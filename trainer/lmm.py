from config import *
from constants import *

from .utils import *
from .loggers import *

import json
import torch
from .verb import ManualVerbalizer
from .qwen_vl_utils import process_vision_info

from .base import *
from transformers import get_cosine_schedule_with_warmup
from transformers import TrainingArguments, EarlyStoppingCallback


def compute_metrics_fn(ks):
    def compute_metrics(eval_pred):
        scores, labels = eval_pred
        scores = torch.tensor(scores)
        labels = torch.tensor(labels).view(-1)
        safe_ks = [k for k in ks if k <= scores.shape[1]]
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels, safe_ks)
        return metrics
    return compute_metrics


def preprocess_logits_for_metrics_fn(verbalizer):
    def preprocess_logits_for_metrics(logits, labels):
        if not isinstance(logits, torch.Tensor):
            logits = logits[0]
        scores = verbalizer.process_logits(logits)
        return scores
    return preprocess_logits_for_metrics


def collate_fn_w_truncation(processor, lmm_max_length, eval=False):
    def collate_fn(batch):
        all_messages = []
        for i in range(len(batch)):
            all_messages.extend(batch[i]["message"])

        image_inputs, _ = process_vision_info(all_messages)
        texts = [batch[i]["text"] for i in range(len(batch))]

        inputs = processor(
            text=texts, 
            images=image_inputs,
            padding=True,
            truncation=True,
            max_length=lmm_max_length,
            return_tensors="pt",
        )

        if eval:
            inputs["labels"] = torch.tensor([batch[i]["labels"] for i in range(len(batch))])
        else:
            labels = inputs["input_ids"].clone().tolist()
            eval_tokens = processor(
                text=[batch[i]["eval_text"] for i in range(len(batch))], 
                images=image_inputs,
                padding=True,
                truncation=True,
                max_length=lmm_max_length,
                return_tensors="pt",
            )

            for i in range(len(batch)):
                label_cutoff = len(eval_tokens["input_ids"][i])
                labels[i][:label_cutoff] = [IGNORE_INDEX] * (len(labels[i][:label_cutoff]))
            inputs["labels"] = torch.tensor(labels).long()
        
        for key in inputs:
            if torch.is_floating_point(inputs[key]):
                inputs[key] = inputs[key].to(torch.float16)

        return inputs
    return collate_fn


class LMMTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            processor,
            export_root,
            use_wandb,
            **kwargs
        ):
        self.original_args = args
        self.export_root = export_root
        self.use_wandb = use_wandb

        self.lmm_max_text_len = args.lmm_max_text_len
        self.rerank_metric_ks = args.rerank_metric_ks
        self.lora_eval_rerank_k = args.lora_eval_rerank_k

        self.verbalizer = ManualVerbalizer(
            tokenizer=processor.tokenizer,
            prefix="",
            post_log_softmax=False,
            classes=list(range(args.lora_eval_rerank_k)),
            label_words={i: chr(ord("A")+i) for i in range(args.lora_eval_rerank_k)},
        ).to(model.device)

        hf_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            do_predict=True,
            optim="adamw_8bit",
            learning_rate=args.lora_lr,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            weight_decay=args.lora_weight_decay,
            num_train_epochs=args.lora_num_epochs,
            per_device_train_batch_size=args.lora_micro_batch_size,
            gradient_accumulation_steps=args.train_batch_size//args.lora_micro_batch_size,
            bf16=True,
            logging_steps=10,
            # eval_accumulation_steps=1,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=args.lora_val_iterations,
            save_steps=args.lora_val_iterations,
            save_total_limit=3,
            output_dir=export_root,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            report_to="wandb" if use_wandb else None,
            run_name=args.model_code+'_'+args.dataset_code if use_wandb else None,
            metric_for_best_model=args.rerank_best_metric,
            greater_is_better=True,
        )
        super().__init__(
            model=model,
            args=hf_args,
            callbacks=[EarlyStoppingCallback(args.lora_early_stopping_patience)],
            **kwargs)  # hf_args is now args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.processor = processor

        self.train_loader.collate_fn = collate_fn_w_truncation(
            self.processor, self.lmm_max_text_len, eval=False)
        self.val_loader.collate_fn = collate_fn_w_truncation(
            self.processor, self.lmm_max_text_len, eval=True)
        self.test_loader.collate_fn = collate_fn_w_truncation(
            self.processor, self.lmm_max_text_len, eval=True)

        self.compute_metrics = compute_metrics_fn(self.rerank_metric_ks)
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics_fn(
            self.verbalizer
        )

        if len(self.label_names) == 0:
            self.label_names = ["labels"]  # in case label name is not set

    def get_train_dataloader(self) -> DataLoader:
        return self.accelerator.prepare(self.train_loader)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        return self.accelerator.prepare(self.val_loader)

    def get_test_dataloader(self, test_dataset: Optional[Dataset] = None) -> DataLoader:
        return self.accelerator.prepare(self.test_loader)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def test(self, test_retrieval, save_metrics=True):
        print('******************** Testing Best Model ********************')
        average_metrics = self.predict(test_dataset=None).metrics
        print('Ranking Performance on Subset:', average_metrics)
        if save_metrics:
            with open(os.path.join(self.export_root, 'subset_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)

        overall_metrics = {}
        retrieval_size = test_retrieval['retrieval_size']
        original_size = test_retrieval['original_size']
        for key in test_retrieval['non_retrieval_metrics'].keys():
            if 'test_' + key in average_metrics:
                cur_k = int(key.split('@')[-1])
                if cur_k > self.lora_eval_rerank_k: continue
                overall_metrics['test_' + key] = average_metrics['test_' + key] * retrieval_size / original_size
        
        # print('Original Performance:', test_retrieval['original_metrics'])
        print('Overall Performance of Our Framework:', overall_metrics)
        if save_metrics:
            with open(os.path.join(self.export_root, 'overall_metrics.json'), 'w') as f:
                json.dump(overall_metrics, f, indent=4)

        return average_metrics