import json
from abc import ABCMeta
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import json
from abc import *
from pathlib import Path
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
from transformers import LogitsProcessor
from vllm import SamplingParams
from PIL import Image
from tqdm import tqdm

from model import *
from config import *
from constants import *

from .utils import *
from .loggers import *
from .verb import ManualVerbalizer
from .qwen_vl_utils import process_vision_info


def construct_item_text(args,
                        processor,
                        item,
                        meta,
                        image_dict,
                        image_paths,
                        ):
    item_attrs = []
    if item in image_dict:
        image_paths.append(image_dict[item])
        item_attrs.append('"image": ' + DEFAULT_IMAGE_TOKEN)

    for key in args.lmm_text_attributes:
        if key not in meta: continue

        attr = meta[key]
        numerical = False
        if isinstance(attr, list):
            attr = ", ".join(attr[0])
        elif isinstance(attr, float) or isinstance(attr, int):
            attr = str(attr)
            numerical = True
        attr = processor.tokenizer.tokenize(attr)[:args.lmm_max_attr_len]
        if numerical:
            item_attrs.append('"' + key + '": ' + processor.tokenizer.convert_tokens_to_string(attr))
        else:
            item_attrs.append('"' + key + '": "' + processor.tokenizer.convert_tokens_to_string(attr) + '"')

    return json.dumps("{" + ", ".join(item_attrs) + "}")[1:-1]


def prepare_multimodal_input(args,
                             seq,
                             candidates,
                             label,
                             meta_dict,
                             image_dict,
                             processor,
                             eval=False
                             ):
    image_paths = []
    seq_t = "\n".join([construct_item_text(
        args, processor, item, meta_dict[item], image_dict, image_paths) for item in seq])
    can_t = "\n".join(["(" + chr(ord("A") + idx) + ") " + construct_item_text(
        args, processor, item, meta_dict[item], image_dict, image_paths) for idx, item in enumerate(candidates)])
    output = chr(ord("A") + candidates.index(label))

    raw_prompt = args.lmm_instruct_template.format(seq_t, can_t)
    split_prompt = raw_prompt.split(DEFAULT_IMAGE_TOKEN)
    image_paths = [str(x) for x in image_paths] + [None]

    multimodal_content = []
    for text, image in zip(split_prompt, image_paths):
        multimodal_content.append({"type": "text", "text": text})
        if image is not None:
            multimodal_content.append({"type": "image", "image": image})

    messages = [
        {
            "role": "user",
            "content": multimodal_content,
        }
    ]
    eval_prompt = processor.apply_chat_template(
        messages, tokenize=False, 
        add_generation_prompt=True,
    )

    full_messages = deepcopy(messages)
    full_messages.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": output}
            ]
        }
    )
    full_prompt = processor.apply_chat_template(
        full_messages, tokenize=False, 
        add_generation_prompt=False,
    )

    if eval:
        return {
            "text": eval_prompt,
            "message": messages,
            "images": image_paths[:-1],
            "labels": ord(output) - ord("A"),
        }
    else:
        return {
            "text": full_prompt,
            "message": messages,
            "images": image_paths[:-1],
            "eval_text": eval_prompt,  # for label generation
        }


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


class LogitsWarper(LogitsProcessor):
    def __init__(self, logits):
        self.logits = logits

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores
        if self.logits is not None:
            self.logits.append(scores.cpu())
        return scores_processed


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, ref, model, ranker, ranker_processor, train_loader, 
                 val_loader, test_loader, dataloader, export_root, use_wandb=True):
        self.args = args
        self.device = args.device
        self.ref = ref.to(self.device)
        self.ref.eval()
        self.ranker = ranker
        # self.ranker.eval()
        self.model = model.to(self.device)
        self.processor = ranker_processor

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dataloader = dataloader
        self.meta_dict = dataloader.meta_dict
        self.image_dict = dataloader.image_dict

        self.verbalizer = ManualVerbalizer(
            tokenizer=self.processor.tokenizer,
            prefix="",
            post_log_softmax=False,
            classes=list(range(args.lora_eval_rerank_k)),
            label_words={i: chr(ord("A")+i) for i in range(args.lora_eval_rerank_k)},
        )

        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            if args.enable_lr_warmup:
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, args.warmup_steps, len(self.train_loader) * self.num_epochs)
            else:
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=args.decay_step, gamma=args.gamma)
            
        self.export_root = export_root
        if not os.path.exists(self.export_root):
            Path(self.export_root).mkdir(parents=True)
        
        with open(Path(self.export_root).joinpath('config.yml'), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(
                name=self.args.model_code+'_'+self.args.dataset_code,
                project=PROJECT_NAME,
                config=args,
            )
            writer = wandb
        else:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(
                log_dir=Path(self.export_root).joinpath('logs'),
                comment=self.args.model_code+'_'+self.args.dataset_code,
            )
        self.val_loggers, self.test_loggers = self._create_loggers()
        self.logger_service = LoggerService(
            self.args, writer, self.val_loggers, self.test_loggers, use_wandb)
        
        print(args)

    def train(self, test=False):
        accum_iter = 0
        self.exit_training = self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            if self.args.val_strategy == 'epoch':
                self.exit_training = self.validate(epoch, accum_iter)  # val after every epoch
            if self.exit_training:
                print('Early stopping triggered. Exit training')
                break

        if test: self.test()
        self.logger_service.complete()

    def train_one_epoch(self, epoch, accum_iter):
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            self.model.train()
            loss = self.train_one_iter(batch)

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += 1
            if self.args.val_strategy == 'iteration' and accum_iter % self.args.val_iterations == 0:
                self.exit_training = self.validate(epoch, accum_iter)  # val after certain iterations
                if self.exit_training: break

        return accum_iter
    
    def train_one_iter(self, batch):
        self.model.eval()
        with torch.no_grad():
            batch = self.to_device(batch)
            scores, labels = self.get_scores_and_labels(batch)
            samples = torch.multinomial(scores, num_samples=2*(self.args.lmm_negative_size+1),
                                        replacement=False).split(self.args.lmm_negative_size+1, -1)
            
            samples_0, samples_1 = samples[0], samples[1]
            for i in range(len(samples_0)):  # make sure label is within candidates
                if labels[i] not in samples_0[i]:
                    samples_0[i, 0] = labels[i]
                if labels[i] not in samples_1[i]:
                    samples_1[i, 0] = labels[i]
            
            sampled_scores_0 = torch.gather(scores, 1, samples_0)
            _, sorted_indices = torch.sort(sampled_scores_0, dim=1, descending=True)
            samples_0_sorted = torch.gather(samples_0, 1, sorted_indices)

            sampled_scores_1 = torch.gather(scores, 1, samples_1)
            _, sorted_indices = torch.sort(sampled_scores_1, dim=1, descending=True)
            samples_1_sorted = torch.gather(samples_1, 1, sorted_indices)

            chosen, rejected = [], []
            samples_0_ranks = self.compute_ranks(batch[0], samples_0_sorted, labels)
            samples_1_ranks = self.compute_ranks(batch[0], samples_1_sorted, labels)
            for idx, (rank_0, rank_1) in enumerate(zip(samples_0_ranks, samples_1_ranks)):
                if rank_0 <= rank_1:
                    chosen.append(samples_0_sorted[idx])
                    rejected.append(samples_1_sorted[idx])
                else:
                    chosen.append(samples_1_sorted[idx])
                    rejected.append(samples_0_sorted[idx])
            
            chosen = torch.vstack(chosen).contiguous()
            rejected = torch.vstack(rejected).contiguous()

        self.model.train()
        loss = self.calculate_loss(batch, chosen, rejected)
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_gradients(self.args.max_grad_norm)
        self.optimizer.step()

        return loss

    def compute_ranks(self, item_ids, candidate_sets, labels):
        projected_labels, all_logits = [], []
        for seq, candidates, label in zip(item_ids, candidate_sets, labels):
            start = (seq != 0).nonzero(as_tuple=True)[0][0].item()
            projected_labels.append(candidates.tolist().index(label.item()))
            inputs = prepare_multimodal_input(args, seq[start:].tolist(), candidates.tolist(), label.item(),
                                              self.meta_dict, self.image_dict, self.processor, eval=True)

            processor = LogitsWarper(logits=[])
            images = [Image.open(x) for x in inputs['images']]
            _ = self.ranker.generate(
                {"prompt": inputs['text'], "multi_modal_data": {"image": images}},
                SamplingParams(max_tokens=1, logits_processors=[processor]),
            )
            all_logits.append(processor.logits)

        import pdb; pdb.set_trace()
        ranks = []
        scores = self.verbalizer.process_logits(torch.vstack(all_logits))
        for score, label in zip(scores, projected_labels):
            _, sorted_indices = torch.sort(score, descending=True)
            ranks.append((sorted_indices == label).nonzero(as_tuple=True)[0].item())

        return ranks

    def validate(self, epoch, accum_iter):
        self.model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
        
        return self.logger_service.log_val(log_data)  # early stopping

    def test(self, epoch=-1, accum_iter=-1, save_name=None):
        print('******************** Testing Best Model ********************')
        best_model_dict = torch.load(os.path.join(
            self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
        self.model.load_state_dict(best_model_dict)
        self.model.eval()

        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            average_metrics = average_meter_set.averages()
            log_data.update(average_metrics)
            self.logger_service.log_test(log_data)

            print('******************** Testing Metrics ********************')
            print(average_metrics)
            file_name = 'test_metrics.json' if save_name is None else save_name
            with open(os.path.join(self.export_root, file_name), 'w') as f:
                json.dump(average_metrics, f, indent=4)
        
        return average_metrics
    
    def to_device(self, batch):
        return [x.to(self.device) for x in batch]

    @abstractmethod
    def calculate_loss(self, batch):
        pass
    
    @abstractmethod
    def calculate_metrics(self, batch):
        pass
    
    def clip_gradients(self, limit=1.0):
        nn.utils.clip_grad_norm_(self.model.parameters(), limit)

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[-2:]] + \
            ['Recall@%d' % k for k in self.metric_ks[-2:]]
        description = 'Eval: ' + \
            ', '.join(s + ' {:.4f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(
            *(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        else:
            raise NotImplementedError

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _create_loggers(self):
        root = Path(self.export_root)
        model_checkpoint = root.joinpath('models')

        val_loggers, test_loggers = [], []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation', use_wandb=self.use_wandb))
            val_loggers.append(
                MetricGraphPrinter(key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation', use_wandb=self.use_wandb))
            val_loggers.append(
                MetricGraphPrinter(key='MRR@%d' % k, graph_name='MRR@%d' % k, group_name='Validation', use_wandb=self.use_wandb))

        val_loggers.append(RecentModelLogger(self.args, model_checkpoint))
        val_loggers.append(BestModelLogger(self.args, model_checkpoint, metric_key=self.best_metric))

        for k in self.metric_ks:
            test_loggers.append(
                MetricGraphPrinter(key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Test', use_wandb=self.use_wandb))
            test_loggers.append(
                MetricGraphPrinter(key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Test', use_wandb=self.use_wandb))
            test_loggers.append(
                MetricGraphPrinter(key='MRR@%d' % k, graph_name='MRR@%d' % k, group_name='Test', use_wandb=self.use_wandb))

        return val_loggers, test_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.state_dict(),
            # OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }


class LRUDPOTrainer(BaseTrainer):
    def dpo_loss(self, logits, ref_logits, chosen, rejected, beta=1.0, label_smoothing=0.1):
        chosen_logits = torch.gather(logits, 1, chosen).sum(-1)
        rejected_logits = torch.gather(logits, 1, rejected).sum(-1)

        chosen_ref_logits = torch.gather(ref_logits, 1, chosen).sum(-1)
        rejected_ref_logits = torch.gather(ref_logits, 1, rejected).sum(-1)

        pi_logratios = chosen_logits - rejected_logits
        ref_logratios = chosen_ref_logits - rejected_ref_logits

        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

        return loss.mean()

    def calculate_loss(self, batch, chosen, rejected):
        seqs, labels = batch
        with torch.no_grad():
            ref_logits = self.ref(seqs)[:, -1, :]
            ref_logits = ref_logits.view(-1, ref_logits.size(-1))
        
        logits = self.model(seqs)[:, -1, :]
        logits = logits.view(-1, logits.size(-1))

        loss = self.dpo_loss(logits, ref_logits, chosen, rejected)
        return loss

    def get_scores_and_labels(self, batch):
        scores = self.model(batch[0])[:, -1, :]
        scores[:, 0] = -1e9
        return torch.softmax(scores, -1), batch[1][:, -1]

    def calculate_metrics(self, batch):
        seqs, labels = batch
        
        scores = self.model(seqs)[:, -1, :]
        scores[:, 0] = -1e9  # padding
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        return metrics