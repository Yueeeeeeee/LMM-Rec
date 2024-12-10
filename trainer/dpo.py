import json
from abc import ABCMeta
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import json
from abc import *
from pathlib import Path
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange

from model import *
from config import *
from constants import *
from trainer.lmm import *
from dataloader.utils import *

from .utils import *
from .loggers import *
from .verb import ManualVerbalizer


class LMMDataset(data_utils.Dataset):
    def __init__(self, args, seqs, candidates, labels,
                    processor, meta_dict, image_dict):
        self.args = args
        self.processor = processor
        self.meta_dict = meta_dict
        self.image_dict = image_dict
        self.data = list(zip(seqs, candidates, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, candidates, label = self.data[idx]
        seq = seq[-self.args.lmm_max_len:]
        return prepare_multimodal_input(self.args, seq, candidates, label, self.meta_dict,
                                        self.image_dict, self.processor, eval=True)


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, ref, model, ranker, ranker_processor, train_loader, 
                 val_loader, test_loader, dataloader, export_root, use_wandb=True):
        self.args = args
        self.device = args.device
        self.ref = ref.to(self.device)
        self.ref.eval()
        self.ranker = ranker
        self.ranker.eval()
        self.model = model.to(self.device)
        self.processor = ranker_processor

        self.num_epochs = args.dpo_epochs
        self.metric_ks = args.rerank_metric_ks
        self.best_metric = args.rerank_best_metric
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
        self.lr_scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer, args.dpo_warmup_steps, len(self.train_loader) * self.num_epochs)
            
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

    def train(self, test=False):
        accum_iter = 0
        self.exit_training = self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            if self.args.dpo_val_strategy == 'epoch':
                self.exit_training = self.validate(epoch, accum_iter)
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
            if self.args.dpo_val_strategy == 'iteration' and accum_iter % self.args.dpo_val_iterations == 0:
                self.exit_training = self.validate(epoch, accum_iter)
                if self.exit_training: break

        return accum_iter

    def train_one_iter(self, batch):
        self.model.eval()
        with torch.no_grad():
            batch = self.to_device(batch)
            scores, flat_seqs, labels = self.get_scores_seqs_labels(batch)
            samples_0 = torch.multinomial(
                scores, num_samples=self.args.lora_eval_rerank_k, replacement=False)
            samples_1 = torch.multinomial(
                scores, num_samples=self.args.lora_eval_rerank_k, replacement=False)
            
            ranking_indices, skip_indices = [], []
            for i in range(len(samples_0)):
                if labels[i] not in samples_0[i] and labels[i] not in samples_1[i]:
                    samples_0[i, 0] = labels[i]
            
                if labels[i] in samples_0[i] and labels[i] in samples_1[i]:
                    ranking_indices.append(i)
                else:
                    skip_indices.append(i)

            sampled_scores_0 = torch.gather(scores, 1, samples_0)
            sampled_scores_1 = torch.gather(scores, 1, samples_1)
            reordered_seqs, reordered_labels, chosen, rejected = [], [], [], []

            if len(ranking_indices) > 0:
                sorted_0 = torch.gather(samples_0[ranking_indices], -1,
                                        torch.argsort(sampled_scores_0[ranking_indices], -1, descending=True))
                sorted_1 = torch.gather(samples_1[ranking_indices], -1,
                                        torch.argsort(sampled_scores_1[ranking_indices], -1, descending=True))

                ranking_seqs = [flat_seqs[i] for i in ranking_indices]
                samples_0_ranks = self.estimate_ranks(ranking_seqs, sorted_0, labels[ranking_indices])
                samples_1_ranks = self.estimate_ranks(ranking_seqs, sorted_1, labels[ranking_indices])
                for i, rank_0, rank_1 in zip(ranking_indices, samples_0_ranks, samples_1_ranks):
                    if rank_0 <= rank_1:
                        chosen.append(samples_0[i])
                        rejected.append(samples_1[i])
                    else:
                        chosen.append(samples_1[i])
                        rejected.append(samples_0[i])

                reordered_seqs.extend(ranking_seqs)
                reordered_labels.extend(labels[ranking_indices].tolist())

            if len(skip_indices) > 0:
                for i in skip_indices:
                    if labels[i] in samples_0[i]:
                        chosen.append(samples_0[i])
                        rejected.append(samples_1[i])
                    else:
                        chosen.append(samples_1[i])
                        rejected.append(samples_0[i])

                reordered_seqs.extend([flat_seqs[i] for i in skip_indices])
                reordered_labels.extend(labels[skip_indices].tolist())
            
            chosen = torch.vstack(chosen).contiguous()
            rejected = torch.vstack(rejected).contiguous()
            batch_max_len = max([len(seq) for seq in reordered_seqs])
            padded_seqs = [[0] * (batch_max_len - len(seq)) + seq for seq in reordered_seqs]
            padded_seqs = torch.tensor(padded_seqs).to(self.device)
            reordered_labels = torch.tensor(reordered_labels).to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()

        batch = (padded_seqs, reordered_labels, chosen, rejected)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.clip_gradients(self.args.dpo_max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss

    def estimate_ranks(self, flat_seqs, candidate_sets, labels):
        if isinstance(candidate_sets, torch.Tensor):
            candidate_sets = candidate_sets.tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        ranking_dataset = LMMDataset(self.args, flat_seqs, candidate_sets, labels,
                                     self.processor, self.meta_dict, self.image_dict)
        collate_fn = collate_fn_w_truncation(self.processor, self.args.lmm_max_text_len, eval=True)
        ranking_loader = data_utils.DataLoader(ranking_dataset, batch_size=self.args.lora_micro_batch_size,
                                               shuffle=False, collate_fn=collate_fn, pin_memory=True,
                                               num_workers=self.args.num_workers)
    
        with torch.no_grad(), \
            torch.autocast(device_type=str(self.ranker.device), dtype=self.ranker.config.torch_dtype):
            all_logits, projected_labels = [], []
            for batch_idx, batch in enumerate(ranking_loader):
                outputs = self.ranker(**batch)
                all_logits.append(outputs.logits.cpu())
                del outputs; torch.cuda.empty_cache()
        
        ranks = []
        scores = self.verbalizer.process_logits(torch.vstack(all_logits))
        for score, label in zip(scores, projected_labels):
            _, sorted_indices = torch.sort(score, descending=True)
            ranks.append((sorted_indices == label).nonzero()[0].item())

        return ranks

    def construct_eval_data(self, mode='val'):
        class LRUDataset(data_utils.Dataset):
            def __init__(self, seqs, labels):
                self.data = list(zip(seqs, labels))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]
        
        def collate_fn(batch):
            padded_seqs, labels, seqs = [], [], []
            for i, (seq, label) in enumerate(batch):
                tokens = seq[-self.args.lru_max_len:]
                mask_len = self.args.lru_max_len - len(tokens)
                padded = [0] * mask_len + deepcopy(tokens)
                padded_seqs.append(padded)
                labels.append(label)
                seqs.append(tokens)
            return torch.tensor(padded_seqs), labels, seqs
    
        self.model.eval()
        eval_seqs, eval_labels = [], []
        dataset = self.dataloader.val if mode == 'val' else self.dataloader.test
        for u in dataset.keys():
            if mode == 'val':
                eval_seqs.append(self.dataloader.train[u])
                eval_labels.append(self.dataloader.val[u][0])
            else:
                eval_seqs.append(self.dataloader.train[u]+self.dataloader.val[u])
                eval_labels.append(self.dataloader.test[u][0])
        
        retrieval_dataset = LRUDataset(eval_seqs, eval_labels)
        retrieval_loader = data_utils.DataLoader(retrieval_dataset, batch_size=self.args.val_batch_size,
                                                 shuffle=False, collate_fn=collate_fn, pin_memory=True,
                                                 num_workers=self.args.num_workers)
        
        filtered_seqs, filtered_candidates, filtered_labels = [], [], []
        tqdm_dataloader = tqdm(retrieval_loader)
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_labels, batch_seqs = batch[1], batch[2]
            with torch.no_grad():
                scores = self.model(
                    batch[0].to(self.device))[:, -1, :]
                scores[:, 0] = -1e9  # padding
                for seq, l, p in zip(batch_seqs, batch_labels, scores):
                    candidates = torch.topk(p, self.args.lora_eval_rerank_k).indices
                    if l in candidates:
                        filtered_seqs.append(seq)
                        filtered_candidates.append(candidates.tolist())
                        filtered_labels.append(l)
                tqdm_dataloader.set_description(
                    f'Retrieval success for {len(filtered_seqs)} examples')

        return filtered_seqs, filtered_candidates, filtered_labels

    # def validate(self, epoch, accum_iter):
    #     self.model.eval()
    #     average_meter_set = AverageMeterSet()
    #     _ = self.construct_eval_data('val')
    #     with torch.no_grad():
    #         tqdm_dataloader = tqdm(self.val_loader)
    #         for batch_idx, batch in enumerate(tqdm_dataloader):
    #             batch = self.to_device(batch)
    #             metrics = self.calculate_metrics(batch)
    #             self._update_meter_set(average_meter_set, metrics)
    #             self._update_dataloader_metrics(
    #                 tqdm_dataloader, average_meter_set)

    #         log_data = {
    #             'state_dict': (self._create_state_dict()),
    #             'epoch': epoch+1,
    #             'accum_iter': accum_iter,
    #         }
    #         log_data.update(average_meter_set.averages())
        
    #     return self.logger_service.log_val(log_data)  # early stopping

    def validate(self, epoch, accum_iter):
        average_meter_set = AverageMeterSet()
        eval_seqs, eval_candidates, eval_labels = self.construct_eval_data('val')
        
        ranking_dataset = LMMDataset(self.args, eval_seqs, eval_candidates, eval_labels,
                                     self.processor, self.meta_dict, self.image_dict)
        collate_fn = collate_fn_w_truncation(self.processor, self.args.lmm_max_text_len, eval=True)
        ranking_loader = data_utils.DataLoader(ranking_dataset, batch_size=self.args.lora_micro_batch_size,
                                               shuffle=False, collate_fn=collate_fn, pin_memory=True,
                                               num_workers=self.args.num_workers)

        retrieval_size = len(eval_seqs)
        original_size = len(self.dataloader.val)
        tqdm_dataloader = tqdm(ranking_loader)
        with torch.no_grad(), \
            torch.autocast(device_type=str(self.ranker.device), dtype=self.ranker.config.torch_dtype):
            for batch_idx, batch in enumerate(tqdm_dataloader):
                outputs = self.ranker(**batch)
                scores = self.verbalizer.process_logits(outputs.logits)
                del outputs; torch.cuda.empty_cache()

                safe_ks = [k for k in self.metric_ks if k <= scores.shape[1]]
                metrics = absolute_recall_mrr_ndcg_for_ks(
                    scores, batch['labels'].cpu(), safe_ks)
                metrics = {k: v * retrieval_size / original_size
                           for k, v in metrics.items()}  # estimate full metrics
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

    def test(self, epoch=-1, accum_iter=-1, save_metrics=True):
        print('******************** Testing Best Model ********************')
        best_model_dict = torch.load(os.path.join(
            self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
        self.model.load_state_dict(best_model_dict)

        average_meter_set = AverageMeterSet()
        eval_seqs, eval_candidates, eval_labels = self.construct_eval_data('test')

        ranking_dataset = LMMDataset(self.args, eval_seqs, eval_candidates, eval_labels,
                                     self.processor, self.meta_dict, self.image_dict)
        collate_fn = collate_fn_w_truncation(self.processor, self.args.lmm_max_text_len, eval=True)
        ranking_loader = data_utils.DataLoader(ranking_dataset, batch_size=self.args.lora_micro_batch_size,
                                               shuffle=False, collate_fn=collate_fn, pin_memory=True,
                                               num_workers=self.args.num_workers)

        retrieval_size = len(eval_seqs)
        original_size = len(self.dataloader.test)
        tqdm_dataloader = tqdm(ranking_loader)
        with torch.no_grad(), \
            torch.autocast(device_type=str(self.ranker.device), dtype=self.ranker.config.torch_dtype):
            for batch_idx, batch in enumerate(tqdm_dataloader):
                outputs = self.ranker(**batch)
                scores = self.verbalizer.process_logits(outputs.logits)
                del outputs; torch.cuda.empty_cache()

                safe_ks = [k for k in self.metric_ks if k <= scores.shape[1]]
                metrics = absolute_recall_mrr_ndcg_for_ks(
                    scores, batch['labels'].cpu(), safe_ks)
                # metrics = {k: v * retrieval_size / original_size
                #            for k, v in metrics.items()}  # preserve subset metrics
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)
        
        average_metrics = average_meter_set.averages()
        print('Ranking Performance on Subset:', average_metrics)
        if save_metrics:
            with open(os.path.join(self.export_root, 'subset_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)

        overall_metrics = {}
        for key in average_metrics.keys():
            cur_k = int(key.split('@')[-1])
            if cur_k > self.args.lora_eval_rerank_k: continue
            overall_metrics[key] = average_metrics[key] * retrieval_size / original_size
        
        print('Overall Performance of Our Framework:', overall_metrics)
        if save_metrics:
            with open(os.path.join(self.export_root, 'overall_metrics.json'), 'w') as f:
                json.dump(overall_metrics, f, indent=4)

        return overall_metrics
    
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
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
            ['Recall@%d' % k for k in self.metric_ks[:3]]
        description = 'Subset eval: ' + \
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
                'weight_decay': args.dpo_weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.dpo_lr, eps=args.dpo_adam_epsilon)
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
            val_loggers.append(MetricGraphPrinter(
                key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation', use_wandb=self.use_wandb))
            val_loggers.append(MetricGraphPrinter(
                key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation', use_wandb=self.use_wandb))
            val_loggers.append(MetricGraphPrinter(
                key='MRR@%d' % k, graph_name='MRR@%d' % k, group_name='Validation', use_wandb=self.use_wandb))

        val_loggers.append(RecentModelLogger(self.args, model_checkpoint))
        val_loggers.append(BestModelLogger(self.args, model_checkpoint, metric_key=self.best_metric))

        for k in self.metric_ks:
            test_loggers.append(MetricGraphPrinter(
                key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Test', use_wandb=self.use_wandb))
            test_loggers.append(MetricGraphPrinter(
                key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Test', use_wandb=self.use_wandb))
            test_loggers.append(MetricGraphPrinter(
                key='MRR@%d' % k, graph_name='MRR@%d' % k, group_name='Test', use_wandb=self.use_wandb))

        return val_loggers, test_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }


class LRUDPOTrainer(BaseTrainer):
    def calculate_metrics(self, batch):
        seqs, labels = batch
        
        scores = self.model(seqs)[:, -1, :]
        scores[:, 0] = -1e9  # padding
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        return metrics
    
    def get_scores_seqs_labels(self, batch):
        seqs, labels = batch
        label_indices = seqs.flatten().nonzero().squeeze()
        flat_seqs = []
        for seq in seqs:
            start_idx = seq.nonzero().squeeze()[0]
            for i in range(start_idx, len(seq)):
                flat_seqs.append(seq[start_idx: i+1].tolist())
        
        scores = self.model(seqs)
        scores = scores.reshape(-1, scores.size(-1))[label_indices]
        scores[:, 0] = -1e9
        return torch.softmax(scores, -1), flat_seqs, labels.flatten()[label_indices]

    # # dpo
    # def dpo_loss(self, logits, ref_logits, chosen, rejected, beta=0.1, smoothing=0.1):
    #     chosen_logits = torch.gather(logits, 1, chosen).sum(-1)
    #     rejected_logits = torch.gather(logits, 1, rejected).sum(-1)

    #     chosen_ref_logits = torch.gather(ref_logits, 1, chosen).sum(-1)
    #     rejected_ref_logits = torch.gather(ref_logits, 1, rejected).sum(-1)

    #     pi_logratios = chosen_logits - rejected_logits
    #     ref_logratios = chosen_ref_logits - rejected_ref_logits

    #     logits = pi_logratios - ref_logratios
    #     loss = -F.logsigmoid(beta * logits) * (1 - smoothing) - F.logsigmoid(-beta * logits) * smoothing
    #     return loss.mean()

    # def calculate_loss(self, batch):
    #     seqs, labels, chosen, rejected = batch
        
    #     with torch.no_grad():
    #         ref_logits = self.ref(seqs)[:, -1, :]
        
    #     logits = self.model(seqs)[:, -1, :]

    #     loss = 0.
    #     if self.args.dpo_with_cross_entropy:
    #         loss += F.cross_entropy(logits, labels)
        
    #     loss += self.dpo_loss(logits, ref_logits, chosen, rejected,
    #                          beta=self.args.dpo_beta, smoothing=self.args.dpo_smoothing)
    #     return loss

    def calculate_loss(self, batch):
        seqs, labels, chosen, rejected = batch

        logits = self.model(seqs)[:, -1, :]
        pos_loss = F.cross_entropy(logits, labels)

        all_probs = torch.softmax(logits, -1)
        pos_prob = torch.gather(all_probs, 1, chosen).sum(-1)
        neg_prob = torch.gather(all_probs, 1, rejected).sum(-1)

        pos_odds = pos_prob / (1 - pos_prob)
        neg_odds = neg_prob / (1 - neg_prob)
        
        sig_ratio = torch.nn.functional.sigmoid(torch.log(pos_odds/neg_odds))
        ratio = torch.log(sig_ratio)

        loss = pos_loss - torch.mean(self.args.dpo_beta * ratio)
        
        return loss

    # # simpo
    # def calculate_loss(self, batch, gamma_beta_ratio=0.5):
    #     seqs, labels, chosen, rejected = batch

    #     logits = self.model(seqs)[:, -1, :]
    #     pos_loss = F.cross_entropy(logits, labels)

    #     # all_probs = torch.softmax(logits, -1)
    #     chosen_prob = torch.gather(logits, 1, chosen).sum(-1)
    #     rejected_prob = torch.gather(logits, 1, rejected).sum(-1)

    #     pi_logratios = chosen_prob - rejected_prob
    #     logits = pi_logratios.to(self.device) - gamma_beta_ratio

    #     losses = (
    #         -F.logsigmoid(self.args.dpo_beta * logits) * (1 - self.args.dpo_smoothing)
    #         - F.logsigmoid(-self.args.dpo_beta * logits) * self.args.dpo_smoothing
    #     )
    #     loss = pos_loss - torch.mean(losses)

    #     return loss