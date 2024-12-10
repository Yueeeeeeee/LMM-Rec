from .base import AbstractDataloader

import torch
import random
import numpy as np
from pathlib import Path
import torch.utils.data as data_utils

import os
import json
from tqdm import trange
from copy import deepcopy
from transformers import AutoProcessor

from trainer import batched_absolute_recall_mrr_ndcg_for_ks
from model import LRURec
from constants import *

from .utils import *


class LMMDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.lmm_max_len = args.lmm_max_len  # num training item

        # processor
        self.processor = AutoProcessor.from_pretrained(
            args.lmm_base_processor,
        )
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.truncation_side = "left"
        self.processor.tokenizer.clean_up_tokenization_spaces = True
        self.processor.tokenizer.model_max_length = args.lmm_max_text_len

        self.processor.image_processor.min_pixels = args.lmm_min_resolution
        self.processor.image_processor.max_pixels = args.lmm_max_resolution

        # retriever
        print('Loading retriever from {}...'.format(args.initial_retriever_path))
        retriever = LRURec(args)
        best_model_dict = torch.load(os.path.join(
            args.initial_retriever_path, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
        retriever.load_state_dict(best_model_dict)
        retriever = retriever.to(args.device)
        retriever.eval()

        # construct training data
        self.train_sequences = []
        for u in self.train.keys():
            seq = self.train[u]
            if len(seq) < 5:
                self.train_sequences.append(seq)
            else:
                for start_idx in range(5, len(seq)+1):
                    self.train_sequences.append(seq[:start_idx])

        # compute popularity for negative sampling
        self.popularity = {k: 1 for k in 
                           range(args.num_items+1)}
        self.popularity[0] = 0  # avoid padding item
        for user in self.train:
            if user in self.val:
                seq = self.train[user] + self.val[user]
            else:
                seq = self.train[user]
            for item in seq:
                self.popularity[item] += 1
        popularity_sum = sum(list(self.popularity.values()))
        self.popularity = [self.popularity[k] / popularity_sum 
                           for k in range(args.num_items+1)]

        def batching_fn(batch):
            for i, seq in enumerate(batch):
                tokens = seq[-args.lru_max_len:]
                mask_len = self.args.lru_max_len - len(tokens)
                batch[i] = [0] * mask_len + tokens
            return batch

        # construct validation data
        batch_size = args.train_batch_size
        val_sequences, val_labels, val_users = [], [], []
        for u in self.val.keys():
            val_sequences.append(self.train[u])
            val_labels.append(self.val[u][0])
            val_users.append(u)

        val_probs, self.val_users, self.val_candidates = [], [], []
        num_batches = int(np.ceil(len(val_sequences)/batch_size))
        for i in trange(num_batches, desc='Constructing validation data'):
            batch_users = val_users[i*batch_size:(i+1)*batch_size]
            batch_labels = val_labels[i*batch_size:(i+1)*batch_size]
            batch = val_sequences[i*batch_size:(i+1)*batch_size]
            batch = torch.tensor(batching_fn(batch), dtype=torch.long, device=args.device)
            with torch.no_grad():
                scores = retriever(batch)[:, -1, :]
                scores[:, 0] = -1e9  # padding
                val_probs.extend(scores.tolist())
                for u, l, p in zip(batch_users, batch_labels, scores):
                    candidates = torch.topk(p, args.lora_eval_rerank_k).indices
                    if l in candidates:
                        self.val_users.append(u)
                        self.val_candidates.append(candidates.tolist())

        # construct test data
        test_sequences, test_labels, test_users = [], [], []
        for u in self.test.keys():
            test_sequences.append(self.train[u]+self.val[u])
            test_labels.append(self.test[u][0])
            test_users.append(u)
        
        test_probs, non_test_users = [], []
        self.test_users, self.test_candidates = [], []
        num_batches = int(np.ceil(len(test_sequences)/batch_size))
        for i in trange(num_batches, desc='Constructing test data'):
            batch_users = test_users[i*batch_size:(i+1)*batch_size]
            batch_labels = test_labels[i*batch_size:(i+1)*batch_size]
            batch = test_sequences[i*batch_size:(i+1)*batch_size]
            batch = torch.tensor(batching_fn(batch), dtype=torch.long, device=args.device)
            with torch.no_grad():
                scores = retriever(batch)[:, -1, :]
                scores[:, 0] = -1e9  # padding
                test_probs.extend(scores.tolist())
                for u, l, p in zip(batch_users, batch_labels, scores):
                    candidates = torch.topk(p, args.lora_eval_rerank_k).indices
                    if l in candidates:
                        self.test_users.append(u)
                        self.test_candidates.append(candidates.tolist())
                    else:
                        non_test_users.append(u)

        # metrics for evaluation
        print('Computing retriever metrics...')
        original_metrics = batched_absolute_recall_mrr_ndcg_for_ks(
            test_probs, test_labels, args.metric_ks)
        val_metrics = batched_absolute_recall_mrr_ndcg_for_ks(
            [val_probs[i] for (i, u) in enumerate(self.val) if u in self.val_users],
            [val_labels[i] for (i, u) in enumerate(self.val) if u in self.val_users],
            args.metric_ks)
        retrieval_metrics = batched_absolute_recall_mrr_ndcg_for_ks(
            [test_probs[i] for (i, u) in enumerate(self.test) if u in self.test_users],
            [test_labels[i] for (i, u) in enumerate(self.test) if u in self.test_users],
            args.metric_ks)
        non_retrieval_metrics = batched_absolute_recall_mrr_ndcg_for_ks(
            [test_probs[i] for (i, u) in enumerate(self.test) if u not in self.test_users],
            [test_labels[i] for (i, u) in enumerate(self.test) if u not in self.test_users],
            args.metric_ks)

        self.test_retrieval = {
            'original_size': len(self.test),
            'retrieval_size': len(self.test_users),
            'original_metrics': original_metrics,
            'retrieval_metrics': retrieval_metrics,
            'non_retrieval_metrics': non_retrieval_metrics,
        }
        print('Val examples: {}, test examples: {}.\nOriginal val metrics --> {}'.format(
            len(self.val_users), len(self.test_users),
            ', '.join([str(k) + ': ' + '{:.4f}'.format(v) for k, v in val_metrics.items()]),
        ))

        del retriever

    @classmethod
    def code(cls):
        return 'lmm'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader
    
    def _get_train_dataset(self):
        dataset = LMMTrainDataset(self.args, self.train_sequences, self.popularity, self.lmm_max_len,
                                  self.rng, self.meta_dict, self.image_dict, self.processor)
        return dataset
    
    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = LMMValidDataset(self.args, self.train, self.val, self.lmm_max_len, self.rng,
                                      self.meta_dict, self.image_dict, self.processor,
                                      self.val_users, self.val_candidates)
        elif mode == 'test':
            dataset = LMMTestDataset(self.args, self.train, self.val, self.test, self.lmm_max_len,
                                     self.rng, self.meta_dict, self.image_dict, self.processor,
                                     self.test_users, self.test_candidates)
        return dataset
    
    def _get_train_loader(self):
        return data_utils.DataLoader(self._get_train_dataset(), batch_size=self.args.lora_micro_batch_size, 
                                     shuffle=True, pin_memory=True, num_workers=self.args.num_workers)
    
    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                           pin_memory=True, num_workers=self.args.num_workers)
        return dataloader
    
    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')


class LMMTrainDataset(data_utils.Dataset):
    def __init__(self,
                 args,
                 seqs,
                 popularity,
                 max_len,
                 rng,
                 meta_dict,
                 image_dict,
                 processor,
                 ):
        self.args = args
        self.seqs = seqs
        self.popularity = popularity
        self.max_len = max_len
        self.rng = rng
        
        self.num_items = args.num_items
        self.meta_dict = meta_dict
        self.image_dict = image_dict
        self.processor = processor

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        tokens = self.seqs[index]
        seq = tokens[:-1][-self.max_len:]
        answer = tokens[-1]

        sample_size = self.args.lmm_negative_size + 2
        samples = self.rng.choice(
            self.num_items+1,
            size=sample_size,
            p=self.popularity,
            replace=False,
        )

        cur_idx, candidates = 0, [answer]
        while len(candidates) < self.args.lmm_negative_size + 1:
            item = samples[cur_idx]
            cur_idx += 1
            if item == answer or item == 0:
                continue
            candidates.append(item)
        
        self.rng.shuffle(candidates)

        return prepare_multimodal_input(self.args, seq, candidates, answer, self.meta_dict,
                                        self.image_dict, self.processor, eval=False)


class LMMValidDataset(data_utils.Dataset):
    def __init__(self,
                 args,
                 u2seq,
                 u2answer,
                 max_len,
                 rng,
                 meta_dict,
                 image_dict,
                 processor,
                 val_users,
                 val_candidates
                 ):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        self.max_len = max_len
        self.rng = rng

        self.meta_dict = meta_dict
        self.image_dict = image_dict
        self.processor = processor

        self.val_users = val_users
        self.val_candidates = val_candidates

    def __len__(self):
        return len(self.val_users)

    def __getitem__(self, index):
        user = self.val_users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user][0]
        
        seq = seq[-self.max_len:]
        candidates = self.val_candidates[index]
        assert answer in candidates
        
        return prepare_multimodal_input(self.args, seq, candidates, answer, self.meta_dict,
                                        self.image_dict, self.processor, eval=True)


class LMMTestDataset(data_utils.Dataset):
    def __init__(self,
                 args,
                 u2seq,
                 u2val,
                 u2answer,
                 max_len,
                 rng,
                 meta_dict,
                 image_dict,
                 processor,
                 test_users,
                 test_candidates,
                 ):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        self.max_len = max_len
        self.rng = rng
        
        self.meta_dict = meta_dict
        self.image_dict = image_dict
        self.processor = processor

        self.test_users = test_users
        self.test_candidates = test_candidates
    
    def __len__(self):
        return len(self.test_users)
    
    def __getitem__(self, index):
        user = self.test_users[index]
        seq = self.u2seq[user] + self.u2val[user]
        answer = self.u2answer[user][0]

        seq = seq[-self.max_len:]
        candidates = self.test_candidates[index]
        assert answer in candidates

        return prepare_multimodal_input(self.args, seq, candidates, answer, self.meta_dict,
                                        self.image_dict, self.processor, eval=True)