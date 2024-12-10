import numpy as np
import random
import torch
import argparse

import yaml


def set_template(args):
    if args.dataset_code == None:
        print('******************** Dataset Selection ********************')
        dataset_code = {'b': 'beauty', 'c': 'clothing', 's': 'sports', 't': 'toys'}
        args.dataset_code = dataset_code[input('Input {}: '.format(', '.join([str(k) + ' for ' + v 
                                                                              for k, v in dataset_code.items()])))]

    args.lru_max_len = 50
    if 'lmm' in args.model_code: 
        batch, mini_batch = 32, 2
        args.train_batch_size = batch
        args.val_batch_size = mini_batch * 2
        args.test_batch_size = mini_batch * 2
        args.lora_micro_batch_size = mini_batch
    else: 
        batch, mini_batch = 64, 8
        args.train_batch_size = batch
        args.val_batch_size = batch * 2
        args.test_batch_size = batch * 2
        args.lora_micro_batch_size = mini_batch

    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default=None)
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)

################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)

################
# ID-based Retrieval
################
# optimization #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=False)
parser.add_argument('--decay_step', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=False)
parser.add_argument('--warmup_steps', type=int, default=100)

# evaluation #
parser.add_argument('--val_strategy', type=str, default='epoch', choices=['epoch', 'iteration'])
parser.add_argument('--val_iterations', type=int, default=1000)  # only for iteration val_strategy
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 25])
parser.add_argument('--rerank_metric_ks', nargs='+', type=int, default=[1, 5, 10, 25])
parser.add_argument('--best_metric', type=str, default='Recall@25')
parser.add_argument('--rerank_best_metric', type=str, default='NDCG@10')
parser.add_argument('--use_wandb', type=bool, default=True)

# model #
parser.add_argument('--model_code', type=str, default=None)
parser.add_argument('--lru_max_len', type=int, default=50)
parser.add_argument('--lru_hidden_units', type=int, default=64)
parser.add_argument('--lru_num_blocks', type=int, default=2)
parser.add_argument('--lru_num_heads', type=int, default=2)
parser.add_argument('--lru_head_size', type=int, default=32)
parser.add_argument('--lru_dropout', type=float, default=0.6)
parser.add_argument('--lru_attn_dropout', type=float, default=0.6)
parser.add_argument('--initial_retriever_path', type=str, default=None)

################
# LMM-based Ranking
################
parser.add_argument('--lmm_base_model', type=str, default="Qwen/Qwen2-VL-2B-Instruct")
parser.add_argument('--lmm_base_processor', type=str, default="Qwen/Qwen2-VL-2B-Instruct")
parser.add_argument('--lmm_min_resolution', type=int, default=56*56)
parser.add_argument('--lmm_max_resolution', type=int, default=168*168)
parser.add_argument('--lmm_max_text_len', type=int, default=16384)  # 16k token length
parser.add_argument('--lmm_max_len', type=int, default=25)  # history item length
parser.add_argument('--lmm_text_attributes', type=list,  # image is always included if available
                    default=['title', 'price', 'brand', 'categories'])
parser.add_argument('--lmm_max_attr_len', type=int, default=32)  # item attribute tokens
parser.add_argument('--lmm_negative_size', type=int, default=24)  # 1 label item + N negatives
sequence = "Based on the user's browsing history and a collection of candidate items (both " \
           "provided in JSON format), recommend the most suitable item by specifying " \
           "its index letter.\n\nBrowsing history:\n{}\n\nCandidate items:\n{}"
parser.add_argument('--lmm_instruct_template', type=str, default=sequence)

# lmm lora #
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_lr', type=float, default=1e-4)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_weight_decay', type=float, default=0.01)
parser.add_argument('--lora_target_modules', type=str, default=None)
parser.add_argument('--lora_warmup_ratio', type=float, default=0.03)
parser.add_argument('--lora_num_epochs', type=int, default=1)
parser.add_argument('--lora_val_iterations', type=int, default=100)
parser.add_argument('--lora_eval_rerank_k', type=int, default=25)
parser.add_argument('--lora_early_stopping_patience', type=int, default=5)
parser.add_argument('--lora_micro_batch_size', type=int, default=4)
parser.add_argument('--lora_ranker_path', type=str, default=None)
parser.add_argument('--lora_path_postfix', type=str, default=None)

################
# DPO Retriever Tuning
################
parser.add_argument('--dpo_epochs', type=int, default=2)
parser.add_argument('--dpo_lr', type=float, default=5e-5)
parser.add_argument('--dpo_dropout', type=float, default=0.1)
parser.add_argument('--dpo_warmup_steps', type=int, default=100)
parser.add_argument('--dpo_weight_decay', type=float, default=0.0)
parser.add_argument('--dpo_adam_epsilon', type=float, default=1e-9)
parser.add_argument('--dpo_max_grad_norm', type=float, default=5.0)
parser.add_argument('--dpo_beta', type=float, default=0.1)
parser.add_argument('--dpo_smoothing', type=float, default=0.1)
parser.add_argument('--dpo_with_cross_entropy', type=bool, default=True)
parser.add_argument('--dpo_val_strategy', type=str, default='iteration', choices=['epoch', 'iteration'])
parser.add_argument('--dpo_val_iterations', type=int, default=100)  # only for iteration val_strategy


################

args = parser.parse_args()