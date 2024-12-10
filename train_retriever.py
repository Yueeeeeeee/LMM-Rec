import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from model import *
from dataloader import *
from trainer import *

from config import *
from constants import *

from transformers import set_seed

try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    set_seed(args.seed)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = LRURec(args)
    if export_root is None:
        export_root = EXPERIMENT_ROOT + '/' + args.model_code + '_initial/' + args.dataset_code
    
    trainer = LRUTrainer(args, model, train_loader, val_loader, test_loader, export_root, args.use_wandb)
    trainer.train(test=True)


if __name__ == "__main__":
    args.model_code = 'lru'
    set_template(args)
    main(args, export_root=None)