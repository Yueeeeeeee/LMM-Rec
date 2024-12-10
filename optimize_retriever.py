import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from config import *
from model import *
from dataloader import *
from trainer import *

from transformers import set_seed
from transformers import AutoConfig
from transformers import BitsAndBytesConfig
from peft import (
    PeftModel,
    LoraConfig,
    prepare_model_for_kbit_training,
)

try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    print(args)
    set_seed(args.seed)
    if args.initial_retriever_path is None:
        args.initial_retriever_path = EXPERIMENT_ROOT + '/lru_initial/' + args.dataset_code  # lru retriver
    if export_root is None:
        export_root = EXPERIMENT_ROOT + '/lru_optimized/' + args.dataset_code  # lru retriver
    if args.lora_ranker_path is None:
        args.lora_ranker_path = EXPERIMENT_ROOT + '/' + args.lmm_base_model.split('/')[-1] + '/' + args.dataset_code
        if args.lora_path_postfix is not None:
            args.lora_ranker_path += '/' +  args.lora_path_postfix
        else:
            lora_paths = [x for x in os.listdir(args.lora_ranker_path) if x.startswith('checkpoint')]
            args.lora_ranker_path += '/' +  sorted(lora_paths)[0]

    # prepare data and initialize retriever / ranker
    train_loader, val_loader, test_loader, processor, dataloader = dataloader_factory(args)

    # initialize model config
    config = AutoConfig.from_pretrained(
        args.lmm_base_model,
    )
    config.torch_dtype = torch.bfloat16
    
    # intialize model in 4bit
    ranker = LMMForRecommendation.from_pretrained(
        args.lmm_base_model,
        config=config,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        ),
    )
    
    # apply lora and merge ranker
    print('Loading from', args.lora_ranker_path)
    ranker = PeftModel.from_pretrained(ranker, args.lora_ranker_path)

    # load reference and retriever model
    args.lru_dropout = args.dpo_dropout
    args.lru_attn_dropout = args.dpo_dropout
    ref, model = LRURec(args), LRURec(args)
    best_model_dict = torch.load(os.path.join(
        args.initial_retriever_path, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
    ref.load_state_dict(best_model_dict)
    model.load_state_dict(best_model_dict)

    trainer = LRUDPOTrainer(
        args,
        ref,
        model,
        ranker,
        processor,
        train_loader,
        val_loader,
        test_loader,
        dataloader,
        export_root,
        args.use_wandb
    )
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    args.model_code = 'dpo'
    set_template(args)
    main(args, export_root=None)