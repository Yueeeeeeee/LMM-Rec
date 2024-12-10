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
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        skip_keywords = ["visual", "lm_head"]
        for name, module in model.named_modules():
            if any(keyword in name for keyword in skip_keywords):
                continue
            if isinstance(module, cls):
                lora_module_names.add(name)
        return list(lora_module_names)

    print(args)
    set_seed(args.seed)
    if args.initial_retriever_path is None:
        args.initial_retriever_path = EXPERIMENT_ROOT + '/lru_initial/' + args.dataset_code  # lru retriver
    if export_root is None:
        export_root = EXPERIMENT_ROOT + '/' + args.lmm_base_model.split('/')[-1] + '/' + args.dataset_code

    # prepare data and initialize retriever / processor
    train_loader, val_loader, test_loader, processor, test_retrieval = dataloader_factory(args)

    # initialize model config
    config = AutoConfig.from_pretrained(
        args.lmm_base_model,
    )
    config.torch_dtype = torch.bfloat16
    
    # intialize model in 4bit
    model = LMMForRecommendation.from_pretrained(
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
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    
    # apply lora and fix dtype issues
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=args.lora_dropout,
        use_dora=False,
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    trainer = LMMTrainer(
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        processor,
        export_root,
        args.use_wandb
    )
    trainer.train()
    trainer.test(test_retrieval)


if __name__ == "__main__":
    args.model_code = 'lmm'
    set_template(args)
    main(args, export_root=None)