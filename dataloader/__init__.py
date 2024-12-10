from dataset import dataset_factory
from transformers import AutoProcessor

from .lru import *
from .lmm import *


def dataloader_factory(args):
    dataset = dataset_factory(args)
    if args.model_code in ['lru', 'dpo']:
        dataloader = LRUDataloader(args, dataset)
    elif args.model_code == 'lmm':
        dataloader = LMMDataloader(args, dataset)
    else:
        raise NotImplementedError
    
    train, val, test = dataloader.get_pytorch_dataloaders()
    if args.model_code == 'lru':
        return train, val, test
    
    elif args.model_code == 'lmm':
        processor = dataloader.processor
        test_retrieval = dataloader.test_retrieval
        
        return train, val, test, processor, test_retrieval
    
    elif args.model_code == 'dpo':
        processor = AutoProcessor.from_pretrained(
            args.lmm_base_processor,
        )
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.truncation_side = "left"
        processor.tokenizer.clean_up_tokenization_spaces = True
        processor.tokenizer.model_max_length = args.lmm_max_text_len
        
        processor.image_processor.min_pixels = args.lmm_min_resolution
        processor.image_processor.max_pixels = args.lmm_max_resolution
        
        return train, val, test, processor, dataloader