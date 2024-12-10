# Data

## Download interaction data and images from https://drive.google.com/drive/u/1/folders/1AjM8Gx4A3xo8seYFWwNUBHpM9uRbfydR

# Train Retriever

An example of training LRU retriever on Beauty:
```
CUDA_VISIBLE_DEVICES=0 python train_retriever.py --dataset_code beauty
```

# Train LMM Ranker

An example of training Qwen2 VL ranker on Beauty:
```
CUDA_VISIBLE_DEVICES=0 python train_ranker.py --dataset_code beauty
```

# Optimize Retriever Model

An example of optimize LRU retriever on Beauty:
```
CUDA_VISIBLE_DEVICES=0 python optimize_retriever.py --dataset_code beauty;
```