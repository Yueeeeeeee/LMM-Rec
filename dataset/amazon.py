import os
import copy
import gzip
import torch
from PIL import Image

import json
import pickle
from pathlib import Path

from .base import *
from constants import *


class AmazonDataset(AbstractDataset):
    def load_datamap(self, path):
        with open(path.joinpath('datamaps.json'), "r") as f:
            datamaps = json.load(f)
        
        user2id = datamaps['user2id']
        user2id = {k: int(v) for k, v in user2id.items()}
        
        item2id = datamaps['item2id']
        item2id = {k: int(v) for k, v in item2id.items()}

        id2item = datamaps['id2item']
        id2item = {int(k): v for k, v in id2item.items()}
        
        return user2id, item2id, id2item

    def load_ratings(self, path):
        lines = []
        with open(path.joinpath('sequential_data.txt'), 'r') as f:
            for line in f:
                lines.append(line.rstrip('\n'))
        
        user_items = {}
        for line in lines:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[int(user)] = items

        return user_items
    
    def load_metadict(self, path, item2id):
        def parse(path):
            g = gzip.open(path, 'r')
            for l in g:
                yield eval(l)   
        
        meta_dict = {}
        for meta in parse(path.joinpath('meta.json.gz')):
            if meta['asin'] in item2id:
                meta_dict[item2id[meta['asin']]] = meta
            else:
                continue

        return meta_dict
    
    def load_imagedict(self, path, image_root, item2id):
        def load_pickle(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)

        mapped_dict = {} 
        image_dict = load_pickle(path.joinpath('item2img_dict.pkl'))
        for k, v in image_dict.items():  # should match exactly
            mapped_dict[item2id[k]] = Path(image_root).joinpath(
                self.code(), v.split('/')[-1])
    
        return mapped_dict
    

class BeautyDataset(AmazonDataset):
    @classmethod
    def code(cls):
        return 'beauty'
    

class ClothingDataset(AmazonDataset):
    @classmethod
    def code(cls):
        return 'clothing'
    

class SportsDataset(AmazonDataset):
    @classmethod
    def code(cls):
        return 'sports'
    

class ToysDataset(AmazonDataset):
    @classmethod
    def code(cls):
        return 'toys'