from pathlib import Path
from abc import *
from constants import *

from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self,
                 args,
                 data_dict_path=None,
                 meta_dict_path=None,
                 image_dict_path=None,
                 ):
        super().__init__()
        self.args = args

        if data_dict_path and meta_dict_path and image_dict_path:
            self.data_dict_path = data_dict_path
            self.meta_dict_path = meta_dict_path
            self.image_dict_path = image_dict_path
        else:
            self.data_dict_path = Path(RAW_DATA_ROOT_FOLDER).joinpath(self.code())
            self.meta_dict_path = Path(RAW_DATA_ROOT_FOLDER).joinpath(self.code())
            self.image_dict_path = Path(RAW_DATA_ROOT_FOLDER).joinpath(self.code())

        print('Loading ratings...')
        self.user_items = self.load_ratings(self.data_dict_path)
        self.user_count = len(self.user_items)
        self.item_count = len(set().union(*self.user_items.values()))

        self.train = {k: v[:-2] for k, v in self.user_items.items()}
        self.val = {k: [v[-2]] for k, v in self.user_items.items()}
        self.test = {k: [v[-1]] for k, v in self.user_items.items()}

        print('Loading meta data...')
        self.user2id, self.item2id, self.id2item = self.load_datamap(self.data_dict_path)
        self.meta_dict = self.load_metadict(self.meta_dict_path, self.item2id)
        self.image_dict = self.load_imagedict(self.image_dict_path,
                                              RAW_IMAGE_ROOT_FOLDER,
                                              self.item2id)

    @classmethod
    def raw_code(cls):
        return cls.code()
    
    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def load_ratings(self):
        pass