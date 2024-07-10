from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        self.save_folder = dataset._get_preprocessed_folder_path()
        rating_set, train_set, val_set, test_set, statistics = dataset.load_dataset()
        self.rating=rating_set
        self.train = train_set
        self.val = val_set
        self.test = test_set


    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
