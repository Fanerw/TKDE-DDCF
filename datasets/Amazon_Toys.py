from .base import AbstractDataset

import pandas as pd

from datetime import date


class AmazonToys(AbstractDataset):
    @classmethod
    def code(cls):
        return 'Amazon_Toys'

    @classmethod
    def url(cls):
        pass

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['Amazon_Toys.csv',]

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('Amazon_Toys.csv')
        df = pd.read_csv(file_path, sep=',', header=None)
        df.columns = [ 'sid','uid', 'rating']
        return df


