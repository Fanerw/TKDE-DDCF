from .base import AbstractDataset

import pandas as pd

from datetime import date


class DoubanBookDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'Douban_Book'

    @classmethod
    def url(cls):
        pass

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['bookreviews_cleaned.txt']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('bookreviews_cleaned.txt')
        df = pd.read_csv(file_path,sep='\s+')
        df.columns = ['uid', 'sid', 'rating','labels',"comment", 'timestamp'	,"ID"]
        return df


