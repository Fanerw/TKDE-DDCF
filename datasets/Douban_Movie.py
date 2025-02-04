from .base import AbstractDataset

import pandas as pd

from datetime import date


class DoubanMovieDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'Douban_Movie'

    @classmethod
    def url(cls):
        pass

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['moviereviews_cleaned.txt']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('moviereviews_cleaned.txt')
        df = pd.read_csv(file_path,sep='\s+')
        df.columns = ["uid",	"sid",	"rating",	"comment",	"timestamp",	"labels",	"useful_num"	,"CategoryID",	"ID"]
        return df


