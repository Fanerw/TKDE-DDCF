from .base import AbstractDataset

import pandas as pd

from datetime import date


class ML100KDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-100k'

    @classmethod
    def url(cls):
        pass

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['ML_data_100k.csv']


    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ML_data_100k.csv')
        df = pd.read_csv(file_path, header=0)
        df.columns = ['_','uid', 'sid', 'rating']
        return df


