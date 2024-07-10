from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils
from scipy import sparse
import numpy as np
from copy import deepcopy
import pandas as pd

class AEDataloader(AbstractDataloader):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        self.rating_values = dict(self.rating.groupby('uid').progress_apply(lambda d: list(d['rating'])))
        self.rating = dict(self.rating.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        self.train_rating=dict(self.train.groupby('uid').progress_apply(lambda d: list(d['rating'])))
        self.train = dict(self.train.groupby('uid').progress_apply(lambda d: list(d['sid'])))

        self.val_rating=dict(self.val.groupby('uid').progress_apply(lambda d: list(d['rating'])))
        self.val=dict(self.val.groupby('uid').progress_apply(lambda d: list(d['sid'])))

        self.test_rating=dict(self.test.groupby('uid').progress_apply(lambda d: list(d['rating'])))
        self.test = dict(self.test.groupby('uid').progress_apply(lambda d: list(d['sid'])))


        unique_train_users=set()
        unique_train_items=set()
        for user, items in self.train.items():
            unique_train_users.add(user)
            unique_train_items.update(items)


        unique_val_users = set()
        unique_val_items = set()
        for user, items in self.val.items():
            unique_val_users.add(user)
            unique_val_items.update(items)


        unique_test_users = set()
        unique_test_items = set()
        for user, items in self.test.items():
            unique_test_users.add(user)
            unique_test_items.update(items)


        rating_num=0
        for useritem in self.train.values():
            rating_num+=len(useritem)

        col_users=unique_train_users&unique_val_users&unique_test_users
        col_items=unique_train_items&unique_val_items&unique_test_items



        rating_set = {}
        rating_values_set = {}
        for user, items in self.rating.items():
            if user in col_users:
                temp = []
                ratings = []
                for idx, item in enumerate(items):
                    if item in col_items:
                        temp.append(item)
                        ratings.append(self.rating_values[user][idx])
                # assert len(temp) == 0
                rating_set[user] = temp
                rating_values_set[user] = ratings


        train_set={}
        train_rating_set = {}
        for user,items in self.train.items():
            if user in col_users:
                temp=[]
                ratings=[]
                for idx,item in enumerate(items):
                    if item in col_items:
                        temp.append(item)
                        ratings.append(self.train_rating[user][idx])
                # assert len(temp) == 0
                train_set[user]=temp
                train_rating_set[user]=ratings

        val_set = {}
        val_rating_set = {}
        for user, items in self.val.items():
            if user in col_users:
                temp = []
                ratings=[]
                for idx,item in enumerate(items):
                    if item in col_items:
                        temp.append(item)
                        ratings.append(self.val_rating[user][idx])
                val_set[user] = temp
                val_rating_set[user]=ratings

        test_set = {}
        test_rating_set={}
        for user, items in self.test.items():
            if user in col_users:
                temp = []
                ratings=[]
                for idx,item in enumerate(items):
                    if item in col_items:
                        temp.append(item)
                        ratings.append(self.test_rating[user][idx])
                test_set[user] = temp
                test_rating_set[user]=ratings

        self.rating=rating_set
        self.rating_values=rating_values_set
        self.train=train_set
        self.train_rating=train_rating_set
        self.val=val_set
        self.val_rating=val_rating_set
        self.test=test_set
        self.test_rating=test_rating_set


        self.umap={u:i for i, u in enumerate(col_users)}
        self.smap={s:i for i, s in enumerate(col_items)}
        self.user_count=len(col_users)
        self.item_count=len(col_items)

        args.num_items = self.item_count
        args.num_users=self.user_count


        remap_items=lambda items: [self.smap[item] for item in items]
        self.rating={self.umap[user] : remap_items(items) for user, items in self.rating.items()}
        self.rating_values={self.umap[user]:ratings for user, ratings in self.rating_values.items()}
        self.train={self.umap[user] : remap_items(items) for user, items in self.train.items()}
        self.train_rating={self.umap[user]:ratings for user, ratings in self.train_rating.items()}
        self.val={self.umap[user] : remap_items(items) for user, items in self.val.items()}
        self.val_rating = {self.umap[user]: ratings for user, ratings in self.val_rating.items()}
        self.test={self.umap[user]:remap_items(items) for user, items in self.test.items()}
        self.test_rating = {self.umap[user]: ratings for user, ratings in self.test_rating.items()}





    @classmethod
    def code(cls):
        return 'ae'

    def get_pytorch_dataloaders(self):
        self._get_rating_dataset()
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):

        dataset = self._get_train_dataset()
        if self.args.dual:
            t_dataset=deepcopy(dataset)
            x=dataset.data
            x=torch.t(x)
            t_dataset.data=x
            return (data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=False, pin_memory=True),data_utils.DataLoader(t_dataset, batch_size=self.args.train_batch_size,
                                           shuffle=False, pin_memory=True))

        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_val_loader(self):
        dataset=self._get_val_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_test_loader(self):
        dataset=self._get_test_dataset()
        dataloader=data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_rating_dataset(self):
        AEDataset(self.rating,self.rating_values,self.user_count,self.item_count,path='Data/'+self.args.dataset_code+'/rating.npz')


    def _get_train_dataset(self):
        dataset = AEDataset(self.train,self.train_rating,self.user_count,self.item_count,path='Data/'+self.args.dataset_code+'/train.npz')
        return dataset

    def _get_val_dataset(self):
        dataset = AEDataset(self.val,self.val_rating,self.user_count,self.item_count,path='Data/'+self.args.dataset_code+'/val.npz')
        return dataset

    def _get_test_dataset(self):
        dataset = AEDataset(self.test,self.test_rating,self.user_count,self.item_count,path='Data/'+self.args.dataset_code+'/test.npz')
        return dataset







class AEDataset(data_utils.Dataset):
    def __init__(self, user2item,user2rating, user_count, item_count,path):

        user_row = []
        for user, useritem in user2item.items():
            for _ in range(len(useritem)):
                user_row.append(user)
        item_col = []
        for useritem in user2item.values():
            item_col.extend(useritem)


        ratings=[]
        for userratings in user2rating.values():
            ratings.extend(userratings)

        assert len(user_row) == len(item_col) & len(user_row)==len(ratings)

        sparse_data = sparse.csr_matrix((np.array(ratings), (user_row, item_col)), dtype='float64',
                                        shape=(user_count, item_count))
        sparse.save_npz(path, sparse_data)
        self.data = torch.FloatTensor(sparse_data.toarray())


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]



