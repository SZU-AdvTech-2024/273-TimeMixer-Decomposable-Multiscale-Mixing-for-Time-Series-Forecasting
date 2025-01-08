import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import Normalizer, interpolate_missing
import warnings
import h5py
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self,root_path='/home1/hwj/TimeMixer-main/dataset/SST'):
        self.root_path = root_path
        data_sst = h5py.File(os.path.join(self.root_path,'201101.h5'))['sst']
        # test_dataset = h5py.File(os.path.join(self.root_path,'201102.h5'))['sst']
        self.data_sst = data_sst
        # print(data_sst.shape)


    # 返回给定索引处的样本
    def __getitem__(self, index, seq_len=3, pred_len=2):
        self.seq_len = seq_len
        x_sst = self.data_sst[index:index+seq_len]
        y_sst = self.data_sst[index+seq_len:index+seq_len+pred_len]

        return x_sst, y_sst
    # 返回数据集的总样本数
    def __len__(self):
        return self.data_sst.shape[0] - 4


if __name__ == '__main__':
    dataset = Dataset_Custom()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    # matplotlib.use('Agg')
    for x, y in dataloader:
        # print(x.shape, y.shape)
        x_plot = x.cpu().detach().numpy()
        x_plot = x_plot[0, 0, ::-1]
        x_plot = x_plot[720:840, 720:840]
        # print(x_plot.shape)
        plt.imshow(x_plot)
        plt.show()
        break

        # /home5/yyz/data/Sea_Surface/orModas
        # /home8/ty/data/adt/adt_2010_global.h5
        # /home8/ty/data/../..._2010_global.h5

