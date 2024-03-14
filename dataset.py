# %%

import os
import sys
import numpy as np
import pandas as pd
from typing import Any, Tuple

from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# import util.transformations as transformations
# import util.augmentations as augmentations

# %%
class EEGDataset(Dataset):

    def __init__(self, data_path, labels_path, # downstream_task=None, 
                 train=False, number_samples=None, length_samples=100, args=None) -> None:
        """load data and labels from files"""
        
        # self.downstream_task = downstream_task # "classification"
        
        self.train = train 
        self.args = args
        self.number_samples = number_samples

        self.X = torch.load(data_path, map_location=torch.device('cpu')) # load to ram
        self.y = torch.load(labels_path, map_location=torch.device('cpu'))#[..., None] # load to ram

        # X, y, length_sample, number_samples=None):

        if number_samples:
            indices = [i for i in range(number_samples)] #np.random.randint(0, number_samples) #.choice(range(len(X[0])), number_samples, replace=False)
            np.random.shuffle(indices) 
            # indices = range(number_samples) #np.random.randint(0, number_samples) #.choice(range(len(X[0])), number_samples, replace=False)
            # indices.shuffle() 
            self.X = self.X[indices]
            self.y = self.y[indices]
        else: 
            self.X = self.X
            self.y = self.y

        self.length_samples = length_samples

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""

        # get random starting point
        last_useful_index = self.X.shape[-1]-self.length_samples
        index = np.random.randint(0,last_useful_index)

        # get 2000 timesteps long data 
        data = self.X[idx][:,index:index+self.length_samples]
        label = self.y[idx]


        # if self.train == False:
        #     transform = transforms.Compose([
        #         augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False),
        #         # transformations.PowerSpectralDensity(fs=100, nperseg=1000, return_onesided=False),
        #         # transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise")
        #     ])
        # else:
        #     transform = transforms.Compose([
        #         augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
        #         # transformations.PowerSpectralDensity(fs=100, nperseg=1000, return_onesided=False),
        #         # transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
        #         augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5),
        #         augmentations.Jitter(sigma=self.args.jitter_sigma),
        #         augmentations.Rescaling(sigma=self.args.rescaling_sigma),
        #         # augmentations.TimeFlip(prob=0.33),
        #         # augmentations.SignFlip(prob=0.33)
        #     ])
        # data = transform(data)



        return data, label
    
