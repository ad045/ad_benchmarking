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





# def build_transform(): 

folder_path = "/vol/aimspace/users/dena/Documents/mae/data/lemon"

data_raw_test = torch.load(os.path.join(folder_path, "data_raw_test.pt"))
data_raw_train = torch.load(os.path.join(folder_path, "data_raw_train.pt"))
data_raw_val = torch.load(os.path.join(folder_path, "data_raw_val.pt"))

# labels_raw_test = torch.load(os.path.join(folder_path, "labels_raw_test.pt"))
# labels_raw_train = torch.load(os.path.join(folder_path, "labels_raw_train.pt"))
# labels_raw_val = torch.load(os.path.join(folder_path, "labels_raw_val.pt"))

folder_path_classification = "/vol/aimspace/users/dena/Documents/ad_benchmarking/ad_benchmarking/data"
labels_raw_train = torch.load(os.path.join(folder_path_classification, "labels_bin_train.pt"))
labels_raw_val = torch.load(os.path.join(folder_path_classification, "labels_bin_val.pt"))
labels_raw_test = torch.load(os.path.join(folder_path_classification, "labels_bin_test.pt"))

# %%
class EEGDataset(Dataset):

    def __init__(self, data_path, labels_path, # downstream_task=None, 
                 train=False, number_samples=None, length_samples=200,  args=None) -> None:
        """load data and labels from files"""
        
        # self.downstream_task = downstream_task # "classification"
        
        self.train = train 
        self.args = args

        self.X = torch.load(data_path, map_location=torch.device('cpu')) # load to ram
        self.y = torch.load(labels_path, map_location=torch.device('cpu'))#[..., None] # load to ram

    def __init__(self, data_path=args.data_path, train=True, args=args):
        # X, y, length_sample, number_samples=None):

        if number_samples:
            indices = [i for i in range(number_samples)] #np.random.randint(0, number_samples) #.choice(range(len(X[0])), number_samples, replace=False)
            np.random.shuffle(indices) 
            # indices = range(number_samples) #np.random.randint(0, number_samples) #.choice(range(len(X[0])), number_samples, replace=False)
            # indices.shuffle() 
            self.X = X[indices]
            self.y = y[indices]
        else: 
            self.X = X
            self.y = y

        self.length_sample = length_sample

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""

        # get random starting point
        last_useful_index = self.X.shape[-1]-self.length_sample
        index = np.random.randint(0,last_useful_index)

        # get 2000 timesteps long data 
        data = self.X[idx][:,index:index+self.length_sample]
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

        # if self.downstream_task == 'classification':
        #     label = label.type(torch.LongTensor).argmax(dim=-1)
        #     label_mask = torch.ones_like(label)


        return data, label
    


    

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""
        if self.downstream_task == 'regression':
            data, label, label_mask = self.data[idx], self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd], self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            data, label, label_mask = self.data[idx], self.labels[idx], self.labels_mask[idx]
        
        if self.train == False:
            transform = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False),
                # transformations.PowerSpectralDensity(fs=100, nperseg=1000, return_onesided=False),
                # transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise")
            ])
        else:
            transform = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
                # transformations.PowerSpectralDensity(fs=100, nperseg=1000, return_onesided=False),
                # transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
                augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5),
                augmentations.Jitter(sigma=self.args.jitter_sigma),
                augmentations.Rescaling(sigma=self.args.rescaling_sigma),
                # augmentations.TimeFlip(prob=0.33),
                # augmentations.SignFlip(prob=0.33)
            ])
        data = transform(data)
        
        if self.downstream_task == 'classification':
            label = label.type(torch.LongTensor).argmax(dim=-1)
            label_mask = torch.ones_like(label)

        return data, label, label_mask


# %%
    
def build_dataloader(set_type="train", length_sample = 200, batch_size=4, number_samples=None):
    if set_type == "train": 
        dataset = EEGDataset(data_raw_train, labels_raw_train, length_sample, number_samples) #, h_params["number_samples"]) #, number_samples=32) #!!! CHANGE AGAIN!! 

    elif set_type == "val": 
        dataset = EEGDataset(data_raw_val, labels_raw_val, length_sample, number_samples) 

    elif set_type == "test": 
        dataset = EEGDataset(data_raw_test, labels_raw_test, length_sample, number_samples) #, h_params["number_samples"]) #, number_samples=32) #!!! CHANGE AGAIN!! 
    else: 
        print("ERROR - this set_type is unknown.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader




# %%
# import matplotlib.pyplot as plt

# for data in enumerate(train_dataset): 
#     index = data[0]
#     X = data[1][0]
#     y = data[1][1]

#     print(index)
#     print(X.shape)
#     print(y.shape) 
#     plt.plot(X)
#     break

# %%



