# %%

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# import util.transformations as transformations
# import util.augmentations as augmentations


# %%
def build_dataset(is_train, args):
    # transform = build_transform(is_train, args)

    folder_path = "/vol/aimspace/users/dena/Documents/mae/data/lemon"

    if is_train == True: 
        data_raw_train = torch.load(os.path.join(folder_path, "data_raw_train.pt"))

    elif is_train == False: 
        data_raw_train = torch.load(os.path.join(folder_path, "data_raw_train.pt"))

    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    # dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset




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

    def __getitem__(self, idx):

        # get random starting point
        last_useful_index = self.X.shape[-1]-self.length_sample
        index = np.random.randint(0,last_useful_index)

        # get 2000 timesteps long data 
        participant_trials = self.X[idx][:,index:index+self.length_sample]
        label = self.y[idx]
        return participant_trials, label


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



