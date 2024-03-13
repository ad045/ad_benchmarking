# %%
import os
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# %%
length_sample = 100 #2000
number_samples = 1

# %%
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
    def __init__(self, X, y, number_samples=None):
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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        # get random starting point
        last_useful_index = self.X.shape[-1]-length_sample
        index = np.random.randint(0,last_useful_index)

        # get 2000 timesteps long data 
        participant_trials = self.X[idx][:,index:index+length_sample]
        label = self.y[idx]
        return participant_trials, label


train_dataset = EEGDataset(data_raw_train, labels_raw_train, number_samples) #, h_params["number_samples"]) #, number_samples=32) #!!! CHANGE AGAIN!! 
val_dataset = EEGDataset(data_raw_val, labels_raw_val, number_samples) #, h_params["number_samples"]) #, number_samples=32) #!!! CHANGE AGAIN!! 
test_dataset = EEGDataset(data_raw_test, labels_raw_test, number_samples) #, h_params["number_samples"]) #, number_samples=32) #!!! CHANGE AGAIN!! 

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



