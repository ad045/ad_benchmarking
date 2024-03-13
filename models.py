import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifierNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):

        super().__init__()

        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer; Input size is 61*2000 since we flatten the cropped tensor
        # self.maxpool1 = nn.Maxpool2D()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
        self.fc3 = nn.Linear(hidden_size2, 1)  # Output layer for regression
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.final_activation(x)
        return x


# class InspiredEEGNet(nn.Module): # inspired by Shallow ConvNet model from Schirrmeister et al 2017.
#     def __init__(self, n_channels, n_classes, input_time_length, n_filters=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15, drop_prob=0.5):
#         super(InspiredEEGNet, self).__init__()
        
#         self.temporal_conv = nn.Conv2d(n_channels, n_filters, (1, filter_time_length), padding='same')
#         self.batch_norm = nn.BatchNorm2d(n_filters)
#         # self.spatial_conv = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False) # assuming n_channels is the spatial dimension
#         self.spatial_conv = nn.Conv2d(n_filters, n_filters, (1, 1), bias=False) # assuming n_channels is the spatial dimension
#         # self.pooling = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride) 
#         self.pooling = nn.AvgPool2d((1,2))# stride=(1,3)) #pool_stride_length, stride=pool_time_stride) #), ceil_mode=True)

#         self.dropout = nn.Dropout(drop_prob)
#         # Regression?/ Classifier convolutional layer, assuming the output from previous layer is flattened
#         self.classifier = nn.Conv2d(n_filters, n_classes, (1, input_time_length // pool_time_stride - pool_time_length // pool_time_stride + 1))

#     def forward(self, x):
#         x = F.relu(self.temporal_conv(x))
#         x = self.batch_norm(x)
#         x = F.relu(self.spatial_conv(x))        
#         x = self.pooling(x)
#         x = self.dropout(x)
#         x = self.classifier(x)
#         # Flatten the output for classification
#         x = x.squeeze(3).mean(2)
#         return x # F.log_softmax(x, dim=1)



# class SimpleNN(nn.Module):
#     def __init__(self, activation=nn.Sigmoid(),
#                  input_size=61*2000, hidden_size1 = h_params["hidden_size1"], hidden_size2=h_params["hidden_size2"]):

#         super().__init__()

#         self.flatten = nn.Flatten() 
#         self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer; Input size is 61*2000 since we flatten the cropped tensor
#         # self.maxpool1 = nn.Maxpool2D()
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
#         self.fc3 = nn.Linear(hidden_size2, 1)  # Output layer for regression
#         self.activation = activation

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.fc2(x)
#         x = self.activation(x)
#         x = self.fc3(x)
#         return x