# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import numpy as np

import wandb

import util.misc as misc

import matplotlib.pyplot as plt


# def train_one_epoch(model: torch.nn.Module,
#                     data_loader: Iterable, 
#                     optimizer: torch.optim.Optimizer, 
#                     criterion,
#                     device: torch.device, epoch: int,
#                     # loss_scaler,
#                     # log_writer=None,
#                     args=None):
#     model.train(True)

#     # accum_iter = args.accum_iter

#     optimizer.zero_grad()

#     cumu_loss = 0
#     # loss_in_years_item = 0 
#     # loss_in_years = 0 

#     # loss_MAE = torch.nn.L1Loss()

#     still_counting_loss_in_years = True

#     for _, (data, target) in enumerate(data_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()

#         # ➡ Forward pass
#         output = model(data)
#         loss = criterion(output, target)
#         # loss_in_years_item += loss_MAE(output, target)
        
#         # loss_in_years += loss_in_years_item

#         # if loss_in_years_item != np.inf and still_counting_loss_in_years: #
#         #     loss_in_years += loss_in_years_item
#         # else: 
#         #     loss_in_years=np.inf
#         #     still_counting_loss_in_years = False
#         #     loss_in_years = None
            
            
#         cumu_loss += loss.item()



#         # ⬅ Backward pass + weight update
#         loss.backward()
#         optimizer.step()

#         # wandb.log({"train BCE loss per batch": loss.item(),
#         #            "train loss in years per batch": loss_in_years.item(), 
#         #            "epoch": epoch})

        
#     mean_bce_loss = cumu_loss/len(data_loader)
#     # mean_mae_loss = loss_in_years/len(data_loader) if loss_in_years != np.inf else None
        
#     return mean_bce_loss #, mean_mae_loss



# @torch.no_grad()
# def evaluate(model, data_loader, criterion, device, epoch, log_writer=None, args=None):
#     model.eval()

#     cumu_loss = 0
#     # loss_in_years = 0 
#     # loss_in_years_item = 0 
#     # still_counting_loss_in_years = True

#     # loss_MAE = torch.nn.L1Loss()

#     for _, (data, target) in enumerate(data_loader):
#         data, target = data.to(device), target.to(device)

#         # ➡ Forward pass
#         output = model(data)
#         loss = criterion(output, target)
#         # loss_in_years_item = loss_MAE(output, target)
#         # loss_in_years += loss_in_years_item

#         # if loss_in_years_item != np.inf and still_counting_loss_in_years: #
#         #     loss_in_years += loss_in_years_item
#         # else: 
#         #     loss_in_years=np.inf
#         #     still_counting_loss_in_years = False
#         #     loss_in_years = None

#         # print(loss_in_years_item)
#         print(f"Target: {target}, output: {output}")
#         cumu_loss += loss.item()
    
#         # wandb.log({"val BCE loss for batch": loss.item(), 
#         #            "val loss in years per batch": loss_in_years.item(),
#         #            "epoch": epoch})

#     mean_bce_loss = cumu_loss/len(data_loader)
#     # mean_mae_loss = loss_in_years/len(data_loader) if loss_in_years != np.inf else None
        
#     return target, output, mean_bce_loss #, mean_mae_loss



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, 
                    criterion,
                    device: torch.device, epoch: int,
                    # loss_scaler,
                    # log_writer=None,
                    scaled=True,
                    args=None):
    
    model.train(True)
    optimizer.zero_grad()
    running_loss = 0.0
    correct = 0 
    total = 0 

    for _, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        output = model(data)
        loss = criterion(output, target)

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        _, target_index = torch.max(target, 1)
        total += target.size(0) 
        correct += (predicted == target_index).sum().item()

    train_loss = running_loss / len(data_loader)
    train_acc = correct / total
        
    return train_loss, train_acc


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, args=None): #, epoch, scaled=True, log_writer=None, args=None):
    
    model.eval()
    running_loss = 0.0
    correct = 0 
    total = 0 

    for _, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        # ➡ Forward pass
        output = model(data)
        loss = criterion(output, target)
       
        # print(f"Target: {target}, output: {output}")

        running_loss += loss.item()

        _, predicted = torch.max(output, 1)
        _, target_index = torch.max(target, 1)

        total += target.size(0) 
        correct += (predicted == target_index).sum().item()

    val_loss = running_loss / len(data_loader)
    val_acc = correct / total
        
    return val_loss, val_acc
    