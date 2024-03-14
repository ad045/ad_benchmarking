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


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, 
                    criterion,
                    device: torch.device, epoch: int,
                    # loss_scaler,
                    # log_writer=None,
                    args=None):
    model.train(True)

    print_freq = 20

    # accum_iter = args.accum_iter

    optimizer.zero_grad()

    cumu_loss = 0

    for _, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        output = model(data)
        loss = criterion(output, target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch train loss": loss.item()})

    return cumu_loss / len(data_loader)



@torch.no_grad()
def evaluate(model, data_loader, criterion, device, epoch, log_writer=None, args=None):
    model.eval()

    cumu_loss = 0

    for _, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        # ➡ Forward pass
        output = model(data)
        loss = criterion(output, target)
        cumu_loss += loss.item()
    
        wandb.log({"batch val loss": loss.item()})

    return cumu_loss / len(data_loader)
