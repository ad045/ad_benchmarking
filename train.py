# %%
import os
import argparse 

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import json as json

import torch.optim as optim

import matplotlib.pyplot as plt
plt.style.use('dark_background')

import models as models

import wandb
# from os import Path

import models 
import datasets
import dataset

import numpy as np
import time as time 
import util.misc as misc
# from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.callbacks import EarlyStop

from util.engine_train import train_one_epoch, evaluate # evaluate_online


wandb.login()


def get_args_parser():
    parser = argparse.ArgumentParser("NN training")

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--acum_iter', default=1, type=int) 

    parser.add_argument('--model', default='shallow_conv_net', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    # Model parameters
    parser.add_argument('--input_channels', type=int, default=1, metavar='N',
                        help='input channels')
    parser.add_argument('--input_electrodes', type=int, default=61, metavar='N',
                        help='input electrodes')
    parser.add_argument('--time_steps', type=int, default=100, metavar='N',
                        help='input length')
    # parser.add_argument('--length_samples', default=200, 
    #                     help='length of samples') 

    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default="adam_w", 
                        help='optimizer type') 
    parser.add_argument('--criterion', type=str, default="bce", 
                        help='loss type') 
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate') 

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')
    parser.add_argument('--sufficient_accuracy', default=-np.inf, type=float,
                        help='Sufficient accuracy that also leads to early stopping (default: -inf)')
 



    # Dataset parameters
    parser.add_argument('--data_path', 
                        # default='_.pt',
                        default="/vol/aimspace/users/dena/Documents/mae/data/lemon/data_raw_train.pt",
                        type=str,
                        help='train dataset path')

    parser.add_argument('--labels_path', 
                        # default='_.pt', 
                        default="/vol/aimspace/users/dena/Documents/ad_benchmarking/ad_benchmarking/data/labels_bin_train.pt", #labels_raw_train.pt",
                        type=str,
                        help='train labels path')
    parser.add_argument('--val_data_path', 
                        # default='', 
                        default="/vol/aimspace/users/dena/Documents/mae/data/lemon/data_raw_val.pt",
                        type=str,
                        help='validation dataset path')
    parser.add_argument('--val_labels_path', 
                        # default='_.pt', 
                        default="/vol/aimspace/users/dena/Documents/ad_benchmarking/ad_benchmarking/data/labels_bin_val.pt", # "labels_raw_val.pt"
                        type=str,
                        help='validation labels path')
    parser.add_argument('--number_samples', default=4, type=int, # | str, 
                        help='number of samples on which network should train on. "None" means all samples.')
    parser.add_argument('--num_workers', default=4, type=int, # | str, 
                        help='number workers for dataloader.')
    
    
    # Wandb parameters
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', default='',
                        help='project where to wandb log')
    parser.add_argument('--wandb_id', default='', type=str,
                        help='id of the current run')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # Saving Parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    
    return parser
    

def main(args): 

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_train = dataset.EEGDataset(data_path=args.data_path, labels_path=args.labels_path, 
                               train=True, number_samples=args.number_samples, length_samples=args.time_steps,
                               args=args)
    dataset_val = dataset.EEGDataset(data_path=args.data_path, labels_path=args.labels_path, 
                               train=True, number_samples=args.number_samples, length_samples=args.time_steps,
                               args=args)

    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_train = torch.utils.data.RandomSampler(dataset_train) 

    # # wandb logging
    # if args.wandb == True:
    #     config = vars(args)
    #     if args.wandb_id:
    #         wandb.init(project=args.wandb_project, id=args.wandb_id, config=config)
    #     else:
    #         wandb.init(project=args.wandb_project, config=config)
    wandb.init(project=args.wandb_project, config=vars(args))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        # shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = models.__dict__[args.model](
        n_channels=args.input_electrodes, 
        input_time_length=args.time_steps, 
    )

    model.to(device)

    # eval_criterion = "bce"
    criterion = torch.nn.BCELoss() 


    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr)
    elif args.optimizer == "adamw": 
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    else: 
        print("Attention: No optimier chosen.")

    # Define callbacks
    # early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")

    min_val_metric = np.inf
    counter = 0 

    for epoch in range(args.epochs): 
        
        mean_loss_epoch_train = train_one_epoch(model, data_loader_train, optimizer, criterion, device, epoch, args=args) #loss_scaler, criterion
        print(f"Loss / BCE on {len(dataset_train)} train samples: {mean_loss_epoch_train}")

        mean_loss_epoch_val = evaluate(model, data_loader_val, criterion, device, epoch, args=args) 
        print(f"Loss / BCE on {len(dataset_val)} val samples: {mean_loss_epoch_val}")

        # Early Stopping
        if args.patience > -1: 
            if mean_loss_epoch_val < min_val_metric: 
                min_val_metric = mean_loss_epoch_val
                counter == 0
            elif mean_loss_epoch_val > min_val_metric: 
                counter += 1
                if counter > args.patience:
                    print(f"stopped early at epoch {epoch}.")
                    break 

        wandb.log({"epoch: ", epoch})

    



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)

