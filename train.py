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
    parser.add_argument('--epochs', default=400, type=int)
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
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate') 

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')


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
    parser.add_argument('--number_samples', default=1, type=int, # | str, 
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
    
    # parser.add_argument('--mode', type=str, default="train")

    return parser
    



# def build_optimizer(model, optimizer, learning_rate):
#     if optimizer == "sgd":
#         optimizer = optim.SGD(model.parameters(),
#                               lr=learning_rate, momentum=0.9)
#     elif optimizer == "adam":
#         optimizer = optim.Adam(model.parameters(),
#                                lr=learning_rate)
#     if optimizer == "adamw": 
#         optimizer = optim.AdamW(model.parameters(), lr=learning_rate) 

#     return optimizer


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
        # n_filters=args.n_filters,         # in konkreten Modellen definiert
        # filter_time_length=args.filter_time_length, 
        # pool_time_length=args.pool_time_length, 
        # pool_time_stride=args.pool_time_stride,
        # n_classes=args.n_classes, 
    )

    model.to(device)

    # eval_criterion = "bce"
    # criterion = nn.BCELoss()  # !!!! 

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")


    for epoch in range(args.epochs): 
        # start_time = time.time()
        
        mean_loss_epoch_train = train_one_epoch(model, data_loader_train, optimizer, device, epoch, args=args) #loss_scaler, criterion
        print(f"Loss / BCE on {len(dataset_train)} train samples: {mean_loss_epoch_train}")

        mean_loss_epoch_val = evaluate(model, data_loader_val, device, epoch, args=args) 
        print(f"Loss / BCE on {len(dataset_val)} val samples: {mean_loss_epoch_val}")
    
        
        # # wandb
        # if args.wandb == True:
        #     test_history['epoch'] = epoch
        #     test_history['val_loss'] =mean_loss_epoch_train 



    # total_time = time.time() - start_time
    # if args.wandb:
    #     wandb.log(train_history | test_history | {"Time per epoch [sec]": total_time})

            
    # ------

    # argparser = argpars()

    # optimizer = None
    # criterion = None # loss

    # if mode== "train":
    #     dataset = EEGDataset()
    #     train_loader = Dataloader(dataset)

    # model = None

    # train_epoch(model, optimozer, criterion, dataloader)




if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)





# model = models.__dict__[args.model](n_channels = 61, n_classes = 1, input_time_length = 100, n_filters=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15) #, drop_prob=0.5)

# model = models.ShallowConvNet(n_channels = 61, n_classes = 1, input_time_length = 100, n_filters=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15) #, drop_prob=0.5)
# model.to(device)


# cumu_loss = 0
# for _, (data, target) in enumerate(loader):
#     data, target = data.to(device), target.to(device)
#     optimizer.zero_grad()

#     # ➡ Forward pass
#     loss = F.nll_loss(model(data), target)
#     cumu_loss += loss.item()

#     # ⬅ Backward pass + weight update
#     loss.backward()
#     optimizer.step()

#     wandb.log({"batch loss": loss.item()})

# return cumu_loss / len(loader)



