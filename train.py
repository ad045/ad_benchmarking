# %%
import os
import argparse 

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
plt.style.use('dark_background')

import models as models

import wandb
from os import Path

import models 
import datasets
import dataset

import numpy as np
import time as time 
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.callbacks import EarlyStop


wandb.login()

def get_args_parser():
    parser = argparse.ArgumentParser("NN training")

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    # parser.add_argument('--acum_iter', default=1, type=int) 

    parser.add_argument('--model', default='shallow_conv_net', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    # Model parameters
    parser.add_argument('--input_channels', type=int, default=1, metavar='N',
                        help='input channels')
    parser.add_argument('--input_electrodes', type=int, default=61, metavar='N',
                        help='input electrodes')
    parser.add_argument('--time_steps', type=int, default=200, metavar='N',
                        help='input length')

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
    parser.add_argument('--data_path', default='_.pt', type=str,
                        help='train dataset path')
    parser.add_argument('--labels_path', default='_.pt', type=str,
                        help='train labels path')
    parser.add_argument('--val_data_path', default='', type=str,
                        help='validation dataset path')
    parser.add_argument('--val_labels_path', default='_.pt', type=str,
                        help='validation labels path')
    parser.add_argument('--number_samples', default=None, type=int | str, 
                        help='number of samples on which network should train on. "None" means all samples.')
    parser.add_argument('--length_samples', default=200, 
                        help='length of samples') 
    
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
    



def build_optimizer(model, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    if optimizer == "adamw": 
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate) 

    return optimizer


def train_epoch(model, loader, optimizer, device):

    # # Get dataloaders
    # train_loader = dl.build_dataloader("train", h_params["length_sample"], h_params["batch_size"], h_params["number_samples"])
    # val_loader = dl.build_dataloader("val", h_params["length_sample"], h_params["batch_size"], h_params["number_samples"])
    # test_loader = dl.build_dataloader("test", h_params["length_sample"], h_params["batch_size"], h_params["number_samples"])

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Get model
    # model = models.SimpleClassifierNN(61*100, 256, 32)

    model = models.__dict__[args.model](n_channels = 61, n_classes = 1, input_time_length = 100, n_filters=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15) #, drop_prob=0.5)

    model = models.ShallowConvNet(n_channels = 61, n_classes = 1, input_time_length = 100, n_filters=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15) #, drop_prob=0.5)
    model.to(device)


    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss = F.nll_loss(model(data), target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)




def main(args): 
    args.input_size = (args.input_channels, args.input_electrodes, args.time_steps)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # cudnn.benchmark = True # selects the best out of multiple CNNs

    # Load data     
    def __init__(self, data_path, labels_path, # downstream_task=None, 
                 train=False, number_samples=None, length_samples=200,  args=None) -> None:
        """load data and labels from files"""
        
    dataset_train = EEGDataset(data_path=args.data_path, labels_path=args.labels_path, 
                               train=True, number_samples=args.number_samples, length_samples=args.length_samples,
                               args=args)
    dataset_val = EEGDataset(data_path=args.data_path, labels_path=args.labels_path, 
                               train=True, number_samples=args.number_samples, length_samples=args.length_samples,
                               args=args)
    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_train = torch.utils.data.RandomSampler(dataset_train) 

    # wandb logging
    if args.wandb == True:
        config = vars(args)
        if args.wandb_id:
            wandb.init(project=args.wandb_project, id=args.wandb_id, config=config)
        else:
            wandb.init(project=args.wandb_project, config=config)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        # shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = models.__dict__[args.model](
        input_size=args.input_size, # self, n_channels, n_classes, input_time_length, n_filters=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15, drop_prob=0.5):
        !!!
    )

    models.to(device)

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # n_params_encoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "decoder" not in n)
    # n_params_decoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "decoder" in n)

    # model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    # print('Number of params (M): %.2f' % (n_parameters / 1.e6))
    # print('Number of encoder params (M): %.2f' % (n_params_encoder / 1.e6))
    # print('Number of decoder params (M): %.2f' % (n_params_decoder / 1.e6))

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")

    eval_criterion = "bce"

    best_stats = {'loss':np.inf, 'ncc':0.0}
    best_eval_scores = {'count':0, 'nb_ckpts_max':5, 'eval_criterion':[best_stats[eval_criterion]]}
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        
    train_stats, train_history = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, args=args)
    # if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
    #     misc.save_model(
    #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
    #         loss_scaler=loss_scaler, epoch=epoch)

    val_stats, test_history = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, args=args)
    print(f"Loss / Normalized CC of the network on the {len(dataset_val)} val images: {val_stats['loss']:.4f}\
        / {val_stats['ncc']:.2f}")
    

    best_stats['loss'] = min(best_stats['loss'], val_stats['loss'])
    best_stats['ncc'] = max(best_stats['ncc'], val_stats['ncc'])
    
    if eval_criterion == "loss":
        if early_stop.evaluate_decreasing_metric(val_metric=val_stats[eval_criterion]):
            break
        if args.output_dir and val_stats[eval_criterion] <= max(best_eval_scores['eval_criterion']):
            # save the best 5 (nb_ckpts_max) checkpoints, even if they appear after the best checkpoint wrt time
            if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                best_eval_scores['count'] += 1
            else:
                best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'])
                best_eval_scores['eval_criterion'].pop()
            best_eval_scores['eval_criterion'].append(val_stats[eval_criterion])

            misc.save_best_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion, 
                mode="decreasing")
    else:
        if early_stop.evaluate_increasing_metric(val_metric=val_stats[eval_criterion]):
            break
        if args.output_dir and val_stats[eval_criterion] >= min(best_eval_scores['eval_criterion']):
            # save the best 5 (nb_ckpts_max) checkpoints, even if they appear after the best checkpoint wrt time
            if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                best_eval_scores['count'] += 1
            else:
                best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=True)
                best_eval_scores['eval_criterion'].pop()
            best_eval_scores['eval_criterion'].append(val_stats[eval_criterion])

            misc.save_best_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion, 
                mode="increasing")
            
    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    if args.wandb:
        wandb.log(train_history | test_history | {"Time per epoch [sec]": total_time})

            
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
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)










# # %%
# def train_model_wandb(model, train_loader, val_loader, h_params, device, config=None):
    
#     with wandb.init(config=config):
#         run = wandb.init(
#             project="classifing-eeg", 
#             config={
#                 "learning_rate": h_params["lr"], 
#                 "epochs": h_params["num_epochs"], 
#                 "batch_size": h_params["batch_size"], 
#             },
#         )   
#         config = wandb.config
#         loader = build_dataset(config.batch_size)
#         network = build_network(config.fc_layer_size, config.dropout)
#         optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

#         for epoch in range(config.epochs):

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)

#     criterion = nn.BCELoss()  
#     optimizer = optim.AdamW(model.parameters(), lr=h_params["lr"])

#     for epoch in range(h_params["num_epochs"]): 
#         model.train()
#         train_running_loss = 0.0
#         counter = 0
        
#         for i, (X,y) in enumerate(train_loader):
#             X, y = X.to(device), y.to(device)
            
#             optimizer.zero_grad()
            
#             outputs = model(X)
#             loss = criterion(outputs, y)
            
#             loss.backward()
#             optimizer.step()
            
#             train_running_loss += loss.item() 
#             #np.sqrt(loss.item()) 
            
#             counter += 1

#         wandb.log({"train_loss_accumulating": train_running_loss}) 

            

#         # wandb.log({"train_loss_accumulating": train_running_loss/len(train_loader)}) 

#         # writer.add_scalar('Loss_in_years/train', train_running_loss, epoch * len(train_loader) + i)

#         avg_train_loss = train_running_loss/counter
#         train_acc = 1-avg_train_loss # Is that correct? 

#         # if epoch % rate_of_updates == 0: 
#         #     print(f'Fold {fold+1}, Epoch [{epoch+1}/{h_params["num_epochs"]}], Train Loss: {avg_train_loss:.4f}')
        
#         # print(f'Fold {fold+1}, Epoch [{epoch+1}/{h_params["num_epochs"]}], Train Loss: {avg_train_loss:.4f}')


#         # Validation phase
#         model.eval()
#         val_running_loss = 0.0
#         counter = 0 
        
#         with torch.no_grad():
#             for i, (X,y) in enumerate(val_loader):
#                 X, y = X.to(device), y.to(device)
                
#                 outputs = model(X)
#                 val_loss = criterion(outputs, y)
                
#                 val_running_loss += val_loss.item() 
#                 # np.sqrt(val_loss.item())

#                 # avg_val_loss = val_running_loss / len(val_loader)
#                 # writer.add_scalar('Loss_in_years/val', np.sqrt(val_loss.item()), epoch * len(val_loader) + i)

#                 counter += 1

#         wandb.log({"val_loss_accumulating": val_running_loss}) 



#         avg_val_loss = val_running_loss / counter # len(val_loader)
#         val_acc = 1 - avg_val_loss
        
#         print(f'Epoch [{epoch+1}/{h_params["num_epochs"]}], Validation Loss: {avg_val_loss:.4f}') #Fold {fold+1}
#         wandb.log({"train_accuracy": train_acc, "val_accuracy": val_acc, "train_loss": avg_train_loss, "val_loss": avg_val_loss})


#     # Log fold results
#     # fold_results.append(avg_val_loss)









# %%
def train_model(model, train_loader, val_loader, h_params, device):
    run = wandb.init(
        project="classifing-eeg", 
        config={
            "learning_rate": h_params["lr"], 
            "epochs": h_params["num_epochs"], 
            "batch_size": h_params["batch_size"], 
        },
    )   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    criterion = nn.BCELoss()  
    optimizer = optim.AdamW(model.parameters(), lr=h_params["lr"])

    for epoch in range(h_params["num_epochs"]): 
        model.train()
        train_running_loss = 0.0
        counter = 0
        
        for i, (X,y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item() 
            #np.sqrt(loss.item()) 
            
            counter += 1

        wandb.log({"train_loss_accumulating": train_running_loss}) 

            

        # wandb.log({"train_loss_accumulating": train_running_loss/len(train_loader)}) 

        # writer.add_scalar('Loss_in_years/train', train_running_loss, epoch * len(train_loader) + i)

        avg_train_loss = train_running_loss/counter
        train_acc = 1-avg_train_loss # Is that correct? 

        # if epoch % rate_of_updates == 0: 
        #     print(f'Fold {fold+1}, Epoch [{epoch+1}/{h_params["num_epochs"]}], Train Loss: {avg_train_loss:.4f}')
        
        # print(f'Fold {fold+1}, Epoch [{epoch+1}/{h_params["num_epochs"]}], Train Loss: {avg_train_loss:.4f}')


        # Validation phase
        model.eval()
        val_running_loss = 0.0
        counter = 0 
        
        with torch.no_grad():
            for i, (X,y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                
                outputs = model(X)
                val_loss = criterion(outputs, y)
                
                val_running_loss += val_loss.item() 
                # np.sqrt(val_loss.item())

                # avg_val_loss = val_running_loss / len(val_loader)
                # writer.add_scalar('Loss_in_years/val', np.sqrt(val_loss.item()), epoch * len(val_loader) + i)

                counter += 1

        wandb.log({"val_loss_accumulating": val_running_loss}) 



        avg_val_loss = val_running_loss / counter # len(val_loader)
        val_acc = 1 - avg_val_loss
        
        print(f'Epoch [{epoch+1}/{h_params["num_epochs"]}], Validation Loss: {avg_val_loss:.4f}') #Fold {fold+1}
        wandb.log({"train_accuracy": train_acc, "val_accuracy": val_acc, "train_loss": avg_train_loss, "val_loss": avg_val_loss})


    # Log fold results
    # fold_results.append(avg_val_loss)
