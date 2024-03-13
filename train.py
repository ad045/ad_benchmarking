# %%

import torch
import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt
plt.style.use('dark_background')

import models as models
import dataloader as dl

import wandb

wandb.login()



# %%
def train_model(model, train_loader, val_loader, h_params, device):
    run = wandb.init(
        project="classifing-eeg", 
        config={
            "learning_rate": h_params["lr"], 
            "epochs": h_params["num_epochs"], 
        },
    )   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    criterion = nn.BCELoss()  
    optimizer = optim.AdamW(model.parameters())

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
