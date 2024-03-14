# # %%


# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset

# # import util.transformations as transformations
# # import util.augmentations as augmentations

# import EEGDataset from dataset


# # %%
# def build_dataset(is_train, args=None):
#     # transform = build_transform(is_train, args)

#     if is_train == True: 
#         data_raw_train = torch.load(args.data_path) #os.path.join(folder_path, "data_raw_train.pt"))
#         labels_raw_train = torch.load(args.labels_path)
#         dataset = EEGDataset(data_raw_train, labels_raw_train, args.time_steps, args.number_samples) #, h_params["number_samples"]) #, number_samples=32) #!!! CHANGE AGAIN!! 

#     elif is_train == False: 
#         data_raw_val = torch.load(args.val_data_path) #os.path.join(folder_path, "data_raw_train.pt"))
#         labels_raw_val = torch.load(args.val_labels_path)
#         dataset = EEGDataset(data_raw_val, labels_raw_val, args.time_steps, args.number_samples) #, h_params["number_samples"]) #, number_samples=32) #!!! CHANGE AGAIN!! 

#     print(dataset)

#     return dataset




# # def build_transform(): 

# # folder_path = "/vol/aimspace/users/dena/Documents/mae/data/lemon"

# # data_raw_test = torch.load(os.path.join(folder_path, "data_raw_test.pt"))
# # data_raw_train = torch.load(os.path.join(folder_path, "data_raw_train.pt"))
# # data_raw_val = torch.load(os.path.join(folder_path, "data_raw_val.pt"))

# # labels_raw_test = torch.load(os.path.join(folder_path, "labels_raw_test.pt"))
# # labels_raw_train = torch.load(os.path.join(folder_path, "labels_raw_train.pt"))
# # labels_raw_val = torch.load(os.path.join(folder_path, "labels_raw_val.pt"))

# # folder_path_classification = "/vol/aimspace/users/dena/Documents/ad_benchmarking/ad_benchmarking/data"
# # labels_raw_train = torch.load(os.path.join(folder_path_classification, "labels_bin_train.pt"))
# # labels_raw_val = torch.load(os.path.join(folder_path_classification, "labels_bin_val.pt"))
# # labels_raw_test = torch.load(os.path.join(folder_path_classification, "labels_bin_test.pt"))



# # %%
    
# # def build_dataloader(set_type="train", length_sample = 200, batch_size=4, number_samples=None):
# #     if set_type == "train": 
# #         dataset = EEGDataset(data_raw_train, labels_raw_train, length_sample, number_samples) #, h_params["number_samples"]) #, number_samples=32) #!!! CHANGE AGAIN!! 

# #     elif set_type == "val": 
# #         dataset = EEGDataset(data_raw_val, labels_raw_val, length_sample, number_samples) 

# #     elif set_type == "test": 
# #         dataset = EEGDataset(data_raw_test, labels_raw_test, length_sample, number_samples) #, h_params["number_samples"]) #, number_samples=32) #!!! CHANGE AGAIN!! 
# #     else: 
# #         print("ERROR - this set_type is unknown.")

# #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# #     return loader




# # %%
# # import matplotlib.pyplot as plt

# # for data in enumerate(train_dataset): 
# #     index = data[0]
# #     X = data[1][0]
# #     y = data[1][1]

# #     print(index)
# #     print(X.shape)
# #     print(y.shape) 
# #     plt.plot(X)
# #     break

# # %%



