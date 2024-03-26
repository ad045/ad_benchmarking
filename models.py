import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleClassifierNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):

        super().__init__()

        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer; Input size is 61*2000 since we flatten the cropped tensor
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


class SimpleRegressorNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):

        super().__init__()

        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer; Input size is 61*2000 since we flatten the cropped tensor
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
        self.fc3 = nn.Linear(hidden_size2, 1)  # Output layer for regression
        self.activation = nn.ReLU()
        # self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        # x = self.final_activation(x)
        return x



class DeepConvNet(nn.Module): # Schirrmeister et al 2017.
    def __init__(self, n_channels=1, input_time_length=100, n_classes=1):
        #n-channels is useless! Replace it! 
        super(DeepConvNet, self).__init__()
        
        n_filters = 25
        filter_time_length = 10

        self.temporal_conv = nn.Conv2d(1, 25, (1, filter_time_length), padding='valid') # same')
        self.spatial_conv = nn.Conv2d(25, 25, (61, 1), padding='valid') #, bias=False) 
        # batchnorm, elu
        self.batch_norm1 = nn.BatchNorm2d(25)
        self.pooling = nn.MaxPool2d((1, 3), stride=(3,1))

        self.conv_2 = nn.Conv2d(25, 50, (1, 10), padding='valid') #same') #, bias=False) 
        # batchnorm, elu
        self.batch_norm2 = nn.BatchNorm2d(50)
        self.pool_2 = nn.MaxPool2d((1, 3), stride=(3,1)) 
        
        self.conv_3 = nn.Conv2d(50, 100, (1, 10), padding='valid') #, bias=False) 
        self.batch_norm3 = nn.BatchNorm2d(100)
        # batchnorm, elu
        self.pool_3 = nn.MaxPool2d((1, 3), stride=(3,1)) 
        
        self.conv_4 = nn.Conv2d(100, 200, (1, 10), padding='valid') #, bias=False) 
        self.batch_norm4 = nn.BatchNorm2d(200)
        # batchnorm, elu
        self.pool_4 = nn.MaxPool2d((1, 3), stride=(3,1)) 
        
        self.elu = nn.ELU()
        # self.batch_norm = nn.BatchNorm2d()

        self.flatten = nn.Flatten() 
        # last_layer_time_length = int((int(input_time_length-10+1/2
        # self.linear_classification = nn.Linear(last_layer_time_length*40, n_classes)
        # self.final_activation = nn.Sigmoid() # in Braindecode Paper a Softmax

        self.linear_classification = nn.Linear(33200, 1)


    def forward(self, x):
        x = x.unsqueeze(1).permute(0,1,2,3)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)  

        x = self.batch_norm1(x)
        x = self.elu(x)
        x = self.pooling(x) 

        x = self.conv_2(x)
        x = self.batch_norm2(x)
        x = self.elu(x)
        x = self.pool_2(x)

        x = self.conv_3(x)


        x = self.batch_norm3(x)
        x = self.elu(x)
        x = self.pool_3(x)

        x = self.conv_4(x)

        x = self.batch_norm4(x)
        x = self.elu(x)

        x = self.pool_4(x)

        x = self.flatten(x) 

        x = self.linear_classification(x)
        # # x = self.final_activation(x)
        return x 
    

class ShallowConvNet(nn.Module): # inspired by Shallow ConvNet model from Schirrmeister et al 2017.
    def __init__(self, n_channels, input_time_length, n_classes=20, n_filters=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15): #, drop_prob=0.5):
        super(ShallowConvNet, self).__init__()
        
        self.temporal_conv = nn.Conv2d(n_channels, n_filters, (1, filter_time_length), padding='valid') # same')
        self.spatial_conv = nn.Conv2d(n_filters, n_filters, (n_channels, n_filters), padding='same') #, bias=False) 
        # self.square_nonlin = 
        self.pooling = nn.AvgPool2d((1, pool_time_length), stride=(pool_time_stride,1)) 
        # self.log_nonlin = nn.LogSigmoid() # "A log non-linearity >> is this wat is meant by that? or a true log fctn?"
        self.flatten = nn.Flatten() 
        last_layer_time_length = int(((input_time_length-filter_time_length+1)-75+pool_time_stride+1)/pool_time_stride+1) # IS THAT CORRECT? 
        self.linear_classification = nn.Linear(last_layer_time_length*40, n_classes)
        self.final_activation = nn.Softmax() # in Braindecode Paper a Softmax

        self.batch_norm = nn.BatchNorm2d(40)

    def forward(self, x):
        x = x.unsqueeze(1).permute(0,2,1,3)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)  
        x = self.batch_norm(x) # somewhere else, too? 
        x = torch.square(x) # ? 

        x = self.pooling(x)
        x = self.flatten(x) 
        x = torch.log(x)    # ?

        x = self.linear_classification(x)
        x = self.final_activation(x)
        return x 


class ShallowConvNet_Regression(nn.Module): # inspired by Shallow ConvNet model from Schirrmeister et al 2017.
    def __init__(self, n_channels, input_time_length, n_classes=1, n_filters=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15): #, drop_prob=0.5):
        super(ShallowConvNet_Regression, self).__init__()
        
        self.temporal_conv = nn.Conv2d(n_channels, n_filters, (1, filter_time_length), padding='valid') # same')
        self.spatial_conv = nn.Conv2d(n_filters, n_filters, (n_channels, n_filters), padding='same') #, bias=False) 
        # self.square_nonlin = 
        self.pooling = nn.AvgPool2d((1, pool_time_length), stride=(pool_time_stride,1)) 
        # self.log_nonlin = nn.LogSigmoid() # "A log non-linearity >> is this wat is meant by that? or a true log fctn?"
        self.flatten = nn.Flatten() 
        last_layer_time_length = int(((input_time_length-filter_time_length+1)-75+pool_time_stride+1)/pool_time_stride+1) # IS THAT CORRECT? 
        self.linear_classification = nn.Linear(last_layer_time_length*40, n_classes)
        self.final_activation = nn.Sigmoid() # in Braindecode Paper a Softmax

        self.batch_norm = nn.BatchNorm2d(40)

    def forward(self, x):
        x = x.unsqueeze(1).permute(0,2,1,3)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)  
        x = self.batch_norm(x) # somewhere else, too? 
        x = torch.square(x) # ? 

        x = self.pooling(x)
        x = self.flatten(x) 
        x = torch.log(x)    # ?

        x = self.linear_classification(x)
        # x = self.final_activation(x)
        return x 



def first_simple_classifier(**kwargs): 
    model = SimpleClassifierNN(
        input_size=61*100, hidden_size1=4096, hidden_size2=128
    )
    return model 


def shallow_net_20_classes(**kwargs): 
    model = ShallowConvNet(n_channels=61, n_classes=10, 
        input_time_length=100, n_filters=40, filter_time_length=25, 
        pool_time_length=75, pool_time_stride=15)
    
    return model

def first_shallow_conv_net_regression(**kwargs): 
    model = ShallowConvNet_Regression(n_channels=61, n_classes=1, 
        input_time_length=100, n_filters=40, filter_time_length=25, 
        pool_time_length=75, pool_time_stride=15)
    
    return model

def first_simple_regressor(**kwargs): 
    model = SimpleRegressorNN(
        input_size=61*100, hidden_size1=4096, hidden_size2=128
    )
    return model 


def deep_conv_net(**kwargs): 
    model = DeepConvNet(n_channels=61, input_time_length=100, n_classes=1)
    
    return model

# set recommended archs
simple_classifier = first_simple_classifier

shallow_conv_net = shallow_net_20_classes

simple_regressor = first_simple_regressor

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