import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        # define all layers, here
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.Dropout(0.3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Sigmoid()
        )
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code, here
        x = self.layer1(x)
        x = self.layer2(x)
        
        return x.squeeze()