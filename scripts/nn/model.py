#!/usr/bin/env python

import torch

class NNModel(torch.nn.Module):

    def __init__(self,inputsize):
        '''
        Purpose: Define a feedforward neural network (NN) for precipitation prediction.
        Args:
        - inputsize (int): number of input features per sample (for 3D variables it's 1 per variable; for 4D variables it's
          'nlevels' per variable; for experiments with more than one variable it's the sum across variables)
        '''
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize,256), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256,128),       torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128,64),        torch.nn.GELU(), torch.nn.Dropout(0.1),            
            torch.nn.Linear(64,32),         torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(32,1))

    def forward(self,X):
        '''
        Purpose: Forward pass through the NN.
        Args:
        - X (torch.Tensor): input features tensor of shape (nsamples, inputsize)
        Returns:
        - torch.Tensor: raw prediction tensor of shape (nsamples, 1)
        '''
        return self.layers(X)