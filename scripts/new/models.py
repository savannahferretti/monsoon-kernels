#!/usr/bin/env python

import torch

class MainNN(torch.nn.Module):

    def __init__(self,nfeatures):
        '''
        Purpose: Initialize a feed-forward neural network that nonlinearly maps a feature vector to a scalar prediction.
        Args:
        - nfeatures (int): number of input features per sample (after any flattening, kernel integration, and/or concatenation)
        '''
        super().__init__()
        nfeatures = int(nfeatures)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(nfeatures,256), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256,128),       torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128,64),        torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64,32),         torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(32,1))

    def forward(self, X):
        '''
        Purpose: Forward pass through MainNN.
        Args:
        - X (torch.Tensor): input features with shape (nbatch, nfeatures)
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        return self.layers(X).squeeze()

class BaselineNN(torch.nn.Module):

    def __init__(self,nfieldvars,patchshape,nlocalvars,uselocal):
        '''
        Purpose: Initialize a neural network that directly ingests space-height-time patches for one or more predictor 
        fields, then passes the patches plus optional local inputs to MainNN.
        Args:
        - nfieldvars (int): number of predictor fields
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nlocalvars (int): number of local inputs
        - uselocal (bool): whether to use local inputs
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.patchshape = tuple(int(x) for x in patchshape)
        self.nlocalvars = int(nlocalvars)
        self.uselocal   = uselocal
        plats,plons,plevs,ptimes = self.patchshape
        nfeatures = self.nfieldvars*(plats*plons*plevs*ptimes)
        if self.uselocal:
            nfeatures += self.nlocalvars
        self.model = MainNN(nfeatures)

    def forward(self,patch,local=None):
        '''
        Purpose: Forward pass through BaselineNN.
        Args:
        - patch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - local (torch.Tensor | None): local inputs with shape (nbatch, nlocalvars) if uselocal is True, otherwise None
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        nbatch = patch.shape[0]
        patch  = patch.reshape(nbatch,-1)
        if self.uselocal:
            if local is None:
                raise ValueError('`local` must be provided when `uselocal` is True')
            X = torch.cat([patch,local],dim=1)
        else:
            X = patch
        return self.model(X)

class KernelNN(torch.nn.Module):

    def __init__(self,kernellayer,nlocalvars,uselocal):
        '''
        Purpose: Initialize a neural network that applies either non-parametric or parametric kernels over selected 
        dimensions of each predictor patch, then passes the resulting kernel features plus optional local inputs to MainNN.
        Args:
        - kernellayer (torch.nn.Module): instance of NonparametricKernelLayer or ParametricKernelLayer
        - nlocalvars (int): number of local inputs
        - uselocal (bool): whether to use local inputs
        '''
        super().__init__()
        self.kernellayer = kernellayer
        self.nfieldvars  = int(kernellayer.nfieldvars)
        self.nkernels    = int(kernellayer.nkernels)
        self.kerneldims  = tuple(kernellayer.kerneldims)
        self.nlocalvars  = int(nlocalvars)
        self.uselocal    = uselocal
        nfeatures = self.nfieldvars*self.nkernels
        if self.uselocal:
            nfeatures += self.nlocalvars
        self.model = MainNN(nfeatures)

    def forward(self,patch,quad,local=None,returnfeatures=False):
        '''
        Purpose: Forward pass through KernelNN.
        Args:
        - patch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quad (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        - local (torch.Tensor | None): local inputs with shape (nbatch, nlocalvars) if uselocal is True, otherwise None
        - returnfeatures (bool): if True, return kernel-integrated features
        Returns:
        - torch.Tensor: predictions with shape (nbatch,) or tuple (predictions, features) if 'returnfeatures' is True
        '''
        kernelfeatures = self.kernellayer(patch,quad)
        if self.uselocal:
            if local is None:
                raise ValueError('`local` must be provided when `uselocal` is True')
            X = torch.cat([kernelfeatures,local],dim=1)
        else:
            X = kernelfeatures
        if returnfeatures:
            return self.model(X),kernelfeatures
        return self.model(X)

class ModelFactory:
    
    def build(name,modelconfig,nfieldvars,patchshape,nlocalvars):
        '''
        Purpose: Build a model instance from configuration.
        Args:
        - name (str): model name
        - modelconfig (dict): model configuration
        - nfieldvars (int): number of predictor fields
        - patchshape (?): (plats, plons, plevs, ptimes)
        - nlocalvars (int): number of local inputs
        Returns:
        - torch.nn.Module: initialized model
        '''
        modeltype = modelconfig['type']
        uselocal  = modelconfig['uselocal']
        if modeltype=='baseline':
            model = BaselineNN(nfieldvars,patchshape,nlocalvars,uselocal)
        elif modeltype=='nonparametric':
            nkernels    = modelconfig['nkernels']
            kerneldims  = modelconfig['kerneldims']
            kernellayer = NonparametricKernelLayer(nfieldvars,nkernels,kerneldims)
            model = KernelNN(kernellayer,nlocalvars,uselocal)
        elif modeltype=='parametric':
            nkernels    = modelconfig['nkernels']
            kerneldict  = modelconfig['kerneldict']
            kernellayer = ParametricKernelLayer(nfieldvars,nkernels,kernelconfig)
            model = KernelNN(kernellayer,nlocalvars,uselocal)
        model.nparams = sum(param.numel() for param in model.parameters())
        return model