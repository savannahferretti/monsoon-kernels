#!/usr/bin/env python

import torch
from kernels import NonparametricKernelLayer,ParametricKernelLayer


class MainNN(torch.nn.Module):
    
    def __init__(self,nfeatures):
        '''
        Purpose: Initialize a feed-forward NN that maps a flat feature vector to a scalar precipitation prediction.
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
    
    def forward(self,X):
        '''
        Purpose: Forward pass through MainNN.
        Args:
        - X (torch.Tensor): input feature tensor of shape (nbatch, nfeatures)
        Returns:
        - torch.Tensor: raw precipitation prediction tensor of shape (nbatch, 1)
        '''
        return self.layers(X)


class BaselineNN(torch.nn.Module):
    
    def __init__(self,nfieldvars,patchshape,nlocalvars,uselocal):
        '''
        Purpose: Initialize the baseline NN that directly ingests space-height-time patches for one or more predictor 
        fields, optionally concatenated with local inputs, and maps them to a scalar precipitation prediction via MainNN.
        Args:
        - nfieldvars (int): number of predictor fields in each patch
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nlocalvars (int): number of local input variables
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
        - patch (torch.Tensor): (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - local (torch.Tensor | None): if uselocal is True, (nbatch, nlocalvars) local inputs; otherwise None
        Returns:
        - torch.Tensor: (nbatch, 1) raw precipitation prediction
        '''
        nbatch = patch.shape[0]
        patch  = patch.reshape(nbatch,-1)
        if self.uselocal:
            if local is None:
                raise ValueError('`local` must be provided when uselocal is True')
            X = torch.cat([patch,local],dim=1)
        else:
            X = patch
        return self.model(X)


class KernelNN(torch.nn.Module):

    def __init__(self,kernellayer,nlocalvars,uselocal):
        '''
        Purpose: Initialize a kernel NN that applies either non-parametric or parametric kernels over selected 
        dimensions of each predictor patch, then passes the resulting kernel features plus optional local inputs 
        to MainNN.
        Args:
        - kernellayer (torch.nn.Module): instance of NonparametricKernelLayer or ParametricKernelLayer with attributes 
          'nfieldvars', 'patchshape', 'nkernels', 'kerneldims'
        - nlocalvars (int): number of local input variables
        - uselocal (bool): whether to use local inputs
        '''
        super().__init__()
        self.kernellayer = kernellayer
        self.nfieldvars  = int(kernellayer.nfieldvars)
        self.patchshape  = tuple(int(x) for x in kernellayer.patchshape)
        self.nkernels    = int(kernellayer.nkernels)
        self.kerneldims  = tuple(kernellayer.kerneldims)
        self.nlocalvars  = int(nlocalvars)
        self.uselocal    = uselocal
        nfeatures = self.nfieldvars*self.nkernels
        if self.uselocal:
            nfeatures += self.nlocalvars
        self.model = MainNN(nfeatures)
    
    def forward(self,patch,local=None,quadweights=None):
        '''
        Purpose: Forward pass through KernelNN.
        Args:
        - patch (torch.Tensor): (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - local (torch.Tensor | None): if uselocal is True, (nbatch, nlocalvars) local inputs; otherwise None
        - quadweights (torch.Tensor | None): (plats, plons, plevs, ptimes); if None, unit weights are used internally
        Returns:
        - torch.Tensor: (nbatch, 1) raw precipitation prediction
        '''
        plats,plons,plevs,ptimes = self.patchshape
        device,dtype = patch.device,patch.dtype
        if quadweights is None:
            quadweights = torch.ones(plats,plons,plevs,ptimes,device=device,dtype=dtype)
        kernelfeatures = self.kernellayer(patch,quadweights=quadweights)
        if self.uselocal:
            if local is None:
                raise ValueError('`local` must be provided when uselocal is True')
            X = torch.cat([kernelfeatures,local],dim=1)
        else:
            X = kernelfeatures
        return self.model(X)


def build_model_from_config(modelcfg,patchshape,nfieldvars,nlocalvars):
    '''
    Purpose: Build a model instance from one configs["models"] entry.
    Args:
    - modelcfg (dict): one element from configs["models"]
    - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
    - nfieldvars (int): number of predictor fields
    - nlocalvars (int): number of local input variables
    Returns:
    - torch.nn.Module: BaselineNN or KernelNN instance
    '''
    mtype    = modelcfg['type']
    uselocal = modelcfg.get('uselocal',True)
    if mtype=='baseline':
        return BaselineNN(
            nfieldvars=nfieldvars,
            patchshape=patchshape,
            nlocalvars=nlocalvars,
            uselocal=uselocal)
    elif mtype=='nonparametric':
        nkernels   = int(modelcfg['nkernels'])
        kerneldims = modelcfg['kernel']['dims']
        kernellayer = NonparametricKernelLayer(
            nfieldvars=nfieldvars,
            patchshape=patchshape,
            nkernels=nkernels,
            kerneldims=kerneldims)
        return KernelNN(
            kernellayer=kernellayer,
            nlocalvars=nlocalvars,
            uselocal=uselocal)
    elif mtype=='parametric':
        nkernels   = int(modelcfg['nkernels'])
        families   = dict(modelcfg['kernel'])
        kerneldims = list(families.keys())
        kernellayer = ParametricKernelLayer(
            nfieldvars=nfieldvars,
            patchshape=patchshape,
            nkernels=nkernels,
            kerneldims=kerneldims,
            families=families)
        return KernelNN(
            kernellayer=kernellayer,
            nlocalvars=nlocalvars,
            uselocal=uselocal)
    else:
        raise ValueError(f'Unknown model type `{mtype}`')