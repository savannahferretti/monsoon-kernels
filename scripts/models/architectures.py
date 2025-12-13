#!/usr/bin/env python

import torch
from math import prod
from scripts.models.kernels import NonparametricKernelLayer,ParametricKernelLayer

class MainNN(torch.nn.Module):

    def __init__(self,nfeatures):
        '''
        Purpose: Initialize a feed-forward neural network that nonlinearly maps a feature vector to a scalar prediction.
        Args:
        - nfeatures (int): number of input features per sample (after any flattening, kernel integration, 
          and/or concatenation with local inputs)
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
        - X (torch.Tensor): input features with shape (nbatch, nfeatures)
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        return self.layers(X).squeeze()

class BaselineNN(torch.nn.Module):

    def __init__(self,patchshape,nfieldvars,nlocalvars,uselocal):
        '''
        Purpose: Initialize a neural network that directly ingests space-height-time patches for one or more predictor 
        fields, then passes the patches plus optional local inputs to MainNN.
        Args:
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nfieldvars (int): number of predictor fields
        - nlocalvars (int): number of local inputs
        - uselocal (bool): whether to use local inputs
        '''
        super().__init__()
        self.patchshape = patchshape
        self.nfieldvars = int(nfieldvars)
        self.nlocalvars = int(nlocalvars)
        self.uselocal   = bool(uselocal)
        nfeatures = self.nfieldvars*prod(self.patchshape)
        if self.uselocal:
            nfeatures += self.nlocalvars
        self.model = MainNN(nfeatures)

    def forward(self,fieldpatch,localvalues=None):
        '''
        Purpose: Forward pass through BaselineNN.
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - localvalues (torch.Tensor | None): local input values with shape (nbatch, nlocalvars) if uselocal is True, otherwise None
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        fieldpatch = fieldpatch.flatten(1)
        if self.uselocal:
            if localvalues is None:
                raise ValueError('`localvalues` must be provided when `uselocal` is True')
            X = torch.cat([fieldpatch,localvalues],dim=1)
        else:
            X = fieldpatch
        return self.model(X)

class KernelNN(torch.nn.Module):

    def __init__(self,intkernel,nlocalvars,uselocal,patchshape):
        # UPDATED: Added patchshape parameter to calculate preserved dimensions
        '''
        Purpose: Initialize a neural network that applies either non-parametric or parametric kernels over selected 
        dimensions of each predictor patch, then passes the resulting kernel features plus optional local inputs to MainNN.
        
        Notes:
        - Feature count depends on which dimensions are preserved (not integrated over by the kernel)
        - Example: vertical-only kernel with patch (3,3,16,7) preserves (3,3,7) → nfeatures = nfields*nkernels*3*3*7
        
        Args:
        - intkernel (torch.nn.Module): instance of NonparametricKernelLayer or ParametricKernelLayer
        - nlocalvars (int): number of local inputs
        - uselocal (bool): whether to use local inputs
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes) used to calculate preserved dimensions
        '''
        super().__init__()
        self.intkernel   = intkernel
        self.nfieldvars  = int(intkernel.nfieldvars)
        self.nlocalvars  = int(nlocalvars)
        self.uselocal    = bool(uselocal)
        self.nkernels    = int(intkernel.nkernels)
        self.kerneldims  = tuple(intkernel.kerneldims)
        
        # UPDATED: Calculate number of features based on preserved (non-integrated) dimensions
        # Dimensions where kernel doesn't vary are preserved in the output
        plats, plons, plevs, ptimes = patchshape
        preserved_size = 1
        if 'lat' not in self.kerneldims:
            preserved_size *= plats
        if 'lon' not in self.kerneldims:
            preserved_size *= plons
        if 'lev' not in self.kerneldims:
            preserved_size *= plevs
        if 'time' not in self.kerneldims:
            preserved_size *= ptimes
        
        nfeatures = self.nfieldvars * self.nkernels * preserved_size
        if self.uselocal:
            nfeatures += self.nlocalvars
        self.model = MainNN(nfeatures)

    def forward(self,fieldpatch,dareapatch,dlevpatch,dtimepatch,localvalues=None):
        '''
        Purpose: Forward pass through KernelNN.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights patch with shape (nbatch, ptimes)
        - localvalues (torch.Tensor | None): local input values with shape (nbatch, nlocalvars) if uselocal is True, otherwise None
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        features = self.intkernel(fieldpatch,dareapatch,dlevpatch,dtimepatch)
        if self.uselocal:
            if localvalues is None:
                raise ValueError('`localvalues` must be provided when `uselocal` is True')
            X = torch.cat([features,localvalues],dim=1)
        else:
            X = features
        return self.model(X)

class ModelFactory:

    @staticmethod
    def build(name,modelconfig,patchshape,nfieldvars,nlocalvars):
        '''
        Purpose: Build a model instance from configuration.
        Args:
        - name (str): model name
        - modelconfig (dict[str,object]): model configuration
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nfieldvars (int): number of predictor fields
        - nlocalvars (int): number of local inputs
        Returns:
        - torch.nn.Module: initialized model
        '''
        kind     = modelconfig['kind']
        uselocal = modelconfig['uselocal']
        if kind=='baseline':
            model = BaselineNN(patchshape,nfieldvars,nlocalvars,uselocal)
        elif kind=='nonparametric':
            nkernels   = modelconfig['nkernels']
            kerneldims = modelconfig['kerneldims']
            intkernel  = NonparametricKernelLayer(nfieldvars,nkernels,kerneldims)
            model = KernelNN(intkernel,nlocalvars,uselocal,patchshape)
        elif kind=='parametric':
            nkernels   = modelconfig['nkernels']
            kerneldict = modelconfig['kerneldict']
            intkernel = ParametricKernelLayer(nfieldvars,nkernels,kerneldict)
            model = KernelNN(intkernel,nlocalvars,uselocal,patchshape)
        else:
            raise ValueError(f'Unknown model kind `{kind}`')
        model.nparams = sum(param.numel() for param in model.parameters())
        return model