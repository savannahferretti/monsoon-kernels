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

        plats, plons, plevs, ptimes = patchshape

        # ---------------------------------------------------------------
        # EDIT: preserved_size calculation updated to match KernelModule.integrate()
        # We now slice to the center for dims NOT in kerneldims (local), so they
        # contribute factor 1 instead of full patch length.
        preserved_size = 1
        # If you ever intentionally want to preserve a dimension, you'd do so by
        # *not* slicing it in KernelModule.integrate. With the current definition,
        # dims not in kerneldims are local -> preserved factor remains 1.
        # ---------------------------------------------------------------

        nfeatures = self.nfieldvars * self.nkernels * preserved_size  # EDIT: now just F*K for vertical kernel
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
        
        # ---------------------------------------------------------------
        # EDIT (optional but helpful): one-time shape print to verify (B, F*K)
        if not hasattr(self, "_printed_kernel_shape"):
            self._printed_kernel_shape = True
            print("[KernelNN] kerneldims =", self.kerneldims, "| features shape =", tuple(features.shape))
        # ---------------------------------------------------------------
        
        if not hasattr(self, "_printed_kernel_stats"):
            self._printed_kernel_stats = True
            print("[KernelNN] features mean/std:",
                  features.mean().item(), features.std().item())

        if self.uselocal:
            if localvalues is None:
                raise ValueError('`localvalues` must be provided when `uselocal` is True')
            X = torch.cat([features,localvalues],dim=1)
        else:
            X = features
        return self.model(X)

