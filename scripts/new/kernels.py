#!/usr/bin/env python

import torch
import logging
import numpy as np

class NonparametricKernelLayer(torch.nn.Module):

    def __init__(self,nfieldvars,patchshape,nkernels,kerneldims):
        '''
        Purpose: Initialize free-form (non-parametric) kernels along selected dimensions.
        Args:
        - nfieldvars (int): number of predictor fields
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldims (list[str] | tuple[str]): dimensions the kernel varies along
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.patchshape = tuple(int(x) for x in patchshape)
        self.nkernels   = int(nkernels)
        self.kerneldims = tuple(kerneldims)
        kernelshape = [size if dim in self.kerneldims else 1 for dim,size in zip(('lat','lon','lev','time'),self.patchshape)]
        self.kernel = torch.nn.Parameter(torch.ones(self.nfieldvars,self.nkernels,*kernelshape))

    @torch.no_grad()
    def weights(self,quadweights,device,asarray=False):
        '''
        Purpose: Obtain normalized non-parametric kernel weights over a patch.
        Args:
        - quadweights (torch.Tensor): quadrature weights of shape (plats, plons, plevs, ptimes)
        - device (str | torch.device): device to place PyTorch tensors on
        - asarray (bool): if True, return a NumPy array
        Returns:
        - torch.Tensor | np.ndarray: normalized kernel weights of shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        kernelvalues = self.kernel.to(device=device)
        quadweights  = quadweights.to(device=device)
        integral = torch.einsum('fkyxpt,yxpt->fk',kernelvalues,quadweights)+1e-4
        kernelweights = kernelvalues/integral[:,:,None,None,None,None]
        if asarray:
            return kernelweights.detach().cpu().numpy()
        return kernelweights
        
    def forward(self,patch,quadweights):
        '''
        Purpose: Apply learned non-parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadweights (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        nbatch,device  = patch.shape[0],patch.device
        kernelweights  = self.weights(quadweights,device,asarray=False)
        kernelfeatures = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk',patch,kernelweights,quadweights)
        return kernelfeatures.reshape(nbatch,self.nfieldvars*self.nkernels)

class ParametricKernelLayer(torch.nn.Module):
         
    class GaussianFunction(torch.nn.Module):

        def __init__(self,nfieldvars,nkernels):
            '''
            Purpose: Initialize Gaussian kernel parameters along one dimension.
            Args:
            - nfieldvars (int): number of predictor fields
            - nkernels (int): number of kernels to learn per predictor field
            '''
            super().__init__()
            self.nfieldvars = int(nfieldvars)
            self.nkernels   = int(nkernels)
            self.mean       = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))
            self.logstd     = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))

        def forward(self,length,device):
            '''
            Purpose: Evaluate Gaussian kernel along a symmetric coordinate in [-1,1].
            Args:
            - length (int): number of points along the axis
            - device (str): device to place PyTorch tensors on ('cpu' | 'cuda')
            Returns:
            - torch.Tensor: Gaussian kernel values with shape (nfieldvars, nkernels, length)
            '''
            coord = torch.linspace(-1.0,1.0,steps=length,device=device)
            std   = torch.exp(self.logstd)
            return torch.exp(-0.5*((coord[None,None,:]-self.mean[...,None])/std[...,None])**2)

    class ExponentialFunction(torch.nn.Module):

        def __init__(self,nfieldvars,nkernels):
            '''
            Purpose: Initialize exponential-decay kernel parameters along one dimension.
            Args:
            - nfieldvars (int): number of predictor fields
            - nkernels (int): number of kernels to learn per predictor field
            '''
            super().__init__()
            self.nfieldvars = int(nfieldvars)
            self.nkernels   = int(nkernels)
            self.logtau     = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))

        def forward(self,length,device):
            '''
            Purpose: Evaluate causal exponential kernel along non-negative coordinate [0,1,2,...].
            Args:
            - length (int): number of points along the axis
            - device (str): device to place PyTorch tensors on ('cpu' | 'cuda')
            Returns:
            - torch.Tensor: exponential kernel values with shape (nfieldvars, nkernels, length)
            '''
            coord = torch.arange(length,device=device)
            tau   = torch.exp(self.logtau)+1e-4
            return torch.exp(-coord[None,None,:]/tau[...,None])

    def __init__(self,nfieldvars,nkernels,kerneldict):
        '''
        Purpose: Initialize smooth parametric kernels along selected dimensions.
        Args:
        - nfieldvars (int): number of predictor fields
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldict (dict[str,str]): mapping of dimensions the kernel varies along to a function ('gaussian' | 'exponential')
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.nkernels   = int(nkernels)
        self.kerneldims = tuple(kerneldict.keys())
        self.kerneldict = dict(kerneldict)
        self.functions  = torch.nn.ModuleDict()
        for dim,function in self.kerneldict.items():
            if function=='gaussian':
                self.functions[dim] = self.GaussianFunction(self.nfieldvars,self.nkernels)
            elif function=='exponential':
                self.functions[dim] = self.ExponentialFunction(self.nfieldvars,self.nkernels)
            else:
                raise ValueError(f'Unknown function type `{function}`, must be either `gaussian` or `exponential`')

    @torch.no_grad()
    def weights(self,quadweights,device,asarray=False):
        '''
        Purpose: Obtain normalized parametric kernel weights over a patch.
        Args:
        - quadweights (torch.Tensor): quadrature weights of shape (plats, plons, plevs, ptimes)
        - device (str | torch.device): device to place PyTorch tensors on
        - asarray (bool): if True, return a NumPy array
        Returns:
        - torch.Tensor | np.ndarray: normalized kernel weights of shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        (plats,plons,plevs,ptimes),dtype = quadweights.shape,quadweights.dtype
        quadweights  = quadweights.to(device=device)
        kernelvalues = torch.ones(self.nfieldvars,self.nkernels,plats,plons,plevs,ptimes,device=device,dtype=dtype)
        dimnames = ('lat','lon','lev','time')
        for axis,dim in enumerate(dimnames,start=2):
            if dim not in self.kerneldims:
                continue
            dimlength = kernelvalues.shape[axis]
            function  = self.functions[dim]
            kernel1D  = func(dimlength,device)
            kernelshape       = [self.nfieldvars,self.nkernels,1,1,1,1]
            kernelshape[axis] = dimlength
            kernelvalues      = kernelvalues*kernel1D.view(*kernelshape)
        integral = torch.einsum('fkyxpt,yxpt->fk',kernelvalues,quadweights)+1e-4
        kernelweights = kernelvalues/integral[:,:,None,None,None,None]
        if asarray:
            return kernelweights.detach().cpu().numpy()
        return kernelweights
        
    def forward(self,patch,quadweights):
        '''
        Purpose: Apply learned parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadweights (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        nbatch,device,dtype = patch.shape[0],patch.device,patch.dtype
        kernelweights  = self.weights(quadweights,device,asarray=False)
        kernelfeatures = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk',patch,kernelweights,quadweights)
        return kernelfeatures.reshape(nbatch,self.nfieldvars*self.nkernels)