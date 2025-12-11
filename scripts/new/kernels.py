#!/usr/bin/env python

import torch
import logging
import numpy as np

class NonparametricKernelLayer(torch.nn.Module):

    def __init__(self,nfieldvars,nkernels,kerneldims):
        '''
        Purpose: Initialize free-form (non-parametric) kernels along selected dimensions.
        Args:
        - nfieldvars (int): number of predictor fields
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldims (list[str] | tuple[str]): dimensions the kernel varies along
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.nkernels   = int(nkernels)
        self.kerneldims = tuple(kerneldims)
        self.kernel     = None

    @torch.no_grad()
    def weights(self,quad,device,asarray=False):
        '''
        Purpose: Obtain normalized non-parametric kernel weights over a patch.
        Args:
        - quad (torch.Tensor): quadrature weights of shape (plats, plons, plevs, ptimes)
        - device (str): device to use
        - asarray (bool): if True, return a NumPy array
        Returns:
        - torch.Tensor | np.ndarray: normalized kernel weights of shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        if self.kernel is None:
            plats,plons,plevs,ptimes = quad.shape
            kernelshape = [size if dim in self.kerneldims else 1 for dim,size in zip(('lat','lon','lev','time'),(plats,plons,plevs,ptimes))]
            self.kernel = torch.nn.Parameter(torch.ones(self.nfieldvars,self.nkernels,*kernelshape,device=device,dtype=quad.dtype))
        kernel = self.kernel
        quad   = quad.to(device)
        integral = torch.einsum('fkyxpt,yxpt->fk',kernel,quad)+1e-4
        weights  = kernel/integral[:,:,None,None,None,None]
        if asarray:
            return weights.detach().cpu().numpy()
        return weights
        
    def forward(self,patch,quad):
        '''
        Purpose: Apply learned non-parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quad (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        weights  = self.weights(quad,patch.device,asarray=False)
        features = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk',patch,weights,quad)
        return features.reshape(patch.shape[0],self.nfieldvars*self.nkernels)

class ParametricKernelLayer(torch.nn.Module):
         
    class GaussianKernel(torch.nn.Module):

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
            - device (str): device to use
            Returns:
            - torch.Tensor: Gaussian kernel values with shape (nfieldvars, nkernels, length)
            '''
            coord = torch.linspace(-1.0,1.0,steps=length,device=device)
            std   = torch.exp(self.logstd)
            return torch.exp(-0.5*((coord[None,None,:]-self.mean[...,None])/std[...,None])**2)

    class ExponentialKernel(torch.nn.Module):

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
            - device (str): device to use
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
                self.functions[dim] = self.GaussianKernel(self.nfieldvars,self.nkernels)
            elif function=='exponential':
                self.functions[dim] = self.ExponentialKernel(self.nfieldvars,self.nkernels)
            else:
                raise ValueError(f'Unknown function type `{function}`, must be either `gaussian` or `exponential`')

    @torch.no_grad()
    def weights(self,quad,device,asarray=False):
        '''
        Purpose: Obtain normalized parametric kernel weights over a patch.
        Args:
        - quad (torch.Tensor): quadrature weights of shape (plats, plons, plevs, ptimes)
        - device (str): device to use
        - asarray (bool): if True, return a NumPy array
        Returns:
        - torch.Tensor | np.ndarray: normalized kernel weights of shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        (plats,plons,plevs,ptimes) = quad.shape
        quad   = quad.to(device)
        kernel = torch.ones(self.nfieldvars,self.nkernels,plats,plons,plevs,ptimes,device=device,dtype=quad.dtype)
        for ax,dim in enumerate(('lat','lon','lev','time'),start=2):
            if dim not in self.kerneldims:
                continue
            dimlength = kernel.shape[ax]
            function  = self.functions[dim]
            kernel1D  = function(dimlength,device)
            kernelshape     = [self.nfieldvars,self.nkernels,1,1,1,1]
            kernelshape[ax] = dimlength
            kernel          = kernel*kernel1D.view(*kernelshape)
        integral = torch.einsum('fkyxpt,yxpt->fk',kernel,quad)+1e-4
        weights  = kernel/integral[:,:,None,None,None,None]
        if asarray:
            return weights.detach().cpu().numpy()
        return weights
        
    def forward(self,patch,quad):
        '''
        Purpose: Apply learned parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quad (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        weights  = self.weights(quad,patch.device,asarray=False)
        features = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk',patch,weights,quad)
        return features.reshape(patch.shape[0],self.nfieldvars*self.nkernels)