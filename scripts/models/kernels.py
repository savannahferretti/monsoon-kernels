#!/usr/bin/env python

import torch
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
    def weights(self,quadpatch,device,asarray=False):
        '''
        Purpose: Obtain normalized non-parametric kernel weights over a patch.
        Args:
        - quadpatch (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        - device (str | torch.device): device to use
        - asarray (bool): if True, return a NumPy array
        Returns:
        - torch.Tensor | np.ndarray: normalized kernel weights of shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        quadpatch = quadpatch.to(device)
        if self.kernel is None:
            kernelshape = [size if dim in self.kerneldims else 1 for dim,size in zip(('lat','lon','lev','time'),quadpatch.shape)]
            self.kernel = torch.nn.Parameter(torch.ones(self.nfieldvars,self.nkernels,*kernelshape,dtype=quadpatch.dtype,device=device))
        integral = torch.einsum('fkyxpt,yxpt->fk',self.kernel,quadpatch)+1e-4
        weights  = self.kernel/integral[:,:,None,None,None,None]
        if asarray:
            return weights.detach().cpu().numpy()
        return weights
        
    def forward(self,fieldpatch,quadpatch):
        '''
        Purpose: Apply learned non-parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadpatch (torch.Tensor): quadrature weights patch with shape (nbatch, plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        weights  = self.weights(quadpatch.mean(dim=0),fieldpatch.device,asarray=False) 
        features = torch.einsum('bfyxpt,fkyxpt,byxpt->bfk',fieldpatch,weights,quadpatch) 
        return features.flatten(1)

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
            Purpose: Evaluate a Gaussian kernel along a symmetric coordinate in [-1,1].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: Gaussian kernel values with shape (nfieldvars, nkernels, length)
            '''
            coord    = torch.linspace(-1.0,1.0,steps=length,device=device)
            std      = torch.exp(self.logstd)
            kernel1D = torch.exp(-0.5*((coord[None,None,:]-self.mean[...,None])/std[...,None])**2)
            return kernel1D

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
            Purpose: Evaluate an exponential kernel along non-negative coordinate [0,1,2,...].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: exponential kernel values with shape (nfieldvars, nkernels, length)
            '''
            coord    = torch.arange(length,device=device)
            tau      = torch.exp(self.logtau)+1e-4
            kernel1D = torch.exp(-coord[None,None,:]/tau[...,None])
            return kernel1D

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
        self.kerneldict = dict(kerneldict)
        self.kerneldims = tuple(kerneldict.keys())
        self.functions  = torch.nn.ModuleDict()
        for dim,function in self.kerneldict.items():
            if function=='gaussian':
                self.functions[dim] = self.GaussianKernel(self.nfieldvars,self.nkernels)
            elif function=='exponential':
                self.functions[dim] = self.ExponentialKernel(self.nfieldvars,self.nkernels)
            else:
                raise ValueError(f'Unknown function type `{function}`; must be either `gaussian` or `exponential`')

    @torch.no_grad()
    def weights(self,quadpatch,device,asarray=False):
        '''
        Purpose: Obtain normalized parametric kernel weights over a patch.
        Args:
        - quadpatch (torch.Tensor): quadrature weights of shape (plats, plons, plevs, ptimes)
        - device (str | torch.device): device to use
        - asarray (bool): if True, return a NumPy array
        Returns:
        - torch.Tensor | np.ndarray: normalized kernel weights of shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        quadpatch = quadpatch.to(device)
        kernel = torch.ones(self.nfieldvars,self.nkernels,*quadpatch.shape,dtype=quadpatch.dtype,device=device)
        for ax,dim in enumerate(('lat','lon','lev','time'),start=2):
            if dim not in self.kerneldims:
                continue
            kernel1D = self.functions[dim](kernel.shape[ax],device)
            view     = [1]*kernel.ndim
            view[:2] = (self.nfieldvars,self.nkernels)
            view[ax] = kernel.shape[ax]
            kernel   = kernel*kernel1D.view(*view)
        integral = torch.einsum('fkyxpt,yxpt->fk',kernel,quadpatch)+1e-4
        weights  = kernel/integral[:,:,None,None,None,None]
        if asarray:
            return weights.detach().cpu().numpy()
        return weights
        
    def forward(self,fieldpatch,quadpatch):
        '''
        Purpose: Apply learned parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadpatch (torch.Tensor): quadrature weights patch with shape (nbatch, plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        weights  = self.weights(quadpatch.mean(dim=0),fieldpatch.device,asarray=False) 
        features = torch.einsum('bfyxpt,fkyxpt,byxpt->bfk',fieldpatch,weights,quadpatch) 
        return features.flatten(1)