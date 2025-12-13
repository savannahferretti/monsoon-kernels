#!/usr/bin/env python

import torch
import numpy as np

class KernelModule:

    @staticmethod
    def normalize(kernel,dareapatch,dlevpatch,dtimepatch,kerneldims,epsilon=1e-4):
        '''
        Purpose: Normalize a kernel so it integrates to ~1 over its active dimensions.
        
        Notes:
        - Normalization should include quadrature factors only along dimensions the kernel varies along.
          For example:
          - If kerneldims includes ('lat','lon','lev','time'): use ΔA*Δp*Δt
          - If kerneldims includes ('lev',): use Δp only (ΔA and Δt are not included in the normalization)
          - If kerneldims includes ('time',): use Δt only
          - If kerneldims includes ('lat','lon'): use ΔA only
        - This avoids renormalizing with weights over dimensions that are collapsed/constant in the kernel.
        
        Args:
        - kernel (torch.Tensor): unnormalized kernel with shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (plevs,)
        - dtimepatch (torch.Tensor): time step weights patch with shape (ptimes,)
        - kerneldims (tuple[str] | list[str]): dimensions the kernel varies along
        - epsilon (float): stabilizer to avoid divide-by-zero (defaults to 1e-4)
        Returns:
        - torch.Tensor: normalized kernel weights with same shape as kernel
        '''
        kerneldims = tuple(kerneldims)
        plats,plons = dareapatch.shape
        plevs       = dlevpatch.numel()
        ptimes      = dtimepatch.numel()

        norm = torch.ones((plats,plons,plevs,ptimes),dtype=dareapatch.dtype,device=dareapatch.device)

        if ('lat' in kerneldims) or ('lon' in kerneldims):
            norm = norm*dareapatch[:,:,None,None]
        if 'lev' in kerneldims:
            norm = norm*dlevpatch[None,None,:,None]
        if 'time' in kerneldims:
            norm = norm*dtimepatch[None,None,None,:]

        integral = torch.einsum('fkyxpt,yxpt->fk',kernel,norm)+epsilon
        weights  = kernel/integral[:,:,None,None,None,None]
        return weights

    @staticmethod
    def integrate(fieldpatch,weights,quadpatch):
        '''
        Purpose: Integrate the predictor fields using the normalized kernel weights.

        Notes:
        - The weights are already normalized to integrate to 1 with quadrature factors included in the
          normalization step, so the feature integral is a simple weighted sum without applying quadpatch again.
        - The quadpatch parameter is kept for API compatibility but is not used in the computation.

        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - weights (torch.Tensor): normalized kernel weights with shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        - quadpatch (torch.Tensor): product measure patch with shape (nbatch, plats, plons, plevs, ptimes) [NOT USED]
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        features = torch.einsum('bfyxpt,fkyxpt->bfk',fieldpatch,weights)
        return features.flatten(1)

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
        self.weights    = None
        self.features   = None

    def get_weights(self,dareapatch,dlevpatch,dtimepatch,device):
        '''
        Purpose: Obtain normalized non-parametric kernel weights over a patch.
        Args:
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (plevs,)
        - dtimepatch (torch.Tensor): time step weights patch with shape (ptimes,)
        - device (str | torch.device): device to use
        Returns:
        - torch.Tensor: normalized kernel weights of shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        dareapatch = dareapatch.to(device)
        dlevpatch  = dlevpatch.to(device)
        dtimepatch = dtimepatch.to(device)

        plats,plons = dareapatch.shape
        plevs       = dlevpatch.numel()
        ptimes      = dtimepatch.numel()

        if self.kernel is None:
            kernelshape = [
                plats  if 'lat' in self.kerneldims else 1,
                plons  if 'lon' in self.kerneldims else 1,
                plevs  if 'lev' in self.kerneldims else 1,
                ptimes if 'time' in self.kerneldims else 1]
            kernel = torch.ones(self.nfieldvars,self.nkernels,*kernelshape,dtype=dareapatch.dtype,device=device)
            kernel = kernel+torch.randn_like(kernel)*0.01
            self.kernel = torch.nn.Parameter(kernel)

        self.weights = KernelModule.normalize(self.kernel,dareapatch,dlevpatch,dtimepatch,self.kerneldims)
        return self.weights
        
    def forward(self,fieldpatch,quadpatch,dareapatch,dlevpatch,dtimepatch):
        '''
        Purpose: Apply learned non-parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadpatch (torch.Tensor): product measure patch with shape (nbatch, plats, plons, plevs, ptimes)
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights patch with shape (nbatch, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        dareamean = dareapatch.mean(dim=0)
        dlevmean  = dlevpatch.mean(dim=0)
        dtimemean = dtimepatch.mean(dim=0)
        weights   = self.get_weights(dareamean,dlevmean,dtimemean,fieldpatch.device)
        self.features = KernelModule.integrate(fieldpatch,weights,quadpatch)
        return self.features
        
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
            self.mean   = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))
            self.logstd = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))

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
            self.logtau     = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))

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
        self.weights    = None 
        self.features   = None
        self.functions  = torch.nn.ModuleDict()
        for dim,function in self.kerneldict.items():
            if function=='gaussian':
                self.functions[dim] = self.GaussianKernel(self.nfieldvars,self.nkernels)
            elif function=='exponential':
                self.functions[dim] = self.ExponentialKernel(self.nfieldvars,self.nkernels)
            else:
                raise ValueError(f'Unknown function type `{function}`; must be either `gaussian` or `exponential`')

    def get_weights(self,dareapatch,dlevpatch,dtimepatch,device):
        '''
        Purpose: Obtain normalized parametric kernel weights over a patch.
        Args:
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (plevs,)
        - dtimepatch (torch.Tensor): time step weights patch with shape (ptimes,)
        - device (str | torch.device): device to use
        Returns:
        - torch.Tensor: normalized kernel weights of shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        dareapatch = dareapatch.to(device)
        dlevpatch  = dlevpatch.to(device)
        dtimepatch = dtimepatch.to(device)

        plats,plons = dareapatch.shape
        plevs       = dlevpatch.numel()
        ptimes      = dtimepatch.numel()

        kernel = torch.ones(self.nfieldvars,self.nkernels,plats,plons,plevs,ptimes,dtype=dareapatch.dtype,device=device)

        for ax,dim in enumerate(('lat','lon','lev','time'),start=2):
            if dim in self.kerneldims:
                kernel1D = self.functions[dim](kernel.shape[ax],device)
                view     = [1,1]+[kernel.shape[ax] if i==ax-2 else 1 for i in range(4)]
                kernel   = kernel*kernel1D.view(*view)

        self.weights = KernelModule.normalize(kernel,dareapatch,dlevpatch,dtimepatch,self.kerneldims)
        return self.weights
        
    def forward(self,fieldpatch,quadpatch,dareapatch,dlevpatch,dtimepatch):
        '''
        Purpose: Apply learned parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadpatch (torch.Tensor): product measure patch with shape (nbatch, plats, plons, plevs, ptimes)
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights patch with shape (nbatch, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        dareamean = dareapatch.mean(dim=0)
        dlevmean  = dlevpatch.mean(dim=0)
        dtimemean = dtimepatch.mean(dim=0)
        weights   = self.get_weights(dareamean,dlevmean,dtimemean,fieldpatch.device)
        self.features = KernelModule.integrate(fieldpatch,weights)
        return self.features