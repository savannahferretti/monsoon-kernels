#!/usr/bin/env python

import torch
import numpy as np

class KernelModule:

    @staticmethod
    def normalize(kernel,dareapatch,dlevpatch,dtimepatch,kerneldims,epsilon=1e-4):
        '''
        Purpose: Normalize a kernel so it integrates to ~1 over its active dimensions.
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
        if 'lat' not in kerneldims:
            integral = integral/plats
        if 'lon' not in kerneldims:
            integral = integral/plons
        if 'lev' not in kerneldims:
            integral = integral/plevs
        if 'time' not in kerneldims:
            integral = integral/ptimes
        weights = kernel/integral[:,:,None,None,None,None]
        return weights

    @staticmethod
    def integrate(fieldpatch,weights,dareapatch,dlevpatch,dtimepatch,kerneldims):
        '''
        Purpose: Integrate predictor fields using normalized kernel weights over kerneled dimensions, preserving 
        non-kerneled dimensions.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - weights (torch.Tensor): normalized kernel weights with shape (nfieldvars, nkernels, platsor 1, plons or 1, plevs or 1, 
          ptimes or 1)
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights patch with shape (nbatch, ptimes)
        - kerneldims (tuple[str] | list[str]): dimensions the kernel varies along
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars, nkernels, ...) 
          where ... are preserved non-kerneled dimensions
        '''
        weighted = fieldpatch.unsqueeze(2)*weights.unsqueeze(0)
        quad = 1.0
        if ('lat' in kerneldims) or ('lon' in kerneldims):
            quad = quad*dareapatch[:,None,None,:,:,None,None]
        if 'lev' in kerneldims:
            quad = quad*dlevpatch[:,None,None,None,None,:,None]
        if 'time' in kerneldims:
            quad = quad*dtimepatch[:,None,None,None,None,None,:]
        weighted = weighted*quad
        dimstosum = []
        if 'lat' in kerneldims:
            dimstosum.append(3)
        if 'lon' in kerneldims:
            dimstosum.append(4)
        if 'lev' in kerneldims:
            dimstosum.append(5)
        if 'time' in kerneldims:
            dimstosum.append(6)
        return weighted.sum(dim=dimstosum) if dimstosum else weighted


class NonparametricKernelLayer(torch.nn.Module):

    def __init__(self,nfieldvars,nkernels,kerneldims,patchshape):
        '''
        Purpose: Initialize free-form (non-parametric) kernels along selected dimensions.
        Args:
        - nfieldvars (int): number of predictor fields
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldims (list[str] | tuple[str]): dimensions the kernel varies along
        - patchshape (tuple[int,int,int,int]): patch shape as (plats, plons, plevs, ptimes)
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.nkernels   = int(nkernels)
        self.kerneldims = tuple(kerneldims)
        self.weights    = None
        self.features   = None
        plats,plons,plevs,ptimes = patchshape
        kernelshape = [
            plats  if 'lat' in self.kerneldims else 1,
            plons  if 'lon' in self.kerneldims else 1,
            plevs  if 'lev' in self.kerneldims else 1,
            ptimes if 'time' in self.kerneldims else 1]
        kernel = torch.ones(self.nfieldvars,self.nkernels,*kernelshape)
        kernel = kernel+torch.randn_like(kernel)*0.2
        self.kernel = torch.nn.Parameter(kernel)

    def get_weights(self,dareapatch,dlevpatch,dtimepatch,device):
        '''
        Purpose: Obtain normalized non-parametric kernel weights over a patch.
        Args:
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (plats, plons) or (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (plevs,) or (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights patch with shape (ptimes,) or (nbatch, ptimes)
        - device (str | torch.device): device to use
        Returns:
        - torch.Tensor: normalized kernel weights of shape (nfieldvars, nkernels, plats_or_1, plons_or_1, plevs_or_1, ptimes_or_1)
        '''
        self.kernel = self.kernel.to(device)
        dareapatch  = dareapatch.to(device)
        dlevpatch   = dlevpatch.to(device)
        dtimepatch  = dtimepatch.to(device)
        if dareapatch.dim()==3:
            dareapatch0 = dareapatch[0]
        else:
            dareapatch0 = dareapatch
        if dlevpatch.dim()==2:
            dlevpatch0 = dlevpatch[0]
        else:
            dlevpatch0 = dlevpatch
        if dtimepatch.dim()==2:
            dtimepatch0 = dtimepatch[0]
        else:
            dtimepatch0 = dtimepatch
        self.weights = KernelModule.normalize(self.kernel,dareapatch0,dlevpatch0,dtimepatch0,self.kerneldims,epsilon=1e-6)
        return self.weights

    def forward(self,fieldpatch,dareapatch,dlevpatch,dtimepatch):
        '''
        Purpose: Apply learned non-parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights patch with shape (nbatch, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels*preserved_dims)
        '''
        nbatch = fieldpatch.shape[0]
        feats_list = []

        for i in range(nbatch):
            weights_i = self.get_weights(dareapatch[i],dlevpatch[i],dtimepatch[i],device=fieldpatch.device)
            feats_i = KernelModule.integrate(
                fieldpatch[i:i+1],weights_i,
                dareapatch[i:i+1],dlevpatch[i:i+1],dtimepatch[i:i+1],
                self.kerneldims)
            feats_list.append(feats_i)

        feats = torch.cat(feats_list,dim=0)
        self.features = feats
        return feats.flatten(1)

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
            self.logtau = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))

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

    def forward(self,fieldpatch,dareapatch,dlevpatch,dtimepatch):
        '''
        Purpose: Apply learned parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights patch with shape (nbatch, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels*preserved_dims)
        '''
        dareamean = dareapatch.mean(dim=0)
        dlevmean  = dlevpatch.mean(dim=0)
        dtimemean = dtimepatch.mean(dim=0)
        weights   = self.get_weights(dareamean,dlevmean,dtimemean,fieldpatch.device)
        feats = KernelModule.integrate(fieldpatch,weights,dareapatch,dlevpatch,dtimepatch,self.kerneldims)
        self.features = feats
        return feats.flatten(1)