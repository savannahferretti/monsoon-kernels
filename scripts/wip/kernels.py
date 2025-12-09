#!/usr/bin/env python

import torch

class KernelLayer(torch.nn.Module):
    
    def __init__(self,nfieldvars,patchshape,nkernels,kerneldims):
        '''
        Purpose: Initialize common kernel layer attributes.
        Args:
        - nfieldvars (int): number of predictor fields in each patch
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldims (list[str]): subset of ('lat','lon','lev','time') specifying dimensions the kernel varies over
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.patchshape = tuple(int(dim) for dim in patchshape)
        self.nkernels   = int(nkernels)
        self.kerneldims = tuple(kerneldims)
    
    def normalize(self,quadweights,device,dtype):
        '''
        Purpose: Normalize kernel values by their quadrature-weighted integral.
        Args:
        - quadweights (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        - device (torch.device): target device for computation
        - dtype (torch.dtype): target data type for computation
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        quadweights = quadweights.to(device=device,dtype=dtype)
        kernel      = self.get_kernel(device,dtype)
        integrated  = torch.einsum('fkyxpt,yxpt->fk',kernelparams,quadweights)+1e-8
        weights     = kernel/integrated[:,:,None,None,None,None]
        return weights
    
    def weights(self,quadweights):
        '''
        Purpose: Return normalized kernel weights for plotting and diagnostics.
        Args:
        - quadweights (torch.Tensor): quadrature weights with shape (plats,plons,plevs,ptimes)
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars,nkernels,plats,plons,plevs,ptimes)
        '''
        device,dtype = next(self.parameters()).device,next(self.parameters()).dtype
        return self._normalized_weights(quadweights,device,dtype)
    
    def forward(self,patch,quadweights):
        '''
        Purpose: Apply learned kernels to batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): input patches with shape (nbatch,nfieldvars,plats,plons,plevs,ptimes)
        - quadweights (torch.Tensor): quadrature weights with shape (plats,plons,plevs,ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch,nfieldvars*nkernels)
        '''
        (nbatch,nfieldvars,_,_,_,_),device,dtype = patch.shape,patch.device,patch.dtype
        if nfieldvars!=self.nfieldvars:
            raise ValueError(f'Expected {self.nfieldvars} field variables, got {nfieldvars}')
        kernelweights = self._normalized_weights(quadweights,device,dtype)
        features = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk',patch,kernelweights,quadweights.to(device=device,dtype=dtype))
        return features.reshape(nbatch,self.nfieldvars*self.nkernels)
    
    def _get_kernel_params(self,device,dtype):
        '''
        Purpose: Get kernel parameters on specified device and dtype (implemented by subclasses).
        Args:
        - device (torch.device): target device
        - dtype (torch.dtype): target data type
        Returns:
        - torch.Tensor: kernel parameters with shape (nfieldvars,nkernels,plats,plons,plevs,ptimes)
        '''
        raise NotImplementedError


class NonparametricKernelLayer(KernelLayer):
    
    def __init__(self,nfieldvars,patchshape,nkernels,kerneldims):
        '''
        Purpose: Initialize non-parametric kernel parameters as free learnable weights.
        Args:
        - nfieldvars (int): number of predictor fields in each patch
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldims (list[str]): subset of ('lat','lon','lev','time') specifying dimensions the kernel varies over
        '''
        super().__init__(nfieldvars,patchshape,nkernels,kerneldims)
        paramsizes = [size if dim in kerneldims else 1 for dim,size in zip(('lat','lon','lev','time'),patchshape)]
        self.kernelparams = torch.nn.Parameter(torch.zeros(nfieldvars,nkernels,*paramsizes))
    
    def get_kernel_params(self,device,dtype):
        return self.kernelparams.to(device=device,dtype=dtype)


class ParametricKernelLayer(KernelLayer):

    def __init__(self,nfieldvars,patchshape,nkernels,kerneldims,families):
        '''
        Purpose: Initialize parametric kernel parameters (means, scales) for smooth basis functions.
        Args:
        - nfieldvars (int): number of predictor fields in each patch
        - patchshape (tuple[int,int,int,int]): (plats,plons,plevs,ptimes)
        - nkernels (int): number of kernels to learn per predictor field variable
        - kerneldims (list[str]): subset of ('lat','lon','lev','time') specifying dimensions the kernel varies over
        - families (dict[str,str]): mapping of dimension name to kernel type ('gaussian' or 'exponential')
        '''
        super().__init__(nfieldvars,patchshape,nkernels,kerneldims)
        self.families = {dim:families.get(dim,'gaussian') for dim in kerneldims}
        for family in self.families.values():
            if family not in ('gaussian','exponential'):
                raise ValueError(f'Unknown family: {family}')
        self.logtau,self.mu,self.logsigma = torch.nn.ParameterDict(),torch.nn.ParameterDict(),torch.nn.ParameterDict()
        for dim in kerneldims:
            if self.families[dim]=='gaussian':
                self.mu[dim] = torch.nn.Parameter(torch.zeros(nfieldvars,nkernels))
                self.logsigma[dim] = torch.nn.Parameter(torch.zeros(nfieldvars,nkernels))
            else:
                self.logtau[dim] = torch.nn.Parameter(torch.zeros(nfieldvars,nkernels))

    def get_kernel_params(self,device,dtype):
        '''
        Purpose: Build parametric kernels by multiplying 1D basis functions along each dimension.
        Args:
        - device (torch.device): target device
        - dtype (torch.dtype): target data type
        Returns:
        - torch.Tensor: kernel parameters with shape (nfieldvars,nkernels,plats,plons,plevs,ptimes)
        '''
        plats,plons,plevs,ptimes = self.patchshape
        sizebydim = {'lat':plats,'lon':plons,'lev':plevs,'time':ptimes}
        axisbydim = {'lat':2,'lon':3,'lev':4,'time':5}
        kernelparams = torch.ones(self.nfieldvars,self.nkernels,plats,plons,plevs,ptimes,device=device,dtype=dtype)
        for dim in self.kerneldims:
            length,axis = sizebydim[dim],axisbydim[dim]
            if self.families[dim]=='gaussian':
                coord = torch.linspace(-1.0,1.0,steps=length,device=device,dtype=dtype)
                mu,logsigma = self.mu[dim].to(device=device,dtype=dtype),self.logsigma[dim].to(device=device,dtype=dtype)
                sigma = torch.exp(logsigma)
                kernel1d = torch.exp(-0.5*((coord[None,None,:]-mu[...,None])/sigma[...,None])**2)
            else:
                coord = torch.arange(length,device=device,dtype=dtype)
                tau = torch.exp(self.logtau[dim].to(device=device,dtype=dtype))+1e-6
                kernel1d = torch.exp(-coord[None,None,:]/tau[...,None])
            shape = [self.nfieldvars,self.nkernels,1,1,1,1]
            shape[axis] = length
            kernelparams = kernelparams*kernel1d.view(*shape)
        return kernelparams