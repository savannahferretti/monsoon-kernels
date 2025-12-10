#!/usr/bin/env python

import torch

class NonparametricKernelLayer(torch.nn.Module):
    '''
    Purpose: Learn free-form (non-parametric) kernels along selected dimensions and compute 
    quadrature-weighted integrals of predictor patches.
    '''
    def __init__(self,nfieldvars,patchshape,nkernels,kerneldims):
        '''
        Args:
        - nfieldvars (int): number of predictor fields in each patch
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldims (list[str]): subset of ('lat','lon','lev','time') specifying which dims the 
          kernel may vary over
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.patchshape = tuple(int(dimlength) for dimlength in patchshape)
        self.nkernels   = int(nkernels)
        self.kerneldims = tuple(kerneldims)
        kernelshape = [size if dim in self.kerneldims else 1 for dim,size in zip(('lat','lon','lev','time'),self.patchshape)]
        self.kernel = torch.nn.Parameter(torch.ones(self.nfieldvars,self.nkernels,*kernelshape))

    def forward(self,patch,quadweights):
        '''
        Purpose: Apply learned kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): predictor patch data with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadweights (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        (nbatch,nfieldvars,_,_,_,_),device,dtype = patch.shape,patch.device,patch.dtype
        if nfieldvars!=self.nfieldvars:
            raise ValueError(f'Expected {self.nfieldvars} field variables, got {nfieldvars}')
        quadweights,kernel = quadweights.to(device=device,dtype=dtype),self.kernel.to(device=device,dtype=dtype)
        kernelweights  = kernel/(torch.einsum('fkyxpt,yxpt->fk',kernel,quadweights)+1e-4)[:,:,None,None,None,None]
        kernelfeatures = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk',patch,kernelweights,quadweights)
        return kernelfeatures.reshape(nbatch,self.nfieldvars*self.nkernels)

class ParametricKernelLayer(torch.nn.Module):
    '''
    Purpose: Learn smooth parametric kernels along selected dimensions and compute 
    quadrature-weighted integrals of predictor patches.
    '''
    def __init__(self,nfieldvars,patchshape,nkernels,kerneldims,families):
        '''
        Args:
        - nfieldvars (int): number of predictor fields in each patch
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldims (list[str]): subset of ('lat','lon','lev','time') specifying which dims the 
          kernel may vary over
        - families (dict[str,str]): mapping of dimension name to kernel type ('gaussian' | 'exponential')
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.patchshape = tuple(int(dimlength) for dimlength in patchshape)
        self.nkernels   = int(nkernels)
        self.kerneldims = tuple(kerneldims)
        self.families   = {}
        families = families or {}
        for dim in self.kerneldims:
            family = families.get(dim,'gaussian')
            if family not in ('gaussian','exponential'):
                raise ValueError(f'Unknown family `{family}` for `{dim}` dimension')
            self.families[dim] = family
        self.logtau = torch.nn.ParameterDict()
        self.mu = torch.nn.ParameterDict()
        self.logsigma = torch.nn.ParameterDict()
        for dim in self.kerneldims:
            family = self.families[dim]
            if family=='gaussian':
                self.mu[dim]       = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))
                self.logsigma[dim] = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))
            elif family=='exponential':
                self.logtau[dim] = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))

    def forward(self, patch, quadweights):
        '''
        Purpose: Apply learned parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): predictor patch data with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadweights (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: (nbatch, nfieldvars*nkernels)
        '''
        (nbatch,nfieldvars,_,_,_,_),device,dtype = patch.shape,patch.device,patch.dtype
        if nfieldvars!=self.nfieldvars:
            raise ValueError(f'Expected {self.nfieldvars} field variables, got {nfieldvars}')
        plats,plons,plevs,ptimes = self.patchshape
        sizebydim = {'lat':plats, 'lon':plons, 'lev':plevs, 'time':ptimes}
        axisbydim = {'lat':2, 'lon':3, 'lev':4, 'time':5}
        kernelparams = torch.ones(self.nfieldvars, self.nkernels, plats, plons, plevs, ptimes, device=device, dtype=dtype)
        for dim in self.kerneldims:
            length = sizebydim[dim]
            axis = axisbydim[dim]
            family = self.families[dim]
            if family == 'gaussian':
                coord = torch.linspace(-1.0, 1.0, steps=length, device=device, dtype=dtype)
                mu = self.mu[dim].to(device=device, dtype=dtype)
                logsigma = self.logsigma[dim].to(device=device, dtype=dtype)
                sigma = torch.exp(logsigma)
                kernel1d = torch.exp(-0.5*((coord[None,None,:]-mu[...,None])/sigma[...,None])**2)
            elif family == 'exponential':
                coord = torch.arange(length, device=device, dtype=dtype)
                logtau = self.logtau[dim].to(device=device, dtype=dtype)
                tau = torch.exp(logtau) + 1e-6
                kernel1d = torch.exp(-coord[None,None,:]/tau[...,None])
            shape = [self.nfieldvars, self.nkernels, 1, 1, 1, 1]
            shape[axis] = length
            kernelparams = kernelparams*kernel1d.view(*shape)
        quadweights = quadweights.to(device=device, dtype=dtype)
        integrated = torch.einsum('fkyxpt,yxpt->fk', kernelparams, quadweights) + 1e-4
        kernelweights = kernelparams / integrated[:,:,None,None,None,None]
        features = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk', patch, kernelweights, quadweights)
        return features.reshape(nbatch, self.nfieldvars*self.nkernels)


class MainNN(torch.nn.Module):
    '''
    Purpose: Feed-forward NN that maps a flat feature vector to a scalar precipitation prediction.
    '''
    def __init__(self, nfeatures):
        '''
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
        - X (torch.Tensor): input feature tensor of shape (nbatch, nfeatures)
        Returns:
        - torch.Tensor: raw precipitation prediction tensor of shape (nbatch, 1)
        '''
        return self.layers(X)


class BaselineNN(torch.nn.Module):
    '''
    Purpose: Baseline NN that directly ingests space-height-time patches for one or more predictor 
    fields, optionally concatenated with local inputs, and maps them to a scalar precipitation prediction via MainNN.
    '''
    def __init__(self, nfieldvars, patchshape, nlocalvars, uselocal):
        '''
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
        self.uselocal = uselocal
        plats,plons,plevs,ptimes = self.patchshape
        nfeatures = self.nfieldvars*(plats*plons*plevs*ptimes)
        if self.uselocal:
            nfeatures += self.nlocalvars
        self.model = MainNN(nfeatures)

    def forward(self, patch, local=None):
        '''
        Purpose: Forward pass through BaselineNN.
        Args:
        - patch (torch.Tensor): (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - local (torch.Tensor | None): if uselocal is True, (nbatch, nlocalvars) local inputs; otherwise None
        Returns:
        - torch.Tensor: (nbatch, 1) raw precipitation prediction
        '''
        nbatch = patch.shape[0]
        patch = patch.reshape(nbatch,-1)
        if self.uselocal:
            if local is None:
                raise ValueError('`local` must be provided when uselocal is True')
            X = torch.cat([patch,local], dim=1)
        else:
            X = patch
        return self.model(X)


class KernelNN(torch.nn.Module):
    '''
    Purpose: Kernel NN that applies either non-parametric or parametric kernels over selected 
    dimensions of each predictor patch, then passes the resulting kernel features plus optional local inputs to MainNN.
    '''
    def __init__(self, kernellayer, nlocalvars, uselocal):
        '''
        Args:
        - kernellayer (torch.nn.Module): instance of NonparametricKernelLayer or ParametricKernelLayer
        - nlocalvars (int): number of local input variables
        - uselocal (bool): whether to use local inputs
        '''
        super().__init__()
        self.kernellayer = kernellayer
        self.nfieldvars = int(kernellayer.nfieldvars)
        self.patchshape = tuple(int(x) for x in kernellayer.patchshape)
        self.nkernels = int(kernellayer.nkernels)
        self.kerneldims = tuple(kernellayer.kerneldims)
        self.nlocalvars = int(nlocalvars)
        self.uselocal = uselocal
        nfeatures = self.nfieldvars*self.nkernels
        if self.uselocal:
            nfeatures += self.nlocalvars
        self.model = MainNN(nfeatures)

    def forward(self, patch, local=None, quadweights=None):
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
            quadweights = torch.ones(plats, plons, plevs, ptimes, device=device, dtype=dtype)
        kernelfeatures = self.kernellayer(patch, quadweights=quadweights)
        if self.uselocal:
            if local is None:
                raise ValueError('`local` must be provided when uselocal is True')
            X = torch.cat([kernelfeatures,local], dim=1)
        else:
            X = kernelfeatures
        return self.model(X)
