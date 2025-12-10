#!/usr/bin/env python

import torch

class NonparametricKernelLayer(torch.nn.Module):

    def __init__(self,nfieldvars,patchshape,nkernels,kernelconfig):
        '''
        Purpose: Initialize free-form (non-parametric) kernels along selected dimensions.
        Args:
        - nfieldvars (int): number of predictor fields
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nkernels (int): number of kernels to learn per predictor field
        - kernelconfig (dict[str,str]): kernel confugiration
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.patchshape = tuple(int(dimlength) for dimlength in patchshape)
        self.nkernels   = int(nkernels)
        self.kerneldims = tuple(kernelconfig.get('dims'))
        kernelshape = [size if dim in self.kerneldims else 1 for dim,size in zip(('lat','lon','lev','time'),self.patchshape)]
        self.kernel = torch.nn.Parameter(torch.ones(self.nfieldvars,self.nkernels,*kernelshape))

    def forward(self,patch,quadweights):
        '''
        Purpose: Apply learned free-form kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadweights (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        (nbatch,nfieldvars,_,_,_,_),device,dtype = patch.shape,patch.device,patch.dtype
        kernel      = self.kernel.to(device=device,dtype=dtype)
        quadweights = quadweights.to(device=device,dtype=dtype)
        kernelweights  = kernel/(torch.einsum('fkyxpt,yxpt->fk',kernel,quadweights)+1e-4)[:,:,None,None,None,None]
        kernelfeatures = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk',patch,kernelweights,quadweights)
        return kernelfeatures.reshape(nbatch,self.nfieldvars*self.nkernels)

class ParametricKernelLayer(torch.nn.Module):

    def __init__(self,nfieldvars,patchshape,nkernels,kernelconfig):
        '''
        Purpose: Initialize smooth parametric kernels along selected dimensions.
        Args:
        - nfieldvars (int): number of predictor fields
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nkernels (int): number of kernels to learn per predictor field
        - kernelconfig (dict[str,str]): kernel configuration
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.patchshape = tuple(int(dimlength) for dimlength in patchshape)
        self.nkernels   = int(nkernels)
        self.kerneldims = tuple(kernelconfig.keys())
        self.functions  = {dim:function for dim,function in kernelconfig.items()}
        self.logtau = torch.nn.ParameterDict()
        self.mean   = torch.nn.ParameterDict()
        self.logstd = torch.nn.ParameterDict()
        for dim,function in self.functions.items():
            if function=='gaussian':
                self.mean[dim]   = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))
                self.logstd[dim] = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))
            elif function=='exponential':
                self.logtau[dim] = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))
            else:
                raise ValueError(f'Unknown function type `{function}`, must be either `gaussian` or `exponential`')
    
    @staticmethod
    def gaussian(coord,mean,logstd):
        ''' 
        Purpose: Evaluate a Gaussian kernel along one axis for each (field, kernel) pair.
        Args:
        - coord (torch.Tensor): 1D coordinate tensor
        - mean (torch.Tensor): center parameters with shape (nfieldvars, nkernels)
        - logstd (torch.Tensor): log standard deviation parameters with shape (nfieldvars, nkernels)
        Returns:
        - torch.Tensor: Gaussian kernel values with shape (nfieldvars, nkernels, len(coord))
        '''
        std = torch.exp(logstd)
        return torch.exp(-0.5*((coord[None,None,:]-mean[...,None])/std[...,None])**2)
    
    @staticmethod
    def exponential(coord,logtau):
        ''' 
        Purpose: Evaluate an exponential-decay kernel along one axis for each (field, kernel) pair.
        Args:
        - coord (torch.Tensor): 1D coordinate tensor
        - logtau (torch.Tensor): log decay scale parameters with shape (nfieldvars, nkernels)
        Returns:
        - torch.Tensor: exponential kernel values with shape (nfieldvars, nkernels, len(coord))
        '''
        tau = torch.exp(logtau)+1e-4
        return torch.exp(-coord[None,None,:]/tau[...,None])
        
    def forward(self,patch,quadweights):
        '''
        Purpose: Apply learned parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadweights (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels)
        '''
        (nbatch,nfieldvars,plats,plons,plevs,ptimes),device,dtype = patch.shape,patch.device,patch.dtype
        kernel = torch.ones(self.nfieldvars,self.nkernels,plats,plons,plevs,ptimes,device=device,dtype=dtype)
        for dim in self.kerneldims:
            dimidx  = ('lat','lon','lev','time').index(dim)     
            length  = self.patchshape[dimidx]    
            axis    = 2+dimidx            
            function = self.functions[dim]
            if function=='gaussian':
                coord    = torch.linspace(-1.0,1.0,steps=length,device=device,dtype=dtype)
                mean     = self.mean[dim].to(device=device,dtype=dtype)
                logstd   = self.logstd[dim].to(device=device,dtype=dtype)
                kernel1D = self.gaussian(coord,mean,logstd)
            elif function=='exponential':  
                coord    = torch.arange(length,device=device,dtype=dtype)
                logtau   = self.logtau[dim].to(device=device,dtype=dtype)
                kernel1D = self.exponential(coord,logtau)
            kernelshape = [self.nfieldvars,self.nkernels,1,1,1,1]
            kernelshape[axis] = length
            kernel = kernel*kernel1D.view(*kernelshape)
        quadweights = quadweights.to(device=device,dtype=dtype)
        kernelweights  = kernel/(torch.einsum('fkyxpt,yxpt->fk',kernel,quadweights)+1e-4)[:,:,None,None,None,None]
        kernelfeatures = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk',patch,kernelweights,quadweights)
        return kernelfeatures.reshape(nbatch,self.nfieldvars*self.nkernels)

class MainNN(torch.nn.Module):

    def __init__(self,nfeatures):
        '''
        Purpose: Initialize a feed-forward neural network that nonlinearly maps a feature vector to a scalar prediction.
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
        - X (torch.Tensor): input features with shape (nbatch, nfeatures)
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        return self.layers(X).squeeze()

class BaselineNN(torch.nn.Module):

    def __init__(self,nfieldvars,patchshape,nlocalvars,uselocal):
        '''
        Purpose: Initialize a neural network that directly ingests space-height-time patches for one or more predictor 
        fields, then passes the patches plus optional local inputs to MainNN.
        Args:
        - nfieldvars (int): number of predictor fields
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nlocalvars (int): number of local inputs
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
        - patch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - local (torch.Tensor | None): local inputs with shape (nbatch, nlocalvars) if uselocal is True, otherwise None
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        nbatch = patch.shape[0]
        patch  = patch.reshape(nbatch,-1)
        if self.uselocal:
            if local is None:
                raise ValueError('`local` must be provided when `uselocal` is True')
            X = torch.cat([patch,local],dim=1)
        else:
            X = patch
        return self.model(X)

class KernelNN(torch.nn.Module):

    def __init__(self,kernellayer,nlocalvars,uselocal):
        '''
        Purpose: Initialize a neural network that applies either non-parametric or parametric kernels over selected 
        dimensions of each predictor patch, then passes the resulting kernel features plus optional local inputs to MainNN.
        Args:
        - kernellayer (torch.nn.Module): instance of NonparametricKernelLayer or ParametricKernelLayer
        - nlocalvars (int): number of local inputs
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

    def forward(self,patch,quadweights,local=None):
        '''
        Purpose: Forward pass through KernelNN.
        Args:
        - patch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadweights (torch.Tensor): quadrature weights with shape (plats, plons, plevs, ptimes)
        - local (torch.Tensor | None): local inputs with shape (nbatch, nlocalvars) if uselocal is True, otherwise None
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        (plats,plons,plevs,ptimes),device,dtype = self.patchshape,patch.device,patch.dtype
        kernelfeatures = self.kernellayer(patch,quadweights)
        if self.uselocal:
            if local is None:
                raise ValueError('`local` must be provided when `uselocal` is True')
            X = torch.cat([kernelfeatures,local],dim=1)
        else:
            X = kernelfeatures
        return self.model(X)