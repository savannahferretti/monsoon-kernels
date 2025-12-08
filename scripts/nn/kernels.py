#!/usr/bin/env python

import torch

class QuadratureWeights:
    
    @staticmethod
    def spacing(coord):
        '''
        Purpose: Estimate spacing between neighboring coordinate points using centered differences in the interior 
        and one-sided differences at the boundaries.
        Args:
        - coord (torch.Tensor): 1D tensor of monotonically increasing coordinate values
        Returns:
        - torch.Tensor: 1D tensor of estimated spacing between grid points with the same length as 'coord'
        '''
        spacing = torch.empty_like(coord)
        spacing[1:-1] = 0.5*(coord[2:]-coord[:-2])
        spacing[0]    = coord[1]-coord[0]
        spacing[-1]   = coord[-1]-coord[-2]
        return torch.abs(spacing)

    @staticmethod
    def compute(lat,lon,lev,time,rearth=6.371e6):
        '''
        Purpose: Compute ΔA Δp Δt quadrature weights for a 4D grid (lat, lon, lev, time). These weights ensure that 
        a sum over grid points approximates a physical integral over space and time.
        Args:
        - lat (torch.Tensor): (nlats,) latitudes in degrees
        - lon (torch.Tensor): (nlons,) longitudes in degrees
        - lev (torch.Tensor): (nlevs,) vertical coordinate
        - time (torch.Tensor): (ntimes,) time coordinate
        - rearth (float): Earth's radius [m]
        Returns:
        - torch.Tensor: (nlats, nlons, nlevs, ntimes) ΔA Δp Δt weights
        '''
        if not (lat.ndim==lon.ndim==lev.ndim==time.ndim==1):
            raise ValueError('All inputs to `QuadratureWeights.compute()` must be 1D tensors')
        nlats  = lat.numel()
        nlons  = lon.numel()
        nlevs  = lev.numel()
        ntimes = time.numel()
        dlat  = QuadratureWeights.spacing(torch.deg2rad(lat))
        dlon  = QuadratureWeights.spacing(torch.deg2rad(lon))
        dlev  = QuadratureWeights.spacing(lev)
        dtime = QuadratureWeights.spacing(time)
        area  = ((rearth**2)*torch.cos(torch.deg2rad(lat))*dlat).reshape(nlats,1)*dlon.reshape(1,nlons)
        quadweights = area.reshape(nlats,nlons,1,1)*dlev.reshape(1,1,nlevs,1)*dtime.reshape(1,1,1,ntimes)
        return quadweights.to(dtype=torch.float32)


class NonparametricKernelLayer(torch.nn.Module):
    
    def __init__(self,nfieldvars,patchshape,nkernels,kerneldims):
        '''
        Purpose: Initialize free-form (non-parametric) kernels along selected dimensions and compute 
        quadrature-weighted integrals of predictor patches.
        Args:
        - nfieldvars (int): number of predictor fields in each patch
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nkernels (int): number of kernels to learn per predictor field variable
        - kerneldims (list[str]): subset of ('lat','lon','lev','time') specifying which dims the kernel may vary over
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.patchshape = tuple(int(dimlength) for dimlength in patchshape)
        self.nkernels   = int(nkernels)
        self.kerneldims = tuple(kerneldims)
        paramsizes = [size if dim in self.kerneldims else 1
                      for dim,size in zip(('lat','lon','lev','time'),self.patchshape)]
        self.kernelparams = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels,*paramsizes))

    def _normalized_weights(self,quadweights,device,dtype):
        '''
        Purpose: Normalize kernel parameters by their quadrature-weighted integral.
        Args:
        - quadweights (torch.Tensor): (plats, plons, plevs, ptimes)
        - device (torch.device): target device
        - dtype (torch.dtype): target dtype
        Returns:
        - torch.Tensor: normalized kernel weights (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        quadweights  = quadweights.to(device=device,dtype=dtype)
        kernelparams = self.kernelparams.to(device=device,dtype=dtype)
        integrated    = torch.einsum('fkyxpt,yxpt->fk',kernelparams,quadweights)+1e-8
        kernelweights = kernelparams/integrated[:,:,None,None,None,None]
        return kernelweights

    def weights(self,quadweights):
        '''
        Purpose: Return normalized kernel weights for plotting/diagnostics.
        Args:
        - quadweights (torch.Tensor): (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        device = self.kernelparams.device
        dtype  = self.kernelparams.dtype
        return self._normalized_weights(quadweights,device,dtype)
    
    def forward(self,patch,quadweights):
        '''
        Purpose: Apply learned free-form kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadweights (torch.Tensor): (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: (nbatch, nfieldvars*nkernels)
        '''
        (nbatch,nfieldvars,_,_,_,_),device,dtype = patch.shape,patch.device,patch.dtype
        if nfieldvars!=self.nfieldvars:
            raise ValueError(f'Expected {self.nfieldvars} field variables, got {nfieldvars}')
        kernelweights = self._normalized_weights(quadweights,device,dtype)
        features      = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk',patch,kernelweights,quadweights.to(device=device,dtype=dtype))
        return features.reshape(nbatch,self.nfieldvars*self.nkernels)
        

class ParametricKernelLayer(torch.nn.Module):

    def __init__(self,nfieldvars,patchshape,nkernels,kerneldims,families):
        '''
        Purpose: Initialize smooth parametric kernels along selected dimensions and compute 
        quadrature-weighted integrals of predictor patches.
        Args:
        - nfieldvars (int): number of predictor fields in each patch
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nkernels (int): number of kernels to learn per predictor field variable
        - kerneldims (list[str]): subset of ('lat','lon','lev','time') specifying which dims the kernel may vary over
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
        self.logtau   = torch.nn.ParameterDict()
        self.mu       = torch.nn.ParameterDict()
        self.logsigma = torch.nn.ParameterDict()
        for dim in self.kerneldims:
            family = self.families[dim]
            if family=='gaussian':
                self.mu[dim]       = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))
                self.logsigma[dim] = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))
            elif family=='exponential':
                self.logtau[dim]   = torch.nn.Parameter(torch.zeros(self.nfieldvars,self.nkernels))

    @staticmethod
    def gaussian(coord,mu,logsigma):
        ''' 
        Purpose: Evaluate a Gaussian kernel along one axis for each (field, kernel) pair.
        '''
        sigma = torch.exp(logsigma)
        return torch.exp(-0.5*((coord[None,None,:]-mu[...,None])/sigma[...,None])**2)
    
    @staticmethod
    def exponential(coord,logtau):
        ''' 
        Purpose: Evaluate an exponential-decay kernel along one axis for each (field, kernel) pair.
        '''
        tau = torch.exp(logtau)+1e-6
        return torch.exp(-coord[None,None,:]/tau[...,None])

    def _build_kernel_params(self,device,dtype):
        '''
        Purpose: Construct parametric kernel values over the patch grid.
        Args:
        - device (torch.device): target device
        - dtype (torch.dtype): target dtype
        Returns:
        - torch.Tensor: (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        plats,plons,plevs,ptimes = self.patchshape
        sizebydim = {'lat':plats,'lon':plons,'lev':plevs,'time':ptimes}
        axisbydim = {'lat':2,'lon':3,'lev':4,'time':5}
        kernelparams = torch.ones(
            self.nfieldvars,self.nkernels,plats,plons,plevs,ptimes,
            device=device,dtype=dtype)
        for dim in self.kerneldims:
            length = sizebydim[dim]
            axis   = axisbydim[dim]
            family = self.families[dim]
            if family=='gaussian':
                coord    = torch.linspace(-1.0,1.0,steps=length,device=device,dtype=dtype)
                mu       = self.mu[dim].to(device=device,dtype=dtype)
                logsigma = self.logsigma[dim].to(device=device,dtype=dtype)
                kernel1D = self.gaussian(coord,mu,logsigma)
            elif family=='exponential':
                coord  = torch.arange(length,device=device,dtype=dtype)
                logtau = self.logtau[dim].to(device=device,dtype=dtype)
                kernel1D = self.exponential(coord,logtau)
            shape = [self.nfieldvars,self.nkernels,1,1,1,1]
            shape[axis] = length
            kernelparams = kernelparams*kernel1D.view(*shape)
        return kernelparams

    def _normalized_weights(self,quadweights,device,dtype):
        '''
        Purpose: Normalize parametric kernels by their quadrature-weighted integral.
        Returns:
        - torch.Tensor: (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        quadweights  = quadweights.to(device=device,dtype=dtype)
        kernelparams = self._build_kernel_params(device,dtype)
        integrated    = torch.einsum('fkyxpt,yxpt->fk',kernelparams,quadweights)+1e-8
        kernelweights = kernelparams/integrated[:,:,None,None,None,None]
        return kernelweights

    def weights(self,quadweights):
        '''
        Purpose: Return normalized parametric kernel weights for plotting/diagnostics.
        Args:
        - quadweights (torch.Tensor): (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        return self._normalized_weights(quadweights,device,dtype)

    def forward(self,patch,quadweights):
        '''
        Purpose: Apply learned parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - patch (torch.Tensor): (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - quadweights (torch.Tensor): (plats, plons, plevs, ptimes)
        Returns:
        - torch.Tensor: (nbatch, nfieldvars*nkernels)
        '''
        (nbatch,nfieldvars,_,_,_,_),device,dtype = patch.shape,patch.device,patch.dtype
        if nfieldvars!=self.nfieldvars:
            raise ValueError(f'Expected {self.nfieldvars} field variables, got {nfieldvars}')
        kernelweights = self._normalized_weights(quadweights,device,dtype)
        features      = torch.einsum('bfyxpt,fkyxpt,yxpt->bfk',patch,kernelweights,quadweights.to(device=device,dtype=dtype))
        return features.reshape(nbatch,self.nfieldvars*self.nkernels)