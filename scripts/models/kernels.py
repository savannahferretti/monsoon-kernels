#!/usr/bin/env python

import torch
import numpy as np

class KernelModule:

    @staticmethod
    def normalize(kernel,dareapatch,dlevpatch,dtimepatch,kerneldims,epsilon=1e-6):
        '''
        Purpose: Normalize kernel so that sum(k * quadrature_weights) = 1 over kerneled dimensions.
        Args:
        - kernel (torch.Tensor): unnormalized kernel with shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        - dareapatch (torch.Tensor): horizontal area weights with shape (plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights with shape (plevs,)
        - dtimepatch (torch.Tensor): time step weights with shape (ptimes,)
        - kerneldims (tuple[str] | list[str]): dimensions the kernel varies along
        - epsilon (float): stabilizer to avoid divide-by-zero (defaults to 1e-6)
        Returns:
        - torch.Tensor: normalized kernel weights with same shape as kernel
        '''
        kerneldims = tuple(kerneldims)
        plats,plons = dareapatch.shape
        plevs = dlevpatch.numel()
        ptimes = dtimepatch.numel()
        quad = torch.ones(1,1,plats,plons,plevs,ptimes,dtype=kernel.dtype,device=kernel.device)
        if ('lat' in kerneldims) or ('lon' in kerneldims):
            quad = quad*dareapatch[None,None,:,:,None,None]
        if 'lev' in kerneldims:
            quad = quad*dlevpatch[None,None,None,None,:,None]
        if 'time' in kerneldims:
            quad = quad*dtimepatch[None,None,None,None,None,:]
        kernelsum = (kernel*quad).sum(dim=(2,3,4,5))
        if 'lat' not in kerneldims:
            kernelsum = kernelsum/plats
        if 'lon' not in kerneldims:
            kernelsum = kernelsum/plons
        if 'lev' not in kerneldims:
            kernelsum = kernelsum/plevs
        if 'time' not in kerneldims:
            kernelsum = kernelsum/ptimes
        weights = kernel/(kernelsum[:,:,None,None,None,None]+epsilon)
        checksum = (weights*quad).sum(dim=(2,3,4,5))
        if 'lat' not in kerneldims:
            checksum = checksum/plats
        if 'lon' not in kerneldims:
            checksum = checksum/plons
        if 'lev' not in kerneldims:
            checksum = checksum/plevs
        if 'time' not in kerneldims:
            checksum = checksum/ptimes
        assert torch.allclose(checksum,torch.ones_like(checksum),atol=1e-4),f"Kernel normalization failed: weights sum to {checksum.mean().item():.6f} instead of 1.0"
        return weights

    @staticmethod
    def integrate(fieldpatch,weights,dareapatch,dlevpatch,dtimepatch,kerneldims):
        '''
        Purpose: Integrate predictor fields using normalized kernel weights with quadrature over kerneled dimensions.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - weights (torch.Tensor): normalized kernel weights with shape (nfieldvars, nkernels, plats or 1, plons or 1, plevs or 1, ptimes or 1)
        - dareapatch (torch.Tensor): horizontal area weights with shape (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights with shape (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights with shape (nbatch, ptimes)
        - kerneldims (tuple[str] | list[str]): dimensions the kernel varies along
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars, nkernels, ...) where ... are preserved non-kerneled dimensions
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
        return torch.nansum(weighted,dim=dimstosum) if dimstosum else weighted


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
        self.dlevfull   = None
        plats,plons,plevs,ptimes = patchshape
        kernelshape = [
            plats  if 'lat' in self.kerneldims else 1,
            plons  if 'lon' in self.kerneldims else 1,
            plevs  if 'lev' in self.kerneldims else 1,
            ptimes if 'time' in self.kerneldims else 1]
        kernel = torch.ones(self.nfieldvars,self.nkernels,*kernelshape)
        kernel = kernel+torch.randn_like(kernel)*0.2
        self.kernel = torch.nn.Parameter(kernel)

    def get_weights(self,dareapatch,dlevfull,dtimepatch,device):
        '''
        Purpose: Obtain normalized non-parametric kernel weights using fixed grid quadrature.
        Args:
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (plats, plons) or (nbatch, plats, plons)
        - dlevfull (torch.Tensor): full vertical thickness weights from fixed grid with shape (nlevs,)
        - dtimepatch (torch.Tensor): time step weights patch with shape (ptimes,) or (nbatch, ptimes)
        - device (str | torch.device): device to use
        Returns:
        - torch.Tensor: normalized kernel weights of shape (nfieldvars, nkernels, plats_or_1, plons_or_1, plevs_or_1, ptimes_or_1)
        '''
        self.kernel = self.kernel.to(device)
        dareapatch  = dareapatch.to(device)
        dlevfull    = dlevfull.to(device)
        dtimepatch  = dtimepatch.to(device)
        if self.dlevfull is None:
            self.dlevfull = dlevfull
        if dareapatch.dim()==3:
            dareapatch0 = dareapatch[0]
        else:
            dareapatch0 = dareapatch
        if dtimepatch.dim()==2:
            dtimepatch0 = dtimepatch[0]
        else:
            dtimepatch0 = dtimepatch
        self.weights = KernelModule.normalize(self.kernel,dareapatch0,self.dlevfull,dtimepatch0,self.kerneldims,epsilon=1e-6)
        return self.weights

    def forward(self,fieldpatch,dareapatch,dlevpatch,dtimepatch,dlevfull):
        '''
        Purpose: Apply learned non-parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights patch with shape (nbatch, ptimes)
        - dlevfull (torch.Tensor): full vertical thickness weights from fixed grid with shape (nlevs,)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels*preserved_dims)
        '''
        weights = self.get_weights(dareapatch,dlevfull,dtimepatch,fieldpatch.device)
        feats = KernelModule.integrate(fieldpatch,weights,dareapatch,dlevpatch,dtimepatch,self.kerneldims)
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
            Notes:
            - Implements k^(G)_s(s; μ_s, σ_s) = exp(-d(s,μ_s)²/(2σ_s²)) / normalization
            - Parameters μ (mean) and σ (std) are learned in normalized coordinate space [-1, 1]
            '''
            super().__init__()
            self.mean   = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))
            self.logstd = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))

        def forward(self,length,device):
            '''
            Purpose: Evaluate a Gaussian kernel along a coordinate in [-1,1].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: Gaussian kernel values with shape (nfieldvars, nkernels, length)
            Notes:
            - Uses normalized coordinates s ∈ [-1, 1]
            - For vertical: -1 ≈ top of atmosphere, +1 ≈ surface
            - Distance: d(s, μ_s) = |s - μ_s|
            '''
            coord = torch.linspace(-1.0,1.0,steps=length,device=device)
            std = torch.exp(self.logstd)
            kernel1d = torch.exp(-0.5*((coord[None,None,:]-self.mean[...,None])/std[...,None])**2)
            return kernel1d

    class TopHatKernel(torch.nn.Module):

        def __init__(self,nfieldvars,nkernels):
            '''
            Purpose: Initialize top-hat (uniform) kernel parameters along one dimension.
            Args:
            - nfieldvars (int): number of predictor fields
            - nkernels (int): number of kernels to learn per predictor field
            Notes:
            - Implements k^(TH)_s(s; s₁, s₂) = I(s ∈ [s₁, s₂]) / normalization
            - Parameters s₁ and s₂ are learned bounds in normalized coordinate space [-1, 1]
            - Assigns uniform weight within [s₁, s₂], zero outside
            '''
            super().__init__()
            # Initialize bounds to cover middle 50% of domain
            self.lower = torch.nn.Parameter(torch.full((int(nfieldvars),int(nkernels)), -0.5))
            self.upper = torch.nn.Parameter(torch.full((int(nfieldvars),int(nkernels)), 0.5))

        def forward(self,length,device):
            '''
            Purpose: Evaluate a top-hat kernel along a coordinate in [-1,1].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: top-hat kernel values with shape (nfieldvars, nkernels, length)
            Notes:
            - Uses normalized coordinates s ∈ [-1, 1]
            - For vertical: defines pressure layer between p₁ and p₂
            - Indicator function: I(s ∈ [s₁, s₂])
            '''
            coord = torch.linspace(-1.0,1.0,steps=length,device=device)

            # Ensure lower < upper by using sorted bounds
            s1 = torch.min(self.lower, self.upper)
            s2 = torch.max(self.lower, self.upper)

            # Indicator function: 1 if s ∈ [s₁, s₂], 0 otherwise
            kernel1d = ((coord[None,None,:] >= s1[...,None]) &
                       (coord[None,None,:] <= s2[...,None])).float()

            # Add small epsilon to avoid all-zero kernels
            kernel1d = kernel1d + 1e-8
            return kernel1d

    class ExponentialKernel(torch.nn.Module):

        def __init__(self,nfieldvars,nkernels):
            '''
            Purpose: Initialize exponential-decay kernel parameters along one dimension.
            Args:
            - nfieldvars (int): number of predictor fields
            - nkernels (int): number of kernels to learn per predictor field
            Notes:
            - Implements k^(EXP)(τ; τ₀) = exp(-τ/τ₀) · I(τ ∈ [0, τ_max]) / normalization
            - Parameter τ₀ (timescale) controls decay rate
            - For time: τ = t₀ - t (lag from present)
            - For vertical: τ = distance from surface or TOA
            '''
            super().__init__()
            self.logtau = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))

        def forward(self,length,device):
            '''
            Purpose: Evaluate an exponential kernel along a lag coordinate [0, τ_max].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: exponential kernel values with shape (nfieldvars, nkernels, length)
            Notes:
            - Uses lag coordinates τ ∈ [0, 1, 2, ..., length-1]
            - For time: τ=0 is present, τ increasing is further into past
            - For vertical: τ=0 could be surface, τ increasing is upward
            - Decay timescale τ₀ = exp(logtau)
            '''
            coord = torch.arange(length,device=device,dtype=torch.float32)
            tau = torch.exp(self.logtau)+1e-4
            kernel1d = torch.exp(-coord[None,None,:]/tau[...,None])
            return kernel1d

    class CosineKernel(torch.nn.Module):

        def __init__(self,nfieldvars,nkernels):
            '''
            Purpose: Initialize oscillating cosine kernel parameters along one dimension.
            Args:
            - nfieldvars (int): number of predictor fields
            - nkernels (int): number of kernels to learn per predictor field
            Notes:
            - Implements k^(C)_s(s; f, φ) = 0.5(1 + cos(2πfs + φ))
            - Parameters f (frequency) and φ (phase) are learned in normalized coordinate space [-1, 1]
            - Creates oscillating patterns with learnable number of cycles and phase offset
            - Unlike raised cosine (single bump), this has multiple peaks and valleys
            '''
            super().__init__()
            self.logfreq = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))  # log(frequency) for positivity
            self.phase = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))     # phase offset

        def forward(self,length,device):
            '''
            Purpose: Evaluate an oscillating cosine kernel along a coordinate in [-1,1].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: oscillating cosine kernel values with shape (nfieldvars, nkernels, length)
            Notes:
            - Uses normalized coordinates s ∈ [-1, 1]
            - For vertical: creates repeating oscillatory pattern with learnable frequency and phase
            - Frequency determines number of complete cycles over the domain
            - Phase shifts the oscillation pattern
            '''
            coord = torch.linspace(-1.0, 1.0, steps=length, device=device)
            freq = torch.exp(self.logfreq).clamp(min=0.5, max=10.0)  # Constrain to reasonable frequency range

            # Oscillating cosine: 0.5 * (1 + cos(2π * f * s + φ))
            # This gives values in [0, 1] with multiple oscillations
            kernel1d = 0.5 * (1.0 + torch.cos(
                2.0 * torch.pi * freq[..., None] * coord[None, None, :] + self.phase[..., None]
            ))

            # Add small epsilon to avoid all-zero kernels
            kernel1d = kernel1d + 1e-8
            return kernel1d

    def __init__(self,nfieldvars,nkernels,kerneldict):
        '''
        Purpose: Initialize parametric kernels along selected dimensions.
        Args:
        - nfieldvars (int): number of predictor fields
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldict (dict[str,str]): mapping of dimensions to kernel type
          ('gaussian' | 'tophat' | 'exponential' | 'cosine')
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.nkernels   = int(nkernels)
        self.kerneldict = dict(kerneldict)
        self.kerneldims = tuple(kerneldict.keys())
        self.weights    = None
        self.features   = None
        self.dlevfull   = None
        self.functions  = torch.nn.ModuleDict()
        for dim,function in self.kerneldict.items():
            if function=='gaussian':
                self.functions[dim] = self.GaussianKernel(self.nfieldvars,self.nkernels)
            elif function=='tophat':
                self.functions[dim] = self.TopHatKernel(self.nfieldvars,self.nkernels)
            elif function=='exponential':
                self.functions[dim] = self.ExponentialKernel(self.nfieldvars,self.nkernels)
            elif function=='cosine':
                self.functions[dim] = self.CosineKernel(self.nfieldvars,self.nkernels)
            else:
                raise ValueError(f'Unknown function type `{function}`; must be `gaussian`, `tophat`, `exponential`, or `cosine`')

    def get_weights(self,dareapatch,dlevfull,dtimepatch,device):
        '''
        Purpose: Obtain normalized parametric kernel weights using fixed grid quadrature.
        Args:
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (plats, plons) or (nbatch, plats, plons)
        - dlevfull (torch.Tensor): full vertical thickness weights from fixed grid with shape (nlevs,)
        - dtimepatch (torch.Tensor): time step weights patch with shape (ptimes,) or (nbatch, ptimes)
        - device (str | torch.device): device to use
        Returns:
        - torch.Tensor: normalized kernel weights of shape (nfieldvars, nkernels, plats, plons, plevs, ptimes)
        '''
        dareapatch = dareapatch.to(device)
        dlevfull   = dlevfull.to(device)
        dtimepatch = dtimepatch.to(device)
        if self.dlevfull is None:
            self.dlevfull = dlevfull
        if dareapatch.dim()==3:
            dareapatch0 = dareapatch[0]
        else:
            dareapatch0 = dareapatch
        if dtimepatch.dim()==2:
            dtimepatch0 = dtimepatch[0]
        else:
            dtimepatch0 = dtimepatch
        plats,plons = dareapatch0.shape
        plevs       = self.dlevfull.numel()
        ptimes      = dtimepatch0.numel()
        kernel = torch.ones(self.nfieldvars,self.nkernels,plats,plons,plevs,ptimes,dtype=dareapatch0.dtype,device=device)
        for ax,dim in enumerate(('lat','lon','lev','time'),start=2):
            if dim in self.kerneldims:
                kernel1d = self.functions[dim](kernel.shape[ax],device)
                # kernel1d has shape (nfieldvars, nkernels, length)
                # Need to reshape to (nfieldvars, nkernels, 1, 1, 1, 1) with length at position ax
                view = [kernel.shape[0], kernel.shape[1], 1, 1, 1, 1]
                view[ax] = kernel.shape[ax]
                kernel = kernel*kernel1d.view(*view)
        self.weights = KernelModule.normalize(kernel,dareapatch0,self.dlevfull,dtimepatch0,self.kerneldims)
        return self.weights

    def forward(self,fieldpatch,dareapatch,dlevpatch,dtimepatch,dlevfull):
        '''
        Purpose: Apply learned parametric kernels to a batch of patches and compute kernel-integrated features.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights patch with shape (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights patch with shape (nbatch, ptimes)
        - dlevfull (torch.Tensor): full vertical thickness weights from fixed grid with shape (nlevs,)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*nkernels*preserved_dims)
        '''
        weights = self.get_weights(dareapatch,dlevfull,dtimepatch,fieldpatch.device)
        feats = KernelModule.integrate(fieldpatch,weights,dareapatch,dlevpatch,dtimepatch,self.kerneldims)
        self.features = feats
        return feats.flatten(1)