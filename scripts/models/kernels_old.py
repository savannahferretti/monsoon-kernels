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

    class ExponentialKernel(torch.nn.Module):

        def __init__(self,nfieldvars,nkernels):
            '''
            Purpose: Initialize exponential-decay kernel parameters along one dimension.
            Args:
            - nfieldvars (int): number of predictor fields
            - nkernels (int): number of kernels to learn per predictor field
            Notes:
            - Implements k^(EXP)(τ; τ₀, d) = exp(-τ/τ₀) · I(τ ∈ [0, τ_max]) / normalization
            - Parameter τ₀ (timescale) controls decay rate
            - Parameter d (direction) controls whether decay is from TOA or surface:
              * d → -∞ (sigmoid→0): decay from TOA downward (τ=0 at top, increases toward surface)
              * d → +∞ (sigmoid→1): decay from surface upward (τ=0 at bottom, increases toward TOA)
            - For time: τ = t₀ - t (lag from present)
            '''
            super().__init__()
            self.logtau = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))
            self.direction = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))

        def forward(self,length,device):
            '''
            Purpose: Evaluate an exponential kernel along a lag coordinate [0, τ_max].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: exponential kernel values with shape (nfieldvars, nkernels, length)
            Notes:
            - Coordinate convention (matching GaussianKernel): index 0 = TOA, index (length-1) = surface
            - Direction parameter determines lag direction:
              * direction < 0: lag from TOA (0, 1, 2, ..., length-1) — peaks at TOA
              * direction > 0: lag from surface (length-1, ..., 2, 1, 0) — peaks at surface
            - Decay timescale τ₀ = exp(logtau)
            - Uses differentiable sigmoid blending for smooth gradient flow
            '''
            coord = torch.arange(length,device=device,dtype=torch.float32)
            tau = torch.exp(self.logtau)+1e-4
            from_surface = torch.sigmoid(self.direction)
            lag_from_top = coord[None,None,:]  
            lag_from_bottom = (length-1) - coord[None,None,:]  
            lag = (1.0 - from_surface[...,None]) * lag_from_top + from_surface[...,None] * lag_from_bottom
            kernel1d = torch.exp(-lag/tau[...,None])
            return kernel1d


    class MixtureGaussianKernel(torch.nn.Module):

        def __init__(self,nfieldvars,nkernels):
            '''
            Purpose: Initialize mixture-of-Gaussians kernel parameters along one dimension.
            Args:
            - nfieldvars (int): number of predictor fields
            - nkernels (int): number of kernels to learn per predictor field
            Notes:
            - Implements k^(MG)_s(s; μ₁, σ₁, μ₂, σ₂, w₁, w₂) = w₁·N(μ₁,σ₁²) + w₂·N(μ₂,σ₂²)
            - Two Gaussians with learnable centers (μ₁, μ₂), widths (σ₁, σ₂), and independent weights (w₁, w₂)
            - Weights are unconstrained - can be positive or negative for full flexibility:
              * w₁ > 0, w₂ > 0: two positive contributions (e.g., surface + lower free-troposphere)
              * w₁ > 0, w₂ < 0: positive + negative (e.g., boundary layer positive, free-troposphere negative)
            - Useful for bimodal patterns or opposing contributions at different levels
            '''
            super().__init__()
            self.center1 = torch.nn.Parameter(torch.full((int(nfieldvars),int(nkernels)), -0.5))
            self.center2 = torch.nn.Parameter(torch.full((int(nfieldvars),int(nkernels)), 0.5))
            self.logwidth1 = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))
            self.logwidth2 = torch.nn.Parameter(torch.zeros(int(nfieldvars),int(nkernels)))
            self.weight1 = torch.nn.Parameter(torch.ones(int(nfieldvars),int(nkernels)))
            self.weight2 = torch.nn.Parameter(torch.ones(int(nfieldvars),int(nkernels)))

        def get_components(self,length,device):
            '''
            Purpose: Compute individual Gaussian components separately (for visualization).
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - tuple: (component1, component2) each with shape (nfieldvars, nkernels, length)
            Notes:
            - Returns weighted Gaussian components before they are combined
            - Useful for visualizing each component separately in plots
            '''
            coord = torch.linspace(-1.0, 1.0, steps=length, device=device)
            width1 = torch.exp(self.logwidth1).clamp(min=0.1, max=2.0)
            width2 = torch.exp(self.logwidth2).clamp(min=0.1, max=2.0)
            dist1 = coord[None,None,:] - self.center1[...,None]
            dist2 = coord[None,None,:] - self.center2[...,None]
            gauss1 = torch.exp(-dist1**2 / (2 * width1[...,None]**2))
            gauss2 = torch.exp(-dist2**2 / (2 * width2[...,None]**2))
            component1 = self.weight1[...,None] * gauss1
            component2 = self.weight2[...,None] * gauss2
            return component1, component2

        def forward(self,length,device):
            '''
            Purpose: Evaluate a mixture-of-Gaussians kernel along a coordinate in [-1,1].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: mixture kernel values with shape (nfieldvars, nkernels, length)
            Notes:
            - Uses normalized coordinates s ∈ [-1, 1]
            - Combines two Gaussian bumps with independent learnable weights
            - Allows positive/positive, positive/negative, or any combination
            - Examples: two peaks, center-surround, or single peak with negative surround
            '''
            component1, component2 = self.get_components(length, device)
            kernel1d = component1 + component2
            kernel1d = kernel1d + 1e-8
            return kernel1d

    def __init__(self,nfieldvars,nkernels,kerneldict):
        '''
        Purpose: Initialize parametric kernels along selected dimensions.
        Args:
        - nfieldvars (int): number of predictor fields
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldict (dict[str,str|list[str]]): mapping of dimensions to kernel type(s)
          Supports two formats:
          1. Single kernel for all fields: {"lev": "gaussian"}
          2. Per-field kernels: {"lev": ["exponential", "gaussian", "cosine"]}
          Valid kernel types: 'gaussian', 'tophat', 'exponential', 'cosine', 'mixture', 'bidirectional'
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.nkernels   = int(nkernels)
        self.kerneldict = dict(kerneldict)
        self.kerneldims = tuple(kerneldict.keys())
        self.weights    = None
        self.component_weights = None  # For mixture kernels: store separate component weights
        self.features   = None
        self.dlevfull   = None
        self.functions  = torch.nn.ModuleDict()
        self.perfield   = {}  # Track which dimensions use per-field kernels

        for dim,function_spec in self.kerneldict.items():
            # Check if per-field specification (list) or single specification (string)
            if isinstance(function_spec, list):
                # Per-field kernels: one kernel type per field variable
                if len(function_spec) != self.nfieldvars:
                    raise ValueError(f'Per-field kernel list for dim `{dim}` must have length {self.nfieldvars}, got {len(function_spec)}')
                self.perfield[dim] = True
                self.functions[dim] = torch.nn.ModuleList([
                    self._create_kernel(func, 1, self.nkernels) for func in function_spec
                ])
            else:
                # Single kernel for all fields
                self.perfield[dim] = False
                self.functions[dim] = self._create_kernel(function_spec, self.nfieldvars, self.nkernels)

    def _create_kernel(self, function, nfieldvars, nkernels):
        '''
        Purpose: Factory method to create a kernel instance from a function name.
        Args:
        - function (str): kernel type name
        - nfieldvars (int): number of field variables for this kernel
        - nkernels (int): number of kernels per field
        Returns:
        - torch.nn.Module: kernel instance
        '''
        if function=='gaussian':
            return self.GaussianKernel(nfieldvars,nkernels)
        elif function=='tophat':
            return self.TopHatKernel(nfieldvars,nkernels)
        elif function=='exponential':
            return self.ExponentialKernel(nfieldvars,nkernels)
        elif function=='cosine':
            return self.CosineKernel(nfieldvars,nkernels)
        elif function=='mixture':
            return self.MixtureGaussianKernel(nfieldvars,nkernels)
        elif function=='bidirectional':
            return self.BidirectionalExponentialKernel(nfieldvars,nkernels)
        else:
            raise ValueError(f'Unknown function type `{function}`; must be `gaussian`, `tophat`, `exponential`, `cosine`, `mixture`, or `bidirectional`')

    def get_weights(self,dareapatch,dlevfull,dtimepatch,device,compute_components=False):
        '''
        Purpose: Obtain normalized parametric kernel weights using fixed grid quadrature.
        Args:
        - dareapatch (torch.Tensor): horizontal area weights patch with shape (plats, plons) or (nbatch, plats, plons)
        - dlevfull (torch.Tensor): full vertical thickness weights from fixed grid with shape (nlevs,)
        - dtimepatch (torch.Tensor): time step weights patch with shape (ptimes,) or (nbatch, ptimes)
        - device (str | torch.device): device to use
        - compute_components (bool): whether to compute component weights for mixture kernels (default: False)
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
                if self.perfield[dim]:
                    # Per-field kernels: concatenate results from each field's kernel
                    kernel1d_list = []
                    for field_idx, field_kernel in enumerate(self.functions[dim]):
                        kernel1d_field = field_kernel(kernel.shape[ax], device)  # Shape: (1, nkernels, length)
                        kernel1d_list.append(kernel1d_field)
                    kernel1d = torch.cat(kernel1d_list, dim=0)  # Shape: (nfieldvars, nkernels, length)
                else:
                    # Single kernel for all fields
                    kernel1d = self.functions[dim](kernel.shape[ax],device)
                # kernel1d has shape (nfieldvars, nkernels, length)
                # Need to reshape to (nfieldvars, nkernels, 1, 1, 1, 1) with length at position ax
                view = [kernel.shape[0], kernel.shape[1], 1, 1, 1, 1]
                view[ax] = kernel.shape[ax]
                kernel = kernel*kernel1d.view(*view)
        self.weights = KernelModule.normalize(kernel,dareapatch0,self.dlevfull,dtimepatch0,self.kerneldims)

        # Compute component weights for mixture kernels (for separate visualization)
        # Only compute when explicitly requested (e.g., during evaluation, not training)
        self.component_weights = None
        if not compute_components:
            return self.weights

        has_mixture = False
        for dim in self.kerneldims:
            if self.perfield[dim]:
                # Check if any field uses mixture kernel
                for field_kernel in self.functions[dim]:
                    if isinstance(field_kernel, self.MixtureGaussianKernel):
                        has_mixture = True
                        break
            else:
                if isinstance(self.functions[dim], self.MixtureGaussianKernel):
                    has_mixture = True
            if has_mixture:
                break

        if has_mixture:
            # Compute separate component kernels for mixture kernels
            # We'll store components as a list of 2 kernels (component1 and component2)
            kernel_c1 = torch.ones(self.nfieldvars,self.nkernels,plats,plons,plevs,ptimes,dtype=dareapatch0.dtype,device=device)
            kernel_c2 = torch.ones(self.nfieldvars,self.nkernels,plats,plons,plevs,ptimes,dtype=dareapatch0.dtype,device=device)

            for ax,dim in enumerate(('lat','lon','lev','time'),start=2):
                if dim in self.kerneldims:
                    if self.perfield[dim]:
                        # Per-field kernels
                        kernel1d_c1_list = []
                        kernel1d_c2_list = []
                        for field_idx, field_kernel in enumerate(self.functions[dim]):
                            if isinstance(field_kernel, self.MixtureGaussianKernel):
                                c1, c2 = field_kernel.get_components(kernel_c1.shape[ax], device)
                                kernel1d_c1_list.append(c1)
                                kernel1d_c2_list.append(c2)
                            else:
                                # For non-mixture kernels, both components are the same
                                kernel1d = field_kernel(kernel_c1.shape[ax], device)
                                kernel1d_c1_list.append(kernel1d)
                                kernel1d_c2_list.append(kernel1d)
                        kernel1d_c1 = torch.cat(kernel1d_c1_list, dim=0)
                        kernel1d_c2 = torch.cat(kernel1d_c2_list, dim=0)
                    else:
                        # Single kernel for all fields
                        if isinstance(self.functions[dim], self.MixtureGaussianKernel):
                            kernel1d_c1, kernel1d_c2 = self.functions[dim].get_components(kernel_c1.shape[ax], device)
                        else:
                            # For non-mixture kernels, both components are the same
                            kernel1d = self.functions[dim](kernel_c1.shape[ax], device)
                            kernel1d_c1 = kernel1d
                            kernel1d_c2 = kernel1d

                    view = [kernel_c1.shape[0], kernel_c1.shape[1], 1, 1, 1, 1]
                    view[ax] = kernel_c1.shape[ax]
                    kernel_c1 = kernel_c1 * kernel1d_c1.view(*view)
                    kernel_c2 = kernel_c2 * kernel1d_c2.view(*view)

            # Normalize component kernels
            weights_c1 = KernelModule.normalize(kernel_c1,dareapatch0,self.dlevfull,dtimepatch0,self.kerneldims)
            weights_c2 = KernelModule.normalize(kernel_c2,dareapatch0,self.dlevfull,dtimepatch0,self.kerneldims)
            # Stack as [2, nfieldvars, nkernels, ...]
            self.component_weights = torch.stack([weights_c1, weights_c2], dim=0)

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