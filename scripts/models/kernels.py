#!/usr/bin/env python

import torch
import numpy as np

class KernelModule:

    @staticmethod
    def normalize(kernel,dareapatch,dlevpatch,dtimepatch,kerneldims,epsilon=1e-6):
        '''
        Purpose: Normalize kernel so that sum(k * quadrature_weights) = 1 over kerneled dimensions.
        Args:
        - kernel (torch.Tensor): unnormalized kernel with shape (nfieldvars, plats, plons, plevs, ptimes)
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
        quad = torch.ones(1,plats,plons,plevs,ptimes,dtype=kernel.dtype,device=kernel.device)
        if ('lat' in kerneldims) or ('lon' in kerneldims):
            quad = quad*dareapatch[None,:,:,None,None]
        if 'lev' in kerneldims:
            quad = quad*dlevpatch[None,None,None,:,None]
        if 'time' in kerneldims:
            quad = quad*dtimepatch[None,None,None,None,:]
        kernelsum = (kernel*quad).sum(dim=(1,2,3,4))
        if 'lat' not in kerneldims:
            kernelsum = kernelsum/plats
        if 'lon' not in kerneldims:
            kernelsum = kernelsum/plons
        if 'lev' not in kerneldims:
            kernelsum = kernelsum/plevs
        if 'time' not in kerneldims:
            kernelsum = kernelsum/ptimes
        weights = kernel/(kernelsum[:,None,None,None,None]+epsilon)
        checksum = (weights*quad).sum(dim=(1,2,3,4))
        if 'lat' not in kerneldims:
            checksum = checksum/plats
        if 'lon' not in kerneldims:
            checksum = checksum/plons
        if 'lev' not in kerneldims:
            checksum = checksum/plevs
        if 'time' not in kerneldims:
            checksum = checksum/ptimes
        assert torch.allclose(checksum,torch.ones_like(checksum),atol=1e-2),f"Kernel normalization failed: weights sum to {checksum.mean().item():.6f} instead of 1.0"
        return weights

    @staticmethod
    def integrate(fieldpatch,weights,dareapatch,dlevpatch,dtimepatch,kerneldims):
        '''
        Purpose: Integrate predictor fields using normalized kernel weights with quadrature over kerneled dimensions.
        Args:
        - fieldpatch (torch.Tensor): predictor fields patch with shape (nbatch, nfieldvars, plats, plons, plevs, ptimes)
        - weights (torch.Tensor): normalized kernel weights with shape (nfieldvars, plats or 1, plons or 1, plevs or 1, ptimes or 1)
        - dareapatch (torch.Tensor): horizontal area weights with shape (nbatch, plats, plons)
        - dlevpatch (torch.Tensor): vertical thickness weights with shape (nbatch, plevs)
        - dtimepatch (torch.Tensor): time step weights with shape (nbatch, ptimes)
        - kerneldims (tuple[str] | list[str]): dimensions the kernel varies along
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars, ...) where ... are preserved non-kerneled dimensions
        '''
        weighted = fieldpatch*weights.unsqueeze(0)
        quad = 1.0
        if ('lat' in kerneldims) or ('lon' in kerneldims):
            quad = quad*dareapatch[:,None,:,:,None,None]
        if 'lev' in kerneldims:
            quad = quad*dlevpatch[:,None,None,None,:,None]
        if 'time' in kerneldims:
            quad = quad*dtimepatch[:,None,None,None,None,:]
        weighted = weighted*quad
        dimstosum = []
        if 'lat' in kerneldims:
            dimstosum.append(2)
        if 'lon' in kerneldims:
            dimstosum.append(3)
        if 'lev' in kerneldims:
            dimstosum.append(4)
        if 'time' in kerneldims:
            dimstosum.append(5)
        return torch.nansum(weighted,dim=dimstosum) if dimstosum else weighted


class NonparametricKernelLayer(torch.nn.Module):

    def __init__(self,nfieldvars,kerneldims,patchshape):
        '''
        Purpose: Initialize free-form (non-parametric) kernels along selected dimensions.
        Args:
        - nfieldvars (int): number of predictor fields
        - kerneldims (list[str] | tuple[str]): dimensions the kernel varies along
        - patchshape (tuple[int,int,int,int]): patch shape as (plats, plons, plevs, ptimes)
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
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
        kernel = torch.ones(self.nfieldvars,*kernelshape)
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
        - torch.Tensor: normalized kernel weights of shape (nfieldvars, plats_or_1, plons_or_1, plevs_or_1, ptimes_or_1)
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
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*preserved_dims)
        '''
        weights = self.get_weights(dareapatch,dlevfull,dtimepatch,fieldpatch.device)
        feats = KernelModule.integrate(fieldpatch,weights,dareapatch,dlevpatch,dtimepatch,self.kerneldims)
        self.features = feats
        return feats.flatten(1)

class ParametricKernelLayer(torch.nn.Module):

    class GaussianKernel(torch.nn.Module):

        def __init__(self,nfieldvars,dim):
            '''
            Purpose: Initialize Gaussian kernel parameters along one dimension.
            Args:
            - nfieldvars (int): number of predictor fields
            - dim (str): dimension name ('lat', 'lon', 'lev', 'time', or 'horizontal')
            Notes:
            - Implements k^(G)(s; μ, σ) = exp(-||s-μ||²/(2σ²))
            - Parameters μ (mean) and σ (std) are learned in normalized coordinate space [-1, 1]
            '''
            super().__init__()
            self.dim = dim
            self.mean = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            self.logstd = torch.nn.Parameter(torch.zeros(int(nfieldvars)))

        def forward(self,length,device):
            '''
            Purpose: Evaluate a Gaussian kernel along a coordinate in [-1,1].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: Gaussian kernel values with shape (nfieldvars, length)
            Notes:
            - Uses normalized coordinates s ∈ [-1, 1]
            - For vertical: -1 ≈ top of atmosphere, +1 ≈ surface
            '''
            coord = torch.linspace(-1.0,1.0,steps=length,device=device)
            std = torch.exp(self.logstd)
            kernel1d = torch.exp(-0.5*((coord[None,:]-self.mean[:,None])/std[:,None])**2)
            return kernel1d

    class TopHatKernel(torch.nn.Module):

        def __init__(self,nfieldvars,dim):
            '''
            Purpose: Initialize top-hat (uniform) kernel parameters along one dimension.
            Args:
            - nfieldvars (int): number of predictor fields
            - dim (str): dimension name ('lat', 'lon', 'lev', 'time', or 'horizontal')
            Notes:
            - Implements smooth approximation: k^(TH)(s; a, b) ≈ σ((s-a)/τ) * σ((b-s)/τ)
            - Uses sigmoid transitions instead of hard boundaries for differentiability
            - Parameters a and b are learned bounds in normalized coordinate space [-1, 1]
            - Approximates uniform weight within bounds, near-zero outside
            - Handles boundaries naturally (e.g., for vertical: a or b can be at -1 or +1)
            '''
            super().__init__()
            self.dim = dim
            self.lower = torch.nn.Parameter(torch.full((int(nfieldvars),),-0.5))
            self.upper = torch.nn.Parameter(torch.full((int(nfieldvars),),0.5))

        def forward(self,length,device):
            '''
            Purpose: Evaluate a top-hat kernel along a coordinate in [-1,1].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: top-hat kernel values with shape (nfieldvars, length)
            Notes:
            - Uses normalized coordinates s ∈ [-1, 1]
            - For vertical: -1 ≈ top of atmosphere, +1 ≈ surface
            - Implements a sharp boxcar function: constant weight inside layer, exactly zero outside
            - Width is constrained to prevent uniform distribution across all levels
            '''
            coord = torch.linspace(-1.0,1.0,steps=length,device=device)
            s1 = torch.min(self.lower,self.upper)
            s2 = torch.max(self.lower,self.upper)

            # Constrain width to be at most 75% of full range to prevent uniform kernels
            # This ensures the kernel must select specific layers, not all of them
            max_width = 1.5  # 75% of full range [-1, 1]
            width = s2 - s1
            # Soft constraint: if width exceeds max, compress it
            width_constrained = torch.where(width > max_width,
                                           max_width * torch.tanh(width / max_width),
                                           width)
            s2_constrained = s1 + width_constrained

            # Use very sharp sigmoids to approximate boxcar function
            # temperature controls sharpness - very small for near-boxcar behavior
            temperature = 0.01  # Very sharp transitions for boxcar-like behavior
            left_edge = torch.sigmoid((coord[None,:] - s1[:,None]) / temperature)
            right_edge = torch.sigmoid((s2_constrained[:,None] - coord[None,:]) / temperature)
            kernel1d = left_edge * right_edge

            # Apply strict threshold to make values exactly zero outside the layer
            # This creates a true boxcar: constant inside, exactly zero outside
            threshold = 0.5  # Values below this are set to exactly zero
            kernel1d = torch.where(kernel1d > threshold, kernel1d, torch.zeros_like(kernel1d))

            return kernel1d

    class ExponentialKernel(torch.nn.Module):

        def __init__(self,nfieldvars,dim):
            '''
            Purpose: Initialize exponential-decay kernel parameters with dimension-specific behavior.
            Args:
            - nfieldvars (int): number of predictor fields
            - dim (str): dimension name ('lat', 'lon', 'lev', 'time', or 'horizontal')
            Notes:
            - Implements k^(EXP)(s; τ₀) = exp(-ℓ(s)/τ₀) where ℓ(s) is distance from anchor
            - For 'horizontal': radial decay from center (2D circus tent pattern)
            - For 'lev': learned decay from top OR bottom via mixing parameter α
            - For 'time': decay backward from current timestep
            - For 'lat' or 'lon' individually: standard 1D exponential decay
            '''
            super().__init__()
            self.dim = dim
            self.logtau = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            if dim=='lev':
                self.logitalpha = torch.nn.Parameter(torch.zeros(int(nfieldvars)))

        def forward(self,length,device):
            '''
            Purpose: Evaluate exponential kernel based on dimension type.
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: exponential kernel values with shape (nfieldvars, length)
            Notes:
            - Time: distance ℓ(j) = (N-1)-j (decay from present into past)
            - Vertical: distance ℓ(j) = (1-α)·j + α·(N-1-j) where α∈(0,1) learned
            - Lat/lon: distance ℓ(s) from coordinate origin
            '''
            tau = torch.exp(self.logtau).clamp(min=1e-4,max=100.0)
            if self.dim=='time':
                distance = torch.arange(length-1,-1,-1,device=device,dtype=torch.float32)
                kernel1d = torch.exp(-distance[None,:]/tau[:,None])
            elif self.dim=='lev':
                alpha = torch.sigmoid(self.logitalpha)
                j = torch.arange(length,device=device,dtype=torch.float32)
                distance = (1.0-alpha[:,None])*j[None,:]+(alpha[:,None])*(length-1-j[None,:])
                kernel1d = torch.exp(-distance/tau[:,None])
            else:
                coord = torch.linspace(-1.0,1.0,steps=length,device=device)
                distance = coord.abs()
                kernel1d = torch.exp(-distance[None,:]/tau[:,None])
            return kernel1d

        def forward_horizontal(self,shape,device):
            '''
            Purpose: Evaluate 2D radial exponential kernel for horizontal dimensions.
            Args:
            - shape (tuple[int,int]): (plats, plons)
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: 2D exponential kernel with shape (nfieldvars, plats, plons)
            Notes:
            - Computes radial distance from center: ℓ(x) = ||x - x₀||
            - Creates circus tent decay pattern
            '''
            plats,plons = shape
            tau = torch.exp(self.logtau).clamp(min=1e-4,max=100.0)
            lats = torch.linspace(-1.0,1.0,steps=plats,device=device)
            lons = torch.linspace(-1.0,1.0,steps=plons,device=device)
            latgrid,longrid = torch.meshgrid(lats,lons,indexing='ij')
            distance = torch.sqrt(latgrid**2+longrid**2)
            kernel2d = torch.exp(-distance[None,:,:]/tau[:,None,None])
            return kernel2d

    class MixtureGaussianKernel(torch.nn.Module):

        def __init__(self,nfieldvars,dim):
            '''
            Purpose: Initialize mixture-of-Gaussians kernel parameters along one dimension.
            Args:
            - nfieldvars (int): number of predictor fields
            - dim (str): dimension name ('lat', 'lon', 'lev', 'time', or 'horizontal')
            Notes:
            - Implements k^(MG)(s; μ₁, σ₁, μ₂, σ₂, w₁, w₂) = w₁·N(μ₁,σ₁²) + w₂·N(μ₂,σ₂²)
            - Two Gaussians with learnable centers (μ₁, μ₂), widths (σ₁, σ₂), and independent weights (w₁, w₂)
            - Weights are unconstrained (can be positive or negative):
              * w₁ > 0, w₂ > 0: reinforcing contributions from different regions
              * w₁ > 0, w₂ < 0: canceling contributions (e.g., boundary layer positive, free-troposphere negative)
            - Useful for bimodal patterns or opposing contributions
            '''
            super().__init__()
            self.dim = dim
            self.center1 = torch.nn.Parameter(torch.full((int(nfieldvars),),-0.5))
            self.center2 = torch.nn.Parameter(torch.full((int(nfieldvars),),0.5))
            self.logwidth1 = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            self.logwidth2 = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            self.weight1 = torch.nn.Parameter(torch.ones(int(nfieldvars)))
            self.weight2 = torch.nn.Parameter(torch.ones(int(nfieldvars)))

        def get_components(self,length,device):
            '''
            Purpose: Compute individual Gaussian components separately (for visualization).
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - tuple: (component1, component2) each with shape (nfieldvars, length)
            Notes:
            - Returns weighted Gaussian components before they are combined
            - Useful for visualizing each component separately in plots
            '''
            coord = torch.linspace(-1.0, 1.0, steps=length, device=device)
            width1 = torch.exp(self.logwidth1).clamp(min=0.1, max=2.0)
            width2 = torch.exp(self.logwidth2).clamp(min=0.1, max=2.0)

            # Compute two Gaussian components
            dist1 = coord[None,:] - self.center1[:,None]
            dist2 = coord[None,:] - self.center2[:,None]
            gauss1 = torch.exp(-dist1**2 / (2 * width1[:,None]**2))
            gauss2 = torch.exp(-dist2**2 / (2 * width2[:,None]**2))

            # Return weighted components separately
            component1 = self.weight1[:,None] * gauss1
            component2 = self.weight2[:,None] * gauss2

            return component1, component2

        def forward(self,length,device):
            '''
            Purpose: Evaluate a mixture-of-Gaussians kernel along a coordinate in [-1,1].
            Args:
            - length (int): number of points along the axis
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: mixture kernel values with shape (nfieldvars, length)
            Notes:
            - Uses normalized coordinates s ∈ [-1, 1]
            - Combines two Gaussian bumps with independent learnable weights
            - Allows positive/positive, positive/negative, or any combination
            - Examples: two peaks, center-surround, or single peak with negative surround
            '''
            component1, component2 = self.get_components(length, device)

            # Combine components
            kernel1d = component1 + component2

            # Add small epsilon to avoid all-zero kernels
            kernel1d = kernel1d + 1e-8
            return kernel1d

    def __init__(self,nfieldvars,kerneldict):
        '''
        Purpose: Initialize parametric kernels along selected dimensions.
        Args:
        - nfieldvars (int): number of predictor fields
        - kerneldict (dict[str,str|list[str]]): mapping of dimensions to kernel type(s)
          Supports two formats:
          1. Single kernel for all fields: {"lev": "gaussian"}
          2. Per-field kernels: {"lev": ["exponential", "gaussian", "mixgaussian"]}
          Valid kernel types: 'gaussian', 'tophat', 'exponential', 'mixgaussian'
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.kerneldict = dict(kerneldict)
        self.kerneldims = tuple(kerneldict.keys())
        self.weights = None
        self.component_weights = None
        self.features = None
        self.dlevfull = None
        self.functions = torch.nn.ModuleDict()
        self.perfield = {}
        has_horizontal_exponential = ('lat' in kerneldict and 'lon' in kerneldict and
            ((isinstance(kerneldict['lat'],str) and kerneldict['lat']=='exponential') or
             (isinstance(kerneldict['lat'],list) and 'exponential' in kerneldict['lat'])) and
            ((isinstance(kerneldict['lon'],str) and kerneldict['lon']=='exponential') or
             (isinstance(kerneldict['lon'],list) and 'exponential' in kerneldict['lon'])))
        for dim,function_spec in self.kerneldict.items():
            if has_horizontal_exponential and dim in ('lat','lon'):
                if dim=='lat':
                    if isinstance(function_spec,list):
                        raise ValueError('Per-field kernels not supported for horizontal exponential (lat+lon must use same exponential)')
                    self.perfield['horizontal'] = False
                    self.functions['horizontal'] = self._create_kernel('exponential',self.nfieldvars,'horizontal')
                continue
            if isinstance(function_spec,list):
                if len(function_spec)!=self.nfieldvars:
                    raise ValueError(f'Per-field kernel list for dim `{dim}` must have length {self.nfieldvars}, got {len(function_spec)}')
                self.perfield[dim] = True
                self.functions[dim] = torch.nn.ModuleList([
                    self._create_kernel(func,1,dim) for func in function_spec
                ])
            else:
                self.perfield[dim] = False
                self.functions[dim] = self._create_kernel(function_spec,self.nfieldvars,dim)

    def _create_kernel(self,function,nfieldvars,dim):
        '''
        Purpose: Factory method to create a kernel instance from a function name.
        Args:
        - function (str): kernel type name
        - nfieldvars (int): number of field variables for this kernel
        - dim (str): dimension name ('lat', 'lon', 'lev', 'time', or 'horizontal')
        Returns:
        - torch.nn.Module: kernel instance
        '''
        if function=='gaussian':
            return self.GaussianKernel(nfieldvars,dim)
        elif function=='tophat':
            return self.TopHatKernel(nfieldvars,dim)
        elif function=='exponential':
            return self.ExponentialKernel(nfieldvars,dim)
        elif function=='mixgaussian':
            return self.MixtureGaussianKernel(nfieldvars,dim)
        else:
            raise ValueError(f'Unknown function type `{function}`; must be `gaussian`, `tophat`, `exponential`, or `mixgaussian`')

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
        - torch.Tensor: normalized kernel weights of shape (nfieldvars, plats, plons, plevs, ptimes)
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
        plevs = self.dlevfull.numel()
        ptimes = dtimepatch0.numel()
        kernel = torch.ones(self.nfieldvars,plats,plons,plevs,ptimes,dtype=dareapatch0.dtype,device=device)
        has_horizontal = 'horizontal' in self.functions
        if has_horizontal:
            kernel2d = self.functions['horizontal'].forward_horizontal((plats,plons),device)
            kernel = kernel*kernel2d.view(self.nfieldvars,plats,plons,1,1)
        for ax,dim in enumerate(('lat','lon','lev','time'),start=1):
            if has_horizontal and dim in ('lat','lon'):
                continue
            if dim in self.kerneldims:
                if self.perfield.get(dim,False):
                    kernel1d_list = []
                    for field_idx,field_kernel in enumerate(self.functions[dim]):
                        kernel1d_field = field_kernel(kernel.shape[ax],device)
                        kernel1d_list.append(kernel1d_field)
                    kernel1d = torch.cat(kernel1d_list,dim=0)
                else:
                    kernel1d = self.functions[dim](kernel.shape[ax],device)
                view = [kernel.shape[0],1,1,1,1]
                view[ax] = kernel.shape[ax]
                kernel = kernel*kernel1d.view(*view)
        normkerneldims = list(self.kerneldims)
        if has_horizontal and 'lat' not in normkerneldims:
            normkerneldims.extend(['lat','lon'])
        self.weights = KernelModule.normalize(kernel,dareapatch0,self.dlevfull,dtimepatch0,normkerneldims)

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
            kernel_c1 = torch.ones(self.nfieldvars,plats,plons,plevs,ptimes,dtype=dareapatch0.dtype,device=device)
            kernel_c2 = torch.ones(self.nfieldvars,plats,plons,plevs,ptimes,dtype=dareapatch0.dtype,device=device)
            if has_horizontal:
                kernel2d = self.functions['horizontal'].forward_horizontal((plats,plons),device)
                kernel_c1 = kernel_c1*kernel2d.view(self.nfieldvars,plats,plons,1,1)
                kernel_c2 = kernel_c2*kernel2d.view(self.nfieldvars,plats,plons,1,1)
            for ax,dim in enumerate(('lat','lon','lev','time'),start=1):
                if has_horizontal and dim in ('lat','lon'):
                    continue
                if dim in self.kerneldims:
                    if self.perfield.get(dim,False):
                        kernel1d_c1_list = []
                        kernel1d_c2_list = []
                        for field_idx,field_kernel in enumerate(self.functions[dim]):
                            if isinstance(field_kernel,self.MixtureGaussianKernel):
                                c1,c2 = field_kernel.get_components(kernel_c1.shape[ax],device)
                                kernel1d_c1_list.append(c1)
                                kernel1d_c2_list.append(c2)
                            else:
                                kernel1d = field_kernel(kernel_c1.shape[ax],device)
                                kernel1d_c1_list.append(kernel1d)
                                kernel1d_c2_list.append(kernel1d)
                        kernel1d_c1 = torch.cat(kernel1d_c1_list,dim=0)
                        kernel1d_c2 = torch.cat(kernel1d_c2_list,dim=0)
                    else:
                        if isinstance(self.functions[dim],self.MixtureGaussianKernel):
                            kernel1d_c1,kernel1d_c2 = self.functions[dim].get_components(kernel_c1.shape[ax],device)
                        else:
                            kernel1d = self.functions[dim](kernel_c1.shape[ax],device)
                            kernel1d_c1 = kernel1d
                            kernel1d_c2 = kernel1d
                    view = [kernel_c1.shape[0],1,1,1,1]
                    view[ax] = kernel_c1.shape[ax]
                    kernel_c1 = kernel_c1*kernel1d_c1.view(*view)
                    kernel_c2 = kernel_c2*kernel1d_c2.view(*view)
            weights_c1 = KernelModule.normalize(kernel_c1,dareapatch0,self.dlevfull,dtimepatch0,normkerneldims)
            weights_c2 = KernelModule.normalize(kernel_c2,dareapatch0,self.dlevfull,dtimepatch0,normkerneldims)
            self.component_weights = torch.stack([weights_c1,weights_c2],dim=0)

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
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars*preserved_dims)
        '''
        weights = self.get_weights(dareapatch,dlevfull,dtimepatch,fieldpatch.device)
        feats = KernelModule.integrate(fieldpatch,weights,dareapatch,dlevpatch,dtimepatch,self.kerneldims)
        self.features = feats
        return feats.flatten(1)