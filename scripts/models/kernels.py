#!/usr/bin/env python

import torch
import numpy as np


class KernelModule:

    @staticmethod
    def normalize(kernel, dareapatch, dlevpatch, dtimepatch, kerneldims, epsilon=1e-6):
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
        plats, plons = dareapatch.shape
        plevs = dlevpatch.numel()
        ptimes = dtimepatch.numel()

        quad = torch.ones(1, 1, plats, plons, plevs, ptimes, dtype=kernel.dtype, device=kernel.device)
        if ('lat' in kerneldims) or ('lon' in kerneldims):
            quad = quad * dareapatch[None, None, :, :, None, None]
        if 'lev' in kerneldims:
            quad = quad * dlevpatch[None, None, None, None, :, None]
        if 'time' in kerneldims:
            quad = quad * dtimepatch[None, None, None, None, None, :]

        kernelsum = (kernel * quad).sum(dim=(2, 3, 4, 5))
        if 'lat' not in kerneldims:
            kernelsum = kernelsum / plats
        if 'lon' not in kerneldims:
            kernelsum = kernelsum / plons
        if 'lev' not in kerneldims:
            kernelsum = kernelsum / plevs
        if 'time' not in kerneldims:
            kernelsum = kernelsum / ptimes

        weights = kernel / (kernelsum[:, :, None, None, None, None] + epsilon)

        checksum = (weights * quad).sum(dim=(2, 3, 4, 5))
        if 'lat' not in kerneldims:
            checksum = checksum / plats
        if 'lon' not in kerneldims:
            checksum = checksum / plons
        if 'lev' not in kerneldims:
            checksum = checksum / plevs
        if 'time' not in kerneldims:
            checksum = checksum / ptimes

        assert torch.allclose(checksum, torch.ones_like(checksum), atol=1e-4), \
            f"Kernel normalization failed: weights sum to {checksum.mean().item():.6f} instead of 1.0"
        return weights

    @staticmethod
    def integrate(fieldpatch, weights, dareapatch, dlevpatch, dtimepatch, kerneldims):
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
        weighted = fieldpatch.unsqueeze(2) * weights.unsqueeze(0)
        quad = 1.0
        if ('lat' in kerneldims) or ('lon' in kerneldims):
            quad = quad * dareapatch[:, None, None, :, :, None, None]
        if 'lev' in kerneldims:
            quad = quad * dlevpatch[:, None, None, None, None, :, None]
        if 'time' in kerneldims:
            quad = quad * dtimepatch[:, None, None, None, None, None, :]
        weighted = weighted * quad

        dimstosum = []
        if 'lat' in kerneldims:
            dimstosum.append(3)
        if 'lon' in kerneldims:
            dimstosum.append(4)
        if 'lev' in kerneldims:
            dimstosum.append(5)
        if 'time' in kerneldims:
            dimstosum.append(6)

        return torch.nansum(weighted, dim=dimstosum) if dimstosum else weighted


class NonparametricKernelLayer(torch.nn.Module):

    def __init__(self, nfieldvars, nkernels, kerneldims, patchshape):
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

        plats, plons, plevs, ptimes = patchshape
        kernelshape = [
            plats  if 'lat' in self.kerneldims else 1,
            plons  if 'lon' in self.kerneldims else 1,
            plevs  if 'lev' in self.kerneldims else 1,
            ptimes if 'time' in self.kerneldims else 1
        ]
        kernel = torch.ones(self.nfieldvars, self.nkernels, *kernelshape)
        kernel = kernel + torch.randn_like(kernel) * 0.2
        self.kernel = torch.nn.Parameter(kernel)

    def get_weights(self, dareapatch, dlevfull, dtimepatch, device):
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

        dareapatch0 = dareapatch[0] if dareapatch.dim() == 3 else dareapatch
        dtimepatch0 = dtimepatch[0] if dtimepatch.dim() == 2 else dtimepatch

        self.weights = KernelModule.normalize(self.kernel, dareapatch0, self.dlevfull, dtimepatch0, self.kerneldims, epsilon=1e-6)
        return self.weights

    def forward(self, fieldpatch, dareapatch, dlevpatch, dtimepatch, dlevfull):
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
        weights = self.get_weights(dareapatch, dlevfull, dtimepatch, fieldpatch.device)
        feats = KernelModule.integrate(fieldpatch, weights, dareapatch, dlevpatch, dtimepatch, self.kerneldims)
        self.features = feats
        return feats.flatten(1)


class ParametricKernelLayer(torch.nn.Module):

    # -------------------------
    #  1D kernel families
    # -------------------------

    class GaussianKernel(torch.nn.Module):

        def __init__(self, nfieldvars, nkernels):
            '''
            Purpose: Initialize Gaussian kernel parameters along one dimension.
            Notes:
            - k^(G)(s) = exp(-(s-μ)^2/(2σ^2))
            - Parameters μ and σ are learned in normalized coordinate space [-1, 1]
            '''
            super().__init__()
            self.mean   = torch.nn.Parameter(torch.zeros(int(nfieldvars), int(nkernels)))
            self.logstd = torch.nn.Parameter(torch.zeros(int(nfieldvars), int(nkernels)))

        def forward(self, length, device):
            coord = torch.linspace(-1.0, 1.0, steps=length, device=device)
            std = torch.exp(self.logstd) + 1e-6
            kernel1d = torch.exp(-0.5 * ((coord[None, None, :] - self.mean[..., None]) / std[..., None]) ** 2)
            return kernel1d

    class MixtureGaussianKernel(torch.nn.Module):

        def __init__(self, nfieldvars, nkernels):
            '''
            Purpose: Initialize two-component mixture-of-Gaussians kernel parameters (1D).
            Notes:
            - k^(MG)(s) = w1 * G(s; μ1, σ1) + w2 * G(s; μ2, σ2)
            - w1, w2 are unconstrained and may be positive or negative
            '''
            super().__init__()
            self.center1   = torch.nn.Parameter(torch.full((int(nfieldvars), int(nkernels)), -0.5))
            self.center2   = torch.nn.Parameter(torch.full((int(nfieldvars), int(nkernels)),  0.5))
            self.logwidth1 = torch.nn.Parameter(torch.zeros(int(nfieldvars), int(nkernels)))
            self.logwidth2 = torch.nn.Parameter(torch.zeros(int(nfieldvars), int(nkernels)))
            self.weight1   = torch.nn.Parameter(torch.ones(int(nfieldvars), int(nkernels)))
            self.weight2   = torch.nn.Parameter(torch.ones(int(nfieldvars), int(nkernels)))

        def get_components(self, length, device):
            coord = torch.linspace(-1.0, 1.0, steps=length, device=device)
            width1 = torch.exp(self.logwidth1) + 1e-6
            width2 = torch.exp(self.logwidth2) + 1e-6

            dist1 = coord[None, None, :] - self.center1[..., None]
            dist2 = coord[None, None, :] - self.center2[..., None]
            gauss1 = torch.exp(-0.5 * (dist1 / width1[..., None]) ** 2)
            gauss2 = torch.exp(-0.5 * (dist2 / width2[..., None]) ** 2)

            component1 = self.weight1[..., None] * gauss1
            component2 = self.weight2[..., None] * gauss2
            return component1, component2

        def forward(self, length, device):
            c1, c2 = self.get_components(length, device)
            kernel1d = c1 + c2
            kernel1d = kernel1d + 1e-8
            return kernel1d

    class TopHatKernel(torch.nn.Module):

        def __init__(self, nfieldvars, nkernels, sharpness=50.0):
            '''
            Purpose: Initialize top-hat kernel parameters (1D).
            Notes:
            - Ideal: I(s in [min(a,b), max(a,b)]).
            - We implement a sharp but differentiable approximation using sigmoids.
            '''
            super().__init__()
            self.a = torch.nn.Parameter(torch.full((int(nfieldvars), int(nkernels)), -0.25))
            self.b = torch.nn.Parameter(torch.full((int(nfieldvars), int(nkernels)),  0.25))
            self.sharpness = float(sharpness)

        def forward(self, length, device):
            coord = torch.linspace(-1.0, 1.0, steps=length, device=device)
            lo = torch.minimum(self.a, self.b)
            hi = torch.maximum(self.a, self.b)
            # Smooth indicator: sigmoid(k*(s-lo)) * sigmoid(k*(hi-s))
            k = self.sharpness
            left  = torch.sigmoid(k * (coord[None, None, :] - lo[..., None]))
            right = torch.sigmoid(k * (hi[..., None] - coord[None, None, :]))
            kernel1d = left * right
            kernel1d = kernel1d + 1e-8
            return kernel1d

    class BoundaryExponentialKernel(torch.nn.Module):

        def __init__(self, nfieldvars, nkernels):
            '''
            Purpose: Initialize exponential-decay kernel parameters (1D) that decay away from a boundary.
            Notes:
            - k^(EXP)(s) = exp(-ell(s)/tau0)
            - tau0 > 0 is learned
            - For pressure (lev): boundary anchor is learned via alpha=sigmoid(d) mixing top vs bottom:
                ell(j) = (1-alpha)*j + alpha*(N-1-j)
              so decay is always from *one of the two boundaries* (model chooses which).
            '''
            super().__init__()
            self.logtau = torch.nn.Parameter(torch.zeros(int(nfieldvars), int(nkernels)))
            self.direction = torch.nn.Parameter(torch.zeros(int(nfieldvars), int(nkernels)))

        def forward(self, length, device):
            coord = torch.arange(length, device=device, dtype=torch.float32)
            tau0 = torch.exp(self.logtau) + 1e-4
            alpha = torch.sigmoid(self.direction)  # alpha ~0 => from top; alpha ~1 => from bottom
            ell = (1.0 - alpha[..., None]) * coord[None, None, :] + alpha[..., None] * ((length - 1) - coord[None, None, :])
            kernel1d = torch.exp(-ell / tau0[..., None])
            return kernel1d

    class TimeExponentialKernel(torch.nn.Module):

        def __init__(self, nfieldvars, nkernels):
            '''
            Purpose: Initialize exponential-decay kernel parameters for time (1D).
            Notes:
            - Anchor fixed to prediction time (end of patch).
            - ell(j) = (N-1) - j so influence decays backward into the past.
            '''
            super().__init__()
            self.logtau = torch.nn.Parameter(torch.zeros(int(nfieldvars), int(nkernels)))

        def forward(self, length, device):
            coord = torch.arange(length, device=device, dtype=torch.float32)
            tau0 = torch.exp(self.logtau) + 1e-4
            ell = (length - 1) - coord[None, None, :]
            kernel1d = torch.exp(-ell / tau0[..., None])
            return kernel1d

    # -------------------------
    #  2D horizontal exponential (radial) kernel family
    # -------------------------

    class HorizontalExponentialKernel(torch.nn.Module):

        def __init__(self, nfieldvars, nkernels):
            '''
            Purpose: Initialize exponential-decay kernel parameters on the horizontal (2D).
            Notes:
            - Anchor fixed to prediction location x0 (assumed to be patch center).
            - ell(x_n) = ||x_n - x0|| using Euclidean distance on the patch index grid.
            - k^(EXP)(x) = exp(-ell(x)/tau0)
            '''
            super().__init__()
            self.logtau = torch.nn.Parameter(torch.zeros(int(nfieldvars), int(nkernels)))

        def forward(self, plats, plons, device):
            tau0 = torch.exp(self.logtau) + 1e-4

            # Anchor at patch center (prediction location).
            cy = (plats - 1) / 2.0
            cx = (plons - 1) / 2.0
            yy = torch.arange(plats, device=device, dtype=torch.float32)[:, None]
            xx = torch.arange(plons, device=device, dtype=torch.float32)[None, :]
            dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)  # (plats, plons)

            kernel2d = torch.exp(-dist[None, None, :, :] / tau0[..., None, None])
            return kernel2d

    # -------------------------
    #  Layer plumbing
    # -------------------------

    def __init__(self, nfieldvars, nkernels, kerneldict):
        '''
        Purpose: Initialize parametric kernels along selected dimensions.

        Args:
        - nfieldvars (int): number of predictor fields
        - nkernels (int): number of kernels to learn per predictor field
        - kerneldict (dict[str,str|list[str]]): mapping of dimensions to kernel type(s)

          Supported dimensions:
          - "lev", "time"  (1D)
          - "horiz"        (2D horizontal, uses both lat+lon together)

          Supported kernel types (matching Appendix):
          - "gaussian"
          - "mixture"      (mixture-of-Gaussians)
          - "tophat"
          - "exponential"  (boundary-decay for lev; backward-in-time for time; radial for horiz)

          Two formats:
          1) Single kernel for all fields: {"lev": "gaussian"}
          2) Per-field kernels: {"lev": ["exponential", "gaussian", ...]}  (length must equal nfieldvars)

        Notes:
        - For horizontal exponential-decay as written in the appendix, use dim="horiz".
          (A radial 2D kernel is not representable as a product of separate lat/lon 1D kernels.)
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.nkernels   = int(nkernels)
        self.kerneldict = dict(kerneldict)

        # Internal: expand dims for normalization/integration.
        # - "horiz" means kernel varies along BOTH lat and lon.
        self.kerneldims = []
        for dim in self.kerneldict.keys():
            if dim == 'horiz':
                self.kerneldims.extend(['lat', 'lon'])
            else:
                self.kerneldims.append(dim)
        self.kerneldims = tuple(self.kerneldims)

        self.weights    = None
        self.component_weights = None
        self.features   = None
        self.dlevfull   = None

        self.functions  = torch.nn.ModuleDict()
        self.perfield   = {}

        for dim, function_spec in self.kerneldict.items():
            if isinstance(function_spec, list):
                if len(function_spec) != self.nfieldvars:
                    raise ValueError(f'Per-field kernel list for dim `{dim}` must have length {self.nfieldvars}, got {len(function_spec)}')
                self.perfield[dim] = True
                self.functions[dim] = torch.nn.ModuleList([
                    self._create_kernel(dim, func, 1, self.nkernels) for func in function_spec
                ])
            else:
                self.perfield[dim] = False
                self.functions[dim] = self._create_kernel(dim, function_spec, self.nfieldvars, self.nkernels)

    def _create_kernel(self, dim, function, nfieldvars, nkernels):
        '''
        Purpose: Factory method to create a kernel instance from a (dim, function) spec.
        '''
        if function == 'gaussian':
            return self.GaussianKernel(nfieldvars, nkernels)
        if function == 'mixture':
            return self.MixtureGaussianKernel(nfieldvars, nkernels)
        if function == 'tophat':
            return self.TopHatKernel(nfieldvars, nkernels)
        if function == 'exponential':
            if dim == 'time':
                return self.TimeExponentialKernel(nfieldvars, nkernels)
            if dim == 'lev':
                return self.BoundaryExponentialKernel(nfieldvars, nkernels)
            if dim == 'horiz':
                return self.HorizontalExponentialKernel(nfieldvars, nkernels)
            raise ValueError(f'`exponential` kernel only supported for dim in {{"lev","time","horiz"}}, got `{dim}`')
        raise ValueError(f'Unknown function type `{function}`; must be `gaussian`, `mixture`, `tophat`, or `exponential`')

    def _eval_kernel_1d(self, dim, field_kernel, length, device):
        # One place to evaluate 1D kernels (keeps get_weights cleaner).
        return field_kernel(length, device)

    def get_weights(self, dareapatch, dlevfull, dtimepatch, device, compute_components=False):
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

        dareapatch0 = dareapatch[0] if dareapatch.dim() == 3 else dareapatch
        dtimepatch0 = dtimepatch[0] if dtimepatch.dim() == 2 else dtimepatch

        plats, plons = dareapatch0.shape
        plevs        = self.dlevfull.numel()
        ptimes       = dtimepatch0.numel()

        kernel = torch.ones(self.nfieldvars, self.nkernels, plats, plons, plevs, ptimes,
                            dtype=dareapatch0.dtype, device=device)

        # Horizontal (2D) kernel: dim == "horiz"
        if 'horiz' in self.kerneldict:
            if self.perfield['horiz']:
                k2_list = []
                for field_kernel in self.functions['horiz']:
                    k2_list.append(field_kernel(plats, plons, device))  # (1, nkernels, plats, plons)
                kernel2d = torch.cat(k2_list, dim=0)  # (nfieldvars, nkernels, plats, plons)
            else:
                kernel2d = self.functions['horiz'](plats, plons, device)
            kernel = kernel * kernel2d[:, :, :, :, None, None]

        # Separable 1D kernels for lev/time (and optionally lat/lon if you ever add them)
        for ax, dim in enumerate(('lat', 'lon', 'lev', 'time'), start=2):
            if dim == 'lat' or dim == 'lon':
                # lat/lon are only included as explicit dims when you *do not* use "horiz".
                # If you want horizontal localization, use "horiz" so the kernel can be truly 2D (radial distance).
                if dim in self.kerneldict:
                    raise ValueError(f'Use dim="horiz" for horizontal kernels; `{dim}` kernels are not supported here.')
                continue

            if dim in self.kerneldict:
                spec = self.kerneldict[dim]
                if self.perfield[dim]:
                    kernel1d_list = []
                    for field_kernel in self.functions[dim]:
                        kernel1d_field = self._eval_kernel_1d(dim, field_kernel, kernel.shape[ax], device)
                        kernel1d_list.append(kernel1d_field)
                    kernel1d = torch.cat(kernel1d_list, dim=0)
                else:
                    kernel1d = self._eval_kernel_1d(dim, self.functions[dim], kernel.shape[ax], device)

                view = [kernel.shape[0], kernel.shape[1], 1, 1, 1, 1]
                view[ax] = kernel.shape[ax]
                kernel = kernel * kernel1d.view(*view)

        self.weights = KernelModule.normalize(kernel, dareapatch0, self.dlevfull, dtimepatch0, self.kerneldims)

        # Optional: compute separate component weights for mixture kernels (for plotting)
        self.component_weights = None
        if not compute_components:
            return self.weights

        has_mixture = False
        for dim in self.kerneldict.keys():
            if self.perfield[dim]:
                for field_kernel in self.functions[dim]:
                    if isinstance(field_kernel, self.MixtureGaussianKernel):
                        has_mixture = True
                        break
            else:
                if isinstance(self.functions[dim], self.MixtureGaussianKernel):
                    has_mixture = True
            if has_mixture:
                break

        if not has_mixture:
            return self.weights

        kernel_c1 = torch.ones_like(kernel)
        kernel_c2 = torch.ones_like(kernel)

        # horiz components: only mixture-of-Gaussians is 1D, so horiz won't contribute components
        if 'horiz' in self.kerneldict:
            # horiz kernel is 2D exponential in this appendix set; no mixture components for horiz.
            if self.perfield['horiz']:
                k2_list = []
                for field_kernel in self.functions['horiz']:
                    k2_list.append(field_kernel(plats, plons, device))
                kernel2d = torch.cat(k2_list, dim=0)
            else:
                kernel2d = self.functions['horiz'](plats, plons, device)
            kernel_c1 = kernel_c1 * kernel2d[:, :, :, :, None, None]
            kernel_c2 = kernel_c2 * kernel2d[:, :, :, :, None, None]

        # lev/time components
        for ax, dim in enumerate(('lat', 'lon', 'lev', 'time'), start=2):
            if dim == 'lat' or dim == 'lon':
                continue
            if dim not in self.kerneldict:
                continue

            if self.perfield[dim]:
                k1_list, k2_list = [], []
                for field_kernel in self.functions[dim]:
                    if isinstance(field_kernel, self.MixtureGaussianKernel):
                        c1, c2 = field_kernel.get_components(kernel.shape[ax], device)
                        k1_list.append(c1)
                        k2_list.append(c2)
                    else:
                        k = self._eval_kernel_1d(dim, field_kernel, kernel.shape[ax], device)
                        k1_list.append(k)
                        k2_list.append(k)
                k1 = torch.cat(k1_list, dim=0)
                k2 = torch.cat(k2_list, dim=0)
            else:
                fk = self.functions[dim]
                if isinstance(fk, self.MixtureGaussianKernel):
                    k1, k2 = fk.get_components(kernel.shape[ax], device)
                else:
                    k = self._eval_kernel_1d(dim, fk, kernel.shape[ax], device)
                    k1, k2 = k, k

            view = [kernel.shape[0], kernel.shape[1], 1, 1, 1, 1]
            view[ax] = kernel.shape[ax]
            kernel_c1 = kernel_c1 * k1.view(*view)
            kernel_c2 = kernel_c2 * k2.view(*view)

        weights_c1 = KernelModule.normalize(kernel_c1, dareapatch0, self.dlevfull, dtimepatch0, self.kerneldims)
        weights_c2 = KernelModule.normalize(kernel_c2, dareapatch0, self.dlevfull, dtimepatch0, self.kerneldims)
        self.component_weights = torch.stack([weights_c1, weights_c2], dim=0)

        return self.weights

    def forward(self, fieldpatch, dareapatch, dlevpatch, dtimepatch, dlevfull):
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
        weights = self.get_weights(dareapatch, dlevfull, dtimepatch, fieldpatch.device)
        feats = KernelModule.integrate(fieldpatch, weights, dareapatch, dlevpatch, dtimepatch, self.kerneldims)
        self.features = feats
        return feats.flatten(1)
