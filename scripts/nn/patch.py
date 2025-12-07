#!/usr/bin/env python

import numpy as np
import xarray as xr
from typing import Union, Tuple, List, Optional


class PatchExtractor:
    """
    Flexible patch extraction for neural network input.

    Extracts spatial-temporal-vertical neighborhoods around target points (x₀,t₀).
    Supports flexible dimensionality - can extract patches over any combination of
    spatial (lat, lon), temporal, and vertical (pressure levels) dimensions.

    Example use cases:
        - Nonlocal inputs: Extract 3D neighborhoods (x, p, t) for RH, θₑ, θₑ*
        - Local inputs: Extract only at (x₀,t₀) for land fraction, heat fluxes
        - Vertical-only: Extract full column (p) at (x₀,t₀) for kernel layers
        - Spatial-only: Extract 2D neighborhood (x) at t₀ for surface variables
    """

    def __init__(self,
                 spatial_window: Tuple[int, int] = (0, 0),
                 temporal_window: int = 0,
                 vertical: Union[str, List[int], np.ndarray] = 'all',
                 boundary: str = 'valid'):
        """
        Initialize patch extractor with desired neighborhood configuration.

        Args:
            spatial_window: (lat_radius, lon_radius) defining spatial neighborhood size
                - (0, 0): local only (single grid cell at x₀)
                - (1, 1): 3×3 neighborhood (1 cell in each direction)
                - (2, 2): 5×5 neighborhood (2 cells in each direction)
            temporal_window: number of past timesteps to include
                - 0: current time only (t₀)
                - 1: current + 1 past timestep (t₀, t₋₁)
                - 2: current + 2 past timesteps (t₀, t₋₁, t₋₂)
            vertical: which pressure levels to include
                - 'all': include all pressure levels
                - 'none': exclude vertical dimension (for 3D variables)
                - List/array of indices: subset of specific levels
            boundary: how to handle spatial/temporal boundaries
                - 'valid': only extract patches where full neighborhood fits (reduces nsamples)
                - 'reflect': reflect/mirror at boundaries (preserves nsamples)
                - 'constant': pad with zeros at boundaries (preserves nsamples)
        """
        self.lat_radius = spatial_window[0]
        self.lon_radius = spatial_window[1]
        self.temporal_window = temporal_window
        self.vertical = vertical
        self.boundary = boundary

        # Calculate spatial patch dimensions
        self.lat_size = 2 * self.lat_radius + 1  # e.g., (1,1) → 3×3
        self.lon_size = 2 * self.lon_radius + 1

    def extract(self, da: xr.DataArray, local_only: bool = False) -> np.ndarray:
        """
        Extract patches from a DataArray for all valid target points.

        Args:
            da: xarray.DataArray with dims (time, lat, lon) or (time, lat, lon, lev)
            local_only: if True, override settings and extract only values at (x₀,t₀)
                       Useful for local-only variables (land fraction, heat fluxes)

        Returns:
            np.ndarray of shape (nsamples, nfeatures) where:
                - nsamples: number of valid target points (depends on boundary mode)
                - nfeatures: total number of features per patch

        Example:
            >>> # Extract 3×3 spatial, 2 past timesteps, all 16 pressure levels
            >>> extractor = PatchExtractor(spatial_window=(1,1), temporal_window=2, vertical='all')
            >>> # For 4D variable (time, lat, lon, lev): nfeatures = 3×3×3×16 = 432
            >>> patches = extractor.extract(rh_data)
            >>> patches.shape  # (nsamples, 432)
        """
        # Handle local-only mode
        if local_only:
            return self._extract_local(da)

        # Check if vertical dimension exists
        has_vertical = 'lev' in da.dims

        # Handle vertical dimension selection
        if has_vertical:
            da = self._select_vertical_levels(da)

        # Sort dimensions for consistency
        if has_vertical:
            da = da.sortby('lev', ascending=False)

        # Extract patches based on boundary mode
        if self.boundary == 'valid':
            patches = self._extract_valid(da)
        elif self.boundary == 'reflect':
            patches = self._extract_padded(da, mode='reflect')
        elif self.boundary == 'constant':
            patches = self._extract_padded(da, mode='constant')
        else:
            raise ValueError(f"Unknown boundary mode '{self.boundary}'. Expected 'valid', 'reflect', or 'constant'.")

        return patches

    def _select_vertical_levels(self, da: xr.DataArray) -> xr.DataArray:
        """Select specified vertical levels from 4D DataArray."""
        if self.vertical == 'all':
            return da
        elif self.vertical == 'none':
            raise ValueError("DataArray has 'lev' dimension but vertical='none' was specified.")
        elif isinstance(self.vertical, (list, np.ndarray)):
            # Select specific level indices
            levels = da.lev.values[self.vertical]
            return da.sel(lev=levels)
        else:
            raise ValueError(f"Invalid vertical specification: {self.vertical}")

    def _extract_local(self, da: xr.DataArray) -> np.ndarray:
        """
        Extract only local values at (x₀,t₀) - no neighborhood.

        Returns:
            np.ndarray of shape (nsamples, nfeatures) where nfeatures depends on vertical dim
        """
        if 'lev' in da.dims:
            # 4D: (time, lat, lon, lev) → (nsamples, nlevels)
            da = self._select_vertical_levels(da)
            da = da.sortby('lev', ascending=False)
            arr = da.transpose('time', 'lat', 'lon', 'lev').values.reshape(-1, da.lev.size)
        else:
            # 3D: (time, lat, lon) → (nsamples, 1)
            arr = da.transpose('time', 'lat', 'lon').values.reshape(-1, 1)
        return arr

    def _extract_valid(self, da: xr.DataArray) -> np.ndarray:
        """
        Extract patches only where full neighborhood fits (no padding).

        This reduces the number of samples by excluding edge regions where
        the full spatial/temporal window doesn't fit.

        Returns:
            np.ndarray of shape (nsamples_valid, nfeatures)
        """
        # Get array and dimensions
        has_vertical = 'lev' in da.dims
        if has_vertical:
            arr = da.transpose('time', 'lat', 'lon', 'lev').values
            nt, nlat, nlon, nlev = arr.shape
        else:
            arr = da.transpose('time', 'lat', 'lon').values
            nt, nlat, nlon = arr.shape
            nlev = 1
            arr = arr[..., np.newaxis]  # Add dummy vertical dim for consistency

        # Calculate valid ranges (exclude boundaries)
        t_start = self.temporal_window
        t_end = nt
        lat_start = self.lat_radius
        lat_end = nlat - self.lat_radius
        lon_start = self.lon_radius
        lon_end = nlon - self.lon_radius

        # Calculate patch size
        n_temporal = self.temporal_window + 1
        n_features = self.lat_size * self.lon_size * n_temporal * nlev

        # Calculate number of valid samples
        n_valid_t = t_end - t_start
        n_valid_lat = lat_end - lat_start
        n_valid_lon = lon_end - lon_start
        n_samples = n_valid_t * n_valid_lat * n_valid_lon

        # Pre-allocate output
        patches = np.zeros((n_samples, n_features), dtype=arr.dtype)

        # Extract patches
        idx = 0
        for t in range(t_start, t_end):
            for lat in range(lat_start, lat_end):
                for lon in range(lon_start, lon_end):
                    # Extract spatial-temporal-vertical patch
                    patch_parts = []
                    for dt in range(self.temporal_window + 1):
                        t_idx = t - dt  # Current and past timesteps
                        spatial_patch = arr[
                            t_idx,
                            lat - self.lat_radius : lat + self.lat_radius + 1,
                            lon - self.lon_radius : lon + self.lon_radius + 1,
                            :
                        ]
                        patch_parts.append(spatial_patch.flatten())

                    patches[idx] = np.concatenate(patch_parts)
                    idx += 1

        # Remove dummy vertical dimension if it was added
        if not has_vertical:
            n_features_no_lev = self.lat_size * self.lon_size * n_temporal
            patches = patches[:, :n_features_no_lev]

        return patches

    def _extract_padded(self, da: xr.DataArray, mode: str = 'reflect') -> np.ndarray:
        """
        Extract patches with boundary padding to preserve all samples.

        Pads the spatial and temporal dimensions so that patches can be extracted
        for all grid points, including those at boundaries.

        Args:
            mode: numpy padding mode ('reflect', 'constant', etc.)

        Returns:
            np.ndarray of shape (nsamples_all, nfeatures)
        """
        # Get array and dimensions
        has_vertical = 'lev' in da.dims
        if has_vertical:
            arr = da.transpose('time', 'lat', 'lon', 'lev').values
            nt, nlat, nlon, nlev = arr.shape
        else:
            arr = da.transpose('time', 'lat', 'lon').values
            nt, nlat, nlon = arr.shape
            nlev = 1
            arr = arr[..., np.newaxis]

        # Pad array
        # Padding: ((before_time, after_time), (before_lat, after_lat), (before_lon, after_lon), (0, 0))
        pad_width = (
            (self.temporal_window, 0),  # Pad past times only
            (self.lat_radius, self.lat_radius),
            (self.lon_radius, self.lon_radius),
            (0, 0)  # No padding for vertical
        )

        if mode == 'constant':
            arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=0)
        else:
            arr_padded = np.pad(arr, pad_width, mode=mode)

        # Calculate patch size
        n_temporal = self.temporal_window + 1
        n_features = self.lat_size * self.lon_size * n_temporal * nlev

        # Calculate number of samples
        n_samples = nt * nlat * nlon

        # Pre-allocate output
        patches = np.zeros((n_samples, n_features), dtype=arr.dtype)

        # Extract patches (now all positions are valid due to padding)
        idx = 0
        for t in range(nt):
            t_padded = t + self.temporal_window  # Offset due to temporal padding
            for lat in range(nlat):
                lat_padded = lat + self.lat_radius  # Offset due to spatial padding
                for lon in range(nlon):
                    lon_padded = lon + self.lon_radius

                    # Extract spatial-temporal-vertical patch
                    patch_parts = []
                    for dt in range(self.temporal_window + 1):
                        t_idx = t_padded - dt
                        spatial_patch = arr_padded[
                            t_idx,
                            lat_padded - self.lat_radius : lat_padded + self.lat_radius + 1,
                            lon_padded - self.lon_radius : lon_padded + self.lon_radius + 1,
                            :
                        ]
                        patch_parts.append(spatial_patch.flatten())

                    patches[idx] = np.concatenate(patch_parts)
                    idx += 1

        # Remove dummy vertical dimension if it was added
        if not has_vertical:
            n_features_no_lev = self.lat_size * self.lon_size * n_temporal
            patches = patches[:, :n_features_no_lev]

        return patches

    def get_feature_count(self, nlev: Optional[int] = None, has_vertical: bool = True) -> int:
        """
        Calculate the number of features per patch.

        Args:
            nlev: number of vertical levels (required if has_vertical=True)
            has_vertical: whether the variable has a vertical dimension

        Returns:
            int: total number of features per patch

        Example:
            >>> extractor = PatchExtractor(spatial_window=(1,1), temporal_window=2, vertical='all')
            >>> extractor.get_feature_count(nlev=16, has_vertical=True)
            432  # 3×3×3×16
        """
        n_temporal = self.temporal_window + 1
        n_spatial = self.lat_size * self.lon_size

        if has_vertical:
            if nlev is None:
                raise ValueError("nlev must be provided for variables with vertical dimension")
            return n_spatial * n_temporal * nlev
        else:
            return n_spatial * n_temporal
