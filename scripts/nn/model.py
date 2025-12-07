#!/usr/bin/env python

import torch
from patch import PatchExtractor

class NNModel(torch.nn.Module):

    def __init__(self,inputsize):
        '''
        Purpose: Define a feedforward neural network (NN) for precipitation prediction.
        Args:
        - inputsize (int): number of input features per sample (for 3D variables it's 1 per variable; for 4D variables it's
          'nlevels' per variable; for experiments with more than one variable it's the sum across variables)
        '''
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize,256), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256,128),       torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128,64),        torch.nn.GELU(), torch.nn.Dropout(0.1),            
            torch.nn.Linear(64,32),         torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(32,1))

    def forward(self,X):
        '''
        Purpose: Forward pass through the NN.
        Args:
        - X (torch.Tensor): input features tensor of shape (nsamples, inputsize)
        Returns:
        - torch.Tensor: raw prediction tensor of shape (nsamples, 1)
        '''
        return self.layers(X)


class BaselineNN(torch.nn.Module):

    def __init__(self,
                 nonlocal_vars=None,
                 local_vars=None,
                 spatial_window=(1, 1),
                 temporal_window=2,
                 vertical='all',
                 boundary='valid',
                 nlev=16):
        '''
        Purpose: Define a baseline NN that uses spatial-temporal-vertical patches for nonlocal inputs
                 and local-only values for local inputs.

        Args:
        - nonlocal_vars (list[str]): Variables that use neighborhood patches (e.g., ['rh', 'thetae', 'thetaestar'])
        - local_vars (list[str]): Variables that use only local (x₀,t₀) values (e.g., ['sdo', 'lhf', 'shf'])
        - spatial_window (tuple): (lat_radius, lon_radius) for spatial neighborhood
            - (0, 0): no spatial neighborhood (local only)
            - (1, 1): 3×3 grid
            - (2, 2): 5×5 grid
        - temporal_window (int): number of past timesteps to include
            - 0: current time only
            - 1: current + 1 past
            - 2: current + 2 past
        - vertical (str or list): which pressure levels to include
            - 'all': all pressure levels
            - list of indices: subset of levels
        - boundary (str): 'valid' | 'reflect' | 'constant'
        - nlev (int): number of vertical pressure levels (default: 16)

        Example:
            >>> # Model with nonlocal RH/θₑ using 3×3×3 patches, local land fraction
            >>> model = BaselineNN(
            ...     nonlocal_vars=['rh', 'thetae'],
            ...     local_vars=['sdo'],
            ...     spatial_window=(1, 1),
            ...     temporal_window=2,
            ...     vertical='all'
            ... )
        '''
        super().__init__()

        # Store configuration
        self.nonlocal_vars = nonlocal_vars or []
        self.local_vars = local_vars or []
        self.nlev = nlev

        # Create patch extractor for nonlocal variables
        self.nonlocal_extractor = PatchExtractor(
            spatial_window=spatial_window,
            temporal_window=temporal_window,
            vertical=vertical,
            boundary=boundary
        )

        # Create patch extractor for local variables (no neighborhood)
        self.local_extractor = PatchExtractor(
            spatial_window=(0, 0),
            temporal_window=0,
            vertical=vertical,
            boundary=boundary
        )

        # Calculate input size
        # Each nonlocal 4D variable contributes (spatial × temporal × vertical) features
        n_temporal = temporal_window + 1
        lat_size = 2 * spatial_window[0] + 1
        lon_size = 2 * spatial_window[1] + 1
        nonlocal_features_per_var = lat_size * lon_size * n_temporal * nlev
        total_nonlocal_features = len(self.nonlocal_vars) * nonlocal_features_per_var

        # Each local 3D variable contributes 1 feature, 4D contributes nlev features
        # For now, assume local vars are 3D (can be extended)
        total_local_features = len(self.local_vars)

        inputsize = total_nonlocal_features + total_local_features

        # Create the neural network
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize, 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),       torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),        torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 32),         torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 1))

    def forward(self, X):
        '''
        Purpose: Forward pass through the baseline NN with patch-based inputs.

        Args:
        - X (torch.Tensor): input features tensor of shape (nsamples, inputsize)
                           This should be the concatenated patches from all variables

        Returns:
        - torch.Tensor: raw prediction tensor of shape (nsamples, 1)
        '''
        return self.layers(X)

    def extract_features(self, dataset_dict):
        '''
        Purpose: Extract patches from raw xarray data for all variables.

        Args:
        - dataset_dict (dict): dictionary mapping variable names to xr.DataArray objects
                              Each DataArray should have dims (time, lat, lon) or (time, lat, lon, lev)

        Returns:
        - np.ndarray: concatenated feature matrix of shape (nsamples, inputsize)

        Example:
            >>> import xarray as xr
            >>> dataset = {
            ...     'rh': xr.open_dataarray('rh.nc'),
            ...     'thetae': xr.open_dataarray('thetae.nc'),
            ...     'sdo': xr.open_dataarray('sdo.nc')
            ... }
            >>> features = model.extract_features(dataset)
            >>> predictions = model(torch.tensor(features))
        '''
        import numpy as np

        feature_arrays = []

        # Extract nonlocal features (with neighborhoods)
        for var in self.nonlocal_vars:
            if var not in dataset_dict:
                raise ValueError(f"Variable '{var}' not found in dataset_dict")
            patches = self.nonlocal_extractor.extract(dataset_dict[var], local_only=False)
            feature_arrays.append(patches)

        # Extract local features (no neighborhoods)
        for var in self.local_vars:
            if var not in dataset_dict:
                raise ValueError(f"Variable '{var}' not found in dataset_dict")
            patches = self.local_extractor.extract(dataset_dict[var], local_only=True)
            feature_arrays.append(patches)

        # Concatenate all features
        if len(feature_arrays) == 0:
            raise ValueError("No variables specified for feature extraction")

        features = np.concatenate(feature_arrays, axis=1)
        return features