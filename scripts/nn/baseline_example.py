#!/usr/bin/env python
"""
Example script demonstrating how to use PatchExtractor and BaselineNN for
spatial-temporal-vertical patch-based precipitation prediction.

This example shows:
1. How to configure patch extractors for different use cases
2. How to extract features from xarray datasets
3. How to train a BaselineNN model
4. Different configurations for nonlocal vs local inputs
"""

import torch
import xarray as xr
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import BaselineNN
from patch import PatchExtractor


# =============================================================================
# EXAMPLE 1: Basic usage with PatchExtractor
# =============================================================================

def example_1_basic_patch_extraction():
    """
    Demonstrates basic patch extraction for a single variable.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Patch Extraction")
    print("=" * 80)

    # Load some sample data (replace with actual file paths)
    # For this example, we'll create dummy data
    times = np.arange(10)
    lats = np.arange(20)
    lons = np.arange(20)
    levs = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10])

    # Create dummy 4D variable (time, lat, lon, lev)
    rh_data = np.random.randn(len(times), len(lats), len(lons), len(levs))
    rh = xr.DataArray(
        rh_data,
        coords={'time': times, 'lat': lats, 'lon': lons, 'lev': levs},
        dims=['time', 'lat', 'lon', 'lev'],
        name='rh'
    )

    # Configuration 1: Extract full 3D patches (spatial + temporal + vertical)
    print("\n--- Configuration 1: Full 3D patches (3×3 spatial, 3 temporal, all 16 levels) ---")
    extractor_3d = PatchExtractor(
        spatial_window=(1, 1),      # 3×3 spatial neighborhood
        temporal_window=2,           # Current + 2 past timesteps = 3 total
        vertical='all',              # All 16 pressure levels
        boundary='valid'             # Only extract where full patch fits
    )

    patches_3d = extractor_3d.extract(rh)
    print(f"Patches shape: {patches_3d.shape}")
    print(f"Features per sample: {3*3*3*16} = {patches_3d.shape[1]}")
    print(f"Number of samples: {patches_3d.shape[0]}")

    # Configuration 2: Extract only vertical profiles (no spatial/temporal neighborhood)
    print("\n--- Configuration 2: Vertical profiles only (local at x₀,t₀) ---")
    extractor_vertical = PatchExtractor(
        spatial_window=(0, 0),       # No spatial neighborhood
        temporal_window=0,            # No temporal window
        vertical='all',               # All 16 pressure levels
        boundary='valid'
    )

    patches_vertical = extractor_vertical.extract(rh)
    print(f"Patches shape: {patches_vertical.shape}")
    print(f"Features per sample: {16} = {patches_vertical.shape[1]}")

    # Configuration 3: Extract spatial patches only (single timestep, single level)
    print("\n--- Configuration 3: Spatial patches only (5×5 grid, surface level) ---")
    # First, select a single level
    rh_surface = rh.isel(lev=0)  # Shape: (time, lat, lon)

    extractor_spatial = PatchExtractor(
        spatial_window=(2, 2),       # 5×5 spatial neighborhood
        temporal_window=0,            # No temporal window
        vertical='none',              # No vertical dimension
        boundary='reflect'            # Use reflection at boundaries
    )

    patches_spatial = extractor_spatial.extract(rh_surface)
    print(f"Patches shape: {patches_spatial.shape}")
    print(f"Features per sample: {5*5} = {patches_spatial.shape[1]}")

    print()


# =============================================================================
# EXAMPLE 2: Using BaselineNN with multiple variables
# =============================================================================

def example_2_baseline_nn_training():
    """
    Demonstrates how to train a BaselineNN model with nonlocal and local inputs.
    """
    print("=" * 80)
    print("EXAMPLE 2: BaselineNN Training with Nonlocal + Local Inputs")
    print("=" * 80)

    # Create dummy dataset
    times = np.arange(100)
    lats = np.arange(30)
    lons = np.arange(30)
    levs = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10])

    # Nonlocal 4D variables (need spatial-temporal-vertical neighborhoods)
    rh = xr.DataArray(
        np.random.randn(len(times), len(lats), len(lons), len(levs)),
        coords={'time': times, 'lat': lats, 'lon': lons, 'lev': levs},
        dims=['time', 'lat', 'lon', 'lev'],
        name='rh'
    )

    thetae = xr.DataArray(
        np.random.randn(len(times), len(lats), len(lons), len(levs)),
        coords={'time': times, 'lat': lats, 'lon': lons, 'lev': levs},
        dims=['time', 'lat', 'lon', 'lev'],
        name='thetae'
    )

    # Local 3D variables (only need values at x₀,t₀)
    sdo = xr.DataArray(
        np.random.randn(len(lats), len(lons)),  # Static field (no time dimension)
        coords={'lat': lats, 'lon': lons},
        dims=['lat', 'lon'],
        name='sdo'
    )
    # Expand to match time dimension for processing
    sdo = sdo.expand_dims(time=times)

    # Target variable (precipitation)
    pr = xr.DataArray(
        np.random.randn(len(times), len(lats), len(lons)),
        coords={'time': times, 'lat': lats, 'lon': lons},
        dims=['time', 'lat', 'lon'],
        name='pr'
    )

    # Create dataset dictionary
    dataset = {
        'rh': rh,
        'thetae': thetae,
        'sdo': sdo,
        'pr': pr
    }

    # Initialize BaselineNN model
    print("\n--- Initializing BaselineNN ---")
    model = BaselineNN(
        nonlocal_vars=['rh', 'thetae'],  # Use 3D patches for RH and θₑ
        local_vars=['sdo'],               # Use local values only for surface orography
        spatial_window=(1, 1),            # 3×3 spatial neighborhood
        temporal_window=2,                # Current + 2 past timesteps
        vertical='all',                   # All 16 pressure levels
        boundary='valid',                 # Only use points where full patch fits
        nlev=16
    )

    # Extract features
    print("\n--- Extracting features ---")
    X = model.extract_features(dataset)
    print(f"Feature matrix shape: {X.shape}")
    print(f"  - RH patches: 3×3×3×16 = 432 features")
    print(f"  - θₑ patches: 3×3×3×16 = 432 features")
    print(f"  - sdo local: 1 feature")
    print(f"  - Total: 432 + 432 + 1 = {X.shape[1]} features")

    # Extract target (use same patch extractor to match samples)
    pr_extractor = PatchExtractor(
        spatial_window=(1, 1),
        temporal_window=2,
        vertical='none',
        boundary='valid'
    )
    y = pr_extractor.extract(pr, local_only=True)
    print(f"Target shape: {y.shape}")

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create DataLoader
    dataset_torch = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset_torch, batch_size=64, shuffle=True)

    # Simple training loop (just one epoch as demonstration)
    print("\n--- Training model (1 epoch) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions.squeeze(-1), batch_y.squeeze(-1))
        loss.backward()
        optimizer.step()

    print(f"Final batch loss: {loss.item():.6f}")
    print()


# =============================================================================
# EXAMPLE 3: Different patch configurations for different models
# =============================================================================

def example_3_flexible_configurations():
    """
    Demonstrates how to use different patch configurations for different model types.
    """
    print("=" * 80)
    print("EXAMPLE 3: Flexible Patch Configurations")
    print("=" * 80)

    times = np.arange(20)
    lats = np.arange(20)
    lons = np.arange(20)
    levs = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10])

    rh = xr.DataArray(
        np.random.randn(len(times), len(lats), len(lons), len(levs)),
        coords={'time': times, 'lat': lats, 'lon': lons, 'lev': levs},
        dims=['time', 'lat', 'lon', 'lev']
    )

    print("\n--- Use Case 1: Kernel model (vertical integration only) ---")
    # For kernel models: extract only vertical profiles at (x₀,t₀)
    kernel_extractor = PatchExtractor(
        spatial_window=(0, 0),
        temporal_window=0,
        vertical='all',
        boundary='valid'
    )
    kernel_patches = kernel_extractor.extract(rh)
    print(f"Kernel patches shape: {kernel_patches.shape}")
    print(f"Each sample has 16 vertical levels")

    print("\n--- Use Case 2: Spatial convolution model (2D spatial patches) ---")
    # For spatial convolution: extract spatial patches at single level
    rh_700 = rh.sel(lev=700, method='nearest').isel(time=0)  # Single time, single level
    conv_extractor = PatchExtractor(
        spatial_window=(2, 2),  # 5×5 patch for convolution
        temporal_window=0,
        vertical='none',
        boundary='reflect'
    )
    conv_patches = conv_extractor.extract(rh_700)
    print(f"Convolution patches shape: {conv_patches.shape}")
    print(f"Each sample is a 5×5 = 25 pixel patch")

    print("\n--- Use Case 3: Temporal sequence model (time series at point) ---")
    # For temporal models: extract time series at single point
    temporal_extractor = PatchExtractor(
        spatial_window=(0, 0),
        temporal_window=5,      # 6 timesteps total (current + 5 past)
        vertical='all',
        boundary='valid'
    )
    temporal_patches = temporal_extractor.extract(rh)
    print(f"Temporal patches shape: {temporal_patches.shape}")
    print(f"Each sample has 6 timesteps × 16 levels = {6*16} features")

    print("\n--- Use Case 4: Full 3D patches (as in BaselineNN) ---")
    # For baseline NN: full spatial-temporal-vertical patches
    baseline_extractor = PatchExtractor(
        spatial_window=(1, 1),
        temporal_window=2,
        vertical='all',
        boundary='valid'
    )
    baseline_patches = baseline_extractor.extract(rh)
    print(f"Baseline patches shape: {baseline_patches.shape}")
    print(f"Each sample has 3×3×3×16 = {3*3*3*16} features")

    print()


# =============================================================================
# EXAMPLE 4: Using subset of vertical levels
# =============================================================================

def example_4_vertical_subset():
    """
    Demonstrates how to extract patches using only a subset of pressure levels.
    """
    print("=" * 80)
    print("EXAMPLE 4: Using Subset of Vertical Levels")
    print("=" * 80)

    times = np.arange(10)
    lats = np.arange(15)
    lons = np.arange(15)
    levs = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10])

    q = xr.DataArray(
        np.random.randn(len(times), len(lats), len(lons), len(levs)),
        coords={'time': times, 'lat': lats, 'lon': lons, 'lev': levs},
        dims=['time', 'lat', 'lon', 'lev']
    )

    print("\n--- Using all 16 levels ---")
    extractor_all = PatchExtractor(
        spatial_window=(0, 0),
        temporal_window=0,
        vertical='all',
        boundary='valid'
    )
    patches_all = extractor_all.extract(q)
    print(f"Patches shape: {patches_all.shape}")
    print(f"Features per sample: {patches_all.shape[1]}")

    print("\n--- Using only lower troposphere (levels 0-5) ---")
    # Select only first 6 levels: 1000, 925, 850, 700, 600, 500 hPa
    extractor_lower = PatchExtractor(
        spatial_window=(0, 0),
        temporal_window=0,
        vertical=[0, 1, 2, 3, 4, 5],  # Indices of levels to use
        boundary='valid'
    )
    patches_lower = extractor_lower.extract(q)
    print(f"Patches shape: {patches_lower.shape}")
    print(f"Features per sample: {patches_lower.shape[1]}")

    print("\n--- Using only upper troposphere/stratosphere (levels 10-15) ---")
    # Select last 6 levels: 150, 100, 70, 50, 30, 10 hPa
    extractor_upper = PatchExtractor(
        spatial_window=(0, 0),
        temporal_window=0,
        vertical=[10, 11, 12, 13, 14, 15],
        boundary='valid'
    )
    patches_upper = extractor_upper.extract(q)
    print(f"Patches shape: {patches_upper.shape}")
    print(f"Features per sample: {patches_upper.shape[1]}")

    print()


# =============================================================================
# Main execution
# =============================================================================

if __name__ == '__main__':
    print("\n")
    print("*" * 80)
    print("PATCH EXTRACTION AND BASELINENN EXAMPLES")
    print("*" * 80)
    print()

    # Run all examples
    example_1_basic_patch_extraction()
    example_2_baseline_nn_training()
    example_3_flexible_configurations()
    example_4_vertical_subset()

    print("*" * 80)
    print("All examples completed successfully!")
    print("*" * 80)
    print()
