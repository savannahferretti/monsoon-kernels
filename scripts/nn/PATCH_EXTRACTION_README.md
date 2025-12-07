# Patch Extraction for Spatial-Temporal-Vertical Neighborhoods

This directory contains a flexible patch extraction system for precipitation prediction using neural networks. The system allows you to extract multi-dimensional neighborhoods around target points for both nonlocal (neighborhood-based) and local (point-based) inputs.

## Overview

For precipitation prediction at a given grid cell location and time `y(x₀,t₀)`, the model needs:

1. **Nonlocal inputs**: Variables that benefit from spatial-temporal-vertical context
   - Extract a "neighborhood" around `(x₀,t₀)` including:
     - Surrounding grid cells (spatial neighborhood)
     - Full pressure column for each grid cell (vertical dimension)
     - Past time steps (temporal history)
   - Examples: RH (relative humidity), θₑ (equivalent potential temperature), θₑ*

2. **Local inputs**: Variables that only need values at the target point `(x₀,t₀)`
   - No spatial or temporal neighborhood
   - Examples: land fraction, latent heat flux, sensible heat flux

## Files

- **`patch.py`**: Core `PatchExtractor` class for flexible neighborhood extraction
- **`model.py`**: Contains `BaselineNN` class that uses patch extraction
- **`baseline_example.py`**: Comprehensive examples showing different use cases

## Quick Start

### Example 1: Basic Patch Extraction

```python
from patch import PatchExtractor
import xarray as xr

# Load your data (time, lat, lon, lev)
rh = xr.open_dataarray('rh.nc')

# Create extractor for 3×3 spatial, 3 temporal (current + 2 past), all 16 levels
extractor = PatchExtractor(
    spatial_window=(1, 1),      # 3×3 grid
    temporal_window=2,           # 3 time steps
    vertical='all',              # All pressure levels
    boundary='valid'             # Only where full patch fits
)

# Extract patches
patches = extractor.extract(rh)
# Shape: (nsamples, 3×3×3×16) = (nsamples, 432)
```

### Example 2: BaselineNN with Multiple Variables

```python
from model import BaselineNN
import torch

# Initialize model
model = BaselineNN(
    nonlocal_vars=['rh', 'thetae', 'thetaestar'],  # 3D neighborhoods
    local_vars=['sdo', 'lhf', 'shf'],               # Local only
    spatial_window=(1, 1),                          # 3×3 spatial
    temporal_window=2,                              # 3 time steps
    vertical='all',                                 # All 16 levels
    boundary='valid'
)

# Load data
dataset = {
    'rh': xr.open_dataarray('rh.nc'),
    'thetae': xr.open_dataarray('thetae.nc'),
    'thetaestar': xr.open_dataarray('thetaestar.nc'),
    'sdo': xr.open_dataarray('sdo.nc'),
    'lhf': xr.open_dataarray('lhf.nc'),
    'shf': xr.open_dataarray('shf.nc')
}

# Extract features automatically
X = model.extract_features(dataset)

# Make predictions
predictions = model(torch.tensor(X, dtype=torch.float32))
```

## PatchExtractor Configuration

### Spatial Window

Controls the spatial neighborhood size around each target point.

```python
spatial_window=(lat_radius, lon_radius)
```

| Configuration | Grid Size | Example |
|---------------|-----------|---------|
| `(0, 0)` | 1×1 | Local only (no neighborhood) |
| `(1, 1)` | 3×3 | 1 cell in each direction |
| `(2, 2)` | 5×5 | 2 cells in each direction |
| `(3, 3)` | 7×7 | 3 cells in each direction |

### Temporal Window

Controls how many past timesteps to include.

```python
temporal_window=n  # Number of past timesteps
```

| Configuration | Timesteps Included | Total |
|---------------|-------------------|-------|
| `0` | t₀ | 1 |
| `1` | t₀, t₋₁ | 2 |
| `2` | t₀, t₋₁, t₋₂ | 3 |
| `5` | t₀, t₋₁, ..., t₋₅ | 6 |

### Vertical Levels

Controls which pressure levels to include.

```python
vertical='all'              # All pressure levels
vertical=[0, 1, 2, 3, 4]   # Specific level indices
```

| Configuration | Description |
|---------------|-------------|
| `'all'` | Include all pressure levels (typically 16) |
| `[0,1,2,3,4,5]` | Lower troposphere only (1000-500 hPa) |
| `[10,11,12,13,14,15]` | Upper troposphere/stratosphere (150-10 hPa) |

### Boundary Handling

Controls how to handle edges where full patches don't fit.

```python
boundary='valid'    # Only extract where full patch fits (reduces nsamples)
boundary='reflect'  # Mirror/reflect at boundaries (preserves nsamples)
boundary='constant' # Pad with zeros at boundaries (preserves nsamples)
```

## Feature Count Calculation

For a 4D variable (time, lat, lon, lev), the number of features per sample is:

```
nfeatures = (2×lat_radius + 1) × (2×lon_radius + 1) × (temporal_window + 1) × nlevels
```

Examples:
- `spatial_window=(1,1), temporal_window=2, vertical='all'` (16 levels):
  - `3 × 3 × 3 × 16 = 432 features`

- `spatial_window=(2,2), temporal_window=1, vertical=[0,1,2,3,4,5]` (6 levels):
  - `5 × 5 × 2 × 6 = 300 features`

For a 3D variable (time, lat, lon), omit the vertical dimension:

```
nfeatures = (2×lat_radius + 1) × (2×lon_radius + 1) × (temporal_window + 1)
```

## BaselineNN Architecture

The `BaselineNN` class automatically:

1. Creates separate `PatchExtractor` instances for nonlocal and local variables
2. Extracts features from all specified variables
3. Concatenates features into a single input vector
4. Passes through a feedforward neural network

### Input Size Calculation

```
total_features = (n_nonlocal_vars × features_per_nonlocal) + (n_local_vars × features_per_local)
```

Example:
- Nonlocal: `rh`, `thetae` (2 variables) with 3×3×3×16 = 432 features each
- Local: `sdo` (1 variable) with 1 feature
- Total: `2×432 + 1×1 = 865 features`

## Use Cases

### 1. Kernel Integration Models

Extract only vertical profiles (no spatial/temporal neighborhood):

```python
extractor = PatchExtractor(
    spatial_window=(0, 0),
    temporal_window=0,
    vertical='all',
    boundary='valid'
)
# Output: (nsamples, 16) for vertical integration
```

### 2. Spatial Convolution Models

Extract 2D spatial patches at single level/time:

```python
rh_surface = rh.isel(lev=0, time=0)
extractor = PatchExtractor(
    spatial_window=(2, 2),  # 5×5 patch
    temporal_window=0,
    vertical='none',
    boundary='reflect'
)
# Output: (nsamples, 25) for 2D convolution
```

### 3. Temporal Sequence Models

Extract time series at single location:

```python
extractor = PatchExtractor(
    spatial_window=(0, 0),
    temporal_window=5,      # 6 timesteps
    vertical='all',
    boundary='valid'
)
# Output: (nsamples, 6×16=96) for LSTM/RNN
```

### 4. Full 3D Baseline Model

Extract spatial-temporal-vertical neighborhoods:

```python
extractor = PatchExtractor(
    spatial_window=(1, 1),  # 3×3
    temporal_window=2,       # 3 timesteps
    vertical='all',          # 16 levels
    boundary='valid'
)
# Output: (nsamples, 432) for baseline NN
```

## Data Requirements

### Input Data Format

All input data should be `xarray.DataArray` objects with standard dimension names:

- **4D variables** (e.g., RH, θₑ, q, T): `(time, lat, lon, lev)`
- **3D variables** (e.g., precipitation, land fraction): `(time, lat, lon)`
- **Static 2D fields**: `(lat, lon)` - will be expanded to match time dimension

### Dimension Conventions

- `time`: Temporal dimension (e.g., 3-hourly timesteps)
- `lat`: Latitude dimension (grid cells)
- `lon`: Longitude dimension (grid cells)
- `lev`: Vertical pressure levels in hPa (e.g., [1000, 925, 850, ..., 10])

The code automatically sorts pressure levels in descending order (surface to top of atmosphere).

## Sample Count Considerations

### Valid Boundary Mode

When using `boundary='valid'`, samples are excluded at the edges:

```python
n_valid_samples = (n_time - temporal_window) × (n_lat - 2×lat_radius) × (n_lon - 2×lon_radius)
```

Example:
- Original grid: 100 time × 50 lat × 50 lon = 250,000 points
- With `spatial_window=(1,1), temporal_window=2, boundary='valid'`:
  - Valid samples: 98 × 48 × 48 = 225,792 (90% of original)

### Reflect/Constant Boundary Modes

Preserve all samples by padding:

```python
n_samples = n_time × n_lat × n_lon  # All points included
```

## Training Integration

### Complete Training Example

```python
import torch
import xarray as xr
from torch.utils.data import TensorDataset, DataLoader
from model import BaselineNN
from patch import PatchExtractor

# 1. Load data
train_data = {
    'rh': xr.open_dataset('train.h5')['rh'],
    'thetae': xr.open_dataset('train.h5')['thetae'],
    'sdo': xr.open_dataset('train.h5')['sdo'],
    'pr': xr.open_dataset('train.h5')['pr']  # Target
}

# 2. Initialize model
model = BaselineNN(
    nonlocal_vars=['rh', 'thetae'],
    local_vars=['sdo'],
    spatial_window=(1, 1),
    temporal_window=2,
    vertical='all',
    boundary='valid'
)

# 3. Extract features
X_train = model.extract_features(train_data)

# 4. Extract target (match the sample reduction from boundary='valid')
target_extractor = PatchExtractor(
    spatial_window=(1, 1),
    temporal_window=2,
    vertical='none',
    boundary='valid'
)
y_train = target_extractor.extract(train_data['pr'], local_only=True)

# 5. Create DataLoader
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 6. Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(10):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch.squeeze())
        loss.backward()
        optimizer.step()
```

## Troubleshooting

### Issue: Shape mismatch between features and target

**Solution**: Ensure you use the same boundary mode and window sizes when extracting target:

```python
# Features extraction
X = model.extract_features(dataset)  # Uses model's boundary='valid'

# Target extraction - must match!
target_extractor = PatchExtractor(
    spatial_window=model.nonlocal_extractor.spatial_window,  # Same as model
    temporal_window=model.nonlocal_extractor.temporal_window,
    vertical='none',
    boundary=model.nonlocal_extractor.boundary  # Must match!
)
y = target_extractor.extract(dataset['pr'], local_only=True)
```

### Issue: Out of memory errors

**Solutions**:
1. Use smaller spatial/temporal windows
2. Extract features in batches
3. Use `boundary='valid'` to reduce sample count
4. Subset vertical levels instead of using all

### Issue: Different variables have different time dimensions

**Solution**: Expand static fields to match temporal dimension:

```python
# Static field (lat, lon)
sdo = xr.open_dataarray('sdo.nc')

# Expand to match time dimension
sdo = sdo.expand_dims(time=target_times)
```

## Performance Tips

1. **Pre-extract features once**: Don't re-extract on every epoch
2. **Use boundary='valid'**: Reduces sample count and avoids padding overhead
3. **Batch processing**: For very large datasets, extract features in chunks
4. **Subset levels**: Use only relevant pressure levels if possible
5. **Cache results**: Save extracted features to disk for reuse

## References

For more details on the monsoon precipitation prediction project, see:
- Main README: `../../README.md`
- Model architecture: `model.py`, `kernel_model.py`
- Training scripts: `train.py`, `eval.py`
- Example usage: `baseline_example.py`
