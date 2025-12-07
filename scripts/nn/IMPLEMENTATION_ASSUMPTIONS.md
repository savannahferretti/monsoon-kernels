# Implementation Assumptions & Verification Checklist

**⚠️ IMPORTANT: This code is UNTESTED and should be treated as a reference implementation / proof of concept.**

This document lists all assumptions made during implementation that are **not explicitly grounded in the existing codebase**. Please verify each assumption before integrating this code into your workflow.

---

## 🔴 Critical Assumptions (High Priority - Verify First)

### 1. All Local Variables are 3D (No Vertical Dimension)

**Location:** `model.py`, line 107

```python
total_local_features = len(self.local_vars)  # Assumes 1 feature per variable
```

**Assumption:** Local variables only contribute 1 feature each (i.e., they're 3D: `time, lat, lon`)

**What could go wrong:** If you want to use a 4D variable locally (e.g., full vertical profile of temperature at x₀,t₀ only), the input size calculation will be wrong and cause shape mismatches.

**How to verify:**
```python
# Check your local variables
for var in ['sdo', 'lhf', 'shf']:
    da = xr.open_dataarray(f'{var}.nc')
    print(f"{var}: {da.dims}")  # Should be (lat, lon) or (time, lat, lon)
```

**Fix if needed:** Modify the `BaselineNN` class to detect whether local variables have a vertical dimension and calculate features accordingly.

---

### 2. All Nonlocal Variables are 4D (Have Vertical Dimension)

**Location:** `model.py`, line 102

```python
nonlocal_features_per_var = lat_size * lon_size * n_temporal * nlev
```

**Assumption:** All nonlocal variables have a vertical dimension and use `nlev` levels.

**What could go wrong:** If you want spatial neighborhoods for a 3D variable (e.g., 3×3 patch of surface temperature), the feature count will be wrong.

**How to verify:**
```python
# Check your nonlocal variables
for var in ['rh', 'thetae', 'thetaestar']:
    da = xr.open_dataarray(f'{var}.nc')
    print(f"{var}: {da.dims}")  # Should be (time, lat, lon, lev)
    if 'lev' in da.dims:
        print(f"  Levels: {da.lev.size}")  # Should be 16
```

**Fix if needed:** Add logic to detect variable dimensionality and calculate features per variable type.

---

### 3. Sample Count Reduction from `boundary='valid'` is Acceptable

**Location:** `patch.py`, `_extract_valid()` method

**Assumption:** Losing edge samples is acceptable. With default settings (`spatial_window=(1,1)`, `temporal_window=2`), you lose approximately:
- 2 timesteps at the start (temporal boundary)
- 2 lat cells on top/bottom edges (spatial boundary)
- 2 lon cells on left/right edges (spatial boundary)

**What could go wrong:**
- Sample count might not match your expected batch size (current code uses `batchsize=2208`)
- Important geographic regions at boundaries might be excluded (e.g., coastal monsoon regions)
- May create train/valid/test split imbalances

**How to verify:**
```python
# Calculate expected sample reduction
original_samples = n_time * n_lat * n_lon
valid_samples = (n_time - temporal_window) * (n_lat - 2*lat_radius) * (n_lon - 2*lon_radius)
reduction_pct = 100 * (1 - valid_samples/original_samples)
print(f"Sample reduction: {reduction_pct:.1f}%")

# Check if batch size divides evenly
print(f"Valid samples: {valid_samples}")
print(f"Batch size: 2208")
print(f"Batches per epoch: {valid_samples / 2208}")
```

**Alternatives:**
- Use `boundary='reflect'` to preserve all samples
- Use `boundary='constant'` with zero-padding
- Adjust batch size to accommodate new sample count

---

### 4. All Variables Have Aligned Coordinates

**Location:** `model.py`, `extract_features()` method

**Assumption:** All variables in `dataset_dict` have identical and perfectly aligned `time`, `lat`, `lon` coordinates.

**What could go wrong:**
- If variables are on different grids (e.g., different lat/lon resolution), extraction will fail or produce misaligned features
- If time coordinates don't match exactly, concatenation will fail

**How to verify:**
```python
# Check coordinate alignment
dataset = xr.open_dataset('normtrain.h5')
vars_to_check = ['rh', 'thetae', 'sdo', 'pr']

# Compare coordinates
for var in vars_to_check[1:]:
    assert np.array_equal(dataset[vars_to_check[0]].lat.values, dataset[var].lat.values)
    assert np.array_equal(dataset[vars_to_check[0]].lon.values, dataset[var].lon.values)
    if 'time' in dataset[var].dims:
        assert np.array_equal(dataset[vars_to_check[0]].time.values, dataset[var].time.values)
```

**Fix if needed:** Add coordinate alignment checks or automatic resampling to `extract_features()`.

---

### 5. No Missing Data (NaN) Handling

**Location:** Entire codebase - no NaN checks anywhere

**Assumption:** Input data has no NaN or missing values.

**What could go wrong:**
- Real climate data often has gaps (sensor failures, quality control filtering)
- NaNs will propagate through patches and corrupt multiple samples
- Model training will fail or learn nonsense patterns

**How to verify:**
```python
# Check for NaNs in your data
dataset = xr.open_dataset('normtrain.h5')
for var in dataset.data_vars:
    nan_count = np.isnan(dataset[var].values).sum()
    if nan_count > 0:
        print(f"{var}: {nan_count} NaN values")
```

**Fix if needed:**
- Pre-process data to fill or remove NaNs before patch extraction
- Add NaN detection to `PatchExtractor.extract()` with configurable handling (skip, fill, error)

---

### 6. Target Variable Extraction Must Match Feature Boundaries

**Location:** Not automated - user must do this manually

**Assumption:** User will remember to create a matching `PatchExtractor` for the target variable with identical boundary settings.

**What could go wrong:** This is VERY error-prone. If you forget to match boundaries:
```python
# WRONG - will cause shape mismatch
X = model.extract_features(dataset)  # Uses boundary='valid'
y = dataset['pr'].values.reshape(-1, 1)  # Uses all samples

# X.shape[0] != y.shape[0] → training will crash
```

**How to verify:**
```python
# Always verify shapes match
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
assert X.shape[0] == y.shape[0], "Sample count mismatch!"
```

**Fix recommended:** Add a helper method to `BaselineNN`:
```python
def extract_target(self, target_da):
    """Extract target with matching boundaries."""
    target_extractor = PatchExtractor(
        spatial_window=self.nonlocal_extractor.spatial_window,
        temporal_window=self.nonlocal_extractor.temporal_window,
        vertical='none',
        boundary=self.nonlocal_extractor.boundary
    )
    return target_extractor.extract(target_da, local_only=True)
```

---

## 🟡 Moderate Assumptions (Should Verify)

### 7. 16 Pressure Levels

**Location:** `model.py`, default `nlev=16`

**Assumption:** Based on `demo.ipynb` showing 16 levels.

**How to verify:**
```python
dataset = xr.open_dataset('normtrain.h5')
for var in ['rh', 'thetae', 'q', 't']:
    if 'lev' in dataset[var].dims:
        print(f"{var}: {dataset[var].lev.size} levels")
```

**What to check:** Is 16 consistent across all files (train, valid, test)?

---

### 8. Data Comes Pre-Normalized

**Location:** No normalization in patch extraction

**Assumption:** Patch extraction happens after the normalization step (based on existing pipeline: download → calculate → normalize → split).

**What could go wrong:** If you try to extract patches from raw (unnormalized) data, model performance will be poor.

**How to verify:** Check that you're loading from `normtrain.h5`, `normvalid.h5`, `normtest.h5` (not `train.h5`).

---

### 9. Static Fields Can Be Expanded with `expand_dims`

**Location:** Examples and documentation

```python
sdo = sdo.expand_dims(time=target_times)
```

**Assumption:** 2D static fields (like topography `sdo`) can be broadcast to all timesteps using xarray's `expand_dims`.

**What could go wrong:**
- Dimension ordering might not match
- Memory inefficiency (duplicating data across time)

**How to verify:**
```python
# Test with actual sdo data
sdo = xr.open_dataset('normtrain.h5')['sdo']
print(f"Original dims: {sdo.dims}")

if 'time' not in sdo.dims:
    target_times = xr.open_dataset('normtrain.h5')['pr'].time
    sdo_expanded = sdo.expand_dims(time=target_times)
    print(f"Expanded dims: {sdo_expanded.dims}")
    print(f"Expected shape: (time={len(target_times)}, lat={sdo.lat.size}, lon={sdo.lon.size})")
    print(f"Actual shape: {sdo_expanded.shape}")
```

---

### 10. Memory Can Hold Full Feature Matrix

**Location:** `extract_features()` - loads everything at once

```python
features = np.concatenate(feature_arrays, axis=1)  # All at once
```

**Assumption:** Full feature matrix fits in memory.

**What could go wrong:** With large configurations:
- Example: 1M samples × 5000 features × 4 bytes = 20 GB
- System runs out of RAM and crashes or swaps heavily

**How to calculate:**
```python
# Estimate memory usage
n_samples = (n_time - temporal_window) * (n_lat - 2*lat_radius) * (n_lon - 2*lon_radius)
n_features = len(nonlocal_vars) * (lat_size * lon_size * n_temporal * nlev) + len(local_vars)
memory_gb = (n_samples * n_features * 4) / (1024**3)  # 4 bytes per float32
print(f"Estimated memory: {memory_gb:.2f} GB")
```

**Fix if needed:** Implement batch-based feature extraction for large datasets.

---

### 11. Pressure Levels Sorted Descending (Surface → Top)

**Location:** `patch.py`, line in `_select_vertical_levels()`

```python
da = da.sortby('lev', ascending=False)
```

**Assumption:** Based on existing `train.py` line 46, but didn't verify this is universal.

**How to verify:**
```python
dataset = xr.open_dataset('normtrain.h5')
print(f"Pressure levels: {dataset['rh'].lev.values}")
# Should be: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 10]
```

**What to check:** Is this ordering consistent across all your data files?

---

### 12. NumPy Padding Modes Are Physically Sensible

**Location:** `patch.py`, `_extract_padded()` method

```python
arr_padded = np.pad(arr, pad_width, mode='reflect')
```

**Assumption:** Reflecting or zero-padding at spatial/temporal boundaries makes sense for atmospheric data.

**What could go wrong:**
- Reflecting atmospheric conditions at geographic boundaries may not be physically meaningful
- For longitude, periodic boundary conditions might be more appropriate (wrapping around the globe)
- For time, reflecting past data may create spurious patterns

**Consider:**
- Using `mode='wrap'` for longitude (periodic)
- Using `mode='edge'` (replicate edge values) instead of reflect
- Or just stick with `boundary='valid'` to avoid the issue

---

### 13. No Integration with Existing `reshape()` Function

**Location:** `train.py` line 37-50 has an existing `reshape()` function

**Observation:** My implementation is a complete replacement, not an extension of existing code.

**Implication:** You'll need to modify `train.py` significantly to use the new patch-based approach. It won't be a drop-in replacement.

**What needs changing in train.py:**
- Replace `load()` function to use `BaselineNN.extract_features()`
- Remove or modify existing `reshape()` function
- Update input size calculation

---

## 🟢 Minor Assumptions (Good to Verify)

### 14. Feature Concatenation Order

**Location:** `extract_features()` method

```python
# Extract nonlocal first, then local
for var in self.nonlocal_vars: ...
for var in self.local_vars: ...
```

**Order:** `[nonlocal_var1_patches][nonlocal_var2_patches]...[local_var1][local_var2]...`

**Why it matters:** If you have existing trained models with different feature ordering, they won't be compatible.

**Recommendation:** Document the exact feature order for your trained models.

---

### 15. Compatible with Existing Training Loop

**Location:** Assumption about `train.py`, `eval.py` integration

**Assumption:** The output tensors work with existing `DataLoader`, optimizer, and training loop.

**What wasn't tested:**
- Integration with the existing `fit()` function
- Compatibility with WandB logging
- Integration with `eval.py` for predictions
- Checkpoint loading/saving with different input sizes

**Recommendation:** Start with a small test run before full training.

---

### 16. Batch Size Still Appropriate

**Location:** `configs.json` specifies `batchsize=2208`

**Assumption:** This batch size still makes sense with the new sample count.

**How to verify:**
```python
# After extracting features
n_samples = X.shape[0]
batch_size = 2208

if n_samples % batch_size != 0:
    print(f"Warning: {n_samples} samples doesn't divide evenly by {batch_size}")
    print(f"Last batch will have {n_samples % batch_size} samples")
```

**Consider:** Adjusting batch size if sample count changed significantly.

---

## Testing Checklist

Before using this code in production, test:

- [ ] **Dimension verification**: All variables have expected dimensions
- [ ] **Shape matching**: `X.shape[0] == y.shape[0]` after extraction
- [ ] **Feature count**: Manual calculation matches `X.shape[1]`
- [ ] **No NaNs**: Check extracted features for NaN values
- [ ] **Memory usage**: Monitor RAM during feature extraction
- [ ] **Small-scale test**: Extract patches from 10 timesteps first
- [ ] **Coordinate alignment**: All variables on same grid
- [ ] **Boundary effects**: Visualize edge samples to check validity
- [ ] **Training integration**: Run 1 epoch with new features
- [ ] **Prediction**: Verify `eval.py` works with new model

---

## Recommended First Steps

1. **Create a minimal test script:**
   ```python
   # test_patch_extraction.py
   import xarray as xr
   from model import BaselineNN

   # Load small subset
   ds = xr.open_dataset('normtrain.h5')
   subset = {
       'rh': ds['rh'].isel(time=slice(0,10)),
       'sdo': ds['sdo'],
       'pr': ds['pr'].isel(time=slice(0,10))
   }

   # Test extraction
   model = BaselineNN(
       nonlocal_vars=['rh'],
       local_vars=['sdo'],
       spatial_window=(1,1),
       temporal_window=2
   )

   X = model.extract_features(subset)
   print(f"Success! X.shape = {X.shape}")
   ```

2. **Verify assumptions systematically** using the checks in this document

3. **Test with full data** once small-scale tests pass

4. **Integrate with training** incrementally

---

## Questions to Discuss

1. Should local variables support 4D (vertical profiles at x₀,t₀)?
2. Is `boundary='valid'` acceptable or do you need all samples?
3. Should we add automatic NaN handling or require clean data?
4. Should we add a helper method for target extraction?
5. Do you want periodic boundary conditions for longitude?
6. Should we implement batch-based extraction for large datasets?

---

## Contact

If you find issues or have questions about these assumptions, please reach out. This is meant as a starting point for discussion and refinement.
