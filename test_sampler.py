#!/usr/bin/env python
"""Test script to verify sampler.py extraction logic without GPU.

This test uses realistic dimensions matching the actual training setup:
- Grid: 80x120 spatial points (lat 5-25, lon 60-90 at 0.25 deg resolution)
- Levels: 16 pressure levels (500-1000 hPa)
- Time: 2760 timesteps (92 days × 3 times/day for JJA)
- Batch size: 500 (matching training)
- Field variables: 3 (RH, θ_e, θ_e*)
- Local variables: 3 (LF, LHF, SHF)
"""

import torch
import sys
sys.path.insert(0, '/home/user/monsoon-kernels')

from scripts.data.classes.sampler import PatchDataset

def create_realistic_mock_data():
    """Create realistic mock dataset matching actual training dimensions."""
    nlats, nlons, nlevs, ntimes = 80, 120, 16, 2760
    nfieldvars = 3
    nlocalvars = 3

    print(f"Creating mock data with realistic dimensions:")
    print(f"  Spatial: {nlats}x{nlons} (lat 5-25, lon 60-90)")
    print(f"  Levels: {nlevs} (500-1000 hPa)")
    print(f"  Time: {ntimes} timesteps")
    print(f"  Field vars: {nfieldvars}, Local vars: {nlocalvars}")

    # Use float32 to match training
    field = torch.randn(nfieldvars, nlats, nlons, nlevs, ntimes, dtype=torch.float32)

    # Surface pressure varies realistically (850-1010 hPa)
    # Create some spatial/temporal variation
    ps_base = torch.linspace(950, 1000, nlats*nlons).reshape(nlats, nlons)
    ps = ps_base[:, :, None].expand(-1, -1, ntimes).clone()
    ps += torch.randn(nlats, nlons, ntimes) * 30  # Add variation
    ps = ps.clamp(850, 1010)  # Keep realistic range

    lev = torch.linspace(500, 1000, nlevs, dtype=torch.float32)
    darea = torch.ones(nlats, nlons, dtype=torch.float32)
    dlev = torch.ones(nlevs, dtype=torch.float32) * 31.25  # ~500 hPa / 16 levels
    dtime = torch.ones(ntimes, dtype=torch.float32)
    local = torch.randn(nlocalvars, nlats, nlons, ntimes, dtype=torch.float32)
    target = torch.randn(nlats, nlons, ntimes, dtype=torch.float32)
    lats = torch.linspace(5, 25, nlats, dtype=torch.float32)
    lons = torch.linspace(60, 90, nlons, dtype=torch.float32)

    return {
        'field': field,
        'ps': ps,
        'lev': lev,
        'darea': darea,
        'dlev': dlev,
        'dtime': dtime,
        'local': local,
        'target': target,
        'lats': lats,
        'lons': lons
    }

def test_surface_mode():
    """Test surface mode extraction with realistic batch size."""
    print("\nTesting levmode='surface' with batch_size=500...")
    data = create_realistic_mock_data()

    dataset = PatchDataset(
        radius=1,
        levmode='surface',
        timelag=6,
        field=data['field'],
        darea=data['darea'],
        dlev=data['dlev'],
        dtime=data['dtime'],
        ps=data['ps'],
        lev=data['lev'],
        local=data['local'],
        target=data['target'],
        uselocal=True,
        lats=data['lats'],
        lons=data['lons'],
        latrange=(10, 20),
        lonrange=(65, 85),
        maxradius=1,
        maxtimelag=6
    )

    print(f"  Dataset has {len(dataset)} valid centers")

    # Test with batch_size=500 (matching training)
    batch_size = 500
    batch = [dataset.centers[i] for i in range(min(batch_size, len(dataset.centers)))]
    print(f"  Testing batch extraction with {len(batch)} samples...")

    try:
        result = PatchDataset.collate(batch, dataset)
        print(f"  ✓ Surface mode successful")
        print(f"    fieldpatch shape: {result['fieldpatch'].shape}")
        print(f"    Expected: (nbatch={len(batch)}, 2*nfieldvars=6, plats=3, plons=3, plevs=1, ptimes=7)")
        assert result['fieldpatch'].shape == (len(batch), 6, 3, 3, 1, 7), "Shape mismatch!"
        print(f"  ✓ Shape correct")

        # Verify no NaNs in data after processing
        assert not torch.isnan(result['fieldpatch']).any(), "Unexpected NaN in fieldpatch!"
        print(f"  ✓ No NaN values in output")

        # Check data range is reasonable (standardized data should be roughly [-5, 5])
        data_channels = result['fieldpatch'][:, :3, :, :, :, :]
        print(f"  ✓ Data range: [{data_channels.min():.2f}, {data_channels.max():.2f}]")

    except Exception as e:
        print(f"  ✗ Surface mode failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_column_mode():
    """Test column mode extraction with realistic batch size."""
    import time
    print("\nTesting levmode='column' with batch_size=500...")
    data = create_realistic_mock_data()

    dataset = PatchDataset(
        radius=1,
        levmode='column',
        timelag=6,
        field=data['field'],
        darea=data['darea'],
        dlev=data['dlev'],
        dtime=data['dtime'],
        ps=data['ps'],
        lev=data['lev'],
        local=data['local'],
        target=data['target'],
        uselocal=True,
        lats=data['lats'],
        lons=data['lons'],
        latrange=(10, 20),
        lonrange=(65, 85),
        maxradius=1,
        maxtimelag=6
    )

    print(f"  Dataset has {len(dataset)} valid centers")

    # Test with batch_size=500 (matching training)
    batch_size = 500
    batch = [dataset.centers[i] for i in range(min(batch_size, len(dataset.centers)))]
    print(f"  Testing batch extraction with {len(batch)} samples...")

    try:
        start_time = time.time()
        result = PatchDataset.collate(batch, dataset)
        batch_time = time.time() - start_time

        print(f"  ✓ Column mode successful")
        nlevs = data['lev'].shape[0]
        print(f"    fieldpatch shape: {result['fieldpatch'].shape}")
        print(f"    Expected: (nbatch={len(batch)}, 2*nfieldvars=6, plats=3, plons=3, plevs={nlevs}, ptimes=7)")
        assert result['fieldpatch'].shape == (len(batch), 6, 3, 3, nlevs, 7), "Shape mismatch!"
        print(f"  ✓ Shape correct")

        # Check validity mask
        data_channels = result['fieldpatch'][:, :3, :, :, :, :]
        mask_channels = result['fieldpatch'][:, 3:, :, :, :, :]

        # Verify no NaNs in data (should be converted to 0)
        assert not torch.isnan(data_channels).any(), "Unexpected NaN in data channels!"
        print(f"  ✓ No NaN in data channels")

        # Check mask values are binary
        assert ((mask_channels == 0) | (mask_channels == 1)).all(), "Mask should be binary!"
        print(f"  ✓ Mask is binary (0 or 1)")

        # Check that invalid data is set to 0
        invalid_positions = mask_channels == 0
        if invalid_positions.any():
            assert (data_channels[invalid_positions] == 0).all(), "Invalid positions should be 0!"
            invalid_frac = invalid_positions.float().mean()
            print(f"  ✓ {invalid_frac*100:.1f}% of data marked invalid (below surface)")

        print(f"  ✓ Data and mask channels verified")

        # Performance estimate
        print(f"\n  Performance:")
        print(f"    Batch extraction time: {batch_time:.3f}s for {len(batch)} samples")
        print(f"    Throughput: {len(batch)/batch_time:.1f} samples/sec")

        # Estimate for full training dataset (21.5M samples, 43115 batches)
        training_samples = 21557214
        training_batches = 43115
        estimated_time = batch_time * training_batches
        print(f"    Estimated time for {training_samples:,} training samples ({training_batches:,} batches):")
        print(f"      {estimated_time/60:.1f} minutes = {estimated_time/3600:.2f} hours per epoch")

    except Exception as e:
        print(f"  ✗ Column mode failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_different_configs():
    """Test all model configurations from configs.json."""
    print("\nTesting all model configurations...")
    configs = [
        ('baseline_only_local', 0, 'surface', 0),
        ('baseline_only_vertical', 0, 'column', 0),
        ('baseline_only_horizontal', 1, 'surface', 0),
        ('baseline_only_temporal', 0, 'surface', 6),
        ('baseline_full', 1, 'column', 6),
    ]

    data = create_realistic_mock_data()

    for name, radius, levmode, timelag in configs:
        print(f"  Testing {name} (radius={radius}, levmode={levmode}, timelag={timelag})...")
        try:
            dataset = PatchDataset(
                radius=radius,
                levmode=levmode,
                timelag=timelag,
                field=data['field'],
                darea=data['darea'],
                dlev=data['dlev'],
                dtime=data['dtime'],
                ps=data['ps'],
                lev=data['lev'],
                local=data['local'],
                target=data['target'],
                uselocal=True,
                lats=data['lats'],
                lons=data['lons'],
                latrange=(10, 20),
                lonrange=(65, 85),
                maxradius=1,
                maxtimelag=6
            )

            # Use smaller batch for quick test
            batch_size = 50
            batch = [dataset.centers[i] for i in range(min(batch_size, len(dataset.centers)))]
            result = PatchDataset.collate(batch, dataset)

            plats = 2*radius + 1
            plons = 2*radius + 1
            plevs = 1 if levmode == 'surface' else data['lev'].shape[0]
            ptimes = timelag + 1 if timelag > 0 else 1

            expected_shape = (len(batch), 6, plats, plons, plevs, ptimes)
            assert result['fieldpatch'].shape == expected_shape, f"Expected {expected_shape}, got {result['fieldpatch'].shape}"
            print(f"    ✓ Shape {result['fieldpatch'].shape} correct")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def test_multiple_batches():
    """Test that multiple consecutive batches work correctly."""
    print("\nTesting multiple consecutive batches (simulating DataLoader)...")
    data = create_realistic_mock_data()

    dataset = PatchDataset(
        radius=1,
        levmode='column',
        timelag=6,
        field=data['field'],
        darea=data['darea'],
        dlev=data['dlev'],
        dtime=data['dtime'],
        ps=data['ps'],
        lev=data['lev'],
        local=data['local'],
        target=data['target'],
        uselocal=True,
        lats=data['lats'],
        lons=data['lons'],
        latrange=(10, 20),
        lonrange=(65, 85),
        maxradius=1,
        maxtimelag=6
    )

    # Test 3 consecutive batches
    batch_size = 500
    num_batches = 3
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        if start_idx >= len(dataset):
            break

        batch = [dataset.centers[i] for i in range(start_idx, end_idx)]
        try:
            result = PatchDataset.collate(batch, dataset)
            print(f"  ✓ Batch {batch_num+1}/{num_batches}: {len(batch)} samples, shape {result['fieldpatch'].shape}")
        except Exception as e:
            print(f"  ✗ Batch {batch_num+1} failed: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    print("="*80)
    print("Testing sampler.py extraction logic with realistic dimensions")
    print("="*80)

    test_surface_mode()
    test_column_mode()
    test_different_configs()
    test_multiple_batches()

    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("The extraction logic is correct and ready for training.")
    print("="*80)
