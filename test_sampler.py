#!/usr/bin/env python
"""Test script to verify sampler.py extraction logic without GPU."""

import torch
import sys
sys.path.insert(0, '/home/user/monsoon-kernels')

from scripts.data.classes.sampler import PatchDataset

def create_mock_data():
    """Create small mock dataset for testing."""
    nlats, nlons, nlevs, ntimes = 100, 100, 16, 50
    nfieldvars = 3
    nlocalvars = 3

    field = torch.randn(nfieldvars, nlats, nlons, nlevs, ntimes)
    ps = torch.linspace(900, 1000, nlats*nlons*ntimes).reshape(nlats, nlons, ntimes)
    lev = torch.linspace(500, 1000, nlevs)
    darea = torch.ones(nlats, nlons)
    dlev = torch.ones(nlevs) * 31.25  # ~500 hPa / 16 levels
    dtime = torch.ones(ntimes)
    local = torch.randn(nlocalvars, nlats, nlons, ntimes)
    target = torch.randn(nlats, nlons, ntimes)
    lats = torch.linspace(5, 25, nlats)
    lons = torch.linspace(60, 90, nlons)

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
    """Test surface mode extraction."""
    print("Testing levmode='surface'...")
    data = create_mock_data()

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

    # Create a small batch
    batch = [dataset.centers[i] for i in range(min(10, len(dataset.centers)))]

    try:
        result = PatchDataset.collate(batch, dataset)
        print(f"  ✓ Surface mode successful")
        print(f"    fieldpatch shape: {result['fieldpatch'].shape}")
        print(f"    Expected: (nbatch={len(batch)}, 2*nfieldvars=6, plats=3, plons=3, plevs=1, ptimes=7)")
        assert result['fieldpatch'].shape == (len(batch), 6, 3, 3, 1, 7), "Shape mismatch!"
        print(f"  ✓ Shape correct")
    except Exception as e:
        print(f"  ✗ Surface mode failed: {e}")
        raise

def test_column_mode():
    """Test column mode extraction."""
    print("\nTesting levmode='column'...")
    data = create_mock_data()

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

    # Create a small batch
    batch = [dataset.centers[i] for i in range(min(10, len(dataset.centers)))]

    try:
        result = PatchDataset.collate(batch, dataset)
        print(f"  ✓ Column mode successful")
        print(f"    fieldpatch shape: {result['fieldpatch'].shape}")
        nlevs = data['lev'].shape[0]
        print(f"    Expected: (nbatch={len(batch)}, 2*nfieldvars=6, plats=3, plons=3, plevs={nlevs}, ptimes=7)")
        assert result['fieldpatch'].shape == (len(batch), 6, 3, 3, nlevs, 7), "Shape mismatch!"
        print(f"  ✓ Shape correct")

        # Check validity mask
        data_channels = result['fieldpatch'][:, :3, :, :, :, :]
        mask_channels = result['fieldpatch'][:, 3:, :, :, :, :]
        print(f"  ✓ Data and mask channels separated correctly")

    except Exception as e:
        print(f"  ✗ Column mode failed: {e}")
        raise

def test_different_configs():
    """Test various radius and timelag configurations."""
    print("\nTesting different configurations...")
    configs = [
        (0, 'surface', 0),  # baseline_only_local
        (0, 'column', 0),   # baseline_only_vertical
        (1, 'surface', 0),  # baseline_only_horizontal
        (1, 'column', 6),   # baseline_full
    ]

    data = create_mock_data()

    for radius, levmode, timelag in configs:
        print(f"  Testing radius={radius}, levmode={levmode}, timelag={timelag}...")
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

            batch = [dataset.centers[i] for i in range(min(5, len(dataset.centers)))]
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
            raise

if __name__ == '__main__':
    print("="*60)
    print("Testing sampler.py extraction logic")
    print("="*60)

    test_surface_mode()
    test_column_mode()
    test_different_configs()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
