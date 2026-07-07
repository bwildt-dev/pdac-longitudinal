"""Unit tests for the numeric preprocessing backbone (pure CPU functions)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from pdac_longitudinal.preprocess.preprocessing import (
    compute_foreground_bbox,
    crop_volume,
    normalise_ct,
    pad_or_crop_to_patch_size,
    resample_volume,
)


def test_bbox_locates_blob_with_margin():
    m = np.zeros((20, 20, 20), dtype=np.uint8)
    m[5:10, 6:8, 7:12] = 1
    start, end = compute_foreground_bbox(m, margin_voxels=0)
    assert start == (5, 6, 7)
    assert end == (10, 8, 12)          # exclusive
    s2, e2 = compute_foreground_bbox(m, margin_voxels=3)
    assert s2 == (2, 3, 4)
    assert e2 == (13, 11, 15)


def test_bbox_clamps_and_handles_empty():
    m = np.zeros((8, 8, 8), dtype=np.uint8)
    m[0, 0, 0] = 1
    start, end = compute_foreground_bbox(m, margin_voxels=5)
    assert start == (0, 0, 0)           # clamped at the low edge
    assert end == (6, 6, 6)
    assert compute_foreground_bbox(np.zeros((4, 4, 4))) == ((0, 0, 0), (4, 4, 4))


def test_crop_matches_bbox():
    a = np.arange(4 * 4 * 4).reshape(4, 4, 4)
    out = crop_volume(a, (1, 1, 1), (3, 3, 3))
    assert out.shape == (2, 2, 2)
    assert np.array_equal(out, a[1:3, 1:3, 1:3])


def test_pad_and_crop_to_patch_size():
    small = np.ones((10, 12, 8), dtype=np.float32)
    padded = pad_or_crop_to_patch_size(small, (16, 16, 16), constant_value=0.0)
    assert padded.shape == (16, 16, 16)
    assert padded.sum() == small.sum()          # zero-pad preserves content
    assert padded[8, 8, 8] == 1.0               # original block centred
    filled = pad_or_crop_to_patch_size(small, (16, 16, 16), constant_value=-1.0)
    assert filled[0, 0, 0] == -1.0              # corner padded with the fill value

    big = np.ones((40, 40, 40), dtype=np.float32)
    cropped = pad_or_crop_to_patch_size(big, (16, 16, 16))
    assert cropped.shape == (16, 16, 16)


def test_normalise_fixed_stats():
    arr = np.array([[[0.0, 100.0]]], dtype=np.float32)
    out = normalise_ct(arr, clip_min=-100, clip_max=200, mean=50.0, std=10.0)
    assert np.allclose(out, [[[-5.0, 5.0]]])


def test_normalise_requires_both_mean_and_std():
    with pytest.raises(ValueError):
        normalise_ct(np.zeros((2, 2, 2)), -100, 200, mean=0.0)


def test_normalise_foreground_zscore_is_standardised():
    rng = np.random.default_rng(0)
    arr = rng.normal(40, 20, (8, 8, 8)).astype(np.float32)
    out = normalise_ct(arr, clip_min=-100, clip_max=200)
    fg = out[arr > -100]
    assert abs(float(fg.mean())) < 1e-4
    assert abs(float(fg.std()) - 1.0) < 1e-4


def test_resample_identity_and_downsample():
    data = np.random.default_rng(0).normal(0, 1, (20, 20, 20)).astype(np.float32)
    img = nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0]))
    same, sp = resample_volume(img, (1.0, 1.0, 1.0), is_mask=False)
    assert same.shape == (20, 20, 20)
    assert np.allclose(sp, (1.0, 1.0, 1.0))
    coarse, _ = resample_volume(img, (2.0, 2.0, 2.0), is_mask=False)
    assert coarse.shape == (10, 10, 10)


def test_resample_mask_stays_binary():
    m = np.zeros((16, 16, 16), dtype=np.float32)
    m[4:12, 4:12, 4:12] = 1.0
    img = nib.Nifti1Image(m, np.diag([1.0, 1.0, 1.0, 1.0]))
    out, _ = resample_volume(img, (2.0, 2.0, 2.0), is_mask=True)
    assert set(np.unique(out)).issubset({0.0, 1.0})   # NN interp, no blur
