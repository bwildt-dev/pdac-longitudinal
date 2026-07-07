"""Unit tests for PT ring / TVI geometry (pdac_longitudinal/data/roi_rings.py)."""

from __future__ import annotations

import numpy as np
import pytest

from pdac_longitudinal.preprocess.roi_rings import create_pt_rings, create_tvi_mask


def _sphere(shape, center, radius):
    zz, yy, xx = np.indices(shape)
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2)
    return d <= radius


def test_rings_are_concentric_and_disjoint_from_it():
    it = _sphere((40, 40, 40), (20, 20, 20), 5)
    rings = create_pt_rings(it, voxel_spacing_mm=(1.0, 1.0, 1.0), ring_radii_mm=(0.0, 5.0, 10.0, 15.0))
    assert len(rings) == 3
    for r in rings:
        assert not np.any(r & it)          # rings never overlap IT
    for r1, r2 in zip(rings, rings[1:]):
        assert not np.any(r1 & r2)         # rings are pairwise disjoint
    # farther rings sit strictly farther from the IT surface on average
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~it)
    means = [dist[r].mean() for r in rings if r.any()]
    assert means == sorted(means)


def test_rings_reject_bad_radii():
    it = _sphere((10, 10, 10), (5, 5, 5), 2)
    with pytest.raises(ValueError):
        create_pt_rings(it, (1.0, 1.0, 1.0), ring_radii_mm=(1.0, 5.0))  # doesn't start at 0
    with pytest.raises(ValueError):
        create_pt_rings(it, (1.0, 1.0, 1.0), ring_radii_mm=(0.0, 5.0, 5.0))  # not strictly increasing


def test_tvi_requires_proximity_to_both_tumour_and_vessel():
    shape = (60, 60, 60)
    it = _sphere(shape, (10, 30, 30), 3)
    near_vessel = _sphere(shape, (14, 30, 30), 3)   # ~4 vox from IT surface: within both zones
    far_vessel = _sphere(shape, (50, 30, 30), 3)    # far from IT: outside tumour proximity

    tvi_near = create_tvi_mask(it, near_vessel, (1.0, 1.0, 1.0), tvi_tumour_mm=10.0, tvi_vessel_mm=10.0)
    assert tvi_near.any()
    assert not np.any(tvi_near & it)  # excludes IT itself

    with pytest.warns(UserWarning):
        tvi_far = create_tvi_mask(it, far_vessel, (1.0, 1.0, 1.0), tvi_tumour_mm=10.0, tvi_vessel_mm=10.0)
    assert not tvi_far.any()


def test_tvi_empty_without_vessels_warns():
    it = _sphere((20, 20, 20), (10, 10, 10), 3)
    empty_vessel = np.zeros_like(it)
    with pytest.warns(UserWarning):
        tvi = create_tvi_mask(it, empty_vessel, (1.0, 1.0, 1.0))
    assert not tvi.any()
