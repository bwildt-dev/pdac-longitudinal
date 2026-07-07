"""Unit tests for pdac_longitudinal/data/mask_utils.py."""

from __future__ import annotations

import numpy as np

from pdac_longitudinal.preprocess.mask_utils import (
    kidney_centroids,
    largest_cc,
    pancreas_anatomy_mask,
)
from pdac_longitudinal.preprocess.segmenter import PANTS_LABELS


def test_largest_cc_keeps_only_biggest_blob():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[0:5, 0:5, 0:5] = True      # 125 vox, the big blob
    mask[15:17, 15:17, 15:17] = True  # 8 vox, small blob
    out = largest_cc(mask)
    assert out.sum() == 125
    assert not out[15, 15, 15]


def test_largest_cc_empty_and_single_component_are_noops():
    empty = np.zeros((5, 5, 5), dtype=bool)
    assert not largest_cc(empty).any()
    single = np.zeros((5, 5, 5), dtype=bool)
    single[1:3, 1:3, 1:3] = True
    assert np.array_equal(largest_cc(single), single)


def test_kidney_centroids_below_threshold_returns_none():
    seg = np.zeros((30, 30, 30), dtype=np.int16)
    seg[0:2, 0:2, 0:2] = PANTS_LABELS["kidney_left"]   # 8 vox, far below 4000
    seg[0:2, 0:2, 28:30] = PANTS_LABELS["kidney_right"]
    cl, cr, union = kidney_centroids(seg)
    assert cl is None and cr is None
    assert union.any()  # union mask still returned


def test_kidney_centroids_above_threshold_returns_centroids():
    seg = np.zeros((40, 40, 40), dtype=np.int16)
    seg[0:20, 0:20, 0:20] = PANTS_LABELS["kidney_left"]    # 8000 vox
    seg[20:40, 20:40, 0:20] = PANTS_LABELS["kidney_right"]  # 8000 vox
    cl, cr, union = kidney_centroids(seg)
    assert cl is not None and cr is not None
    np.testing.assert_allclose(cl, [9.5, 9.5, 9.5])
    np.testing.assert_allclose(cr, [29.5, 29.5, 9.5])


def test_pancreas_anatomy_mask_unions_parenchyma_and_tumor():
    seg = np.zeros((10, 10, 10), dtype=np.int16)
    seg[0:3, 0:3, 0:3] = PANTS_LABELS["pancreas"]
    seg[5:7, 5:7, 5:7] = PANTS_LABELS["tumor"]
    seg[8:9, 8:9, 8:9] = PANTS_LABELS["liver"]  # must not be included
    mask = pancreas_anatomy_mask(seg)
    assert mask.sum() == 27 + 8
    assert not mask[8, 8, 8]
