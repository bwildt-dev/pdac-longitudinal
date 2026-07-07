"""Characterization tests for the CPU cache pipeline (CachePipelineMixin).

These protect the decomposition of `_preprocess_and_cache` into phase methods:
the pipeline must run end to end on a synthetic seg, emit the exact cache
contract, and be deterministic.
"""

from __future__ import annotations

import logging

import numpy as np

from pdac_longitudinal.data.longitudinal_dataset import LongitudinalCTDataset
from pdac_longitudinal.data.registry import ClinicalRegistry

PATCH = (32, 32, 32)

EXPECTED_KEYS = {
    "t0", "t1",
    "mask_it", "mask_it_t1",
    "liver_t0", "liver_t1", "pancreas_t0", "pancreas_t1", "kidneys_t0", "kidneys_t1",
    "mask_pt1", "mask_pt2", "mask_pt3", "mask_tvi",
    "mask_pt1_t1", "mask_pt2_t1", "mask_pt3_t1", "mask_tvi_t1",
    "valid_t0", "valid_t1", "phase_used",
    "anatomy_features_json", "vessel_features_json",
}

_UINT8_KEYS = EXPECTED_KEYS - {"t0", "t1", "phase_used",
                               "anatomy_features_json", "vessel_features_json"}


def _make_ds(cfg, **kw):
    registry = ClinicalRegistry(cfg["labels_csv"])
    return LongitudinalCTDataset(
        nifti_root=cfg["nifti_root"], registry=registry, cache_dir=cfg["cache_dir"],
        weights_path=None, phase=cfg["phase"],
        allowed_regions=["abdomen"], post_nat_tps=["t1"],
        patch_size=PATCH, target_spacing_mm=(1.5, 1.5, 1.5),
        foreground_margin_voxels=4, reuse_saved_segs=True, **kw,
    )


def test_discovery_finds_both_patients(synthetic_dataset):
    ds = _make_ds(synthetic_dataset)
    assert len(ds) == 2
    assert {c["patient_id"] for c in ds.cases} == set(synthetic_dataset["pids"])


def test_pipeline_contract_and_shapes(synthetic_dataset):
    ds = _make_ds(synthetic_dataset)
    arrays = ds._preprocess_and_cache(ds.cases[0])

    assert set(arrays) == EXPECTED_KEYS
    for k in ("t0", "t1"):
        assert arrays[k].shape == PATCH
        assert arrays[k].dtype == np.float32
    for k in _UINT8_KEYS:
        assert arrays[k].shape == PATCH, k
        assert arrays[k].dtype == np.uint8, k
    # valid masks are all-or-part real content, never empty for an in-bounds patch.
    assert arrays["valid_t0"].any()
    # the planted tumour survives to the cached IT mask.
    assert arrays["mask_it"].sum() > 0
    # peritumoural rings are concentric with (and disjoint from) IT.
    assert not np.any(arrays["mask_it"] & arrays["mask_pt1"])


def test_pipeline_is_deterministic(synthetic_dataset):
    ds = _make_ds(synthetic_dataset)
    a = ds._preprocess_and_cache(ds.cases[0])
    b = ds._preprocess_and_cache(ds.cases[0])
    for k in EXPECTED_KEYS:
        assert np.array_equal(a[k], b[k]), f"{k} differs between runs"


def test_cache_roundtrip_matches_recompute(synthetic_dataset):
    """The .npz written by preprocessing reloads to the same arrays (minus skipped)."""
    ds = _make_ds(synthetic_dataset)
    case = ds.cases[0]
    fresh = ds._preprocess_and_cache(case)          # writes the .npz
    loaded = ds._load_case(case)                    # reads it back
    for k in loaded:                                # _SKIP_ON_LOAD keys are absent
        assert np.array_equal(fresh[k], loaded[k]), k
    assert set(loaded) == EXPECTED_KEYS - ds._SKIP_ON_LOAD


def test_kidney_centroid_alignment_path(aligned_dataset, caplog):
    """The shared-frame kidney-alignment branch of _align_and_crop_t1 runs."""
    registry = ClinicalRegistry(aligned_dataset["labels_csv"])
    ds = LongitudinalCTDataset(
        nifti_root=aligned_dataset["nifti_root"], registry=registry,
        cache_dir=aligned_dataset["cache_dir"], weights_path=None,
        phase=aligned_dataset["phase"],
        allowed_regions=["abdomen"], post_nat_tps=["t1"],
        patch_size=(48, 48, 48), target_spacing_mm=(1.0, 1.0, 1.0),
        foreground_margin_voxels=6, shared_crop_frame=True, reuse_saved_segs=True,
    )
    with caplog.at_level(logging.INFO):
        arrays = ds._preprocess_and_cache(ds.cases[0])
    text = caplog.text.lower()
    assert "kidney" in text and "aligned" in text     # kidney-alignment branch taken
    assert arrays["mask_it"].sum() > 0
    assert arrays["t1"].shape == (48, 48, 48)


def _make_shared_frame_ds(cfg, pid):
    registry = ClinicalRegistry(cfg["labels_csv"])
    return LongitudinalCTDataset(
        nifti_root=cfg["nifti_root"], registry=registry, cache_dir=cfg["cache_dir"],
        weights_path=None, phase=cfg["phase"],
        allowed_regions=["abdomen"], post_nat_tps=["t1"],
        patch_size=(48, 48, 48), target_spacing_mm=(1.0, 1.0, 1.0),
        foreground_margin_voxels=6, shared_crop_frame=True, reuse_saved_segs=True,
    )


def test_pancreas_centroid_alignment_fallback(pancreas_aligned_dataset, caplog):
    """No kidneys: alignment falls back to the pancreas centroid."""
    ds = _make_shared_frame_ds(pancreas_aligned_dataset, pancreas_aligned_dataset["pid"])
    with caplog.at_level(logging.INFO):
        arrays = ds._preprocess_and_cache(ds.cases[0])
    text = caplog.text.lower()
    assert "pancreas" in text and "aligned" in text
    assert "kidney union too small" in text  # confirms kidney path was tried and skipped first
    assert arrays["mask_it"].sum() > 0


def test_world_space_projection_fallback(projection_fallback_dataset, caplog):
    """No shared anatomical anchor survives at T1: falls back to projecting
    the T0 anchor into T1's voxel grid via world-space coordinates."""
    ds = _make_shared_frame_ds(projection_fallback_dataset, projection_fallback_dataset["pid"])
    with caplog.at_level(logging.DEBUG):
        arrays = ds._preprocess_and_cache(ds.cases[0])
    text = caplog.text.lower()
    assert "projected" in text
    assert arrays["mask_it"].sum() > 0        # T0 tumour still cached
    assert arrays["mask_it_t1"].sum() == 0    # T1 truly had no tumour
    assert arrays["t1"].shape == (48, 48, 48)


def test_getitem_returns_tensors(synthetic_dataset):
    ds = _make_ds(synthetic_dataset)
    sample = ds[0]
    assert sample["t0"].shape == (1, *PATCH)
    assert sample["t1"].shape == (1, *PATCH)
    assert sample["case_id"] == "PAT-001"
    for k in ("clinical", "anatomy", "vessel", "radiomic", "duration", "event", "phase"):
        assert k in sample
