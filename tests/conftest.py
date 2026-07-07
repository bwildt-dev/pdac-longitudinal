"""Shared fixtures: tiny synthetic datasets that exercise the CPU pipeline.

No GPU / nnU-Net: segmentations are planted directly and loaded via the
`reuse_saved_segs` path, so the whole cache pipeline runs on CPU.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest


def _affine(spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    a = np.eye(4)
    for i, s in enumerate(spacing):
        a[i, i] = s
    return a


def _save_nii(arr, path, spacing=(1.0, 1.0, 1.0), dtype=np.float32) -> None:
    nib.save(nib.Nifti1Image(arr.astype(dtype), _affine(spacing)), str(path))


def synthetic_seg(shape=(48, 48, 48)) -> np.ndarray:
    """A PanTS-style label map (XYZ) with the organs the pipeline reads."""
    seg = np.zeros(shape, dtype=np.int16)
    seg[2:20, 2:20, 2:20]   = 13   # liver
    seg[20:40, 18:30, 18:30] = 1   # pancreas
    seg[26:34, 21:27, 21:27] = 2   # tumor (inside the pancreas region)
    seg[24:36, 20:22, 20:30] = 3   # superior mesenteric artery
    seg[24:36, 28:30, 20:30] = 5   # veins
    seg[10:22, 34:44, 10:22] = 14  # kidney_left
    seg[26:40, 34:44, 10:22] = 15  # kidney_right
    seg[30:44, 2:14, 30:44]  = 12  # spleen
    seg[4:16, 30:44, 30:44]  = 10  # stomach
    seg[34:44, 22:32, 20:30] = 9   # duodenum
    return seg


def synthetic_ct(shape=(48, 48, 48), seed=0) -> np.ndarray:
    """A soft-tissue-ish CT volume (~40 HU) with reproducible noise."""
    rng = np.random.default_rng(seed)
    return rng.normal(40.0, 20.0, shape).astype(np.float32)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Two-patient composer-layout dataset with pre-saved segs (no GPU needed).

    Returns a dict of paths + ids for building a `LongitudinalCTDataset`.
    """
    root = tmp_path / "nifti"
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    phase = "venous"
    pids = ["PAT-001", "PAT-002"]
    seg = synthetic_seg()

    rows = []
    for i, pid in enumerate(pids):
        for tp, tag in [("t0", "T0"), ("t1", "T1")]:
            d = root / phase / pid / tp
            d.mkdir(parents=True)
            _save_nii(synthetic_ct(seed=i * 2 + (tp == "t1")), d / f"{pid}_{tp}.nii.gz")
            _save_nii(seg, cache / f"{pid}_seg_{tag}.nii.gz", dtype=np.int16)
        rows.append(dict(
            patient_id=pid, cohort="cohort_a", time_months=10.0 + i, status=i % 2,
            age_diag=60 + i, bmi=25.0 + i,
        ))
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)

    return {
        "nifti_root": root, "cache_dir": cache, "labels_csv": labels,
        "pids": pids, "phase": phase,
    }


def big_kidney_seg(shape=(64, 64, 64), shift=0) -> np.ndarray:
    """Seg with kidneys large enough (>8000 vox union) to drive kidney alignment.

    `shift` translates the whole map along x to create a small, in-threshold
    T0->T1 centroid drift.
    """
    seg = np.zeros(shape, dtype=np.int16)
    s = shift
    seg[24 + s:44 + s, 24:40, 24:40] = 1    # pancreas
    seg[4 + s:24 + s, 4:20, 40:60]   = 13   # liver
    seg[8 + s:30 + s, 40:58, 8:30]   = 14   # kidney_left  (~8700 vox)
    seg[34 + s:56 + s, 40:58, 8:30]  = 15   # kidney_right (~8700 vox)
    seg[30 + s:36 + s, 30:36, 30:36] = 2    # tumor (planted last, inside pancreas)
    return seg


@pytest.fixture
def aligned_dataset(tmp_path):
    """One-patient dataset whose big, slightly-shifted kidneys trigger the
    shared-frame kidney-centroid alignment path in `_align_and_crop_t1`."""
    root = tmp_path / "nifti"
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    phase = "venous"
    pid = "PAT-100"
    for tp, tag, shift in [("t0", "T0", 0), ("t1", "T1", 2)]:
        d = root / phase / pid / tp
        d.mkdir(parents=True)
        _save_nii(synthetic_ct((64, 64, 64), seed=(tp == "t1")), d / f"{pid}_{tp}.nii.gz")
        _save_nii(big_kidney_seg(shift=shift), cache / f"{pid}_seg_{tag}.nii.gz", dtype=np.int16)
    labels = tmp_path / "labels.csv"
    pd.DataFrame([dict(patient_id=pid, cohort="cohort_a", time_months=12.0, status=1,
                       age_diag=65, bmi=26.0)]).to_csv(labels, index=False)
    return {"nifti_root": root, "cache_dir": cache, "labels_csv": labels,
            "pid": pid, "phase": phase}


def pancreas_only_seg(shape=(64, 64, 64), shift=0) -> np.ndarray:
    """Seg with a >3000-voxel pancreas and no kidneys, so alignment falls
    back to the pancreas centroid once kidneys are unavailable."""
    seg = np.zeros(shape, dtype=np.int16)
    s = shift
    seg[20 + s:40 + s, 20:36, 20:36] = 1   # pancreas, 20*16*16=5120 vox
    seg[26 + s:32 + s, 26:32, 26:32] = 2   # tumor inside the pancreas
    return seg


@pytest.fixture
def pancreas_aligned_dataset(tmp_path):
    """One-patient dataset with no kidneys but a big, slightly-shifted
    pancreas, to trigger the pancreas-centroid alignment fallback."""
    root = tmp_path / "nifti"
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    phase = "venous"
    pid = "PAT-200"
    for tp, tag, shift in [("t0", "T0", 0), ("t1", "T1", 3)]:
        d = root / phase / pid / tp
        d.mkdir(parents=True)
        _save_nii(synthetic_ct((64, 64, 64), seed=(tp == "t1")), d / f"{pid}_{tp}.nii.gz")
        _save_nii(pancreas_only_seg(shift=shift), cache / f"{pid}_seg_{tag}.nii.gz", dtype=np.int16)
    labels = tmp_path / "labels.csv"
    pd.DataFrame([dict(patient_id=pid, cohort="cohort_a", time_months=12.0, status=1,
                       age_diag=65, bmi=26.0)]).to_csv(labels, index=False)
    return {"nifti_root": root, "cache_dir": cache, "labels_csv": labels,
            "pid": pid, "phase": phase}


@pytest.fixture
def projection_fallback_dataset(tmp_path):
    """T0 has a tumour and nothing else; T1's seg is empty. No kidney or
    pancreas anchor exists at either timepoint, so `_align_and_crop_t1`
    must fall through to the world-space projection of the T0 anchor."""
    root = tmp_path / "nifti"
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    phase = "venous"
    pid = "PAT-300"
    shape = (64, 64, 64)
    t0_seg = np.zeros(shape, dtype=np.int16)
    t0_seg[28:36, 28:36, 28:36] = 2   # tumor only, no pancreas/kidneys
    t1_seg = np.zeros(shape, dtype=np.int16)  # fully empty: PanTS found nothing
    for tp, tag, seg in [("t0", "T0", t0_seg), ("t1", "T1", t1_seg)]:
        d = root / phase / pid / tp
        d.mkdir(parents=True)
        _save_nii(synthetic_ct(shape, seed=(tp == "t1")), d / f"{pid}_{tp}.nii.gz")
        _save_nii(seg, cache / f"{pid}_seg_{tag}.nii.gz", dtype=np.int16)
    labels = tmp_path / "labels.csv"
    pd.DataFrame([dict(patient_id=pid, cohort="cohort_a", time_months=12.0, status=1,
                       age_diag=65, bmi=26.0)]).to_csv(labels, index=False)
    return {"nifti_root": root, "cache_dir": cache, "labels_csv": labels,
            "pid": pid, "phase": phase}
