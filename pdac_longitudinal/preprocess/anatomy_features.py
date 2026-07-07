"""Scalar anatomy features derived from segmentations."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from pdac_longitudinal.preprocess.segmenter import PANTS_LABELS, BILIARY_LABELS

logger = logging.getLogger(__name__)



_PER_TP_ORGANS = ("liver", "spleen", "pancreas", "tumor")


def anatomy_feature_columns(include_hu: bool = True) -> Tuple[str, ...]:
    """Canonical column order for the anatomy feature vector.

    Args:
        include_hu: Whether to include per-organ mean/std HU columns.
    """
    cols: list[str] = []
    for organ in _PER_TP_ORGANS:
        cols.append(f"{organ}_t0_mL")
        if include_hu:
            cols += [f"{organ}_t0_mean_HU", f"{organ}_t0_std_HU"]
        cols.append(f"{organ}_t1_mL")
        if include_hu:
            cols += [f"{organ}_t1_mean_HU", f"{organ}_t1_std_HU"]
        cols.append(f"delta_{organ}_pct")
    cols += ["biliary_t0_mL", "biliary_t1_mL", "pancreas_atrophy"]
    return tuple(cols)


ANATOMY_FEATURE_COLS = anatomy_feature_columns(include_hu=True)
ANATOMY_FEATURE_DIM  = len(ANATOMY_FEATURE_COLS)


def features_dict_to_vector(
    feats: Dict[str, float],
    columns: Sequence[str] = ANATOMY_FEATURE_COLS,
    fill: float = 0.0,
) -> np.ndarray:
    """Map a feature dict to a fixed-order float32 numpy vector.

    Args:
        feats: Feature name -> value mapping.
        columns: Column order to emit; missing or non-finite entries fall
            back to `fill`.
        fill: Value used for missing/non-finite entries.

    Returns:
        A float32 array of length `len(columns)`.
    """
    out = np.full(len(columns), fill, dtype=np.float32)
    for i, col in enumerate(columns):
        v = feats.get(col, fill)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            v = fill
        out[i] = float(v)
    return out


def _organ_voxels(seg: np.ndarray, label: int) -> int:
    return int((seg == label).sum())


def _organ_hu_stats(
    ct: np.ndarray, seg: np.ndarray, label: int,
) -> Tuple[float, float]:
    """Mean and std of CT intensity within an organ mask.

    Args:
        ct: CT intensity volume.
        seg: Label volume, same shape as `ct`.
        label: Organ label value to mask on.

    Returns:
        A `(mean, std)` tuple, or `(nan, nan)` if the mask is empty.
    """
    mask = (seg == label)
    if mask.sum() == 0 or ct.shape != seg.shape:
        return float("nan"), float("nan")
    vals = ct[mask]
    return float(vals.mean()), float(vals.std())


class AnatomyFeatureExtractor:
    """Derive scalar anatomy features from PanTS segmentations."""

    def __init__(self) -> None:
        self.labels = PANTS_LABELS


    def extract_one_from_arrays(
        self,
        seg: np.ndarray,
        ct: Optional[np.ndarray],
        vox_mm3: float,
        suffix: str = "",
    ) -> Dict[str, float]:
        """Per-organ volumes (mL) + optional HU stats from in-memory arrays.

        Args:
            seg: PanTS label volume.
            ct: CT intensity volume for HU stats; skipped if `None` or if
                its shape doesn't match `seg`.
            vox_mm3: Voxel volume in mm³.
            suffix: Appended to each output key.

        Returns:
            Feature name -> value mapping.
        """
        if ct is not None and ct.shape != seg.shape:
            logger.debug(
                "Shape mismatch ct %s vs seg %s — HU stats skipped", ct.shape, seg.shape,
            )
            ct = None

        feats: Dict[str, float] = {}
        for organ in _PER_TP_ORGANS:
            label = self.labels[organ]
            n_vox = _organ_voxels(seg, label)
            feats[f"{organ}{suffix}_mL"] = round(n_vox * vox_mm3 / 1000.0, 2)
            if ct is not None:
                mu, sd = _organ_hu_stats(ct, seg, label)
                feats[f"{organ}{suffix}_mean_HU"] = round(mu, 2) if np.isfinite(mu) else float("nan")
                feats[f"{organ}{suffix}_std_HU"]  = round(sd, 2) if np.isfinite(sd) else float("nan")

        biliary_voxels = sum(
            _organ_voxels(seg, lab) for lab in BILIARY_LABELS.values()
        )
        feats[f"biliary{suffix}_mL"] = round(biliary_voxels * vox_mm3 / 1000.0, 2)

        return feats


    @staticmethod
    def _compute_pair_derivatives(feats: Dict[str, float]) -> Dict[str, float]:
        """Deltas + pancreas-atrophy derived from already-computed T0/T1 columns.

        Args:
            feats: Feature dict containing the per-timepoint `_t0`/`_t1` columns.

        Returns:
            The derived `delta_*_pct` and `pancreas_atrophy` columns.
        """
        derived: Dict[str, float] = {}
        for organ in _PER_TP_ORGANS:
            v0 = feats.get(f"{organ}_t0_mL", 0.0)
            v1 = feats.get(f"{organ}_t1_mL", 0.0)
            derived[f"delta_{organ}_pct"] = (
                round((v1 - v0) / v0, 3) if v0 > 0 else float("nan")
            )
        v0p = feats.get("pancreas_t0_mL", 0.0)
        v1p = feats.get("pancreas_t1_mL", 0.0)
        derived["pancreas_atrophy"] = (
            round(1.0 - v1p / v0p, 3) if v0p > 0 else float("nan")
        )
        return derived
