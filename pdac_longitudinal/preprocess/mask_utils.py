"""Segmentation-mask helpers and cache feature decoders."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np


def decode_anatomy_features(arrays: Dict[str, Any]) -> Dict[str, float]:
    """Decode the JSON-encoded anatomy features stashed in a cache `.npz`.

    Args:
        arrays: Loaded `.npz` array mapping.

    Returns:
        The decoded feature dict, or `{}` if the key is absent or decoding fails.
    """
    buf = arrays.get("anatomy_features_json")
    if buf is None:
        return {}
    try:
        return json.loads(bytes(buf).decode("utf-8"))
    except Exception:
        return {}


def decode_vessel_features(arrays: Dict[str, Any]) -> Dict[str, float]:
    """Decode the JSON-encoded vessel features stashed in a cache `.npz`.

    Args:
        arrays: Loaded `.npz` array mapping.

    Returns:
        The decoded feature dict, or `{}` if the key is absent or decoding fails.
    """
    buf = arrays.get("vessel_features_json")
    if buf is None:
        return {}
    try:
        return json.loads(bytes(buf).decode("utf-8"))
    except Exception:
        return {}


def largest_cc(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask.

    Args:
        mask: Boolean array to filter.
    """
    if not mask.any():
        return mask
    from scipy.ndimage import label
    lab, n = label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0  # ignore background
    return lab == int(counts.argmax())


def kidney_centroids(
    seg: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Return the left/right kidney centroids and their union mask.

    Args:
        seg: PanTS label volume.

    Returns:
        A `(left_centroid, right_centroid, bilateral_union_mask)` tuple.
        The centroids are `None` when either kidney falls below the
        minimum-voxel confidence threshold (PanTS truncation guard).
    """
    from pdac_longitudinal.preprocess.segmenter import PANTS_LABELS
    _K_MIN_VOX = 4000
    left = largest_cc((seg == PANTS_LABELS["kidney_left"]).astype(bool))
    right = largest_cc((seg == PANTS_LABELS["kidney_right"]).astype(bool))
    if int(left.sum()) < _K_MIN_VOX or int(right.sum()) < _K_MIN_VOX:
        return None, None, left | right
    cl = np.argwhere(left).mean(axis=0).astype(np.float64)
    cr = np.argwhere(right).mean(axis=0).astype(np.float64)
    return cl, cr, (left | right)


def pancreas_anatomy_mask(seg: np.ndarray) -> np.ndarray:
    """Pancreas-anatomy mask: union of PanTS parenchyma (1) and tumour (2).

    Args:
        seg: PanTS label volume.
    """
    from pdac_longitudinal.preprocess.segmenter import PANTS_LABELS
    return (seg == PANTS_LABELS["pancreas"]) | (seg == PANTS_LABELS["tumor"])
