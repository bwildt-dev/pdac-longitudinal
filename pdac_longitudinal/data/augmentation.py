"""Training-time augmentation: one shared spatial warp per sample pair to preserve the longitudinal Δ."""

from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

import numpy as np

from pdac_longitudinal.config import AugmentationConfig


def augment_sample(
    arrays: Dict[str, np.ndarray], cfg: AugmentationConfig
) -> Dict[str, np.ndarray]:
    """Apply flip, spatial warp, and intensity aug to one sample dict.

    Args:
        arrays: Per-key volumes for one sample pair; 1-D JSON byte buffers
            pass through untouched.
        cfg: Augmentation probabilities and magnitudes.

    Returns:
        The augmented arrays dict (same keys, same shapes).
    """
    if random.random() < cfg.flip_x_prob:
        # Flip 3-D arrays only; the 1-D JSON byte buffers must not be reversed.
        arrays = {
            k: (np.flip(v, axis=0).copy() if v.ndim >= 3 else v)
            for k, v in arrays.items()
        }

    ref = arrays.get("t0")
    if ref is not None and ref.ndim >= 3:
        coords = _spatial_warp_coords(tuple(ref.shape[-3:]), cfg)
        if coords is not None:
            from scipy.ndimage import map_coordinates
            # Masks share one rounded index grid + out-of-bounds mask;
            # CT keeps trilinear interpolation.
            idx = np.round(coords).astype(np.intp)
            oob = np.zeros(coords.shape[1:], dtype=bool)
            for a in range(3):
                oob |= (idx[a] < 0) | (idx[a] >= ref.shape[a])
                np.clip(idx[a], 0, ref.shape[a] - 1, out=idx[a])
            i0, i1, i2 = idx[0], idx[1], idx[2]
            for k, v in arrays.items():
                if v.ndim < 3:
                    continue
                if k in ("t0", "t1"):
                    arrays[k] = map_coordinates(
                        v, coords, order=1, mode="constant", cval=0.0,
                    ).astype(v.dtype, copy=False)
                else:
                    w = v[i0, i1, i2]
                    w[oob] = 0
                    arrays[k] = w.astype(v.dtype, copy=False)

    if random.random() < cfg.intensity_prob:
        for key in ("t0", "t1"):
            if key in arrays:
                arrays[key] = _augment_intensity(arrays[key], cfg)

    return arrays


def _rotation_matrix(ax: float, ay: float, az: float) -> np.ndarray:
    """3×3 rotation matrix R = Rz·Ry·Rx for radians about (X, Y, Z).

    Args:
        ax: Rotation about the X axis, in radians.
        ay: Rotation about the Y axis, in radians.
        az: Rotation about the Z axis, in radians.
    """
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _spatial_warp_coords(
    shape: Tuple[int, int, int], cfg: AugmentationConfig
) -> Optional[np.ndarray]:
    """Sampling coordinates for a single rotation+scale+elastic warp.

    Args:
        shape: Spatial shape `(D, H, W)` of the volume to warp.
        cfg: Augmentation probabilities and magnitudes.

    Returns:
        A `(3, *shape)` sampling-coordinate array, or `None` if neither the
        affine nor the elastic warp was triggered this call.
    """
    do_affine = random.random() < cfg.spatial_prob
    do_elastic = random.random() < cfg.elastic_prob
    if not (do_affine or do_elastic):
        return None
    grid = np.indices(shape, dtype=np.float32)                      # (3, *shape)
    center = np.array([(s - 1) / 2.0 for s in shape],
                      dtype=np.float32).reshape(3, 1, 1, 1)
    coords = grid
    if do_affine:
        ax = math.radians(random.uniform(-cfg.tilt_deg, cfg.tilt_deg))
        ay = math.radians(random.uniform(-cfg.tilt_deg, cfg.tilt_deg))
        az = math.radians(random.uniform(-cfg.axial_deg, cfg.axial_deg))
        scale = random.uniform(*cfg.scale_range)
        # Sampling transform R/scale about the centre.
        M = _rotation_matrix(ax, ay, az) / scale
        flat = (coords - center).reshape(3, -1)
        coords = (M @ flat).astype(np.float32).reshape(3, *shape) + center
    if do_elastic:
        from scipy.ndimage import gaussian_filter, zoom
        coarse = tuple(max(4, s // 4) for s in shape)
        zoom_f = [s / c for s, c in zip(shape, coarse)]
        disp = np.empty((3, *shape), dtype=np.float32)
        for a in range(3):
            fc = gaussian_filter(
                np.random.randn(*coarse).astype(np.float32),
                cfg.elastic_sigma / 4.0, mode="constant")
            f = zoom(fc, zoom_f, order=1).astype(np.float32)
            disp[a] = f / (f.std() + 1e-8) * cfg.elastic_voxels
        coords = coords + disp
    return coords


def _augment_intensity(v: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
    """Acquisition-variation intensity aug for one z-scored CT volume.

    Args:
        v: Z-scored CT volume to augment.
        cfg: Augmentation probabilities and magnitudes.
    """
    v = v.astype(np.float32, copy=True)
    v += random.uniform(-cfg.brightness_bias, cfg.brightness_bias)
    v += np.random.normal(0.0, cfg.noise_sigma, size=v.shape).astype(np.float32)
    if random.random() < cfg.mult_bright_prob:
        v *= random.uniform(*cfg.mult_bright_range)
    if random.random() < cfg.contrast_prob:
        m = float(v.mean())
        v = (v - m) * random.uniform(*cfg.contrast_range) + m
    if random.random() < cfg.blur_prob:
        from scipy.ndimage import gaussian_filter
        v = gaussian_filter(v, random.uniform(*cfg.blur_sigma))
    if random.random() < cfg.lowres_prob:
        # Downsample then upsample to mimic resolution loss.
        from scipy.ndimage import zoom
        f = random.uniform(*cfg.lowres_range)
        small = zoom(v, f, order=0)
        v = zoom(small, [o / s for o, s in zip(v.shape, small.shape)],
                 order=1).astype(np.float32)
    return v.astype(np.float32, copy=False)
