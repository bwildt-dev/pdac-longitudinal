"""CT volume preprocessing for the PDAC longitudinal framework."""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def resample_volume(
    image_nib: nib.Nifti1Image,
    target_spacing_mm: Sequence[float],
    is_mask: bool = False,
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Resample a NIfTI to `target_spacing_mm` (B-spline for CT, NN for masks).

    Args:
        image_nib: Source image to resample.
        target_spacing_mm: Target `(x, y, z)` voxel spacing in mm.
        is_mask: Whether to use nearest-neighbour interpolation and a zero
            default fill value, instead of B-spline with the image minimum.

    Returns:
        A `(array, actual_spacing)` tuple: the resampled array in nibabel
        `(x, y, z)` axis order, and the achieved spacing in mm.
    """
    sitk_image = sitk.ReadImage(image_nib.get_filename() or _nib_to_tempfile(image_nib))

    orig_spacing = sitk_image.GetSpacing()
    orig_size = sitk_image.GetSize()

    # Clamp each axis to ≥1 voxel; a thin axis could otherwise round to 0.
    new_size = [
        max(1, int(round(orig_size[i] * orig_spacing[i] / target_spacing_mm[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(list(target_spacing_mm))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(
        float(sitk.GetArrayFromImage(sitk_image).min()) if not is_mask else 0.0
    )
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline
    )

    resampled = resampler.Execute(sitk_image)

    array = sitk.GetArrayFromImage(resampled)  # (z, y, x)
    array = array.transpose(2, 1, 0).astype(np.float32)  # -> nibabel (x, y, z)

    actual_spacing = tuple(float(s) for s in resampled.GetSpacing())
    logger.debug(
        "Resample: %s → %s  |  spacing %s → %s mm",
        orig_size, new_size, orig_spacing, actual_spacing,
    )
    return array, actual_spacing


def _nib_to_tempfile(image_nib: nib.Nifti1Image) -> str:
    """Write an in-memory NIfTI to a temp file so SimpleITK can open it."""
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    nib.save(image_nib, path)
    return path


def normalise_ct(
    array: np.ndarray,
    clip_min: float,
    clip_max: float,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    fg_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Clip to `[clip_min, clip_max]` HU, then z-score.

    Args:
        array: CT intensity volume, in HU.
        clip_min: Lower HU clip bound.
        clip_max: Upper HU clip bound.
        mean: Fixed mean to z-score against; omit to compute a per-case
            foreground mean instead.
        std: Fixed std to z-score against; must be given together with `mean`.
        fg_mask: Optional foreground mask for the per-case z-score path;
            defaults to voxels above `clip_min`.

    Returns:
        The clipped, z-scored volume; all-zero if every voxel clips to
        `clip_min`.

    Raises:
        ValueError: If exactly one of `mean`/`std` is provided.
    """
    if (mean is None) != (std is None):
        raise ValueError(
            "normalise_ct: mean and std must be provided together, or both omitted."
        )

    arr = np.clip(array, clip_min, clip_max).astype(np.float32)

    if mean is not None and std is not None:
        # Fixed-stat path; matches pretrained encoder's input distribution.
        return (arr - float(mean)) / max(float(std), 1e-8)

    if fg_mask is not None:
        fg = arr[fg_mask.astype(bool)]
    else:
        fg = arr[arr > clip_min]
    if fg.size == 0:
        # All voxels clipped; degenerate input, return zeros to avoid NaN.
        return np.zeros_like(arr)
    fg_mean = float(fg.mean())
    fg_std  = max(float(fg.std()), 1e-8)
    return (arr - fg_mean) / fg_std


def compute_foreground_bbox(
    mask: np.ndarray,
    margin_voxels: int = 0,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return inclusive start and exclusive end indices of the non-zero region in `mask`.

    Args:
        mask: Volume to find the foreground bounding box in.
        margin_voxels: Extra voxels to pad the box by on each side, clamped
            to `mask`'s bounds.

    Returns:
        A `(start, end)` tuple of `(x, y, z)` indices. `((0, 0, 0), mask.shape)`
        when `mask` is entirely zero.
    """
    nonzero = np.argwhere(mask > 0)
    if len(nonzero) == 0:
        return (0, 0, 0), tuple(mask.shape)  # type: ignore[return-value]

    lo = nonzero.min(axis=0)
    hi = nonzero.max(axis=0) + 1  # make exclusive

    lo = np.maximum(lo - margin_voxels, 0)
    hi = np.minimum(hi + margin_voxels, np.array(mask.shape))

    return tuple(lo.tolist()), tuple(hi.tolist())  # type: ignore[return-value]


def crop_volume(
    array: np.ndarray,
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
) -> np.ndarray:
    """Crop array to the bounding box `[start, end)` along each axis.

    Args:
        array: Volume to crop.
        start: Inclusive start index per axis.
        end: Exclusive end index per axis.
    """
    return array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def pad_or_crop_to_patch_size(
    array: np.ndarray,
    patch_size: Tuple[int, int, int],
    mode: str = "constant",
    constant_value: float = 0.0,
) -> np.ndarray:
    """Centre-crop or symmetrically zero-pad array to exactly `patch_size`.

    Args:
        array: Volume to resize.
        patch_size: Target `(x, y, z)` shape.
        mode: `numpy.pad` mode used when padding is needed.
        constant_value: Fill value when `mode="constant"`.

    Returns:
        The array resized to exactly `patch_size`.
    """
    result = array
    pad_widths = []
    slices = []

    for dim, (curr, target) in enumerate(zip(result.shape, patch_size)):
        if curr < target:
            total_pad = target - curr
            pad_lo = total_pad // 2
            pad_hi = total_pad - pad_lo
            pad_widths.append((pad_lo, pad_hi))
            slices.append(slice(None))
        elif curr > target:
            start = (curr - target) // 2
            slices.append(slice(start, start + target))
            pad_widths.append((0, 0))
        else:
            pad_widths.append((0, 0))
            slices.append(slice(None))

    result = result[tuple(slices)]
    if any(p != (0, 0) for p in pad_widths):
        # constant_values is only valid for mode="constant"; pass conditionally.
        if mode == "constant":
            result = np.pad(result, pad_widths, mode=mode, constant_values=constant_value)
        else:
            result = np.pad(result, pad_widths, mode=mode)

    return result


