"""Peritumoural ring (PT) masks and tumour-vessel interface (TVI) from binary segmentation volumes."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)

PANTS_VASCULAR_LABELS: Dict[str, int] = {
    "superior_mesenteric_artery": 3,
    "celiac_artery": 4,
    "veins": 5,
    "postcava": 6,
}

DEFAULT_RING_RADII_MM: Tuple[float, ...] = (0.0, 5.0, 10.0, 15.0)  # must start with 0.0
DEFAULT_TVI_TUMOUR_MM: float = 10.0
DEFAULT_TVI_VESSEL_MM: float = 10.0



def _edt_from_mask(
    binary_mask: np.ndarray,
    voxel_spacing_mm: Sequence[float],
) -> np.ndarray:
    """Euclidean distance (mm) from each outside voxel to the nearest mask surface (all zeros if `binary_mask` is empty)."""
    mask_bool = binary_mask.astype(bool)
    if not mask_bool.any():
        warnings.warn("Empty mask supplied to _edt_from_mask; returning all-zero distance map.")
        return np.zeros(binary_mask.shape, dtype=np.float64)

    dist = distance_transform_edt(~mask_bool, sampling=tuple(voxel_spacing_mm))
    return dist


def create_pt_rings(
    it_mask: np.ndarray,
    voxel_spacing_mm: Sequence[float],
    ring_radii_mm: Sequence[float] = DEFAULT_RING_RADII_MM,
) -> List[np.ndarray]:
    """Generate concentric peritumoural ring masks.

    Each ring is the shell between two radii (mm) measured from the tumour (IT) surface.
    `ring_radii_mm` must start at 0.0 (the IT surface itself) and be strictly increasing.

    Raises:
        ValueError: If `ring_radii_mm` does not start at 0.0 or is not strictly increasing.
    """
    radii = list(ring_radii_mm)
    if radii[0] != 0.0:
        raise ValueError(
            f"ring_radii_mm must start with 0.0, got {radii[0]}.  "
            "The inner boundary of ring 1 is the IT surface itself."
        )
    if any(r2 <= r1 for r1, r2 in zip(radii, radii[1:])):
        raise ValueError(
            f"ring_radii_mm must be strictly increasing, got {radii}."
        )

    dist = _edt_from_mask(it_mask, voxel_spacing_mm)

    rings: List[np.ndarray] = []
    for inner_r, outer_r in zip(radii[:-1], radii[1:]):
        ring = (dist > inner_r) & (dist <= outer_r)
        rings.append(ring)
        logger.debug(
            "PT ring (%.1f, %.1f] mm: %d voxels",
            inner_r, outer_r, ring.sum(),
        )

    return rings


def create_tvi_mask(
    it_mask: np.ndarray,
    vessel_masks: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    voxel_spacing_mm: Sequence[float],
    tvi_tumour_mm: float = DEFAULT_TVI_TUMOUR_MM,
    tvi_vessel_mm: float = DEFAULT_TVI_VESSEL_MM,
    exclude_it: bool = True,
) -> np.ndarray:
    """Generate the tumour-vessel interface (TVI) mask.

    A voxel is part of the TVI if it lies within `tvi_tumour_mm` of the tumour surface AND
    within `tvi_vessel_mm` of any vessel. `vessel_masks` may be a single boolean array, a
    list of per-vessel arrays, or a `{name: mask}` dict, unioned into one mask before use.
    Empty if there are no vessel voxels or the proximity zones don't overlap.
    """
    if isinstance(vessel_masks, np.ndarray):
        vessel_union = vessel_masks.astype(bool)
    elif isinstance(vessel_masks, dict):
        arrays = list(vessel_masks.values())
        vessel_union = np.zeros_like(it_mask, dtype=bool)
        for arr in arrays:
            vessel_union |= arr.astype(bool)
    else:  # list
        vessel_union = np.zeros_like(it_mask, dtype=bool)
        for arr in vessel_masks:
            vessel_union |= arr.astype(bool)

    if not vessel_union.any():
        warnings.warn(
            "No vessel voxels found in vessel_masks.  TVI mask will be empty.  "
            "Check that vessel segmentation was supplied correctly."
        )
        return np.zeros_like(it_mask, dtype=bool)

    dist_to_it = _edt_from_mask(it_mask, voxel_spacing_mm)
    dist_to_vessels = _edt_from_mask(vessel_union, voxel_spacing_mm)

    near_tumour = dist_to_it <= tvi_tumour_mm
    near_vessels = dist_to_vessels <= tvi_vessel_mm

    tvi = near_tumour & near_vessels

    if exclude_it:
        tvi &= ~it_mask.astype(bool)

    logger.debug(
        "TVI mask: %d voxels  (tumour prox=%.1f mm, vessel prox=%.1f mm)",
        tvi.sum(), tvi_tumour_mm, tvi_vessel_mm,
    )

    if not tvi.any():
        warnings.warn(
            f"TVI mask is empty with tvi_tumour_mm={tvi_tumour_mm} and "
            f"tvi_vessel_mm={tvi_vessel_mm}.  The tumour and vessel "
            "neighbourhoods do not overlap — consider increasing one or both "
            "proximity thresholds."
        )

    return tvi



def extract_vessel_masks_from_segmentation(
    seg_array: np.ndarray,
    label_map: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Extract per-vessel binary masks from a multi-label segmentation array (`label_map` defaults to `PANTS_VASCULAR_LABELS`)."""
    if label_map is None:
        label_map = PANTS_VASCULAR_LABELS
    return {
        name: (seg_array == label_id).astype(bool)
        for name, label_id in label_map.items()
    }


def generate_all_roi_masks(
    it_nib: nib.Nifti1Image,
    vessel_nib_map: Optional[Union[Dict[str, nib.Nifti1Image], nib.Nifti1Image]] = None,
    seg_nib: Optional[nib.Nifti1Image] = None,
    ring_radii_mm: Sequence[float] = DEFAULT_RING_RADII_MM,
    tvi_tumour_mm: float = DEFAULT_TVI_TUMOUR_MM,
    tvi_vessel_mm: float = DEFAULT_TVI_VESSEL_MM,
) -> Dict[str, np.ndarray]:
    """Generate all ROI masks for one case from NIfTI inputs.

    Provide either `vessel_nib_map` (per-vessel masks, or a single pre-unioned mask) or
    `seg_nib` (full multi-label PanTS segmentation) — mutually exclusive.

    Returns:
        A dict with `IT`, `vessel_union`, `PT_ring1..N`, and `TVI` boolean masks.

    Raises:
        ValueError: If neither `vessel_nib_map` nor `seg_nib` is provided.
    """
    if vessel_nib_map is None and seg_nib is None:
        raise ValueError(
            "Provide either vessel_nib_map (per-vessel binary masks) or "
            "seg_nib (full multi-label PanTS segmentation)."
        )

    # nibabel loads data as (x, y, z); zooms are in the same order.
    zooms = it_nib.header.get_zooms()[:3]
    voxel_spacing_mm = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    it_array = it_nib.get_fdata(dtype=np.float32).astype(bool)

    if seg_nib is not None:
        seg_array = np.round(seg_nib.get_fdata(dtype=np.float32)).astype(np.int32)
        vessel_mask_arrays = extract_vessel_masks_from_segmentation(seg_array)
    elif isinstance(vessel_nib_map, dict):
        vessel_mask_arrays = {
            name: nib_img.get_fdata(dtype=np.float32).astype(bool)
            for name, nib_img in vessel_nib_map.items()
        }
    else:
        vessel_mask_arrays = {
            "vessel_union": vessel_nib_map.get_fdata(dtype=np.float32).astype(bool)
        }

    vessel_union = np.zeros_like(it_array, dtype=bool)
    for arr in vessel_mask_arrays.values():
        vessel_union |= arr

    pt_rings = create_pt_rings(it_array, voxel_spacing_mm, ring_radii_mm)

    tvi = create_tvi_mask(
        it_array,
        vessel_union,
        voxel_spacing_mm,
        tvi_tumour_mm=tvi_tumour_mm,
        tvi_vessel_mm=tvi_vessel_mm,
        exclude_it=True,
    )

    result: Dict[str, np.ndarray] = {"IT": it_array, "vessel_union": vessel_union}
    for i, ring in enumerate(pt_rings, start=1):
        result[f"PT_ring{i}"] = ring
    result["TVI"] = tvi

    logger.info(
        "ROI masks generated for shape=%s  spacing=%.2f×%.2f×%.2f mm",
        it_array.shape, *voxel_spacing_mm,
    )
    for name, arr in result.items():
        logger.info("  %-14s  %6d voxels  (%.2f cm³)",
                    name, arr.sum(),
                    arr.sum() * np.prod(voxel_spacing_mm) / 1000.0)

    return result


def save_roi_masks(
    roi_masks: Dict[str, np.ndarray],
    out_dir: Union[str, Path],
    ref_nib: nib.Nifti1Image,
    filename_map: Optional[Dict[str, str]] = None,
    dtype: type = np.uint8,
) -> None:
    """Save ROI mask arrays as NIfTI files, inheriting geometry from `ref_nib` (default filenames are `mask_{name}.nii.gz`, override via `filename_map`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_names = {name: f"mask_{name}.nii.gz" for name in roi_masks}
    if filename_map:
        default_names.update(filename_map)

    for name, array in roi_masks.items():
        fname = default_names[name]
        nib_img = nib.Nifti1Image(
            array.astype(dtype), affine=ref_nib.affine, header=ref_nib.header
        )
        nib.save(nib_img, out_dir / fname)
        logger.info("Saved %s → %s", name, out_dir / fname)


# CLI entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PT ring and TVI masks from IT and vessel masks."
    )
    parser.add_argument("--it_mask", required=True, help="IT mask NIfTI path.")
    parser.add_argument(
        "--vessel_masks",
        nargs="+",
        default=[],
        help="Vessel mask paths as name=path pairs, e.g. sma=seg_sma.nii.gz",
    )
    parser.add_argument(
        "--seg",
        default=None,
        help="Full multi-label PanTS segmentation NIfTI (alternative to --vessel_masks).",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--ring_radii_mm",
        nargs="+",
        type=float,
        default=list(DEFAULT_RING_RADII_MM),
        help="Ring boundary values in mm (default: 0 5 10 15).",
    )
    parser.add_argument(
        "--tvi_tumour_mm", type=float, default=DEFAULT_TVI_TUMOUR_MM,
        help="Tumour proximity threshold for TVI (mm)."
    )
    parser.add_argument(
        "--tvi_vessel_mm", type=float, default=DEFAULT_TVI_VESSEL_MM,
        help="Vessel proximity threshold for TVI (mm)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    it_nib = nib.load(args.it_mask)

    vessel_nib_map: Optional[Dict[str, nib.Nifti1Image]] = None
    seg_nib_loaded: Optional[nib.Nifti1Image] = None

    if args.seg:
        seg_nib_loaded = nib.load(args.seg)
    elif args.vessel_masks:
        vessel_nib_map = {}
        for pair in args.vessel_masks:
            if "=" not in pair:
                parser.error(f"--vessel_masks entries must be name=path, got: {pair!r}")
            name, path = pair.split("=", 1)
            vessel_nib_map[name] = nib.load(path)
    else:
        parser.error("Provide either --seg or --vessel_masks.")

    rois = generate_all_roi_masks(
        it_nib=it_nib,
        vessel_nib_map=vessel_nib_map,
        seg_nib=seg_nib_loaded,
        ring_radii_mm=args.ring_radii_mm,
        tvi_tumour_mm=args.tvi_tumour_mm,
        tvi_vessel_mm=args.tvi_vessel_mm,
    )
    save_roi_masks(rois, out_dir=args.out_dir, ref_nib=it_nib)
    print(f"Saved {len(rois)} ROI masks to {args.out_dir}")
