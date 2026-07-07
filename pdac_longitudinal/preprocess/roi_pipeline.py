"""Raw CT -> PanTS nnU-Net segmentation -> ROI mask files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np

from pdac_longitudinal.preprocess.roi_rings import (
    DEFAULT_RING_RADII_MM,
    DEFAULT_TVI_TUMOUR_MM,
    DEFAULT_TVI_VESSEL_MM,
    create_pt_rings,
    create_tvi_mask,
)
from pdac_longitudinal.preprocess.segmenter import PanTSSegmenter
from pdac_longitudinal.preprocess.vessel_tumor_metrics import (
    compute_vessel_tumor_metrics,
    format_metrics_report,
)

logger = logging.getLogger(__name__)


def _json_default(obj):
    """Coerce numpy scalars / inf / arrays to JSON-friendly Python values (non-finite floats become `None`)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def _array_xyz_to_reference_nifti(
    arr_xyz: np.ndarray,
    reference_img: nib.spatialimages.SpatialImage,
    dtype: np.dtype,
) -> nib.Nifti1Image:
    """Create a NIfTI image in the exact geometry of a reference image.

    `arr_xyz` must be in `(X, Y, Z)` axis order, matching `reference_img`.
    """
    header = reference_img.header.copy()
    header.set_data_dtype(dtype)
    out = nib.Nifti1Image(arr_xyz.astype(dtype), affine=reference_img.affine, header=header)
    qform, qform_code = reference_img.get_qform(coded=True)
    sform, sform_code = reference_img.get_sform(coded=True)
    out.set_qform(qform, code=int(qform_code))
    out.set_sform(sform, code=int(sform_code))
    return out


class ROIPipeline:
    """All-in-one pipeline: raw CT -> PanTS segmentation -> ROI masks.

    Args:
        ring_radii_mm: Strictly increasing peritumoural ring boundary radii in mm; see `create_pt_rings`.
        segment_t1: Whether `process_timepoints` also segments the T1 CT (masks/metrics stay T0-only).
        include_biliary: Whether to also extract pancreatic-duct and common-bile-duct masks.
        device: Torch device for segmentation. Defaults to CUDA if available, else CPU.
        **segmenter_kwargs: Forwarded to `PanTSSegmenter`.
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        ring_radii_mm: Sequence[float] = DEFAULT_RING_RADII_MM,
        tvi_tumour_mm: float = DEFAULT_TVI_TUMOUR_MM,
        tvi_vessel_mm: float = DEFAULT_TVI_VESSEL_MM,
        segment_t1: bool = False,
        include_biliary: bool = False,
        device: Optional[Union[str, "torch.device"]] = None,
        **segmenter_kwargs,
    ) -> None:
        self.ring_radii_mm = tuple(ring_radii_mm)
        self.tvi_tumour_mm = tvi_tumour_mm
        self.tvi_vessel_mm = tvi_vessel_mm
        self.segment_t1 = segment_t1
        self.include_biliary = include_biliary

        logger.info("Loading PanTSSegmenter from %s", weights_path)
        self.segmenter = PanTSSegmenter(
            weights_path=weights_path,
            device=device,
            **segmenter_kwargs,
        )

    def _segment_and_build_masks(
        self,
        ct_path: Union[str, Path],
        label: str = "T0",
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[float, float, float], np.ndarray]:
        """Segment one CT and extract tumour + vessel binary masks."""
        logger.info("[%s] Segmenting %s …", label, ct_path)
        seg_array, spacing_zyx, affine = self.segmenter.segment(ct_path)

        logger.info("[%s] Extracting binary masks …", label)
        binary_masks = self.segmenter.extract_masks(
            seg_array, include_biliary=self.include_biliary
        )

        return seg_array, binary_masks, spacing_zyx, affine

    def _generate_roi_masks(
        self,
        tumor_mask: np.ndarray,
        vessel_union: np.ndarray,
        spacing_zyx: Tuple[float, float, float],
    ) -> Dict[str, np.ndarray]:
        """Build PT rings and TVI from tumour + vessel masks (all arrays in (Z,Y,X)); returns
        `IT`, `PT_ring1..N`, and `TVI` masks.
        """
        sz, sy, sx = spacing_zyx
        voxel_spacing = (sz, sy, sx)

        logger.info(
            "Generating PT rings %s mm and TVI (tumour prox %.1f mm, vessel prox %.1f mm) …",
            self.ring_radii_mm, self.tvi_tumour_mm, self.tvi_vessel_mm,
        )

        pt_rings = create_pt_rings(tumor_mask, voxel_spacing, self.ring_radii_mm)
        tvi = create_tvi_mask(
            tumor_mask,
            vessel_union,
            voxel_spacing,
            tvi_tumour_mm=self.tvi_tumour_mm,
            tvi_vessel_mm=self.tvi_vessel_mm,
            exclude_it=True,
        )

        rois: Dict[str, np.ndarray] = {"IT": tumor_mask}
        for i, ring in enumerate(pt_rings, start=1):
            rois[f"PT_ring{i}"] = ring
        rois["TVI"] = tvi

        sz_f, sy_f, sx_f = spacing_zyx
        vox_vol_cm3 = (sz_f * sy_f * sx_f) / 1000.0
        for name, arr in rois.items():
            logger.info(
                "  %-14s  %7d voxels  %.2f cm³",
                name, arr.sum(), arr.sum() * vox_vol_cm3,
            )

        return rois

    def process_timepoints(
        self,
        t0_path: Union[str, Path],
        t1_path: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        save_segmentations: bool = True,
    ) -> Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]:
        """Process one T0 CT (and optionally T1) into ROI masks.

        `t1_path` is only used when `self.segment_t1` is `True`. Returns a dict with
        `roi_masks`, `seg_T0`, `spacing_zyx`, `affine`, `vessel_tumor_metrics`, and
        (if `self.segment_t1` and `t1_path` were given) `seg_T1`.
        """
        t0_path = Path(t0_path)

        seg_T0, masks_T0, spacing_zyx, affine = self._segment_and_build_masks(
            t0_path, label="T0"
        )
        roi_masks = self._generate_roi_masks(
            tumor_mask=masks_T0["tumor"],
            vessel_union=masks_T0["vessel_union"],
            spacing_zyx=spacing_zyx,
        )

        vessel_masks_subset = {
            name: masks_T0[name]
            for name in ("sma", "celiac", "veins", "postcava")
            if name in masks_T0
        }
        vt_metrics = compute_vessel_tumor_metrics(
            tumor_mask=masks_T0["tumor"],
            vessel_masks=vessel_masks_subset,
            spacing_zyx=spacing_zyx,
        )
        logger.info("Vessel-tumour metrics:\n%s", format_metrics_report(vt_metrics))

        result: Dict = {
            "roi_masks": roi_masks,
            "seg_T0": seg_T0,
            "spacing_zyx": spacing_zyx,
            "affine": affine,
            "vessel_tumor_metrics": vt_metrics,
        }

        if self.segment_t1:
            if t1_path is None:
                logger.warning(
                    "segment_t1=True but t1_path was not provided — skipping T1 segmentation."
                )
            else:
                seg_T1, _, _, _ = self._segment_and_build_masks(
                    t1_path, label="T1"
                )
                result["seg_T1"] = seg_T1

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(
                result=result,
                out_dir=out_dir,
                save_segmentations=save_segmentations,
                t0_path=t0_path,
                t1_path=t1_path,
            )

        return result

    def _save_results(
        self,
        result: dict,
        out_dir: Path,
        save_segmentations: bool,
        t0_path: Path,
        t1_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Write all mask and segmentation NIfTIs to `out_dir`.

        `t1_path` supplies T1's reference geometry when given; falls back to T0 otherwise.
        """
        t0_ref_img = nib.load(str(t0_path))

        for name, arr in result["roi_masks"].items():
            arr_xyz = arr.transpose(2, 1, 0)
            fname = f"mask_{name}.nii.gz"
            nib.save(
                _array_xyz_to_reference_nifti(arr_xyz, t0_ref_img, np.uint8),
                out_dir / fname,
            )
            logger.info("Saved %s → %s", name, out_dir / fname)

        if save_segmentations:
            seg_T0_xyz = result["seg_T0"].transpose(2, 1, 0)
            seg_T0_nib = _array_xyz_to_reference_nifti(seg_T0_xyz, t0_ref_img, np.int16)
            nib.save(seg_T0_nib, out_dir / "seg_T0.nii.gz")
            logger.info("Saved T0 segmentation → %s", out_dir / "seg_T0.nii.gz")

            if "seg_T1" in result:
                t1_ref_img = t0_ref_img
                if t1_path is not None and Path(t1_path).exists():
                    t1_ref_img = nib.load(str(t1_path))
                seg_T1_xyz = result["seg_T1"].transpose(2, 1, 0)
                seg_T1_nib = _array_xyz_to_reference_nifti(seg_T1_xyz, t1_ref_img, np.int16)
                nib.save(seg_T1_nib, out_dir / "seg_T1.nii.gz")
                logger.info("Saved T1 segmentation → %s", out_dir / "seg_T1.nii.gz")

        if "vessel_tumor_metrics" in result:
            metrics_path = out_dir / "vessel_tumor_metrics.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(result["vessel_tumor_metrics"], f, indent=2, default=_json_default)
            logger.info("Saved vessel-tumour metrics → %s", metrics_path)

    def process_dataset(
        self,
        root_dir: Union[str, Path],
        case_ids: List[str],
        t0_filename: str = "T0.nii.gz",
        t1_filename: str = "T1.nii.gz",
        save_segmentations: bool = True,
        skip_existing: bool = True,
        error_on_failure: bool = False,
    ) -> Dict[str, str]:
        """Process all cases in sequence.

        `skip_existing` skips cases with an existing `mask_IT.nii.gz`; `error_on_failure`
        re-raises on the first per-case failure instead of recording it and continuing.
        Returns `{case_id: 'ok'|'skipped'|'error: <msg>'}`.
        """
        root_dir = Path(root_dir)
        status: Dict[str, str] = {}
        n = len(case_ids)

        for i, case_id in enumerate(case_ids, start=1):
            case_dir = root_dir / case_id
            t0_path = case_dir / t0_filename
            t1_path = case_dir / t1_filename if (case_dir / t1_filename).exists() else None
            out_dir = case_dir

            logger.info("[%d/%d] Processing case: %s", i, n, case_id)

            if skip_existing and (out_dir / "mask_IT.nii.gz").exists():
                logger.info("  → Skipping (mask_IT.nii.gz already exists).")
                status[case_id] = "skipped"
                continue

            if not t0_path.exists():
                msg = f"T0 not found: {t0_path}"
                logger.error(msg)
                status[case_id] = f"error: {msg}"
                continue

            try:
                self.process_timepoints(
                    t0_path=t0_path,
                    t1_path=t1_path,
                    out_dir=out_dir,
                    save_segmentations=save_segmentations,
                )
                status[case_id] = "ok"
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                logger.error("  → FAILED for %s: %s", case_id, msg)
                status[case_id] = f"error: {msg}"
                if error_on_failure:
                    raise

        ok = sum(1 for v in status.values() if v == "ok")
        skipped = sum(1 for v in status.values() if v == "skipped")
        errors = sum(1 for v in status.values() if v.startswith("error"))
        logger.info(
            "Dataset processing complete: %d OK, %d skipped, %d errors (out of %d).",
            ok, skipped, errors, n,
        )
        return status


# CLI entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run PanTS segmentation + ROI ring generation on a dataset."
    )
    parser.add_argument(
        "--weights", required=True,
        help="Path to pretrained PanTS checkpoint.",
    )
    parser.add_argument(
        "--root_dir",
        help="Dataset root directory (case subdirectories within).",
    )
    parser.add_argument(
        "--t0", dest="t0_path",
        help="Single-case T0 CT path (mutually exclusive with --root_dir).",
    )
    parser.add_argument(
        "--t1", dest="t1_path", default=None,
        help="Single-case T1 CT path (optional).",
    )
    parser.add_argument(
        "--out_dir",
        help="Output directory for single-case mode.",
    )
    parser.add_argument(
        "--splits", default=None,
        help="JSON splits file (used with --root_dir).",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test", "all"],
        help="Which split to process.",
    )
    parser.add_argument(
        "--ring_radii_mm", nargs="+", type=float, default=list(DEFAULT_RING_RADII_MM),
    )
    parser.add_argument("--tvi_tumour_mm", type=float, default=DEFAULT_TVI_TUMOUR_MM)
    parser.add_argument("--tvi_vessel_mm", type=float, default=DEFAULT_TVI_VESSEL_MM)
    parser.add_argument("--segment_t1", action="store_true")
    parser.add_argument("--no_seg_save", action="store_true",
                        help="Do not save full segmentation NIfTIs (saves disk space).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_mirroring", action="store_true")
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip cases where mask_IT.nii.gz already exists."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = ROIPipeline(
        weights_path=args.weights,
        ring_radii_mm=args.ring_radii_mm,
        tvi_tumour_mm=args.tvi_tumour_mm,
        tvi_vessel_mm=args.tvi_vessel_mm,
        segment_t1=args.segment_t1,
        device=args.device,
        use_mirroring=args.use_mirroring,
    )

    if args.t0_path:
        out = args.out_dir or str(Path(args.t0_path).parent)
        pipeline.process_timepoints(
            t0_path=args.t0_path,
            t1_path=args.t1_path,
            out_dir=out,
            save_segmentations=not args.no_seg_save,
        )
    elif args.root_dir:
        if args.splits:
            with open(args.splits) as f:
                splits_data = json.load(f)
            if args.split == "all":
                case_ids = [c for ids in splits_data.values() for c in ids]
            else:
                case_ids = splits_data[args.split]
        else:
            case_ids = sorted(
                d.name for d in Path(args.root_dir).iterdir() if d.is_dir()
            )
        pipeline.process_dataset(
            root_dir=args.root_dir,
            case_ids=case_ids,
            save_segmentations=not args.no_seg_save,
            skip_existing=args.skip_existing,
        )
    else:
        parser.error("Provide either --t0 or --root_dir.")
