"""Cache-building pipeline for the longitudinal dataset.

`CachePipelineMixin` turns one discovered case into the preprocessed arrays that
get cached to `.npz`: segment both timepoints, derive ROI masks + anatomy/vessel
features, resample, re-clean, crop to a shared anatomical frame, pad, and assemble.
`_preprocess_and_cache` runs the phases in order; each phase reads and updates a
working dict `w` carrying the case's arrays.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np

from pdac_longitudinal.preprocess.anatomy_features import AnatomyFeatureExtractor
from pdac_longitudinal.preprocess.mask_utils import (
    kidney_centroids as _kidney_centroids,
    largest_cc as _largest_cc,
    pancreas_anatomy_mask as _pancreas_anatomy_mask,
)
from pdac_longitudinal.preprocess.preprocessing import (
    compute_foreground_bbox,
    crop_volume,
    normalise_ct,
    pad_or_crop_to_patch_size,
    resample_volume,
)
from pdac_longitudinal.preprocess.roi_rings import (
    create_pt_rings,
    create_tvi_mask,
    extract_vessel_masks_from_segmentation,
)
from pdac_longitudinal.preprocess.segmenter import PanTSSegmenter
from pdac_longitudinal.preprocess.vessel_features import VesselFeatureExtractor

logger = logging.getLogger(__name__)

_PANC_MIN_VOXELS      = 3000
_KID_UNION_MIN_VOXELS = 8000 


def _arr_to_nib(arr: np.ndarray, ref: "nib.Nifti1Image") -> "nib.Nifti1Image":
    """Wrap a numpy array in a NIfTI header copied from `ref`."""
    return nib.Nifti1Image(arr.astype(np.float32), ref.affine, ref.header)


def _stable_centroid(mask: np.ndarray, min_voxels: int) -> Optional[np.ndarray]:
    """Return the voxel centroid of `mask`, or None if too small to be stable."""
    if int(mask.sum()) < min_voxels:
        return None
    return np.argwhere(mask > 0).mean(axis=0).astype(np.float64)


class CachePipelineMixin:
    """Segmentation + preprocessing that builds one case's cached arrays."""

    # Segmentation

    def _get_segmenter(self) -> PanTSSegmenter:
        """Return the PanTS segmenter, constructing it lazily on first use.

        Raises:
            RuntimeError: If neither `segmenter` nor `weights_path` was provided.
        """
        if self._segmenter is None:
            if self._weights_path is None:
                raise RuntimeError(
                    "PanTS weights_path is required for on-the-fly segmentation. "
                    "Pass weights_path= or precompute caches and point cache_dir "
                    "to the precomputed directory."
                )
            self._segmenter = PanTSSegmenter(
                weights_path=self._weights_path,
                device=self._segmenter_device,
                use_mirroring=False,
                use_gaussian=False,
                # Host-RAM aggregation buffer; GPU mode OOMs on very large CTs.
                perform_everything_on_device=False,
            )
        return self._segmenter

    def _estimate_seg_resources(self, nib_img: "nib.Nifti1Image") -> Tuple[int, float]:
        """Estimate nnUNet tile count and peak host-RAM for a CT volume.

        Returns:
            `(n_tiles, peak_gb)`.
        """
        shape_xyz = nib_img.shape[:3]
        shape_zyx = (shape_xyz[2], shape_xyz[1], shape_xyz[0])
        zooms = nib_img.header.get_zooms()
        spacing_zyx = (float(zooms[2]), float(zooms[1]), float(zooms[0]))

        target = self._pants_target_spacing_zyx
        patch  = self._pants_patch_zyx
        step   = self._pants_step

        resampled = tuple(
            max(p, int(math.ceil(s * sp / tp)))
            for s, sp, tp, p in zip(shape_zyx, spacing_zyx, target, patch)
        )
        strides = tuple(int(p * step) for p in patch)
        n_tiles = 1
        for ds, ps, ss in zip(resampled, patch, strides):
            n_tiles *= int(math.ceil((ds - ps) / ss)) + 1

        n_voxels = int(resampled[0]) * int(resampled[1]) * int(resampled[2])
        buf_gb   = 19 * n_voxels * 4 / 1024 ** 3
        peak_gb  = buf_gb * 3.5

        return n_tiles, peak_gb

    def _segment(
        self, ct_path, pid: str, tag: str,
    ) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]:
        """Segment one timepoint, or reuse a saved seg from cache.

        Args:
            ct_path: Path to the CT NIfTI file.
            pid: Patient identifier, used to name/locate the saved seg.
            tag: Timepoint tag used in the seg filename.

        Returns:
            `(seg_zyx, spacing_zyx, affine)`.
        """
        seg_path = (
            self.cache_dir / f"{pid}_seg_{tag}.nii.gz" if self.cache_dir is not None else None
        )
        if self.reuse_saved_segs and seg_path is not None and seg_path.exists():
            nii = nib.load(str(seg_path))
            seg = nii.get_fdata().astype(np.int16).transpose(2, 1, 0)  # XYZ->ZYX
            zooms = nii.header.get_zooms()[:3]
            spacing_zyx = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
            affine = nii.affine
            logger.info("%s %s seg reused from %s (PanTS skipped)", pid, tag, seg_path)
            return seg, spacing_zyx, affine

        seg, spacing_zyx, affine = self._get_segmenter().segment(ct_path)
        if self.cache_dir is not None:
            nib.save(
                nib.Nifti1Image(seg.astype(np.int16).transpose(2, 1, 0), affine),
                self.cache_dir / f"{pid}_seg_{tag}.nii.gz",
            )
        return seg, spacing_zyx, affine

    def _process_timepoint(
        self, ct_path, seg: np.ndarray, spacing_zyx: Tuple[float, float, float], suffix: str,
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
        """Build ROI masks (XYZ) and anatomy/vessel features for one timepoint.

        Args:
            ct_path: Path to the CT NIfTI file.
            seg: PanTS segmentation label map in ZYX order.
            spacing_zyx: Voxel spacing (Z, Y, X) in mm.
            suffix: Timepoint suffix appended to feature names.

        Returns:
            `(masks, anat_feats, vessel_feats)`. `masks` holds `tumor_xyz`,
            `pt_rings` (3-tuple), `tvi_xyz`, `liver_xyz`, `pancreas_xyz`,
            `kidneys_xyz`.
        """
        from pdac_longitudinal.preprocess.segmenter import PANTS_LABELS

        tumor = _largest_cc((seg == 2).astype(bool))  # drop FP blobs
        vessel_masks = extract_vessel_masks_from_segmentation(seg)
        spacing_xyz = (spacing_zyx[2], spacing_zyx[1], spacing_zyx[0])

        # Native CT in the seg's ZYX frame.
        try:
            ct_native = nib.load(str(ct_path)).get_fdata().astype(np.float32)
            ct_zyx = ct_native.transpose(2, 1, 0) if ct_native.shape != seg.shape else ct_native
        except Exception:
            ct_zyx = None
        vox_mm3 = float(np.prod(spacing_zyx))
        anat = AnatomyFeatureExtractor().extract_one_from_arrays(
            seg, ct_zyx, vox_mm3, suffix=suffix,
        )
        vessel = VesselFeatureExtractor().extract_one_from_arrays(
            seg, spacing_zyx, suffix=suffix, ct=ct_zyx,
        )

        tumor_xyz = tumor.transpose(2, 1, 0)
        masks = {
            "tumor_xyz": tumor_xyz,
            "pt_rings": create_pt_rings(it_mask=tumor_xyz, voxel_spacing_mm=spacing_xyz),
            "tvi_xyz": create_tvi_mask(
                it_mask=tumor_xyz,
                vessel_masks={k: v.transpose(2, 1, 0) for k, v in vessel_masks.items()},
                voxel_spacing_mm=spacing_xyz,
            ),
            "liver_xyz": (seg == PANTS_LABELS["liver"]).astype(bool).transpose(2, 1, 0),
            "pancreas_xyz": _pancreas_anatomy_mask(seg).transpose(2, 1, 0),
            "kidneys_xyz": _kidney_centroids(seg)[2].transpose(2, 1, 0),
        }
        return masks, anat, vessel

    def _normalise_pair(self, t0_arr, t1_arr, pid):
        """Clip and z-score the T0/T1 CT pair per the `ct_norm_*` settings.

        Priority: fixed global stats, then shared-pair foreground stats,
        then per-timepoint foreground z-score.

        Args:
            t0_arr: T0 CT array in HU.
            t1_arr: T1 CT array in HU.
            pid: Patient identifier, used for logging.

        Returns:
            `(t0_arr, t1_arr)`.
        """
        if self.ct_norm_mean is not None and self.ct_norm_std is not None:
            mean_t0 = mean_t1 = self.ct_norm_mean
            std_t0  = std_t1  = self.ct_norm_std
        elif self.ct_norm_shared_pair:
            t0_clip = np.clip(t0_arr, self.ct_clip_min, self.ct_clip_max)
            t1_clip = np.clip(t1_arr, self.ct_clip_min, self.ct_clip_max)
            fg0 = t0_clip[t0_clip > self.ct_clip_min]
            fg1 = t1_clip[t1_clip > self.ct_clip_min]
            if fg0.size + fg1.size == 0:
                logger.warning("%s: empty foreground union — falling back to zeros", pid)
                mean_t0 = mean_t1 = 0.0
                std_t0  = std_t1  = 1.0
            else:
                joint = np.concatenate([fg0, fg1])
                shared_mean = float(joint.mean())
                shared_std  = max(float(joint.std()), 1e-8)
                mean_t0 = mean_t1 = shared_mean
                std_t0  = std_t1  = shared_std
            mean_t0_only = float(fg0.mean()) if fg0.size else float("nan")
            mean_t1_only = float(fg1.mean()) if fg1.size else float("nan")
            logger.info(
                "%s norm: fg_mean_T0=%.1f HU  fg_mean_T1=%.1f HU  Δ=%+.1f HU  "
                "(shared joint μ=%.1f σ=%.1f)",
                pid, mean_t0_only, mean_t1_only,
                mean_t1_only - mean_t0_only, mean_t0, std_t0,
            )
            del t0_clip, t1_clip, fg0, fg1
        else:
            mean_t0 = std_t0 = None
            mean_t1 = std_t1 = None  # -> per-timepoint foreground z-score
        t0_arr = normalise_ct(
            t0_arr, self.ct_clip_min, self.ct_clip_max,
            mean=mean_t0, std=std_t0,
        )
        t1_arr = normalise_ct(
            t1_arr, self.ct_clip_min, self.ct_clip_max,
            mean=mean_t1, std=std_t1,
        )
        return t0_arr, t1_arr

    def _render_cache_viz(self, pid: str, arrays: Dict[str, np.ndarray]) -> None:
        """Render an ROI-overlay PNG every Nth case; always advances the counter.

        Args:
            pid: Patient identifier.
            arrays: Preprocessed arrays for this case.
        """
        if (self.viz_cache_every > 0
                and self.cache_dir is not None
                and (self._viz_counter % self.viz_cache_every) == 0):
            try:
                from pdac_longitudinal.visualisation.attention_viz import render_roi_overlay
                viz_dir = self.cache_dir / "_viz"
                masks_t0 = {
                    "mask_it":     arrays["mask_it"],
                    "mask_pt1":    arrays["mask_pt1"],
                    "mask_pt2":    arrays["mask_pt2"],
                    "mask_pt3":    arrays["mask_pt3"],
                    "mask_tvi":    arrays["mask_tvi"],
                    "liver_t0":    arrays["liver_t0"],
                    "pancreas_t0": arrays["pancreas_t0"],
                    "kidneys_t0":  arrays["kidneys_t0"],
                }
                masks_t1 = {
                    "mask_it_t1":  arrays["mask_it_t1"],
                    "mask_pt1_t1": arrays["mask_pt1_t1"],
                    "mask_pt2_t1": arrays["mask_pt2_t1"],
                    "mask_pt3_t1": arrays["mask_pt3_t1"],
                    "mask_tvi_t1": arrays["mask_tvi_t1"],
                    "liver_t1":    arrays["liver_t1"],
                    "pancreas_t1": arrays["pancreas_t1"],
                    "kidneys_t1":  arrays["kidneys_t1"],
                }
                out = render_roi_overlay(
                    ct_t0=arrays["t0"], ct_t1=arrays["t1"],
                    masks_t0=masks_t0, masks_t1=masks_t1,
                    out_path=viz_dir / f"{pid}_roi.png",
                    case_id=pid,
                )
                logger.info("%s ROI overlay → %s", pid, out)
                # Also push to W&B (no-op if inactive) so previews don't need an rsync.
                try:
                    from pdac_longitudinal.training.wandb_setup import log_cache_roi
                    log_cache_roi(pid, out)
                except Exception as _wb_exc:
                    logger.warning("%s W&B ROI push failed (%s); PNG still on disk.",
                                   pid, _wb_exc)
            except Exception as _viz_exc:
                logger.warning("%s ROI overlay failed (%s); continuing.", pid, _viz_exc)
        self._viz_counter += 1

    # Preprocessing pipeline

    def _preprocess_and_cache(
        self,
        case: Dict,
        max_seg_tiles: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Run the full preprocessing pipeline for one case and cache the result.

        Args:
            case: Case record with at least `patient_id`, `t0`, `t1` paths.
            max_seg_tiles: Abort before segmenting if the estimated tile
                count would exceed this; None disables the check.

        Returns:
            The preprocessed arrays dict, also cached to `.npz` if
            `cache_dir` is set.

        Raises:
            RuntimeError: If the estimated segmentation tile count exceeds
                `max_seg_tiles`.
        """
        pid = case["patient_id"]
        logger.info("Preprocessing %s (no cache hit)", pid)

        m0, m1, anat_feats, vessel_feats = self._segment_and_featurize(case, max_seg_tiles, pid)
        w = self._resample_case(case, m0, m1)
        self._reclean_and_rebuild_rings(w, pid)
        w["t0_arr"], w["t1_arr"] = self._normalise_pair(w["t0_arr"], w["t1_arr"], pid)
        self._crop_t0(w, pid)
        self._align_and_crop_t1(w, pid)
        self._pad_to_patch(w, pid)
        self._maybe_deformable_register_t1(w, pid)
        arrays = self._assemble_arrays(w, pid, anat_feats, vessel_feats)

        cp = self._cache_path(pid)
        if cp is not None:
            np.savez_compressed(cp, **arrays)
            logger.debug("Cached %s → %s", pid, cp)

        self._render_cache_viz(pid, arrays)
        return arrays

    def _segment_and_featurize(
        self, case: Dict, max_seg_tiles: Optional[int], pid: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float], Dict[str, float]]:
        """Segment both timepoints and derive their masks + paired features.

        Returns:
            `(m0, m1, anat_feats, vessel_feats)`.

        Raises:
            RuntimeError: If the estimated tile count exceeds `max_seg_tiles`.
        """
        # Pre-flight: estimate RAM before launching segmentation.
        t0_nib = nib.load(str(case["t0"]))
        n_tiles, peak_gb = self._estimate_seg_resources(t0_nib)
        logger.info(
            "%s T0 pre-flight: estimated %d nnUNet tiles, peak RAM ~%.0f GB",
            pid, n_tiles, peak_gb,
        )
        if max_seg_tiles is not None and n_tiles > max_seg_tiles:
            raise RuntimeError(
                f"{pid}: estimated {n_tiles} tiles exceeds max_seg_tiles={max_seg_tiles}; "
                f"skipping (peak RAM would be ~{peak_gb:.0f} GB)"
            )
        del t0_nib

        seg_t0, spacing_zyx, _ = self._segment(case["t0"], pid, "T0")
        m0, anat0, ves0 = self._process_timepoint(case["t0"], seg_t0, spacing_zyx, "_t0")
        del seg_t0
        seg_t1, spacing_zyx_t1, _ = self._segment(case["t1"], pid, "T1")
        m1, anat1, ves1 = self._process_timepoint(case["t1"], seg_t1, spacing_zyx_t1, "_t1")
        del seg_t1

        anat_feats = {**anat0, **anat1}
        anat_feats.update(AnatomyFeatureExtractor._compute_pair_derivatives(anat_feats))
        vessel_feats = {**ves0, **ves1}
        vessel_feats.update(VesselFeatureExtractor._compute_pair_derivatives(vessel_feats))
        logger.info(
            "%s anatomy: liver_t0=%.0fmL  delta_liver=%s  pancreas_atrophy=%s",
            pid, anat_feats.get("liver_t0_mL", 0.0),
            anat_feats.get("delta_liver_pct", "n/a"),
            anat_feats.get("pancreas_atrophy", "n/a"),
        )
        logger.info(
            "%s vessels: arterial_enc_t0=%.0f°  arterial_enc_t1=%.0f°  delta=%.0f°  "
            "resectability=%s→%s", pid,
            vessel_feats.get("arterial_enc_deg_t0", 0.0),
            vessel_feats.get("arterial_enc_deg_t1", 0.0),
            vessel_feats.get("delta_arterial_enc_deg", 0.0),
            vessel_feats.get("resectability_t0", "?"),
            vessel_feats.get("resectability_t1", "?"),
        )
        return m0, m1, anat_feats, vessel_feats

    def _resample_case(self, case: Dict, m0: Dict[str, Any], m1: Dict[str, Any]) -> Dict[str, Any]:
        """Resample the CT pair and every ROI mask to the target spacing.

        Returns:
            A working dict `w` holding the resampled arrays, `new_spacing`, and
            the loaded `t0_nib`/`t1_nib` (reused by the crop phases).
        """
        tumor_t0_xyz, tumor_t1_xyz = m0["tumor_xyz"], m1["tumor_xyz"]
        pt_rings, pt_rings_t1 = m0["pt_rings"], m1["pt_rings"]
        tvi_mask_arr, tvi_mask_t1_arr = m0["tvi_xyz"], m1["tvi_xyz"]
        liver_t0_xyz, liver_t1_xyz = m0["liver_xyz"], m1["liver_xyz"]
        pancreas_t0_xyz, pancreas_t1_xyz = m0["pancreas_xyz"], m1["pancreas_xyz"]
        kidneys_t0_xyz, kidneys_t1_xyz = m0["kidneys_xyz"], m1["kidneys_xyz"]

        t0_nib = nib.load(str(case["t0"]))
        t1_nib = nib.load(str(case["t1"]))

        t0_arr, new_spacing = resample_volume(t0_nib, self.target_spacing_mm, is_mask=False)
        t1_arr, _           = resample_volume(t1_nib, self.target_spacing_mm, is_mask=False)
        it_arr,  _ = resample_volume(_arr_to_nib(tumor_t0_xyz.astype(np.float32), t0_nib),   self.target_spacing_mm, is_mask=True)
        it1_arr, _ = resample_volume(_arr_to_nib(tumor_t1_xyz.astype(np.float32), t1_nib),   self.target_spacing_mm, is_mask=True)
        liver_t0_arr, _    = resample_volume(_arr_to_nib(liver_t0_xyz.astype(np.float32),    t0_nib), self.target_spacing_mm, is_mask=True)
        liver_t1_arr, _    = resample_volume(_arr_to_nib(liver_t1_xyz.astype(np.float32),    t1_nib), self.target_spacing_mm, is_mask=True)
        pancreas_t0_arr, _ = resample_volume(_arr_to_nib(pancreas_t0_xyz.astype(np.float32), t0_nib), self.target_spacing_mm, is_mask=True)
        pancreas_t1_arr, _ = resample_volume(_arr_to_nib(pancreas_t1_xyz.astype(np.float32), t1_nib), self.target_spacing_mm, is_mask=True)
        kidneys_t0_arr,  _ = resample_volume(_arr_to_nib(kidneys_t0_xyz .astype(np.float32), t0_nib), self.target_spacing_mm, is_mask=True)
        kidneys_t1_arr,  _ = resample_volume(_arr_to_nib(kidneys_t1_xyz .astype(np.float32), t1_nib), self.target_spacing_mm, is_mask=True)
        pt1_arr,    _ = resample_volume(_arr_to_nib(pt_rings[0].astype(np.float32),       t0_nib), self.target_spacing_mm, is_mask=True)
        pt2_arr,    _ = resample_volume(_arr_to_nib(pt_rings[1].astype(np.float32),       t0_nib), self.target_spacing_mm, is_mask=True)
        pt3_arr,    _ = resample_volume(_arr_to_nib(pt_rings[2].astype(np.float32),       t0_nib), self.target_spacing_mm, is_mask=True)
        tvi_arr,    _ = resample_volume(_arr_to_nib(tvi_mask_arr.astype(np.float32),      t0_nib), self.target_spacing_mm, is_mask=True)
        pt1_t1_arr, _ = resample_volume(_arr_to_nib(pt_rings_t1[0].astype(np.float32),   t1_nib), self.target_spacing_mm, is_mask=True)
        pt2_t1_arr, _ = resample_volume(_arr_to_nib(pt_rings_t1[1].astype(np.float32),   t1_nib), self.target_spacing_mm, is_mask=True)
        pt3_t1_arr, _ = resample_volume(_arr_to_nib(pt_rings_t1[2].astype(np.float32),   t1_nib), self.target_spacing_mm, is_mask=True)
        tvi_t1_arr, _ = resample_volume(_arr_to_nib(tvi_mask_t1_arr.astype(np.float32),  t1_nib), self.target_spacing_mm, is_mask=True)

        return {
            "t0_nib": t0_nib, "t1_nib": t1_nib, "new_spacing": new_spacing,
            "t0_arr": t0_arr, "t1_arr": t1_arr,
            "it_arr": it_arr, "it1_arr": it1_arr,
            "liver_t0_arr": liver_t0_arr, "liver_t1_arr": liver_t1_arr,
            "pancreas_t0_arr": pancreas_t0_arr, "pancreas_t1_arr": pancreas_t1_arr,
            "kidneys_t0_arr": kidneys_t0_arr, "kidneys_t1_arr": kidneys_t1_arr,
            "pt1_arr": pt1_arr, "pt2_arr": pt2_arr, "pt3_arr": pt3_arr, "tvi_arr": tvi_arr,
            "pt1_t1_arr": pt1_t1_arr, "pt2_t1_arr": pt2_t1_arr, "pt3_t1_arr": pt3_t1_arr,
            "tvi_t1_arr": tvi_t1_arr,
        }

    def _reclean_and_rebuild_rings(self, w: Dict[str, Any], pid: str) -> None:
        """Re-clean masks after resampling and rebuild PT rings from the cleaned IT.

        Interpolate-then-threshold can re-fragment thin shells, so rings and TVI
        are rebuilt from the cleaned largest-CC IT to stay concentric. Also
        captures the pancreas/kidney alignment centroids.
        """
        it_arr = w["it_arr"]
        it1_arr = w["it1_arr"]
        liver_t0_arr = w["liver_t0_arr"]
        liver_t1_arr = w["liver_t1_arr"]
        pancreas_t0_arr = w["pancreas_t0_arr"]
        pancreas_t1_arr = w["pancreas_t1_arr"]
        kidneys_t0_arr = w["kidneys_t0_arr"]
        kidneys_t1_arr = w["kidneys_t1_arr"]

        ts = tuple(self.target_spacing_mm)  # (x, y, z), matches create_pt_rings
        # Two IT tracks: `_full` (all components) is cached as mask_it; `_largest`
        # (dominant lesion) drives the PT rings, TVI, and the crop anchor.
        it_arr_full      = it_arr.astype(bool)
        it1_arr_full     = it1_arr.astype(bool)
        it_arr_largest   = _largest_cc(it_arr_full)
        it1_arr_largest  = _largest_cc(it1_arr_full)
        liver_t0_arr     = _largest_cc(liver_t0_arr.astype(bool)).astype(np.float32)
        liver_t1_arr     = _largest_cc(liver_t1_arr.astype(bool)).astype(np.float32)
        pancreas_t0_arr  = _largest_cc(pancreas_t0_arr.astype(bool)).astype(np.float32)
        pancreas_t1_arr  = _largest_cc(pancreas_t1_arr.astype(bool)).astype(np.float32)
        # Kidneys were already cleaned per-side at native spacing; re-CC'ing the
        # resampled union would drop a lobe, so just cast.
        kidneys_t0_arr = kidneys_t0_arr.astype(np.float32)
        kidneys_t1_arr = kidneys_t1_arr.astype(np.float32)

        # Pancreas centroids in resampled XYZ voxel coords, captured before any
        # crop. A minimum size guards against a truncated pancreas giving a biased
        # centroid; below it, the T1 crop falls back to the tumour anchor.
        pancreas_t0_centroid = _stable_centroid(pancreas_t0_arr, _PANC_MIN_VOXELS)
        pancreas_t1_centroid = _stable_centroid(pancreas_t1_arr, _PANC_MIN_VOXELS)

        # Kidneys are the primary alignment anchor; pancreas is the fallback.
        kidney_t0_centroid: Optional[np.ndarray] = None
        kidney_t1_centroid: Optional[np.ndarray] = None
        if (int(kidneys_t0_arr.sum()) >= _KID_UNION_MIN_VOXELS
                and int(kidneys_t1_arr.sum()) >= _KID_UNION_MIN_VOXELS):
            kidney_t0_centroid = (
                np.argwhere(kidneys_t0_arr > 0).mean(axis=0).astype(np.float64)
            )
            kidney_t1_centroid = (
                np.argwhere(kidneys_t1_arr > 0).mean(axis=0).astype(np.float64)
            )

        if pancreas_t0_centroid is None or pancreas_t1_centroid is None:
            logger.warning(
                "%s pancreas mask too small for alignment "
                "(t0_voxels=%d t1_voxels=%d, threshold=%d).",
                pid, int(pancreas_t0_arr.sum()), int(pancreas_t1_arr.sum()),
                _PANC_MIN_VOXELS,
            )
        if kidney_t0_centroid is None:
            logger.info(
                "%s kidney union too small for alignment "
                "(t0_voxels=%d t1_voxels=%d, threshold=%d) — will try pancreas.",
                pid, int(kidneys_t0_arr.sum()), int(kidneys_t1_arr.sum()),
                _KID_UNION_MIN_VOXELS,
            )
        # Rebuild PT rings from the cleaned largest-CC IT (empty IT -> empty rings).
        if it_arr_largest.any():
            pt_rings_rs = create_pt_rings(it_mask=it_arr_largest, voxel_spacing_mm=ts)
            pt1_arr, pt2_arr, pt3_arr = (r.astype(np.float32) for r in pt_rings_rs)
        else:
            pt1_arr = np.zeros_like(it_arr); pt2_arr = np.zeros_like(it_arr); pt3_arr = np.zeros_like(it_arr)
        if it1_arr_largest.any():
            pt_rings_t1_rs = create_pt_rings(it_mask=it1_arr_largest, voxel_spacing_mm=ts)
            pt1_t1_arr, pt2_t1_arr, pt3_t1_arr = (r.astype(np.float32) for r in pt_rings_t1_rs)
        else:
            pt1_t1_arr = np.zeros_like(it1_arr); pt2_t1_arr = np.zeros_like(it1_arr); pt3_t1_arr = np.zeros_like(it1_arr)
        # mask_it caches all PanTS-detected components, not just the largest CC
        it_arr  = it_arr_full.astype(np.float32)
        it1_arr = it1_arr_full.astype(np.float32)
        # CC summary for auditing multi-focal cases vs. noisy detections
        from scipy.ndimage import label as _cclabel
        _ncc_t0 = int(_cclabel(it_arr_full)[1]) if it_arr_full.any() else 0
        _ncc_t1 = int(_cclabel(it1_arr_full)[1]) if it1_arr_full.any() else 0
        logger.info(
            "%s post-resample IT: t0 vox=%d (cc=%d, largest=%d)  "
            "t1 vox=%d (cc=%d, largest=%d)  liver_t0=%d liver_t1=%d",
            pid,
            int(it_arr.sum()), _ncc_t0, int(it_arr_largest.sum()),
            int(it1_arr.sum()), _ncc_t1, int(it1_arr_largest.sum()),
            int(liver_t0_arr.sum()), int(liver_t1_arr.sum()),
        )

        w.update({
            "it_arr": it_arr, "it1_arr": it1_arr,
            "it_arr_largest": it_arr_largest, "it1_arr_largest": it1_arr_largest,
            "liver_t0_arr": liver_t0_arr, "liver_t1_arr": liver_t1_arr,
            "pancreas_t0_arr": pancreas_t0_arr, "pancreas_t1_arr": pancreas_t1_arr,
            "kidneys_t0_arr": kidneys_t0_arr, "kidneys_t1_arr": kidneys_t1_arr,
            "pt1_arr": pt1_arr, "pt2_arr": pt2_arr, "pt3_arr": pt3_arr,
            "pt1_t1_arr": pt1_t1_arr, "pt2_t1_arr": pt2_t1_arr, "pt3_t1_arr": pt3_t1_arr,
            "pancreas_t0_centroid": pancreas_t0_centroid,
            "pancreas_t1_centroid": pancreas_t1_centroid,
            "kidney_t0_centroid": kidney_t0_centroid,
            "kidney_t1_centroid": kidney_t1_centroid,
        })

    def _crop_t0(self, w: Dict[str, Any], pid: str) -> None:
        """Pick the T0 crop anchor (largest-CC IT, else pancreas) and crop T0 arrays."""
        t0_nib = w["t0_nib"]
        it_arr_largest = w["it_arr_largest"]
        t0_arr = w["t0_arr"]
        it_arr = w["it_arr"]
        pt1_arr, pt2_arr, pt3_arr, tvi_arr = w["pt1_arr"], w["pt2_arr"], w["pt3_arr"], w["tvi_arr"]
        liver_t0_arr = w["liver_t0_arr"]
        pancreas_t0_arr = w["pancreas_t0_arr"]
        kidneys_t0_arr = w["kidneys_t0_arr"]

        # T0 crop anchor (with fallback): largest-CC IT so the crop centres on the
        # dominant lesion. Only affects the bbox; the full mask_it is still cached.
        anchor_t0_xyz = None
        anchor_source_t0 = "none"
        if it_arr_largest.any():
            anchor_t0_xyz = it_arr_largest.astype(np.float32)
            anchor_source_t0 = "tumor"
        else:
            try:
                _seg_T0 = (
                    nib.load(str(self.cache_dir / f"{pid}_seg_T0.nii.gz"))
                    .get_fdata().astype(np.int16)
                )
                # Pancreas anatomy (parenchyma ∪ tumour) as the crop anchor.
                pancreas_mask = _pancreas_anatomy_mask(_seg_T0)
                if pancreas_mask.any():
                    # get_fdata() is already (x, y, z), as _arr_to_nib expects;
                    # do not transpose here or the bbox mis-indexes t0_arr.
                    pancreas_xyz_rs, _ = resample_volume(
                        _arr_to_nib(pancreas_mask.astype(np.float32), t0_nib),
                        self.target_spacing_mm, is_mask=True,
                    )
                    if pancreas_xyz_rs.any():
                        anchor_t0_xyz = pancreas_xyz_rs
                        anchor_source_t0 = "pancreas"
            except Exception as _exc:
                logger.debug("Pancreas T0 fallback unavailable for %s: %s", pid, _exc)

        start_t0_kept: Optional[Tuple[int, int, int]] = None
        end_t0_kept:   Optional[Tuple[int, int, int]] = None
        if self.crop_to_foreground and anchor_t0_xyz is not None:
            start, end = compute_foreground_bbox(anchor_t0_xyz, self.foreground_margin_voxels)
            if any(e <= s for s, e in zip(start, end)):
                logger.warning(
                    "%s T0: foreground bbox is degenerate (start=%s end=%s) — "
                    "skipping crop, keeping full resampled volume",
                    pid, start, end,
                )
            else:
                t0_arr          = crop_volume(t0_arr,           start, end)
                it_arr          = crop_volume(it_arr,           start, end)
                pt1_arr         = crop_volume(pt1_arr,          start, end)
                pt2_arr         = crop_volume(pt2_arr,          start, end)
                pt3_arr         = crop_volume(pt3_arr,          start, end)
                tvi_arr         = crop_volume(tvi_arr,          start, end)
                liver_t0_arr    = crop_volume(liver_t0_arr,     start, end)
                pancreas_t0_arr = crop_volume(pancreas_t0_arr,  start, end)
                kidneys_t0_arr  = crop_volume(kidneys_t0_arr,   start, end)
                start_t0_kept, end_t0_kept = tuple(start), tuple(end)
                if anchor_source_t0 != "tumor":
                    logger.warning(
                        "%s T0: no tumour mask, falling back to %s for crop anchor",
                        pid, anchor_source_t0,
                    )
        elif self.crop_to_foreground:
            logger.error(
                "%s T0: no tumour AND no pancreas mask — using center crop "
                "of full resampled volume. This case is likely UNUSABLE for "
                "training; consider excluding it.", pid,
            )

        w.update({
            "t0_arr": t0_arr, "it_arr": it_arr,
            "pt1_arr": pt1_arr, "pt2_arr": pt2_arr, "pt3_arr": pt3_arr, "tvi_arr": tvi_arr,
            "liver_t0_arr": liver_t0_arr, "pancreas_t0_arr": pancreas_t0_arr,
            "kidneys_t0_arr": kidneys_t0_arr,
            "anchor_t0_xyz": anchor_t0_xyz, "anchor_source_t0": anchor_source_t0,
            "start_t0_kept": start_t0_kept, "end_t0_kept": end_t0_kept,
        })

    def _align_and_crop_t1(self, w: Dict[str, Any], pid: str) -> None:
        """Anchor the T1 bbox to T0's via a landmark offset, then crop T1 arrays.

        Preference: bilateral kidneys (30 mm drift cap), then pancreas (60 mm),
        then T1's own tumour, then a world-space projection of the T0 anchor.
        """
        t0_nib, t1_nib = w["t0_nib"], w["t1_nib"]
        t1_arr = w["t1_arr"]
        it1_arr = w["it1_arr"]
        pt1_t1_arr, pt2_t1_arr, pt3_t1_arr = w["pt1_t1_arr"], w["pt2_t1_arr"], w["pt3_t1_arr"]
        tvi_t1_arr = w["tvi_t1_arr"]
        liver_t1_arr = w["liver_t1_arr"]
        pancreas_t1_arr = w["pancreas_t1_arr"]
        kidneys_t1_arr = w["kidneys_t1_arr"]
        it1_arr_largest = w["it1_arr_largest"]
        anchor_t0_xyz = w["anchor_t0_xyz"]
        anchor_source_t0 = w["anchor_source_t0"]
        start_t0_kept, end_t0_kept = w["start_t0_kept"], w["end_t0_kept"]
        pancreas_t0_centroid = w["pancreas_t0_centroid"]
        pancreas_t1_centroid = w["pancreas_t1_centroid"]
        kidney_t0_centroid = w["kidney_t0_centroid"]
        kidney_t1_centroid = w["kidney_t1_centroid"]

        # T1 crop anchor. With shared_crop_frame the T1 bbox is the T0 bbox
        # projected to T1 voxel space, so both patches view the same world region;
        # otherwise each timepoint is cropped on its own anchor (below).
        anchor_t1_xyz = None
        anchor_source_t1 = "none"
        start_t1_kept: Optional[Tuple[int, int, int]] = None
        end_t1_kept:   Optional[Tuple[int, int, int]] = None

        def _resampled_affine(orig_nib: nib.Nifti1Image) -> np.ndarray:
            """Return the 4x4 affine matching `resample_volume`'s output space.

            Same origin and direction as `orig_nib`; columns rescaled to
            `target_spacing_mm`.
            """
            A = orig_nib.affine.copy().astype(np.float64)
            orig_zooms = np.array(orig_nib.header.get_zooms()[:3], dtype=np.float64)
            tgt = np.array(self.target_spacing_mm, dtype=np.float64)
            for i in range(3):
                if orig_zooms[i] > 0:
                    A[:3, i] *= tgt[i] / orig_zooms[i]
            return A

        def _try_centroid_alignment(
            c0: np.ndarray, c1: np.ndarray,
            kind: str, drift_max_mm: float,
        ) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]]:
            """Project the T0 crop bbox onto T1 via a landmark centroid offset.

            Rejects the alignment if T0->T1 centroid drift exceeds `drift_max_mm`.

            Args:
                c0: Landmark centroid in T0 resampled voxel space.
                c1: Landmark centroid in T1 resampled voxel space.
                kind: Landmark name, used in logging and the anchor-source tag.
                drift_max_mm: Maximum plausible centroid drift, in mm.

            Returns:
                `(start, end, anchor_source)`, or None if the drift check
                or resulting bbox failed.
            """
            _vox_mm = float(self.target_spacing_mm[0])
            _drift_mm = float(np.linalg.norm(c1 - c0)) * _vox_mm
            if _drift_mm > drift_max_mm:
                logger.warning(
                    "%s %s centroid drift %.1f mm > %.0f mm — implausible, "
                    "treating as truncated mask.",
                    pid, kind, _drift_mm, drift_max_mm,
                )
                return None
            s0 = np.asarray(start_t0_kept, dtype=np.float64)
            e0 = np.asarray(end_t0_kept,   dtype=np.float64)
            bbox_center_t0 = (s0 + e0) / 2.0
            bbox_size      = e0 - s0
            offset = bbox_center_t0 - c0
            bbox_center_t1 = c1 + offset
            sf = bbox_center_t1 - bbox_size / 2.0
            ef = bbox_center_t1 + bbox_size / 2.0
            t1_shape = np.array(t1_arr.shape, dtype=np.int64)
            sa = np.clip(np.floor(sf).astype(np.int64), 0, t1_shape)
            ea = np.clip(np.ceil (ef).astype(np.int64), 0, t1_shape)
            if any(ea[i] <= sa[i] for i in range(3)):
                logger.warning(
                    "%s %s-aligned T1 bbox degenerate (start=%s end=%s).",
                    pid, kind, sa.tolist(), ea.tolist(),
                )
                return None
            sk = tuple(int(v) for v in sa)
            ek = tuple(int(v) for v in ea)
            logger.info(
                "%s %s-aligned T1: c0=%s c1=%s drift=%.1fmm offset=%s bbox=%s..%s",
                pid, kind, c0.round(1).tolist(), c1.round(1).tolist(),
                _drift_mm, offset.round(1).tolist(), sk, ek,
            )
            return sk, ek, f"{kind}_aligned_from_t0({anchor_source_t0})"

        if (self.shared_crop_frame and self.crop_to_foreground
                and start_t0_kept is not None and end_t0_kept is not None):
            # 1) Try kidneys
            if kidney_t0_centroid is not None and kidney_t1_centroid is not None:
                _r = _try_centroid_alignment(
                    kidney_t0_centroid, kidney_t1_centroid, "kidney", 30.0,
                )
                if _r is not None:
                    start_t1_kept, end_t1_kept, anchor_source_t1 = _r
            # 2) Fall back to pancreas
            if (start_t1_kept is None
                    and pancreas_t0_centroid is not None
                    and pancreas_t1_centroid is not None):
                _r = _try_centroid_alignment(
                    pancreas_t0_centroid, pancreas_t1_centroid, "pancreas", 60.0,
                )
                if _r is not None:
                    start_t1_kept, end_t1_kept, anchor_source_t1 = _r

        # Per-timepoint anchor: only when shared-frame is off or its projection
        # gave no bbox. Anchor on the largest CC, as with T0.
        if start_t1_kept is None:
            if it1_arr_largest.any():
                anchor_t1_xyz = it1_arr_largest.astype(np.float32)
                anchor_source_t1 = "tumor"
            else:
                try:
                    seg_t1_path = self.cache_dir / f"{pid}_seg_T1.nii.gz"
                    if seg_t1_path.exists():
                        _seg_T1 = (
                            nib.load(str(seg_t1_path)).get_fdata().astype(np.int16)
                        )
                        pmask = _pancreas_anatomy_mask(_seg_T1)
                        if pmask.any():
                            pxyz, _ = resample_volume(
                                _arr_to_nib(pmask.astype(np.float32), t1_nib),
                                self.target_spacing_mm, is_mask=True,
                            )
                            if pxyz.any():
                                anchor_t1_xyz = pxyz
                                anchor_source_t1 = "pancreas_t1"
                except Exception as _exc:
                    logger.debug("Pancreas T1 fallback unavailable for %s: %s", pid, _exc)

        if start_t1_kept is None and anchor_t1_xyz is None and anchor_t0_xyz is not None:
            # The T0 anchor lives in T0's voxel grid; project its centroid through
            # world space into T1's grid.
            try:
                affine_t0_rs = _resampled_affine(t0_nib)
                affine_t1_rs = _resampled_affine(t1_nib)

                vox_t0 = np.argwhere(anchor_t0_xyz).mean(axis=0).astype(np.float64)
                world = affine_t0_rs[:3, :3] @ vox_t0 + affine_t0_rs[:3, 3]
                inv_t1 = np.linalg.inv(affine_t1_rs)
                vox_t1 = inv_t1[:3, :3] @ world + inv_t1[:3, 3]

                vox_t1_int = np.clip(
                    np.round(vox_t1).astype(int), 0,
                    np.array(t1_arr.shape, dtype=int) - 1,
                )
                synthetic_anchor = np.zeros(t1_arr.shape, dtype=np.float32)
                synthetic_anchor[tuple(vox_t1_int)] = 1.0
                anchor_t1_xyz   = synthetic_anchor
                anchor_source_t1 = f"{anchor_source_t0}_projected_from_t0"
                logger.debug(
                    "%s T1 anchor projected: T0 vox %s → world %s → T1 vox %s",
                    pid, vox_t0.round(1), world.round(1), vox_t1_int,
                )
            except Exception as _proj_exc:
                logger.warning(
                    "%s T1: world-space projection of T0 anchor failed (%s); "
                    "T1 crop will use full-volume centre.", pid, _proj_exc,
                )
                anchor_t1_xyz   = None
                anchor_source_t1 = "projection_failed"

        # If the shared-frame path gave no bbox, derive one from anchor_t1_xyz.
        if (start_t1_kept is None
                and self.crop_to_foreground and anchor_t1_xyz is not None):
            s1, e1 = compute_foreground_bbox(anchor_t1_xyz, self.foreground_margin_voxels)
            if any(ev <= sv for sv, ev in zip(s1, e1)):
                logger.warning(
                    "%s T1: foreground bbox is degenerate (start=%s end=%s) — "
                    "skipping crop, keeping full resampled volume",
                    pid, s1, e1,
                )
            else:
                start_t1_kept, end_t1_kept = tuple(s1), tuple(e1)

        if start_t1_kept is not None and end_t1_kept is not None:
            t1_arr          = crop_volume(t1_arr,          start_t1_kept, end_t1_kept)
            it1_arr         = crop_volume(it1_arr,         start_t1_kept, end_t1_kept)
            pt1_t1_arr      = crop_volume(pt1_t1_arr,      start_t1_kept, end_t1_kept)
            pt2_t1_arr      = crop_volume(pt2_t1_arr,      start_t1_kept, end_t1_kept)
            pt3_t1_arr      = crop_volume(pt3_t1_arr,      start_t1_kept, end_t1_kept)
            tvi_t1_arr      = crop_volume(tvi_t1_arr,      start_t1_kept, end_t1_kept)
            liver_t1_arr    = crop_volume(liver_t1_arr,    start_t1_kept, end_t1_kept)
            pancreas_t1_arr = crop_volume(pancreas_t1_arr, start_t1_kept, end_t1_kept)
            kidneys_t1_arr  = crop_volume(kidneys_t1_arr,  start_t1_kept, end_t1_kept)
            if anchor_source_t1 != "tumor" and not anchor_source_t1.startswith("shared_frame"):
                logger.warning("%s T1: no tumour mask, anchored on %s for crop",
                               pid, anchor_source_t1)
        logger.info(
            "%s T1 crop anchor: %s  bbox=%s..%s",
            pid, anchor_source_t1,
            start_t1_kept, end_t1_kept,
        )

        w.update({
            "t1_arr": t1_arr, "it1_arr": it1_arr,
            "pt1_t1_arr": pt1_t1_arr, "pt2_t1_arr": pt2_t1_arr, "pt3_t1_arr": pt3_t1_arr,
            "tvi_t1_arr": tvi_t1_arr, "liver_t1_arr": liver_t1_arr,
            "pancreas_t1_arr": pancreas_t1_arr, "kidneys_t1_arr": kidneys_t1_arr,
        })

    def _pad_to_patch(self, w: Dict[str, Any], pid: str) -> None:
        """Pad/crop every array to `patch_size`; build the CT validity masks."""
        t0_arr, t1_arr = w["t0_arr"], w["t1_arr"]
        it_arr, it1_arr = w["it_arr"], w["it1_arr"]
        liver_t0_arr, liver_t1_arr = w["liver_t0_arr"], w["liver_t1_arr"]
        pancreas_t0_arr, pancreas_t1_arr = w["pancreas_t0_arr"], w["pancreas_t1_arr"]
        kidneys_t0_arr, kidneys_t1_arr = w["kidneys_t0_arr"], w["kidneys_t1_arr"]
        pt1_arr, pt2_arr, pt3_arr, tvi_arr = w["pt1_arr"], w["pt2_arr"], w["pt3_arr"], w["tvi_arr"]
        pt1_t1_arr, pt2_t1_arr, pt3_t1_arr = w["pt1_t1_arr"], w["pt2_t1_arr"], w["pt3_t1_arr"]
        tvi_t1_arr = w["tvi_t1_arr"]
        new_spacing = w["new_spacing"]

        # CTs pad with per-volume min (≈ air) to match encoder pretraining; masks zero-pad.
        # valid_mask marks real content vs padding.
        valid_mask_t0 = pad_or_crop_to_patch_size(
            np.ones_like(t0_arr, dtype=np.float32),
            self.patch_size, mode="constant", constant_value=0.0,
        ).astype(bool)
        valid_mask_t1 = pad_or_crop_to_patch_size(
            np.ones_like(t1_arr, dtype=np.float32),
            self.patch_size, mode="constant", constant_value=0.0,
        ).astype(bool)
        # Pad value = per-volume min; fall back to clip_min on a degenerate
        # zero-size array rather than crashing on .min().
        def _safe_pad_val(arr: np.ndarray, label: str) -> float:
            """Return `arr`'s min as the pad value, or `ct_clip_min` if empty."""
            if arr.size > 0:
                return float(arr.min())
            logger.warning(
                "%s %s: zero-size array reached pad step (shape=%s) — "
                "using clip_min as pad value", pid, label, arr.shape,
            )
            return float(self.ct_clip_min)
        t0_pad_val = _safe_pad_val(t0_arr, "T0")
        t1_pad_val = _safe_pad_val(t1_arr, "T1")
        t0_arr = pad_or_crop_to_patch_size(
            t0_arr, self.patch_size, mode="constant", constant_value=t0_pad_val,
        )
        t1_arr = pad_or_crop_to_patch_size(
            t1_arr, self.patch_size, mode="constant", constant_value=t1_pad_val,
        )
        it_arr      = pad_or_crop_to_patch_size(it_arr,      self.patch_size)
        it1_arr     = pad_or_crop_to_patch_size(it1_arr,     self.patch_size)
        liver_t0_arr    = pad_or_crop_to_patch_size(liver_t0_arr,    self.patch_size)
        liver_t1_arr    = pad_or_crop_to_patch_size(liver_t1_arr,    self.patch_size)
        pancreas_t0_arr = pad_or_crop_to_patch_size(pancreas_t0_arr, self.patch_size)
        pancreas_t1_arr = pad_or_crop_to_patch_size(pancreas_t1_arr, self.patch_size)
        kidneys_t0_arr  = pad_or_crop_to_patch_size(kidneys_t0_arr,  self.patch_size)
        kidneys_t1_arr  = pad_or_crop_to_patch_size(kidneys_t1_arr,  self.patch_size)
        pt1_arr     = pad_or_crop_to_patch_size(pt1_arr,     self.patch_size)
        pt2_arr     = pad_or_crop_to_patch_size(pt2_arr,     self.patch_size)
        pt3_arr     = pad_or_crop_to_patch_size(pt3_arr,     self.patch_size)
        tvi_arr     = pad_or_crop_to_patch_size(tvi_arr,     self.patch_size)
        pt1_t1_arr  = pad_or_crop_to_patch_size(pt1_t1_arr,  self.patch_size)
        pt2_t1_arr  = pad_or_crop_to_patch_size(pt2_t1_arr,  self.patch_size)
        pt3_t1_arr  = pad_or_crop_to_patch_size(pt3_t1_arr,  self.patch_size)
        tvi_t1_arr  = pad_or_crop_to_patch_size(tvi_t1_arr,  self.patch_size)

        logger.info(
            "%s: shape=%s  spacing=%.2f×%.2f×%.2f mm  "
            "IT_t0=%d  IT_t1=%d  TVI_t0=%d  TVI_t1=%d vox",
            pid, t0_arr.shape, *new_spacing,
            it_arr.sum(), it1_arr.sum(), tvi_arr.sum(), tvi_t1_arr.sum(),
        )

        w.update({
            "t0_arr": t0_arr, "t1_arr": t1_arr, "it_arr": it_arr, "it1_arr": it1_arr,
            "liver_t0_arr": liver_t0_arr, "liver_t1_arr": liver_t1_arr,
            "pancreas_t0_arr": pancreas_t0_arr, "pancreas_t1_arr": pancreas_t1_arr,
            "kidneys_t0_arr": kidneys_t0_arr, "kidneys_t1_arr": kidneys_t1_arr,
            "pt1_arr": pt1_arr, "pt2_arr": pt2_arr, "pt3_arr": pt3_arr, "tvi_arr": tvi_arr,
            "pt1_t1_arr": pt1_t1_arr, "pt2_t1_arr": pt2_t1_arr, "pt3_t1_arr": pt3_t1_arr,
            "tvi_t1_arr": tvi_t1_arr,
            "valid_mask_t0": valid_mask_t0, "valid_mask_t1": valid_mask_t1,
        })

    def _maybe_deformable_register_t1(self, w: Dict[str, Any], pid: str) -> None:
        """Warp the padded T1 arrays into T0 space with LocalNet, if enabled.

        No-op unless registration.use_deformable_registration is set. T0 and T1
        are patch-sized here, so a per-pair fit gives a DDF that warps the T1
        CT and its masks into T0 space.
        """
        rc = getattr(self, "_registration_cfg", None) or {}
        if not rc.get("use_deformable_registration", False):
            return

        import torch
        from pdac_longitudinal.registration.deformable_registration import (
            DeformableRegistration,
        )

        ln = dict(rc.get("localnet", {}))
        device = torch.device(
            rc.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        reg = DeformableRegistration(
            num_channel_initial=int(ln.get("num_channel_initial", 16)),
            extract_levels=tuple(ln.get("extract_levels", (0, 1, 2, 3))),
            lncc_kernel_size=int(ln.get("lncc_kernel_size", 9)),
            reg_weight=float(ln.get("reg_weight", 1.0)),
        ).to(device)

        def _t(a: np.ndarray) -> "torch.Tensor":
            return torch.as_tensor(
                np.ascontiguousarray(a), dtype=torch.float32, device=device
            )[None, None]

        fixed  = _t(w["t0_arr"])
        moving = _t(w["t1_arr"])
        iters = int(rc.get("iterations", 100))
        _, ddf = reg.fit_pair(moving, fixed, iterations=iters, lr=float(rc.get("lr", 1e-3)))

        warped_ct = reg.warp_image(moving, ddf)[0, 0].detach().cpu().numpy()
        w["t1_arr"] = warped_ct.astype(w["t1_arr"].dtype)

        mask_keys = (
            "it1_arr", "liver_t1_arr", "pancreas_t1_arr", "kidneys_t1_arr",
            "pt1_t1_arr", "pt2_t1_arr", "pt3_t1_arr", "tvi_t1_arr", "valid_mask_t1",
        )
        for k in mask_keys:
            if k not in w:
                continue
            warped = reg.warp_mask(_t(w[k].astype(np.float32)), ddf)[0, 0].detach().cpu().numpy()
            w[k] = warped.astype(w[k].dtype)

        logger.info("%s: deformable T1->T0 registration applied (%d iters)", pid, iters)

    def _assemble_arrays(
        self, w: Dict[str, Any], pid: str,
        anat_feats: Dict[str, float], vessel_feats: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        """Pack the padded arrays + feature/phase payloads into the cache dict."""
        # JSON blobs in uint8 avoids allow_pickle=True on load
        _anat_json    = json.dumps(anat_feats).encode("utf-8")
        _vessel_json  = json.dumps(vessel_feats).encode("utf-8")
        anatomy_payload = np.frombuffer(_anat_json,   dtype=np.uint8)
        vessel_payload  = np.frombuffer(_vessel_json, dtype=np.uint8)

        arrays = {
            "t0": w["t0_arr"], "t1": w["t1_arr"],
            "mask_it":    w["it_arr"].astype(np.uint8),
            "mask_it_t1": w["it1_arr"].astype(np.uint8),
            "liver_t0":    w["liver_t0_arr"].astype(np.uint8),
            "liver_t1":    w["liver_t1_arr"].astype(np.uint8),
            "pancreas_t0": w["pancreas_t0_arr"].astype(np.uint8),
            "pancreas_t1": w["pancreas_t1_arr"].astype(np.uint8),
            "kidneys_t0":  w["kidneys_t0_arr"].astype(np.uint8),
            "kidneys_t1":  w["kidneys_t1_arr"].astype(np.uint8),
            "mask_pt1": w["pt1_arr"].astype(np.uint8),
            "mask_pt2": w["pt2_arr"].astype(np.uint8),
            "mask_pt3": w["pt3_arr"].astype(np.uint8),
            "mask_tvi": w["tvi_arr"].astype(np.uint8),
            "mask_pt1_t1": w["pt1_t1_arr"].astype(np.uint8),
            "mask_pt2_t1": w["pt2_t1_arr"].astype(np.uint8),
            "mask_pt3_t1": w["pt3_t1_arr"].astype(np.uint8),
            "mask_tvi_t1": w["tvi_t1_arr"].astype(np.uint8),
            # 1 = real content, 0 = padding; masks padded keys in cross-attention.
            "valid_t0": w["valid_mask_t0"].astype(np.uint8),
            "valid_t1": w["valid_mask_t1"].astype(np.uint8),
            # Resolved phase, stored as bytes to survive the .npz round-trip.
            "phase_used": np.frombuffer(
                (self._resolve_phase(pid) or self.phase).encode("utf-8"), dtype=np.uint8,
            ),
            "anatomy_features_json": anatomy_payload,
            "vessel_features_json":  vessel_payload,
        }
        return arrays
