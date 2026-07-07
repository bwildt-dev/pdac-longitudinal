"""Generic longitudinal PDAC dataset: paired (T0, T1) CTs with survival labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pdac_longitudinal.config import AugmentationConfig

from pdac_longitudinal.preprocess.anatomy_features import (
    features_dict_to_vector as anatomy_features_to_vector,
)
from pdac_longitudinal.preprocess.vessel_features import (
    features_dict_to_vector as vessel_features_to_vector,
)
from pdac_longitudinal.radiomics.feature_schema import (
    RadiomicScaler,
    decode_radiomic_features,
    radiomic_dict_to_vector,
    signed_log,
)
from pdac_longitudinal.data.augmentation import augment_sample
from pdac_longitudinal.preprocess.cache_pipeline import CachePipelineMixin
from pdac_longitudinal.data.case_discovery import CaseDiscoveryMixin
from pdac_longitudinal.preprocess.mask_utils import (
    decode_anatomy_features,
    decode_vessel_features,
)
from pdac_longitudinal.data.registry import ClinicalRegistry
from pdac_longitudinal.preprocess.segmenter import PanTSSegmenter

logger = logging.getLogger(__name__)


_CACHE_VERSION = "v3.5"


# Dataset

class LongitudinalCTDataset(CachePipelineMixin, CaseDiscoveryMixin, Dataset):
    """Paired (T0, T1) PDAC CTs with survival labels.

    Args:
        nifti_root: Root directory of the NIfTI dataset.
        registry: Clinical registry supplying survival labels and tensors.
        cache_dir: Directory for cached preprocessing outputs; None disables caching.
        weights_path: PanTS segmenter weights, required for on-the-fly segmentation.
        phase: One phase, or candidate phases in priority order.
        patch_size: Output patch size (X, Y, Z) after crop/pad.
        target_spacing_mm: Resampling target spacing in mm (X, Y, Z).
        ct_clip_min: Lower HU clip bound.
        ct_clip_max: Upper HU clip bound.
        ct_norm_mean: Fixed global normalisation mean; None computes per-case stats.
        ct_norm_std: Fixed global normalisation std; None computes per-case stats.
        ct_norm_shared_pair: Pool T0/T1 foreground statistics for shared normalisation.
        crop_to_foreground: Crop around the anatomical anchor.
        foreground_margin_voxels: Margin added around the foreground bbox.
        shared_crop_frame: Project the T0 crop bbox onto T1 via a landmark offset.
        reuse_saved_segs: Reuse a previously saved PanTS segmentation.
        viz_cache_every: Render an ROI-overlay PNG every Nth case; 0 disables it.
        augment: Apply training-time augmentation in `__getitem__`.
        patient_ids: Restrict case discovery to these patient IDs.
        segmenter_device: Device for on-the-fly PanTS inference.
        segmenter: Pre-built `PanTSSegmenter` to reuse.
        allowed_regions: Composer body-region labels accepted when discovering cases.
        classification_state: Path to the composer `classification_state.json`.
        post_nat_tps: Post-neoadjuvant-therapy timepoint tags considered for T1.
        cache_version: Cache-busting tag included in cache filenames.
        augmentation: Augmentation config.
        pants_target_spacing_zyx: PanTS inference spacing (Z, Y, X).
        pants_patch_zyx: PanTS inference patch size (Z, Y, X).
        pants_step: PanTS sliding-window step fraction.
    """

    def __init__(
        self,
        nifti_root: Path,
        registry: ClinicalRegistry,
        cache_dir: Optional[Path] = None,
        weights_path: Optional[Path] = None,
        phase: "str | Sequence[str]" = "venous",
        patch_size: Tuple[int, int, int] = (192, 192, 192),
        target_spacing_mm: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        ct_clip_min: float = -150.0,
        ct_clip_max: float =  250.0,
        ct_norm_mean: Optional[float] = None,
        ct_norm_std:  Optional[float] = None,
        ct_norm_shared_pair: bool = True,
        crop_to_foreground: bool = True,
        foreground_margin_voxels: int = 100,
        shared_crop_frame: bool = True,
        reuse_saved_segs: bool = True,
        viz_cache_every: int = 0,
        augment: bool = False,
        patient_ids: Optional[List[str]] = None,
        segmenter_device: Optional[str] = None,
        segmenter: Optional[PanTSSegmenter] = None,
        allowed_regions: Optional[List[str]] = None,
        classification_state: Optional[Path] = None,
        post_nat_tps: Optional[List[str]] = None,
        cache_version: str = _CACHE_VERSION,
        augmentation: "Optional[AugmentationConfig]" = None,
        pants_target_spacing_zyx: Tuple[float, float, float] = (1.0, 0.787, 0.801),
        pants_patch_zyx: Tuple[int, int, int] = (128, 192, 288),
        pants_step: float = 0.5,
        registration: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.nifti_root = nifti_root
        self.registry   = registry
        self.cache_dir     = Path(cache_dir) if cache_dir else None
        self.cache_version = cache_version
        self.phase_preference: List[str] = (
            [phase] if isinstance(phase, str) else list(phase)
        )
        self.phase      = self.phase_preference[0]
        self._phase_for_pid: Dict[str, str] = {}   # resolved per patient (cached)
        self.patch_size = patch_size
        self.target_spacing_mm = target_spacing_mm
        self.ct_clip_min = ct_clip_min
        self.ct_clip_max = ct_clip_max
        self.ct_norm_mean        = ct_norm_mean
        self.ct_norm_std         = ct_norm_std
        self.ct_norm_shared_pair = ct_norm_shared_pair
        self.crop_to_foreground = crop_to_foreground
        self.foreground_margin_voxels = foreground_margin_voxels
        self.shared_crop_frame = shared_crop_frame
        self.reuse_saved_segs = reuse_saved_segs
        self.viz_cache_every = int(viz_cache_every)
        self._viz_counter = 0
        self.augment = augment
        self._weights_path = weights_path
        self._segmenter_device = segmenter_device
        self._segmenter: Optional[PanTSSegmenter] = segmenter

        self._pants_target_spacing_zyx = pants_target_spacing_zyx
        self._pants_patch_zyx = pants_patch_zyx
        self._pants_step = pants_step
        self._registration_cfg = dict(registration or {})
        from pdac_longitudinal.config import AugmentationConfig
        self._aug_cfg = augmentation or AugmentationConfig()

        if not allowed_regions:
            raise ValueError("allowed_regions is required (set data.allowed_regions in config).")
        self.allowed_regions = list(allowed_regions)
        self.post_nat_tps = list(post_nat_tps or ["t1"])

        # keyed by basename so scratch and output paths don't collide
        self._region_by_name: Dict[str, str] = {}
        if classification_state is None:
            classification_state = self.nifti_root / "_state" / "classification_state.json"
        if classification_state.exists():
            with open(classification_state) as f:
                clf = json.load(f)
            self._region_by_name = {
                Path(rec["nifti_path"]).name: rec.get("body_region", "")
                for rec in clf.values()
            }
            logger.info(
                "Loaded %d classification records from %s (keyed by filename)",
                len(self._region_by_name), classification_state,
            )
        else:
            logger.warning(
                "classification_state.json not found at %s — region filter disabled "
                "(all NIfTIs in `phase` dir will be used)",
                classification_state,
            )

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cases = self._discover_cases(patient_ids)
        # fold-internal scaler fit on train, shared to val/test; None = bare signed-log
        self._radiomic_scaler: Optional[RadiomicScaler] = None
        n_events = sum(c["event"] for c in self.cases)
        logger.info(
            "%s: %d cases (phase=%s augment=%s events=%d censored=%d)",
            type(self).__name__, len(self.cases), phase, augment,
            n_events, len(self.cases) - n_events,
        )

    # Caching

    def _cache_path(self, patient_id: str) -> Optional[Path]:
        """Return the `.npz` cache path for a patient, or None if caching is off."""
        if self.cache_dir is None:
            return None
        sp = "_".join(f"{s:.2f}" for s in self.target_spacing_mm)
        ps = "_".join(str(p) for p in self.patch_size)
        ph = self._resolve_phase(patient_id) or self.phase
        return self.cache_dir / f"{patient_id}_{ph}_{sp}_{ps}_{self.cache_version}.npz"

    # Cached only for preprocessing-time alignment; skipped on load to save host
    # RAM. Drop a mask from here if you add it to attn_guidance_roi.
    _SKIP_ON_LOAD = frozenset({
        "kidneys_t0", "kidneys_t1", "pancreas_t0", "pancreas_t1",
    })

    def _load_case(self, case: Dict) -> Dict[str, np.ndarray]:
        """Load a case's arrays from cache, or preprocess and cache them.

        Args:
            case: Case record with at least `patient_id`.

        Returns:
            The arrays dict, excluding `_SKIP_ON_LOAD` keys when loaded
            from cache.
        """
        cp = self._cache_path(case["patient_id"])
        if cp is not None and cp.exists():
            # Context manager closes the zipfile handle deterministically (else
            # DataLoader workers leak file descriptors at scale).
            with np.load(cp) as data:
                return {k: data[k] for k in data.files
                        if k not in self._SKIP_ON_LOAD}
        return self._preprocess_and_cache(case)

    # Radiomic fold-internal normalisation (StandardScaler on signed-log)
    def set_radiomic_scaler(self, scaler: Optional[RadiomicScaler]) -> None:
        """Inject the train-fold scaler (so val/test use train stats).

        Args:
            scaler: Fitted scaler from the train fold, or None for bare signed-log.
        """
        self._radiomic_scaler = scaler

    def fit_radiomic_scaler(self, n_components: int = 0) -> Optional[RadiomicScaler]:
        """Fit the fold-internal radiomic normaliser on this dataset's patients.

        Args:
            n_components: PCA components to retain; 0 keeps the full
                signed-log feature space.

        Returns:
            The fitted scaler, or None if no cached radiomic payloads were found.
        """
        rows = []
        for case in self.cases:
            cp = self._cache_path(case["patient_id"])
            if cp is None or not cp.exists():
                continue
            with np.load(cp, allow_pickle=False) as z:
                if "radiomic_features_json" not in z.files:
                    continue
                feats = decode_radiomic_features(
                    {"radiomic_features_json": z["radiomic_features_json"]})
            rows.append(radiomic_dict_to_vector(feats))
        if not rows:
            logger.warning("fit_radiomic_scaler: no radiomic payloads — using bare signed-log.")
            return None
        mat = np.stack(rows)
        self._radiomic_scaler = RadiomicScaler().fit(mat, n_components=n_components)
        logger.info("Radiomic scaler fit on %d train patients (in=%d, out=%d%s).",
                    len(rows), mat.shape[1], self._radiomic_scaler.out_dim,
                    f", PCA k={n_components}" if n_components > 0 else "")
        return self._radiomic_scaler

    # Dataset protocol

    def __len__(self) -> int:
        """Return the number of cases."""
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one sample dict, preprocessing/caching and augmenting as needed.

        Args:
            idx: Case index.

        Returns:
            Dict keyed by: `t0`, `t1` (CT volumes); `mask_it`, `mask_it_t1`,
            `mask_pt1/2/3[_t1]`, `mask_tvi[_t1]`, `liver_t0/t1`,
            `pancreas_t0/t1`, `kidneys_t0/t1` (ROI masks); `valid_t0`,
            `valid_t1` (padding-validity masks); `phase` (int) and
            `phase_name` (str); `anatomy`, `vessel`, `radiomic` (feature
            vectors); `duration`, `event` (survival label); `clinical`
            (registry tensor); and `case_id`.
        """
        case   = self.cases[idx]
        arrays = self._load_case(case)

        if self.augment:
            arrays = augment_sample(arrays, self._aug_cfg)

        # skip JSON byte buffers here; default_collate can't handle variable-length arrays
        _JSON_KEYS = frozenset({
            "anatomy_features_json", "vessel_features_json",
            "radiomic_features_json", "phase_used",
        })
        sample: Dict[str, Any] = {}
        for k, v in arrays.items():
            if k in _JSON_KEYS:
                continue
            # Copies only if augment left a non-contiguous view.
            arr = np.ascontiguousarray(v)
            # masks stay uint8 on host; cast to float on GPU to keep prefetch buffers small
            if k in ("t0", "t1"):
                sample[k] = torch.from_numpy(arr).float().unsqueeze(0)
            elif k in ("valid_t0", "valid_t1"):
                sample[k] = torch.from_numpy(arr).unsqueeze(0)
            else:
                sample[k] = torch.from_numpy(arr)

        # Phase index for the adversary head; decoded from the cached byte buffer.
        phase_str = "arterial"
        if "phase_used" in arrays:
            try:
                phase_str = bytes(arrays["phase_used"]).decode("utf-8")
            except Exception:
                pass
        # Stable integer encoding across runs.
        _PHASE_IDX = {"arterial": 0, "venous": 1}
        sample["phase"]      = torch.tensor(_PHASE_IDX.get(phase_str, 0), dtype=torch.long)
        sample["phase_name"] = phase_str

        anat_feats = decode_anatomy_features(arrays)
        sample["anatomy"] = torch.from_numpy(
            anatomy_features_to_vector(anat_feats)
        ).float()

        vessel_feats = decode_vessel_features(arrays)
        sample["vessel"] = torch.from_numpy(
            vessel_features_to_vector(vessel_feats)
        ).float()

        # radio_raw is T0|T1|Δ concatenated.
        radio_raw = radiomic_dict_to_vector(decode_radiomic_features(arrays))
        radio_vec = (self._radiomic_scaler.transform(radio_raw)
                     if self._radiomic_scaler is not None
                     else signed_log(radio_raw))   # safe fallback if unfit
        sample["radiomic"] = torch.from_numpy(radio_vec).float()

        sample["duration"] = torch.tensor(case["duration"], dtype=torch.float32)
        sample["event"]    = torch.tensor(case["event"],    dtype=torch.float32)
        sample["clinical"] = self.registry.get_clinical_tensor(case["patient_id"])
        sample["case_id"]  = case["patient_id"]
        return sample
