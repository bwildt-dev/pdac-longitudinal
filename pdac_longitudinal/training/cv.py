"""K-fold cross-validation orchestration."""

from __future__ import annotations

import gc
import hashlib
import logging
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from pdac_longitudinal.config import Config
from pdac_longitudinal.data.longitudinal_dataset import LongitudinalCTDataset
from pdac_longitudinal.data.registry import ClinicalRegistry
from pdac_longitudinal.preprocess.segmenter import PanTSSegmenter

logger = logging.getLogger(__name__)


def build_dataset_kwargs(cfg: Config, registry: ClinicalRegistry) -> dict:
    """Return kwargs shared by all three dataset splits (train/val/test)."""
    dc  = cfg.data
    pc  = cfg.preprocessing

    return dict(
        nifti_root             = dc.nifti_dir or dc.root_dir,
        registry               = registry,
        cache_dir              = dc.cache_dir,
        weights_path           = dc.segmenter_weights_path,
        phase                  = dc.phase,
        patch_size             = tuple(pc.patch_size),
        target_spacing_mm      = tuple(pc.target_spacing),
        ct_clip_min            = pc.ct_clip_min,
        ct_clip_max            = pc.ct_clip_max,
        ct_norm_mean           = pc.ct_norm_mean,
        ct_norm_std            = pc.ct_norm_std,
        ct_norm_shared_pair    = pc.ct_norm_shared_pair,
        crop_to_foreground     = pc.crop_to_foreground,
        foreground_margin_voxels = pc.foreground_margin_voxels,
        shared_crop_frame      = pc.shared_crop_frame,
        reuse_saved_segs       = pc.reuse_saved_segs,
        viz_cache_every        = pc.viz_cache_every,
        allowed_regions        = list(dc.allowed_regions),
        post_nat_tps           = list(dc.post_nat_tps),
        segmenter_device       = cfg.training.device,
        cache_version          = dc.cache_version,
        augmentation           = cfg.augmentation,
        pants_target_spacing_zyx = tuple(pc.pants_target_spacing_zyx),
        pants_patch_zyx        = tuple(pc.pants_patch_zyx),
        pants_step             = pc.pants_step,
        registration           = dict(cfg.registration),
    )


def build_shared_segmenter(cfg: Config) -> Optional[PanTSSegmenter]:
    """Build a single shared PanTS segmenter if a weights path is set."""
    dc  = cfg.data
    roi = cfg.roi_pipeline
    weights_path = dc.segmenter_weights_path
    if weights_path is None:
        return None

    seg_cfg = roi.get("segmenter", {})
    return PanTSSegmenter(
        weights_path=weights_path,
        device=cfg.training.device,
        use_mirroring=bool(seg_cfg.get("use_mirroring", False)),
        use_gaussian=bool(seg_cfg.get("use_gaussian", False)),
        perform_everything_on_device=False,
    )


def release_segmenter(
    segmenter: Optional[PanTSSegmenter],
    datasets: List[LongitudinalCTDataset],
) -> None:
    """Release the shared segmenter and free GPU VRAM before training starts."""
    if segmenter is not None:
        segmenter.release()
    for ds in datasets:
        ds._segmenter = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_gib = torch.cuda.mem_get_info()[0] / 1024 ** 3
        logger.info("Segmenter released — GPU free: %.1f GiB", free_gib)


def shard_filter_fn(shard_spec: str) -> Callable[[str], bool]:
    """Parse `"K/N"` shard spec and return a filter function `pid -> bool`.

    Raises SystemExit on a malformed spec.
    """
    try:
        k_str, n_str = shard_spec.split("/")
        shard_k, shard_n = int(k_str), int(n_str)
        assert 0 <= shard_k < shard_n
    except Exception:
        raise SystemExit(f"--shard must be 'K/N' with 0 <= K < N, got {shard_spec!r}")

    def _filter(pid: str) -> bool:
        return int(hashlib.md5(pid.encode()).hexdigest(), 16) % shard_n == shard_k

    return _filter


def cache_all_cases(
    datasets: List[LongitudinalCTDataset],
    max_seg_tiles: Optional[int] = None,
    shard_fn: Optional[Callable[[str], bool]] = None,
    use_wandb: bool = False,
) -> int:
    """Pre-cache all cases across *datasets* before DataLoader workers start."""
    from pdac_longitudinal.training.wandb_setup import log_seg_overlay

    if shard_fn is not None:
        for ds in datasets:
            n_before = len(ds.cases)
            ds.cases = [c for c in ds.cases if shard_fn(c["patient_id"])]
            logger.info(
                "Shard filter: %s subset %d → %d cases",
                ds.__class__.__name__, n_before, len(ds.cases),
            )

    total_cases = sum(len(ds) for ds in datasets)
    n_cached = 0
    logger.info("Pre-caching %d cases before starting DataLoader workers...", total_cases)

    for ds in datasets:
        for case in list(ds.cases):
            pid = case["patient_id"]
            cp  = ds._cache_path(pid)
            if cp is None or not cp.exists():
                logger.info("  Caching %s ...", pid)
                try:
                    arrays = ds._preprocess_and_cache(case, max_seg_tiles=max_seg_tiles)
                except Exception as exc:
                    logger.error("Failed to preprocess %s: %s", pid, exc, exc_info=True)
                    ds.cases = [c for c in ds.cases if c["patient_id"] != pid]
                    total_cases -= 1
                    continue
                if use_wandb:
                    log_seg_overlay(pid, arrays)
            n_cached += 1
            logger.info("  [%d/%d] done", n_cached, total_cases)

    logger.info("All cases cached.")
    return n_cached


def build_cv_loaders(
    cfg: Config,
    registry: ClinicalRegistry,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    segmenter: Optional[PanTSSegmenter] = None,
    max_seg_tiles: Optional[int] = None,
    shard_fn: Optional[Callable[[str], bool]] = None,
    cache_only: bool = False,
    use_wandb: bool = False,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Build DataLoaders for one training run (or one CV fold).

    Returns `(None, None, None)` instead of loaders when `cache_only=True`.
    """
    tc = cfg.training
    common_kw = build_dataset_kwargs(cfg, registry)

    train_set = LongitudinalCTDataset(
        **common_kw, augment=True,  patient_ids=train_ids, segmenter=segmenter,
    )
    val_set = LongitudinalCTDataset(
        **common_kw, augment=False, patient_ids=val_ids,   segmenter=segmenter,
    )
    test_set = LongitudinalCTDataset(
        **common_kw, augment=False, patient_ids=test_ids,  segmenter=segmenter,
    )

    cache_all_cases(
        [train_set, val_set, test_set],
        max_seg_tiles=max_seg_tiles,
        shard_fn=shard_fn,
        use_wandb=use_wandb,
    )

    if cache_only:
        release_segmenter(segmenter, [train_set, val_set, test_set])
        return None, None, None

    release_segmenter(segmenter, [train_set, val_set, test_set])

    # Fold-internal radiomic normalisation (signed-log -> z-score -> optional PCA):
    # fit on train only, then apply the same scaler to val/test.
    if cfg.modules.radiomics:
        scaler = train_set.fit_radiomic_scaler(
            n_components=getattr(cfg.data, "radiomic_pca_components", 0))
        for ds in (val_set, test_set):
            ds.set_radiomic_scaler(scaler)

    loader_kw = dict(
        num_workers=tc.num_workers,
        pin_memory=(tc.device != "cpu" if tc.device else torch.cuda.is_available()),
        persistent_workers=(tc.num_workers > 0),
    )
    train_loader = DataLoader(train_set, batch_size=tc.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   batch_size=tc.batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_set,  batch_size=tc.batch_size, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader
