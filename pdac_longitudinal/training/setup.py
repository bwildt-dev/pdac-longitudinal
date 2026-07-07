"""Shared setup for the CLI commands."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from pdac_longitudinal.config import Config
from pdac_longitudinal.preprocess.anatomy_features import ANATOMY_FEATURE_DIM
from pdac_longitudinal.data.registry import ClinicalRegistry
from pdac_longitudinal.data.split import load_or_make_kfolds, make_split
from pdac_longitudinal.preprocess.vessel_features import VESSEL_FEATURE_DIM
from pdac_longitudinal.radiomics.feature_schema import RADIOMIC_FEATURE_DIM
from pdac_longitudinal.training.wandb_setup import init_wandb, update_wandb_config

logger = logging.getLogger(__name__)


@dataclass
class ModuleDims:
    """Per-token feature dimensions implied by the module toggles. 0 means the module is off."""

    clinical: int
    anatomy: int
    vessel: int
    radiomic: int


@dataclass
class RunContext:
    """Everything the commands share after the common preamble."""

    cfg: Config
    device: torch.device
    registry: ClinicalRegistry
    dims: ModuleDims
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    folds: Optional[List[Tuple[List[str], List[str]]]]
    cv_fold: Optional[int]
    phases_by_pid: Dict[str, str]
    use_wandb: bool


# Logging / determinism / device

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: Config) -> torch.device:
    if cfg.training.device:
        return torch.device(cfg.training.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Config overrides

def apply_cli_overrides(
    cfg: Config,
    *,
    cv_fold: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Config:
    """Apply the operational CLI flags to the config."""
    overrides: Dict[str, Any] = {}
    if cv_fold is not None:
        overrides["cv"] = {"enabled": True, "fold": cv_fold}
    if output_dir is not None:
        overrides["training"] = {"output_dir": str(output_dir)}
    return cfg.with_overrides(**overrides) if overrides else cfg


# Module dims / fusion patching

def resolve_module_dims(cfg: Config, registry: ClinicalRegistry) -> ModuleDims:
    """Map the module toggles to concrete per-token feature dimensions."""
    clinical = registry.clinical_dim if cfg.modules.clinical else 0
    anatomy = ANATOMY_FEATURE_DIM if cfg.modules.anatomy else 0
    vessel = VESSEL_FEATURE_DIM if cfg.modules.vessel else 0
    # Fold-internal PCA
    pca_k = int(getattr(cfg.data, "radiomic_pca_components", 0) or 0)
    radiomic = (
        (pca_k if pca_k > 0 else RADIOMIC_FEATURE_DIM) if cfg.modules.radiomics else 0
    )
    return ModuleDims(clinical=clinical, anatomy=anatomy, vessel=vessel, radiomic=radiomic)


def patch_fusion_dims(cfg: Config, dims: ModuleDims) -> Config:
    """Write the runtime token dims into the fusion section."""
    fusion = dict(cfg.fusion)
    fusion["clinical_feature_dim"] = dims.clinical
    fusion["anatomy_feature_dim"] = dims.anatomy or None
    fusion["vessel_feature_dim"] = dims.vessel or None
    fusion["radiomic_feature_dim"] = dims.radiomic or None
    return cfg.with_overrides(fusion=fusion)


def resolve_output_dir(cfg: Config, cv_fold: Optional[int]) -> Path:
    """Per-run output directory: `<output_dir>/fold<K>` or `.../single_run`."""
    base = Path(cfg.training.output_dir)
    return base / f"fold{cv_fold}" if cv_fold is not None else base / "single_run"


# W&B

def init_run_wandb(
    cfg: Config,
    *,
    stage: str,
    cv_fold: Optional[int],
    shard: Optional[str],
    cache_mode: bool = False,
) -> bool:
    """Start the W&B run."""
    base_name = cfg.wandb.run_name or "run"
    slurm_job = os.getenv("SLURM_JOB_ID")
    suffix = f"-{slurm_job}" if slurm_job else ""
    if cache_mode:
        shard_id = os.getenv("SLURM_ARRAY_TASK_ID", shard or "0")
        run_name = f"{base_name}-cache-{shard_id}{suffix}"
    elif cv_fold is not None:
        run_name = f"{base_name}-fold{cv_fold}{suffix}"
    else:
        run_name = f"{base_name}{suffix}"
    return init_wandb(
        cfg,
        run_name=run_name,
        extra_config={
            "stage": stage,
            "shard": shard or None,
            "cv_fold": cv_fold,
            "slurm_job_id": slurm_job,
        },
    )


def push_wandb_data_extras(ctx: "RunContext") -> None:
    """Log the dataset-derived sizes onto the already-initialised W&B run."""
    update_wandb_config(
        {
            "cv_fold": ctx.cv_fold,
            "clinical_dim": ctx.dims.clinical,
            "anatomy_dim": ctx.dims.anatomy,
            "vessel_dim": ctx.dims.vessel,
            "n_train": len(ctx.train_ids),
            "n_val": len(ctx.val_ids),
            "n_test": len(ctx.test_ids),
        }
    )


# Registry / split / folds

def build_registry(cfg: Config) -> ClinicalRegistry:
    if cfg.data.labels_csv is None:
        raise SystemExit("cfg.data.labels_csv is required (set labels_csv: in config).")
    return ClinicalRegistry(
        cfg.data.labels_csv,
        include_cohorts=cfg.data.include_cohorts,
        completeness_weighting=getattr(cfg.data, "clinical_completeness_weighting", False),
        missingness_flags=getattr(cfg.data, "clinical_missingness_flags", False),
    )


def build_split(
    cfg: Config, registry: ClinicalRegistry
) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:
    """Enumerate patients with imaging + labels and make the train/val/test split."""
    dc = cfg.data

    from pdac_longitudinal.data.longitudinal_dataset import LongitudinalCTDataset

    discovery = LongitudinalCTDataset(
        nifti_root=dc.nifti_dir or dc.root_dir,
        registry=registry,
        phase=dc.phase,
        allowed_regions=list(dc.allowed_regions),
        post_nat_tps=list(dc.post_nat_tps),
    )
    all_ids = [c["patient_id"] for c in discovery.cases]
    if dc.exclude_cases:
        excluded = set(dc.exclude_cases)
        all_ids = [p for p in all_ids if p not in excluded]
        logger.info("Excluded %d cases: %s", len(excluded), sorted(excluded))
    if dc.include_cohorts:
        wanted = {c.lower() for c in dc.include_cohorts}
        before = len(all_ids)
        all_ids = [p for p in all_ids if registry.get_cohort(p).lower() in wanted]
        logger.info(
            "Cohort filter %s: kept %d / %d patients.",
            sorted(dc.include_cohorts), len(all_ids), before,
        )
    if cfg.training.max_cases:
        all_ids = all_ids[: cfg.training.max_cases]
        logger.info("Capped to %d cases (max_cases)", cfg.training.max_cases)

    if not all_ids:
        raise SystemExit(
            "No patients found.  Check that patient IDs match between "
            "nifti_dir and labels_csv, and that the composer layout is correct."
        )
    logger.info(
        "Patients with imaging + labels: %d / %d in registry",
        len(all_ids), len(registry.all_ids()),
    )


    phases_by_pid = {pid: (discovery.get_phase(pid) or "unknown") for pid in all_ids}

    train_ids, val_ids, test_ids = make_split(
        all_ids,
        registry=registry,
        seed=cfg.training.seed,
        split_file=dc.splits_file,
        phases=phases_by_pid,
    )
    return train_ids, val_ids, test_ids, phases_by_pid


def build_folds(
    cfg: Config,
    registry: ClinicalRegistry,
    train_ids: List[str],
    val_ids: List[str],
    phases_by_pid: Dict[str, str],
) -> List[Tuple[List[str], List[str]]]:
    """Build and persist the stratified k-fold partition of the train+val pool."""
    dc, cv = cfg.data, cfg.cv
    pool_ids = [p for p in train_ids + val_ids if registry.has(p)]
    pool_events = [registry.get_survival(p)[1] for p in pool_ids]
    pool_cohorts = [registry.get_cohort(p) for p in pool_ids]
    pool_phases = [phases_by_pid.get(p, "unknown") for p in pool_ids]

    if cv.folds_file:
        folds_file = cv.folds_file
    elif dc.splits_file:
        tag = "-".join(sorted(dc.include_cohorts)) if dc.include_cohorts else "all"
        folds_file = str(
            Path(dc.splits_file).expanduser().parent
            / f"cv_folds_{tag}_k{cv.n_folds}_seed{cv.seed}.json"
        )
    else:
        folds_file = None
    return load_or_make_kfolds(
        pool_ids, pool_events, n_folds=cv.n_folds, seed=cv.seed,
        cohorts=pool_cohorts, phases=pool_phases, folds_file=folds_file,
    )


def validate_cv_fold(cfg: Config, cv_fold: int) -> None:
    if not (0 <= cv_fold < cfg.cv.n_folds):
        raise SystemExit(f"--cv-fold must be in [0, {cfg.cv.n_folds}), got {cv_fold}")


# Orchestration

def prepare_run(
    config_path: Path,
    *,
    stage: str,
    cv_fold: Optional[int] = None,
    output_dir: Optional[Path] = None,
    shard: Optional[str] = None,
    cache_mode: bool = False,
    require_fold: bool = True,
    skip_wandb: bool = False,
) -> RunContext:
    """Run the shared setup and return a `RunContext`.

    Args:
        require_fold: Build/select a CV fold; False in evaluate.py, which
            refits the registry per fold itself.
        skip_wandb: Bypass `init_run_wandb` for commands that don't need
            a tracked run.
    """
    cfg = Config.from_yaml(config_path)
    logger.info("Loaded config from %s", config_path)
    cfg = apply_cli_overrides(cfg, cv_fold=cv_fold, output_dir=output_dir)

    seed_everything(cfg.training.seed)
    device = resolve_device(cfg)
    logger.info("Device: %s", device)

    use_wandb = False if skip_wandb else init_run_wandb(
        cfg, stage=stage, cv_fold=cv_fold, shard=shard, cache_mode=cache_mode,
    )

    registry = build_registry(cfg)
    dims = resolve_module_dims(cfg, registry)
    logger.info(
        "Modules  imaging=%s  clinical=%s (dim=%d)  anatomy=%s (dim=%d)"
        "  vessel=%s (dim=%d)  radiomics=%s (dim=%d)",
        cfg.modules.imaging,
        cfg.modules.clinical, dims.clinical,
        cfg.modules.anatomy, dims.anatomy,
        cfg.modules.vessel, dims.vessel,
        cfg.modules.radiomics, dims.radiomic,
    )
    cfg = patch_fusion_dims(cfg, dims)

    train_ids, val_ids, test_ids, phases_by_pid = build_split(cfg, registry)

    folds: Optional[List[Tuple[List[str], List[str]]]] = None
    resolved_fold: Optional[int] = cfg.cv.fold if cfg.cv.enabled else None

    if not require_fold:

        folds = build_folds(cfg, registry, train_ids, val_ids, phases_by_pid)
    else:
        if resolved_fold is not None:
            validate_cv_fold(cfg, resolved_fold)
            folds = build_folds(cfg, registry, train_ids, val_ids, phases_by_pid)
            train_ids, val_ids = folds[resolved_fold]
            logger.info(
                "CV mode: fold %d/%d  train=%d  val=%d  (test=%d held out)",
                resolved_fold + 1, cfg.cv.n_folds,
                len(train_ids), len(val_ids), len(test_ids),
            )
        # Fold-internal clinical preprocessing
        if cfg.modules.clinical:
            registry.fit(train_ids)

    return RunContext(
        cfg=cfg, device=device, registry=registry, dims=dims,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        folds=folds, cv_fold=resolved_fold, phases_by_pid=phases_by_pid,
        use_wandb=use_wandb,
    )
