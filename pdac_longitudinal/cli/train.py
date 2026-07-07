"""`pdac_longitudinal train`; train one CV fold or the full split."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch

from pdac_longitudinal.config import Config
from pdac_longitudinal.losses import BinaryHorizonLoss, CoxPHLoss
from pdac_longitudinal.models.longitudinal_model import build_model_from_config
from pdac_longitudinal.radiomics.feature_schema import RADIOMIC_FEATURE_DIM
from pdac_longitudinal.training.checkpointing import load_checkpoint
from pdac_longitudinal.training.cv import build_cv_loaders, build_shared_segmenter
from pdac_longitudinal.training.loop import fit, train_step
from pdac_longitudinal.training.setup import (
    ModuleDims,
    configure_logging,
    patch_fusion_dims,
    prepare_run,
    push_wandb_data_extras,
    resolve_device,
    resolve_output_dir,
    seed_everything,
)
from pdac_longitudinal.training.wandb_setup import finish_wandb, log_metrics

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for `pdac_longitudinal train`."""
    p = argparse.ArgumentParser(
        prog="pdac_longitudinal train",
        description="Train the PDAC longitudinal model from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    p.add_argument("--cv-fold", type=int, default=None, metavar="N", dest="cv_fold",
                   help="Run CV fold N (0-indexed).  Overrides cfg.cv.fold.")
    p.add_argument("--output-dir", type=Path, default=None, dest="output_dir",
                   help="Override cfg.training.output_dir (per-run output location).")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume training from this checkpoint .pth file.")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Build model, run one fake batch forward+backward, exit.")
    return p


# Loss / optimiser

def build_criterion(cfg: Config):
    """Pick the loss by task: Cox PH (survival, default) or BCE (binary horizon).

    Args:
        cfg: Run config.
    """
    task = getattr(cfg.training, "task", "survival")
    if task == "binary":
        horizon = getattr(cfg.training, "survival_horizon_months", 12.0)
        logger.info("Task: binary survival @ %.0f months (BCE + AUC)", horizon)
        return BinaryHorizonLoss(horizon_months=horizon)
    logger.info("Task: survival (Cox PH + C-index)")
    return CoxPHLoss()


def build_lr_scheduler(optimizer, tc, resume_epoch: int = 0):
    """Build a cosine LR schedule with optional linear warmup.

    Args:
        optimizer: Optimizer to schedule.
        tc: Training config.
        resume_epoch: Epoch to resume from; skips warmup when nonzero.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup = int(getattr(tc, "warmup_epochs", 0) or 0)
    if resume_epoch == 0 and 0 < warmup < tc.num_epochs:
        warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        cos = CosineAnnealingLR(optimizer, T_max=tc.num_epochs - warmup)
        return SequentialLR(optimizer, [warm, cos], milestones=[warmup])
    return CosineAnnealingLR(optimizer, T_max=tc.num_epochs - resume_epoch)


def build_optimiser_and_scheduler(model: torch.nn.Module, cfg: Config, resume_epoch: int = 0) -> tuple:
    """Return `(optimizer, scheduler, scaler)`.

    Args:
        model: Model whose trainable parameters are optimized.
        cfg: Run config.
        resume_epoch: Epoch to resume the LR schedule from.
    """
    tc = cfg.training
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tc.learning_rate, weight_decay=tc.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, tc, resume_epoch)
    scaler: Any = None
    if tc.use_amp:
        device = resolve_device(cfg)
        if device.type == "cuda":
            # bf16 keeps a disabled (but non-None) scaler; scale/step stay pass-through.
            use_fp16 = getattr(tc, "amp_dtype", "float16") == "float16"
            scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    return optimizer, scheduler, scaler


def _dry_run(cfg: Config, dims: ModuleDims) -> None:
    """Build model, run one fake forward+backward, log timing, exit.

    Args:
        cfg: Run config.
        dims: Per-branch feature dims used to build the fake batch.
    """
    device = resolve_device(cfg)
    tc, pc = cfg.training, cfg.preprocessing
    patch = tuple(min(p, 64) for p in pc.patch_size)
    B = min(tc.batch_size, 2)
    logger.info("=== DRY RUN ===  device=%s  patch=%s  B=%d", device, patch, B)

    model = build_model_from_config(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer, _, scaler = build_optimiser_and_scheduler(model, cfg)

    fake = {
        "t0": torch.randn(B, 1, *patch),
        "t1": torch.randn(B, 1, *patch),
        "valid_t0": torch.ones(B, 1, *patch, dtype=torch.bool),
        "valid_t1": torch.ones(B, 1, *patch, dtype=torch.bool),
        "duration": torch.tensor([12.0, 8.0])[:B],
        "event": torch.tensor([1.0, 0.0])[:B],
        "clinical": torch.zeros(B, dims.clinical),
        "anatomy": torch.zeros(B, dims.anatomy) if dims.anatomy > 0 else torch.zeros(B, 0),
        "vessel": torch.zeros(B, dims.vessel) if dims.vessel > 0 else torch.zeros(B, 0),
        "radiomic": torch.zeros(B, dims.radiomic) if dims.radiomic > 0 else torch.zeros(B, 0),
    }
    t0 = time.time()
    metrics, _, _, _ = train_step(
        model, fake, criterion, optimizer, device, scaler,
        dims.clinical, dims.anatomy, dims.vessel, dims.radiomic,
    )
    logger.info("Dry-run done in %.1fs  loss=%.4f  device=%s", time.time() - t0, metrics["total"], device)
    logger.info("Model feature dims: %s", model.feature_dims)


# Clinical-only baseline (inline, same fold + cohort as the imaging model)

def _resolve_clinical_baseline(cfg: Config, train_ids, val_ids) -> tuple:
    """Return `(baseline_dict_or_None, val_predictions_or_None)`.

    Args:
        cfg: Run config.
        train_ids: Patient IDs in the train fold.
        val_ids: Patient IDs in the val fold.
    """
    results_path = cfg.training.clinical_baseline_results
    if results_path is not None:
        try:
            baseline = json.loads(Path(results_path).read_text())
            logger.info("=" * 70)
            logger.info("CLINICAL-ONLY BASELINE  (loaded from %s — OVERRIDE)", results_path)
            logger.info("-" * 70)
            logger.info("  Val  C-index   : %.3f", baseline.get("best_val_c", float("nan")))
            logger.info("  Train C-index  : %.3f", baseline.get("best_train_c", float("nan")))
            logger.info("  Test C-index   : %.3f", baseline.get("best_test_c", float("nan")))
            logger.info("  Val  AUC@horiz : %.3f", baseline.get("best_val_auc", float("nan")))
            logger.info("  Test AUC@horiz : %.3f", baseline.get("best_test_auc", float("nan")))
            logger.info("=" * 70)
            return baseline, None
        except Exception as exc:
            logger.warning("Could not load clinical baseline from %s: %s", results_path, exc)
            return None, None

    if not cfg.training.clinical_baseline_enabled:
        return None, None

    from pdac_longitudinal.baselines import train_clinical_baseline
    try:
        baseline = train_clinical_baseline(
            labels_csv=cfg.data.labels_csv,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=None,
            horizon=getattr(cfg.training, "survival_horizon_months", 12.0),
            include_cohorts=cfg.data.include_cohorts,
            missingness_flags=getattr(cfg.data, "clinical_missingness_flags", False),
        )
    except Exception as exc:
        logger.warning("Inline clinical baseline failed (%s); continuing without.", exc)
        return None, None
    if not baseline:
        return None, None
    val_predictions = baseline.pop("val_predictions", None)
    logger.info("=" * 70)
    logger.info("CLINICAL-ONLY BASELINE  (inline lifelines CoxPH — same fold/cohort)")
    logger.info("-" * 70)
    logger.info("  Cohort         : %s", list(cfg.data.include_cohorts) or "all")
    logger.info("  Val  C-index   : %.3f", baseline.get("best_val_c", float("nan")))
    logger.info("  Train C-index  : %.3f", baseline.get("best_train_c", float("nan")))
    logger.info("  Val  AUC@horiz : %.3f", baseline.get("best_val_auc", float("nan")))
    logger.info("=" * 70)
    return baseline, val_predictions


# Entry point

def main(argv: Optional[list] = None) -> None:
    """Entry point for `pdac_longitudinal train`.

    Args:
        argv: Command-line args; defaults to `sys.argv[1:]`.
    """
    configure_logging()
    args = build_parser().parse_args(argv)

    # Dry run: no registry / data needed
    if args.dry_run:
        from pdac_longitudinal.training.setup import apply_cli_overrides

        cfg = apply_cli_overrides(
            Config.from_yaml(args.config),
            cv_fold=args.cv_fold, output_dir=args.output_dir,
        )
        seed_everything(cfg.training.seed)
        fc = cfg.fusion
        dims = ModuleDims(
            clinical=int(fc.get("clinical_feature_dim", 0) or 0) if cfg.modules.clinical else 0,
            anatomy=int(fc.get("anatomy_feature_dim", 0) or 0) if cfg.modules.anatomy else 0,
            vessel=int(fc.get("vessel_feature_dim", 0) or 0) if cfg.modules.vessel else 0,
            radiomic=RADIOMIC_FEATURE_DIM if cfg.modules.radiomics else 0,
        )
        cfg = patch_fusion_dims(cfg, dims)
        _dry_run(cfg, dims)
        return

    ctx = prepare_run(
        args.config, stage="train",
        cv_fold=args.cv_fold, output_dir=args.output_dir,
    )
    cfg, device, registry, dims = ctx.cfg, ctx.device, ctx.registry, ctx.dims

    clinical_baseline, clinical_val_predictions = _resolve_clinical_baseline(
        cfg, ctx.train_ids, ctx.val_ids,
    )

    output_dir = resolve_output_dir(cfg, ctx.cv_fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    if clinical_val_predictions:
        pred_path = output_dir / "clinical_val_predictions.csv"
        with open(pred_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["patient_id", "risk", "duration", "event"])
            w.writeheader()
            w.writerows(clinical_val_predictions)
        logger.info("Clinical baseline val predictions → %s", pred_path)

    push_wandb_data_extras(ctx)

    segmenter = build_shared_segmenter(cfg)
    train_loader, val_loader, test_loader = build_cv_loaders(
        cfg=cfg, registry=registry,
        train_ids=ctx.train_ids, val_ids=ctx.val_ids, test_ids=ctx.test_ids,
        segmenter=segmenter, max_seg_tiles=cfg.training.max_seg_tiles,
        use_wandb=ctx.use_wandb,
    )
    logger.info(
        "Dataset  train=%d  val=%d  test=%d (held out)",
        len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset),
    )

    model = build_model_from_config(cfg).to(device)
    logger.info("Built model; feature dims: %s", model.feature_dims)

    criterion = build_criterion(cfg)
    # pos_weight = neg/pos on this fold's train ids; patients
    # censored before the horizon are excluded.
    if isinstance(criterion, BinaryHorizonLoss):
        H = float(getattr(cfg.training, "survival_horizon_months", 12.0))
        pos = neg = 0
        for pid in ctx.train_ids:
            t, s = registry.get_survival(pid)
            if s == 1 and t < H:
                pos += 1
            elif t >= H:
                neg += 1
        if pos > 0:
            criterion.set_pos_weight(neg / pos)
            logger.info("Binary class-balanced BCE: pos_weight=%.2f (neg=%d / pos=%d, train fold)",
                        neg / pos, neg, pos)
        else:
            logger.warning("Binary task: 0 positives in train fold — pos_weight unset")

    resume_epoch, best_val_metric = 0, 0.0
    optimizer, scheduler, scaler = build_optimiser_and_scheduler(model, cfg)
    if args.resume is not None:
        logger.info("Resuming from %s", args.resume)
        state = load_checkpoint(args.resume, model=model, optimizer=optimizer,
                                scheduler=scheduler, scaler=scaler, map_location=device, strict=True)
        resume_epoch = state.get("epoch", 0)
        best_val_metric = state.get("val_metric", 0.0)
        logger.info("Resumed: epoch=%d  best_val_metric=%.3f", resume_epoch, best_val_metric)
        # Restart the LR schedule from the resumed epoch so the cosine curve continues.
        scheduler = build_lr_scheduler(optimizer, cfg.training, resume_epoch)

    result = fit(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        cfg=cfg, device=device, output_dir=output_dir,
        clinical_dim=dims.clinical, anatomy_dim=dims.anatomy,
        vessel_dim=dims.vessel, radiomic_dim=dims.radiomic,
        amp_dtype=(torch.bfloat16 if getattr(cfg.training, "amp_dtype", "float16") == "bfloat16"
                   else torch.float16),
        use_wandb=ctx.use_wandb, resume_epoch=resume_epoch,
        best_val_metric=best_val_metric, clinical_baseline=clinical_baseline,
    )

    logger.info("Run complete.  Best val score: %.3f", result["best_val_metric"])
    logger.info("Best checkpoint: %s", result["best_checkpoint"])
    logger.info("Test set (%d patients) in %s — run `pdac_longitudinal evaluate` for final metrics.",
                len(ctx.test_ids), cfg.data.splits_file)

    results = {
        "cv_fold": ctx.cv_fold,
        "best_val_metric": float(result["best_val_metric"]),
        "final_epoch": int(result["final_epoch"]),
        "best_checkpoint": str(result["best_checkpoint"]),
        "n_train": len(ctx.train_ids), "n_val": len(ctx.val_ids), "n_test": len(ctx.test_ids),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("Wrote fold summary → %s", output_dir / "results.json")

    if ctx.use_wandb:
        log_metrics({
            "summary/best_val_metric": result["best_val_metric"],
            "summary/final_epoch": float(result["final_epoch"]),
        })
    finish_wandb()


if __name__ == "__main__":
    main()
