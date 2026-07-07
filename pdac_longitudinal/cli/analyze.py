"""`pdac_longitudinal analyze`; feature importance for a trained checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from pdac_longitudinal.analysis.feature_importance import run_feature_importance
from pdac_longitudinal.models.longitudinal_model import build_model_from_config
from pdac_longitudinal.training.checkpointing import load_checkpoint
from pdac_longitudinal.training.cv import build_cv_loaders, build_shared_segmenter
from pdac_longitudinal.training.setup import (
    configure_logging,
    prepare_run,
    push_wandb_data_extras,
    resolve_output_dir,
)
from pdac_longitudinal.training.wandb_setup import finish_wandb

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for `pdac_longitudinal analyze`."""
    p = argparse.ArgumentParser(
        prog="pdac_longitudinal analyze",
        description="Permutation / SHAP feature importance for a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Trained checkpoint .pth to analyse (use the run's config + --cv-fold).")
    p.add_argument("--cv-fold", type=int, default=None, metavar="N", dest="cv_fold",
                   help="CV fold N (0-indexed) the checkpoint was trained on.")
    p.add_argument("--output-dir", type=Path, default=None, dest="output_dir",
                   help="Override cfg.training.output_dir.")
    p.add_argument("--attention-cases", nargs="+", default=None, dest="attention_cases",
                   help="Render cross-timepoint attention overlays + per-stage montages for "
                        "these case IDs, then exit.")
    p.add_argument("--attention-on-test", action="store_true", dest="attention_on_test",
                   help="Render attention on the held-out test split. The test "
                        "split is shared across folds, so any fold's checkpoint can render it.")
    return p


def main(argv: Optional[list] = None) -> None:
    """Entry point for `pdac_longitudinal analyze`.

    Args:
        argv: Command-line args; defaults to `sys.argv[1:]`.

    Raises:
        SystemExit: If the requested attention split is empty, or if
            `analysis.on_test` is set but no test split is available.
    """
    configure_logging()
    args = build_parser().parse_args(argv)

    ctx = prepare_run(
        args.config, stage="analyze",
        cv_fold=args.cv_fold, output_dir=args.output_dir,
    )
    cfg, device, registry, dims = ctx.cfg, ctx.device, ctx.registry, ctx.dims
    ana = cfg.analysis

    output_dir = resolve_output_dir(cfg, ctx.cv_fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    push_wandb_data_extras(ctx)

    segmenter = build_shared_segmenter(cfg)
    _, val_loader, test_loader = build_cv_loaders(
        cfg=cfg, registry=registry,
        train_ids=ctx.train_ids, val_ids=ctx.val_ids, test_ids=ctx.test_ids,
        segmenter=segmenter, max_seg_tiles=cfg.training.max_seg_tiles,
        use_wandb=ctx.use_wandb,
    )

    model = build_model_from_config(cfg).to(device)
    logger.info("Loading %s", args.checkpoint)
    load_checkpoint(str(args.checkpoint), model=model, map_location=device, strict=True)

    # Attention-rendering mode: overlays + per-stage montages for chosen cases, then exit.
    if args.attention_cases:
        from pdac_longitudinal.visualisation.attention_viz import render_attention_maps

        attn_loader = test_loader if args.attention_on_test else val_loader
        if attn_loader is None or len(attn_loader.dataset) == 0:
            raise SystemExit("No loader available for attention rendering "
                             f"({'test' if args.attention_on_test else 'val'} split empty).")
        viz_dir = output_dir / "attention_viz"
        paths = render_attention_maps(
            model, attn_loader, device,
            clinical_dim=dims.clinical, out_dir=viz_dir,
            n_cases=len(args.attention_cases),
            anatomy_dim=dims.anatomy, vessel_dim=dims.vessel,
            skip_attn_stages=tuple(cfg.training.attention_viz_skip_stages),
            case_ids=args.attention_cases,
        )
        logger.info("Attention overlays + per-stage montages (%d cases) → %s", len(paths), viz_dir)
        if not paths:
            logger.warning("No cases matched %s in the %s split.",
                           args.attention_cases, "test" if args.attention_on_test else "val")
        finish_wandb()
        return

    # Optionally analyse the held-out test set, using this fold's already-fitted radiomic scaler.
    analysis_loader = test_loader if ana.on_test else val_loader
    if ana.on_test and (test_loader is None or len(test_loader.dataset) == 0):
        raise SystemExit("analysis.on_test: no test split available for this run.")
    if ana.on_test:
        logger.info("Feature importance on the HELD-OUT TEST set (n=%d), fold-%d scaler",
                    len(analysis_loader.dataset), ctx.cv_fold)

    # Radiomic PC->feature load-back: fold's fitted PCA loadings + canonical names.
    radiomic_loadings = radiomic_cols = None
    if dims.radiomic > 0:
        from pdac_longitudinal.radiomics.feature_schema import RADIOMIC_FEATURE_COLS

        scaler = getattr(analysis_loader.dataset, "_radiomic_scaler", None)
        if scaler is not None and scaler.pca_comp is not None:
            radiomic_loadings = scaler.pca_comp  # (k, n_features)
            radiomic_cols = list(RADIOMIC_FEATURE_COLS)
        elif dims.radiomic == len(RADIOMIC_FEATURE_COLS):
            radiomic_cols = list(RADIOMIC_FEATURE_COLS)

    run_feature_importance(
        model, analysis_loader, device,
        clinical_dim=dims.clinical, anatomy_dim=dims.anatomy, vessel_dim=dims.vessel,
        registry=registry, output_dir=output_dir, fold=ctx.cv_fold, do_shap=ana.shap,
        radiomic_dim=dims.radiomic,
        radiomic_loadings=radiomic_loadings, radiomic_cols=radiomic_cols,
        perm_through_pca=ana.perm_through_pca,
    )
    finish_wandb()


if __name__ == "__main__":
    main()
