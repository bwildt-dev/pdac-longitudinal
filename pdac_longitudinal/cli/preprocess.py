"""`pdac_longitudinal preprocess`; build the CT cache without training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from pdac_longitudinal.training.cv import (
    build_cv_loaders,
    build_shared_segmenter,
    shard_filter_fn,
)
from pdac_longitudinal.training.setup import configure_logging, prepare_run
from pdac_longitudinal.training.wandb_setup import finish_wandb

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for `pdac_longitudinal preprocess`."""
    p = argparse.ArgumentParser(
        prog="pdac_longitudinal preprocess",
        description="Pre-build the preprocessed CT cache and exit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    p.add_argument("--cv-fold", type=int, default=None, metavar="N", dest="cv_fold",
                   help="Restrict to fold N's patients (default: all patients).")
    p.add_argument("--shard", type=str, default=None, metavar="K/N",
                   help="Cache only patients where md5(pid) %% N == K (SLURM array).")
    return p


def main(argv: Optional[list] = None) -> None:
    """Entry point for `pdac_longitudinal preprocess`.

    Args:
        argv: Command-line args; defaults to `sys.argv[1:]`.
    """
    configure_logging()
    args = build_parser().parse_args(argv)

    ctx = prepare_run(
        args.config, stage="cache",
        cv_fold=args.cv_fold, shard=args.shard, cache_mode=True,
    )
    segmenter = build_shared_segmenter(ctx.cfg)
    build_cv_loaders(
        cfg=ctx.cfg, registry=ctx.registry,
        train_ids=ctx.train_ids, val_ids=ctx.val_ids, test_ids=ctx.test_ids,
        segmenter=segmenter, max_seg_tiles=ctx.cfg.training.max_seg_tiles,
        shard_fn=shard_filter_fn(args.shard) if args.shard else None,
        cache_only=True, use_wandb=ctx.use_wandb,
    )
    logger.info("--cache-only: exiting after cache build.")

    if ctx.cfg.modules.radiomics:
        _run_radiomics_if_available(args.config, ctx.cfg)

    finish_wandb()


def _run_radiomics_if_available(config_path: Path, cfg) -> None:
    """Extract radiomics in-process if pyradiomics is installed, else point at
    the standalone extractor. Never blocks preprocessing."""
    try:
        import radiomics  # noqa: F401
    except ImportError:
        logger.warning(
            "modules.radiomics is enabled but pyradiomics isn't installed here — "
            "skipping in-process extraction. Install it with `uv sync --extra "
            "radiomics`, or run `python -m pdac_longitudinal.preprocess.radiomics_features "
            "--config %s` separately.",
            config_path,
        )
        return
    if cfg.data.cache_dir is None:
        logger.warning("modules.radiomics is enabled but data.cache_dir is unset — "
                       "skipping in-process radiomics extraction.")
        return
    from pdac_longitudinal.preprocess.radiomics_features import run_radiomics_extraction
    run_radiomics_extraction(
        config_path=config_path,
        cache_dir=Path(cfg.data.cache_dir),
        version=cfg.data.cache_version,
    )


if __name__ == "__main__":
    main()
