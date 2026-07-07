"""`pdac_longitudinal verify`; validate a config and report split sizes without training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from pdac_longitudinal.training.setup import configure_logging, prepare_run

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for `pdac_longitudinal verify`."""
    p = argparse.ArgumentParser(
        prog="pdac_longitudinal verify",
        description="Load a config, build the patient split (and fold, if given), and exit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    p.add_argument("--cv-fold", type=int, default=None, metavar="N", dest="cv_fold",
                   help="Also validate fold N (default: validate the base split only).")
    return p


def main(argv: Optional[list] = None) -> None:
    """Entry point for `pdac_longitudinal verify`.

    Args:
        argv: Command-line args; defaults to `sys.argv[1:]`.
    """
    configure_logging()
    args = build_parser().parse_args(argv)

    ctx = prepare_run(args.config, stage="verify", cv_fold=args.cv_fold, skip_wandb=True)

    logger.info("Config OK: %s", args.config)
    logger.info("Patients  train=%d  val=%d  test=%d",
                len(ctx.train_ids), len(ctx.val_ids), len(ctx.test_ids))
    logger.info(
        "Modules   imaging=%s  clinical=%s (dim=%d)  anatomy=%s (dim=%d)"
        "  vessel=%s (dim=%d)  radiomics=%s (dim=%d)",
        ctx.cfg.modules.imaging,
        ctx.cfg.modules.clinical, ctx.dims.clinical,
        ctx.cfg.modules.anatomy, ctx.dims.anatomy,
        ctx.cfg.modules.vessel, ctx.dims.vessel,
        ctx.cfg.modules.radiomics, ctx.dims.radiomic,
    )
    if ctx.folds is not None:
        logger.info("CV folds  n_folds=%d  fold=%s", ctx.cfg.cv.n_folds, ctx.cv_fold)


if __name__ == "__main__":
    main()
