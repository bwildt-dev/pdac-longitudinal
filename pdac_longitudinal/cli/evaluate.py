"""`pdac_longitudinal evaluate`; held-out-test ensemble eval of a CV run."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from pdac_longitudinal.training.cv import build_shared_segmenter
from pdac_longitudinal.training.evaluate import run_ensemble_eval
from pdac_longitudinal.training.setup import configure_logging, prepare_run
from pdac_longitudinal.training.wandb_setup import finish_wandb

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for `pdac_longitudinal evaluate`."""
    p = argparse.ArgumentParser(
        prog="pdac_longitudinal evaluate",
        description="Average a CV run's fold checkpoints on the held-out test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    p.add_argument("--run-dir", type=Path, required=True, dest="run_dir",
                   help="CV run dir containing fold*/checkpoints/checkpoint_best_*.pth.")
    return p


def main(argv: Optional[list] = None) -> None:
    """Entry point for `pdac_longitudinal evaluate`.

    Args:
        argv: Command-line args; defaults to `sys.argv[1:]`.
    """
    configure_logging()
    args = build_parser().parse_args(argv)

    ctx = prepare_run(args.config, stage="evaluate", require_fold=False)
    logger.info("Ensemble-eval mode — averaging %d folds from %s on %d test patients",
                len(ctx.folds), args.run_dir, len(ctx.test_ids))
    run_ensemble_eval(
        ctx.cfg, ctx.registry, ctx.folds, ctx.test_ids, args.run_dir, ctx.device,
        build_shared_segmenter(ctx.cfg),
        ctx.dims.clinical, ctx.dims.anatomy, ctx.dims.vessel, ctx.dims.radiomic,
        tta=ctx.cfg.analysis.tta,
    )
    finish_wandb()


if __name__ == "__main__":
    main()
