"""W&B run initialisation and logging helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import wandb as _wandb
except ImportError:
    _wandb = None  # type: ignore[assignment]


def _flatten(d: Dict, parent: str = "") -> Dict[str, str]:
    """Flatten a nested dict to dot-separated keys, truncating long values."""
    flat: Dict[str, str] = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = str(v)[:200]
    return flat


def init_wandb(
    cfg: "Config",  # type: ignore[name-defined]
    run_name: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Initialise a W&B run from the typed config."""
    if _wandb is None:
        logger.warning("wandb not installed; experiment tracking disabled.")
        return False

    wc = cfg.wandb
    if not wc.enabled:
        return False

    if wc.mode == "online" and not os.getenv("WANDB_API_KEY"):
        logger.warning("W&B mode=online but WANDB_API_KEY not set; disabling.")
        return False

    try:
        flat_cfg = _flatten(cfg._raw)
        if extra_config:
            flat_cfg.update({k: str(v)[:200] for k, v in extra_config.items()})


        array_job = os.getenv("SLURM_ARRAY_JOB_ID")
        plain_job = os.getenv("SLURM_JOB_ID")
        wandb_group = (
            os.getenv("WANDB_RUN_GROUP")
            or (f"array-{array_job}" if array_job
                else (f"job-{plain_job}" if plain_job else None))
        )
        _wandb.init(
            project=os.getenv("WANDB_PROJECT") or wc.project,
            entity=os.getenv("WANDB_ENTITY") or (wc.entity or None),
            name=run_name or wc.run_name or None,
            mode=wc.mode,
            dir=str(wc.dir) if wc.dir else None,
            tags=list(wc.tags.values()) if wc.tags else None,
            group=wandb_group,
            config=flat_cfg,
        )
        logger.info("W&B run initialised: %s/%s", wc.project, run_name or wc.run_name)
        return True

    except Exception as exc:
        logger.warning("W&B init failed (%s); disabling.", exc)
        return False


def finish_wandb() -> None:
    if _wandb is not None and _wandb.run is not None:
        _wandb.finish()


def update_wandb_config(extras: Dict[str, Any]) -> None:
    if _wandb is None or _wandb.run is None:
        return
    try:
        _wandb.config.update(
            {k: (str(v)[:200] if v is not None else None) for k, v in extras.items()},
            allow_val_change=True,
        )
    except Exception as exc:
        logger.warning("update_wandb_config failed: %s", exc)


def log_seg_overlay(
    pid: str,
    arrays: Dict[str, Any],
) -> None:
    """Log T0 + T1 CT slices with IT tumour mask overlay to the active W&B run."""
    if _wandb is None or _wandb.run is None:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap

        t0  = arrays["t0"]
        t1  = arrays["t1"]
        it0 = arrays["mask_it"]
        it1 = arrays.get("mask_it_t1", np.zeros_like(it0))

        z_scores = it0.sum(axis=(0, 1))
        z = int(z_scores.argmax()) if z_scores.max() > 0 else it0.shape[2] // 2

        def _slice(vol: Any) -> Any:
            s = vol[:, :, z].T.astype(float)
            s[s == 0] = float("nan")
            return s

        def _display_window(vol: np.ndarray) -> tuple:
            fg = vol[vol > vol.min() + 1e-6]
            if fg.size == 0:
                return float(vol.min()), float(vol.max())
            return float(np.percentile(fg, 1.0)), float(np.percentile(fg, 99.0))

        vmin0, vmax0 = _display_window(t0)
        vmin1, vmax1 = _display_window(t1)

        red_cmap = ListedColormap(["red"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"{pid}  —  axial z={z}  "
            f"IT_t0={int(it0.sum())} vox  IT_t1={int(it1.sum())} vox"
        )

        axes[0].imshow(t0[:, :, z].T, cmap="gray", origin="lower", vmin=vmin0, vmax=vmax0)
        axes[0].imshow(_slice(it0), cmap=red_cmap, alpha=0.45, origin="lower")
        axes[0].set_title("T0 — pre-NAT")
        axes[0].axis("off")

        axes[1].imshow(t1[:, :, z].T, cmap="gray", origin="lower", vmin=vmin1, vmax=vmax1)
        axes[1].imshow(_slice(it0), cmap=red_cmap, alpha=0.45, origin="lower")
        axes[1].set_title("T1 — post-NAT (T0 ROI)")
        axes[1].axis("off")

        axes[2].imshow(t1[:, :, z].T, cmap="gray", origin="lower", vmin=vmin1, vmax=vmax1)
        axes[2].imshow(_slice(it1), cmap=red_cmap, alpha=0.45, origin="lower")
        axes[2].set_title("T1 — post-NAT (T1 ROI)")
        axes[2].axis("off")

        fig.tight_layout()
        _wandb.log({f"segmentation/{pid}": _wandb.Image(fig)})
        plt.close(fig)

    except Exception as exc:
        logger.warning("log_seg_overlay failed for %s: %s", pid, exc)


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
) -> None:
    """Log a flat metrics dict to the active W&B run. No-op if not active."""
    if _wandb is not None and _wandb.run is not None:
        _wandb.log(metrics, step=step)


def log_attention_images(paths: List[Path], step: Optional[int] = None) -> None:
    """Log a list of attention overlay image paths to W&B. No-op if not active."""
    if _wandb is not None and _wandb.run is not None and paths:
        _wandb.log(
            {"val/attention": [_wandb.Image(str(p)) for p in paths]},
            step=step,
        )


def log_cache_roi(pid: str, path: Path) -> None:
    """Log a per-patient cache-time ROI overlay PNG to W&B."""
    if _wandb is None or _wandb.run is None:
        return
    try:
        _wandb.log({f"cache_roi/{pid}": _wandb.Image(str(path), caption=pid)})
    except Exception as exc:
        logger.warning("W&B log_cache_roi failed for %s (%s); continuing.", pid, exc)
