"""Checkpoint save/load utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointPaths:
    """Where each checkpoint kind is written."""
    output_dir: Path
    best_metric: Path
    latest: Path

    @classmethod
    def make(
        cls, output_dir: Path, prefix: str = "checkpoint", metric: str = "cindex",
    ) -> "CheckpointPaths":
        """Build the standard checkpoint paths under `<output_dir>/checkpoints/`."""
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            output_dir   = output_dir,
            best_metric  = ckpt_dir / f"{prefix}_best_{metric}.pth",
            latest       = ckpt_dir / f"{prefix}_latest.pth",
        )


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a checkpoint atomically (rename-after-write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    payload: Dict[str, Any] = {
        "epoch":        int(epoch),
        "model_state":  model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler_state"] = scaler.state_dict()
    if extra:
        payload["extra"] = dict(extra)

    torch.save(payload, tmp)
    tmp.replace(path)
    logger.info("Saved checkpoint → %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Optional[Any] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint in place; return the `extra` dict + epoch.

    optimizer/scheduler/scaler are restored only if both their saved state
    and a matching `load_state_dict` are present; missing either skips
    that part silently
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=strict)
    if missing or unexpected:
        logger.warning(
            "load_checkpoint: missing=%s unexpected=%s",
            sorted(missing), sorted(unexpected),
        )
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and "scaler_state" in payload and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(payload["scaler_state"])
    return {"epoch": int(payload.get("epoch", 0)), **payload.get("extra", {})}
