"""Survival metrics used during training/validation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def concordance_index(risk: np.ndarray, dur: np.ndarray, evt: np.ndarray) -> float:
    """Compute Harrell's C-index."""
    from lifelines.utils import concordance_index as ci

    valid = np.isfinite(risk)
    if not valid.all():
        n_bad = int((~valid).sum())
        logger.warning("concordance_index: dropping %d NaN/Inf risk scores", n_bad)
        risk, dur, evt = risk[valid], dur[valid], evt[valid]

    if len(risk) < 2 or evt.sum() == 0:
        logger.warning(
            "concordance_index: not computable — n=%d events=%d",
            len(risk), int(evt.sum()),
        )
        return float("nan")

    return float(ci(dur, -risk, evt))


def horizon_auc(
    risk: np.ndarray, dur: np.ndarray, evt: np.ndarray, horizon: float,
) -> float:
    """ROC-AUC for survival past *horizon* (binary task)."""
    from sklearn.metrics import roc_auc_score

    risk = np.asarray(risk, dtype=float)
    dur = np.asarray(dur, dtype=float)
    evt = np.asarray(evt, dtype=float)

    died = (evt > 0.5) & (dur < horizon)
    valid = ((dur >= horizon) | (evt > 0.5)) & np.isfinite(risk)
    y = died[valid].astype(int)
    s = risk[valid]

    if len(y) < 2 or len(np.unique(y)) < 2:
        logger.warning(
            "horizon_auc: not computable — usable=%d positives=%d",
            len(y), int(y.sum()),
        )
        return float("nan")
    return float(roc_auc_score(y, s))


def epoch_metric(
    task: str, risk: np.ndarray, dur: np.ndarray, evt: np.ndarray,
    horizon: float = 12.0,
) -> float:
    """Dispatch to the task's primary metric: C-index (survival) or AUC (binary)."""
    if task == "binary":
        return horizon_auc(risk, dur, evt, horizon)
    return concordance_index(risk, dur, evt)
