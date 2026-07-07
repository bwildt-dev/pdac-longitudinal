"""Binary survival-past-horizon loss."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdac_longitudinal.losses._utils import as_1d_tensor


class BinaryHorizonLoss(nn.Module):
    """Binary cross-entropy for survival past a fixed horizon.

    Args:
        horizon_months: Horizon in months; the binary label is "died before
            this horizon".
        pos_weight: Positive-class weight for the BCE loss; unweighted when `None`.
    """

    def __init__(self, horizon_months: float = 12.0,
                 pos_weight: Optional[float] = None) -> None:
        super().__init__()
        self.horizon = float(horizon_months)
        self._pos_weight: Optional[torch.Tensor] = (
            None if pos_weight is None
            else torch.tensor(float(pos_weight), dtype=torch.float32)
        )

    def set_pos_weight(self, value: Optional[float]) -> None:
        """Set/clear the positive-class weight.

        Args:
            value: New positive-class weight, or `None` to clear it.
        """
        self._pos_weight = (
            None if value is None
            else torch.tensor(float(value), dtype=torch.float32)
        )

    @staticmethod
    def labels_and_mask(
        durations: torch.Tensor, events: torch.Tensor, horizon: float,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Return `(label, valid)`; label = died-within-horizon, valid = usable.

        Args:
            durations: Observed time-to-event or censoring, shape `(B,)`.
            events: Event indicator (1 = event, 0 = censored), shape `(B,)`.
            horizon: Horizon in the same units as `durations`.

        Returns:
            Tuple of the binary label (died within `horizon`) and a valid
            mask (censored-before-horizon cases are excluded as unusable).
        """
        d = as_1d_tensor(durations).float()
        e = as_1d_tensor(events).float()
        died = ((e > 0.5) & (d < horizon)).float()
        valid = (d >= horizon) | (e > 0.5)
        return died, valid

    def forward(
        self,
        risk_scores: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted BCE between risk scores and binary horizon labels.

        Args:
            risk_scores: Predicted risk logits, shape `(B,)` or `(B, 1)`.
            durations: Observed time-to-event or censoring, same shape.
            events: Event indicator (1 = event, 0 = censored), same shape.

        Returns:
            Scalar loss, `0` (graph-connected) if no sample is valid (see
            `labels_and_mask`).
        """
        risk = as_1d_tensor(risk_scores).float()
        label, valid = self.labels_and_mask(durations, events, self.horizon)
        if int(valid.sum().item()) == 0:
            return risk.sum() * 0.0  # graph-connected zero, no signal
        pw = self._pos_weight.to(risk.device) if self._pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            risk[valid], label[valid], pos_weight=pw,
        )
