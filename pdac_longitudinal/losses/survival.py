"""Cox proportional-hazards survival loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from pdac_longitudinal.losses._utils import as_1d_tensor


class CoxPHLoss(nn.Module):
    """Cox partial negative log-likelihood with Efron tie handling.

    Args:
        eps: Small constant to keep `log()` finite when a risk-set sum is
            near zero.
        reduction: `'mean'` (average over observed events) or `'sum'`.

    Raises:
        ValueError: If `reduction` is not one of `{'mean', 'sum'}`.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be one of {'mean', 'sum'}")
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        risk_scores: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar Cox PH loss for a batch of right-censored observations.

        The partial likelihood is computed within this batch; risk sets are
        formed from the batch's own durations.

        Args:
            risk_scores: Predicted risk scores, shape `(B,)` or `(B, 1)`.
            durations: Observed time-to-event or censoring, same shape.
            events: Event indicator (1 = event, 0 = censored), same shape.

        Returns:
            Scalar loss, `0` (graph-connected) if the batch has no observed
            events.

        Raises:
            ValueError: If the three inputs don't have matching lengths.
        """
        risk_scores = as_1d_tensor(risk_scores).float()
        durations = as_1d_tensor(durations).float()
        events = as_1d_tensor(events).float()

        if not (risk_scores.shape[0] == durations.shape[0] == events.shape[0]):
            raise ValueError("risk_scores, durations, and events must have the same length.")

        if risk_scores.numel() == 0:
            return torch.zeros((), dtype=risk_scores.dtype, device=risk_scores.device)

        # Sort descending by time so risk sets are prefix subsets.
        order = torch.argsort(durations, descending=True)
        t = durations[order]
        e = events[order]
        r = risk_scores[order]

        r = r - r.max()  # shift for numerical stability (Cox loss is shift-invariant)
        exp_r = torch.exp(r)

        event_mask = e > 0.5
        if int(event_mask.sum().item()) == 0:
            # No observed events return zero connected to the graph
            return r.sum() * 0.0

        unique_event_times = torch.unique(t[event_mask])
        loglik = r.new_zeros(())

        for ti in unique_event_times:
            deaths = (t == ti) & event_mask
            d_i = int(deaths.sum().item())
            if d_i == 0:
                continue

            risk_set = t >= ti
            sum_risk = exp_r[risk_set].sum()
            sum_deaths = exp_r[deaths].sum()
            loglik = loglik + r[deaths].sum()

            # Efron tie correction.
            for l in range(d_i):
                frac = float(l) / float(d_i)
                denom = sum_risk - frac * sum_deaths
                loglik = loglik - torch.log(denom.clamp_min(self.eps))

        neg_loglik = -loglik
        if self.reduction == "mean":
            return neg_loglik / event_mask.sum().clamp_min(1.0)
        return neg_loglik
