"""Unit tests for survival losses (CPU, deterministic)."""

from __future__ import annotations

import torch

from pdac_longitudinal.losses.binary import BinaryHorizonLoss
from pdac_longitudinal.losses.survival import CoxPHLoss


def test_cox_loss_rewards_correct_risk_ordering():
    dur = torch.tensor([1.0, 2.0, 3.0, 4.0])
    evt = torch.tensor([1.0, 1.0, 1.0, 1.0])
    good = torch.tensor([4.0, 3.0, 2.0, 1.0])   # higher risk = earlier death
    bad  = torch.tensor([1.0, 2.0, 3.0, 4.0])   # reversed
    lg = CoxPHLoss()(good, dur, evt)
    lb = CoxPHLoss()(bad, dur, evt)
    assert torch.isfinite(lg)
    assert lg < lb


def test_cox_loss_gradient_flows():
    risk = torch.zeros(4, requires_grad=True)
    dur = torch.tensor([1.0, 2.0, 3.0, 4.0])
    evt = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = CoxPHLoss()(risk, dur, evt)
    loss.backward()
    assert risk.grad is not None
    assert torch.isfinite(risk.grad).all()


def test_binary_horizon_loss_is_finite():
    logits = torch.tensor([0.5, -0.5, 2.0, -1.0])
    dur = torch.tensor([6.0, 24.0, 6.0, 24.0])
    evt = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = BinaryHorizonLoss(horizon_months=12.0)(logits, dur, evt)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
