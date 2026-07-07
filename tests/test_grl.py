"""Unit tests for the gradient-reversal layer and phase adversary."""

from __future__ import annotations

import torch

from pdac_longitudinal.models.grl import PhaseAdversary, grl


def test_grl_forward_is_identity():
    x = torch.randn(4, 8)
    assert torch.equal(grl(x, lambd=1.0), x)


def test_grl_backward_negates_and_scales_gradient():
    x = torch.randn(5, requires_grad=True)
    y = grl(x, lambd=2.5)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -2.5))


def test_phase_adversary_forward_shape():
    adv = PhaseAdversary(in_dim=16, n_phases=3)
    embedding = torch.randn(4, 16)
    logits = adv(embedding, lambd=1.0)
    assert logits.shape == (4, 3)


def test_phase_adversary_gradient_reverses_into_shared_embedding():
    adv = PhaseAdversary(in_dim=8, n_phases=2)
    embedding = torch.randn(4, 8, requires_grad=True)
    logits = adv(embedding, lambd=1.0)
    loss = logits.sum()
    loss.backward()

    assert embedding.grad is not None
    assert torch.isfinite(embedding.grad).all()
