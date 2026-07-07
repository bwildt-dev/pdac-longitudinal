"""Gradient-reversal layer + phase adversary for domain-adversarial training."""

from __future__ import annotations

import torch
import torch.nn as nn


class _GRL(torch.autograd.Function):
    """Identity in forward, multiply-by-(-λ) in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grl(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Apply the gradient-reversal layer with strength `lambd`.

    Args:
        x: Input tensor, passed through unchanged in the forward pass.
        lambd: Gradient-reversal strength.
    """
    return _GRL.apply(x, lambd)


class PhaseAdversary(nn.Module):
    """Tiny MLP that predicts CT phase from the fusion embedding.

    Args:
        in_dim: Dimension of the input embedding.
        n_phases: Number of CT phase classes to predict.
        hidden: Hidden layer width.
    """

    def __init__(self, in_dim: int, n_phases: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_phases),
        )

    def forward(self, embedding: torch.Tensor, lambd: float) -> torch.Tensor:
        """Apply GRL then predict CT phase logits.

        Args:
            embedding: Fusion embedding, shape `(B, in_dim)`.
            lambd: Gradient-reversal strength passed to `grl`.

        Returns:
            Phase logits, shape `(B, n_phases)`.
        """
        return self.net(grl(embedding, lambd))
