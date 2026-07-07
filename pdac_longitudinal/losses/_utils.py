"""Shared tensor helpers for the loss modules."""

from __future__ import annotations

import torch


def as_1d_tensor(x: torch.Tensor) -> torch.Tensor:
    """Flatten `(B, 1)` style tensors to `(B,)` while keeping batch semantics."""
    if x.dim() == 0:
        return x.unsqueeze(0)
    if x.dim() == 2 and x.shape[1] == 1:
        return x.squeeze(1)
    if x.dim() != 1:
        raise ValueError(f"Expected 1-D tensor or shape (B, 1), got {tuple(x.shape)}.")
    return x
