"""Segmentation-mask-guided attention regularizer."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionGuidanceRegularizer(nn.Module):
    """Area-normalised in/out contrastive regularizer: `mean(β − α)`.

    Args:
        eps: Small constant to avoid division by zero when an inside/outside
            region has zero area.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        heatmap: torch.Tensor,
        roi: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean attention outside minus inside the ROI.

        Samples with an empty ROI are excluded from the batch mean.

        Args:
            heatmap: Attention heatmap in `[0, 1]`, shape `(B, D, H, W)`.
            roi: Binary ROI mask at heatmap resolution, shape `(B, D, H, W)`.

        Returns:
            Scalar loss; `mean(attn_out - attn_in)` over samples with a
            non-empty ROI, or a graph-connected zero if none have one.
        """
        roi = roi.to(heatmap.dtype)
        inside = (roi > 0.5).to(heatmap.dtype)
        outside = 1.0 - inside

        dims = (1, 2, 3)
        attn_in = (heatmap * inside).sum(dims) / (inside.sum(dims) + self.eps)
        attn_out = (heatmap * outside).sum(dims) / (outside.sum(dims) + self.eps)

        has_roi = (inside.sum(dims) > 0).to(heatmap.dtype)
        if has_roi.sum() < 1:
            return heatmap.sum() * 0.0    # zero with grad -> stable for AMP / autograd graph
        per_sample = (attn_out - attn_in) * has_roi
        return per_sample.sum() / has_roi.sum()
