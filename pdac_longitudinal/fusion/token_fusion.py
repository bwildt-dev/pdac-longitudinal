"""Transformer fusion head for longitudinal PDAC modelling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_2d_tensor(x: Union[torch.Tensor, float, int]) -> torch.Tensor:
    """Convert a scalar or tensor to shape `(B, F)` when possible."""
    tensor = torch.as_tensor(x)
    if tensor.dim() == 0:
        tensor = tensor.view(1, 1)
    elif tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.float()


def radiomic_mapping_to_tensor(
    features: Mapping[str, Union[torch.Tensor, float, int]],
) -> torch.Tensor:
    """Stack a radiomic feature mapping into a `(B, F)` tensor.

    Args:
        features: Mapping of feature name to scalar or tensor value.

    Returns:
        Tensor of shape `(B, F)`, columns ordered by sorted feature name.

    Raises:
        ValueError: If `features` is empty, or the per-feature tensors don't
            all share the same shape.
    """
    if not features:
        raise ValueError("Radiomic feature mapping is empty.")
    stacked = [_ensure_2d_tensor(features[key]) for key in sorted(features)]
    shapes = {tuple(t.shape) for t in stacked}
    if len(shapes) != 1:
        raise ValueError(
            f"Radiomic feature tensors must share the same shape; got {sorted(shapes)}."
        )
    return torch.cat(stacked, dim=-1)


class _NormProjection(nn.Module):
    """Per-feature BatchNorm + linear projection for tabular tokens.

    Args:
        in_dim: Number of input features.
        embed_dim: Output token embedding dimension.
        normalize: Whether to apply BatchNorm before the linear projection.
    """

    def __init__(self, in_dim: int, embed_dim: int, normalize: bool = True) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim) if normalize else None
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise (optional) and project tabular features to the token space.

        Args:
            x: Tabular features, shape `(B, N, F)`.

        Returns:
            Projected tokens, shape `(B, N, embed_dim)`.
        """
        if self.norm is not None:
            #Treat inf/nan as missing (0) before BatchNorm.
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            b, n, f = x.shape
            x = self.norm(x.reshape(b * n, f)).reshape(b, n, f)
        return self.proj(x)


class TokenFusionHead(nn.Module):
    """Fuse deep, radiomic, and clinical tokens via a Transformer.

    Args:
        deep_feature_dims: Channel dims for each deep feature source.
        embed_dim: Shared token embedding dimension.
        num_layers: Number of Transformer encoder layers.
        num_heads: Attention heads; auto-chosen from `embed_dim` when `None`.
        num_classes: Number of output classes.
        dropout: Dropout in projections and Transformer.
        radiomic_feature_dim: Fixed projection dim for radiomic token; `LazyLinear` when `None`.
        clinical_feature_dim: Fixed projection dim for clinical token; `LazyLinear` when `None`.
        anatomy_feature_dim: Fixed projection dim for anatomy token; `LazyLinear` when `None`.
        vessel_feature_dim: Fixed projection dim for vessel token; `LazyLinear` when `None`.
    """

    def __init__(
        self,
        deep_feature_dims: Sequence[int],
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: Optional[int] = None,
        num_classes: int = 2,
        dropout: float = 0.1,
        radiomic_feature_dim: Optional[int] = None,
        clinical_feature_dim: Optional[int] = None,
        anatomy_feature_dim: Optional[int] = None,
        vessel_feature_dim: Optional[int] = None,
        roi_pool_regions: Optional[Sequence[str]] = None,
        roi_pool_stages: Optional[Sequence[int]] = None,
        token_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Token dropout: randomly mask non-CLS input tokens during training.
        self.token_dropout = float(token_dropout)

        if num_heads is None:
            num_heads = max(1, min(8, embed_dim // 32))
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.deep_feature_dims = tuple(int(d) for d in deep_feature_dims)
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.deep_projections = nn.ModuleList(
            [nn.Linear(dim, embed_dim) for dim in self.deep_feature_dims]
        )

        # Anatomy/vessel features get per-feature BatchNorm before projection;
        # clinical and radiomic are already normalised upstream, so they don't.
        self.radiomic_projection = (
            _NormProjection(radiomic_feature_dim, embed_dim, normalize=False)
            if radiomic_feature_dim is not None
            else nn.LazyLinear(embed_dim)
        )
        self.clinical_projection = (
            nn.Linear(clinical_feature_dim, embed_dim)
            if clinical_feature_dim is not None
            else nn.LazyLinear(embed_dim)
        )
        self.anatomy_projection = (
            _NormProjection(anatomy_feature_dim, embed_dim, normalize=True)
            if anatomy_feature_dim is not None
            else nn.LazyLinear(embed_dim)
        )
        self.vessel_projection = (
            _NormProjection(vessel_feature_dim, embed_dim, normalize=True)
            if vessel_feature_dim is not None
            else nn.LazyLinear(embed_dim)
        )

        # Per-compartment masked mean+std pool at selected stages.
        self.roi_pool_regions = tuple(roi_pool_regions) if roi_pool_regions else ()
        if self.roi_pool_regions:
            n_stage = len(self.deep_feature_dims)
            stages = tuple(roi_pool_stages) if roi_pool_stages else (0, -1)
            self.roi_pool_stages = tuple(int(s) % n_stage for s in stages)
            self.roi_projections = nn.ModuleList(
                [
                    nn.Linear(2 * self.deep_feature_dims[s], embed_dim)
                    for s in self.roi_pool_stages
                ]
            )
            self.roi_region_embed = nn.Embedding(len(self.roi_pool_regions), embed_dim)
        else:
            self.roi_pool_stages = ()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Embedding(256, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _pool_deep_features(
        self, deep_features: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Global-average-pool each deep feature map and project to the token space.

        Args:
            deep_features: One tensor per encoder stage, each either
                `(B, C, D, H, W)` or already-pooled `(B, C)`.

        Returns:
            Deep tokens, shape `(B, n_stage, embed_dim)`.

        Raises:
            ValueError: If the number of tensors doesn't match the configured
                deep feature dims, or a tensor has an unsupported rank.
        """
        if len(deep_features) != len(self.deep_projections):
            raise ValueError(
                f"Expected {len(self.deep_projections)} deep feature tensors, got {len(deep_features)}."
            )

        tokens: List[torch.Tensor] = []
        for feat, proj in zip(deep_features, self.deep_projections):
            if feat.dim() == 5:
                feat_vec = F.adaptive_avg_pool3d(feat, output_size=1).flatten(1)
            elif feat.dim() == 2:
                feat_vec = feat
            else:
                raise ValueError(
                    f"Deep features must have shape (B, C, D, H, W) or (B, C); got {tuple(feat.shape)}."
                )
            tokens.append(proj(feat_vec).unsqueeze(1))

        return torch.cat(tokens, dim=1)

    def _pool_roi_features(
        self,
        deep_features: Sequence[torch.Tensor],
        roi_masks: Optional[Mapping[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Masked mean⊕std per compartment at the configured stages -> ROI tokens.

        An empty mask (count 0) yields a constant zero token rather than NaN.

        Args:
            deep_features: One tensor per encoder stage, `(B, C, D, H, W)` for
                the stages used for ROI pooling.
            roi_masks: Maps region name -> `(B, 1, D, H, W)` (or `(B, D, H,
                W)`) mask at input resolution; each is nearest-downsampled to
                the stage's grid.

        Returns:
            ROI tokens, shape `(B, n_stage*n_region, embed_dim)`, or `None`
            when ROI pooling is off or no masks were supplied.
        """
        if not self.roi_pool_regions or roi_masks is None:
            return None
        tokens: List[torch.Tensor] = []
        for proj, s in zip(self.roi_projections, self.roi_pool_stages):
            feat = deep_features[s]
            if feat.dim() != 5:
                continue
            shape = feat.shape[-3:]
            for ri, region in enumerate(self.roi_pool_regions):
                m = roi_masks.get(region)
                if m is None:
                    continue
                m = m.to(feat.dtype)
                if m.dim() == 4:
                    m = m.unsqueeze(1)
                if tuple(m.shape[-3:]) != tuple(shape):
                    m = F.interpolate(m, size=shape, mode="nearest")
                cnt = m.sum(dim=(2, 3, 4)).clamp_min(1.0)  # (B, 1)
                mean = (feat * m).sum(dim=(2, 3, 4)) / cnt  # (B, C)
                meansq = (feat * feat * m).sum(dim=(2, 3, 4)) / cnt
                # Floor variance before sqrt to avoid NaN gradients.
                std = (meansq - mean * mean).clamp_min(1e-6).sqrt()
                vec = torch.cat([mean, std], dim=1)  # (B, 2C)
                tok = proj(vec) + self.roi_region_embed.weight[ri]
                tokens.append(tok.unsqueeze(1))
        if not tokens:
            return None
        return torch.cat(tokens, dim=1)

    def _project_tabular_features(
        self,
        features: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ],
        projection: nn.Module,
        feature_name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Coerce tabular features to `(B, 1, F)` and project to the token space.

        Args:
            features: Feature tensor (`(B, F)` or `(B, N, F)`) or a mapping
                convertible via `radiomic_mapping_to_tensor`; `None` skips.
            projection: Module that maps `(B, N, F)` -> `(B, N, embed_dim)`.
            feature_name: Name used in error messages.
            batch_size: Expected batch size; a size-1 input is broadcast.
            device: Device to move `features` to.
            dtype: Dtype to cast `features` to.

        Returns:
            Projected token(s), shape `(B, N, embed_dim)`, or `None` when
            `features` is `None`.

        Raises:
            ValueError: If `features` has an unsupported rank, or its batch
                size doesn't match `batch_size` and can't be broadcast.
        """
        if features is None:
            return None

        if isinstance(features, Mapping):
            features = radiomic_mapping_to_tensor(features)

        features = torch.as_tensor(features, device=device, dtype=dtype)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.dim() == 2:
            features = features.unsqueeze(1)
        elif features.dim() != 3:
            raise ValueError(
                f"{feature_name} must have shape (B, F) or (B, N, F); got {tuple(features.shape)}."
            )

        if features.shape[0] != batch_size:
            if features.shape[0] == 1 and batch_size > 1:
                features = features.expand(batch_size, -1, -1)
            else:
                raise ValueError(
                    f"{feature_name} batch size mismatch: expected {batch_size}, got {features.shape[0]}."
                )

        projected = projection(features)
        return projected

    def forward(
        self,
        deep_features: Sequence[torch.Tensor],
        radiomic_features: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ] = None,
        clinical_features: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ] = None,
        anatomy_features: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ] = None,
        vessel_features: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ] = None,
        roi_masks: Optional[Mapping[str, torch.Tensor]] = None,
        return_tokens: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fuse all tokens and return `(logits, aux_dict)`.

        Args:
            deep_features: One tensor per encoder stage; see
                `_pool_deep_features`.
            radiomic_features: Optional radiomic feature tensor or mapping.
            clinical_features: Optional clinical feature tensor or mapping.
            anatomy_features: Optional anatomy feature tensor or mapping.
            vessel_features: Optional vessel feature tensor or mapping.
            roi_masks: Optional per-region masks for ROI-pooled tokens; see
                `_pool_roi_features`.
            return_tokens: If `True`, the aux dict includes the full encoded
                token sequence and every per-source token tensor; otherwise
                it contains only `'embedding'`.

        Returns:
            Tuple of logits `(B, num_classes)` and an aux dict containing
            `'embedding'` and, when `return_tokens=True`, the full token sequence.
        """
        deep_tokens = self._pool_deep_features(deep_features)
        roi_tokens = self._pool_roi_features(deep_features, roi_masks)
        batch_size = deep_tokens.shape[0]
        device = deep_tokens.device
        dtype = deep_tokens.dtype

        radiomic_tokens = self._project_tabular_features(
            features=radiomic_features,
            projection=self.radiomic_projection,
            feature_name="radiomic_features",
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        clinical_tokens = self._project_tabular_features(
            features=clinical_features,
            projection=self.clinical_projection,
            feature_name="clinical_features",
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        anatomy_tokens = self._project_tabular_features(
            features=anatomy_features,
            projection=self.anatomy_projection,
            feature_name="anatomy_features",
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        vessel_tokens = self._project_tabular_features(
            features=vessel_features,
            projection=self.vessel_projection,
            feature_name="vessel_features",
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        tokens = deep_tokens
        if roi_tokens is not None:
            tokens = torch.cat([tokens, roi_tokens], dim=1)
        if radiomic_tokens is not None:
            tokens = torch.cat([tokens, radiomic_tokens], dim=1)
        if clinical_tokens is not None:
            tokens = torch.cat([tokens, clinical_tokens], dim=1)
        if anatomy_tokens is not None:
            tokens = torch.cat([tokens, anatomy_tokens], dim=1)
        if vessel_tokens is not None:
            tokens = torch.cat([tokens, vessel_tokens], dim=1)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        pos_ids = torch.arange(tokens.shape[1], device=device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)
        tokens = self.input_norm(tokens)
        tokens = self.dropout(tokens)

        key_padding_mask: Optional[torch.Tensor] = None
        if self.training and self.token_dropout > 0.0:
            drop = (
                torch.rand(tokens.shape[0], tokens.shape[1], device=device)
                < self.token_dropout
            )
            drop[:, 0] = False
            key_padding_mask = drop

        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        pooled = self.final_norm(encoded[:, 0])
        logits = self.classifier(pooled)

        aux: Dict[str, torch.Tensor] = {
            "tokens": encoded,
            "embedding": pooled,
            "deep_tokens": deep_tokens,
        }
        if roi_tokens is not None:
            aux["roi_tokens"] = roi_tokens
        if radiomic_tokens is not None:
            aux["radiomic_tokens"] = radiomic_tokens
        if clinical_tokens is not None:
            aux["clinical_tokens"] = clinical_tokens
        if anatomy_tokens is not None:
            aux["anatomy_tokens"] = anatomy_tokens
        if vessel_tokens is not None:
            aux["vessel_tokens"] = vessel_tokens

        if return_tokens:
            return logits, aux
        return logits, {"embedding": pooled}
