"""Full longitudinal PDAC response model: Siamese encoder -> cross-timepoint attention -> fusion head."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from pdac_longitudinal.fusion.token_fusion import TokenFusionHead
from pdac_longitudinal.models.cross_timepoint_attention import (
    CrossTimepointAttentionStack,
)
from pdac_longitudinal.models.siamese_encoder import SiameseResEncLEncoder

if TYPE_CHECKING:
    from pdac_longitudinal.config import Config


class LongitudinalResponseModel(nn.Module):
    """Siamese longitudinal encoder with cross-timepoint attention and fusion.

    Args:
        weights_path: Path to the pretrained nnU-Net v2 ResEncL checkpoint.
        input_channels: Number of input image channels.
        encoder_arch_kwargs: Architecture kwargs forwarded to
            `SiameseResEncLEncoder`; defaults to `RESENCL_3D_DEFAULTS` there.
        num_classes: Number of response classes for the fusion head.
        encoder_freeze: If `True`, freeze encoder parameters on construction.
        attention_window_sizes: Per-stage window sizes forwarded to
            `CrossTimepointAttentionStack`.
        attention_num_heads: Per-stage attention head counts forwarded to
            `CrossTimepointAttentionStack`.
        attention_use_gradient_checkpointing: Forwarded to
            `CrossTimepointAttentionStack`.
        attention_pass_through_stages: Forwarded to
            `CrossTimepointAttentionStack`.
        fusion_embed_dim: Shared token embedding dimension for the fusion head.
        fusion_layers: Number of Transformer layers in the fusion head.
        fusion_heads: Attention heads in the fusion head; auto-chosen when
            `None`.
        radiomic_feature_dim: Fixed radiomic feature dim; `LazyLinear` when
            `None`.
        clinical_feature_dim: Fixed clinical feature dim; `LazyLinear` when
            `None`.
        anatomy_feature_dim: Fixed anatomy feature dim; `LazyLinear` when
            `None`.
        vessel_feature_dim: Fixed vessel feature dim; `LazyLinear` when
            `None`.
        fusion_dropout: Dropout used in the attention stack and fusion head.
        use_imaging: If `False`, the deep imaging tokens are zeroed in
            `forward()`; the encoder still runs.
        roi_pool_regions: Per-compartment ROI regions forwarded to
            `TokenFusionHead`.
        roi_pool_stages: Stages used for ROI pooling, forwarded to
            `TokenFusionHead`.
        token_dropout: Token dropout forwarded to `TokenFusionHead`.
    """

    def __init__(
        self,
        weights_path: Union[str, "Path"],
        input_channels: int = 1,
        encoder_arch_kwargs: Optional[Dict[str, Any]] = None,
        num_classes: int = 2,
        encoder_freeze: bool = True,
        attention_window_sizes: Optional[Sequence[Tuple[int, int, int]]] = None,
        attention_num_heads: Optional[Sequence[Optional[int]]] = None,
        attention_use_gradient_checkpointing: bool = False,
        attention_pass_through_stages: Optional[Sequence[int]] = None,
        fusion_embed_dim: int = 128,
        fusion_layers: int = 2,
        fusion_heads: Optional[int] = None,
        radiomic_feature_dim: Optional[int] = None,
        clinical_feature_dim: Optional[int] = None,
        anatomy_feature_dim: Optional[int] = None,
        vessel_feature_dim: Optional[int] = None,
        fusion_dropout: float = 0.1,
        use_imaging: bool = True,
        roi_pool_regions: Optional[Sequence[str]] = None,
        roi_pool_stages: Optional[Sequence[int]] = None,
        token_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.use_imaging = use_imaging
        self.encoder = SiameseResEncLEncoder(
            weights_path=weights_path,
            input_channels=input_channels,
            arch_kwargs=encoder_arch_kwargs,
            freeze_encoder=encoder_freeze,
        )
        self.attention_stack = CrossTimepointAttentionStack(
            features_per_stage=self.encoder.features_per_stage,
            window_sizes=attention_window_sizes,
            num_heads_per_stage=list(attention_num_heads)
            if attention_num_heads is not None
            else None,
            dropout=fusion_dropout,
            use_gradient_checkpointing=attention_use_gradient_checkpointing,
            pass_through_stages=attention_pass_through_stages,
        )
        self.fusion_head = TokenFusionHead(
            deep_feature_dims=self.encoder.features_per_stage,
            embed_dim=fusion_embed_dim,
            num_layers=fusion_layers,
            num_heads=fusion_heads,
            num_classes=num_classes,
            dropout=fusion_dropout,
            radiomic_feature_dim=radiomic_feature_dim,
            clinical_feature_dim=clinical_feature_dim,
            anatomy_feature_dim=anatomy_feature_dim,
            vessel_feature_dim=vessel_feature_dim,
            roi_pool_regions=roi_pool_regions,
            roi_pool_stages=roi_pool_stages,
            token_dropout=token_dropout,
        )
        self.risk_head = nn.Linear(fusion_embed_dim, 1)
        from pdac_longitudinal.models.grl import PhaseAdversary

        self.phase_adv = PhaseAdversary(in_dim=fusion_embed_dim, n_phases=2)

    @property
    def feature_dims(self) -> Tuple[int, ...]:
        """Channel dimensions of the encoder stages."""
        return self.encoder.features_per_stage

    @staticmethod
    def _build_valid_pairs(
        valid_T0: Optional[torch.Tensor],
        valid_T1: Optional[torch.Tensor],
        stage_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]]:
        """Downsample valid masks to each stage's resolution.

        Args:
            valid_T0: Optional T0 valid mask at input resolution.
            valid_T1: Optional T1 valid mask at input resolution.
            stage_pairs: Per-stage `(feat_T0, feat_T1)` tensors, used only
                for their target spatial shape.

        Returns:
            One `(valid_T0, valid_T1)` tuple per stage, downsampled to that
            stage's resolution, or `None` when both inputs are `None`.
        """
        if valid_T0 is None and valid_T1 is None:
            return None
        import torch.nn.functional as F

        def _down(
            m: Optional[torch.Tensor], ref: torch.Tensor
        ) -> Optional[torch.Tensor]:
            if m is None:
                return None
            target = ref.shape[-3:]
            if tuple(m.shape[-3:]) == tuple(target):
                return m.bool() if m.dtype != torch.bool else m
            return F.interpolate(m.float(), size=target, mode="nearest").bool()

        out: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = []
        for feat_T0, feat_T1 in stage_pairs:
            out.append((_down(valid_T0, feat_T0), _down(valid_T1, feat_T1)))
        return out

    def encode(
        self,
        x_t0: torch.Tensor,
        x_t1: torch.Tensor,
        valid_T0: Optional[torch.Tensor] = None,
        valid_T1: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        skip_attn_stages: Sequence[int] = (),
    ) -> Tuple[
        List[torch.Tensor],
        List[Dict[str, torch.Tensor]],
        List[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Run the Siamese encoder and cross-timepoint attention stack.

        Args:
            x_t0: T0 input volume, shape `(B, C, D, H, W)`.
            x_t1: T1 input volume, same shape as `x_t0`.
            valid_T0: Optional T0 valid mask at input resolution.
            valid_T1: Optional T1 valid mask at input resolution.
            return_attn: If `True`, materialise per-window attention weight
                tensors (memory-heavy).
            skip_attn_stages: Stage indices to exclude from attention
                materialisation even when `return_attn=True`.

        Returns:
            Tuple of the fused feature map per stage, the attention-map dict
            per stage, and the raw `(feat_T0, feat_T1)` encoder pairs.
        """
        stage_pairs = self.encoder(x_t0, x_t1)
        valid_pairs = self._build_valid_pairs(valid_T0, valid_T1, stage_pairs)
        fused_maps, attn_maps = self.attention_stack(
            stage_pairs,
            valid_pairs=valid_pairs,
            return_attn=return_attn,
            skip_attn_stages=skip_attn_stages,
        )
        return fused_maps, attn_maps, stage_pairs

    def forward(
        self,
        x_t0: torch.Tensor,
        x_t1: torch.Tensor,
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
        valid_T0: Optional[torch.Tensor] = None,
        valid_T1: Optional[torch.Tensor] = None,
        return_tokens: bool = False,
        return_attn: bool = False,
        skip_attn_stages: Sequence[int] = (),
    ) -> Dict[str, Any]:
        """Full forward pass: encode, fuse, and predict risk/response.

        Args:
            x_t0: T0 input volume, shape `(B, C, D, H, W)`.
            x_t1: T1 input volume, same shape as `x_t0`.
            radiomic_features: Optional radiomic feature tensor or mapping.
            clinical_features: Optional clinical feature tensor or mapping.
            anatomy_features: Optional anatomy feature tensor or mapping.
            vessel_features: Optional vessel feature tensor or mapping.
            roi_masks: Optional per-region masks for ROI-pooled tokens.
            valid_T0: Optional T0 valid mask at input resolution.
            valid_T1: Optional T1 valid mask at input resolution.
            return_tokens: If `True`, include the fusion head's token
                tensors in the output dict.
            return_attn: If `True`, materialise cross-attention maps.
            skip_attn_stages: Stage indices to skip when materialising
                attention maps.

        Returns:
            Dict with `'risk'`, `'response_logits'`/`'logits'`, `'embedding'`,
            `'attention_maps'`, `'fused_maps'`, `'stage_pairs'`, and, when
            `return_tokens=True`, the per-source token tensors.
        """
        fused_maps, attn_maps, stage_pairs = self.encode(
            x_t0,
            x_t1,
            valid_T0=valid_T0,
            valid_T1=valid_T1,
            return_attn=return_attn,
            skip_attn_stages=skip_attn_stages,
        )
        # Imaging ablation: zero the deep tokens.
        if not self.use_imaging:
            fused_maps = [torch.zeros_like(f) for f in fused_maps]
        logits, fusion_aux = self.fusion_head(
            deep_features=fused_maps,
            radiomic_features=radiomic_features,
            clinical_features=clinical_features,
            anatomy_features=anatomy_features,
            vessel_features=vessel_features,
            roi_masks=roi_masks,
            return_tokens=True,
        )
        risk = self.risk_head(fusion_aux["embedding"]).squeeze(1)

        output: Dict[str, Any] = {
            "risk": risk,
            "response_logits": logits,
            "logits": logits,
            "embedding": fusion_aux["embedding"],
            "attention_maps": attn_maps,
            "fused_maps": fused_maps,
            "stage_pairs": stage_pairs,
        }

        if return_tokens:
            output["tokens"] = fusion_aux["tokens"]
            output["deep_tokens"] = fusion_aux["deep_tokens"]
            if "roi_tokens" in fusion_aux:
                output["roi_tokens"] = fusion_aux["roi_tokens"]
            if "radiomic_tokens" in fusion_aux:
                output["radiomic_tokens"] = fusion_aux["radiomic_tokens"]
            if "clinical_tokens" in fusion_aux:
                output["clinical_tokens"] = fusion_aux["clinical_tokens"]
            if "anatomy_tokens" in fusion_aux:
                output["anatomy_tokens"] = fusion_aux["anatomy_tokens"]
            if "vessel_tokens" in fusion_aux:
                output["vessel_tokens"] = fusion_aux["vessel_tokens"]

        return output


def build_model_from_config(cfg: "Config") -> LongitudinalResponseModel:
    """Convenience constructor driven by a typed `Config`.

    Args:
        cfg: Loaded `Config`; reads its `encoder`, `fusion`, `attention`,
            and `modules` sections. See `LongitudinalResponseModel` for the
            individual settings each maps to.

    Returns:
        A configured `LongitudinalResponseModel`.
    """
    encoder_cfg = cfg.encoder
    fusion_cfg = cfg.fusion
    attention_cfg = cfg.attention

    return LongitudinalResponseModel(
        weights_path=encoder_cfg.get(
            "weights_path", "./weights/nnunet_resencl_pdac.pth"
        ),
        input_channels=encoder_cfg.get("input_channels", 1),
        encoder_arch_kwargs=encoder_cfg.get("arch_kwargs"),
        num_classes=fusion_cfg.get("num_classes", 2),
        encoder_freeze=encoder_cfg.get("freeze_on_init", True),
        attention_window_sizes=attention_cfg.get("window_sizes"),
        attention_num_heads=attention_cfg.get("num_heads_per_stage"),
        attention_use_gradient_checkpointing=attention_cfg.get(
            "use_gradient_checkpointing", False
        ),
        attention_pass_through_stages=attention_cfg.get("pass_through_stages"),
        fusion_embed_dim=fusion_cfg.get("radiomic_embed_dim", 128),
        fusion_layers=fusion_cfg.get("num_fusion_layers", 2),
        fusion_heads=fusion_cfg.get("num_heads"),
        radiomic_feature_dim=fusion_cfg.get("radiomic_feature_dim"),
        clinical_feature_dim=fusion_cfg.get("clinical_feature_dim"),
        anatomy_feature_dim=fusion_cfg.get("anatomy_feature_dim"),
        vessel_feature_dim=fusion_cfg.get("vessel_feature_dim"),
        fusion_dropout=fusion_cfg.get("dropout", 0.1),
        use_imaging=cfg.modules.imaging,
        roi_pool_regions=fusion_cfg.get("roi_pool_regions"),
        roi_pool_stages=fusion_cfg.get("roi_pool_stages"),
        token_dropout=fusion_cfg.get("token_dropout", 0.0),
    )
