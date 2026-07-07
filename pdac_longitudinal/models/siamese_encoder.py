"""Siamese wrapper around the pretrained nnU-Net v2 encoder."""

from __future__ import annotations

import logging
import pydoc
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder

logger = logging.getLogger(__name__)


RESENCL_3D_DEFAULTS: Dict = {
    "n_stages": 6,
    "features_per_stage": [32, 64, 128, 256, 320, 320],
    "conv_op": "torch.nn.modules.conv.Conv3d",
    "kernel_sizes": [[3, 3, 3]] * 6,
    "strides": [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
    "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
    "conv_bias": True,
    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
    "dropout_op": None,
    "dropout_op_kwargs": None,
    "nonlin": "torch.nn.LeakyReLU",
    "nonlin_kwargs": {"inplace": True},
}


class SiameseResEncLEncoder(nn.Module):
    """Siamese encoder wrapping the nnU-Net v2 ResEncL for longitudinal CT encoding.

    Args:
        weights_path: Path to the nnunetv2 checkpoint to load encoder
            weights from.
        input_channels: Number of input image channels.
        num_classes: Number of segmentation classes for the full model built
            to receive the checkpoint.
        arch_kwargs: Architecture kwargs for `ResidualEncoderUNet`; defaults
            to `RESENCL_3D_DEFAULTS` when `None`.
        freeze_encoder: If `True`, freeze all encoder parameters on
            construction; call `unfreeze_encoder()` before fine-tuning.
    """

    def __init__(
        self,
        weights_path: str,
        input_channels: int = 1,
        num_classes: int = 19,
        arch_kwargs: Optional[Dict] = None,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()

        if arch_kwargs is None:
            arch_kwargs = RESENCL_3D_DEFAULTS

        full_model = self._build_full_model(input_channels, num_classes, arch_kwargs)
        self._load_encoder_weights(full_model, weights_path)

        self.encoder: ResidualEncoder = full_model.encoder
        del full_model

        if not getattr(self.encoder, "return_skips", False):
            logger.warning(
                "Encoder.return_skips is False — forcing True so the Siamese "
                "wrapper sees every U-Net skip level, not only the bottleneck."
            )
            self.encoder.return_skips = True

        self.features_per_stage: Tuple[int, ...] = tuple(arch_kwargs["features_per_stage"])
        self.n_stages: int = arch_kwargs["n_stages"]

        if freeze_encoder:
            self.freeze_encoder()
            logger.info("Encoder frozen. Call unfreeze_encoder() before fine-tuning.")

    @staticmethod
    def _build_full_model(
        input_channels: int,
        num_classes: int,
        arch_kwargs: Dict,
    ) -> ResidualEncoderUNet:
        """Instantiate `ResidualEncoderUNet` from `arch_kwargs`, resolving dotted class paths.

        Args:
            input_channels: Number of input image channels.
            num_classes: Number of segmentation output classes.
            arch_kwargs: Architecture kwargs, with dotted class-path strings
                resolved to classes.

        Returns:
            The instantiated `ResidualEncoderUNet`.

        Raises:
            ImportError: If a dotted class path in `arch_kwargs` can't be
                resolved.
        """
        kwargs = dict(arch_kwargs)

        for key in ("conv_op", "norm_op", "dropout_op", "nonlin"):
            if key in kwargs and isinstance(kwargs[key], str):
                resolved = pydoc.locate(kwargs[key])
                if resolved is None:
                    raise ImportError(
                        f"Could not resolve class reference for '{key}': "
                        f"{kwargs[key]!r}.  Check the arch_kwargs / config.yaml."
                    )
                kwargs[key] = resolved

        return ResidualEncoderUNet(
            input_channels=input_channels,
            num_classes=num_classes,
            **kwargs,
        )

    def _load_encoder_weights(
        self,
        full_model: ResidualEncoderUNet,
        weights_path: Union[str, Path],
    ) -> None:
        """Load `encoder.*` parameters from a nnunetv2 checkpoint into `full_model`.

        Args:
            full_model: Model to load the encoder weights into, in place.
            weights_path: Path to the nnunetv2 checkpoint.

        Raises:
            FileNotFoundError: If `weights_path` doesn't exist.
            ValueError: If the checkpoint format isn't recognised.
            RuntimeError: If any `encoder.*` parameter fails to load.
        """
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Pretrained weights not found at: {path}\n"
                "Set the correct path in your config's encoder.weights_path"
            )

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "network_weights" in checkpoint:
            raw_state_dict: Dict[str, torch.Tensor] = checkpoint["network_weights"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            raw_state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            raw_state_dict = checkpoint
        else:
            raise ValueError(
                f"Unrecognised checkpoint format in {path}.  Expected a dict "
                "with key 'network_weights', 'state_dict', or a bare state dict."
            )

        # Strip DDP 'module.' prefix.
        state_dict: Dict[str, torch.Tensor] = {}
        for k, v in raw_state_dict.items():
            state_dict[k[len("module."):] if k.startswith("module.") else k] = v

        total_ckpt_keys = len(state_dict)

        encoder_state: Dict[str, torch.Tensor] = {
            k: v for k, v in state_dict.items() if k.startswith("encoder.")
        }
        skipped_keys: List[str] = [
            k for k in state_dict if not k.startswith("encoder.")
        ]

        # encoder_state only has encoder.* keys, so strict=False just tolerates
        # full_model's decoder params being absent from it; still raises on a missing encoder key.
        missing_keys, unexpected_keys = full_model.load_state_dict(
            encoder_state, strict=False
        )

        missing_encoder: List[str] = [
            k for k in missing_keys if k.startswith("encoder.")
        ]

        sep = "-" * 66
        print(sep)
        print("  Pretrained encoder weight loading report")
        print(sep)
        print(f"  Checkpoint      : {path}")
        print(f"  Total ckpt keys : {total_ckpt_keys}")
        print(f"  Encoder loaded  : {len(encoder_state)} keys")
        print(
            f"  Decoder skipped : {len(skipped_keys)} keys  "
        )

        if missing_encoder:
            print(f"\n  [FAIL] Missing encoder keys ({len(missing_encoder)}):")
            for k in missing_encoder:
                print(f"    ✗ {k}")
        else:
            print("\n  [OK] All encoder keys matched — no missing encoder parameters.")

        if unexpected_keys:
            print(f"\n  [WARN] Unexpected keys supplied but not consumed ({len(unexpected_keys)}):")
            for k in unexpected_keys[:10]:
                print(f"    ? {k}")
            if len(unexpected_keys) > 10:
                print(f"    … and {len(unexpected_keys) - 10} more")

        if skipped_keys:
            prefixes: set = set()
            for k in skipped_keys:
                parts = k.split(".")
                prefixes.add(".".join(parts[:2]) if len(parts) > 1 else parts[0])
            print("\n  Skipped parameter groups (decoder / segmentation head):")
            for prefix in sorted(prefixes):
                count = sum(1 for k in skipped_keys if k.startswith(prefix))
                print(f"    - {prefix}.*  ({count} tensors)")

        print(sep)

        if missing_encoder:
            raise RuntimeError(
                f"Failed to load {len(missing_encoder)} encoder parameter(s) — "
                "verify that arch_kwargs in config.yaml matches the checkpoint "
                "architecture exactly."
            )

        if unexpected_keys and any(k.startswith("encoder.") for k in unexpected_keys):
            logger.warning(
                "Some encoder keys from the checkpoint were not consumed: %s",
                [k for k in unexpected_keys if k.startswith("encoder.")],
            )

    def freeze_encoder(self) -> None:
        """Set `requires_grad=False` on all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.debug("Encoder frozen (%d parameter tensors).", sum(1 for _ in self.encoder.parameters()))

    def unfreeze_encoder(self) -> None:
        """Set `requires_grad=True` on all encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.debug("Encoder unfrozen — all parameters trainable.")

    def unfreeze_stages(self, stage_indices: Sequence[int]) -> int:
        """Unfreeze only the given encoder stage indices.

        Args:
            stage_indices: Stage indices to unfreeze; negative indices wrap.
                The stem is unfrozen too when stage 0 is included.

        Returns:
            Number of parameter tensors made trainable.
        """
        stages = self.encoder.stages
        n = len(stages)
        idx = {(i % n) for i in stage_indices}
        n_trainable = 0
        for i, stage in enumerate(stages):
            req = i in idx
            for p in stage.parameters():
                p.requires_grad_(req)
                if req:
                    n_trainable += 1
        stem = getattr(self.encoder, "stem", None)
        if stem is not None:
            train_stem = 0 in idx
            for p in stem.parameters():
                p.requires_grad_(train_stem)
                if train_stem:
                    n_trainable += 1
        logger.info(
            "Encoder partial-unfreeze: stages %s trainable (%d param tensors); "
            "stages %s remain frozen.",
            sorted(idx), n_trainable, sorted(set(range(n)) - idx),
        )
        return n_trainable

    def forward(
        self,
        x_T0: torch.Tensor,
        x_T1: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return a list of `(feat_T0, feat_T1)` tuples, one per encoder stage.

        Args:
            x_T0: T0 input volume, shape `(B, C, D, H, W)`.
            x_T1: T1 input volume, same shape as `x_T0`.
        """
        feats_T0: List[torch.Tensor] = self.encoder(x_T0)
        feats_T1: List[torch.Tensor] = self.encoder(x_T1)

        return list(zip(feats_T0, feats_T1))
