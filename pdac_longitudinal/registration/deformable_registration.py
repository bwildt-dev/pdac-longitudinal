"""MONAI LocalNet deformable registration: T1 (moving) -> T0 (fixed) space."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

from monai.losses import BendingEnergyLoss, LocalNormalizedCrossCorrelationLoss
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet

logger = logging.getLogger(__name__)


class DeformableRegistration(nn.Module):
    """MONAI LocalNet deformable registration with LNCC + BendingEnergy loss.

    Args:
        spatial_dims: Spatial dimensionality; only 3 is supported.
        in_channels: Channels fed to LocalNet (moving + fixed stacked).
        num_channel_initial: LocalNet's initial encoder channel width.
        extract_levels: LocalNet decoder levels to extract skip features from.
        lncc_kernel_size: LNCC local-window size, in voxels.
        reg_weight: Weight on the bending-energy regularizer in the combined loss.

    Raises:
        ValueError: If `spatial_dims` is not 3.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 2,
        num_channel_initial: int = 16,
        extract_levels: Tuple[int, ...] = (0, 1, 2, 3),
        lncc_kernel_size: int = 9,
        reg_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if spatial_dims != 3:
            raise ValueError(
                f"DeformableRegistration only supports spatial_dims=3, got {spatial_dims}."
            )

        self.reg_weight = reg_weight

        self.localnet = LocalNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            extract_levels=extract_levels,
            out_activation=None,
            out_channels=spatial_dims,
        )

        self.warp_bilinear = Warp(mode="bilinear", padding_mode="border")
        self.warp_nearest  = Warp(mode="nearest",  padding_mode="border")

        self.lncc_loss = LocalNormalizedCrossCorrelationLoss(
            spatial_dims=spatial_dims,
            kernel_size=lncc_kernel_size,
            kernel_type="rectangular",
            reduction="mean",
        )
        self.bending_energy = BendingEnergyLoss()

        logger.info(
            "DeformableRegistration: LocalNet (initial_ch=%d, levels=%s)",
            num_channel_initial, extract_levels,
        )

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(warped, ddf)`; T1 warped into T0 space and the DDF `(B,3,D,H,W)`.

        Args:
            moving: T1 volume(s) to warp, `(B,1,D,H,W)`.
            fixed: T0 volume(s) defining the target space, `(B,1,D,H,W)`.
        """
        net_input = torch.cat([moving, fixed], dim=1)
        ddf = self.localnet(net_input)
        warped = self.warp_bilinear(moving, ddf)
        return warped, ddf

    def compute_loss(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        ddf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(total, lncc, bending)` scalars for the combined registration loss.

        Args:
            warped: Moving volume warped by `ddf`, from `forward`.
            fixed: Target (T0) volume.
            ddf: Dense displacement field from `forward`.
        """
        lncc = self.lncc_loss(warped, fixed)
        bending = self.bending_energy(ddf)
        total = lncc + self.reg_weight * bending
        return total, lncc, bending

    def fit_pair(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        iterations: int = 100,
        lr: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit the LocalNet on a single moving/fixed pair; return `(warped, ddf)`.

        The LocalNet has no pretrained weights, so it is optimised from scratch
        on this one pair against the combined LNCC + bending-energy loss.

        Args:
            moving: T1 volume to warp, `(1,1,D,H,W)`.
            fixed: T0 volume defining the target space, `(1,1,D,H,W)`.
            iterations: Optimisation steps.
            lr: Adam learning rate.
        """
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(max(1, iterations)):
            opt.zero_grad(set_to_none=True)
            warped, ddf = self(moving=moving, fixed=fixed)
            total, _, _ = self.compute_loss(warped, fixed, ddf)
            total.backward()
            opt.step()
        self.eval()
        with torch.no_grad():
            warped, ddf = self(moving=moving, fixed=fixed)
        return warped, ddf

    def warp_image(self, image: torch.Tensor, ddf: torch.Tensor) -> torch.Tensor:
        """Warp image with bilinear interpolation (CT intensities).

        Args:
            image: Volume to warp.
            ddf: Dense displacement field, from `forward`.
        """
        return self.warp_bilinear(image, ddf)

    def warp_mask(self, mask: torch.Tensor, ddf: torch.Tensor) -> torch.Tensor:
        """Warp mask with nearest-neighbour interpolation (preserves integer labels).

        Args:
            mask: Label mask to warp.
            ddf: Dense displacement field, from `forward`.
        """
        return self.warp_nearest(mask, ddf)
