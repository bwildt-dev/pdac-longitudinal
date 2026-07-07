"""Per-pair instance optimisation should reduce the registration loss."""

import torch

from pdac_longitudinal.registration.deformable_registration import DeformableRegistration


def _shifted_blob(shape=(24, 24, 24), shift=(3, 0, 0)):
    """A smooth Gaussian blob and a copy shifted by `shift` voxels."""
    zz, yy, xx = torch.meshgrid(
        *[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij"
    )
    c = [s / 2 for s in shape]
    fixed = torch.exp(-(((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / 20.0))
    cm = [c[i] + shift[i] for i in range(3)]
    moving = torch.exp(-(((zz - cm[0]) ** 2 + (yy - cm[1]) ** 2 + (xx - cm[2]) ** 2) / 20.0))
    return moving[None, None], fixed[None, None]


def test_fit_pair_reduces_loss():
    torch.manual_seed(0)
    moving, fixed = _shifted_blob()
    reg = DeformableRegistration(
        num_channel_initial=4, extract_levels=(0, 1), lncc_kernel_size=3, reg_weight=0.1
    )

    warped0, ddf0 = reg(moving=moving, fixed=fixed)
    before, _, _ = reg.compute_loss(warped0, fixed, ddf0)

    warped1, ddf1 = reg.fit_pair(moving, fixed, iterations=30, lr=1e-2)
    after, _, _ = reg.compute_loss(warped1, fixed, ddf1)

    assert after < before
    assert warped1.shape == moving.shape
