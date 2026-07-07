"""Unit tests for SiameseResEncLEncoder's checkpoint loading and forward pass.

Uses a tiny architecture (not the real ResEncL) so this runs in milliseconds
on CPU with no pretrained weights.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pdac_longitudinal.models.siamese_encoder import SiameseResEncLEncoder

_TINY_ARCH = dict(
    n_stages=2, features_per_stage=[4, 8],
    conv_op=nn.Conv3d, kernel_sizes=[[3, 3, 3]] * 2,
    strides=[[1, 1, 1], [2, 2, 2]], n_blocks_per_stage=[1, 1],
    n_conv_per_stage_decoder=[1],
    conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
    dropout_op=None, dropout_op_kwargs=None,
    nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
)


def _save_full_checkpoint(tmp_path, drop_keys=()):
    full = SiameseResEncLEncoder._build_full_model(1, 3, _TINY_ARCH)
    sd = {k: v for k, v in full.state_dict().items() if k not in drop_keys}
    path = tmp_path / "ckpt.pth"
    torch.save({"network_weights": sd}, path)
    return path


def test_loads_encoder_weights_and_forward_shapes(tmp_path):
    ckpt = _save_full_checkpoint(tmp_path)
    enc = SiameseResEncLEncoder(
        str(ckpt), input_channels=1, num_classes=3, arch_kwargs=_TINY_ARCH, freeze_encoder=True,
    )
    assert enc.encoder.return_skips is True
    assert all(not p.requires_grad for p in enc.encoder.parameters())

    x0 = torch.randn(1, 1, 16, 16, 16)
    x1 = torch.randn(1, 1, 16, 16, 16)
    feats = enc(x0, x1)
    assert len(feats) == enc.n_stages
    for (f0, f1), c in zip(feats, enc.features_per_stage):
        assert f0.shape[:2] == (1, c)
        assert f1.shape == f0.shape


def test_unfreeze_encoder_makes_all_params_trainable(tmp_path):
    ckpt = _save_full_checkpoint(tmp_path)
    enc = SiameseResEncLEncoder(str(ckpt), 1, 3, arch_kwargs=_TINY_ARCH, freeze_encoder=True)
    enc.unfreeze_encoder()
    assert all(p.requires_grad for p in enc.encoder.parameters())


def test_unfreeze_stages_only_trains_selected_stages(tmp_path):
    ckpt = _save_full_checkpoint(tmp_path)
    enc = SiameseResEncLEncoder(str(ckpt), 1, 3, arch_kwargs=_TINY_ARCH, freeze_encoder=True)
    enc.unfreeze_stages([1])  # last stage only
    trainable_stages = [
        i for i, stage in enumerate(enc.encoder.stages)
        if all(p.requires_grad for p in stage.parameters())
    ]
    assert trainable_stages == [1]
    assert all(not p.requires_grad for p in enc.encoder.stages[0].parameters())


def test_missing_encoder_key_raises_runtime_error(tmp_path):
    ckpt = _save_full_checkpoint(tmp_path, drop_keys=("encoder.stem.convs.0.conv.weight",))
    with pytest.raises(RuntimeError):
        SiameseResEncLEncoder(str(ckpt), 1, 3, arch_kwargs=_TINY_ARCH)


def test_missing_weights_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        SiameseResEncLEncoder(str(tmp_path / "nope.pth"), 1, 3, arch_kwargs=_TINY_ARCH)
