"""Unit tests for the TokenFusionHead transformer fusion head."""

from __future__ import annotations

import pytest
import torch

from pdac_longitudinal.fusion.token_fusion import TokenFusionHead, radiomic_mapping_to_tensor


def test_radiomic_mapping_to_tensor_stacks_sorted_and_rejects_bad_input():
    vec = radiomic_mapping_to_tensor({"b": 2.0, "a": 1.0})
    assert torch.equal(vec, torch.tensor([[1.0, 2.0]]))  # sorted by key: a, b
    with pytest.raises(ValueError):
        radiomic_mapping_to_tensor({})
    with pytest.raises(ValueError):
        radiomic_mapping_to_tensor({"a": torch.zeros(2), "b": torch.zeros(3)})


def test_forward_with_only_deep_features():
    head = TokenFusionHead(deep_feature_dims=[8, 16], embed_dim=32, num_layers=1)
    deep = [torch.randn(3, 8, 4, 4, 4), torch.randn(3, 16, 2, 2, 2)]
    logits, aux = head(deep)
    assert logits.shape == (3, 2)
    assert aux["embedding"].shape == (3, 32)


def test_forward_with_all_tabular_branches_and_gradients_flow():
    head = TokenFusionHead(
        deep_feature_dims=[8], embed_dim=32, num_layers=1,
        radiomic_feature_dim=5, clinical_feature_dim=4,
        anatomy_feature_dim=6, vessel_feature_dim=3,
    )
    deep = [torch.randn(2, 8, 4, 4, 4, requires_grad=True)]
    logits, aux = head(
        deep,
        radiomic_features=torch.randn(2, 5),
        clinical_features=torch.randn(2, 4),
        anatomy_features=torch.randn(2, 6),
        vessel_features=torch.randn(2, 3),
        return_tokens=True,
    )
    assert logits.shape == (2, 2)
    for k in ("radiomic_tokens", "clinical_tokens", "anatomy_tokens", "vessel_tokens"):
        assert k in aux
    logits.sum().backward()
    assert deep[0].grad is not None
    assert torch.isfinite(deep[0].grad).all()


def test_forward_with_radiomic_mapping_input():
    head = TokenFusionHead(deep_feature_dims=[8], embed_dim=32, num_layers=1)
    deep = [torch.randn(2, 8, 4, 4, 4)]
    logits, _ = head(deep, radiomic_features={"T0_foo": 1.0, "T1_foo": 2.0})
    assert logits.shape == (2, 2)


def test_roi_pooling_produces_extra_tokens_and_empty_mask_is_finite():
    head = TokenFusionHead(
        deep_feature_dims=[8, 16], embed_dim=32, num_layers=1,
        roi_pool_regions=["tumour", "liver"], roi_pool_stages=(0, -1),
    )
    deep = [torch.randn(2, 8, 4, 4, 4), torch.randn(2, 16, 2, 2, 2)]
    roi_masks = {
        "tumour": torch.zeros(2, 1, 4, 4, 4),   # empty mask must not NaN
        "liver": torch.ones(2, 1, 4, 4, 4),
    }
    logits, aux = head(deep, roi_masks=roi_masks, return_tokens=True)
    assert torch.isfinite(logits).all()
    assert aux["roi_tokens"].shape[1] == 4  # 2 stages x 2 regions


def test_batch_size_mismatch_raises():
    head = TokenFusionHead(deep_feature_dims=[8], embed_dim=32, num_layers=1, clinical_feature_dim=4)
    deep = [torch.randn(3, 8, 4, 4, 4)]
    with pytest.raises(ValueError):
        head(deep, clinical_features=torch.randn(2, 4))  # batch 2 vs deep batch 3
