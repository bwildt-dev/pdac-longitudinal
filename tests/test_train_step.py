"""End-to-end CPU smoke test for training/loop.py's train_step.

Exercises the real feature-assembly, loss, and backward wiring. Uses a tiny
stand-in model instead of the real SiameseResEncLEncoder, since train_step
only depends on the model's forward() I/O contract, not its internals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from conftest import _save_nii, synthetic_ct, synthetic_seg
from pdac_longitudinal.data.longitudinal_dataset import LongitudinalCTDataset
from pdac_longitudinal.data.registry import ClinicalRegistry
from pdac_longitudinal.losses.survival import CoxPHLoss
from pdac_longitudinal.training.loop import train_step, val_step


class _FakeModel(nn.Module):
    """Pools t0 to a scalar, concatenates clinical features, predicts risk."""

    def __init__(self, clinical_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(1 + clinical_dim, 1)

    def forward(self, x0, x1, clinical_features=None, **_ignored):
        pooled = x0.mean(dim=(1, 2, 3, 4)).unsqueeze(-1)  # (B, 1)
        z = torch.cat([pooled, clinical_features], dim=-1) if clinical_features is not None else pooled
        return {"risk": self.fc(z).squeeze(-1), "embedding": z}


@pytest.fixture
def cox_ready_dataset(tmp_path):
    """Like conftest's synthetic_dataset, but with durations chosen so the Cox
    gradient isn't trivially zero. The event patient must not have the
    longest duration, or its risk set is just itself and the loss is flat."""
    root = tmp_path / "nifti"
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    phase = "venous"
    pids = ["PAT-001", "PAT-002"]
    seg = synthetic_seg()
    rows = []
    for i, (pid, dur, status) in enumerate(zip(pids, [10.0, 11.0], [1, 0])):
        for tp, tag in [("t0", "T0"), ("t1", "T1")]:
            d = root / phase / pid / tp
            d.mkdir(parents=True)
            _save_nii(synthetic_ct(seed=i * 2 + (tp == "t1")), d / f"{pid}_{tp}.nii.gz")
            _save_nii(seg, cache / f"{pid}_seg_{tag}.nii.gz", dtype=np.int16)
        rows.append(dict(patient_id=pid, cohort="cohort_a", time_months=dur, status=status,
                          age_diag=60 + i, bmi=25.0 + i))
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    return {"nifti_root": root, "cache_dir": cache, "labels_csv": labels,
            "pids": pids, "phase": phase}


def _make_loader(dataset, batch_size=2):
    registry = ClinicalRegistry(dataset["labels_csv"])
    ds = LongitudinalCTDataset(
        nifti_root=dataset["nifti_root"], registry=registry,
        cache_dir=dataset["cache_dir"], weights_path=None,
        phase=dataset["phase"],
        allowed_regions=["abdomen"], post_nat_tps=["t1"],
        patch_size=(32, 32, 32), target_spacing_mm=(1.5, 1.5, 1.5),
        foreground_margin_voxels=4, reuse_saved_segs=True,
    )
    clinical_dim = ds[0]["clinical"].shape[-1]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, clinical_dim


def test_train_step_produces_finite_loss_and_updates_weights(cox_ready_dataset):
    loader, clinical_dim = _make_loader(cox_ready_dataset)
    model = _FakeModel(clinical_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = CoxPHLoss()
    before = model.fc.weight.detach().clone()

    batch = next(iter(loader))
    metrics, risk, dur, evt = train_step(
        model, batch, criterion, optimizer, torch.device("cpu"),
        scaler=None, clinical_dim=clinical_dim,
    )

    assert torch.isfinite(torch.tensor(metrics["total"]))
    assert risk.shape == dur.shape == evt.shape
    assert not torch.equal(before, model.fc.weight)  # optimizer.step() actually moved weights


def test_val_step_does_not_update_weights(cox_ready_dataset):
    loader, clinical_dim = _make_loader(cox_ready_dataset)
    model = _FakeModel(clinical_dim)
    criterion = CoxPHLoss()
    before = model.fc.weight.detach().clone()

    batch = next(iter(loader))
    metrics, risk, dur, evt, pids = val_step(
        model, batch, criterion, torch.device("cpu"), clinical_dim=clinical_dim,
    )

    assert torch.isfinite(torch.tensor(metrics["total"]))
    assert torch.equal(before, model.fc.weight)  # no gradient step in eval
    assert set(pids) == {"PAT-001", "PAT-002"}
