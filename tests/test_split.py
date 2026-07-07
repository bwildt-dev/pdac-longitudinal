"""Unit tests for train/val/test and K-fold splitting (determinism + no leakage)."""

from __future__ import annotations

import pandas as pd

from pdac_longitudinal.data.registry import ClinicalRegistry
from pdac_longitudinal.data.split import make_split, stratified_kfold_ids


def _registry(tmp_path):
    rows = [
        {"patient_id": f"p{i}", "cohort": "c", "time_months": 5.0 + i,
         "status": i % 2, "age_diag": 60 + i}
        for i in range(40)
    ]
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    return ClinicalRegistry(labels)


def test_make_split_is_deterministic_and_disjoint(tmp_path):
    reg = _registry(tmp_path)
    ids = reg.all_ids()
    tr1, va1, te1 = make_split(ids, reg, seed=7)
    tr2, va2, te2 = make_split(ids, reg, seed=7)
    assert (tr1, va1, te1) == (tr2, va2, te2)                 # deterministic
    assert set(tr1) | set(va1) | set(te1) == set(ids)         # covers everyone
    assert not (set(tr1) & set(va1))                          # disjoint
    assert not (set(tr1) & set(te1))
    assert not (set(va1) & set(te1))


def test_make_split_seed_changes_partition(tmp_path):
    reg = _registry(tmp_path)
    ids = reg.all_ids()
    a = make_split(ids, reg, seed=1)
    b = make_split(ids, reg, seed=2)
    assert a != b


def test_kfold_folds_partition_val_without_overlap():
    ids = [f"p{i}" for i in range(30)]
    events = [i % 2 for i in range(30)]
    folds = stratified_kfold_ids(ids, events, n_folds=5, seed=0)
    assert len(folds) == 5
    val_union = set()
    for train_ids, val_ids in folds:
        assert not (set(train_ids) & set(val_ids))           # no leakage in a fold
        assert set(train_ids) | set(val_ids) == set(ids)     # fold covers all ids
        val_union |= set(val_ids)
    assert val_union == set(ids)                             # every id validated once
    assert stratified_kfold_ids(ids, events, 5, 0) == folds  # deterministic
