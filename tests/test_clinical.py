"""Unit tests for clinical preprocessing: leakage guard, collinearity, schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdac_longitudinal.data.clinical_prep import (
    RESERVED_COLS,
    FoldStats,
    add_missingness_flags,
    drop_collinear,
)
from pdac_longitudinal.data.registry import ClinicalRegistry


def test_foldstats_uses_train_rows_only():
    # p4's extreme value must NOT influence stats fit on p1..p3.
    raw = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]},
                       index=["p1", "p2", "p3", "p4"])
    fs = FoldStats().fit(raw, ["p1", "p2", "p3"])
    assert fs.medians["a"] == 2.0
    assert fs.mean["a"] == 2.0
    imputed = fs.impute(raw)
    assert imputed.loc["p4", "a"] == 2.0        # test row filled with train median


def test_drop_collinear_removes_duplicate():
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x_copy": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [5.0, 1.0, 4.0, 2.0, 3.0],
    })
    kept, dropped = drop_collinear(df, ["x", "x_copy", "y"])
    assert "x" in kept and "y" in kept
    assert dropped == ["x_copy"]


def test_missingness_flags_toggle():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert add_missingness_flags(df, ["a"], enabled=False) == []
    flags = add_missingness_flags(df, ["a"], enabled=True)
    assert flags == ["a__isna"]
    assert df["a__isna"].tolist() == [0.0, 1.0, 0.0]


def test_registry_derives_features_from_reserved(tmp_path):
    labels = tmp_path / "labels.csv"
    pd.DataFrame([
        {"patient_id": "p1", "cohort": "c", "time_months": 5.0, "status": 1,
         "age_diag": 60, "bmi": 25.0},
        {"patient_id": "p2", "cohort": "c", "time_months": 9.0, "status": 0,
         "age_diag": 70, "bmi": 28.0},
    ]).to_csv(labels, index=False)
    reg = ClinicalRegistry(labels)
    assert set(reg.clinical_cols) == {"age_diag", "bmi"}     # reserved cols excluded
    assert not (set(reg.clinical_cols) & RESERVED_COLS)
    assert reg.get_clinical_tensor("p1").shape[0] == reg.clinical_dim
    assert reg.get_survival("p1") == (5.0, 1)
