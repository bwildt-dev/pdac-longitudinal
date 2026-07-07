"""Unit tests for the concordance and horizon-AUC metrics."""

from __future__ import annotations

import numpy as np

from pdac_longitudinal.training.metrics import concordance_index, horizon_auc


def test_concordance_perfect_and_reversed():
    dur = np.array([1.0, 2.0, 3.0, 4.0])
    evt = np.array([1, 1, 1, 1])
    risk = np.array([4.0, 3.0, 2.0, 1.0])       # higher risk dies earlier
    assert concordance_index(risk, dur, evt) == 1.0
    assert concordance_index(-risk, dur, evt) == 0.0


def test_concordance_nan_without_events():
    val = concordance_index(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0, 0]))
    assert np.isnan(val)


def test_horizon_auc_separable_case():
    risk = np.array([2.0, 2.0, -2.0, -2.0])
    dur  = np.array([6.0, 6.0, 24.0, 24.0])
    evt  = np.array([1, 1, 0, 0])
    assert horizon_auc(risk, dur, evt, horizon=12.0) == 1.0
