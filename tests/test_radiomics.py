"""Unit tests for the radiomic feature schema and fold-internal scaler."""

from __future__ import annotations

import numpy as np

from pdac_longitudinal.radiomics.feature_schema import (
    RADIOMIC_FEATURE_DIM,
    RadiomicScaler,
    decode_radiomic_features,
    radiomic_dict_to_vector,
    signed_log,
)


def test_signed_log_preserves_sign_and_zero():
    x = np.array([-10.0, 0.0, 10.0])
    y = signed_log(x)
    assert y[1] == 0.0
    assert y[0] < 0.0 < y[2]
    assert np.isclose(abs(y[0]), abs(y[2]))  # symmetric for +/-x


def test_radiomic_dict_to_vector_fixed_dim_and_delta():
    vec = radiomic_dict_to_vector({})
    assert vec.shape == (RADIOMIC_FEATURE_DIM,)
    assert np.all(vec == 0.0)


def test_decode_radiomic_features_roundtrip():
    import json
    feats = {"T0_foo": 1.5}
    buf = json.dumps(feats).encode("utf-8")
    assert decode_radiomic_features({"radiomic_features_json": buf}) == feats
    assert decode_radiomic_features({}) == {}


def test_scaler_zscores_training_data_to_unit_stats():
    rng = np.random.default_rng(0)
    raw = rng.normal(5.0, 2.0, size=(50, RADIOMIC_FEATURE_DIM)).astype(np.float32)
    scaler = RadiomicScaler().fit(raw)
    z = scaler.transform(raw)
    assert z.shape == raw.shape
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-4)
    assert scaler.out_dim == RADIOMIC_FEATURE_DIM


def test_scaler_pca_reduces_dim_and_is_deterministic():
    rng = np.random.default_rng(1)
    raw = rng.normal(0.0, 1.0, size=(20, RADIOMIC_FEATURE_DIM)).astype(np.float32)
    scaler = RadiomicScaler().fit(raw, n_components=5)
    assert scaler.out_dim == 5
    a = scaler.transform(raw)
    b = scaler.transform(raw)
    assert a.shape == (20, 5)
    assert np.array_equal(a, b)


def test_scaler_transform_handles_nan_inf():
    rng = np.random.default_rng(2)
    raw = rng.normal(0.0, 1.0, size=(10, RADIOMIC_FEATURE_DIM)).astype(np.float32)
    scaler = RadiomicScaler().fit(raw)
    bad = raw[0].copy()
    bad[0] = np.nan
    bad[1] = np.inf
    z = scaler.transform(bad)
    assert np.all(np.isfinite(z))
