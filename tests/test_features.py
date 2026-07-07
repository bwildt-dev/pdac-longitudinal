"""Unit tests for the fixed-length feature-vector contract (model input dims)."""

from __future__ import annotations

import numpy as np

from pdac_longitudinal.preprocess.anatomy_features import (
    ANATOMY_FEATURE_COLS,
    ANATOMY_FEATURE_DIM,
    features_dict_to_vector as anatomy_to_vector,
)
from pdac_longitudinal.preprocess.vessel_features import (
    VESSEL_FEATURE_COLS,
    VESSEL_FEATURE_DIM,
    features_dict_to_vector as vessel_to_vector,
)


def test_anatomy_vector_has_fixed_dim():
    vec = anatomy_to_vector({})
    assert vec.shape == (ANATOMY_FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.all(vec == 0.0)                       # missing -> fill=0.0
    assert ANATOMY_FEATURE_DIM == len(ANATOMY_FEATURE_COLS)


def test_anatomy_vector_reads_named_columns():
    col = ANATOMY_FEATURE_COLS[0]
    vec = anatomy_to_vector({col: 3.5})
    assert vec[0] == np.float32(3.5)


def test_vessel_vector_has_fixed_dim():
    vec = vessel_to_vector({})
    assert vec.shape == (VESSEL_FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert VESSEL_FEATURE_DIM == len(VESSEL_FEATURE_COLS)
