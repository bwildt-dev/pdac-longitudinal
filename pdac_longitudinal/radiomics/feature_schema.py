"""Canonical radiomic feature schema for the model (T0 + T1 + Δ)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_COLS_PATH = Path(__file__).with_name("radiomic_feature_cols.json")


def _load_base_cols() -> List[str]:
    return list(json.loads(_COLS_PATH.read_text()))


_BASE_COLS: List[str] = _load_base_cols()
T0_COLS: Tuple[str, ...] = tuple(sorted(c for c in _BASE_COLS if c.startswith("T0_")))
T1_COLS: Tuple[str, ...] = tuple(sorted(c for c in _BASE_COLS if c.startswith("T1_")))

# stems common to both timepoints become Δ features
_DELTA_STEMS: Tuple[str, ...] = tuple(
    sorted({c[3:] for c in T0_COLS} & {c[3:] for c in T1_COLS})
)
DELTA_COLS: Tuple[str, ...] = tuple(f"D_{stem}" for stem in _DELTA_STEMS)

# Canonical model layout + dim.
RADIOMIC_FEATURE_COLS: Tuple[str, ...] = T0_COLS + T1_COLS + DELTA_COLS
RADIOMIC_FEATURE_DIM: int = len(RADIOMIC_FEATURE_COLS)   # 2247


def decode_radiomic_features(arrays: Dict[str, Any]) -> Dict[str, float]:
    """Decode the JSON-encoded radiomic features stashed in a cache `.npz`.

    Args:
        arrays: Loaded `.npz` mapping; must contain `radiomic_features_json`
            or the result is empty.

    Returns:
        Dict mapping feature name to value.
    """
    buf = arrays.get("radiomic_features_json")
    if buf is None:
        return {}
    try:
        return json.loads(bytes(buf).decode("utf-8"))
    except Exception:
        return {}


def _finite(x: Any) -> float:
    """float(x) with NaN/Inf/None mapped to 0.0."""
    if x is None:
        return 0.0
    v = float(x)
    return v if np.isfinite(v) else 0.0


def radiomic_dict_to_vector(feats: Dict[str, float]) -> np.ndarray:
    """Pack a `{T0_*, T1_*}` feature dict into the canonical raw feature vector.

    Missing or non-finite values map to 0.0; Δ entries are `T1 - T0` and are
    only set when both timepoints are present and finite.

    Args:
        feats: Dict of `T0_*`/`T1_*` feature name to value.

    Returns:
        `float32` vector of shape `(RADIOMIC_FEATURE_DIM,)`, laid out as
        `[T0 | T1 | Δ]` per `RADIOMIC_FEATURE_COLS`.
    """
    vec = np.zeros(RADIOMIC_FEATURE_DIM, dtype=np.float32)
    n0, n1 = len(T0_COLS), len(T1_COLS)
    for i, col in enumerate(T0_COLS):
        vec[i] = _finite(feats.get(col))
    for j, col in enumerate(T1_COLS):
        vec[n0 + j] = _finite(feats.get(col))
    for k, stem in enumerate(_DELTA_STEMS):
        t0, t1 = feats.get(f"T0_{stem}"), feats.get(f"T1_{stem}")
        if t0 is not None and t1 is not None and np.isfinite(t0) and np.isfinite(t1):
            vec[n0 + n1 + k] = np.float32(t1) - np.float32(t0)
    return vec


def signed_log(x: np.ndarray) -> np.ndarray:
    """`sign(x)·log1p(|x|)`; monotonic, sign-preserving range compression."""
    return np.sign(x) * np.log1p(np.abs(x))


class RadiomicScaler:
    """Fold-internal radiomic normaliser: signed-log -> z-score -> optional PCA.

    Attributes:
        mean: Per-feature mean of the signed-log-transformed training data.
        std: Per-feature std of the signed-log-transformed training data,
            floored at `1e-6`.
        pca_mean: PCA mean vector over the z-scored features (set when
            `n_components > 0`).
        pca_comp: PCA component matrix, shape `(k, RADIOMIC_FEATURE_DIM)`.
    """

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.pca_mean: Optional[np.ndarray] = None
        self.pca_comp: Optional[np.ndarray] = None

    @property
    def out_dim(self) -> int:
        """Return the output feature dimension."""
        return int(self.pca_comp.shape[0]) if self.pca_comp is not None else RADIOMIC_FEATURE_DIM

    def fit(self, raw_matrix: np.ndarray, n_components: int = 0) -> "RadiomicScaler":
        """Fit signed-log z-score stats and optional PCA on `raw_matrix`.

        Args:
            raw_matrix: Raw feature matrix, shape `(n_samples, RADIOMIC_FEATURE_DIM)`.
            n_components: Number of PCA components to fit; 0 disables PCA.

        Returns:
            self, for chaining.
        """
        logm = signed_log(np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0))
        self.mean = logm.mean(axis=0).astype(np.float32)
        self.std = np.maximum(logm.std(axis=0), 1e-6).astype(np.float32)
        if n_components and n_components > 0:
            from sklearn.decomposition import PCA
            Z = (logm - self.mean) / self.std
            k = int(min(n_components, Z.shape[0] - 1, Z.shape[1]))
            pca = PCA(n_components=k, svd_solver="auto", random_state=0).fit(Z)
            self.pca_mean = pca.mean_.astype(np.float32)
            self.pca_comp = pca.components_.astype(np.float32)
        return self

    def transform(self, raw_vec: np.ndarray) -> np.ndarray:
        """Apply signed-log, z-score, and optional PCA to a raw feature vector.

        Args:
            raw_vec: Raw feature vector(s), shape `(..., RADIOMIC_FEATURE_DIM)`.

        Returns:
            `float32` array, PCA-reduced to `(..., k)` when PCA was fit,
            otherwise same shape as `raw_vec`.
        """
        z = signed_log(np.nan_to_num(raw_vec, nan=0.0, posinf=0.0, neginf=0.0))
        if self.mean is not None and self.std is not None:
            z = (z - self.mean) / self.std
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if self.pca_comp is not None:
            z = (z - self.pca_mean) @ self.pca_comp.T
        return z.astype(np.float32)
