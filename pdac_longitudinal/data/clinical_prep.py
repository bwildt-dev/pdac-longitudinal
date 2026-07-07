"""Shared clinical-feature preprocessing."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MISSING_SUFFIX = "__isna"

# Non-feature columns in labels.csv; every other column is treated as a
# clinical covariate.
RESERVED_COLS = {"patient_id", "cohort", "time_months", "status"}


def drop_collinear(
    df: pd.DataFrame, feature_cols: Sequence[str], thresh: float = 0.999,
) -> Tuple[List[str], List[str]]:
    """Drop features that are near-exact linear duplicates of an earlier-kept one.

    Args:
        df: Source dataframe containing `feature_cols`.
        feature_cols: Candidate columns, in priority order.
        thresh: Correlation threshold above which a column is dropped.

    Returns:
        A `(kept, dropped)` tuple of column names.
    """
    kept: List[str] = []
    dropped: List[Tuple[str, str, float]] = []
    for col in feature_cols:
        redundant_with: Optional[Tuple[str, float]] = None
        for k in kept:
            s = df[[col, k]].dropna()
            if len(s) < 3:
                continue
            a, b = s[col].to_numpy(dtype=float), s[k].to_numpy(dtype=float)
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            r = abs(float(np.corrcoef(a, b)[0, 1]))
            if r >= thresh:
                redundant_with = (k, r)
                break
        if redundant_with is None:
            kept.append(col)
        else:
            dropped.append((col, redundant_with[0], redundant_with[1]))
    if dropped:
        logger.info(
            "Collinearity guard dropped %d feature(s): %s",
            len(dropped),
            ", ".join(f"{c} (|r|={r:.3f} with {k})" for c, k, r in dropped),
        )
    return kept, [d[0] for d in dropped]


def add_missingness_flags(
    df: pd.DataFrame, feature_cols: Sequence[str], enabled: bool = False,
) -> List[str]:
    """Append a 0/1 `<col>__isna` indicator for every feature with any NaN.

    Args:
        df: Dataframe to mutate in place with the new indicator columns.
        feature_cols: Feature columns to check for missingness.
        enabled: If `False`, a no-op (no measured benefit by default; enable
            for cohorts with informative missingness).

    Returns:
        Names of the indicator columns added; empty if `enabled` is `False`.
    """
    if not enabled:
        return []
    flags: List[str] = []
    for col in feature_cols:
        if df[col].isna().any():
            flag = f"{col}{MISSING_SUFFIX}"
            df[flag] = df[col].isna().astype("float64")
            flags.append(flag)
    if flags:
        logger.info("Added %d missingness indicator(s): %s", len(flags), flags)
    return flags


class FoldStats:
    """Fit imputation medians + z-score (mean/std) on a chosen id subset.

    Attributes:
        medians: Per-feature imputation medians, set by `fit`.
        mean: Per-feature mean of the imputed data, set by `fit`.
        std: Per-feature std of the imputed data, set by `fit`.
        completeness: Per-feature observed-fraction on the train fold
            (1.0 = fully observed).
    """

    def __init__(self) -> None:
        self.medians: Optional[pd.Series] = None
        self.mean: Optional[pd.Series] = None
        self.std: Optional[pd.Series] = None
        self.completeness: Optional[pd.Series] = None

    def fit(self, raw: pd.DataFrame, ids: Optional[Sequence[str]] = None) -> "FoldStats":
        """Fit imputation medians and z-score stats on the rows identified by `ids`.

        Args:
            raw: Raw feature dataframe, indexed by patient id.
            ids: Subset of ids to fit on; falls back to all rows if `None`
                or none are present.

        Returns:
            `self`, for chaining.
        """
        sub = raw if ids is None else raw.loc[[p for p in ids if p in raw.index]]
        if len(sub) == 0:
            sub = raw
        self.completeness = sub.notna().mean().clip(lower=1e-2)
        self.medians = sub.median().fillna(0.0)
        imputed = sub.fillna(self.medians)
        self.mean = imputed.mean().fillna(0.0)
        self.std = imputed.std().fillna(1.0).replace(0, 1)
        return self

    def impute(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Impute `raw` with the fitted train medians.

        Args:
            raw: Feature dataframe to impute; need not match the fit id set.

        Raises:
            AssertionError: If `fit` has not been called yet.
        """
        assert self.medians is not None, "FoldStats.fit() must be called first"
        return raw.fillna(self.medians)
