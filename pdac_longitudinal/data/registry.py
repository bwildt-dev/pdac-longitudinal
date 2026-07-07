"""Clinical registry; loads labels.csv and serves per-patient survival + covariates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from pdac_longitudinal.data.clinical_prep import (
    RESERVED_COLS,
    FoldStats,
    add_missingness_flags,
    drop_collinear,
)

logger = logging.getLogger(__name__)


class ClinicalRegistry:
    """Loads labels.csv and provides fast lookup by patient_id.

    Args:
        labels_csv: Path to the labels CSV (patient_id, survival, and
            clinical covariate columns).
        min_cohort_coverage: Minimum per-cohort non-null fraction required
            to keep a feature.
        include_cohorts: Cohorts to restrict the coverage filter to; empty
            means all cohorts.
        completeness_weighting: Whether to scale each z-scored clinical
            feature by its train-fold observed fraction.
        missingness_flags: Whether to add a `<col>__isna` indicator feature
            for every clinical column with missing values.
    """

    def __init__(
        self,
        labels_csv: Path,
        min_cohort_coverage: float = 0.0,
        include_cohorts: Sequence[str] = (),
        completeness_weighting: bool = False,
        missingness_flags: bool = False,
    ) -> None:
        df = pd.read_csv(labels_csv)

        # Drop ca19_9_post_nat_log: redundant with ca19_9_log + delta_ca19_9_log.
        _REDUNDANT_FEATURES = ["ca19_9_post_nat_log"]
        df = df.drop(columns=[c for c in _REDUNDANT_FEATURES if c in df.columns])

        # Per-cohort coverage filter, before one-hot encoding (categoricals
        # stay string columns).
        dropped_by_cohort: List[Tuple[str, Dict[str, float]]] = []
        if "cohort" in df.columns:
            candidate_cols = [c for c in df.columns if c not in RESERVED_COLS]
            cov_df = df
            if include_cohorts:
                wanted = {c.lower() for c in include_cohorts}
                cov_df = df[df["cohort"].astype(str).str.lower().isin(wanted)]
                logger.info(
                    "Coverage filter restricted to cohorts %s (%d/%d rows).",
                    sorted(include_cohorts), len(cov_df), len(df),
                )
            cohort_groups = cov_df.groupby("cohort")
            for col in candidate_cols:
                cov = cohort_groups[col].apply(lambda s: s.notna().mean())
                if (cov <= min_cohort_coverage).any():
                    dropped_by_cohort.append((col, cov.to_dict()))
            drop_names = {c for c, _ in dropped_by_cohort}
            if drop_names:
                df = df.drop(columns=list(drop_names))
                logger.info(
                    "Dropped %d clinical features for insufficient per-cohort "
                    "coverage (threshold=%.2f):", len(drop_names),
                    min_cohort_coverage,
                )
                for col, cov in dropped_by_cohort:
                    cov_str = ", ".join(f"{k}={v:.0%}" for k, v in cov.items())
                    logger.info("  %-26s coverage: %s", col, cov_str)

        for col, prefix in [
            ("nat_regimen", "nat_regimen"),
            ("resectability", "resectability"),
        ]:
            if col in df.columns:
                dummies = pd.get_dummies(
                    df[col], prefix=prefix, drop_first=True, dummy_na=False
                )
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        self.df = df.set_index("patient_id")
        feature_cols = [c for c in self.df.columns if c not in RESERVED_COLS]
        feature_cols, _ = drop_collinear(self.df, feature_cols)
        # Missingness indicators, added before imputation so a median-fill
        # can't masquerade as observed.
        flag_cols = add_missingness_flags(self.df, feature_cols, enabled=missingness_flags)
        self.clinical_cols = feature_cols + flag_cols
        self.clinical_dim = len(self.clinical_cols)

        all_nan = self.df[feature_cols].isna().all()
        if all_nan.any():
            logger.warning(
                "Columns entirely NaN (filled with 0): %s",
                all_nan[all_nan].index.tolist(),
            )
        # Raw (pre-impute) feature matrix, so stats can be refit per fold
        # via `fit(train_ids)`.
        self._raw = self.df[self.clinical_cols].copy()
        self._stats = FoldStats()
        self.completeness_weighting = bool(completeness_weighting)
        self._completeness = np.ones(len(self.clinical_cols), dtype=np.float32)
        self.fit(None)

        n_events = int(self.df["status"].sum())
        logger.info(
            "ClinicalRegistry: %d patients | %d features | events=%d censored=%d",
            len(self.df),
            self.clinical_dim,
            n_events,
            len(self.df) - n_events,
        )

    # Lookup helpers

    def fit(self, train_ids: Optional[Sequence[str]] = None) -> "ClinicalRegistry":
        """(Re)compute imputation medians + z-score stats on `train_ids` only.

        Args:
            train_ids: Patient ids to fit the imputation/z-score stats on;
                `None` fits on all rows.

        Returns:
            `self`, for chaining.
        """
        self._stats.fit(self._raw, train_ids)
        self.df.loc[:, self.clinical_cols] = self._stats.impute(self._raw)
        self._mean = self._stats.mean
        self._std = self._stats.std
        if self.completeness_weighting and self._stats.completeness is not None:
            self._completeness = (
                self._stats.completeness.reindex(self.clinical_cols)
                .fillna(1.0).values.astype(np.float32)
            )
        return self

    def has(self, patient_id: str) -> bool:
        """Return True if `patient_id` is present in the registry.

        Args:
            patient_id: Patient id to look up.
        """
        return patient_id in self.df.index

    def get_survival(self, patient_id: str) -> Tuple[float, int]:
        """Return `(time_months, status)` for `patient_id`.

        Args:
            patient_id: Patient id to look up.
        """
        row = self.df.loc[patient_id]
        return float(row["time_months"]), int(row["status"])

    def get_cohort(self, patient_id: str) -> str:
        """Return the cohort label for `patient_id`.

        Args:
            patient_id: Patient id to look up.

        Returns:
            `"unknown"` if the cohort column is absent, the id is missing,
            or the value is NaN.
        """
        if "cohort" not in self.df.columns:
            return "unknown"
        try:
            val = self.df.loc[patient_id, "cohort"]
        except KeyError:
            return "unknown"
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "unknown"
        return str(val)

    def get_clinical_tensor(self, patient_id: str) -> torch.Tensor:
        """Return a z-scored float32 tensor of clinical covariates.

        Args:
            patient_id: Patient id to look up.
        """
        if not self.clinical_cols:
            return torch.zeros(0)
        row = (
            self.df.loc[patient_id][self.clinical_cols]
            .values.astype(np.float32)
        )
        row = (row - self._mean.values) / self._std.values
        if self.completeness_weighting:
            row = row * self._completeness
        return torch.from_numpy(row)

    def all_ids(self) -> List[str]:
        """Return all patient IDs in the registry."""
        return list(self.df.index)
