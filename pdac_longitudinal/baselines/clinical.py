"""Clinical-only Cox baseline (lifelines CoxPH, reusable core)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pdac_longitudinal.data.clinical_prep import (
    RESERVED_COLS,
    FoldStats,
    add_missingness_flags,
    drop_collinear,
)
from pdac_longitudinal.training.metrics import horizon_auc

logger = logging.getLogger(__name__)


# Data

class ClinicalRegistry:
    """Wrap clinical labels CSV with fold-internal median-impute and z-score.

    Args:
        labels_csv: Path to the labels CSV (must have `patient_id`,
            `time_months`, `status`, plus clinical columns).
        min_cohort_coverage: Drop a feature if, in any cohort, its non-missing
            fraction is `<=` this value.
        include_cohorts: If given, restrict the coverage computation to these
            cohorts (case-insensitive).
        missingness_flags: Whether to add a `<col>__isna` indicator feature
            for every clinical column with missing values.
    """

    def __init__(
        self,
        labels_csv: Union[str, Path],
        min_cohort_coverage: float = 0.0,
        include_cohorts: Sequence[str] = (),
        missingness_flags: bool = False,
    ):
        df = pd.read_csv(labels_csv)
        if "cohort" in df.columns:
            candidate_cols = [c for c in df.columns if c not in RESERVED_COLS]
            cov_df = df
            if include_cohorts:
                wanted = {c.lower() for c in include_cohorts}
                cov_df = df[df["cohort"].astype(str).str.lower().isin(wanted)]
            groups = cov_df.groupby("cohort")
            drop = [c for c in candidate_cols
                    if bool((groups[c].apply(lambda s: s.notna().mean())
                             <= min_cohort_coverage).any())]
            if drop:
                df = df.drop(columns=drop)
                logger.info("Clinical baseline: dropped %d features for per-cohort "
                            "coverage<=%.2f (matched to imaging registry): %s",
                            len(drop), min_cohort_coverage, drop)
        for col, prefix in [("nat_regimen", "nat_regimen"),
                             ("resectability", "resectability")]:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=prefix,
                                         drop_first=True, dummy_na=False)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        self.df = df.set_index("patient_id")
        feature_cols = [c for c in self.df.columns if c not in RESERVED_COLS]
        if not feature_cols:
            raise ValueError("No clinical columns found in labels.csv")
        feature_cols, _ = drop_collinear(self.df, feature_cols)
        flag_cols = add_missingness_flags(self.df, feature_cols, enabled=missingness_flags)
        self.cols = feature_cols + flag_cols

        self._raw = self.df[self.cols].copy()
        self._stats = FoldStats()
        self.fit(None)
        logger.info("ClinicalRegistry: %d patients × %d features",
                    len(self.df), len(self.cols))

    def fit(self, train_ids=None):
        """(Re)fit imputation medians + z-score stats on `train_ids` only.

        Args:
            train_ids: Patient IDs to fit stats on; `None` fits over all rows.

        Returns:
            self, for chaining.
        """
        self._stats.fit(self._raw, train_ids)
        self.df.loc[:, self.cols] = self._stats.impute(self._raw)
        self._mean = self._stats.mean
        self._std = self._stats.std
        return self

    def has(self, pid: str) -> bool:
        """Return True if `pid` has a row in the registry."""
        return pid in self.df.index

    def get_features(self, pid: str) -> np.ndarray:
        """Return the z-scored clinical feature vector for `pid`."""
        row = self.df.loc[pid][self.cols].values.astype(np.float32)
        return ((row - self._mean.values) / self._std.values).astype(np.float32)

    def get_survival(self, pid: str) -> Tuple[float, int]:
        """Return `(time_months, event)` for `pid`."""
        row = self.df.loc[pid]
        return float(row["time_months"]), int(row["status"])

    def frame(self, ids: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Return a z-scored `(DataFrame[cols + duration + event], kept_ids)`.

        Args:
            ids: Patient IDs to include; any not present in the registry are
                dropped.

        Returns:
            Tuple of the assembled `DataFrame` and the list of IDs actually
            kept, in the same row order.
        """
        kept = [p for p in ids if self.has(p)]
        dropped = set(ids) - set(kept)
        if dropped:
            logger.warning("Dropped %d IDs not in registry: %s",
                           len(dropped), sorted(dropped)[:10])
        if not kept:
            return pd.DataFrame(columns=self.cols + ["duration", "event"]), []
        X = np.stack([self.get_features(p) for p in kept])
        surv = [self.get_survival(p) for p in kept]
        out = pd.DataFrame(X, columns=self.cols, index=kept)
        out["duration"] = [s[0] for s in surv]
        out["event"] = [s[1] for s in surv]
        return out, kept


def stratified_kfold(
    ids: List[str], events: List[int], n_folds: int, seed: int,
) -> List[Tuple[List[str], List[str]]]:
    """Stratified k-fold by event status. Deterministic given the inputs.

    Args:
        ids: Patient IDs to split.
        events: Event indicator (1=event, 0=censored) aligned with `ids`.
        n_folds: Number of folds.
        seed: Seed for the shuffle.

    Returns:
        List of `(train_ids, val_ids)` pairs, one per fold, each sorted.
    """
    import random as _rnd
    rng = _rnd.Random(seed)

    paired = list(zip(ids, events))
    pos = [pid for pid, e in paired if e == 1]
    neg = [pid for pid, e in paired if e == 0]
    rng.shuffle(pos); rng.shuffle(neg)

    def _chunks(seq, k):
        out = [[] for _ in range(k)]
        for i, x in enumerate(seq):
            out[i % k].append(x)
        return out

    pos_folds = _chunks(pos, n_folds)
    neg_folds = _chunks(neg, n_folds)
    folds = []
    for k in range(n_folds):
        val_ids = sorted(pos_folds[k] + neg_folds[k])
        train_ids = sorted(
            [p for j in range(n_folds) if j != k for p in pos_folds[j] + neg_folds[j]]
        )
        folds.append((train_ids, val_ids))
    return folds


# C-index

def concordance_index(risk: np.ndarray, dur: np.ndarray, evt: np.ndarray) -> float:
    """Concordance for a *risk* score (higher = worse prognosis).

    Args:
        risk: Risk scores; non-finite entries are dropped.
        dur: Observed time-to-event or censoring, aligned with `risk`.
        evt: Event indicator (1=event, 0=censored), aligned with `risk`.

    Returns:
        C-index, or `nan` if fewer than 2 valid samples or no events.
    """
    from lifelines.utils import concordance_index as ci
    risk = np.asarray(risk, dtype=float)
    dur = np.asarray(dur, dtype=float)
    evt = np.asarray(evt, dtype=float)
    valid = np.isfinite(risk)
    if not valid.all():
        risk, dur, evt = risk[valid], dur[valid], evt[valid]
    if len(risk) < 2 or evt.sum() == 0:
        return float("nan")
    return float(ci(dur, -risk, evt))


# High-level inline entry point

def train_clinical_baseline(
    labels_csv: Union[str, Path],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Optional[Sequence[str]] = None,
    *,
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
    horizon: float = 12.0,
    min_cohort_coverage: float = 0.0,
    include_cohorts: Sequence[str] = (),
    missingness_flags: bool = False,
    seed: int = 42,
    device: Optional[Any] = None,
    use_wandb: bool = False,
    log_prefix: str = "[clinical-baseline] ",
    **_ignored: Any,
) -> Dict[str, Any]:
    """Fit a penalised Cox PH model on `train_ids` and score `val_ids`.

    Args:
        labels_csv: Path to the labels CSV; passed through to `ClinicalRegistry`.
        train_ids: Patient IDs to fit the Cox model on.
        val_ids: Patient IDs to score.
        test_ids: Optional held-out patient IDs to also score.
        penalizer: Lifelines `CoxPHFitter` L2/elastic-net penalty strength.
        l1_ratio: Lifelines elastic-net mixing parameter (0=ridge, 1=lasso).
        horizon: Time horizon (months) for the horizon-AUC metric.
        min_cohort_coverage: Per-cohort feature coverage filter, passed to
            `ClinicalRegistry`.
        include_cohorts: Cohorts to restrict the coverage computation to,
            passed to `ClinicalRegistry`.
        missingness_flags: Whether to add `<col>__isna` indicator features,
            passed to `ClinicalRegistry`.
        seed: Accepted for call-site compatibility; unused.
        device: Accepted for call-site compatibility; unused.
        use_wandb: Accepted for call-site compatibility; unused.
        log_prefix: Prefix prepended to log messages.
        **_ignored: Extra kwargs accepted for call-site compatibility.

    Returns:
        Dict with `best_val_c`, `best_train_c`, `best_val_auc`,
        `best_test_c`/`best_test_auc` (NaN if no `test_ids`), `n_train`/`n_val`/
        `n_test`, `features`, `horizon_months`, `penalizer`, and
        `val_predictions`. Returns `{}` if train or val is empty after
        registry filtering.
    """
    from lifelines import CoxPHFitter

    registry = ClinicalRegistry(
        labels_csv,
        min_cohort_coverage=min_cohort_coverage,
        include_cohorts=include_cohorts,
        missingness_flags=missingness_flags,
    )
    registry.fit(train_ids)
    df_tr, ids_tr = registry.frame(train_ids)
    df_va, ids_va = registry.frame(val_ids)
    df_te, ids_te = registry.frame(test_ids or [])

    if len(ids_tr) == 0 or len(ids_va) == 0:
        logger.warning(
            "%sEmpty train/val after registry filter (train=%d val=%d); "
            "skipping clinical baseline.", log_prefix, len(ids_tr), len(ids_va))
        return {}

    # Drop near-constant feature columns *in the training fold*
    feat_cols = list(registry.cols)
    keep_cols = [c for c in feat_cols if df_tr[c].std() > 1e-8]
    dropped = [c for c in feat_cols if c not in keep_cols]
    if dropped:
        logger.info("%sdropped %d near-constant cols in this fold: %s",
                    log_prefix, len(dropped), dropped)

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df_tr[keep_cols + ["duration", "event"]],
            duration_col="duration", event_col="event",
            show_progress=False)

    def _risk(df: pd.DataFrame) -> np.ndarray:
        # Partial hazard: higher = more hazard = worse prognosis.
        return cph.predict_partial_hazard(df[keep_cols]).values.astype(float).ravel()

    r_tr = _risk(df_tr)
    r_va = _risk(df_va)
    d_va, e_va = df_va["duration"].values, df_va["event"].values
    d_tr, e_tr = df_tr["duration"].values, df_tr["event"].values

    val_c = concordance_index(r_va, d_va, e_va)
    train_c = concordance_index(r_tr, d_tr, e_tr)
    val_auc = horizon_auc(r_va, d_va, e_va, horizon)

    result: Dict[str, Any] = {
        "best_val_c": val_c,
        "best_train_c": train_c,
        "best_val_auc": val_auc,
        "best_test_c": float("nan"),
        "best_test_auc": float("nan"),
        "n_train": int(len(ids_tr)),
        "n_val": int(len(ids_va)),
        "n_test": int(len(ids_te)),
        "features": keep_cols,
        "horizon_months": float(horizon),
        "penalizer": float(penalizer),
        # Per-patient out-of-fold predictions for pooled CV concordance.
        "val_predictions": [
            {"patient_id": p, "risk": float(r), "duration": float(d), "event": int(e)}
            for p, r, d, e in zip(ids_va, r_va, d_va, e_va)
        ],
    }

    if len(ids_te) > 0:
        r_te = _risk(df_te)
        d_te, e_te = df_te["duration"].values, df_te["event"].values
        result["best_test_c"] = concordance_index(r_te, d_te, e_te)
        result["best_test_auc"] = horizon_auc(r_te, d_te, e_te, horizon)

    logger.info(
        "%sCoxPH(penalizer=%.3g)  train=%d (ev=%d)  val=%d (ev=%d)  "
        "val_C=%.3f  train_C=%.3f  val_AUC=%.3f",
        log_prefix, penalizer,
        len(ids_tr), int(e_tr.sum()), len(ids_va), int(e_va.sum()),
        val_c, train_c, val_auc,
    )
    return result
