"""Train/val/test split utilities."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pdac_longitudinal.data.registry import ClinicalRegistry

logger = logging.getLogger(__name__)


def make_split(
    all_ids: List[str],
    registry: ClinicalRegistry,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    split_file: Optional[Path] = None,
    phases: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Stratified 70/15/15 train/val/test split.

    Stratifies by `(cohort, event)`, or `(cohort, phase, event)` when
    `phases` is given. Loads and filters a cached split from `split_file`
    if one exists, instead of resplitting.

    Args:
        all_ids: Patient ids to split.
        registry: Clinical registry, used for cohort and event lookup.
        val_fraction: Fraction of each stratum assigned to validation.
        test_fraction: Fraction of each stratum assigned to test.
        seed: RNG seed for the shuffle within each stratum.
        split_file: Path to load a cached split from, or persist a newly
            computed one to.
        phases: Optional patient id -> phase label mapping, added as an
            extra stratification key.

    Returns:
        A `(train_ids, val_ids, test_ids)` tuple.
    """
    if split_file and split_file.exists():
        logger.info("Loading existing split from %s", split_file)
        with open(split_file) as f:
            s = json.load(f)
        # Respect any current filter (exclusions, --max_cases).
        all_set = set(all_ids)
        train = [p for p in s["train"] if p in all_set]
        val   = [p for p in s["val"]   if p in all_set]
        test  = [p for p in s["test"]  if p in all_set]
        dropped = (set(s["train"]) | set(s["val"]) | set(s["test"])) - all_set
        if dropped:
            logger.info(
                "Dropped %d patients from cached split (now excluded): %s",
                len(dropped), sorted(dropped),
            )
        return train, val, test

    def _stratum(p: str) -> str:
        ev = registry.get_survival(p)[1]
        try:
            co = registry.get_cohort(p)
        except Exception:
            co = "unknown"
        if phases is not None:
            ph = phases.get(p, "unknown")
            return f"{co}__{ph}__{ev}"
        return f"{co}__{ev}"

    rng = random.Random(seed)
    groups: Dict[str, List[str]] = {}
    for pid in all_ids:
        groups.setdefault(_stratum(pid), []).append(pid)
    for g in groups.values():
        rng.shuffle(g)

    def _split(g: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(g)
        n_test = max(1, int(n * test_fraction)) if n > 1 else 0
        n_val  = max(1, int(n * val_fraction)) if n - n_test > 1 else 0
        return g[n_val + n_test:], g[n_test:n_val + n_test], g[:n_test]

    train_ids, val_ids, test_ids = [], [], []
    for key in sorted(groups):
        tr, va, te = _split(groups[key])
        train_ids += tr; val_ids += va; test_ids += te
    train_ids, val_ids, test_ids = sorted(train_ids), sorted(val_ids), sorted(test_ids)

    if split_file:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w") as f:
            json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    def _cohort_counts(ids: List[str]) -> str:
        from collections import Counter
        c = Counter(registry.get_cohort(p) for p in ids)
        return ", ".join(f"{k}={v}" for k, v in sorted(c.items()))

    def _phase_counts(ids: List[str]) -> str:
        from collections import Counter
        c = Counter((phases or {}).get(p, "unknown") for p in ids)
        return ", ".join(f"{k}={v}" for k, v in sorted(c.items()))

    def _events(ids: List[str]) -> int:
        return sum(registry.get_survival(p)[1] for p in ids)

    strat_label = "cohort×phase×event" if phases is not None else "cohort×event"
    if phases is not None:
        logger.info(
            "Split (stratified on %s)  "
            "train=%d (ev=%d; %s; phase=%s) | val=%d (ev=%d; %s; phase=%s) | "
            "test=%d (ev=%d; %s; phase=%s)",
            strat_label,
            len(train_ids), _events(train_ids), _cohort_counts(train_ids), _phase_counts(train_ids),
            len(val_ids),   _events(val_ids),   _cohort_counts(val_ids),   _phase_counts(val_ids),
            len(test_ids),  _events(test_ids),  _cohort_counts(test_ids),  _phase_counts(test_ids),
        )
    else:
        logger.info(
            "Split (stratified on %s)  "
            "train=%d (ev=%d; %s) | val=%d (ev=%d; %s) | test=%d (ev=%d; %s)",
            strat_label,
            len(train_ids), _events(train_ids), _cohort_counts(train_ids),
            len(val_ids),   _events(val_ids),   _cohort_counts(val_ids),
            len(test_ids),  _events(test_ids),  _cohort_counts(test_ids),
        )
    logger.info("Test set locked in %s", split_file)
    return train_ids, val_ids, test_ids


def stratified_kfold_ids(
    ids: List[str],
    events: List[int],
    n_folds: int,
    seed: int,
    cohorts: Optional[Sequence[str]] = None,
    phases: Optional[Sequence[str]] = None,
) -> List[Tuple[List[str], List[str]]]:
    """Stratified K-fold by joint `(cohort, event)` when `cohorts` is given.

    Args:
        ids: Patient ids to fold.
        events: Event indicator per id, same order as `ids`.
        n_folds: Number of folds.
        seed: RNG seed for the shuffle within each stratum.
        cohorts: Cohort label per id, same order as `ids`; omit to
            stratify on event only.
        phases: Phase label per id, same order as `ids`; omit to
            stratify on `(cohort, event)` only.

    Returns:
        `n_folds` `(train_ids, val_ids)` tuples.

    Raises:
        ValueError: If `ids`, `events`, `cohorts`, `phases` don't all have
        matching lengths.
    """
    rng = random.Random(seed)
    if cohorts is None:
        cohorts = ["_"] * len(ids)
    if phases is None:
        phases = ["_"] * len(ids)
    if not (len(ids) == len(events) == len(cohorts) == len(phases)):
        raise ValueError("ids, events, cohorts, phases must have matching lengths")

    # Bucket by (cohort, phase, event); "_" placeholder collapses to (cohort, event).
    strata: Dict[str, List[str]] = {}
    for pid, e, co, ph in zip(ids, events, cohorts, phases):
        strata.setdefault(f"{co}__{ph}__{int(e)}", []).append(pid)
    for g in strata.values():
        rng.shuffle(g)

    def _chunks(seq: List[str], k: int) -> List[List[str]]:
        out: List[List[str]] = [[] for _ in range(k)]
        for i, x in enumerate(seq):
            out[i % k].append(x)
        return out

    per_stratum_folds = {key: _chunks(seq, n_folds) for key, seq in strata.items()}

    folds: List[Tuple[List[str], List[str]]] = []
    for k in range(n_folds):
        val_ids: List[str] = []
        train_ids: List[str] = []
        for key, chunks in per_stratum_folds.items():
            val_ids.extend(chunks[k])
            for j in range(n_folds):
                if j != k:
                    train_ids.extend(chunks[j])
        folds.append((sorted(train_ids), sorted(val_ids)))
    return folds


def load_or_make_kfolds(
    pool_ids: List[str],
    events: List[int],
    n_folds: int,
    seed: int,
    cohorts: Optional[List[str]] = None,
    phases: Optional[List[str]] = None,
    folds_file: Optional[str] = None,
) -> List[Tuple[List[str], List[str]]]:
    """K stratified folds, persisted to `folds_file` for reproducibility.

    Args:
        pool_ids: Patient ids to fold.
        events: Event indicator per id, same order as `pool_ids`.
        n_folds: Number of folds.
        seed: RNG seed for the shuffle within each stratum.
        cohorts: Cohort label per id, same order as `pool_ids`.
        phases: Phase label per id, same order as `pool_ids`.
        folds_file: Path to persisted folds; regenerated if missing, corrupt,
            or mismatched with `n_folds`/`pool_ids`.

    Returns:
        `n_folds` `(train_ids, val_ids)` tuples.
    """
    fp = Path(folds_file).expanduser() if folds_file else None
    if fp is not None and fp.exists():
        try:
            data = json.loads(fp.read_text())
            fmap = data["fold_by_pid"]
            if int(data["n_folds"]) == n_folds and set(fmap) == set(pool_ids):
                folds = []
                for k in range(n_folds):
                    val = sorted(p for p in pool_ids if fmap[p] == k)
                    train = sorted(p for p in pool_ids if fmap[p] != k)
                    folds.append((train, val))
                logger.info("Loaded persisted CV folds from %s", fp)
                return folds
            logger.warning("Persisted folds %s don't match current pool/n_folds "
                           "— regenerating.", fp)
        except Exception as exc:
            logger.warning("Could not read %s (%s) — regenerating folds.", fp, exc)

    folds = stratified_kfold_ids(
        pool_ids, events, n_folds=n_folds, seed=seed, cohorts=cohorts, phases=phases,
    )
    if fp is not None:
        fmap = {p: k for k, (_, val) in enumerate(folds) for p in val}
        fp.parent.mkdir(parents=True, exist_ok=True)
        import os as _os
        tmp = fp.with_suffix(f".{_os.getpid()}.tmp")
        tmp.write_text(json.dumps(
            {"n_folds": n_folds, "seed": seed, "fold_by_pid": fmap}, indent=2))
        tmp.replace(fp)
        logger.info("Persisted CV folds → %s (%d patients, %d folds)",
                    fp, len(fmap), n_folds)
    return folds
