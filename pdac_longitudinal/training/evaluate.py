"""Held-out-test ensemble evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import torch

from pdac_longitudinal.models.longitudinal_model import build_model_from_config
from pdac_longitudinal.training.checkpointing import load_checkpoint
from pdac_longitudinal.training.cv import build_cv_loaders

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_test_risks(model, loader, device, clinical_dim, anatomy_dim,
                       vessel_dim, radiomic_dim, amp_dtype, tta=False):
    """Per-patient risk for one fold model over the held-out test loader."""
    from pdac_longitudinal.training.loop import (
        _anatomy_feat, _clinical_feat, _radiomic_feat, _roi_pool_masks, _vessel_feat,
    )
    model.eval()
    out: Dict[str, tuple] = {}
    for batch in loader:
        x0 = batch["t0"].to(device); x1 = batch["t1"].to(device)
        clin = _clinical_feat(batch, clinical_dim, device)
        anat = _anatomy_feat(batch, anatomy_dim, device)
        vessel = _vessel_feat(batch, vessel_dim, device)
        radio = _radiomic_feat(batch, radiomic_dim, device)
        roi = _roi_pool_masks(model, batch, device)
        v0 = batch["valid_t0"].to(device).bool() if "valid_t0" in batch else None
        v1 = batch["valid_t1"].to(device).bool() if "valid_t1" in batch else None
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=amp_dtype):
            risk = model(x0, x1, radiomic_features=radio, clinical_features=clin,
                         anatomy_features=anat, vessel_features=vessel, roi_masks=roi,
                         valid_T0=v0, valid_T1=v1)["risk"].float()
            if tta:
                fr = model(torch.flip(x0, [2]), torch.flip(x1, [2]),
                           radiomic_features=radio, clinical_features=clin,
                           anatomy_features=anat, vessel_features=vessel,
                           roi_masks=({k: torch.flip(v, [-3]) for k, v in roi.items()} if roi else None),
                           valid_T0=(torch.flip(v0, [2]) if v0 is not None else None),
                           valid_T1=(torch.flip(v1, [2]) if v1 is not None else None))["risk"].float()
                risk = 0.5 * (risk + fr)
        pids = list(batch.get("case_id", batch.get("patient_id")))
        for i, pid in enumerate(pids):
            out[pid] = (float(risk[i].cpu()), float(batch["duration"][i]), int(batch["event"][i]))
    return out


def run_ensemble_eval(cfg, registry, folds, test_ids, run_dir, device, segmenter,
                      clinical_dim, anatomy_dim, vessel_dim, radiomic_dim, tta=False):
    """Average the fold models on the held-out test set and report test metrics."""
    import numpy as _np
    from lifelines.utils import concordance_index
    amp_dtype = torch.bfloat16 if getattr(cfg.training, "amp_dtype", "") == "bfloat16" else torch.float16
    horizon = float(getattr(cfg.training, "survival_horizon_months", 12.0))
    per_fold: Dict[str, List[float]] = {}
    meta: Dict[str, tuple] = {}
    fold_cis: List[float] = []

    def _auc_at_horizon(risk, dur, evt):
        """ROC-AUC for death-within-horizon"""
        from sklearn.metrics import roc_auc_score
        lab = _np.full(len(dur), -1)
        lab[dur >= horizon] = 0
        lab[(evt == 1) & (dur < horizon)] = 1
        m = lab >= 0
        if m.sum() > 1 and len(set(lab[m].tolist())) == 2:
            return float(roc_auc_score(lab[m], risk[m]))
        return float("nan")

    def _ipcw_auc_at_horizon(risk, dur, evt):
        """IPCW cumulative/dynamic AUC at the horizon; uses all patients, censored-before-horizon reweighted by 1/G (censoring KM), not dropped."""
        from lifelines import KaplanMeierFitter
        cases = [i for i in range(len(dur)) if dur[i] < horizon and evt[i] == 1]
        ctrls = [j for j in range(len(dur)) if dur[j] > horizon]
        if len(cases) < 1 or len(ctrls) < 1:
            return float("nan")
        kmf = KaplanMeierFitter().fit(dur, event_observed=(evt == 0))  # censoring KM
        num = den = 0.0
        for i in cases:
            w = 1.0 / max(float(kmf.predict(dur[i])), 1e-6)
            num += w * sum((risk[i] > risk[j]) + 0.5 * (risk[i] == risk[j]) for j in ctrls)
            den += w * len(ctrls)
        return float(num / den) if den > 0 else float("nan")

    for k, (tr, va) in enumerate(folds):
        ck_dir = Path(run_dir).expanduser() / f"fold{k}" / "checkpoints"
        ckpt = next((ck_dir / n for n in ("checkpoint_best_auc.pth", "checkpoint_best_cindex.pth")
                     if (ck_dir / n).exists()), None)
        if ckpt is None:
            logger.warning("fold %d: no best checkpoint in %s — skipping", k, ck_dir); continue
        if cfg.modules.clinical:
            registry.fit(tr)  # fold-internal clinical impute/z-score
        _, _, test_loader = build_cv_loaders(
            cfg=cfg, registry=registry, train_ids=tr, val_ids=va, test_ids=test_ids,
            segmenter=segmenter, max_seg_tiles=cfg.training.max_seg_tiles, use_wandb=False)
        model = build_model_from_config(cfg).to(device)
        load_checkpoint(str(ckpt), model=model, map_location=device, strict=True)
        risks = predict_test_risks(model, test_loader, device, clinical_dim, anatomy_dim,
                                   vessel_dim, radiomic_dim, amp_dtype, tta=tta)
        pid = list(risks); r = _np.array([risks[p][0] for p in pid])
        d = _np.array([risks[p][1] for p in pid]); e = _np.array([risks[p][2] for p in pid])
        fold_cis.append(float(concordance_index(d, -r, e)))
        rz = (r - r.mean()) / (r.std() + 1e-8)
        for p, z in zip(pid, rz):
            per_fold.setdefault(p, []).append(float(z))
        for p in pid:
            meta[p] = risks[p][1:]
        logger.info("fold %d test C=%.3f  (n=%d)", k, fold_cis[-1], len(pid))

    pids = list(per_fold)
    ens = _np.array([_np.mean(per_fold[p]) for p in pids])
    dur = _np.array([meta[p][0] for p in pids]); evt = _np.array([meta[p][1] for p in pids])
    coh = _np.array([registry.get_cohort(p) for p in pids])

    def _c(r, d, e): return float(concordance_index(d, -r, e))

    def _boot_ci(fn, r, d, e, B=2000, seed=0):
        """Patient-level bootstrap 95% CI for a metric fn(risk, dur, evt)."""
        rng = _np.random.default_rng(seed); n = len(r); vals = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            try:
                v = fn(r[idx], d[idx], e[idx])
                if v == v:  # not NaN
                    vals.append(v)
            except Exception:
                pass
        if len(vals) < 50:
            return [float("nan"), float("nan")]
        return [round(float(_np.percentile(vals, 2.5)), 3),
                round(float(_np.percentile(vals, 97.5)), 3)]

    res = {"n_test": len(pids), "n_folds_used": len(fold_cis), "horizon_months": horizon,
           "per_fold_test_c": fold_cis, "per_fold_test_c_mean": float(_np.mean(fold_cis)),
           "ensemble_test_c": _c(ens, dur, evt),
           "ensemble_test_c_ci": _boot_ci(_c, ens, dur, evt),
           "ensemble_test_auc": _auc_at_horizon(ens, dur, evt),
           "ensemble_test_ipcw_auc": _ipcw_auc_at_horizon(ens, dur, evt),
           "ensemble_test_ipcw_auc_ci": _boot_ci(_ipcw_auc_at_horizon, ens, dur, evt),
           "n_pos_within_horizon": int(((evt == 1) & (dur < horizon)).sum()),
           "tta": tta,
           # per-patient ensemble risk (z-pooled) for downstream metrics (IPCW, CI bootstrap)
           "per_patient": {p: {"risk": float(r), "dur": float(d), "evt": int(e),
                               "cohort": registry.get_cohort(p)}
                           for p, r, d, e in zip(pids, ens, dur, evt)}}
    for c in sorted(set(coh.tolist())):
        m = coh == c
        if m.sum() > 1:
            res[f"ensemble_test_c_{c}"] = _c(ens[m], dur[m], evt[m])
            res[f"ensemble_test_c_{c}_ci"] = _boot_ci(_c, ens[m], dur[m], evt[m])
            res[f"ensemble_test_auc_{c}"] = _auc_at_horizon(ens[m], dur[m], evt[m])
            res[f"ensemble_test_ipcw_auc_{c}"] = _ipcw_auc_at_horizon(ens[m], dur[m], evt[m])
            res[f"ensemble_test_ipcw_auc_{c}_ci"] = _boot_ci(_ipcw_auc_at_horizon, ens[m], dur[m], evt[m])
    cohorts = sorted(set(coh.tolist()))
    by_cohort_c = "  ".join(
        f"{c} {res.get(f'ensemble_test_c_{c}', float('nan')):.3f} "
        f"{res.get(f'ensemble_test_c_{c}_ci', [])}"
        for c in cohorts
    )
    by_cohort_ipcw = "  ".join(
        f"{c} {res.get(f'ensemble_test_ipcw_auc_{c}', float('nan')):.3f} "
        f"{res.get(f'ensemble_test_ipcw_auc_{c}_ci', [])}"
        for c in cohorts
    )
    logger.info("=" * 64)
    logger.info("HELD-OUT TEST ENSEMBLE  (n=%d, %d folds, TTA=%s, horizon=%.0fmo)",
                res["n_test"], res["n_folds_used"], tta, horizon)
    logger.info("  per-fold test C (mean) : %.3f  %s", res["per_fold_test_c_mean"],
                [round(x, 3) for x in fold_cis])
    logger.info("  ENSEMBLE test C        : %.3f %s  (%s)",
                res["ensemble_test_c"], res["ensemble_test_c_ci"], by_cohort_c)
    logger.info("  ENSEMBLE test IPCW-AUC@%-3.0f : %.3f %s  (%s) "
                "[%d deaths<horizon; naive %.3f]", horizon, res["ensemble_test_ipcw_auc"],
                res["ensemble_test_ipcw_auc_ci"], by_cohort_ipcw,
                res["n_pos_within_horizon"], res["ensemble_test_auc"])
    logger.info("=" * 64)
    out_path = Path(run_dir).expanduser() / f"ensemble_test_eval{'_tta' if tta else ''}.json"
    out_path.write_text(json.dumps(res, indent=2))
    logger.info("Ensemble test eval → %s", out_path)
    return res
