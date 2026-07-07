"""Feature / modality importance for the trained longitudinal model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from pdac_longitudinal.training.loop import (
    _anatomy_feat,
    _clinical_feat,
    _radiomic_feat,
    _vessel_feat,
)

logger = logging.getLogger(__name__)


def _cindex(risk: np.ndarray, dur: np.ndarray, evt: np.ndarray) -> float:
    """Harrell's C (higher risk -> shorter survival).

    Args:
        risk: Risk scores; non-finite entries are dropped.
        dur: Observed time-to-event or censoring, aligned with `risk`.
        evt: Event indicator (1=event, 0=censored), aligned with `risk`.

    Returns:
        Concordance index, or `nan` if no comparable pairs.
    """
    m = np.isfinite(risk)
    risk, dur, evt = risk[m], dur[m], evt[m]
    num = den = 0.0
    for i in range(len(dur)):
        if evt[i] == 1:
            later = dur > dur[i]
            den += later.sum()
            num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return num / den if den > 0 else float("nan")


@torch.no_grad()
def _collect(model, val_loader, device, clinical_dim, anatomy_dim, vessel_dim,
             radiomic_dim: int = 0, enc_chunk: int = 2):
    """Run the encoder once per patient; cache global-pooled deep tokens + tabular.

    Args:
        model: Trained longitudinal model; only `model.encode` is used.
        val_loader: DataLoader providing the batches to encode.
        device: Torch device to run encoding on.
        clinical_dim: Width of the clinical feature vector.
        anatomy_dim: Width of the anatomy feature vector; 0 disables collection.
        vessel_dim: Width of the vessel feature vector; 0 disables collection.
        radiomic_dim: Width of the radiomic feature vector; 0 disables collection.
        enc_chunk: Sub-batch size for the imaging encoder pass.

    Returns:
        Dict with `pooled` (per-stage global-pooled deep tensors),
        `clin`/`anat`/`vessel`/`radiomic` (tabular tensors or `None` if
        unused), `enc` (frozen pre-fusion encoder features), `dur`/`evt`
        (survival arrays), and `case_ids`.
    """
    pooled_stages: Optional[List[List[torch.Tensor]]] = None
    clin_l, anat_l, vessel_l, radio_l, dur_l, evt_l, enc_l = [], [], [], [], [], [], []
    case_ids: List[str] = []
    has_anat = anatomy_dim > 0
    has_vessel = vessel_dim > 0
    has_radiomic = radiomic_dim > 0
    for batch in val_loader:
        # tabular + survival for the whole batch
        clin_l.append(_clinical_feat(batch, clinical_dim, device).cpu())
        if has_anat:
            a = _anatomy_feat(batch, anatomy_dim, device)
            if a is not None:
                anat_l.append(a.cpu())
        if has_vessel:
            ve = _vessel_feat(batch, vessel_dim, device)
            if ve is not None:
                vessel_l.append(ve.cpu())
        if has_radiomic:
            rd = _radiomic_feat(batch, radiomic_dim, device)
            if rd is not None:
                radio_l.append(rd.cpu())
        dur_l.append(batch["duration"].cpu())
        evt_l.append(batch["event"].cpu())
        cid = batch.get("case_id")
        if cid is not None:
            case_ids.extend(str(c) for c in cid)
        # imaging encode in sub-chunks to bound peak GPU memory
        B = batch["t0"].shape[0]
        for s in range(0, B, enc_chunk):
            sl = slice(s, min(s + enc_chunk, B))
            x0 = batch["t0"][sl].to(device)
            x1 = batch["t1"][sl].to(device)
            v0 = batch["valid_t0"][sl].to(device).bool() if "valid_t0" in batch else None
            v1 = batch["valid_t1"][sl].to(device).bool() if "valid_t1" in batch else None
            fused, _, stage_pairs = model.encode(
                x0, x1, valid_T0=v0, valid_T1=v1, return_attn=False)
            # Match training: a use_imaging=False checkpoint never saw real
            # deep tokens, so zero them here too.
            if not getattr(model, "use_imaging", True):
                fused = [torch.zeros_like(f) for f in fused]
            pooled = [
                (F.adaptive_avg_pool3d(f, 1).flatten(1) if f.dim() == 5 else f).cpu()
                for f in fused
            ]
            if pooled_stages is None:
                pooled_stages = [[] for _ in pooled]
            for i, p in enumerate(pooled):
                pooled_stages[i].append(p)
            # frozen pre-fusion encoder features, both timepoints pooled;
            # X for the vessel-geometry probe.
            enc_l.append(torch.cat(
                [F.adaptive_avg_pool3d(ft, 1).flatten(1).cpu()
                 for pair in stage_pairs for ft in pair], dim=1))
            del x0, x1, v0, v1, fused, stage_pairs

    return {
        "pooled": [torch.cat(s, 0) for s in pooled_stages],
        "clin": torch.cat(clin_l, 0),
        "anat": torch.cat(anat_l, 0) if (has_anat and anat_l) else None,
        "vessel": torch.cat(vessel_l, 0) if (has_vessel and vessel_l) else None,
        "radiomic": torch.cat(radio_l, 0) if (has_radiomic and radio_l) else None,
        "enc": torch.cat(enc_l, 0),
        "dur": torch.cat(dur_l).numpy().ravel(),
        "evt": torch.cat(evt_l).numpy().ravel(),
        "case_ids": case_ids,
    }


@torch.no_grad()
def _risk(model, data, device, override=None, chunk: int = 16) -> np.ndarray:
    """Risk from cached deep tokens + tabular through only the fusion head.

    Args:
        model: Trained longitudinal model; only `fusion_head`/`risk_head` are used.
        data: Cached features from `_collect`.
        device: Torch device to run on.
        override: Optional dict overriding one or more of `pooled`/`clin`/
            `anat`/`vessel`/`radiomic` from `data`.
        chunk: Sub-batch size through the fusion head.

    Returns:
        Risk score per patient, in `data`/`override` row order.
    """
    override = override or {}
    pooled = override.get("pooled", data["pooled"])
    clin = override.get("clin", data["clin"])
    anat = override.get("anat", data["anat"])
    vessel = override.get("vessel", data["vessel"])
    radio = override.get("radiomic", data.get("radiomic"))
    n = pooled[0].shape[0]
    out = []
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        deep = [p[sl].to(device) for p in pooled]
        _, aux = model.fusion_head(
            deep_features=deep,
            clinical_features=clin[sl].to(device),
            anatomy_features=anat[sl].to(device) if anat is not None else None,
            vessel_features=vessel[sl].to(device) if vessel is not None else None,
            radiomic_features=radio[sl].to(device) if radio is not None else None,
            return_tokens=False,
        )
        out.append(model.risk_head(aux["embedding"]).squeeze(1).float().cpu().numpy().ravel())
    return np.concatenate(out)


def _permute_delta(model, data, device, key, col, base_c, rng, n_rep) -> float:
    """Mean ΔC from permuting `key` (modality token list, or one tabular column).

    Args:
        model: Trained longitudinal model.
        data: Cached features from `_collect`.
        device: Torch device to run on.
        key: Which entry of `data` to permute; `"pooled"` is the whole
            imaging modality, otherwise a tabular key.
        col: Column index to permute within `data[key]`; `None` permutes the
            whole tensor's row order.
        base_c: Unpermuted baseline C-index to subtract from.
        rng: Random generator for the permutation draws.
        n_rep: Number of permutation repeats to average over.

    Returns:
        Mean ΔC-index (baseline minus permuted) over `n_rep` repeats, or
        `nan` if `data[key]` is `None`.
    """
    n = data["pooled"][0].shape[0]
    drops = []
    for _ in range(n_rep):
        perm = rng.permutation(n)
        if key == "pooled":
            ov = {"pooled": [p[perm] for p in data["pooled"]]}
        else:
            if data[key] is None:
                return float("nan")
            t = data[key].clone()
            if col is None:
                t = t[perm]
            else:
                t[:, col] = t[perm, col]
            ov = {key: t}
        drops.append(base_c - _cindex(_risk(model, data, device, ov), data["dur"], data["evt"]))
    return float(np.mean(drops))


def _run_shap(model, data, device, registry, output_dir, n_samples=16,
              radiomic_loadings=None, radiomic_cols=None):
    """GradientSHAP attributions for imaging + clinical + vessel + radiomic.

    Writes per-patient SHAP values plus summary beeswarm plots to `output_dir`.

    Args:
        model: Trained longitudinal model; `fusion_head`/`risk_head` are used.
        data: Cached features from `_collect`.
        device: Torch device to run on.
        registry: Clinical registry providing `clinical_cols` for labeling.
        output_dir: Directory to write `shap_values.npz` and beeswarm PNGs.
        n_samples: GradientSHAP interpolation samples per attribution call.
        radiomic_loadings: `(k, n_features)` PCA loading matrix; enables the
            PC-to-feature load-back when the radiomic token is PCA-reduced.
        radiomic_cols: Feature names for the radiomic token.

    Returns:
        Dict with `shap_values_npz`, the path to the saved `.npz`.
    """
    from captum.attr import GradientShap
    from pdac_longitudinal.preprocess.vessel_features import VESSEL_FEATURE_COLS

    has_vessel = data["vessel"] is not None
    has_radio = data.get("radiomic") is not None
    clin_names = list(getattr(registry, "clinical_cols", []))[: data["clin"].shape[1]]
    ves_names = list(VESSEL_FEATURE_COLS)[: data["vessel"].shape[1]] if has_vessel else []
    n = data["pooled"][0].shape[0]

    clin_all = data["clin"].to(device).float()
    ves_all = data["vessel"].to(device).float() if has_vessel else None
    anat_all = data["anat"].to(device).float() if data["anat"] is not None else None
    radio_all = data["radiomic"].to(device).float() if has_radio else None
    pooled_all = [p.to(device).float() for p in data["pooled"]]
    stage_sizes = [p.shape[1] for p in pooled_all]
    img_all = torch.cat(pooled_all, dim=1)        # (N, ΣC); imaging as one input
    fh, rh = model.fusion_head, model.risk_head

    sv_clin = np.zeros((n, clin_all.shape[1]), dtype=np.float32)
    sv_ves = np.zeros((n, ves_all.shape[1]), dtype=np.float32) if has_vessel else None
    sv_radio = np.zeros((n, radio_all.shape[1]), dtype=np.float32) if has_radio else None
    # signed per-modality SHAP total per patient (imaging/clinical/vessel/radiomic)
    mod_names = ["imaging", "clinical", "vessel", "radiomic"]
    mod_col = {m: j for j, m in enumerate(mod_names)}
    mod_signed = np.zeros((n, 4), dtype=np.float32)

    # Only attribute modalities the model consumes — imaging always is one.
    # An unused zero-width token makes autograd raise "differentiated Tensor not used".
    attr_specs = []                                       # (name, all_tensor)
    if clin_all.shape[1] > 0:
        attr_specs.append(("clinical", clin_all))
    if has_vessel and ves_all.shape[1] > 0:
        attr_specs.append(("vessel", ves_all))
    attr_specs.append(("imaging", img_all))
    if has_radio and radio_all.shape[1] > 0:
        attr_specs.append(("radiomic", radio_all))
    attr_order = [s[0] for s in attr_specs]

    torch.set_grad_enabled(True)
    for i in range(n):
        anat_i = anat_all[i:i + 1] if anat_all is not None else None

        def fwd(*diff_args, _i=i, _order=attr_order, _anat=anat_i,
                _sizes=stage_sizes, _hv=has_vessel, _hr=has_radio):
            kw = dict(zip(_order, diff_args))
            b = diff_args[0].shape[0]
            deep = list(torch.split(kw["imaging"], _sizes, dim=1))
            clin = kw.get("clinical")
            if clin is None:                              # context (width-0 or off)
                clin = clin_all[_i:_i + 1].expand(b, -1)
            ves = kw.get("vessel") if _hv else None
            radio = kw.get("radiomic") if _hr else None
            anat_b = _anat.expand(b, *_anat.shape[1:]) if _anat is not None else None
            _, aux = fh(deep_features=deep, clinical_features=clin,
                        vessel_features=ves, anatomy_features=anat_b,
                        radiomic_features=radio, return_tokens=False)
            return rh(aux["embedding"]).squeeze(1)

        gs = GradientShap(fwd)
        inputs = tuple(t[i:i + 1].clone() for _, t in attr_specs)
        baselines = tuple(t for _, t in attr_specs)
        attr = gs.attribute(inputs, baselines=baselines,
                            n_samples=n_samples, stdevs=0.09)
        am = dict(zip(attr_order, attr))
        if "clinical" in am:
            sv_clin[i] = am["clinical"].detach().cpu().numpy().ravel()
        if has_vessel and "vessel" in am:
            sv_ves[i] = am["vessel"].detach().cpu().numpy().ravel()
        if has_radio and "radiomic" in am:
            sv_radio[i] = am["radiomic"].detach().cpu().numpy().ravel()
        for nm, a in am.items():
            mod_signed[i, mod_col[nm]] = float(a.sum())
    torch.set_grad_enabled(False)

    # radiomic per-feature load-back
    # sv_radio columns are PCs when PCA is active, raw features otherwise.
    radio_feat_names = radio_vals_out = sv_radio_feat = None
    if has_radio:
        k = sv_radio.shape[1]
        if radiomic_loadings is not None and radiomic_cols is not None \
                and radiomic_loadings.shape[0] == k:
            # project mean|φ_pc| through |loading| -> per-feature importance (unsigned)
            mean_abs_pc = np.abs(sv_radio).mean(0)                     # (k,)
            feat_imp = (mean_abs_pc[:, None] * np.abs(radiomic_loadings)).sum(0)
            radio_feat_names = list(radiomic_cols)
            sv_radio_feat = feat_imp                                  # (n_features,) aggregate
        elif radiomic_cols is not None and len(radiomic_cols) == k:
            radio_feat_names = list(radiomic_cols)                    # raw token -> 1:1
            radio_vals_out = radio_all.cpu().numpy()
            sv_radio_feat = np.abs(sv_radio).mean(0)

    od = Path(output_dir)
    save_kw = dict(
        shap_clinical=sv_clin, clinical_values=clin_all.cpu().numpy(),
        clinical_names=np.array(clin_names),
        shap_vessel=(sv_ves if has_vessel else np.zeros((n, 0))),
        vessel_values=(ves_all.cpu().numpy() if has_vessel else np.zeros((n, 0))),
        vessel_names=np.array(ves_names),
        shap_radiomic=(sv_radio if has_radio else np.zeros((n, 0))),
        radiomic_values=(radio_all.cpu().numpy() if has_radio else np.zeros((n, 0))),
        modality_signed=mod_signed,
        modality_names=np.array(mod_names),
    )
    if radio_feat_names is not None:
        save_kw["radiomic_feature_names"] = np.array(radio_feat_names)
        save_kw["radiomic_feature_importance"] = sv_radio_feat
    np.savez(od / "shap_values.npz", **save_kw)

    mod_imp = np.abs(mod_signed).mean(0)
    logger.info("SHAP modality importance (mean |signed sum| per patient)")
    for nm, v in sorted(zip(mod_names, mod_imp), key=lambda x: -x[1]):
        logger.info("    %-9s mean|SHAP| = %.4f", nm, v)
    logger.info("Top per-feature mean|SHAP| (clinical+vessel)")
    allnames = ["clinical:" + c for c in clin_names] + ["vessel:" + c for c in ves_names]
    allsv = np.concatenate([sv_clin] + ([sv_ves] if has_vessel else []), axis=1)
    if allsv.shape[1]:
        mean_abs = np.abs(allsv).mean(0)
        for idx in np.argsort(-mean_abs)[:12]:
            logger.info("    %-42s mean|SHAP| = %.4f", allnames[idx], mean_abs[idx])
    if radio_feat_names is not None:
        kind = "PC-to-feature load-back" if radiomic_loadings is not None else "per-feature"
        logger.info("Top radiomic features by SHAP (%s)", kind)
        order = np.argsort(-sv_radio_feat)[:15]
        for idx in order:
            logger.info("    %-46s imp = %.4f", radio_feat_names[idx], sv_radio_feat[idx])

    # plots (optional; npz is the durable output)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap as shap_lib
        # per-feature beeswarms
        for tag, sv, vals, names in (
            [("clinical", sv_clin, clin_all.cpu().numpy(), clin_names)]
            + ([("vessel", sv_ves, ves_all.cpu().numpy(), ves_names)] if has_vessel else [])
        ):
            if not len(names):
                continue
            shap_lib.summary_plot(sv, features=vals, feature_names=names, show=False,
                                  max_display=min(20, len(names)))
            plt.tight_layout(); plt.savefig(od / f"shap_beeswarm_{tag}.png", dpi=150)
            plt.close()
        # radiomic beeswarm: raw-feature token plots directly; PCA token plots PCs
        if has_radio:
            r_names = (radio_feat_names if (radiomic_loadings is None and radio_feat_names)
                       else [f"PC{j}" for j in range(sv_radio.shape[1])])
            r_vals = (radio_vals_out if radio_vals_out is not None
                      else radio_all.cpu().numpy())
            shap_lib.summary_plot(sv_radio, features=r_vals, feature_names=r_names,
                                  show=False, max_display=20)
            plt.tight_layout(); plt.savefig(od / "shap_beeswarm_radiomic.png", dpi=150)
            plt.close()
        # modality comparison: beeswarm of signed per-modality SHAP
        shap_lib.summary_plot(mod_signed, features=np.abs(mod_signed),
                              feature_names=mod_names, show=False)
        plt.tight_layout(); plt.savefig(od / "shap_modality.png", dpi=150); plt.close()
        logger.info("SHAP plots written to %s/shap_{beeswarm_*,modality}.png", od)
    except Exception as exc:  # pragma: no cover
        logger.warning("SHAP plotting skipped (%s); values saved to shap_values.npz", exc)
    return {"shap_values_npz": str(od / "shap_values.npz")}


def _linear_probe(X, Y, ynames):
    """5-fold CV R² of a linear readout of frozen-encoder features -> each vessel scalar.

    Args:
        X: Frozen-encoder feature matrix, `(n_samples, n_features)`.
        Y: Vessel scalar matrix, `(n_samples, len(ynames))`.
        ynames: Names of the vessel scalars, aligned with `Y`'s columns.

    Returns:
        Dict mapping vessel scalar name to its cross-validated R²; near-
        constant or fit-failure targets are omitted.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    ncomp = int(max(2, min(40, n - 5, X.shape[1])))
    cv = KFold(5, shuffle=True, random_state=0)
    r2 = {}
    for j, name in enumerate(ynames):
        y = Y[:, j].astype(float)
        if np.std(y) < 1e-8:
            continue
        pipe = make_pipeline(StandardScaler(), PCA(n_components=ncomp),
                             RidgeCV(alphas=np.logspace(-1, 4, 12)))
        try:
            pred = cross_val_predict(pipe, X, y, cv=cv)
            r2[name] = float(r2_score(y, pred))
        except Exception:  # pragma: no cover
            continue
    return r2


def _raw_logz_matrix(loader, scaler, case_ids):
    """Post-signed-log + z (pre-PCA) radiomic matrix, ordered to `case_ids`.

    Args:
        loader: DataLoader whose `.dataset` exposes `cases` and `_cache_path`.
        scaler: Fitted `RadiomicScaler`; its mean/std are applied.
        case_ids: Patient IDs to look up, in the desired output row order.

    Returns:
        `float32` array `(len(case_ids), n_features)`, or `None` if any
        `case_id` is missing cached radiomic features.
    """
    from pdac_longitudinal.radiomics.feature_schema import (
        decode_radiomic_features, radiomic_dict_to_vector, signed_log)
    ds = loader.dataset
    raw = {}
    for case in getattr(ds, "cases", []):
        pid = str(case["patient_id"])
        cp = ds._cache_path(case["patient_id"])
        if cp is None or not cp.exists():
            continue
        with np.load(cp, allow_pickle=False) as z:
            if "radiomic_features_json" not in z.files:
                continue
            feats = decode_radiomic_features(
                {"radiomic_features_json": z["radiomic_features_json"]})
        raw[pid] = radiomic_dict_to_vector(feats)
    if not all(c in raw for c in case_ids):
        return None
    M = np.stack([raw[c] for c in case_ids])
    Z = signed_log(np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0))
    if scaler.mean is not None and scaler.std is not None:
        Z = (Z - scaler.mean) / scaler.std
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _permute_through_pca(model, data, device, Z, scaler, base_c, rng, n_rep,
                         feature_names):
    """Direct per-NAMED-feature ΔC for a PCA-token model.

    Args:
        model: Trained longitudinal model.
        data: Cached features from `_collect`.
        device: Torch device to run on.
        Z: Post-signed-log + z (pre-PCA) radiomic matrix, from `_raw_logz_matrix`.
        scaler: Fitted `RadiomicScaler` supplying `pca_mean`/`pca_comp`.
        base_c: Unpermuted baseline C-index to subtract from.
        rng: Random generator for the permutation draws.
        n_rep: Number of permutation repeats to average over.
        feature_names: Names for `Z`'s columns, used as the output dict keys.

    Returns:
        Dict mapping feature name to mean ΔC-index; 0.0 for near-constant
        columns.
    """
    mu, W = scaler.pca_mean, scaler.pca_comp                    # (F,), (k, F)
    n, F = Z.shape
    out = {}
    for j in range(F):
        col = Z[:, j]
        if float(col.std()) < 1e-9:
            out[feature_names[j]] = 0.0
            continue
        drops = []
        for _ in range(n_rep):
            Zp = Z.copy()
            Zp[:, j] = col[rng.permutation(n)]
            pcs = ((Zp - mu) @ W.T).astype(np.float32)
            r = _risk(model, data, device, {"radiomic": torch.from_numpy(pcs)}, chunk=n)
            drops.append(base_c - _cindex(r, data["dur"], data["evt"]))
        out[feature_names[j]] = float(np.mean(drops))
    return out


def run_feature_importance(model, val_loader, device, clinical_dim, anatomy_dim,
                           vessel_dim, registry, output_dir, fold=None, n_rep=20,
                           do_shap=False, radiomic_dim=0,
                           radiomic_loadings=None, radiomic_cols=None,
                           perm_through_pca=False, pca_perm_n_rep=5):
    """Compute and log permutation + SHAP feature importance for a trained model.

    Writes the combined results to `feature_importance.json` in `output_dir`.

    Args:
        model: Trained longitudinal model in eval mode (set internally).
        val_loader: DataLoader for the fold's validation set.
        device: Torch device to run on.
        clinical_dim: Width of the clinical feature vector.
        anatomy_dim: Width of the anatomy feature vector (0 if unused).
        vessel_dim: Width of the vessel feature vector (0 if unused).
        registry: Clinical registry providing `clinical_cols` for labeling.
        output_dir: Directory to write `feature_importance.json` (and SHAP
            outputs, if `do_shap`).
        fold: Fold index/label, recorded in the output and log lines.
        n_rep: Number of permutation repeats averaged per feature/modality.
        do_shap: If True, also run GradientSHAP.
        radiomic_dim: Width of the radiomic feature vector (0 if unused).
        radiomic_loadings: `(k, n_features)` PCA loading matrix; enables the
            PC-to-feature load-back when the radiomic token is PCA-reduced.
        radiomic_cols: Feature names for the radiomic token.
        perm_through_pca: When True, permutes raw features through the fitted
            PCA transform for direct per-feature ΔC.
        pca_perm_n_rep: Permutation repeats for the permute-through-PCA pass.

    Returns:
        Dict with `fold`, `n`, `events`, `baseline_val_c`,
        `modality_permutation_dC`, `per_feature_permutation_dC`,
        `radiomic_pc_permutation_dC`, `radiomic_feature_importance`,
        `radiomic_feature_permute_pca`, `univariate_vessel_c`, and
        `encoder_probe_r2`. ΔC values are baseline minus permuted C-index.
    """
    from pdac_longitudinal.preprocess.vessel_features import VESSEL_FEATURE_COLS

    model.eval()
    rng = np.random.default_rng(0)
    data = _collect(model, val_loader, device, clinical_dim, anatomy_dim,
                    vessel_dim, radiomic_dim=radiomic_dim)
    n = data["pooled"][0].shape[0]
    base_c = _cindex(_risk(model, data, device), data["dur"], data["evt"])
    logger.info("Feature importance | fold=%s  N=%d  events=%d  baseline val C=%.4f",
                fold, n, int(data["evt"].sum()), base_c)

    # 1. modality (token) permutation
    groups = {"imaging": "pooled", "clinical": "clin", "vessel": "vessel",
              "anatomy": "anat", "radiomic": "radiomic"}
    modality = {}
    for name, key in groups.items():
        if key != "pooled" and data[key] is None:
            continue
        modality[name] = _permute_delta(model, data, device, key, None, base_c, rng, n_rep)
    logger.info("Modality permutation delta-C (higher means model relies on it)")
    for name, d in sorted(modality.items(), key=lambda x: -x[1]):
        logger.info("    %-9s ΔC = %+.4f", name, d)

    # 2. per-feature permutation (vessel + clinical)
    per_feature = {}
    if data["vessel"] is not None:
        for j, c in enumerate(list(VESSEL_FEATURE_COLS)[: data["vessel"].shape[1]]):
            per_feature[f"vessel:{c}"] = _permute_delta(model, data, device, "vessel", j, base_c, rng, n_rep)
    for j, c in enumerate(list(getattr(registry, "clinical_cols", []))[: data["clin"].shape[1]]):
        per_feature[f"clinical:{c}"] = _permute_delta(model, data, device, "clin", j, base_c, rng, n_rep)
    for c, d in sorted(per_feature.items(), key=lambda x: -x[1])[:15]:
        logger.info("    top per-feature  %-38s ΔC = %+.4f", c, d)

    # 2b. radiomic per-component permutation + PC->feature load-back
    radiomic_pc_dC = {}
    radiomic_feature_importance = {}
    if data["radiomic"] is not None:
        k = data["radiomic"].shape[1]
        for j in range(k):
            radiomic_pc_dC[f"PC{j}"] = _permute_delta(
                model, data, device, "radiomic", j, base_c, rng, n_rep)
        for c, d in sorted(radiomic_pc_dC.items(), key=lambda x: -x[1])[:10]:
            logger.info("    radiomic component  %-10s ΔC = %+.4f", c, d)
        # load-back: feature_importance[f] = Σ_pc ΔC[pc]·|loading[pc,f]|
        if radiomic_loadings is not None and radiomic_cols is not None \
                and radiomic_loadings.shape[0] == k:
            w = np.array([max(radiomic_pc_dC[f"PC{j}"], 0.0) for j in range(k)])  # (k,)
            feat_imp = (w[:, None] * np.abs(radiomic_loadings)).sum(axis=0)        # (n_features,)
            for f, v in zip(radiomic_cols, feat_imp):
                radiomic_feature_importance[f] = float(v)
            logger.info("Top radiomic features (PC delta-C times |loading|, load-back)")
            for f, v in sorted(radiomic_feature_importance.items(),
                               key=lambda x: -x[1])[:15]:
                logger.info("    %-46s imp = %.4f", f, v)
        # No PCA: the token is the raw features already, so ΔC maps 1:1 to names.
        elif radiomic_loadings is None and radiomic_cols is not None \
                and len(radiomic_cols) == k:
            for j, f in enumerate(radiomic_cols):
                radiomic_feature_importance[f] = float(radiomic_pc_dC[f"PC{j}"])
            logger.info("Top radiomic features (per-feature permutation delta-C)")
            for f, v in sorted(radiomic_feature_importance.items(),
                               key=lambda x: -x[1])[:15]:
                logger.info("    %-46s ΔC = %+.4f", f, v)

    # 2c. direct per-feature ΔC via permute-through-PCA
    radiomic_feature_permute_pca = {}
    if perm_through_pca and radiomic_loadings is not None and radiomic_cols is not None:
        scaler = getattr(val_loader.dataset, "_radiomic_scaler", None)
        if scaler is not None and scaler.pca_comp is not None and data.get("case_ids"):
            Z = _raw_logz_matrix(val_loader, scaler, data["case_ids"])
            if Z is not None and Z.shape[1] == radiomic_loadings.shape[1]:
                logger.info("Permute-through-PCA: DIRECT per-feature ΔC over %d "
                            "features (n_rep=%d) …", Z.shape[1], pca_perm_n_rep)
                radiomic_feature_permute_pca = _permute_through_pca(
                    model, data, device, Z, scaler, base_c, rng, pca_perm_n_rep,
                    list(radiomic_cols))
                logger.info("Top radiomic features (direct permute-through-PCA delta-C)")
                for f, v in sorted(radiomic_feature_permute_pca.items(),
                                   key=lambda x: -x[1])[:15]:
                    logger.info("    %-46s ΔC = %+.4f", f, v)
            else:
                logger.warning("permute-through-PCA skipped: raw matrix unavailable "
                               "or feature-count mismatch.")

    # 3. univariate vessel-feature concordance (model-free)
    univ = {}
    if data["vessel"] is not None:
        V = data["vessel"].numpy()
        for j, c in enumerate(list(VESSEL_FEATURE_COLS)[: V.shape[1]]):
            univ[c] = _cindex(V[:, j], data["dur"], data["evt"])
        for c, uc in sorted(univ.items(), key=lambda x: -abs(x[1] - 0.5))[:10]:
            logger.info("    univariate vessel  %-38s C = %.3f", c, uc)

    # 4. linear probe: frozen-encoder features -> vessel scalars
    probe = {}
    if data["vessel"] is not None:
        probe = _linear_probe(data["enc"].numpy(), data["vessel"].numpy(),
                              list(VESSEL_FEATURE_COLS)[: data["vessel"].shape[1]])
        if probe:
            vals = np.array(list(probe.values()))
            logger.info("Linear probe: frozen-encoder features to vessel scalars "
                        "(5-fold CV R-squared; high means encoder already encodes it)")
            for c, r in sorted(probe.items(), key=lambda x: -x[1])[:12]:
                logger.info("    %-40s R² = %+.3f", c, r)
            logger.info("    summary: median R²=%.3f  features R²>0.3: %d/%d  "
                        "(encoder dim=%d, N=%d)", float(np.median(vals)),
                        int((vals > 0.3).sum()), len(vals), data["enc"].shape[1], n)

    if do_shap:
        logger.info("Running GradientSHAP on clinical + vessel + radiomic tokens …")
        _run_shap(model, data, device, registry, output_dir,
                  radiomic_loadings=radiomic_loadings, radiomic_cols=radiomic_cols)

    out = {
        "fold": fold, "n": int(n), "events": int(data["evt"].sum()),
        "baseline_val_c": base_c,
        "modality_permutation_dC": modality,
        "per_feature_permutation_dC": per_feature,
        "radiomic_pc_permutation_dC": radiomic_pc_dC,
        "radiomic_feature_importance": radiomic_feature_importance,
        "radiomic_feature_permute_pca": radiomic_feature_permute_pca,
        "univariate_vessel_c": univ,
        "encoder_probe_r2": probe,
    }
    p = Path(output_dir) / "feature_importance.json"
    p.write_text(json.dumps(out, indent=2))
    logger.info("Feature importance written to %s", p)
    return out
