"""Core training loop for the PDAC longitudinal framework."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pdac_longitudinal.config import Config
from pdac_longitudinal.training.checkpointing import CheckpointPaths, save_checkpoint
from pdac_longitudinal.training.metrics import epoch_metric

logger = logging.getLogger(__name__)


# Feature helpers

def _clinical_feat(
    batch: Dict[str, Any],
    clinical_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Return clinical feature tensor, falling back to zeros if unavailable."""
    if clinical_dim > 0 and "clinical" in batch and batch["clinical"].shape[-1] == clinical_dim:
        return batch["clinical"].to(device).float()
    return torch.zeros(batch["t0"].shape[0], clinical_dim, device=device)


def _anatomy_feat(
    batch: Dict[str, Any],
    anatomy_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return anatomy tensor, or `None` when the anatomy branch is off."""
    if anatomy_dim <= 0:
        return None
    if "anatomy" not in batch or batch["anatomy"].shape[-1] != anatomy_dim:
        return None
    return batch["anatomy"].to(device).float()


def _vessel_feat(
    batch: Dict[str, Any],
    vessel_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return vessel-interface tensor, or `None` when the vessel branch is off."""
    if vessel_dim <= 0:
        return None
    if "vessel" not in batch or batch["vessel"].shape[-1] != vessel_dim:
        return None
    return batch["vessel"].to(device).float()


def _radiomic_feat(
    batch: Dict[str, Any],
    radiomic_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return the radiomic (T0|T1|Δ) tensor, or `None` when the branch is off."""
    if radiomic_dim <= 0:
        return None
    if "radiomic" not in batch or batch["radiomic"].shape[-1] != radiomic_dim:
        if not getattr(_radiomic_feat, "_warned", False):
            got = batch["radiomic"].shape[-1] if "radiomic" in batch else "absent"
            logger.warning(
                "radiomic_dim=%d but batch radiomic dim=%s — radiomics is being "
                "SILENTLY DROPPED (model trains without it). Check the radiomic "
                "scaler out_dim (PCA k vs n_train) matches the configured dim.",
                radiomic_dim, got,
            )
            _radiomic_feat._warned = True   # type: ignore[attr-defined]
        return None
    return batch["radiomic"].to(device).float()


# Step functions

def _mask_union(
    batch: Dict[str, Any], keys: Sequence[str],
    target_shape: Tuple[int, int, int], device: torch.device,
) -> torch.Tensor:
    """Union of the given mask keys, resized to `target_shape` (d,h,w)."""
    import torch.nn.functional as F
    B = batch["t0"].shape[0]
    acc: Optional[torch.Tensor] = None
    for k in keys:
        if k not in batch:
            continue
        m = batch[k].to(device).float()
        while m.dim() > 4:           # (B,1,D,H,W) -> (B,D,H,W)
            m = m.squeeze(1)
        m = F.interpolate(m.unsqueeze(1), size=target_shape, mode="nearest").squeeze(1)
        acc = m if acc is None else torch.maximum(acc, m)
    if acc is None:
        return torch.zeros((B, *target_shape), device=device)
    return (acc > 0.5).float()


# Attention-guidance ROI -> mask keys (T0 + T1).
_ATTN_GUIDANCE_ROI_KEYS: Dict[str, Tuple[str, ...]] = {
    "tumour":       ("mask_it", "mask_it_t1"),
    "peritumoural": ("mask_pt1", "mask_pt2", "mask_pt3",
                     "mask_pt1_t1", "mask_pt2_t1", "mask_pt3_t1"),
    "tvi":          ("mask_tvi", "mask_tvi_t1"),
    "liver":        ("liver_t0", "liver_t1"),
    "pancreas":     ("pancreas_t0", "pancreas_t1"),
    "kidneys":      ("kidneys_t0", "kidneys_t1"),
}


def _roi_pool_masks(
    model: Any, batch: Dict[str, Any], device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Per-compartment union masks for the fusion head's ROI pooling."""
    fh = getattr(model, "fusion_head", None)
    regions = getattr(fh, "roi_pool_regions", ()) if fh is not None else ()
    if not regions:
        return None
    shape = tuple(batch["t0"].shape[-3:])
    out: Dict[str, torch.Tensor] = {}
    for r in regions:
        keys = _ATTN_GUIDANCE_ROI_KEYS.get(r)
        if keys is None:
            continue
        out[r] = _mask_union(batch, keys, shape, device).unsqueeze(1)  # (B,1,D,H,W)
    return out or None


def _attn_guidance_term(
    out: Dict[str, Any], batch: Dict[str, Any], device: torch.device,
    stage: int, attn_guidance_reg: Any, roi_regions: Sequence[str],
    roi_weights: Sequence[float] = (),
) -> Optional[torch.Tensor]:
    """Apply the attention guidance regularizer to the *cross-attention* saliency at a stage."""
    amaps = out["attention_maps"][stage]
    if not amaps or "T1_to_T0" not in amaps:
        return None
    sal = 0.5 * (amaps["T1_to_T0"].float() + amaps["T0_to_T1"].float())  # (B,1,D,H,W)
    sal = sal.squeeze(1)                                                  # (B,D,H,W)
    b = sal.shape[0]
    flat = sal.reshape(b, -1)
    smin = flat.min(dim=1).values.view(b, 1, 1, 1)
    smax = flat.max(dim=1).values.view(b, 1, 1, 1)
    sal = (sal - smin) / (smax - smin + 1e-8)
    dhw = tuple(sal.shape[-3:])

    if not roi_weights:
        keys = tuple(k for r in roi_regions for k in _ATTN_GUIDANCE_ROI_KEYS.get(r, ()))
        roi = _mask_union(batch, keys, dhw, device)
        return attn_guidance_reg(sal, roi)

    # Weights control relative importance, not relative mask volume, since
    # each region's mask is normalised independently inside the regularizer.
    total: Optional[torch.Tensor] = None
    wsum = 0.0
    for region, w in zip(roi_regions, roi_weights):
        if w == 0:
            continue
        keys = _ATTN_GUIDANCE_ROI_KEYS.get(region, ())
        roi = _mask_union(batch, keys, dhw, device)
        term = attn_guidance_reg(sal, roi)
        total = w * term if total is None else total + w * term
        wsum += w
    if total is None or wsum == 0:
        return sal.sum() * 0.0
    return total / wsum


def _resolve_guidance_stages(stage_spec: Any, n_stages: int) -> Tuple[int, ...]:
    """Normalise `attn_guidance_stage` (int or list) -> tuple of non-neg indices."""
    specs = stage_spec if isinstance(stage_spec, (list, tuple)) else [stage_spec]
    out: List[int] = []
    for s in specs:
        s = int(s)
        s = s if s >= 0 else n_stages + s
        if 0 <= s < n_stages and s not in out:
            out.append(s)
    return tuple(out) or (n_stages - 1,)


def train_step(
    model: nn.Module,
    batch: Dict[str, Any],
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[Any],
    clinical_dim: int,
    anatomy_dim: int = 0,
    vessel_dim: int = 0,
    radiomic_dim: int = 0,
    attn_guidance_reg: Optional[Any] = None,
    attn_guidance_coef: float = 0.0,
    attn_guidance_stage: int = -1,
    attn_guidance_roi: Sequence[str] = ("tumour", "peritumoural"),
    attn_guidance_roi_weights: Sequence[float] = (),
    phase_adv_lambd: float = 0.0,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Forward + backward pass for one training batch."""
    model.train()
    x0  = batch["t0"].to(device)
    x1  = batch["t1"].to(device)
    dur = batch["duration"].to(device)
    evt = batch["event"].to(device)
    clin   = _clinical_feat(batch, clinical_dim, device)
    anat   = _anatomy_feat(batch, anatomy_dim, device)
    vessel = _vessel_feat(batch, vessel_dim, device)
    radio  = _radiomic_feat(batch, radiomic_dim, device)
    # Optional valid-region masks; when present, cross-attention ignores padded keys.
    valid_t0 = batch["valid_t0"].to(device).bool() if "valid_t0" in batch else None
    valid_t1 = batch["valid_t1"].to(device).bool() if "valid_t1" in batch else None

    if torch.isnan(x0).any():
        logger.warning("NaN in t0")
    if torch.isnan(x1).any():
        logger.warning("NaN in t1")
    if clin is not None and torch.isnan(clin).any():
        logger.warning("NaN in clinical features")
    if anat is not None and torch.isnan(anat).any():
        logger.warning("NaN in anatomy features")
    if vessel is not None and torch.isnan(vessel).any():
        logger.warning("NaN in vessel features")

    optimizer.zero_grad()
    use_attn = attn_guidance_reg is not None and attn_guidance_coef > 0
    amp_on   = scaler is not None

    # Materialise attention weights only at the guided stages
    skip_stages: Tuple[int, ...] = ()
    guided_stages: Tuple[int, ...] = ()
    if use_attn:
        n_stages = len(model.feature_dims)  # type: ignore[union-attr]
        guided_stages = _resolve_guidance_stages(attn_guidance_stage, n_stages)
        skip_stages = tuple(s for s in range(n_stages) if s not in guided_stages)

    roi_masks = _roi_pool_masks(model, batch, device)

    with torch.autocast(device_type=device.type, enabled=amp_on, dtype=amp_dtype):
        out  = model(x0, x1, radiomic_features=radio,
                     clinical_features=clin, anatomy_features=anat,
                     vessel_features=vessel, roi_masks=roi_masks,
                     valid_T0=valid_t0, valid_T1=valid_t1,
                     return_attn=use_attn,
                     skip_attn_stages=skip_stages)
        if torch.isnan(out["risk"]).any():
            logger.warning("NaN in model output risk score")
        loss = criterion(risk_scores=out["risk"], durations=dur, events=evt)

    attn_guidance_val = 0.0
    if use_attn and torch.isfinite(loss):
        # Skip stages that weren't materialised (returns None).
        terms = [
            t for st in guided_stages
            if (t := _attn_guidance_term(
                out, batch, device, st, attn_guidance_reg,
                attn_guidance_roi, attn_guidance_roi_weights)) is not None
        ]
        ta = torch.stack(terms).mean() if terms else None
        if ta is not None:
            loss = loss + attn_guidance_coef * ta
            attn_guidance_val = float(ta.detach().item())

    # Phase-adversarial regulariser (GRL).
    phase_adv_val = 0.0
    phase_adv_correct = 0
    phase_adv_n = 0
    phase_adv_class_counts: Optional[List[int]] = None
    if (phase_adv_lambd > 0.0 and torch.isfinite(loss)
            and "phase" in batch and hasattr(model, "phase_adv")):
        from torch.nn.functional import cross_entropy
        phase = batch["phase"].to(device).long()
        # GRL is inside model.phase_adv; reversed grad scales by lambd.
        phase_logits = model.phase_adv(out["embedding"].float(), lambd=phase_adv_lambd)
        pa = cross_entropy(phase_logits, phase)
        loss = loss + pa
        phase_adv_val = float(pa.detach().item())
        # Baseline is the class prior, not 0.5, since phase labels are imbalanced.
        with torch.no_grad():
            preds = phase_logits.argmax(dim=-1)
            phase_adv_correct = int((preds == phase).sum().item())
            phase_adv_n = int(phase.numel())
            n_phases = int(phase_logits.shape[-1])
            phase_adv_class_counts = torch.bincount(
                phase, minlength=n_phases
            ).cpu().tolist()

    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning(
            "Skipping batch — loss is %s (dur=%s evt=%s)",
            loss.item(), dur.tolist(), evt.tolist(),
        )
        return (
            {"total": float("nan"), "attn_guidance": attn_guidance_val,
             "phase_adv": phase_adv_val,
             "phase_adv_correct": phase_adv_correct, "phase_adv_n": phase_adv_n,
             "phase_adv_class_counts": phase_adv_class_counts},
            out["risk"].detach().float().cpu().numpy().ravel(),
            dur.cpu().numpy().ravel(),
            evt.cpu().numpy().ravel(),
        )

    if amp_on:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return (
        {"total": float(loss.item()),
         "attn_guidance": attn_guidance_val, "phase_adv": phase_adv_val,
         "phase_adv_correct": phase_adv_correct, "phase_adv_n": phase_adv_n,
         "phase_adv_class_counts": phase_adv_class_counts},
        out["risk"].detach().float().cpu().numpy().ravel(),
        dur.cpu().numpy().ravel(),
        evt.cpu().numpy().ravel(),
    )


@torch.no_grad()
def val_step(
    model: nn.Module,
    batch: Dict[str, Any],
    criterion: Any,
    device: torch.device,
    clinical_dim: int,
    anatomy_dim: int = 0,
    vessel_dim: int = 0,
    radiomic_dim: int = 0,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Forward-only pass for one validation batch."""
    model.eval()
    x0  = batch["t0"].to(device)
    x1  = batch["t1"].to(device)
    dur = batch["duration"].to(device)
    evt = batch["event"].to(device)
    clin   = _clinical_feat(batch, clinical_dim, device)
    anat   = _anatomy_feat(batch, anatomy_dim, device)
    vessel = _vessel_feat(batch, vessel_dim, device)
    radio  = _radiomic_feat(batch, radiomic_dim, device)
    # Optional valid-region masks; when present, cross-attention ignores padded keys.
    valid_t0 = batch["valid_t0"].to(device).bool() if "valid_t0" in batch else None
    valid_t1 = batch["valid_t1"].to(device).bool() if "valid_t1" in batch else None

    roi_masks = _roi_pool_masks(model, batch, device)

    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=amp_dtype):
        out  = model(x0, x1, radiomic_features=radio,
                     clinical_features=clin, anatomy_features=anat,
                     vessel_features=vessel, roi_masks=roi_masks,
                     valid_T0=valid_t0, valid_T1=valid_t1)
        loss = criterion(risk_scores=out["risk"], durations=dur, events=evt)

    # The dataset stores the pid under "case_id"; fall back to "patient_id".
    pids = list(batch.get("case_id", batch.get("patient_id", [""] * dur.shape[0])))
    return (
        {"total": float(loss.item())},
        out["risk"].detach().float().cpu().numpy().ravel(),
        dur.cpu().numpy().ravel(),
        evt.cpu().numpy().ravel(),
        pids,
    )


# Main loop

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    cfg: Config,
    device: torch.device,
    output_dir: Path,
    clinical_dim: int,
    anatomy_dim: int = 0,
    vessel_dim: int = 0,
    radiomic_dim: int = 0,
    amp_dtype: torch.dtype = torch.float16,
    use_wandb: bool = False,
    resume_epoch: int = 0,
    best_val_metric: float = 0.0,
    clinical_baseline: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Full training loop."""
    tc = cfg.training
    num_epochs          = tc.num_epochs
    unfreeze_at         = tc.unfreeze_encoder_at_epoch
    unfreeze_stages     = tc.unfreeze_encoder_stages
    save_every          = tc.save_every_n_epochs
    patience            = tc.early_stopping_patience
    viz_every           = tc.attention_viz_every
    viz_cases           = tc.attention_viz_cases
    viz_skip_stages     = tc.attention_viz_skip_stages
    viz_case_ids        = tc.attention_viz_case_ids
    wd                  = tc.weight_decay
    task                = getattr(tc, "task", "survival")
    horizon             = getattr(tc, "survival_horizon_months", 12.0)
    metric_name         = "auc" if task == "binary" else "c_index"
    metric_tag          = "AUC" if task == "binary" else "C"
    ckpt_metric         = "auc" if task == "binary" else "cindex"

    # Attention guidance: supervises materialised cross-attention weights to
    # concentrate on the ROI. Training-only, AMP-compatible.
    attn_guidance_enabled   = getattr(tc, "attn_guidance_enabled", False)
    attn_guidance_coef = getattr(tc, "attn_guidance_coef", 0.0)
    attn_guidance_stage     = getattr(tc, "attn_guidance_stage", -1)
    attn_guidance_roi       = tuple(getattr(tc, "attn_guidance_roi", ("tumour", "peritumoural")))
    attn_guidance_roi_weights = tuple(float(w) for w in getattr(tc, "attn_guidance_roi_weights", ()))
    if attn_guidance_roi_weights and len(attn_guidance_roi_weights) != len(attn_guidance_roi):
        raise ValueError(
            f"attn_guidance_roi_weights ({len(attn_guidance_roi_weights)}) must align with "
            f"attn_guidance_roi ({len(attn_guidance_roi)}): {list(attn_guidance_roi)} vs {list(attn_guidance_roi_weights)}"
        )
    phase_adv_enabled = getattr(tc, "phase_adversarial_enabled", False)
    phase_adv_lambd   = float(getattr(tc, "phase_adv_coef", 0.0)) if phase_adv_enabled else 0.0
    if phase_adv_enabled:
        logger.info("Phase-adversarial (GRL) ON — lambd=%.4f", phase_adv_lambd)
    attn_guidance_reg = None
    if attn_guidance_enabled and attn_guidance_coef > 0:
        from pdac_longitudinal.losses import AttentionGuidanceRegularizer
        attn_guidance_reg = AttentionGuidanceRegularizer()
        logger.info(
            "attention guidance ON — attn_coef=%.2f stage(s)=%s roi=%s "
            "weights=%s (first-order, AMP)",
            attn_guidance_coef, attn_guidance_stage, list(attn_guidance_roi),
            list(attn_guidance_roi_weights) if attn_guidance_roi_weights else "uniform-union",
        )
    elif attn_guidance_enabled:
        logger.info("attention guidance enabled but attn_guidance_coef=0 — no attention supervision applied.")

    ckpt_paths = CheckpointPaths.make(output_dir, metric=ckpt_metric)

    # Sanity-check event counts; C-index requires ≥1 event per split
    train_cases = train_loader.dataset.cases  # type: ignore[union-attr]
    val_cases   = val_loader.dataset.cases    # type: ignore[union-attr]
    tr_events   = sum(c["event"] for c in train_cases)
    va_events   = sum(c["event"] for c in val_cases)
    logger.info(
        "Split sizes  train=%d (events=%d)  val=%d (events=%d)",
        len(train_cases), tr_events, len(val_cases), va_events,
    )
    if tr_events == 0:
        logger.warning("Train split has 0 events — %s will produce no gradient signal!",
                       "Cox loss" if task != "binary" else "binary loss (no positives)")
    if va_events == 0:
        logger.warning("Val split has 0 events — val %s will be NaN!", metric_tag)

    # Binary task: report usable counts after censoring exclusion (Split sizes above overstates N).
    if task == "binary":
        def _bin_counts(cases: Sequence[Dict[str, Any]]) -> Tuple[int, int, int]:
            pos = neg = exc = 0
            for c in cases:
                d, e = float(c["duration"]), float(c["event"])
                if d >= horizon:    neg += 1
                elif e > 0.5:       pos += 1
                else:               exc += 1
            return pos, neg, exc

        for name, cases in (("train", train_cases), ("val", val_cases)):
            pos, neg, exc = _bin_counts(cases)
            usable = pos + neg
            logger.info(
                "Binary @ %.0fmo  %-5s usable=%d  (died<H=%d, survived=%d)  "
                "excluded[censored<H]=%d  of %d total",
                horizon, name, usable, pos, neg, exc, len(cases),
            )
            if usable < 2 or pos == 0 or neg == 0:
                logger.warning(
                    "  %s has degenerate binary labels (pos=%d neg=%d) — "
                    "%s AUC will be NaN.", name, pos, neg, name,
                )

        # Positive-class weight. Computed from train only.
        bpw = getattr(tc, "binary_pos_weight", "auto")
        if hasattr(criterion, "set_pos_weight"):
            if isinstance(bpw, str) and bpw.lower() == "auto":
                tp, tn, _ = _bin_counts(train_cases)
                if tp > 0:
                    w = tn / tp
                    criterion.set_pos_weight(w)
                    logger.info("Binary pos_weight (auto) = %.2f  (neg/pos = %d/%d)",
                                w, tn, tp)
            elif bpw is None or (isinstance(bpw, str) and bpw.lower() in ("none", "null", "")):
                criterion.set_pos_weight(None)
                logger.info("Binary pos_weight: disabled (unweighted BCE)")
            else:
                criterion.set_pos_weight(float(bpw))
                logger.info("Binary pos_weight (fixed) = %.2f", float(bpw))

    best_val_loss   = math.inf
    no_improve_count = 0

    for epoch in range(resume_epoch + 1, num_epochs + 1):

        # Unfreeze the encoder. Full unfreeze overfits and OOMs on small
        # cohorts, so unfreeze_encoder_stages restricts it to a few deep
        # stages instead, keeping the shallow ones frozen and grad-free.
        if unfreeze_at > 0 and epoch == unfreeze_at:
            if unfreeze_stages:
                model.encoder.unfreeze_stages(unfreeze_stages)
            else:
                logger.info("Unfreezing FULL encoder at epoch %d", epoch)
                model.encoder.unfreeze_encoder()
            # Separate param group so the encoder gets a lower LR than the head.
            current_lr = optimizer.param_groups[0]["lr"]
            encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
            encoder_ids = {id(p) for p in encoder_params}
            head_params = [
                p for p in model.parameters()
                if p.requires_grad and id(p) not in encoder_ids
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": head_params,    "lr": current_lr},
                    {"params": encoder_params, "lr": current_lr * 0.1},
                ],
                weight_decay=wd,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - epoch,
            )

        # Train
        tr_metrics: Dict[str, float] = {"total": 0.0}
        if attn_guidance_reg is not None:
            tr_metrics["attn_guidance"] = 0.0
        if phase_adv_lambd > 0:
            tr_metrics["phase_adv"] = 0.0
        tr_risk, tr_dur, tr_evt = [], [], []
        t_start = time.time()
        # Summed counts, not a mean of per-batch accuracies, so uneven final
        # batches don't bias it.
        pa_correct_sum = 0
        pa_n_sum = 0
        pa_class_counts_sum: Optional[List[int]] = None

        for batch in train_loader:
            m, r, d, e = train_step(
                model, batch, criterion, optimizer, device, scaler,
                clinical_dim, anatomy_dim, vessel_dim, radiomic_dim,
                amp_dtype=amp_dtype,
                attn_guidance_reg=attn_guidance_reg,
                attn_guidance_coef=attn_guidance_coef, attn_guidance_stage=attn_guidance_stage,
                attn_guidance_roi=attn_guidance_roi, attn_guidance_roi_weights=attn_guidance_roi_weights,
                phase_adv_lambd=phase_adv_lambd,
            )
            for k in tr_metrics:
                tr_metrics[k] += m.get(k, 0.0)
            if phase_adv_lambd > 0:
                pa_correct_sum += int(m.get("phase_adv_correct", 0) or 0)
                pa_n_sum += int(m.get("phase_adv_n", 0) or 0)
                cc = m.get("phase_adv_class_counts")
                if cc is not None:
                    if pa_class_counts_sum is None:
                        pa_class_counts_sum = list(cc)
                    else:
                        pa_class_counts_sum = [
                            a + b for a, b in zip(pa_class_counts_sum, cc)
                        ]
            tr_risk.append(r)
            tr_dur.append(d)
            tr_evt.append(e)

        nb = max(len(train_loader), 1)
        tr_metrics = {k: v / nb for k, v in tr_metrics.items()}
        # Discriminator accuracy vs. the class-prior baseline, added after the
        # per-batch mean so it isn't divided by nb.
        if phase_adv_lambd > 0 and pa_n_sum > 0:
            disc_acc = pa_correct_sum / pa_n_sum
            disc_baseline = (
                max(pa_class_counts_sum) / pa_n_sum
                if pa_class_counts_sum else float("nan")
            )
            tr_metrics["phase_adv_acc"] = disc_acc
            tr_metrics["phase_adv_baseline"] = disc_baseline
            tr_metrics["phase_adv_acc_above_baseline"] = disc_acc - disc_baseline
        elapsed = time.time() - t_start

        try:
            tr_metric = epoch_metric(
                task,
                np.concatenate(tr_risk),
                np.concatenate(tr_dur),
                np.concatenate(tr_evt),
                horizon,
            )
        except Exception as exc:
            logger.warning("train %s failed: %s", metric_tag, exc)
            tr_metric = float("nan")

        # Val
        va_metrics: Dict[str, float] = {"total": 0.0}
        va_risk, va_dur, va_evt = [], [], []
        va_pids: List[str] = []

        for batch in val_loader:
            m, r, d, e, p = val_step(
                model, batch, criterion, device,
                clinical_dim, anatomy_dim, vessel_dim, radiomic_dim,
                amp_dtype=amp_dtype,
            )
            for k in va_metrics:
                va_metrics[k] += m.get(k, 0.0)
            va_risk.append(r)
            va_dur.append(d)
            va_evt.append(e)
            va_pids.extend(p)

        nvb = max(len(val_loader), 1)
        va_metrics = {k: v / nvb for k, v in va_metrics.items()}

        va_risk_all = np.concatenate(va_risk) if va_risk else np.zeros(0)
        va_dur_all  = np.concatenate(va_dur)  if va_dur  else np.zeros(0)
        va_evt_all  = np.concatenate(va_evt)  if va_evt  else np.zeros(0)

        try:
            va_metric = epoch_metric(task, va_risk_all, va_dur_all, va_evt_all, horizon)
        except Exception as exc:
            logger.warning("val %s failed: %s", metric_tag, exc)
            va_metric = float("nan")

        # Per-stratum val metric (cohort x phase); skipped below 5 patients or
        # 0 events, too noisy otherwise.
        per_stratum_metric: Dict[str, float] = {}
        try:
            ds = val_loader.dataset
            reg = getattr(ds, "registry", None)
            for strat_name, key_fn in (
                ("cohort", lambda p: reg.get_cohort(p) if reg is not None else "unknown"),
                ("phase",  lambda p: ds.get_phase(p) if hasattr(ds, "get_phase") else "unknown"),
            ):
                buckets: Dict[str, List[int]] = {}
                for i, p in enumerate(va_pids):
                    try:
                        buckets.setdefault(str(key_fn(p)) or "unknown", []).append(i)
                    except Exception:
                        buckets.setdefault("unknown", []).append(i)
                for label, idx in buckets.items():
                    if len(idx) < 5:
                        continue
                    e_sub = va_evt_all[idx]
                    if int(e_sub.sum()) < 1:
                        continue
                    try:
                        metric_sub = epoch_metric(
                            task,
                            va_risk_all[idx], va_dur_all[idx], e_sub,
                            horizon,
                        )
                        per_stratum_metric[f"{strat_name}/{label}"] = float(metric_sub)
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("per-stratum metric failed: %s", exc)

        scheduler.step()

        logger.info(
            "Epoch %3d/%d  train[loss=%.4f %s=%.3f]  val[loss=%.4f %s=%.3f]  %.1fs",
            epoch, num_epochs,
            tr_metrics["total"], metric_tag, tr_metric,
            va_metrics["total"], metric_tag, va_metric,
            elapsed,
        )
        if per_stratum_metric:
            logger.info(
                "  val %s by stratum:  %s",
                metric_tag,
                "  ".join(
                    f"{k}={v:.3f}" for k, v in sorted(per_stratum_metric.items())
                ),
            )
        if phase_adv_lambd > 0 and "phase_adv_acc" in tr_metrics:
            _gap = tr_metrics["phase_adv_acc_above_baseline"]
            logger.info(
                "  GRL phase-discriminator: acc=%.3f  prior-baseline=%.3f  "
                "Δ=%+.3f  → %s",
                tr_metrics["phase_adv_acc"],
                tr_metrics["phase_adv_baseline"],
                _gap,
                "at chance, phase scrubbed (GRL working)"
                if _gap <= 0.02 else "phase still leaking (GRL not winning)",
            )

        # Attention viz
        viz_paths: List[Path] = []
        if viz_every > 0 and (epoch == 1 or epoch % viz_every == 0):
            from pdac_longitudinal.visualisation.attention_viz import render_attention_maps
            viz_dir = output_dir / "attention_viz" / f"epoch{epoch:04d}"
            try:
                viz_paths = render_attention_maps(
                    model, val_loader, device,
                    clinical_dim=clinical_dim,
                    out_dir=viz_dir,
                    n_cases=viz_cases,
                    anatomy_dim=anatomy_dim,
                    vessel_dim=vessel_dim,
                    skip_attn_stages=viz_skip_stages,
                    case_ids=viz_case_ids,
                )
                if viz_paths:
                    logger.info("Attention overlays → %s", viz_dir)
            except Exception as exc:
                logger.warning("Attention viz failed epoch %d (%s)", epoch, exc)

        # W&B logging
        if use_wandb:
            from pdac_longitudinal.training.wandb_setup import (
                log_attention_images,
                log_metrics,
            )
            log_d: Dict[str, float] = {f"train/{k}": v for k, v in tr_metrics.items()}
            log_d.update({f"val/{k}": v for k, v in va_metrics.items()})
            log_d.update({
                f"train/{metric_name}": tr_metric,
                f"val/{metric_name}":   va_metric,
                "epoch":                float(epoch),
                "lr":                   optimizer.param_groups[0]["lr"],
            })
            # Per-stratum val metrics, for cohort/phase transfer diagnostics.
            for k, v in per_stratum_metric.items():
                log_d[f"val/{metric_name}/{k}"] = v
            # Logged every epoch so it draws as a flat reference line.
            if clinical_baseline is not None:
                for src_k, dst_k in (
                    ("best_val_c",    f"clinical_baseline/val_{metric_name}"),
                    ("best_train_c",  f"clinical_baseline/train_{metric_name}"),
                    ("best_test_c",   f"clinical_baseline/test_{metric_name}"),
                    ("best_val_auc",  "clinical_baseline/val_auc"),
                    ("best_test_auc", "clinical_baseline/test_auc"),
                ):
                    v = clinical_baseline.get(src_k)
                    if v is not None and v == v:    # filter NaN
                        log_d[dst_k] = float(v)
            log_metrics(log_d, step=epoch)
            if viz_paths:
                log_attention_images(viz_paths, step=epoch)

        # Checkpointing
        is_best_metric = not math.isnan(va_metric) and va_metric > best_val_metric
        is_best_loss   = va_metrics["total"] < best_val_loss

        if is_best_metric:
            best_val_metric = va_metric
            no_improve_count = 0
            save_checkpoint(
                ckpt_paths.best_metric,
                model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler,
                epoch=epoch,
                extra={
                    "val_metric": va_metric, "val_loss": va_metrics["total"],
                    "train_metric": tr_metric, "clinical_dim": clinical_dim,
                    "anatomy_dim": anatomy_dim, "vessel_dim": vessel_dim,
                    "radiomic_dim": radiomic_dim,
                },
            )
            # Per-patient val predictions, for a pooled cross-validated concordance.
            try:
                import csv as _csv
                with open(output_dir / "val_predictions.csv", "w", newline="") as _fh:
                    _w = _csv.writer(_fh)
                    _w.writerow(["patient_id", "risk", "duration", "event", "epoch"])
                    for _p, _r, _d, _e in zip(
                        va_pids, va_risk_all, va_dur_all, va_evt_all
                    ):
                        _w.writerow([_p, float(_r), float(_d), int(_e), epoch])
            except Exception as _exc:
                logger.debug("Could not write val_predictions.csv: %s", _exc)
        else:
            if not math.isnan(va_metric):
                no_improve_count += 1

        if is_best_loss:
            best_val_loss = va_metrics["total"]

        if epoch % save_every == 0:
            save_checkpoint(
                ckpt_paths.latest,
                model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler,
                epoch=epoch,
                extra={
                    "val_metric": va_metric, "val_loss": va_metrics["total"],
                    "best_val_metric": best_val_metric,
                    "clinical_dim": clinical_dim, "anatomy_dim": anatomy_dim,
                    "vessel_dim": vessel_dim, "radiomic_dim": radiomic_dim,
                },
            )

        # Early stopping
        if patience > 0 and no_improve_count >= patience:
            logger.info(
                "Early stopping at epoch %d — no val %s improvement for %d epochs.",
                epoch, metric_tag, patience,
            )
            break

    logger.info(
        "Training complete.  Best val %s: %.3f",
        metric_tag, best_val_metric,
    )

    return {
        "best_val_metric":  best_val_metric,
        "best_checkpoint":  ckpt_paths.best_metric,
        "latest_checkpoint": ckpt_paths.latest,
        "final_epoch":      epoch,
        "n_train":          len(train_cases),
        "n_val":            len(val_cases),
    }
