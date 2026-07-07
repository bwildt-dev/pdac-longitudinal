"""Visualisation of cross-timepoint attention overlaid on CT compartments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# high-contrast colours against the hot heatmap
COMPARTMENT_COLOURS: Dict[str, str] = {
    "mask_pt3": "#4fc3f7",
    "mask_pt2": "#29b6f6",
    "mask_pt1": "#0288d1",
    "mask_tvi": "#d500f9",
    "mask_it":     "#00e676",
    "mask_it_t1":  "#00e676",
    "mask_pt3_t1": "#4fc3f7",
    "mask_pt2_t1": "#29b6f6",
    "mask_pt1_t1": "#0288d1",
    "mask_tvi_t1": "#d500f9",
    "liver_t0":    "#ffb74d",
    "liver_t1":    "#ffb74d",
    "pancreas_t0": "#ff7043",
    "pancreas_t1": "#ff7043",
    "kidneys_t0":  "#9ccc65",
    "kidneys_t1":  "#9ccc65",
}

COMPARTMENT_DISPLAY_NAMES: Dict[str, str] = {
    "mask_it":     "IT (T0)",
    "mask_it_t1":  "IT (T1)",
    "mask_pt1":    "PT ring 1 (0–5 mm, T0)",
    "mask_pt2":    "PT ring 2 (5–10 mm, T0)",
    "mask_pt3":    "PT ring 3 (10–15 mm, T0)",
    "mask_tvi":    "TVI (T0)",
    "mask_pt1_t1": "PT ring 1 (0–5 mm, T1)",
    "mask_pt2_t1": "PT ring 2 (5–10 mm, T1)",
    "mask_pt3_t1": "PT ring 3 (10–15 mm, T1)",
    "mask_tvi_t1": "TVI (T1)",
    "liver_t0":    "Liver (T0)",
    "liver_t1":    "Liver (T1)",
    "pancreas_t0": "Pancreas (T0)",
    "pancreas_t1": "Pancreas (T1)",
    "kidneys_t0":  "Kidneys (T0)",
    "kidneys_t1":  "Kidneys (T1)",
}


def upsample_to_patch(
    attn: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Trilinearly upsample `(B, 1, D_s, H_s, W_s)` to `(B, 1, D, H, W)`.

    Args:
        attn: Attention map, shape `(B, 1, D_s, H_s, W_s)`.
        target_shape: Target spatial shape `(D, H, W)`.

    Returns:
        Upsampled attention map, shape `(B, 1, D, H, W)`; returned unchanged
        if already at `target_shape`.

    Raises:
        ValueError: If `attn` is not 5-D.
    """
    if attn.ndim != 5:
        raise ValueError(f"Expected 5-D attention (B,1,D,H,W), got shape {tuple(attn.shape)}")
    if tuple(attn.shape[-3:]) == tuple(target_shape):
        return attn
    return F.interpolate(
        attn.float(), size=tuple(target_shape), mode="trilinear", align_corners=False
    )


def _to_numpy_3d(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a tensor to a bare 3-D numpy array.

    Args:
        x: Tensor or array with a trailing 3-D volume.

    Returns:
        Bare 3-D numpy array, shape `(D, H, W)`.

    Raises:
        ValueError: If squeezing leading dims doesn't leave a 3-D array.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x)
    while x.ndim > 3:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected a 3-D volume after squeezing, got shape {x.shape}")
    return x


def _mask_to_valid(
    attn: Union[torch.Tensor, np.ndarray],
    valid: Optional[Union[torch.Tensor, np.ndarray]],
) -> np.ndarray:
    """Zero attention outside the valid region.

    Args:
        attn: Attention volume.
        valid: Optional valid mask; nearest-resized to `attn`'s shape if it
            differs. `None` is a no-op.

    Returns:
        Attention volume with values outside `valid` set to 0.
    """
    a = _to_numpy_3d(attn).astype(np.float32)
    if valid is None:
        return a
    v = _to_numpy_3d(valid)
    if v.shape != a.shape:
        # Valid mask at a different resolution; nearest-resize via index scaling.
        import torch.nn.functional as F
        vt = torch.as_tensor(v, dtype=torch.float32)[None, None]
        v = F.interpolate(vt, size=a.shape, mode="nearest")[0, 0].numpy()
    a[v <= 0.5] = 0.0
    return a


def aggregate_stage_attention(
    attn_maps_per_stage: Sequence[Mapping[str, torch.Tensor]],
    target_shape: Tuple[int, int, int],
    direction: str = "T1_to_T0",
    weights: Optional[Sequence[float]] = None,
    sample_index: int = 0,
    skip_stages: Sequence[int] = (),
) -> torch.Tensor:
    """Per-stage min-max normalise, upsample, then weighted-average to `(D,H,W)` in [0,1].

    Args:
        attn_maps_per_stage: One attention-map dict per stage (as returned by
            `CrossTimepointAttentionStack.forward`).
        target_shape: Target spatial shape `(D, H, W)` to upsample each
            stage's map to before averaging.
        direction: Attention direction, `'T1_to_T0'` or `'T0_to_T1'`.
        weights: Optional per-stage weight; uniform when `None`. Stages that
            are skipped or lack `direction` are excluded and the remaining
            weights renormalised.
        sample_index: Index into the batch dimension to extract.
        skip_stages: Stage indices to exclude from aggregation.

    Returns:
        Aggregated attention volume in `[0, 1]`, shape `(D, H, W)`.

    Raises:
        ValueError: If `attn_maps_per_stage` is empty, `weights` has the
            wrong length, no stage is usable for `direction`, or the usable
            weights sum to `<= 0`.
    """
    if not attn_maps_per_stage:
        raise ValueError("attn_maps_per_stage is empty")

    n = len(attn_maps_per_stage)
    if weights is None:
        raw_weights = [1.0] * n
    else:
        if len(weights) != n:
            raise ValueError(f"weights has {len(weights)} entries, expected {n}")
        raw_weights = [float(x) for x in weights]

    skip_set = set(skip_stages)
    usable = [
        (i, attn_maps_per_stage[i], raw_weights[i])
        for i in range(n)
        if i not in skip_set and direction in attn_maps_per_stage[i]
    ]
    if not usable:
        raise ValueError(
            f"No usable attention stages for direction {direction!r} "
            f"(n={n}, skip_stages={sorted(skip_set)})"
        )
    total = sum(w for _, _, w in usable)
    if total <= 0:
        raise ValueError("usable stage weights must sum to > 0")

    agg: Optional[torch.Tensor] = None
    for _stage_idx, stage_maps, raw_w in usable:
        weight = raw_w / total
        a = stage_maps[direction]
        a = a[sample_index : sample_index + 1]
        a = upsample_to_patch(a, target_shape)
        a = a[0, 0]

        a_min = a.min()
        a_max = a.max()
        if (a_max - a_min) > 1e-8:
            a = (a - a_min) / (a_max - a_min)
        else:
            a = torch.zeros_like(a)

        agg = a * weight if agg is None else agg + a * weight

    assert agg is not None
    return agg.clamp(0.0, 1.0)


def compartment_stats(
    attn_volume: Union[torch.Tensor, np.ndarray],
    masks: Mapping[str, Union[torch.Tensor, np.ndarray]],
) -> Dict[str, Dict[str, float]]:
    """Per-compartment attention descriptors including attention_lift = mass_frac / vol_frac.

    Args:
        attn_volume: Attention volume, shape `(D, H, W)`.
        masks: Maps compartment name to a bool mask of the same shape.

    Returns:
        Dict mapping compartment name to a dict of `voxels`,
        `volume_fraction`, `mean_attention`, `peak_attention`,
        `attention_mass_fraction`, and `attention_lift`.

    Raises:
        ValueError: If a mask's shape doesn't match `attn_volume`'s.
    """
    attn = _to_numpy_3d(attn_volume).astype(np.float32)
    total_mass = float(attn.sum())
    total_voxels = int(attn.size)

    out: Dict[str, Dict[str, float]] = {}
    for name, m in masks.items():
        mask = _to_numpy_3d(m).astype(bool)
        if mask.shape != attn.shape:
            raise ValueError(
                f"Mask {name!r} shape {mask.shape} does not match attention "
                f"shape {attn.shape}. Upsample attention first."
            )
        vox = int(mask.sum())
        if vox == 0:
            out[name] = {
                "voxels": 0,
                "volume_fraction": 0.0,
                "mean_attention": 0.0,
                "peak_attention": 0.0,
                "attention_mass_fraction": 0.0,
                "attention_lift": 0.0,
            }
            continue
        region = attn[mask]
        mass = float(region.sum())
        vol_frac = float(vox / total_voxels)
        mass_frac = float(mass / total_mass) if total_mass > 0 else 0.0
        lift = float(mass_frac / vol_frac) if vol_frac > 0 else 0.0
        out[name] = {
            "voxels": vox,
            "volume_fraction": vol_frac,
            "mean_attention": float(region.mean()),
            "peak_attention": float(region.max()),
            "attention_mass_fraction": mass_frac,
            "attention_lift": lift,
        }
    return out


def _pick_center_slices(
    mask_it: np.ndarray,
) -> Tuple[int, int, int]:
    """Return tumour-centred slice indices as `(i0, i1, i2)` in array-axis order.

    Args:
        mask_it: Intra-tumoural mask, shape `(D, H, W)`. Falls back to the
            volume centre when empty.
    """
    n0, n1, n2 = mask_it.shape
    if mask_it.any():
        a0, a1, a2 = np.where(mask_it)
        return int(np.median(a0)), int(np.median(a1)), int(np.median(a2))
    return n0 // 2, n1 // 2, n2 // 2


def _draw_compartment_contours(ax, masks_by_slice: Dict[str, np.ndarray]) -> List:
    """Draw compartment outlines on a 2-D matplotlib axis.

    Args:
        ax: Matplotlib axis to draw on.
        masks_by_slice: Maps compartment name to a 2-D bool slice.

    Returns:
        Legend handles, one per compartment drawn.
    """
    import matplotlib.patches as mpatches

    handles: List = []

    def _is_it(name: str) -> bool:
        return name in ("mask_it", "mask_it_t1")

    items = sorted(masks_by_slice.items(), key=lambda kv: _is_it(kv[0]))
    for name, slc in items:
        if not slc.any():
            continue
        colour = COMPARTMENT_COLOURS.get(name, "#ffffff")
        it = _is_it(name)
        ax.contour(
            slc.astype(float), levels=[0.5], colors=colour,
            linewidths=2.0 if it else 1.2,
            zorder=5 if it else 3,
        )
        handles.append(
            mpatches.Patch(color=colour, label=COMPARTMENT_DISPLAY_NAMES.get(name, name))
        )
    return handles


def render_compartment_overlay(
    ct: Union[torch.Tensor, np.ndarray],
    attn: Union[torch.Tensor, np.ndarray],
    masks: Mapping[str, Union[torch.Tensor, np.ndarray]],
    out_path: Union[str, Path],
    case_id: str = "",
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    attn_alpha: float = 0.60,
    attn_percentile: float = 90.0,
    dpi: int = 140,
) -> Path:
    """Render axial/coronal/sagittal panels of CT + attention heatmap + compartment contours.

    Args:
        ct: CT volume, shape `(D, H, W)`.
        attn: Attention volume, same shape as `ct`.
        masks: Maps compartment name to a bool mask, same shape as `ct`.
        out_path: PNG output path; parent directories are created.
        case_id: Optional case identifier shown in the figure title.
        stats: Optional per-compartment stats (from `compartment_stats`) to
            render as a table under the panels.
        attn_alpha: Opacity of the attention heatmap overlay.
        attn_percentile: Percentile used as the hotspot display threshold.
        dpi: Output image resolution.

    Returns:
        `out_path`.

    Raises:
        ValueError: If `attn`'s shape doesn't match `ct`'s.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    ct_np = _to_numpy_3d(ct)
    attn_np = _to_numpy_3d(attn)
    if attn_np.shape != ct_np.shape:
        raise ValueError(
            f"Attention shape {attn_np.shape} does not match CT shape {ct_np.shape}; "
            "upsample before calling render_compartment_overlay()."
        )

    masks_np = {name: _to_numpy_3d(m).astype(bool) for name, m in masks.items()}
    mask_it = masks_np.get("mask_it", np.zeros_like(ct_np, dtype=bool))
    # (i0, i1, i2) = centroids along array axes 0/1/2 = (x, y, z).
    i0, i1, i2 = _pick_center_slices(mask_it)

    lo, hi = np.percentile(ct_np, [1, 99])
    ct_display = np.clip((ct_np - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    med = float(np.median(attn_np))
    cut = float(np.percentile(attn_np, attn_percentile))
    peak = float(attn_np.max())
    above_median = np.clip(attn_np - med, 0.0, None)
    scale = max(peak - cut, 1e-6)
    hotspot = np.clip((attn_np - cut) / scale, 0.0, 1.0)
    visible = (attn_np > cut) & (above_median > 0)
    attn_display = np.ma.array(hotspot, mask=~visible)

    panels: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, np.ndarray], int]] = [
        ("axial",    ct_display[:, :, i2], attn_display[:, :, i2],
         {k: v[:, :, i2] for k, v in masks_np.items()}, i2),
        ("coronal",  ct_display[:, i1, :], attn_display[:, i1, :],
         {k: v[:, i1, :] for k, v in masks_np.items()}, i1),
        ("sagittal", ct_display[i0, :, :], attn_display[i0, :, :],
         {k: v[i0, :, :] for k, v in masks_np.items()}, i0),
    ]

    fig_h = 5.2 if stats is None else 6.6
    fig, axes = plt.subplots(1, 3, figsize=(13, fig_h))
    legend_handles: List = []
    for ax, (view_name, ct_slc, attn_slc, mask_slc, idx) in zip(axes, panels):
        ax.imshow(ct_slc, cmap="gray", origin="lower", interpolation="nearest")
        ax.imshow(
            attn_slc, cmap="hot", origin="lower",
            interpolation="bilinear",
            alpha=attn_alpha, vmin=0.0, vmax=1.0,
        )
        handles = _draw_compartment_contours(ax, mask_slc)
        if not legend_handles and handles:
            legend_handles = handles
        ax.set_title(f"{view_name}  (idx={idx})")
        ax.set_xticks([])
        ax.set_yticks([])

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.02 if stats is None else 0.30),
        )

    title = "Cross-timepoint attention  ·  compartment overlay"
    if case_id:
        title = f"{title}  ·  {case_id}"
    fig.suptitle(title, fontsize=12)

    if stats is not None:
        _draw_stats_table(fig, stats)
    else:
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out_path


def _draw_stats_table(
    fig,
    stats: Dict[str, Dict[str, float]],
    rect: Tuple[float, float, float, float] = (0.08, 0.05, 0.84, 0.20),
    title_text: Optional[str] = None,
) -> None:
    """Draw a small per-compartment stats table under the image panels.

    Args:
        fig: Matplotlib figure to add the table axis to.
        stats: Per-compartment stats, as returned by `compartment_stats`.
        rect: Table axis position `(left, bottom, width, height)` in figure
            fraction coordinates.
        title_text: Optional title drawn above the table.
    """
    col_labels = ["compartment", "vol frac", "mass frac", "lift", "mean", "peak"]
    order = [
        "mask_it", "mask_pt1", "mask_pt2", "mask_pt3", "mask_tvi",
        "mask_it_t1", "mask_pt1_t1", "mask_pt2_t1", "mask_pt3_t1", "mask_tvi_t1",
    ]
    rows = []
    for key in order:
        if key not in stats:
            continue
        s = stats[key]
        lift = s.get("attention_lift", 0.0)

        lift_str = f"{lift:.2f}×" + (" ↑" if lift > 1.1 else (" ↓" if lift < 0.9 else "  "))
        rows.append([
            COMPARTMENT_DISPLAY_NAMES.get(key, key),
            f"{s.get('volume_fraction', 0.0):.2%}",
            f"{s['attention_mass_fraction']:.2%}",
            lift_str,
            f"{s['mean_attention']:.3f}",
            f"{s['peak_attention']:.3f}",
        ])
    if not rows:
        return

    ax = fig.add_axes(list(rect))
    ax.axis("off")
    if title_text:
        ax.set_title(title_text, fontsize=9, pad=4)
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)


def _prep_attn_display(
    attn_np: np.ndarray, attn_percentile: float
) -> Tuple[np.ma.MaskedArray, float, float, float]:
    """Threshold + masked-array prep shared by single/dual renderers.

    Args:
        attn_np: Attention volume.
        attn_percentile: Percentile used as the hotspot display threshold.

    Returns:
        Tuple of the masked hotspot array (values below the threshold
        masked out), the median, the percentile cutoff, and the peak value.
    """
    med  = float(np.median(attn_np))
    cut  = float(np.percentile(attn_np, attn_percentile))
    peak = float(attn_np.max())
    above_median = np.clip(attn_np - med, 0.0, None)
    scale = max(peak - cut, 1e-6)
    hotspot = np.clip((attn_np - cut) / scale, 0.0, 1.0)
    visible = (attn_np > cut) & (above_median > 0)
    return np.ma.array(hotspot, mask=~visible), med, cut, peak


def render_dual_overlay(
    ct_T0: Union[torch.Tensor, np.ndarray],
    ct_T1: Union[torch.Tensor, np.ndarray],
    attn_T1_to_T0: Union[torch.Tensor, np.ndarray],
    attn_T0_to_T1: Union[torch.Tensor, np.ndarray],
    masks_T0: Mapping[str, Union[torch.Tensor, np.ndarray]],
    masks_T1: Mapping[str, Union[torch.Tensor, np.ndarray]],
    out_path: Union[str, Path],
    case_id: str = "",
    attn_alpha: float = 0.60,
    attn_percentile: float = 90.0,
    rot90: int = 3,
    dpi: int = 140,
) -> Path:
    """Render a 2×3 panel of both timepoints with their respective attention maps.

    Args:
        ct_T0: T0 CT volume, shape `(D, H, W)`.
        ct_T1: T1 CT volume, shape `(D, H, W)`.
        attn_T1_to_T0: T1->T0 attention volume, same shape as `ct_T0`.
        attn_T0_to_T1: T0->T1 attention volume, same shape as `ct_T1`.
        masks_T0: Maps compartment name to a bool mask for T0.
        masks_T1: Maps compartment name to a bool mask for T1.
        out_path: PNG output path; parent directories are created.
        case_id: Optional case identifier shown in the figure title.
        attn_alpha: Opacity of the attention heatmap overlay.
        attn_percentile: Percentile used as the hotspot display threshold.
        rot90: Number of 90-degree rotations applied to axial panels to
            match radiological display convention.
        dpi: Output image resolution.

    Returns:
        `out_path`.

    Raises:
        ValueError: If an attention volume's shape doesn't match its CT.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    ct0_np = _to_numpy_3d(ct_T0)
    ct1_np = _to_numpy_3d(ct_T1)
    a01_np = _to_numpy_3d(attn_T1_to_T0)
    a10_np = _to_numpy_3d(attn_T0_to_T1)
    for label, arr, ref in (
        ("attn_T1_to_T0", a01_np, ct0_np),
        ("attn_T0_to_T1", a10_np, ct1_np),
    ):
        if arr.shape != ref.shape:
            raise ValueError(
                f"{label} shape {arr.shape} does not match CT shape {ref.shape}; "
                "upsample before calling render_dual_overlay()."
            )

    m0 = {name: _to_numpy_3d(m).astype(bool) for name, m in masks_T0.items()}
    m1 = {name: _to_numpy_3d(m).astype(bool) for name, m in masks_T1.items()}

    c0 = _pick_center_slices(m0.get("mask_it", np.zeros_like(ct0_np, dtype=bool)))
    c1 = _pick_center_slices(m1.get("mask_it_t1", np.zeros_like(ct1_np, dtype=bool)))

    def _ct_display(ct_np: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(ct_np, [1, 99])
        return np.clip((ct_np - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    ct0_disp = _ct_display(ct0_np)
    ct1_disp = _ct_display(ct1_np)
    a01_disp, *_ = _prep_attn_display(a01_np, attn_percentile)
    a10_disp, *_ = _prep_attn_display(a10_np, attn_percentile)

    rows = [
        ("T0  ·  T1→T0 attention", ct0_disp, a01_disp, m0, c0),
        ("T1  ·  T0→T1 attention", ct1_disp, a10_disp, m1, c1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    legend_handles: List = []
    seen_labels: set = set()

    for row_idx, (row_label, ct_disp, attn_disp, mask_dict, (i0, i1, i2)) in enumerate(rows):
        panels = [
            ("axial",    ct_disp[:, :, i2], attn_disp[:, :, i2],
             {k: v[:, :, i2] for k, v in mask_dict.items()}, i2),
            ("coronal",  ct_disp[:, i1, :], attn_disp[:, i1, :],
             {k: v[:, i1, :] for k, v in mask_dict.items()}, i1),
            ("sagittal", ct_disp[i0, :, :], attn_disp[i0, :, :],
             {k: v[i0, :, :] for k, v in mask_dict.items()}, i0),
        ]
        for col_idx, (view, ct_slc, attn_slc, mask_slc, idx) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            # Axial panel rotated + L-R flipped to match radiological convention.
            if view == "axial":
                _o = lambda a: np.fliplr(np.rot90(a, rot90) if rot90 else a)
                ct_slc = _o(ct_slc)
                attn_slc = _o(attn_slc)
                mask_slc = {k: _o(v) for k, v in mask_slc.items()}
            ax.imshow(ct_slc, cmap="gray", origin="lower", interpolation="nearest")
            ax.imshow(
                attn_slc, cmap="hot", origin="lower",
                interpolation="bilinear",
                alpha=attn_alpha, vmin=0.0, vmax=1.0,
            )
            handles = _draw_compartment_contours(ax, mask_slc)
            # Accumulate unique legend entries across both rows.
            for h in handles:
                lbl = h.get_label()
                if lbl not in seen_labels:
                    seen_labels.add(lbl)
                    legend_handles.append(h)
            title = f"{view}  (idx={idx})"
            if col_idx == 0:
                title = f"{row_label}\n{title}"
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 5),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.01),
        )

    title = "Cross-timepoint attention  ·  bidirectional overlay"
    if case_id:
        title = f"{title}  ·  {case_id}"
    fig.suptitle(title, fontsize=12)

    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out_path


def render_stage_montage(
    ct: Union[torch.Tensor, np.ndarray],
    masks: Mapping[str, Union[torch.Tensor, np.ndarray]],
    attn_maps_per_stage: Sequence[Mapping[str, torch.Tensor]],
    out_path: Union[str, Path],
    direction: str = "T1_to_T0",
    case_id: str = "",
    view: str = "axial",
    sample_index: int = 0,
    skip_stages: Sequence[int] = (),
    valid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    attn_alpha: float = 0.60,
    attn_percentile: float = 90.0,
    rot90: int = 3,
    scale_modes: Sequence[str] = ("per-panel", "shared"),
    dpi: int = 150,
) -> Path:
    """One-row montage: each stage at native resolution + the combined map.

    Args:
        ct: CT volume, shape `(D, H, W)`.
        masks: Maps compartment name to a bool mask, same shape as `ct`.
        attn_maps_per_stage: One attention-map dict per stage.
        out_path: PNG output path; parent directories are created.
        direction: Attention direction, `'T1_to_T0'` or `'T0_to_T1'`.
        case_id: Optional case identifier shown in the figure title.
        view: Anatomical view to render: `'axial'`, `'coronal'`, or
            `'sagittal'`.
        sample_index: Index into the batch dimension to extract.
        skip_stages: Stage indices to exclude from the montage.
        valid: Optional valid mask used to zero out padded regions.
        attn_alpha: Opacity of the attention heatmap overlay.
        attn_percentile: Percentile used as the hotspot display threshold.
        rot90: Number of 90-degree rotations applied to the axial view to
            match radiological display convention.
        scale_modes: Row(s) to render: `'per-panel'` (each stage min-maxed
            independently) and/or `'shared'` (one threshold/scale for all panels).
        dpi: Output image resolution.

    Returns:
        `out_path`.

    Raises:
        ValueError: If no stage has a usable map for `direction`.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    ct_np = _to_numpy_3d(ct)
    masks_np = {n: _to_numpy_3d(m).astype(bool) for n, m in masks.items()}
    it_key = "mask_it" if "mask_it" in masks_np else "mask_it_t1"
    i0, i1, i2 = _pick_center_slices(masks_np.get(it_key, np.zeros_like(ct_np, dtype=bool)))

    skip = set(skip_stages)
    usable = [
        (i, m[direction]) for i, m in enumerate(attn_maps_per_stage)
        if i not in skip and m and direction in m and m[direction] is not None
    ]
    if not usable:
        raise ValueError(f"No usable stages for direction {direction!r}")

    def _nn_upsample(a: torch.Tensor) -> np.ndarray:
        a = a[sample_index : sample_index + 1].float()
        a = F.interpolate(a, size=tuple(ct_np.shape), mode="nearest")[0, 0]
        v = a.detach().cpu().numpy()
        v = _mask_to_valid(v, valid)
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(v)

    stage_vols = [(idx, _nn_upsample(a)) for idx, a in usable]
    combined = np.mean([v for _, v in stage_vols], axis=0)
    columns: List[Tuple[str, np.ndarray]] = (
        [(f"stage {idx}", v) for idx, v in stage_vols] + [("combined", combined)]
    )

    sl = {"axial": np.s_[:, :, i2], "coronal": np.s_[:, i1, :], "sagittal": np.s_[i0, :, :]}[view]
    idx_shown = {"axial": i2, "coronal": i1, "sagittal": i0}[view]

    rot = rot90 if view == "axial" else 0
    flip_lr = view == "axial"
    def _orient(a):
        if rot:
            a = np.rot90(a, rot)
        if flip_lr:
            a = np.fliplr(a)
        return a

    lo, hi = np.percentile(ct_np, [1, 99])
    ct_slc = _orient(np.clip((ct_np[sl] - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    mask_slc = {k: _orient(v[sl]) for k, v in masks_np.items()}

    def _disp_for(mode: str):
        if mode == "shared":
            cut = float(np.percentile(combined, attn_percentile))
            peak = max(float(v.max()) for _, v in columns)
            scale = max(peak - cut, 1e-6)
            return lambda vol: np.ma.array(np.clip((vol - cut) / scale, 0.0, 1.0),
                                           mask=~(vol > cut))
        return lambda vol: _prep_attn_display(vol, attn_percentile)[0]

    modes = list(scale_modes)
    nrows, ncols = len(modes), len(columns)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.8 * nrows))
    axes = np.atleast_2d(axes)
    legend_handles: List = []
    for ri, mode in enumerate(modes):
        disp = _disp_for(mode)
        for ci, (label, vol) in enumerate(columns):
            ax = axes[ri, ci]
            ax.imshow(ct_slc, cmap="gray", origin="lower", interpolation="nearest")
            ax.imshow(_orient(disp(vol)[sl]), cmap="hot", origin="lower", interpolation="nearest",
                      alpha=attn_alpha, vmin=0.0, vmax=1.0)
            handles = _draw_compartment_contours(ax, mask_slc)
            if not legend_handles and handles:
                legend_handles = handles
            if ri == 0:
                ax.set_title(label, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{mode} scale", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 6), frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 0.0))

    title = f"Per-stage cross-timepoint attention ({direction}, {view} idx={idx_shown})"
    if case_id:
        title = f"{title}  ·  {case_id}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out_path


def save_dual_attention_report(
    case_id: str,
    ct_T0: Union[torch.Tensor, np.ndarray],
    ct_T1: Union[torch.Tensor, np.ndarray],
    masks_T0: Mapping[str, Union[torch.Tensor, np.ndarray]],
    masks_T1: Mapping[str, Union[torch.Tensor, np.ndarray]],
    attn_maps_per_stage: Sequence[Mapping[str, torch.Tensor]],
    out_dir: Union[str, Path],
    stage_weights: Optional[Sequence[float]] = None,
    sample_index: int = 0,
    skip_stages: Sequence[int] = (),
    valid_T0: Optional[Union[torch.Tensor, np.ndarray]] = None,
    valid_T1: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Dict[str, Path]:
    """Aggregate both attention directions, render 2×3 panel, write PNG + JSON.

    Args:
        case_id: Case identifier, used to name output files.
        ct_T0: T0 CT volume, shape `(D, H, W)`.
        ct_T1: T1 CT volume, shape `(D, H, W)`.
        masks_T0: Maps compartment name to a bool mask for T0.
        masks_T1: Maps compartment name to a bool mask for T1.
        attn_maps_per_stage: One attention-map dict per stage.
        out_dir: Output directory; created if missing.
        stage_weights: Optional per-stage aggregation weights.
        sample_index: Index into the batch dimension to extract.
        skip_stages: Stage indices to exclude from aggregation.
        valid_T0: Optional T0 valid mask; attention outside it is zeroed.
        valid_T1: Optional T1 valid mask; attention outside it is zeroed.

    Returns:
        Dict with `'figure'` and `'stats'` paths, plus `'montage'` when the
        per-stage montage render succeeds (it never blocks the report).
    """
    ct0_np = _to_numpy_3d(ct_T0)
    ct1_np = _to_numpy_3d(ct_T1)

    attn_01 = aggregate_stage_attention(
        attn_maps_per_stage=attn_maps_per_stage,
        target_shape=tuple(ct0_np.shape),
        direction="T1_to_T0",
        weights=stage_weights,
        sample_index=sample_index,
        skip_stages=skip_stages,
    )
    attn_10 = aggregate_stage_attention(
        attn_maps_per_stage=attn_maps_per_stage,
        target_shape=tuple(ct1_np.shape),
        direction="T0_to_T1",
        weights=stage_weights,
        sample_index=sample_index,
        skip_stages=skip_stages,
    )

    # Restrict attention to the valid region so padding never shows.
    attn_01 = _mask_to_valid(attn_01, valid_T0)
    attn_10 = _mask_to_valid(attn_10, valid_T1)

    stats_T0 = compartment_stats(attn_01, masks_T0)
    stats_T1 = compartment_stats(attn_10, masks_T1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{case_id}_attention.png"
    render_dual_overlay(
        ct_T0=ct0_np, ct_T1=ct1_np,
        attn_T1_to_T0=attn_01, attn_T0_to_T1=attn_10,
        masks_T0=masks_T0, masks_T1=masks_T1,
        out_path=fig_path, case_id=case_id,
    )

    # Crisp per-stage montage (T1->T0 on T0); never block the report.
    montage_path = out_dir / f"{case_id}_stages.png"
    out_paths = {"figure": fig_path}
    try:
        render_stage_montage(
            ct=ct0_np, masks=masks_T0,
            attn_maps_per_stage=attn_maps_per_stage,
            out_path=montage_path, direction="T1_to_T0",
            case_id=case_id, sample_index=sample_index,
            skip_stages=skip_stages, valid=valid_T0,
        )
        out_paths["montage"] = montage_path
    except Exception as exc:
        logger.warning("stage montage failed for %s: %s", case_id, exc)

    stats_path = out_dir / f"{case_id}_attention_stats.json"
    payload = {
        "case_id": case_id,
        "n_stages": len(attn_maps_per_stage),
        "stage_weights": list(stage_weights) if stage_weights is not None else None,
        "skip_stages": list(skip_stages),
        "T1_to_T0_on_T0": stats_T0,
        "T0_to_T1_on_T1": stats_T1,
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    out_paths["stats"] = stats_path
    return out_paths


def save_attention_report(
    case_id: str,
    ct: Union[torch.Tensor, np.ndarray],
    masks: Mapping[str, Union[torch.Tensor, np.ndarray]],
    attn_maps_per_stage: Sequence[Mapping[str, torch.Tensor]],
    out_dir: Union[str, Path],
    direction: str = "T1_to_T0",
    stage_weights: Optional[Sequence[float]] = None,
    sample_index: int = 0,
) -> Dict[str, Path]:
    """Aggregate attention, compute stats, and write `{case_id}_attention.png` + `.json`.

    Args:
        case_id: Case identifier, used to name output files.
        ct: CT volume, shape `(D, H, W)`.
        masks: Maps compartment name to a bool mask, same shape as `ct`.
        attn_maps_per_stage: One attention-map dict per stage.
        out_dir: Output directory; created if missing.
        direction: Which attention direction to aggregate.
        stage_weights: Optional per-stage aggregation weights.
        sample_index: Index into the batch dimension to extract.

    Returns:
        Dict with `'figure'` and `'stats'` paths.
    """
    ct_np = _to_numpy_3d(ct)
    target_shape = tuple(ct_np.shape)  # (D, H, W)

    attn = aggregate_stage_attention(
        attn_maps_per_stage=attn_maps_per_stage,
        target_shape=target_shape,
        direction=direction,
        weights=stage_weights,
        sample_index=sample_index,
    )
    stats = compartment_stats(attn, masks)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / f"{case_id}_attention.png"
    render_compartment_overlay(
        ct=ct_np,
        attn=attn,
        masks=masks,
        out_path=fig_path,
        case_id=case_id,
        stats=stats,
    )

    stats_path = out_dir / f"{case_id}_attention_stats.json"
    payload = {
        "case_id": case_id,
        "direction": direction,
        "n_stages": len(attn_maps_per_stage),
        "stage_weights": list(stage_weights) if stage_weights is not None else None,
        "compartment_stats": stats,
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)

    logger.info(
        "Attention report written for %s → %s (lift: IT=%.2fx TVI=%.2fx)",
        case_id, fig_path,
        stats.get("mask_it", {}).get("attention_lift", 0.0),
        stats.get("mask_tvi", {}).get("attention_lift", 0.0),
    )
    return {"figure": fig_path, "stats": stats_path}


def render_roi_overlay(
    ct_t0: Union[torch.Tensor, np.ndarray],
    ct_t1: Union[torch.Tensor, np.ndarray],
    masks_t0: Mapping[str, Union[torch.Tensor, np.ndarray]],
    masks_t1: Mapping[str, Union[torch.Tensor, np.ndarray]],
    out_path: Union[str, Path],
    case_id: str = "",
    dpi: int = 130,
) -> Path:
    """Render a side-by-side ROI overlay for the T0 and T1 patches at cache time.

    Args:
        ct_t0: T0 CT volume, shape `(D, H, W)`.
        ct_t1: T1 CT volume, shape `(D, H, W)`.
        masks_t0: Maps compartment name to a bool mask for T0.
        masks_t1: Maps compartment name to a bool mask for T1.
        out_path: PNG output path; parent directories are created.
        case_id: Optional case identifier shown in the figure title.
        dpi: Output image resolution.

    Returns:
        `out_path`.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ct0 = _to_numpy_3d(ct_t0); ct1 = _to_numpy_3d(ct_t1)
    m0 = {n: _to_numpy_3d(m).astype(bool) for n, m in masks_t0.items()}
    m1 = {n: _to_numpy_3d(m).astype(bool) for n, m in masks_t1.items()}

    def _center_indices(masks: Dict[str, np.ndarray], ref_shape: Tuple[int, int, int],
                        primary_key: str) -> Tuple[int, int, int]:
        primary = masks.get(primary_key)
        if primary is not None and primary.any():
            return _pick_center_slices(primary)
        return tuple(s // 2 for s in ref_shape)

    i0_t0, i1_t0, i2_t0 = _center_indices(m0, ct0.shape, "mask_it")
    i0_t1, i1_t1, i2_t1 = _center_indices(m1, ct1.shape, "mask_it_t1")

    def _disp(ct: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(ct, [1, 99])
        return np.clip((ct - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    ct0d = _disp(ct0); ct1d = _disp(ct1)

    panels = [
        ("T0", ct0d, m0, i0_t0, i1_t0, i2_t0),
        ("T1", ct1d, m1, i0_t1, i1_t1, i2_t1),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"ROI overlay  ·  {case_id}", fontsize=12)
    legend_handles: List = []
    seen_labels: set = set()
    for row, (label, ct_disp, masks, i0, i1, i2) in enumerate(panels):
        for col, (view, slc_ct, slc_masks, idx) in enumerate([
            ("axial",    ct_disp[:, :, i2], {k: v[:, :, i2] for k, v in masks.items()}, i2),
            ("coronal",  ct_disp[:, i1, :], {k: v[:, i1, :] for k, v in masks.items()}, i1),
            ("sagittal", ct_disp[i0, :, :], {k: v[i0, :, :] for k, v in masks.items()}, i0),
        ]):
            ax = axes[row][col]
            # No transpose: must match _draw_compartment_contours' orientation.
            ax.imshow(slc_ct, cmap="gray", origin="lower", interpolation="nearest")
            handles = _draw_compartment_contours(ax, slc_masks)
            for h in handles:
                if h.get_label() not in seen_labels:
                    legend_handles.append(h); seen_labels.add(h.get_label())
            ax.set_title(f"{label}  {view}  idx={idx}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    if legend_handles:
        fig.legend(
            handles=legend_handles, loc="lower center",
            ncol=min(5, len(legend_handles)), frameon=False, fontsize=8,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)   # close even if savefig fails, to avoid a figure leak
    return out_path


_MASK_KEYS_T0 = ("mask_it", "mask_pt1", "mask_pt2", "mask_pt3", "mask_tvi")
_MASK_KEYS_T1 = ("mask_it_t1", "mask_pt1_t1", "mask_pt2_t1", "mask_pt3_t1", "mask_tvi_t1")


@torch.no_grad()
def render_attention_maps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    clinical_dim: int,
    out_dir: Path,
    n_cases: int,
    anatomy_dim: int = 0,
    vessel_dim: int = 0,
    skip_attn_stages: Sequence[int] = (0,),
    case_ids: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Render cross-timepoint attention overlays for the first *n_cases* samples.

    Args:
        model: Model to run in eval mode; must accept `return_attn` and
            `skip_attn_stages` kwargs (see `LongitudinalResponseModel.forward`).
        loader: Data loader yielding batches with `t0`, `t1`, and optionally
            `clinical`, `anatomy`, `vessel`, `valid_t0`, `valid_t1`, mask
            keys, and `case_id`.
        device: Device to run inference on.
        clinical_dim: Expected clinical feature dim; falls back to a
            zero vector when the batch doesn't have a matching key.
        out_dir: Output directory for rendered reports; created if missing.
        n_cases: Number of cases to render; ignored (replaced by
            `len(case_ids)`) when `case_ids` is given.
        anatomy_dim: Expected anatomy feature dim; omitted when `<= 0` or the
            batch doesn't have a matching key.
        vessel_dim: Expected vessel feature dim; omitted when `<= 0` or the
            batch doesn't have a matching key.
        skip_attn_stages: Stage indices to exclude from attention-map
            materialisation.
        case_ids: Optional specific case IDs to render; renders the first
            `n_cases` batch samples when `None`.

    Returns:
        Paths to the rendered figure for each case.
    """
    wanted_ids: Optional[set] = set(case_ids) if case_ids else None
    if wanted_ids is not None:
        n_cases = len(wanted_ids)

    if n_cases <= 0:
        return []

    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    rendered_ids: set = set()

    for batch in loader:
        x0 = batch["t0"].to(device)
        x1 = batch["t1"].to(device)
        B  = x0.shape[0]

        clin   = _clinical_feat(batch, clinical_dim, device)
        anat   = _anatomy_feat(batch, anatomy_dim, device)
        vessel = _vessel_feat(batch, vessel_dim, device)
        valid_t0 = batch["valid_t0"].to(device).bool() if "valid_t0" in batch else None
        valid_t1 = batch["valid_t1"].to(device).bool() if "valid_t1" in batch else None

        # return_attn=True is memory-heavy; use a small-batch loader (B<=1).
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(
                x0, x1,
                radiomic_features=None,
                clinical_features=clin,
                anatomy_features=anat,
                vessel_features=vessel,
                valid_T0=valid_t0, valid_T1=valid_t1,
                return_attn=True,
                skip_attn_stages=skip_attn_stages,
            )

        batch_case_ids = batch.get("case_id")

        if wanted_ids is not None:
            if batch_case_ids is None:
                if len(paths) >= n_cases:
                    break
                continue
            indices = [
                i for i in range(B)
                if str(batch_case_ids[i]) in wanted_ids
                and str(batch_case_ids[i]) not in rendered_ids
            ]
        else:
            take = min(n_cases - len(paths), B)
            indices = list(range(take))

        for i in indices:
            cid    = batch_case_ids[i] if batch_case_ids is not None else f"val_{len(paths)}"
            rendered_ids.add(str(cid))
            masks0 = {k: batch[k][i] for k in _MASK_KEYS_T0 if k in batch}
            masks1 = {k: batch[k][i] for k in _MASK_KEYS_T1 if k in batch}
            try:
                report = save_dual_attention_report(
                    case_id=str(cid),
                    ct_T0=x0[i],
                    ct_T1=x1[i],
                    masks_T0=masks0,
                    masks_T1=masks1,
                    attn_maps_per_stage=out["attention_maps"],
                    out_dir=out_dir,
                    sample_index=i,
                    skip_stages=skip_attn_stages,
                    valid_T0=valid_t0[i] if valid_t0 is not None else None,
                    valid_T1=valid_t1[i] if valid_t1 is not None else None,
                )
                paths.append(report["figure"])
            except Exception as exc:
                logger.warning("save_dual_attention_report failed for %s: %s", cid, exc)

        if len(paths) >= n_cases:
            break

    return paths


def _clinical_feat(
    batch: Dict[str, Any],
    clinical_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract the clinical feature tensor from a batch, zero-filled as a fallback.

    Args:
        batch: Data loader batch.
        clinical_dim: Expected clinical feature dim.
        device: Device to move the tensor to.

    Returns:
        Clinical features, shape `(B, clinical_dim)`; zeros when `batch`
        lacks a matching `'clinical'` key.
    """
    if clinical_dim > 0 and "clinical" in batch and batch["clinical"].shape[-1] == clinical_dim:
        return batch["clinical"].to(device).float()
    return torch.zeros(batch["t0"].shape[0], clinical_dim, device=device)


def _anatomy_feat(
    batch: Dict[str, Any],
    anatomy_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Extract the anatomy feature tensor from a batch, if present.

    Args:
        batch: Data loader batch.
        anatomy_dim: Expected anatomy feature dim; `<= 0` disables the token.
        device: Device to move the tensor to.

    Returns:
        Anatomy features, shape `(B, anatomy_dim)`, or `None` when disabled
        or `batch` lacks a matching `'anatomy'` key.
    """
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
    """Extract the vessel feature tensor from a batch, if present.

    Args:
        batch: Data loader batch.
        vessel_dim: Expected vessel feature dim; `<= 0` disables the token.
        device: Device to move the tensor to.

    Returns:
        Vessel features, shape `(B, vessel_dim)`, or `None` when disabled or
        `batch` lacks a matching `'vessel'` key.
    """
    if vessel_dim <= 0:
        return None
    if "vessel" not in batch or batch["vessel"].shape[-1] != vessel_dim:
        return None
    return batch["vessel"].to(device).float()
