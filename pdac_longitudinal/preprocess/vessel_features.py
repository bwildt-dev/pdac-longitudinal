"""Tumour-vessel interface feature extraction for the PDAC longitudinal framework."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from pdac_longitudinal.preprocess.roi_rings import extract_vessel_masks_from_segmentation
from pdac_longitudinal.preprocess.vessel_tumor_metrics import compute_vessel_tumor_metrics

logger = logging.getLogger(__name__)

# Canonical vessels and metrics

_VESSELS: Tuple[str, ...] = ("sma", "celiac", "veins", "postcava")

# Map PanTS long mask names to the short canonical names the metrics use;
# without this the arterial vessel features silently stay 0.
_VESSEL_KEY_ALIASES: Dict[str, str] = {
    "superior_mesenteric_artery": "sma",
    "celiac_artery":              "celiac",
    "veins":                      "veins",
    "postcava":                   "postcava",
}
_PER_VESSEL_METRICS: Tuple[str, ...] = (
    "min_distance_mm",
    "contact_fraction",
    "max_encasement_deg",
    "contact_length_mm",   # craniocaudal extent of tumour-vessel contact
    "stenosis_ratio",
)

# CT-derived peri-vascular interface texture, grouped arterial vs venous.
_INTERFACE_GROUPS:  Tuple[str, ...] = ("arterial", "venous")
_INTERFACE_METRICS: Tuple[str, ...] = (
    "interface_hu_mean",
    "interface_fat_frac",
    "interface_grad_mean",
)

_RESECT_ORDINAL: Dict[str, float] = {
    "no_tumor":       0.0,
    "resectable":     0.0,
    "borderline":     1.0,
    "locally_advanced": 2.0,
}


def _vessel_feature_columns() -> Tuple[str, ...]:
    cols = []
    # Per-vessel × T0
    for v in _VESSELS:
        for m in _PER_VESSEL_METRICS:
            cols.append(f"{v}_{m}_t0")
    # Per-vessel × T1
    for v in _VESSELS:
        for m in _PER_VESSEL_METRICS:
            cols.append(f"{v}_{m}_t1")
    # Aggregates T0
    cols += ["arterial_enc_deg_t0", "venous_enc_deg_t0", "resectability_t0",
             "tumor_max_diameter_mm_t0"]
    # Aggregates T1
    cols += ["arterial_enc_deg_t1", "venous_enc_deg_t1", "resectability_t1",
             "tumor_max_diameter_mm_t1"]
    # CT interface texture, T0 then T1
    for suffix in ("_t0", "_t1"):
        for g in _INTERFACE_GROUPS:
            for m in _INTERFACE_METRICS:
                cols.append(f"{g}_{m}{suffix}")
    # Per-vessel deltas (T1 − T0): encasement arc + min-distance.
    for v in _VESSELS:
        cols.append(f"delta_{v}_enc_deg")
    for v in _VESSELS:
        cols.append(f"delta_{v}_min_dist_mm")
    # Aggregate deltas
    cols += ["delta_arterial_enc_deg", "delta_venous_enc_deg"]
    # Interface-texture response deltas
    cols += ["delta_arterial_interface_fat_frac", "delta_venous_interface_fat_frac",
             "delta_arterial_interface_hu_mean",  "delta_venous_interface_hu_mean"]
    # Tumour volume T0, T1, Δ% and diameter Δ%
    cols += ["tumor_vol_t0_mm3", "tumor_vol_t1_mm3",
             "delta_tumor_vol_pct", "delta_tumor_diameter_pct"]
    return tuple(cols)


VESSEL_FEATURE_COLS: Tuple[str, ...] = _vessel_feature_columns()
VESSEL_FEATURE_DIM:  int              = len(VESSEL_FEATURE_COLS)   # 78


# Vector helper

def features_dict_to_vector(
    feats: Dict[str, float],
    columns: Tuple[str, ...] = VESSEL_FEATURE_COLS,
    fill: float = 0.0,
) -> np.ndarray:
    """Pack a feature dict into a fixed-length float32 array.

    Args:
        feats: Feature name -> value mapping.
        columns: Column order to emit.
        fill: Value used for missing entries; non-finite entries fall back
            to `999.0` for distance columns (sentinel for "no vessel / no
            tumour") or to `fill` otherwise.

    Returns:
        A float32 array of length `len(columns)`.
    """
    vec = np.full(len(columns), fill, dtype=np.float32)
    for i, col in enumerate(columns):
        val = feats.get(col, fill)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            val = 999.0 if "distance" in col else fill
        vec[i] = float(val)
    return vec


# Extractor

class VesselFeatureExtractor:
    """Extract vessel–tumour contact features from PanTS segmentation arrays."""

    def extract_one_from_arrays(
        self,
        seg: np.ndarray,
        spacing_zyx: Tuple[float, float, float],
        suffix: str,
        ct: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Extract per-vessel metrics from one segmentation array.

        Args:
            seg: PanTS label volume (tumour = label 2).
            spacing_zyx: Voxel spacing `(z, y, x)` in mm.
            suffix: Appended to each output key.
            ct: Optional CT intensity volume; enables interface-texture
                metrics when given.

        Returns:
            Feature name -> value mapping. Empty if metric computation fails.
        """
        tumor_mask   = (seg == 2).astype(bool)
        raw_masks    = extract_vessel_masks_from_segmentation(seg)
        # Canonicalise long PanTS keys -> short names the metrics module expects.
        vessel_masks = {
            _VESSEL_KEY_ALIASES.get(name, name): mask
            for name, mask in raw_masks.items()
        }

        try:
            metrics = compute_vessel_tumor_metrics(
                tumor_mask, vessel_masks, spacing_zyx, ct=ct,
            )
        except Exception as exc:
            logger.warning("compute_vessel_tumor_metrics failed%s: %s", suffix, exc)
            return {}

        feats: Dict[str, float] = {}

        # Per-vessel scalar metrics
        per_vessel = metrics.get("per_vessel", {})
        for v in _VESSELS:
            vm = per_vessel.get(v, {})
            for m in _PER_VESSEL_METRICS:
                key = f"{v}_{m}{suffix}"
                val = vm.get(m, 0.0)
                feats[key] = float(val) if val is not None else 0.0

        # Aggregates
        feats[f"arterial_enc_deg{suffix}"] = float(
            metrics.get("max_arterial_encasement_deg", 0.0)
        )
        feats[f"venous_enc_deg{suffix}"] = float(
            metrics.get("max_venous_encasement_deg", 0.0)
        )
        feats[f"resectability{suffix}"] = _RESECT_ORDINAL.get(
            str(metrics.get("resectability_category", "no_tumor")), 0.0
        )
        feats[f"tumor_vol{suffix}_mm3"] = float(
            metrics.get("tumor_volume_mm3", 0.0)
        )
        feats[f"tumor_max_diameter_mm{suffix}"] = float(
            metrics.get("tumor_max_diameter_mm", 0.0)
        )

        # CT-derived interface texture; 0.0 when CT absent.
        for g in _INTERFACE_GROUPS:
            for m in _INTERFACE_METRICS:
                feats[f"{g}_{m}{suffix}"] = float(metrics.get(f"{g}_{m}", 0.0))

        return feats

    @staticmethod
    def _compute_pair_derivatives(feats: Dict[str, float]) -> Dict[str, float]:
        """Compute T1 − T0 deltas from an already-extracted feature dict.

        Args:
            feats: Feature dict containing the per-timepoint `_t0`/`_t1` columns.

        Returns:
            The derived `delta_*` columns.
        """
        deltas: Dict[str, float] = {}

        # Per-vessel encasement deltas
        for v in _VESSELS:
            t0 = feats.get(f"{v}_max_encasement_deg_t0", 0.0)
            t1 = feats.get(f"{v}_max_encasement_deg_t1", 0.0)
            deltas[f"delta_{v}_enc_deg"] = float(t1 - t0)

        # Per-vessel min-distance deltas (positive ⇒ vessel cleared post-NAT);
        # non-finite sentinels (vessel/tumour absent) guard to 0.
        for v in _VESSELS:
            d0 = feats.get(f"{v}_min_distance_mm_t0", 0.0)
            d1 = feats.get(f"{v}_min_distance_mm_t1", 0.0)
            if np.isfinite(d0) and np.isfinite(d1):
                deltas[f"delta_{v}_min_dist_mm"] = float(d1 - d0)
            else:
                deltas[f"delta_{v}_min_dist_mm"] = 0.0

        # Aggregate arterial / venous deltas
        deltas["delta_arterial_enc_deg"] = float(
            feats.get("arterial_enc_deg_t1", 0.0) - feats.get("arterial_enc_deg_t0", 0.0)
        )
        deltas["delta_venous_enc_deg"] = float(
            feats.get("venous_enc_deg_t1", 0.0) - feats.get("venous_enc_deg_t0", 0.0)
        )

        # +fat_frac or −HU post-NAT ⇒ fat plane re-emerging (favourable response).
        for g in _INTERFACE_GROUPS:
            for m in ("interface_fat_frac", "interface_hu_mean"):
                deltas[f"delta_{g}_{m}"] = float(
                    feats.get(f"{g}_{m}_t1", 0.0) - feats.get(f"{g}_{m}_t0", 0.0)
                )

        # Tumour volume delta (%)
        vol_t0 = feats.get("tumor_vol_t0_mm3", 0.0)
        vol_t1 = feats.get("tumor_vol_t1_mm3", 0.0)
        if vol_t0 and vol_t0 > 0:
            deltas["delta_tumor_vol_pct"] = float((vol_t1 - vol_t0) / vol_t0 * 100.0)
        else:
            deltas["delta_tumor_vol_pct"] = 0.0

        # Tumour max-diameter delta (%)
        d_t0 = feats.get("tumor_max_diameter_mm_t0", 0.0)
        d_t1 = feats.get("tumor_max_diameter_mm_t1", 0.0)
        if d_t0 and d_t0 > 0:
            deltas["delta_tumor_diameter_pct"] = float((d_t1 - d_t0) / d_t0 * 100.0)
        else:
            deltas["delta_tumor_diameter_pct"] = 0.0

        return deltas
