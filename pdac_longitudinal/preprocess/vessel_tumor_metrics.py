"""Clinical vessel–tumour contact metrics derived from segmentations."""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    from scipy.spatial import ConvexHull, QhullError
except Exception:  # pragma: no cover - import shim across scipy versions
    ConvexHull = None  # type: ignore

    class QhullError(Exception):  # type: ignore
        """Fallback when scipy.spatial.qhull is unavailable."""

logger = logging.getLogger(__name__)

ARTERIAL_VESSELS: Tuple[str, ...] = ("sma", "celiac")
VENOUS_VESSELS:   Tuple[str, ...] = ("veins", "postcava")


def _tumor_surface(mask: np.ndarray) -> np.ndarray:
    """Return the 1-voxel-thick inner surface of a binary mask (voxels with at least one background 6-neighbour)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    shifted = (
        np.pad(mask, ((1, 0), (0, 0), (0, 0)))[:-1]
        & np.pad(mask, ((0, 1), (0, 0), (0, 0)))[1:]
        & np.pad(mask, ((0, 0), (1, 0), (0, 0)))[:, :-1]
        & np.pad(mask, ((0, 0), (0, 1), (0, 0)))[:, 1:]
        & np.pad(mask, ((0, 0), (0, 0), (1, 0)))[:, :, :-1]
        & np.pad(mask, ((0, 0), (0, 0), (0, 1)))[:, :, 1:]
    )
    return mask & ~shifted


def _longest_arc_degrees(present: np.ndarray) -> float:
    """Return the longest contiguous True run in a 360-bin circular array (degrees), accounting for wrap-around."""
    if not present.any():
        return 0.0
    if present.all():
        return 360.0
    # Duplicate to handle wrap-around in a single pass.
    doubled = np.concatenate([present, present])
    best = cur = 0
    for v in doubled:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return float(min(best, 360))


def _encasement_degrees(
    tumor_mask: np.ndarray,
    vessel_mask: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    ring_radius_mm: float = 8.0,
) -> float:
    """Max axial arc (degrees) of tumour voxels around the vessel centroid; approximates
    circumferential vessel encasement by tumour.
    """
    if not (tumor_mask.any() and vessel_mask.any()):
        return 0.0

    sz, sy, sx = spacing_zyx  # noqa: F841
    max_deg = 0.0

    z_idxs = np.where(vessel_mask.any(axis=(1, 2)))[0]
    for z in z_idxs:
        t_slice = tumor_mask[z]
        v_slice = vessel_mask[z]
        if not (t_slice.any() and v_slice.any()):
            continue
        vy, vx = np.where(v_slice)
        cy = float(vy.mean()) * sy
        cx = float(vx.mean()) * sx
        ty, tx = np.where(t_slice)
        if ty.size == 0:
            continue
        dy = ty * sy - cy
        dx = tx * sx - cx
        dist = np.hypot(dy, dx)
        inside = dist <= ring_radius_mm
        if not inside.any():
            continue
        angles = np.degrees(np.arctan2(dy[inside], dx[inside])) % 360.0
        bins = np.zeros(360, dtype=bool)
        bins[angles.astype(np.int32)] = True
        deg = _longest_arc_degrees(bins)
        if deg > max_deg:
            max_deg = deg
    return float(max_deg)


def _max_diameter_mm(
    mask: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
) -> float:
    """Largest 3-D extent (mm) of a binary mask; a max-Feret-diameter proxy (convex hull,
    falling back to the bounding-box diagonal; 0.0 if `mask` is empty).
    """
    if not mask.any():
        return 0.0
    sz, sy, sx = spacing_zyx
    surf = _tumor_surface(mask)
    zz, yy, xx = np.where(surf if surf.any() else mask)
    pts_mm = np.column_stack([zz * sz, yy * sy, xx * sx]).astype(np.float64)

    if pts_mm.shape[0] > 3 and ConvexHull is not None:
        try:
            hull = ConvexHull(pts_mm)
            verts = pts_mm[hull.vertices]
            diffs = verts[:, None, :] - verts[None, :, :]
            return float(np.sqrt((diffs ** 2).sum(axis=-1)).max())
        except (QhullError, ValueError):
            pass  # fall through to bbox diagonal

    # Bounding-box diagonal fallback.
    ext_z = (zz.max() - zz.min()) * sz
    ext_y = (yy.max() - yy.min()) * sy
    ext_x = (xx.max() - xx.min()) * sx
    return float(np.sqrt(ext_z ** 2 + ext_y ** 2 + ext_x ** 2))


def _vessel_caliber(
    vessel_mask: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Axial cross-sectional caliber of a vessel and its narrowing ratio.

    Returns:
        `(min_csa_mm2, median_csa_mm2, stenosis_ratio)`; ratio is 1.0 (no narrowing) if
        the mask is empty.
    """
    sz, sy, sx = spacing_zyx  # noqa: F841
    area_per_vox = float(sy * sx)
    z_present = np.where(vessel_mask.any(axis=(1, 2)))[0]
    if z_present.size == 0:
        return 0.0, 0.0, 1.0
    areas = vessel_mask[z_present].sum(axis=(1, 2)).astype(np.float64) * area_per_vox
    median_csa = float(np.median(areas))
    min_csa = float(np.percentile(areas, 5))
    ratio = float(min_csa / median_csa) if median_csa > 0 else 1.0
    return min_csa, median_csa, ratio


def _gradient_magnitude(
    ct: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
) -> np.ndarray:
    """Per-voxel CT intensity-gradient magnitude (HU/mm)."""
    sz, sy, sx = (float(s) for s in spacing_zyx)
    gz, gy, gx = np.gradient(ct.astype(np.float32), sz, sy, sx)
    return np.sqrt(gz * gz + gy * gy + gx * gx)


def _empty_interface() -> Dict[str, float]:
    return {
        "interface_hu_mean":   0.0,
        "interface_fat_frac":  0.0,
        "interface_grad_mean": 0.0,
        "interface_n_voxels":  0,
    }


def _group_union(
    vessel_masks: Mapping[str, np.ndarray],
    names: Tuple[str, ...],
    shape: Tuple[int, ...],
) -> np.ndarray:
    """Boolean union of the listed vessels (empty array when none present)."""
    out = np.zeros(shape, dtype=bool)
    for n in names:
        m = vessel_masks.get(n)
        if m is not None and m.size:
            out |= m.astype(bool)
    return out


def _interface_texture(
    ct: np.ndarray,
    grad_mag: np.ndarray,
    tumor_mask: np.ndarray,
    vessel_mask: np.ndarray,
    dist_to_tumor: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    band_mm: float,
    fat_hu: float,
) -> Dict[str, float]:
    """CT texture of the peri-vascular interface band (fat-plane / wall-blur).

    The interface band is the set of non-tumour, non-vessel voxels within `band_mm` of both
    the tumour and the vessel group. Returns `interface_hu_mean`, `interface_fat_frac`,
    `interface_grad_mean`, `interface_n_voxels`; all zero if `vessel_mask` is empty or the
    band contains no voxels.
    """
    if not vessel_mask.any():
        return _empty_interface()
    dist_to_vessel = distance_transform_edt(~vessel_mask, sampling=spacing_zyx)
    band = (
        (dist_to_vessel <= band_mm)
        & (dist_to_tumor <= band_mm)
        & ~tumor_mask
        & ~vessel_mask
    )
    n = int(band.sum())
    if n == 0:
        return _empty_interface()
    hu = ct[band]
    return {
        "interface_hu_mean":   float(hu.mean()),
        "interface_fat_frac":  float((hu < fat_hu).mean()),
        "interface_grad_mean": float(grad_mag[band].mean()),
        "interface_n_voxels":  n,
    }


def compute_vessel_tumor_metrics(
    tumor_mask: np.ndarray,
    vessel_masks: Mapping[str, np.ndarray],
    spacing_zyx: Tuple[float, float, float],
    contact_mm: float = 1.0,
    ring_radius_mm: float = 8.0,
    ct: Optional[np.ndarray] = None,
    interface_band_mm: float = 4.0,
    fat_hu: float = -30.0,
) -> Dict[str, object]:
    """Compute per-vessel contact metrics and aggregate resectability for one case.

    `ct` is optional; if given and shape-matched to `tumor_mask`, peri-vascular interface
    texture is also computed, otherwise those fields are zero-filled.

    Returns:
        Aggregate fields (`tumor_volume_mm3`, `tumor_max_diameter_mm`, `resectability_category`
        in `{"no_tumor", "resectable", "borderline", "locally_advanced"}`, max arterial/venous
        encasement in degrees, interface texture stats) plus a `per_vessel` dict keyed by
        vessel name.
    """
    sz, sy, sx = spacing_zyx
    vox_vol_mm3 = float(sz * sy * sx)
    mean_face_area_mm2 = float((sy * sx + sz * sx + sz * sy) / 3.0)

    tumor_mask = tumor_mask.astype(bool)
    tumor_voxels = int(tumor_mask.sum())
    tumor_volume_mm3 = tumor_voxels * vox_vol_mm3

    if tumor_voxels == 0:
        out = {
            "tumor_voxels": 0,
            "tumor_volume_mm3": 0.0,
            "tumor_surface_voxels": 0,
            "tumor_max_diameter_mm": 0.0,
            "per_vessel": {k: _empty_vessel_metrics() for k in vessel_masks},
            "any_vessel_contact": False,
            "max_arterial_encasement_deg": 0.0,
            "max_venous_encasement_deg": 0.0,
            "resectability_category": "no_tumor",
        }
        for grp in ("arterial", "venous"):
            for k, v in _empty_interface().items():
                out[f"{grp}_{k}"] = v
        return out

    surface = _tumor_surface(tumor_mask)
    surface_voxels = int(surface.sum())
    tumor_max_diameter_mm = _max_diameter_mm(tumor_mask, spacing_zyx)

    per_vessel: Dict[str, Dict[str, float]] = {}
    for name, vmask in vessel_masks.items():
        vmask_b = vmask.astype(bool)
        if not vmask_b.any():
            per_vessel[name] = _empty_vessel_metrics()
            continue

        dist_to_vessel = distance_transform_edt(~vmask_b, sampling=spacing_zyx)
        tumor_dists = dist_to_vessel[tumor_mask]
        min_dist = float(tumor_dists.min())

        contact_mask = surface & (dist_to_vessel <= contact_mm)
        contact_vox = int(contact_mask.sum())

        # Craniocaudal extent of contact: number of axial slices touching, and its length.
        contact_z = np.where(contact_mask.any(axis=(1, 2)))[0]
        contact_n_slices = int(contact_z.size)
        contact_length_mm = float(contact_n_slices) * float(sz)

        encasement_deg = _encasement_degrees(
            tumor_mask, vmask_b, spacing_zyx, ring_radius_mm=ring_radius_mm
        )

        # Vessel caliber / focal narrowing along its axial course.
        min_csa, median_csa, stenosis_ratio = _vessel_caliber(vmask_b, spacing_zyx)

        per_vessel[name] = {
            "min_distance_mm":      min_dist,
            "contact_voxels":       contact_vox,
            "contact_surface_mm2":  contact_vox * mean_face_area_mm2,
            "contact_fraction":     float(contact_vox / max(surface_voxels, 1)),
            "max_encasement_deg":   encasement_deg,
            "contact_n_slices":     contact_n_slices,
            "contact_length_mm":    contact_length_mm,
            "min_csa_mm2":          min_csa,
            "median_csa_mm2":       median_csa,
            "stenosis_ratio":       stenosis_ratio,
        }

    any_contact = any(v["contact_voxels"] > 0 for v in per_vessel.values())
    max_art = max(
        (per_vessel[n]["max_encasement_deg"] for n in ARTERIAL_VESSELS if n in per_vessel),
        default=0.0,
    )
    max_ven = max(
        (per_vessel[n]["max_encasement_deg"] for n in VENOUS_VESSELS if n in per_vessel),
        default=0.0,
    )

    if not any_contact:
        category = "resectable"
    elif max_art > 180.0:
        category = "locally_advanced"
    else:
        category = "borderline"

    # Peri-vascular interface texture (arterial vs venous); only computed when a CT
    # volume is supplied (cache-build time).
    interface: Dict[str, float] = {}
    if ct is not None and ct.shape == tumor_mask.shape:
        grad_mag = _gradient_magnitude(ct, spacing_zyx)
        dist_to_tumor = distance_transform_edt(~tumor_mask, sampling=spacing_zyx)
        for grp, names in (("arterial", ARTERIAL_VESSELS), ("venous", VENOUS_VESSELS)):
            grp_mask = _group_union(vessel_masks, names, tumor_mask.shape)
            tex = _interface_texture(
                ct, grad_mag, tumor_mask, grp_mask, dist_to_tumor,
                spacing_zyx, interface_band_mm, fat_hu,
            )
            for k, v in tex.items():
                interface[f"{grp}_{k}"] = v
    else:
        for grp in ("arterial", "venous"):
            for k, v in _empty_interface().items():
                interface[f"{grp}_{k}"] = v

    return {
        "tumor_voxels":                  tumor_voxels,
        "tumor_volume_mm3":              tumor_volume_mm3,
        "tumor_surface_voxels":          surface_voxels,
        "tumor_max_diameter_mm":         float(tumor_max_diameter_mm),
        "per_vessel":                    per_vessel,
        "any_vessel_contact":            any_contact,
        "max_arterial_encasement_deg":   float(max_art),
        "max_venous_encasement_deg":     float(max_ven),
        "resectability_category":        category,
        **interface,
    }


def _empty_vessel_metrics() -> Dict[str, float]:
    return {
        "min_distance_mm":     float("inf"),
        "contact_voxels":      0,
        "contact_surface_mm2": 0.0,
        "contact_fraction":    0.0,
        "max_encasement_deg":  0.0,
        "contact_n_slices":    0,
        "contact_length_mm":   0.0,
        "min_csa_mm2":         0.0,
        "median_csa_mm2":      0.0,
        "stenosis_ratio":      1.0,  # 1.0 = no narrowing detected (neutral)
    }


def format_metrics_report(metrics: Mapping[str, object], case_id: str = "") -> str:
    """Return a short multi-line string summarising one case's VT metrics."""
    lines = []
    header = "  Vessel-tumour metrics" + (f" [{case_id}]" if case_id else "")
    lines.append(header)
    lines.append(f"    tumour volume        : {metrics['tumor_volume_mm3']/1000:.2f} cm³")
    lines.append(f"    tumour max diameter  : {metrics.get('tumor_max_diameter_mm', 0.0):.1f} mm")
    lines.append(f"    resectability        : {metrics['resectability_category']}")
    lines.append(
        f"    max arterial enc.    : {metrics['max_arterial_encasement_deg']:.0f}°"
        f"   max venous enc.: {metrics['max_venous_encasement_deg']:.0f}°"
    )
    if "arterial_interface_fat_frac" in metrics:
        lines.append(
            f"    interface fat frac   : "
            f"art={metrics.get('arterial_interface_fat_frac', 0.0):.2f}  "
            f"ven={metrics.get('venous_interface_fat_frac', 0.0):.2f}  "
            f"(HU art={metrics.get('arterial_interface_hu_mean', 0.0):.0f})"
        )
    per_vessel = metrics.get("per_vessel", {}) or {}
    for name, vm in per_vessel.items():
        md = vm["min_distance_mm"]
        md_str = f"{md:.1f} mm" if np.isfinite(md) else "n/a"
        lines.append(
            f"    {name:<10s}  dist={md_str:>9s}  "
            f"contact_frac={vm['contact_fraction']:.2f}  "
            f"enc={vm['max_encasement_deg']:.0f}°  "
            f"len={vm.get('contact_length_mm', 0.0):.0f}mm  "
            f"sten={vm.get('stenosis_ratio', 1.0):.2f}"
        )
    return "\n".join(lines)
