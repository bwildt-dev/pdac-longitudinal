"""CPU radiomics extraction, appended to the CT cache.

Runs automatically as the last step of `pdac_longitudinal preprocess`: reads the
original HU CTs and the segs that stage already saved, and writes per-compartment
radiomics back into each patient's .npz. Also runnable standalone via
`python -m pdac_longitudinal.preprocess.radiomics_features`, e.g. to re-extract onto an
existing cache or in a separate torch-free environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PID_RE = re.compile(r"^(.+?)_(arterial|venous)_")


_VASCULAR_LABELS: Dict[str, int] = {
    "superior_mesenteric_artery": 3, "celiac_artery": 4, "veins": 5, "postcava": 6,
}
_RING_RADII_MM = (0.0, 5.0, 10.0, 15.0)
_TVI_TUMOUR_MM = 10.0
_TVI_VESSEL_MM = 10.0


def _edt_from_mask(binary_mask: np.ndarray, spacing_xyz) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    mb = binary_mask.astype(bool)
    if not mb.any():
        return np.zeros(binary_mask.shape, dtype=np.float64)
    return distance_transform_edt(~mb, sampling=tuple(spacing_xyz))


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Largest connected component. Drops the segmenter's spurious FP blobs."""
    from scipy.ndimage import label
    if not mask.any():
        return mask
    lab, n = label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel()); counts[0] = 0
    return lab == int(counts.argmax())


def _pt_rings(it_mask: np.ndarray, spacing_xyz, ring_radii_mm=_RING_RADII_MM) -> List[np.ndarray]:
    dist = _edt_from_mask(it_mask, spacing_xyz)
    return [(dist > a) & (dist <= b)
            for a, b in zip(ring_radii_mm[:-1], ring_radii_mm[1:])]


def _tvi_mask(it_mask: np.ndarray, vessel_union: np.ndarray, spacing_xyz,
              tvi_tumour_mm=_TVI_TUMOUR_MM, tvi_vessel_mm=_TVI_VESSEL_MM) -> np.ndarray:
    if not vessel_union.any():
        return np.zeros_like(it_mask, dtype=bool)
    near_t = _edt_from_mask(it_mask, spacing_xyz) <= tvi_tumour_mm
    near_v = _edt_from_mask(vessel_union, spacing_xyz) <= tvi_vessel_mm
    return (near_t & near_v) & ~it_mask.astype(bool)


def _load_nii(path: Path):
    import nibabel as nib
    nii = nib.load(str(path))
    arr = np.asarray(nii.get_fdata(), dtype=np.float32)        # native order (XYZ)
    spacing_xyz = tuple(float(z) for z in nii.header.get_zooms()[:3])
    return nii, arr, spacing_xyz


def _seg_on_ct_grid(seg_path: Path, ct_nii) -> np.ndarray:
    """Load a seg and guarantee it sits on the CT voxel grid """
    import nibabel as nib
    from nibabel.processing import resample_from_to
    seg_nii = nib.load(str(seg_path))
    same_shape = seg_nii.shape[:3] == ct_nii.shape[:3]
    same_aff = np.allclose(seg_nii.affine, ct_nii.affine, atol=1e-3)
    if not (same_shape and same_aff):
        seg_nii = resample_from_to(seg_nii, (ct_nii.shape[:3], ct_nii.affine), order=0)
    return np.asarray(seg_nii.get_fdata(), dtype=np.int16)


def _masks_from_seg(seg: np.ndarray, spacing_xyz, roi_cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Build every radiomics compartment from the native seg.

    IT/PT_ring1-3/TVI are anchored to the tumour (largest CC of label 2); pancreas
    and liver come straight from their labels.  ROIs with <8 voxels are dropped.
    """
    out: Dict[str, np.ndarray] = {}

    tumor = _largest_cc(seg == 2)
    if tumor.sum() >= 8:
        out["IT"] = tumor.astype(np.uint8)
        ring_radii_mm = roi_cfg.get("ring_radii_mm", _RING_RADII_MM)
        for i, ring in enumerate(_pt_rings(tumor, spacing_xyz, ring_radii_mm), start=1):
            if ring.sum() >= 8:
                out[f"PT_ring{i}"] = ring.astype(np.uint8)
        vessel_union = np.zeros_like(tumor, dtype=bool)
        for lab in _VASCULAR_LABELS.values():
            vessel_union |= (seg == lab)
        tvi = _tvi_mask(
            tumor, vessel_union, spacing_xyz,
            roi_cfg.get("tvi_tumour_mm", _TVI_TUMOUR_MM),
            roi_cfg.get("tvi_vessel_mm", _TVI_VESSEL_MM),
        )
        if tvi.sum() >= 8:
            out["TVI"] = tvi.astype(np.uint8)

    for comp, lab in (("pancreas", 1), ("liver", 13)):
        m = (seg == lab)
        if m.sum() >= 8:
            out[comp] = m.astype(np.uint8)
    return out


def _extract_one(extractor, ct_path: Path, seg_path: Path, prefix: str,
                  roi_cfg: Dict[str, Any], compartments: Optional[List[str]]) -> Dict[str, float]:
    ct_nii, ct, spacing = _load_nii(ct_path)
    seg = _seg_on_ct_grid(seg_path, ct_nii)
    masks = _masks_from_seg(seg, spacing, roi_cfg)
    if not masks:
        logger.warning("  %s: no ROI labels present in seg — skipping timepoint", prefix)
        return {}
    return extractor.extract_all_compartments(
        ct_array=ct, roi_masks=masks, spacing_xyz=spacing, prefix=prefix,
        compartments=compartments,
    )


def _append_to_npz(npz_path: Path, feats: Dict[str, float]) -> None:
    """Rewrite the .npz with an added ``radiomic_features_json`` payload."""
    with np.load(npz_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["radiomic_features_json"] = np.frombuffer(
        json.dumps(feats).encode("utf-8"), dtype=np.uint8
    )
    # NB: np.savez_compressed APPENDS ".npz" unless given a file handle, so write
    # through an open handle to control the exact tmp name, then atomic-swap.
    tmp = npz_path.parent / (npz_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **data)
    tmp.replace(npz_path)


def _discover_pid_to_ct(config_path: Path) -> Dict[str, Tuple[str, str]]:
    """Reuse the training discovery (needs torch) to map pid → (t0_ct, t1_ct)."""
    from pdac_longitudinal.config import Config
    from pdac_longitudinal.data.registry import ClinicalRegistry
    from pdac_longitudinal.data.longitudinal_dataset import LongitudinalCTDataset

    cfg = Config.from_yaml(str(config_path))
    dc = cfg.data
    registry = ClinicalRegistry(dc.labels_csv, include_cohorts=dc.include_cohorts)
    discovery = LongitudinalCTDataset(
        nifti_root=dc.nifti_dir or dc.root_dir,
        registry=registry,
        phase=dc.phase,
        allowed_regions=list(dc.allowed_regions),
        post_nat_tps=list(dc.post_nat_tps),
    )
    mapping = {c["patient_id"]: (str(c["t0"]), str(c["t1"])) for c in discovery.cases}
    logger.info("Discovery resolved %d pid→CT mappings", len(mapping))
    return mapping


def _load_pid_to_ct(pid_map: Optional[Path], config_path: Optional[Path]
                    ) -> Dict[str, Tuple[Path, Path]]:
    """Use the cached JSON map if present (torch-free); else build via discovery
    and write it to ``pid_map`` for reuse."""
    if pid_map and pid_map.exists():
        raw = json.loads(pid_map.read_text())
        logger.info("Loaded %d pid→CT mappings from %s", len(raw), pid_map)
        return {pid: (Path(t0), Path(t1)) for pid, (t0, t1) in raw.items()}
    if not config_path:
        raise SystemExit(
            "No --pid-map cache found and no --config to build one.\n"
            "Generate it once in the uv env:  --dump-only --config <yaml> --pid-map <json>"
        )
    raw = _discover_pid_to_ct(config_path)
    if pid_map:
        pid_map.parent.mkdir(parents=True, exist_ok=True)
        pid_map.write_text(json.dumps(raw, indent=2))
        logger.info("Wrote pid→CT map → %s", pid_map)
    return {pid: (Path(t0), Path(t1)) for pid, (t0, t1) in raw.items()}


def run_radiomics_extraction(
    config_path: Path,
    cache_dir: Path,
    version: str,
    resample_mm: float = 1.5,
    redo: bool = False,
) -> Tuple[int, int, int]:
    """Extract + append radiomics for every cached npz under `cache_dir`.

    The plain extraction path, without sharding or W&B; used by `preprocess`'s
    in-process step. Requires pyradiomics; import it before calling this.

    Returns:
        `(n_ok, n_skip, n_fail)` patient counts.
    """
    from pdac_longitudinal.config import Config
    from pdac_longitudinal.radiomics.extractor import RadiomicsExtractor

    cfg = Config.from_yaml(str(config_path))
    roi_cfg = dict(cfg.roi_pipeline)
    radiomics_cfg = dict(cfg.radiomics)

    extractor = RadiomicsExtractor(
        feature_classes=radiomics_cfg.get("feature_classes"),
        settings_file=radiomics_cfg.get("settings_file"),
        binWidth=radiomics_cfg.get("bin_width", 25.0),
        sigma_values=radiomics_cfg.get("sigma_values") or None,
        resegment_range=radiomics_cfg.get("resegment_range"),
    )
    compartments = radiomics_cfg.get("compartments")
    if resample_mm and resample_mm > 0:
        extractor._extractor.settings["resampledPixelSpacing"] = [resample_mm] * 3
        extractor._extractor.settings["interpolator"] = "sitkBSpline"
        extractor._extractor.settings["preCrop"] = True

    cache_dir = Path(cache_dir)
    pid2ct = _load_pid_to_ct(None, config_path)
    files = sorted(cache_dir.glob(f"*_{version}.npz"))
    logger.info("radiomics: found %d v%s npz files in %s", len(files), version, cache_dir)

    n_ok = n_skip = n_fail = 0
    for f in files:
        m = _PID_RE.match(f.name)
        if not m:
            n_skip += 1; continue
        pid = m.group(1)
        if pid not in pid2ct:
            n_skip += 1; continue
        ct_t0, ct_t1 = pid2ct[pid]
        seg_t0 = cache_dir / f"{pid}_seg_T0.nii.gz"
        seg_t1 = cache_dir / f"{pid}_seg_T1.nii.gz"
        if not (seg_t0.exists() and seg_t1.exists()):
            n_skip += 1; continue
        if not redo:
            with np.load(f, allow_pickle=False) as _z:
                if "radiomic_features_json" in _z.files:
                    n_skip += 1; continue
        try:
            feats: Dict[str, float] = {}
            feats.update(_extract_one(extractor, ct_t0, seg_t0, "T0_", roi_cfg, compartments))
            feats.update(_extract_one(extractor, ct_t1, seg_t1, "T1_", roi_cfg, compartments))
            if not feats:
                n_skip += 1; continue
            _append_to_npz(f, feats)
            n_ok += 1
        except Exception as exc:
            logger.warning("radiomics: %s failed: %s", pid, exc)
            n_fail += 1

    logger.info("radiomics: DONE ok=%d skip=%d fail=%d", n_ok, n_skip, n_fail)
    return n_ok, n_skip, n_fail


def _merge_from_cache(cache: Path, version: str, out: Path) -> None:
    """Build the final table by reading ``radiomic_features_json`` back from every
    npz.  The npz is the authoritative store, so this is robust to sharded/resumed
    runs (no reliance on per-shard intermediates)."""
    files = sorted(cache.glob(f"*_{version}.npz"))
    rows: List[Dict[str, float]] = []
    n_missing = 0
    for f in files:
        m = _PID_RE.match(f.name)
        if not m:
            continue
        with np.load(f, allow_pickle=False) as z:
            if "radiomic_features_json" not in z.files:
                n_missing += 1
                continue
            feats = json.loads(bytes(z["radiomic_features_json"]).decode("utf-8"))
        feats["patient_id"] = m.group(1)
        rows.append(feats)
    if not rows:
        raise SystemExit(f"No npz under {cache} has radiomic_features_json yet.")
    df = pd.DataFrame(rows).set_index("patient_id").sort_index()
    df = df.dropna(axis=1, how="all")
    out.parent.mkdir(parents=True, exist_ok=True)
    (df.to_parquet(out) if out.suffix == ".parquet" else df.to_csv(out))
    cols_path = out.with_name("radiomic_feature_cols.json")
    cols_path.write_text(json.dumps(sorted(df.columns), indent=2))
    logger.info("Merged from cache: %d patients × %d features → %s  (%d npz not yet done)",
                len(df), df.shape[1], out, n_missing)
    logger.info("Canonical schema (%d cols) → %s", df.shape[1], cols_path)


def _init_wandb(args, n_files: int):
    """Start a fail-soft W&B run (one per shard, grouped by the array job).
    Returns the wandb module on success, else None — never raises."""
    try:
        import wandb
    except Exception as exc:
        logger.warning("W&B requested but not importable (%s) — install into the "
                       "radiomics venv:  uv pip install --python .venv-radiomics/bin/python "
                       "wandb.  Continuing without it.", exc)
        return None
    array_job = os.getenv("SLURM_ARRAY_JOB_ID")
    group = f"radiomics-{array_job}" if array_job else None
    name = (f"shard{args.shard_id}" if args.num_shards > 1 else "full")
    try:
        wandb.init(
            project=args.wandb_project,
            entity=os.getenv("WANDB_ENTITY") or None,
            name=name,
            group=group,
            job_type="radiomics-cache",
            mode=os.getenv("WANDB_MODE") or "online",
            config={
                "version": args.version, "resample_mm": args.resample_mm,
                "num_shards": args.num_shards, "shard_id": args.shard_id,
                "n_files_shard": n_files, "redo": args.redo,
                "cache_dir": str(args.cache_dir),
            },
        )
        logger.info("W&B run: %s/%s (group=%s)", args.wandb_project, name, group)
        return wandb
    except Exception as exc:
        logger.warning("W&B init failed (%s); continuing without it.", exc)
        return None


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the standalone radiomics extractor."""
    p = argparse.ArgumentParser(
        prog="python -m pdac_longitudinal.preprocess.radiomics_features",
        description="Extract per-compartment radiomics and append them to the CT cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config: supplies cache_dir/version + cfg.roi_pipeline/"
                        "cfg.radiomics, and drives pid-to-CT discovery for --dump-only.")
    p.add_argument("--pid-map", type=Path, default=None,
                   help="JSON cache of pid→[t0,t1] CT paths (torch-free reuse).")
    p.add_argument("--dump-only", action="store_true",
                   help="Build the pid→CT map (needs --config) and exit; no extraction.")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Dir with the *_<version>.npz + *_seg_T?.nii.gz (default: cfg.data.cache_dir).")
    p.add_argument("--version", default=None,
                   help="Cache version tag in the npz filenames (default: cfg.data.cache_version).")
    p.add_argument("--out", type=Path, default=None,
                   help="Table output path (default: data/radiomics_<version>.parquet).")
    p.add_argument("--limit", type=int, default=0, help="Process only N (smoke test).")
    p.add_argument("--no-append", action="store_true",
                   help="Write the table only; do NOT modify the .npz files.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split the file list into N strided shards (SLURM array).")
    p.add_argument("--shard-id", type=int, default=0, help="This task's shard index [0, N).")
    p.add_argument("--merge", action="store_true",
                   help="Rebuild the table + schema from the npz cache (after the array finishes).")
    p.add_argument("--resample-mm", type=float, default=1.5,
                   help="Isotropic spacing PyRadiomics resamples to before extraction; 0 disables.")
    p.add_argument("--redo", action="store_true",
                   help="Re-extract even if the npz already has radiomic_features_json.")
    p.add_argument("--wandb", action="store_true",
                   help="Log per-patient timing + progress to W&B (one run per shard).")
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "pdac-radiomics-cache"))
    return p


def main(argv: Optional[list] = None) -> None:
    """Entry point for the standalone radiomics extractor.

    Args:
        argv: Command-line args; defaults to `sys.argv[1:]`.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    pid_map = args.pid_map
    config_path = args.config

    # --dump-only: produce the pid→CT JSON in the uv env, then stop (no pyradiomics).
    if args.dump_only:
        _load_pid_to_ct(pid_map, config_path)
        logger.info("dump-only complete")
        return

    # cache_dir/version + cfg.roi_pipeline/cfg.radiomics from the YAML (torch-free);
    # explicit flags override, else fall back to config, then the hardcoded defaults.
    roi_cfg: Dict[str, Any] = {}
    radiomics_cfg: Dict[str, Any] = {}
    if config_path:
        from pdac_longitudinal.config import Config
        cfg = Config.from_yaml(str(config_path))
        roi_cfg = dict(cfg.roi_pipeline)
        radiomics_cfg = dict(cfg.radiomics)
        if args.cache_dir is None and cfg.data.cache_dir is not None:
            args.cache_dir = Path(cfg.data.cache_dir)
        if args.version is None:
            args.version = cfg.data.cache_version
    if args.version is None:
        args.version = "v3.19"
    if args.out is None:
        args.out = Path(f"data/radiomics_{args.version}.parquet")

    # --merge: materialise the final table from the npz cache → out + schema.
    if args.merge:
        if args.cache_dir is None:
            raise SystemExit("--merge needs --cache-dir or --config (reads radiomic_features_json from the npz)")
        _merge_from_cache(Path(args.cache_dir), args.version, Path(args.out))
        return

    if args.cache_dir is None:
        raise SystemExit("--cache-dir or --config (with data.cache_dir) is required for extraction")

    # pyradiomics imported lazily so --dump-only never needs it
    try:
        from pdac_longitudinal.radiomics.extractor import RadiomicsExtractor
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"Could not import RadiomicsExtractor ({exc}).\n"
            "Use a Python 3.9 uv env so pyradiomics installs as a prebuilt wheel:\n"
            "  uv venv --python 3.9 .venv-radiomics\n"
            "  uv pip install --python .venv-radiomics/bin/python pyradiomics "
            "'numpy<2' SimpleITK scipy PyWavelets pykwalify six nibabel pandas pyarrow"
        )

    extractor = RadiomicsExtractor(
        feature_classes=radiomics_cfg.get("feature_classes"),
        settings_file=radiomics_cfg.get("settings_file"),
        binWidth=radiomics_cfg.get("bin_width", 25.0),
        sigma_values=radiomics_cfg.get("sigma_values") or None,
        resegment_range=radiomics_cfg.get("resegment_range"),
    )
    compartments = radiomics_cfg.get("compartments")
    if args.resample_mm and args.resample_mm > 0:
        # Resample to isotropic spacing for IBSI comparability across native spacings.
        # Memory-heavy on big organs; run on fat_rome.
        extractor._extractor.settings["resampledPixelSpacing"] = [args.resample_mm] * 3
        extractor._extractor.settings["interpolator"] = "sitkBSpline"
        extractor._extractor.settings["preCrop"] = True
        logger.info("PyRadiomics resampling=%.2f mm isotropic, preCrop=on", args.resample_mm)

    cache = Path(args.cache_dir)
    pid2ct = _load_pid_to_ct(pid_map, config_path)

    files = sorted(cache.glob(f"*_{args.version}.npz"))
    if args.limit:
        files = files[: args.limit]
    n_total = len(files)
    if args.num_shards > 1:
        files = files[args.shard_id :: args.num_shards]   # strided slice, so each shard writes disjoint npz files
        logger.info("Shard %d/%d: %d of %d files",
                    args.shard_id, args.num_shards, len(files), n_total)
    logger.info("Found %d v%s npz files in %s", n_total, args.version, cache)

    wb = _init_wandb(args, len(files)) if args.wandb else None

    rows: List[Dict[str, float]] = []
    n_ok = n_skip = n_fail = 0
    t_start = time.perf_counter()
    for i, f in enumerate(files):
        m = _PID_RE.match(f.name)
        if not m:
            logger.warning("Skip (cannot parse pid): %s", f.name); n_skip += 1; continue
        pid = m.group(1)
        if pid not in pid2ct:
            logger.warning("Skip %s: no CT path from discovery", pid); n_skip += 1; continue
        ct_t0, ct_t1 = pid2ct[pid]
        seg_t0 = cache / f"{pid}_seg_T0.nii.gz"
        seg_t1 = cache / f"{pid}_seg_T1.nii.gz"
        if not (seg_t0.exists() and seg_t1.exists()):
            logger.warning("Skip %s: missing seg(s)", pid); n_skip += 1; continue
        if not args.redo and not args.no_append:
            with np.load(f, allow_pickle=False) as _z:   # lazy: reads only the zip index
                if "radiomic_features_json" in _z.files:
                    n_skip += 1; continue
        try:
            t_pat = time.perf_counter()
            feats: Dict[str, float] = {}
            feats.update(_extract_one(extractor, ct_t0, seg_t0, "T0_", roi_cfg, compartments))
            feats.update(_extract_one(extractor, ct_t1, seg_t1, "T1_", roi_cfg, compartments))
            if not feats:
                logger.warning("Skip %s: no features extracted", pid); n_skip += 1; continue
            if not args.no_append:
                _append_to_npz(f, feats)
            row = dict(feats); row["patient_id"] = pid
            rows.append(row); n_ok += 1
            dt = time.perf_counter() - t_pat
            if wb is not None:
                n_comp = len({k.split("_", 2)[1] for k in feats if k[:1] == "T"})
                wb.log({"patient_sec": dt, "n_features": len(feats), "n_compartments": n_comp,
                        "ok": n_ok, "skip": n_skip, "fail": n_fail,
                        "done_frac": (i + 1) / max(len(files), 1)}, step=i)
        except Exception as exc:
            logger.warning("%s failed: %s", pid, exc); n_fail += 1
            if wb is not None:
                wb.log({"fail": n_fail, "ok": n_ok, "skip": n_skip}, step=i)
        if (i + 1) % 25 == 0:
            logger.info("  %d / %d (ok=%d skip=%d fail=%d)", i + 1, len(files), n_ok, n_skip, n_fail)

    logger.info("DONE  ok=%d skip=%d fail=%d  (appended to npz: %s)",
                n_ok, n_skip, n_fail, not args.no_append)
    if wb is not None:
        elapsed = time.perf_counter() - t_start
        wb.summary.update({"total_ok": n_ok, "total_skip": n_skip, "total_fail": n_fail,
                           "elapsed_sec": elapsed,
                           "sec_per_ok": (elapsed / n_ok) if n_ok else None})
        wb.finish()

    if args.no_append:
        if not rows:
            raise SystemExit("No radiomic features extracted — check CT paths / segs / labels.")
        df = pd.DataFrame(rows).set_index("patient_id").sort_index().dropna(axis=1, how="all")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        (df.to_parquet(out) if out.suffix == ".parquet" else df.to_csv(out))
        cols_path = out.with_name("radiomic_feature_cols.json")
        cols_path.write_text(json.dumps(sorted(df.columns), indent=2))
        logger.info("Wrote %d patients × %d radiomic features → %s", len(df), df.shape[1], out)
        logger.info("Canonical schema (%d cols) → %s", df.shape[1], cols_path)
    elif args.num_shards == 1:
        logger.info("Run `--merge --cache-dir %s --out %s` to materialise the table.",
                    cache, args.out)


if __name__ == "__main__":
    main()
