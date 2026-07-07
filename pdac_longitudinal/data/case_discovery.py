"""Patient / timepoint discovery for the longitudinal dataset.

`CaseDiscoveryMixin` walks the NIfTI tree, resolves each patient's contrast
phase, matches T0/T1 pairs by slice thickness, and builds the case list.
"""

from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib

logger = logging.getLogger(__name__)


class CaseDiscoveryMixin:
    def _resolve_phase(self, patient_id: str) -> Optional[str]:
        """Pick the patient's effective phase from `self.phase_preference` (first with both t0 and post-NAT data)."""
        if patient_id in self._phase_for_pid:
            return self._phase_for_pid[patient_id]
        chosen: Optional[str] = None
        for ph in self.phase_preference:
            t0_dir = self.nifti_root / ph / patient_id / "t0"
            t1_dirs = [
                self.nifti_root / ph / patient_id / tp for tp in self.post_nat_tps
            ]
            if not (t0_dir.is_dir() and any(t0_dir.glob("*.nii.gz"))):
                continue
            if not any(d.is_dir() and any(d.glob("*.nii.gz")) for d in t1_dirs):
                continue
            chosen = ph
            break
        self._phase_for_pid[patient_id] = chosen
        return chosen

    def get_phase(self, patient_id: str) -> Optional[str]:
        """Public accessor for the resolved phase of `patient_id`."""
        return self._resolve_phase(patient_id)

    def _collect_series(
        self, patient_id: str, timepoint: str
    ) -> List[Tuple[int, Path, float]]:
        """Return all NIfTI candidates as `(tier_priority, path, z_mm)` tuples (`z_mm` is `inf` if unreadable)."""
        candidates: List[Tuple[int, Path, float]] = []

        # Resolve the phase once per patient so t0 and t1 share a phase.
        phase = self._resolve_phase(patient_id)
        if phase is None:
            return candidates
        tp_dir = self.nifti_root / phase / patient_id / timepoint
        if not tp_dir.exists():
            return candidates
        for path in sorted(tp_dir.glob("*.nii.gz")):
            if self._region_by_name:
                region = self._region_by_name.get(path.name, "")
                if region not in self.allowed_regions:
                    continue
                tier_idx = self.allowed_regions.index(region)
            else:
                tier_idx = 0
            try:
                z_mm = float(nib.load(str(path)).header.get_zooms()[2])
            except Exception:
                z_mm = float("inf")
            candidates.append((tier_idx, path, z_mm))
        return candidates

    def _find_matched_niftis(
        self, patient_id: str
    ) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
        """Find the best-matched T0/T1 NIfTI pair for one patient.

        Picks the first post-NAT timepoint with candidates, then the T0/T1 pair minimising
        `(max tier, thickness difference, thinnest slice)`. Returns all `None` if no match found.
        """
        t0_cands = self._collect_series(patient_id, "t0")
        if not t0_cands:
            return None, None, None

        # Walk the post-NAT fallback list; first tp with candidates wins.
        t1_cands: List[Tuple[int, Path, float]] = []
        chosen_tp: Optional[str] = None
        for tp in self.post_nat_tps:
            cands = self._collect_series(patient_id, tp)
            if cands:
                t1_cands = cands
                chosen_tp = tp
                if tp != self.post_nat_tps[0]:
                    logger.info(
                        "%s: no %s scan, falling back to %s as post-NAT",
                        patient_id, self.post_nat_tps[0], tp,
                    )
                break

        if not t1_cands:
            return None, None, None

        best_score: Optional[Tuple] = None
        best_t0: Optional[Path] = None
        best_t1: Optional[Path] = None

        for t0_tier, t0_path, t0_z in t0_cands:
            for t1_tier, t1_path, t1_z in t1_cands:
                if not (math.isfinite(t0_z) and math.isfinite(t1_z)):
                    continue
                score = (
                    max(t0_tier, t1_tier),
                    abs(t0_z - t1_z),
                    min(t0_z, t1_z),
                )
                if best_score is None or score < best_score:
                    best_score, best_t0, best_t1 = score, t0_path, t1_path

        if best_t0 is None:
            best_t0 = t0_cands[0][1]
            best_t1 = t1_cands[0][1]
            logger.warning("%s: could not read z-spacing, using first candidates", patient_id)
            return best_t0, best_t1, chosen_tp

        t0_z_best = next((z for _, p, z in t0_cands if p == best_t0), float("nan"))
        t1_z_best = next((z for _, p, z in t1_cands if p == best_t1), float("nan"))
        z_diff = abs(t0_z_best - t1_z_best)
        if z_diff > 0.5:
            logger.warning(
                "%s: T0 z=%.2f mm vs T1 z=%.2f mm — thickness mismatch %.2f mm "
                "(no better matched pair available)",
                patient_id, t0_z_best, t1_z_best, z_diff,
            )
        else:
            logger.debug(
                "%s: matched z=%.2f mm (T0) / %.2f mm (T1)", patient_id, t0_z_best, t1_z_best,
            )
        return best_t0, best_t1, chosen_tp

    def _is_patient_dir(self, name: str) -> bool:
        """Whether a directory name is a patient ID. Override to restrict by cohort."""
        return not name.startswith("_")

    # Discovery cache: pid -> [t0_path, t1_path, t1_timepoint]. Delete the file to force a re-scan.

    def _discovery_cache_path(self) -> Optional[Path]:
        base = getattr(self, "cache_dir", None) or (self.nifti_root / "_state")
        try:
            base = Path(base)
            base.mkdir(parents=True, exist_ok=True)
            return base / "discovery_cache.json"
        except Exception:
            return None

    def _discovery_cache_key(self) -> str:
        scope = ",".join(self.phase_preference) + "|" + ",".join(self.allowed_regions)
        return "|".join([
            str(self.nifti_root), scope,
            ",".join(self.post_nat_tps), str(getattr(self, "cache_version", "?")),
        ])

    def _load_discovery_cache(self) -> Tuple[Dict[str, list], str]:
        key = self._discovery_cache_key()
        path = self._discovery_cache_path()
        if path is not None and path.exists():
            try:
                blob = json.loads(path.read_text())
                if blob.get("key") == key:
                    return dict(blob.get("pids", {})), key
            except Exception:
                pass
        return {}, key

    def _save_discovery_cache(self, pids: Dict[str, list], key: str) -> None:
        path = self._discovery_cache_path()
        if path is None:
            return
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps({"key": key, "pids": pids}))
            # Atomic replace; concurrent writers race but write identical content, so last-writer-wins is fine.
            os.replace(tmp, path)
        except Exception as exc:
            logger.debug("discovery cache write failed: %s", exc)

    def _discover_cases(self, patient_ids: Optional[List[str]]) -> List[Dict]:
        """Walk the NIfTI tree and build the matched T0/T1 case list.

        Returns one dict per valid case with `patient_id`, `t0`, `t1`, `t1_timepoint`,
        `duration`, `event`; patients missing labels, T0, or T1 are excluded.
        """
        all_imaging: set = set()
        # Walk every preferred phase (not just the first) so venous-only patients
        # are discoverable; the registry check below drops anything without labels.
        for sub in self.phase_preference:
            sub_dir = self.nifti_root / sub
            if not sub_dir.exists():
                continue
            for d in sub_dir.iterdir():
                if d.is_dir() and self._is_patient_dir(d.name):
                    all_imaging.add(d.name)

        if patient_ids is not None:
            all_imaging = all_imaging & set(patient_ids)

        cases, no_t0, no_t1 = [], [], []
        fallback_counts: Dict[str, int] = {}

        labelled = [pid for pid in sorted(all_imaging) if self.registry.has(pid)]
        no_labels = [pid for pid in sorted(all_imaging) if not self.registry.has(pid)]

        # Reuse cached discovery; only pids not in the cache need header reads.
        disc_cache, cache_key = self._load_discovery_cache()
        to_read = [pid for pid in labelled if pid not in disc_cache]

        # Thread-pool the header reads for cache misses.
        if to_read:
            max_workers = min(16, (os.cpu_count() or 4) * 2)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for pid, rec in zip(to_read, ex.map(self._find_matched_niftis, to_read)):
                    t0, t1, t1_tp = rec
                    disc_cache[pid] = [str(t0) if t0 else None,
                                       str(t1) if t1 else None, t1_tp]
            self._save_discovery_cache(disc_cache, cache_key)
        logger.info("Discovery: %d cached, %d read from headers (of %d labelled)",
                    len(labelled) - len(to_read), len(to_read), len(labelled))

        for pid in labelled:
            rec = disc_cache[pid]
            t0 = Path(rec[0]) if rec[0] else None
            t1 = Path(rec[1]) if rec[1] else None
            t1_tp = rec[2]
            if t0 is None:
                no_t0.append(pid)
                continue
            if t1 is None:
                no_t1.append(pid)
                continue
            try:
                duration, event = self.registry.get_survival(pid)
            except Exception as exc:
                logger.error("get_survival failed for %s: %s", pid, exc)
                continue
            logger.debug("case OK: %s  t=%.1f  e=%d  t1_tp=%s", pid, duration, event, t1_tp)
            fallback_counts[t1_tp or "?"] = fallback_counts.get(t1_tp or "?", 0) + 1
            cases.append({
                "patient_id": pid, "t0": t0, "t1": t1,
                "t1_timepoint": t1_tp, "duration": duration, "event": event,
            })

        if fallback_counts:
            tp_breakdown = ", ".join(f"{tp}={n}" for tp, n in sorted(fallback_counts.items()))
            logger.info("Post-NAT timepoint usage: %s", tp_breakdown)

        logger.info(
            "_discover_cases: imaging=%d  no_labels=%d  no_t0=%d  no_t1=%d  valid=%d",
            len(all_imaging), len(no_labels), len(no_t0), len(no_t1), len(cases),
        )
        if no_labels:
            logger.warning("%d patients have imaging but no labels: %s", len(no_labels), no_labels)
        if no_t0:
            logger.warning("%d patients missing t0: %s", len(no_t0), no_t0)
        if no_t1:
            logger.warning("%d patients missing t1: %s", len(no_t1), no_t1)
        return cases
