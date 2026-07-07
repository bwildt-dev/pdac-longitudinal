"""Unit tests for the config schema and YAML loading."""

from __future__ import annotations

from pathlib import Path

from pdac_longitudinal.config import Config, DataConfig

REPO = Path(__file__).resolve().parents[1]


def test_data_defaults():
    dc = DataConfig()
    assert dc.post_nat_tps == ("t1",)                 # t1-only, no t2/t3 fallback
    assert dc.clinical_missingness_flags is False
    assert dc.clinical_completeness_weighting is False


def test_example_config_round_trip():
    cfg = Config.from_yaml(REPO / "configs" / "example_config.yaml")
    assert list(cfg.data.post_nat_tps) == ["t1"]
    # open pass-through sections load as typed dict fields the pipeline reads.
    assert cfg.roi_pipeline["ring_radii_mm"] == [0.0, 5.0, 10.0, 15.0]
    assert "feature_classes" in cfg.radiomics
    assert cfg.data.include_cohorts == ("cohort_a", "cohort_b")


def test_nested_dataclasses_are_typed():
    cfg = Config.from_yaml(REPO / "configs" / "example_config.yaml")
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.preprocessing.patch_size, tuple)
