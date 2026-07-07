"""Unit tests for preprocess.py's optional in-process radiomics step."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

from pdac_longitudinal.cli import preprocess


def _cfg(cache_dir):
    return SimpleNamespace(data=SimpleNamespace(cache_dir=cache_dir, cache_version="v1"))


def test_skips_when_pyradiomics_not_importable(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "radiomics", None)  # forces ImportError on `import radiomics`
    called = []
    monkeypatch.setattr(
        "pdac_longitudinal.preprocess.radiomics_features.run_radiomics_extraction",
        lambda **kw: called.append(kw),
    )
    preprocess._run_radiomics_if_available(Path("cfg.yaml"), _cfg(Path("/tmp/cache")))
    assert not called
    assert "skipping in-process extraction" in caplog.text


def test_skips_when_cache_dir_unset(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "radiomics", types.ModuleType("radiomics"))
    called = []
    monkeypatch.setattr(
        "pdac_longitudinal.preprocess.radiomics_features.run_radiomics_extraction",
        lambda **kw: called.append(kw),
    )
    preprocess._run_radiomics_if_available(Path("cfg.yaml"), _cfg(None))
    assert not called
    assert "cache_dir is unset" in caplog.text


def test_runs_extraction_when_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "radiomics", types.ModuleType("radiomics"))
    called = []
    monkeypatch.setattr(
        "pdac_longitudinal.preprocess.radiomics_features.run_radiomics_extraction",
        lambda **kw: called.append(kw),
    )
    preprocess._run_radiomics_if_available(Path("cfg.yaml"), _cfg(Path("/tmp/cache")))
    assert len(called) == 1
    assert called[0]["config_path"] == Path("cfg.yaml")
    assert called[0]["cache_dir"] == Path("/tmp/cache")
    assert called[0]["version"] == "v1"
