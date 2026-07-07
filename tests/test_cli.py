"""Argparse smoke tests for each CLI subcommand's build_parser().

Only exercises argument parsing (no `main()` calls, which need a real config
and, for most subcommands, a GPU/training loop).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pdac_longitudinal.cli import analyze, evaluate, preprocess, train, verify
from pdac_longitudinal.cli.__main__ import _COMMANDS, main as dispatch_main
from pdac_longitudinal.preprocess import radiomics_features

_CONFIG_ONLY = (preprocess, verify, evaluate)  # --config is their only required arg


@pytest.mark.parametrize("mod", _CONFIG_ONLY)
def test_requires_config(mod):
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args([])


@pytest.mark.parametrize("mod", [preprocess, verify, train])
def test_config_and_cv_fold_parse(mod):
    args = mod.build_parser().parse_args(["--config", "cfg.yaml", "--cv-fold", "2"])
    assert args.config == Path("cfg.yaml")
    assert args.cv_fold == 2


def test_analyze_requires_checkpoint():
    args = analyze.build_parser().parse_args(
        ["--config", "cfg.yaml", "--checkpoint", "ckpt.pth", "--cv-fold", "2"]
    )
    assert args.checkpoint == Path("ckpt.pth")
    assert args.cv_fold == 2
    with pytest.raises(SystemExit):
        analyze.build_parser().parse_args(["--config", "cfg.yaml"])


def test_evaluate_requires_run_dir():
    args = evaluate.build_parser().parse_args(["--config", "cfg.yaml", "--run-dir", "runs/x"])
    assert args.run_dir == Path("runs/x")
    with pytest.raises(SystemExit):
        evaluate.build_parser().parse_args(["--config", "cfg.yaml"])


def test_train_dry_run_flag():
    args = train.build_parser().parse_args(["--config", "cfg.yaml", "--dry-run"])
    assert args.dry_run is True


def test_radiomics_features_config_is_optional():
    args = radiomics_features.build_parser().parse_args([])
    assert args.config is None
    args = radiomics_features.build_parser().parse_args(["--config", "cfg.yaml", "--limit", "5"])
    assert args.config == Path("cfg.yaml")
    assert args.limit == 5


def test_dispatch_rejects_unknown_command(capsys):
    with pytest.raises(SystemExit) as exc:
        dispatch_main(["not-a-command"])
    assert exc.value.code == 2
    assert "unknown command" in capsys.readouterr().err


def test_dispatch_help_lists_commands(capsys):
    dispatch_main(["--help"])
    out = capsys.readouterr().out
    for cmd in _COMMANDS:
        assert cmd in out
