"""Tests for the ensemble scripts' missing-member handling."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(script_name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / script_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_mode_returns_none_when_any_member_missing(tmp_path):
    mod = _load("evaluate_stage1_ensemble.py", "stage1_ens")
    (tmp_path / "a.pt").write_bytes(b"x")
    members = {"a": str(tmp_path / "a.pt"), "b": str(tmp_path / "b.pt")}
    assert mod.available_members(members, allow_missing=False) is None


def test_allow_missing_drops_absent_members(tmp_path):
    mod = _load("evaluate_stage1_ensemble.py", "stage1_ens")
    for name in ("a", "b"):
        (tmp_path / f"{name}.pt").write_bytes(b"x")
    members = {
        "a": str(tmp_path / "a.pt"),
        "b": str(tmp_path / "b.pt"),
        "c": str(tmp_path / "c.pt"),
    }
    got = mod.available_members(members, allow_missing=True)
    assert set(got) == {"a", "b"}


def test_allow_missing_requires_two_members(tmp_path):
    mod = _load("evaluate_stage2_ensemble.py", "stage2_ens")
    (tmp_path / "a.pt").write_bytes(b"x")
    members = {"a": str(tmp_path / "a.pt"), "b": str(tmp_path / "b.pt")}
    assert mod.available_members(members, allow_missing=True) is None


def test_full_pipeline_filters_members(tmp_path):
    mod = _load("evaluate_full_pipeline.py", "full_pipe")
    for name in ("a", "b"):
        (tmp_path / f"{name}.pt").write_bytes(b"x")
    members = {
        "a": str(tmp_path / "a.pt"),
        "b": str(tmp_path / "b.pt"),
        "c": str(tmp_path / "c.pt"),
    }
    assert list(mod.available_members(members, allow_missing=True)) == ["a", "b"]
