"""Tests for checkpoint path resolution."""

import os

from pneumonia import checkpoints


def test_checkpoint_path_uses_env_root(tmp_path, monkeypatch):
    root = tmp_path / "drive" / "checkpoints"
    (root / "stage1_vgg16").mkdir(parents=True)
    target = root / "stage1_vgg16" / "final_model.pt"
    target.write_bytes(b"x")

    monkeypatch.setenv(checkpoints.ENV_VAR, str(root))

    assert checkpoints.checkpoint_root() == str(root)
    assert checkpoints.checkpoint_path("stage1_vgg16") == str(target)


def test_checkpoint_path_falls_back_to_legacy_directory(tmp_path, monkeypatch):
    root = tmp_path / "results"
    legacy = root / "stage2_checkpoints"
    legacy.mkdir(parents=True)
    target = legacy / "final_model.pt"
    target.write_bytes(b"x")

    monkeypatch.delenv(checkpoints.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    path = checkpoints.checkpoint_path("stage2_pneumonet_v1", "stage2_checkpoints")

    assert os.path.abspath(path) == os.path.abspath(target)


def test_checkpoint_path_prefers_canonical_over_legacy(tmp_path, monkeypatch):
    root = tmp_path / "results"
    for name in ("stage2_pneumonet_v1", "stage2_checkpoints"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "final_model.pt").write_bytes(b"x")

    monkeypatch.delenv(checkpoints.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    canonical = root / "stage2_pneumonet_v1" / "final_model.pt"
    resolved = checkpoints.checkpoint_path("stage2_pneumonet_v1", "stage2_checkpoints")
    assert os.path.abspath(resolved) == os.path.abspath(canonical)


def test_checkpoint_path_returns_expected_location_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv(checkpoints.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    path = checkpoints.checkpoint_path("stage1_missing")

    # Canonical path, so the error message points at where the file should be.
    assert path == os.path.join("results", "stage1_missing", "final_model.pt")
    assert not os.path.exists(path)


def test_describe_missing_lists_every_path():
    text = checkpoints.describe_missing({"a": "p/a.pt", "b": "p/b.pt"})
    assert "a: p/a.pt" in text
    assert "b: p/b.pt" in text
