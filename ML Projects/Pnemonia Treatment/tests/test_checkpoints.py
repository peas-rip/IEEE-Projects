"""Tests for checkpoint path resolution."""

import os

from pneumonia import checkpoints


def _make(root, name, content=b"x"):
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    target = path / checkpoints.FILENAME
    target.write_bytes(content)
    return target


def _no_roots(monkeypatch, tmp_path):
    for var in (checkpoints.ENV_VAR,):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_env_override_wins_and_ignores_other_roots(tmp_path, monkeypatch):
    override = tmp_path / "drive"
    _make(override, "stage1_vgg16")
    (tmp_path / "checkpoints").mkdir()
    _make(tmp_path / "checkpoints", "stage1_vgg16", b"other")

    monkeypatch.setenv(checkpoints.ENV_VAR, str(override))

    assert checkpoints.root_candidates() == [str(override)]
    assert checkpoints.checkpoint_path("stage1_vgg16") == str(override / "stage1_vgg16" / checkpoints.FILENAME)


def test_checkpoints_dir_beats_models_and_results(tmp_path, monkeypatch):
    """The shipping layout is authoritative when several copies exist."""
    _no_roots(monkeypatch, tmp_path)
    _make(tmp_path / "results", "stage1_vgg16", b"legacy")
    _make(tmp_path / "Models", "stage1_vgg16", b"alt")
    canonical = _make(tmp_path / "checkpoints", "stage1_vgg16", b"canonical")

    assert os.path.basename(checkpoints.checkpoint_path("stage1_vgg16")) == checkpoints.FILENAME
    resolved = checkpoints.checkpoint_path("stage1_vgg16")
    assert open(resolved, "rb").read() == b"canonical"
    assert resolved == os.path.join("checkpoints", "stage1_vgg16", checkpoints.FILENAME)


def test_models_dir_used_when_checkpoints_absent(tmp_path, monkeypatch):
    _no_roots(monkeypatch, tmp_path)
    _make(tmp_path / "Models", "stage1_vgg16")

    resolved = checkpoints.checkpoint_path("stage1_vgg16")

    assert resolved == os.path.join("Models", "stage1_vgg16", checkpoints.FILENAME)
    assert os.path.exists(resolved)


def test_results_is_last_resort(tmp_path, monkeypatch):
    _no_roots(monkeypatch, tmp_path)
    _make(tmp_path / "results", "stage1_vgg16")

    resolved = checkpoints.checkpoint_path("stage1_vgg16")

    assert resolved == os.path.join("results", "stage1_vgg16", checkpoints.FILENAME)


def test_legacy_directory_name_still_resolves(tmp_path, monkeypatch):
    _no_roots(monkeypatch, tmp_path)
    (tmp_path / "checkpoints" / "stage2_checkpoints").mkdir(parents=True)
    (tmp_path / "checkpoints" / "stage2_checkpoints" / checkpoints.FILENAME).write_bytes(b"x")

    resolved = checkpoints.checkpoint_path(*checkpoints.CHECKPOINTS["stage2_pneumonet_v1"])

    assert "stage2_checkpoints" in resolved


def test_missing_checkpoint_reports_canonical_expectation(tmp_path, monkeypatch):
    _no_roots(monkeypatch, tmp_path)

    resolved = checkpoints.checkpoint_path("stage1_pneumonet_v1")

    assert resolved == os.path.join("checkpoints", "stage1_pneumonet_v1", checkpoints.FILENAME)
    assert not os.path.exists(resolved)


def test_inventory_separates_present_from_missing(tmp_path, monkeypatch):
    _no_roots(monkeypatch, tmp_path)
    _make(tmp_path / "checkpoints", "stage1_vgg16")

    inv = checkpoints.inventory()

    assert set(inv) == set(checkpoints.CHECKPOINTS)
    assert inv["stage1_vgg16"] is not None
    assert inv["stage2_mobilenet"] is None


def test_describe_inventory_mentions_both_states(tmp_path, monkeypatch):
    _no_roots(monkeypatch, tmp_path)
    _make(tmp_path / "checkpoints", "stage1_vgg16")

    text = checkpoints.describe_inventory()

    assert "MISSING" in text
    assert "ok" in text
    assert "stage1_vgg16" in text


def test_sha256_head_is_stable(tmp_path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"abc")

    first = checkpoints.sha256_head(str(target))
    second = checkpoints.sha256_head(str(target))

    assert first == second
    assert len(first) == 16


def test_verify_reports_whether_pneumonet_v2_matches_known_run(tmp_path, monkeypatch):
    """The wrong PneumoNet-v2 run must be detectable."""
    _no_roots(monkeypatch, tmp_path)
    target = _make(tmp_path / "checkpoints", "stage1_pneumonet_v1")
    _make(tmp_path / "checkpoints", "stage1_pneumonet_v2", b"not the right run")

    assert checkpoints.verify("stage1_pneumonet_v2") is False
    # Unhashed models only need to exist.
    assert checkpoints.verify("stage1_pneumonet_v1") is True
    # A missing model can never verify.
    assert checkpoints.verify("stage2_mobilenet") is False


def test_describe_missing_lists_every_path():
    text = checkpoints.describe_missing({"a": "p/a.pt", "b": "p/b.pt"})
    assert "a: p/a.pt" in text
    assert "b: p/b.pt" in text
