"""Tests for scripts/dump_predictions.py.

Covers the two failure modes that would silently corrupt the artifact:

1. Label remapping. ImageFolder sorts classes alphabetically (BACTERIAL=0,
   NORMAL=1, VIRAL=2), but the pipeline emits NORMAL=0, BACTERIAL=1,
   VIRAL=2. Confusing the two produces a plausible-looking but wrong matrix.
2. Blend dtype. Accumulating ensemble weights in float64 rather than float32
   shifts 78 borderline images across the routing threshold, which silently
   inflates pipeline accuracy by ~3.8 points.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import dump_predictions as dp  # noqa: E402


class _FakeImageFolder:
    """Minimal stand-in exposing only what the label remap reads."""

    def __init__(self, classes):
        self.classes = classes
        self.targets = list(range(len(classes)))


def test_imagefolder_alphabetical_order_is_the_hazard():
    # Guards the premise of the remap: sorted() puts NORMAL in the middle.
    assert sorted(["NORMAL", "BACTERIAL", "VIRAL"]) == ["BACTERIAL", "NORMAL", "VIRAL"]


def test_pipeline_order_differs_from_imagefolder_order():
    # This mismatch is the entire reason y_true is remapped on write.
    assert dp.PIPELINE_ORDER != ["BACTERIAL", "NORMAL", "VIRAL"]
    assert dp.PIPELINE_ORDER == ["NORMAL", "BACTERIAL", "VIRAL"]


def test_remap_maps_each_class_to_its_pipeline_index():
    ds = _FakeImageFolder(["BACTERIAL", "NORMAL", "VIRAL"])
    remapped = dp.remap_labels(np.array(ds.targets), ds.classes)

    # targets are 0,1,2 -> BACTERIAL, NORMAL, VIRAL
    assert remapped.tolist() == [1, 0, 2]


def test_binary_labels_collapse_correctly():
    ds = _FakeImageFolder(["BACTERIAL", "NORMAL", "VIRAL"])
    remapped = dp.remap_labels(np.array(ds.targets), ds.classes)
    binary = dp.binary_labels(remapped)

    assert binary.tolist() == [1, 0, 1]  # only NORMAL is negative


def test_blend_matches_reference_sum_and_normalisation():
    probs = {
        "a": np.array([[0.9, 0.1]], dtype=np.float32),
        "b": np.array([[0.1, 0.9]], dtype=np.float32),
    }
    got = dp.blend(probs, {"a": 1.0, "b": 3.0})

    # weights renormalise to 0.25 / 0.75
    expected = 0.25 * probs["a"] + 0.75 * probs["b"]
    assert np.allclose(got, expected, atol=1e-6)


def test_blend_accumulates_in_float32_to_match_live_pipeline():
    """Regression: float64 accumulation changes the routing decision count."""
    probs = {
        "vgg16": np.full((4, 2), 0.5, dtype=np.float32),
        "inception_v3": np.full((4, 2), 0.5, dtype=np.float32),
        "pneumonet_v2": np.full((4, 2), 0.5, dtype=np.float32),
        "pneumonet_v1": np.full((4, 2), 0.5, dtype=np.float32),
    }
    got = dp.blend(probs, dp.PAPER_STAGE1_WEIGHTS)
    assert got.dtype == np.float32


def test_blend_rejects_empty_weights():
    with pytest.raises(ValueError):
        dp.blend({"a": np.zeros((1, 2), dtype=np.float32)}, {})


def test_blend_renames_members_to_canonical_stage1_keys():
    # The dump uses short names (inception_v3), so the paper's weighting
    # must key off those, not the config names (stage1_inception).
    assert set(dp.PAPER_STAGE1_WEIGHTS) == {
        "vgg16", "inception_v3", "pneumonet_v2", "pneumonet_v1"
    }
