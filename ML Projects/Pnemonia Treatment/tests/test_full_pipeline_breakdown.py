"""Tests for the Stage 2 breakdown of the full-pipeline evaluation.

Regression for a crash where pneumonia images that Stage 1 predicted Normal
leaked a -1 sentinel into the binary metrics: scikit-learn then saw three
distinct labels, treated the task as multiclass, and rejected
``average="binary"``.
"""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "full_pipe_breakdown", ROOT / "scripts" / "evaluate_full_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# y_true pipeline order: 0 = Normal, 1 = Bacterial, 2 = Viral.
# Index 6 is a bacterial image that Stage 1 called Normal (a miss), which is what
# used to produce the -1 sentinel.
Y_TRUE = [0, 0, 1, 1, 2, 2, 1, 2, 1, 2]
Y_PRED = [0, 1, 1, 1, 2, 2, 0, 1, 1, 2]


def test_missed_pneumonia_is_excluded_not_fatal():
    mod = _load()
    bd = mod.stage2_breakdown(Y_TRUE, Y_PRED, np.array(Y_TRUE) != 0)

    assert bd["n_pneumonia"] == 8
    assert bd["n_routed"] == 7
    assert bd["n_missed"] == 1
    # 7 routed samples -> confusion matrix must cover exactly those.
    assert bd["confusion_matrix"].shape == (2, 2)
    assert bd["confusion_matrix"].sum() == 7


def test_routed_metrics_ignore_misses_but_overall_counts_them_wrong():
    mod = _load()
    bd = mod.stage2_breakdown(Y_TRUE, Y_PRED, np.array(Y_TRUE) != 0)

    # 6 of the 7 routed images are correct.
    assert bd["accuracy_routed"] == 6 / 7
    # The miss stays in the denominator and is never credited as correct.
    assert bd["accuracy_all"] == 6 / 8


def test_per_class_metrics_and_confusion_matrix():
    mod = _load()
    bd = mod.stage2_breakdown(Y_TRUE, Y_PRED, np.array(Y_TRUE) != 0)

    # rows = true Bacterial/Viral, cols = predicted.
    assert bd["confusion_matrix"].tolist() == [[3, 0], [1, 3]]
    # Bacterial: 3 TP, 1 viral image called bacterial.
    assert bd["precision"][0] == 3 / 4
    assert bd["recall"][0] == 3 / 3
    # Viral: 3 TP, 1 viral image called bacterial.
    assert bd["precision"][1] == 3 / 3
    assert bd["recall"][1] == 3 / 4


def test_normal_images_are_never_scored():
    mod = _load()
    # Adding normal images (correctly or not) must not move any Stage 2 number.
    y_true = [0, 0, 0] + Y_TRUE
    y_pred = [0, 1, 1] + Y_PRED
    bd = mod.stage2_breakdown(y_true, y_pred, np.array(y_true) != 0)

    assert bd["n_pneumonia"] == 8
    assert bd["n_routed"] == 7
    assert bd["confusion_matrix"].sum() == 7


def test_no_pneumonia_routed_yields_zero_overall_accuracy():
    mod = _load()
    # Stage 1 calls everything Normal: nothing reaches Stage 2.
    bd = mod.stage2_breakdown(Y_TRUE, [0] * len(Y_TRUE), np.array(Y_TRUE) != 0)

    assert bd["n_routed"] == 0
    assert bd["accuracy_all"] == 0.0
    assert np.isnan(bd["accuracy_routed"])
    assert bd["confusion_matrix"].sum() == 0


def test_blend_probs_renormalises_and_skips_absent_members():
    mod = _load()
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    b = np.array([[0.4, 0.6], [0.7, 0.3]])
    probs = {"a": a, "b": b}

    # Weights for a member that is not present must be ignored, not silently
    # inflate the total.
    got = mod.blend_probs(probs, {"a": 3.0, "b": 1.0, "missing": 5.0})
    np.testing.assert_allclose(got, 0.75 * a + 0.25 * b)

    single = mod.blend_probs({"b": b}, {"a": 3.0, "b": 1.0})
    np.testing.assert_allclose(single, b)


def test_pipeline_predict_composes_the_two_stages():
    mod = _load()
    det = np.array([[0.9, 0.1],   # -> Normal
                    [0.2, 0.8],   # -> Stage 2
                    [0.3, 0.7]])  # -> Stage 2
    cls = np.array([[0.5, 0.5],
                    [0.8, 0.2],   # -> Bacterial
                    [0.1, 0.9]])  # -> Viral

    assert mod.pipeline_predict(det, cls).tolist() == [0, 1, 2]
