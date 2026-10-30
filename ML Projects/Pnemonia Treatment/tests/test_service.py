"""Tests for pneumonia.service.

The model backbones are stubbed with fixed probabilities so these run on CPU
in milliseconds without touching the 1.05 GiB of real checkpoints. What they
pin down is the arithmetic: how member probabilities become a three-class
verdict, and how re-blending reuses a cached prediction.
"""

import numpy as np
import pytest

from pneumonia.service import (
    DEFAULT_STAGE2_WEIGHTS,
    IMAGE_SIZE,
    PIPELINE_CLASSES,
    PAPER_STAGE1_WEIGHTS,
    Prediction,
    PipelineService,
    build_model,
    blend_probs,
)


def _probs(p_normal: float) -> dict[str, np.ndarray]:
    """A Stage 1 member set where every member agrees on P(pneumonia)."""
    p_pneumonia = 1.0 - p_normal
    return {
        name: np.array([p_normal, p_pneumonia], dtype=np.float32)
        for name in PAPER_STAGE1_WEIGHTS
    }


def _stage2_probs(p_bacterial: float) -> dict[str, np.ndarray]:
    p_viral = 1.0 - p_bacterial
    return {
        name: np.array([p_bacterial, p_viral], dtype=np.float32)
        for name in DEFAULT_STAGE2_WEIGHTS
    }


def _prediction(p_normal: float, run_stage2: bool = True) -> Prediction:
    """A routed-or-not prediction, built by hand and finished through reblend."""
    stage1 = _probs(p_normal)
    stage1_blend = blend_probs(stage1, PAPER_STAGE1_WEIGHTS)
    routed = float(stage1_blend[1]) >= 0.5

    stage2 = _stage2_probs(0.75) if (run_stage2 and routed) else None
    stage2_blend = blend_probs(stage2, DEFAULT_STAGE2_WEIGHTS) if stage2 is not None else None

    # Finished through the public path so no test depends on a private method.
    return _service().reblend(
        Prediction(
            label="", confidence=0.0, class_probs={}, routed=routed,
            stage1=stage1, stage2=stage2,
            stage1_probs=stage1_blend, stage2_probs=stage2_blend,
        ),
        PAPER_STAGE1_WEIGHTS, DEFAULT_STAGE2_WEIGHTS, 0.5,
    )


# --- blend_probs -----------------------------------------------------------


def test_blend_normalises_weights():
    members = _probs(0.9)
    got = blend_probs(members, PAPER_STAGE1_WEIGHTS)

    # Every member is identical, so any weighting yields that member's value.
    assert np.allclose(got, [0.9, 0.1], atol=1e-6)


def test_blend_averages_disagreeing_members_by_weight():
    members = {
        "vgg16": np.array([0.0, 1.0], dtype=np.float32),   # weight 5
        "inception_v3": np.array([1.0, 0.0], dtype=np.float32),  # weight 4
        "pneumonet_v2": np.array([0.0, 1.0], dtype=np.float32),  # weight 3
        "pneumonet_v1": np.array([0.0, 1.0], dtype=np.float32),  # weight 1
    }
    got = blend_probs(members, PAPER_STAGE1_WEIGHTS)

    # P(pneumonia) = (5*1 + 4*0 + 3*1 + 1*1) / 13
    assert np.isclose(got[1], 9.0 / 13.0, atol=1e-6)


def test_blend_drops_members_not_in_weights():
    """An ablation that removes a member must not require re-running inference."""
    members = _probs(0.8)
    members["inception_v3"] = np.array([0.0, 1.0], dtype=np.float32)

    paper = blend_probs(members, PAPER_STAGE1_WEIGHTS)
    dropped = blend_probs(members, {k: v for k, v in PAPER_STAGE1_WEIGHTS.items()
                                    if k != "inception_v3"})

    # Dropping the one dissenting member pulls the blend toward 0.8.
    assert np.isclose(dropped[0], 0.8, atol=1e-6)
    assert paper[0] < dropped[0]


def test_blend_accumulates_in_float32():
    """Regression: float64 accumulation shifts routing decisions."""
    got = blend_probs(_probs(0.9), PAPER_STAGE1_WEIGHTS)
    assert got.dtype == np.float32


def test_blend_rejects_no_matching_members():
    with pytest.raises(ValueError):
        blend_probs(_probs(0.9), {"nonexistent": 1.0})


# --- _compose / prediction -------------------------------------------------


def test_below_threshold_reports_normal_without_stage2():
    pred = _prediction(p_normal=0.9, run_stage2=True)

    assert pred.label == "NORMAL"
    assert pred.routed is False
    assert pred.stage2 is None
    # Non-routed classes are exactly zero, not tiny residuals.
    assert pred.class_probs == {"NORMAL": pytest.approx(0.9), "BACTERIAL": 0.0, "VIRAL": 0.0}


def test_above_threshold_routes_and_splits_probability():
    pred = _prediction(p_normal=0.1)

    assert pred.routed is True
    assert pred.label == "BACTERIAL"
    # Stage 2 stub: 0.75 bacterial. P(pneumonia) = 0.9.
    assert pred.class_probs["BACTERIAL"] == pytest.approx(0.9 * 0.75)
    assert pred.class_probs["VIRAL"] == pytest.approx(0.9 * 0.25)
    # Probabilities over all three classes sum to 1.
    assert sum(pred.class_probs.values()) == pytest.approx(1.0)


def test_label_follows_the_cascade_not_a_three_way_argmax():
    """A routed image must never be labelled NORMAL.

    When Stage 1 routes, P(normal) can still exceed each split subtype because
    Stage 2 divides the pneumonia mass in two. An argmax over all three classes
    would then contradict the routing decision and report NORMAL for an image
    the pipeline already flagged as pneumonia.
    """
    # p_normal = 0.45 -> P(pneumonia) = 0.55, split 0.75/0.25 into 0.4125 / 0.1375.
    pred = _prediction(p_normal=0.45)

    assert pred.routed is True
    assert pred.label == "BACTERIAL"
    assert pred.label != "NORMAL"
    # P(normal) is still the single largest class share, which is why the
    # cascade decision has to win over the argmax.
    assert pred.class_probs["NORMAL"] > pred.class_probs["BACTERIAL"]


def test_class_order_matches_pipeline_convention():
    assert PIPELINE_CLASSES == ("NORMAL", "BACTERIAL", "VIRAL")


def test_to_dict_exposes_only_serialisable_scalars():
    pred = _prediction(p_normal=0.1)
    got = pred.to_dict()

    assert set(got["stage1_members"]) == set(PAPER_STAGE1_WEIGHTS)
    assert isinstance(got["stage1_members"]["vgg16"], float)
    assert got["routed"] is True
    # Raw member arrays stay off the wire; only the blended scalars ship.
    assert "stage1" not in got


# --- reblend ---------------------------------------------------------------


def _service() -> PipelineService:
    return PipelineService.__new__(PipelineService)


def test_reblend_recomputes_without_touching_models():
    pred = _prediction(p_normal=0.1)  # routed, stage2 cached
    rebalanced = _service().reblend(
        pred,
        stage1_weights={k: 1.0 for k in PAPER_STAGE1_WEIGHTS},
        stage2_weights=DEFAULT_STAGE2_WEIGHTS,
        threshold=0.5,
    )

    # Equal Stage 1 weights no longer outvote the dissenting member.
    assert rebalanced.routed is True
    assert rebalanced.stage1 is pred.stage1  # cached arrays reused, not rebuilt


def test_reblend_can_raise_threshold_until_normal():
    pred = _prediction(p_normal=0.1)
    rebalanced = _service().reblend(
        pred, PAPER_STAGE1_WEIGHTS, DEFAULT_STAGE2_WEIGHTS, threshold=0.99,
    )

    assert rebalanced.routed is False
    assert rebalanced.label == "NORMAL"


def test_reblend_refuses_when_stage2_was_never_run():
    pred = _prediction(p_normal=0.9, run_stage2=False)

    with pytest.raises(ValueError, match="never run"):
        _service().reblend(pred, PAPER_STAGE1_WEIGHTS, DEFAULT_STAGE2_WEIGHTS, threshold=0.1)


# --- loading ---------------------------------------------------------------


def test_build_model_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_model("vgg16", str(tmp_path / "nope.pt"))


def test_load_reports_the_missing_checkpoint_by_stage(tmp_path, monkeypatch):
    """Stage-qualified keys: short names repeat across stages."""
    monkeypatch.setenv("PNEUMONIA_CHECKPOINT_DIR", str(tmp_path))
    service = PipelineService(checkpoint_root=str(tmp_path))

    with pytest.raises(FileNotFoundError) as excinfo:
        service.load()

    assert "stage1/pneumonet_v1" in str(excinfo.value)


def test_service_health_reports_all_seven_members(tmp_path, monkeypatch):
    """Health must list every checkpoint even when none are installed."""
    monkeypatch.setenv("PNEUMONIA_CHECKPOINT_DIR", str(tmp_path))
    service = PipelineService(checkpoint_root=str(tmp_path))

    health = service.health()

    assert health["ready"] is False
    # Seven entries, keyed by stage: short names repeat across stages, so an
    # unqualified key would collapse them and under-report the count.
    assert set(health["checkpoints"]) == {
        "stage1/pneumonet_v1", "stage1/pneumonet_v2", "stage1/vgg16",
        "stage1/inception_v3", "stage2/pneumonet_v1", "stage2/mobilenet_v2",
        "stage2/inception_v3",
    }
    assert not any(v["present"] for v in health["checkpoints"].values())
    assert set(health["versions"]) == {"torch", "torchvision", "numpy"}


def test_default_weights_are_the_paper_and_equal():
    assert PAPER_STAGE1_WEIGHTS == {"vgg16": 5, "inception_v3": 4,
                                    "pneumonet_v2": 3, "pneumonet_v1": 1}
    assert set(DEFAULT_STAGE2_WEIGHTS) == {"pneumonet_v1", "mobilenet_v2", "inception_v3"}
    assert IMAGE_SIZE == 224
