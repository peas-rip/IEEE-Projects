"""Tests for the weighted ensemble and two-stage pipeline."""

import pytest
import torch

from pneumonia.ensemble import WeightedEnsemble, DetectionEnsemble, ClassificationEnsemble, TwoStagePipeline
from pneumonia.models import PneumoNetV1


def test_weighted_ensemble_forward():
    models = [PneumoNetV1(num_classes=2) for _ in range(3)]
    ensemble = WeightedEnsemble(models, weights=[0.2, 0.3, 0.5], num_classes=2)
    ensemble.eval()
    x = torch.zeros(4, 3, 224, 224)
    with torch.no_grad():
        out = ensemble(x)
    assert out["probs"].shape == (4, 2)
    assert out["pred"].shape == (4,)
    # probs sum to 1 across classes
    assert torch.allclose(out["probs"].sum(dim=1), torch.ones(4), atol=1e-5)


def test_weight_mismatch_raises():
    models = [PneumoNetV1(num_classes=2) for _ in range(2)]
    with pytest.raises(ValueError):
        WeightedEnsemble(models, weights=[0.5], num_classes=2)


def test_detection_ensemble_output():
    ens = DetectionEnsemble(num_classes=2)
    ens.eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = ens(x)
    assert out["pred"].shape == (2,)
    assert set(out["pred"].tolist()) <= {0, 1}


def test_two_stage_pipeline_output():
    det = DetectionEnsemble(num_classes=2)
    cls = ClassificationEnsemble(num_classes=2)
    pipe = TwoStagePipeline(det, cls)
    pipe.eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = pipe(x)
    assert out["pred"].shape == (2,)
    assert set(out["pred"].tolist()) <= {0, 1, 2}  # Normal / Bacterial / Viral


def test_ensemble_weights_buffer_default_normalization():
    models = [PneumoNetV1(num_classes=2) for _ in range(2)]
    ens = WeightedEnsemble(models, num_classes=2)  # no weights -> uniform 0.5/0.5
    assert torch.allclose(ens.weights, torch.tensor([0.5, 0.5]))


def test_ensemble_weights_buffer_custom():
    models = [PneumoNetV1(num_classes=2) for _ in range(2)]
    ens = WeightedEnsemble(models, weights=[0.25, 0.75], num_classes=2)
    assert torch.allclose(ens.weights, torch.tensor([0.25, 0.75]))
