"""Unit tests for PneumoNet-v1 and PneumoNet-v2 custom CNNs."""

import pytest
import torch

from pneumonia.models import PneumoNetV1, PneumoNetV2


@pytest.mark.parametrize("ModelCls", [PneumoNetV1, PneumoNetV2])
def test_forward_output_shape(ModelCls):
    model = ModelCls(num_classes=2, in_channels=3)
    model.eval()
    x = torch.zeros(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2), f"Expected (4,2) got {out.shape}"


@pytest.mark.parametrize("ModelCls", [PneumoNetV1, PneumoNetV2])
def test_forward_dtype_and_device(ModelCls):
    model = ModelCls(num_classes=2, in_channels=3)
    model.eval()
    x = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert out.dtype == torch.float32


def test_models_have_parameters():
    for ModelCls in [PneumoNetV1, PneumoNetV2]:
        model = ModelCls(num_classes=2)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0


def test_feature_size_matches_classifier_input():
    # Ensure the conv trunk output matches the Linear layer input assumption (512*7*7).
    from pneumonia.models import _infer_feat_size
    for ModelCls in [PneumoNetV1, PneumoNetV2]:
        model = ModelCls(num_classes=2)
        feat = _infer_feat_size(model)
        assert feat == 512 * 7 * 7, f"{ModelCls.__name__} feat size {feat} != {512*7*7}"
