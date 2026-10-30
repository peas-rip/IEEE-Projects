"""Tests for single-image inference (``pneumonia infer``)."""

import numpy as np
import torch
from PIL import Image

from pneumonia.cli import main
from pneumonia.models import PneumoNetV1
from pneumonia.transfer import TransferLearningModel


def _write_grayscale_image(path, size=224):
    rng = np.random.RandomState(0)
    array = (rng.rand(size, size) * 255).astype("uint8")
    Image.fromarray(array, mode="L").save(path)
    return path


def test_infer_defaults_to_pneumonet_v1(tmp_path, capsys):
    checkpoint = tmp_path / "pneumonet.pt"
    torch.save(PneumoNetV1(num_classes=2).state_dict(), checkpoint)
    image = _write_grayscale_image(tmp_path / "xray.png")

    code = main(["infer", "--image", str(image), "--checkpoint", str(checkpoint)])

    output = capsys.readouterr().out
    assert code == 0
    assert "pneumonet_v1" in output
    assert "Predicted class:" in output
    assert "Probabilities:" in output


def test_infer_supports_transfer_learning_checkpoints(tmp_path, capsys):
    """--model must let any of the seven checkpoints load, not just PneumoNet."""
    checkpoint = tmp_path / "mobilenet.pt"
    model = TransferLearningModel(name="mobilenet_v2", num_classes=2, pretrained=False)
    torch.save(model.state_dict(), checkpoint)
    image = _write_grayscale_image(tmp_path / "xray.png")

    code = main([
        "infer",
        "--image", str(image),
        "--checkpoint", str(checkpoint),
        "--model", "mobilenet_v2",
        "--labels", "BACTERIAL,VIRAL",
    ])

    output = capsys.readouterr().out
    assert code == 0
    assert "mobilenet_v2" in output
    assert "BACTERIAL" in output or "VIRAL" in output


def test_infer_rejects_wrong_number_of_labels(tmp_path, capsys):
    checkpoint = tmp_path / "pneumonet.pt"
    torch.save(PneumoNetV1(num_classes=2).state_dict(), checkpoint)
    image = _write_grayscale_image(tmp_path / "xray.png")

    code = main([
        "infer",
        "--image", str(image),
        "--checkpoint", str(checkpoint),
        "--labels", "NORMAL,BACTERIAL,VIRAL",
    ])

    assert code == 1
    assert "3 labels" in capsys.readouterr().out
