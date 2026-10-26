"""Integration tests: training step, checkpoint save/load, evaluate, CLI."""

import os
import pytest
import torch

from pneumonia.data import DatasetConfig, make_synthetic_fixture, build_dataloaders
from pneumonia.models import PneumoNetV1
from pneumonia.train import train_model, save_checkpoint, load_checkpoint, get_optimizer, get_criterion
from pneumonia.evaluate import evaluate_model, compute_metrics


@pytest.fixture
def synthetic_loaders(tmp_path):
    root = os.path.join(str(tmp_path), "fixture")
    # Paper-stated input size is 224x224; smaller inputs over-reduce the conv trunk.
    paths, labels = make_synthetic_fixture(root=root, n_samples=24, image_size=224, seed=42, n_classes=2)
    cfg = DatasetConfig(image_size=224, batch_size=8, train_ratio=0.8, seed=42, num_workers=0)
    return build_dataloaders(paths, labels, cfg)


def test_single_epoch_training_completes(synthetic_loaders):
    train_loader, test_loader = synthetic_loaders
    model = PneumoNetV1(num_classes=2)
    history = train_model(
        model,
        train_loader,
        val_loader=test_loader,
        epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
    )
    assert "train_loss" in history
    assert "train_acc" in history
    assert history["train_loss"][0] > 0
    assert not (float("nan") in history["train_loss"] or float("inf") in history["train_loss"])


def test_checkpoint_save_load(synthetic_loaders, tmp_path):
    train_loader, _ = synthetic_loaders
    model = PneumoNetV1(num_classes=2)
    opt = get_optimizer(model, lr=1e-3)
    ckpt_path = os.path.join(str(tmp_path), "ckpt.pt")
    save_checkpoint(model, opt, epoch=3, path=ckpt_path)
    assert os.path.exists(ckpt_path)

    model2 = PneumoNetV1(num_classes=2)
    opt2 = get_optimizer(model2, lr=1e-3)
    load_checkpoint(ckpt_path, model2, opt2)
    # parameters should match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_evaluate_metrics(synthetic_loaders):
    _, test_loader = synthetic_loaders
    model = PneumoNetV1(num_classes=2)
    model.eval()
    metrics = evaluate_model(model, test_loader, device=torch.device("cpu"))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["precision_per_class"]) == 2
    assert "confusion_matrix" in metrics


def test_compute_metrics_hand_calculated():
    # Tiny hand-calculated example: 2 classes, perfect predictions.
    y_true = [0, 0, 1, 1]
    y_prob = [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
    ]
    import numpy as np
    m = compute_metrics(np.array(y_true), np.array(y_prob))
    assert m["accuracy"] == 1.0
    assert all(p == 1.0 for p in m["precision_per_class"])
    assert all(r == 1.0 for r in m["recall_per_class"])
    assert all(f == 1.0 for f in m["f1_per_class"])
