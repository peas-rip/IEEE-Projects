"""Tests for the data module: transforms, synthetic fixture, dataloaders."""

import os
import pytest
import torch

from pneumonia.data import (
    DatasetConfig,
    make_synthetic_fixture,
    get_transforms,
    build_dataloaders,
    PneumoniaDataset,
)


def test_make_synthetic_fixture(tmp_path):
    root = os.path.join(str(tmp_path), "fixture")
    paths, labels = make_synthetic_fixture(root=root, n_samples=32, image_size=32, seed=42, n_classes=2)
    assert len(paths) == 32
    assert len(labels) == 32
    assert set(labels) == {0, 1}
    assert all(os.path.exists(p) for p in paths)


def test_get_transforms_shapes():
    train_tf, eval_tf = get_transforms(image_size=64, augment=True)
    from PIL import Image, ImageOps
    img = Image.new("L", (32, 32), color=100)
    img = Image.merge("RGB", (img, img, img))  # grayscale -> 3 channels (matches dataset)
    t = train_tf(img)
    e = eval_tf(img)
    assert t.shape == (3, 64, 64)
    assert e.shape == (3, 64, 64)


def test_build_dataloaders_shapes_and_dtypes(tmp_path):
    root = os.path.join(str(tmp_path), "fixture")
    paths, labels = make_synthetic_fixture(root=root, n_samples=64, image_size=64, seed=42, n_classes=2)
    cfg = DatasetConfig(image_size=64, batch_size=8, train_ratio=0.8, seed=42, num_workers=0)
    train_loader, test_loader = build_dataloaders(paths, labels, cfg)
    for images, labels_batch in train_loader:
        assert images.shape == (8, 3, 64, 64)
        assert images.dtype == torch.float32
        assert labels_batch.shape == (8,)
        break


def test_dataset_item(tmp_path):
    root = os.path.join(str(tmp_path), "fixture")
    paths, labels = make_synthetic_fixture(root=root, n_samples=8, image_size=32, seed=42, n_classes=2)
    from torchvision import transforms
    ds = PneumoniaDataset(paths[:4], labels[:4], transform=transforms.ToTensor())
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert label in {0, 1}
