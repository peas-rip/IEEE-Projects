"""Data loading, preprocessing, splits, and synthetic fixtures.

Implements the data contract from paper Section 3.2 (PAPER-STATED):
- Guangzhou Women and Children's Medical Center (Kermany) dataset.
- 5,856 paediatric chest X-rays (ages 1-5): Normal 1,583; Pneumonia 4,280
  (Bacterial 2,766; Viral 1,514).
- Images resized to 224x224, normalized; augmentation: flipping, scaling,
  brightness/contrast, shear.
- Train/test split 80:20 (stratified).
- Augmented training counts: bacterial 5,781; viral 3,149.

Because the real dataset is gated behind an author request (paper Data
Availability section), a synthetic fixture generator is provided for
smoke tests and unit tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


@dataclass
class DatasetConfig:
    image_size: int = 224
    batch_size: int = 32
    train_ratio: float = 0.8
    seed: int = 42
    num_workers: int = 0
    grayscale_to_rgb: bool = True
    augment: bool = True


class PneumoniaDataset(Dataset):
    """Dataset of chest X-ray images with integer labels.

    Label encoding (binary tasks):
      - Detection: 0 = Normal, 1 = Pneumonia
      - Classification: 0 = Bacterial, 1 = Viral
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform: Optional[Callable] = None,
        grayscale_to_rgb: bool = True,
    ):
        self.image_paths = list(image_paths)
        self.labels = [int(l) for l in labels]
        self.transform = transform
        self.grayscale_to_rgb = grayscale_to_rgb

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("L")  # load as grayscale
        if self.grayscale_to_rgb:
            image = ImageOps.grayscale(image)  # ensure single channel
            image = Image.merge("RGB", (image, image, image))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def get_transforms(image_size: int = 224, augment: bool = True):
    """Build torchvision transforms.

    Paper augmentation (PAPER-STATED): horizontal flipping, random scaling,
    brightness/contrast adjustments, shear. We approximate with
    RandomHorizontalFlip, RandomAffine (scale/translate/shear),
    ColorJitter (brightness/contrast), then Resize, ToTensor, Normalize.
    """
    if augment:
        train_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=10
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def make_synthetic_fixture(
    root: str,
    n_samples: int = 64,
    image_size: int = 64,
    seed: int = 42,
    n_classes: int = 2,
) -> Tuple[list[str], list[int]]:
    """Generate a synthetic fixture of random grayscale images for smoke tests.

    Images are saved as PNG files under ``root``. Labels are balanced across
    classes. Intended for unit/integration tests only; not a substitute for
    the real Kermany dataset.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    paths: list[str] = []
    labels: list[int] = []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        for i in range(per_class):
            arr = rng.integers(0, 256, size=(image_size, image_size), dtype=np.uint8)
            # give each class a slightly different mean to make it learnable
            arr = np.clip(arr.astype(np.int16) + c * 40, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")
            fname = f"class{c}_sample{i:04d}.png"
            fpath = os.path.join(root, fname)
            img.save(fpath)
            paths.append(fpath)
            labels.append(c)
    # shuffle
    perm = rng.permutation(len(paths))
    paths = [paths[i] for i in perm]
    labels = [labels[i] for i in perm]
    return paths, labels


def build_dataloaders(
    image_paths: list[str],
    labels: list[int],
    cfg: DatasetConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Split into train/test (stratified) and return DataLoaders."""
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        train_size=cfg.train_ratio,
        random_state=cfg.seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    train_tf, eval_tf = get_transforms(image_size=cfg.image_size, augment=cfg.augment)
    train_ds = PneumoniaDataset(
        train_paths, train_labels, transform=train_tf, grayscale_to_rgb=cfg.grayscale_to_rgb
    )
    test_ds = PneumoniaDataset(
        test_paths, test_labels, transform=eval_tf, grayscale_to_rgb=cfg.grayscale_to_rgb
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, test_loader


def build_eval_loader(root: str, cfg: DatasetConfig) -> DataLoader:
    """Build a DataLoader over the FULL directory (ImageFolder-style).

    Use for held-out test sets: unlike load_real_dataset (which does an 80/20
    split of the given root), this evaluates every image in the folder.
    """
    from torchvision.datasets import ImageFolder

    _, eval_tf = get_transforms(image_size=cfg.image_size, augment=False)
    ds = ImageFolder(root=root, transform=eval_tf)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)


def load_real_dataset(
    root: str,
    cfg: DatasetConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Load a real image-folder dataset.

    Expected directory layout (ImageFolder-style):
        root/
            class0/  (e.g. Normal or Bacterial)
            class1/  (e.g. Pneumonia or Viral)
    The real Kermany dataset must be obtained from the corresponding author
    per the paper's Data Availability section.
    """
    train_ds, test_ds = _load_real_dataset_raw(root, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, test_loader


def _load_real_dataset_raw(
    root: str,
    cfg: DatasetConfig,
):
    """Internal: return raw train/test datasets (for class weight computation, etc.)."""
    from torchvision.datasets import ImageFolder

    train_tf, eval_tf = get_transforms(image_size=cfg.image_size, augment=cfg.augment)
    full_ds = ImageFolder(root=root, transform=train_tf)
    # simple split
    n = len(full_ds)
    n_train = int(cfg.train_ratio * n)
    indices = np.arange(n)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    train_ds = torch.utils.data.Subset(full_ds, train_idx.tolist())
    # rebuild test dataset with eval transforms
    full_ds_eval = ImageFolder(root=root, transform=eval_tf)
    test_ds = torch.utils.data.Subset(full_ds_eval, test_idx.tolist())
    return train_ds, test_ds
