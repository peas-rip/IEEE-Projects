"""Training loop, optimizer, loss, checkpointing.

Paper training settings (PAPER-STATED unless marked otherwise):
- Loss: Categorical Crossentropy.
- Optimizer: SGD, learning rate initialized at 1e-4.
- Mini-batch gradient descent (batch size and epochs not stated -> DEFAULT).
- Momentum SGD update per paper Eq. (3)-(4) (DEFAULT: use PyTorch SGD with momentum).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np


def get_optimizer(model: nn.Module, lr: float = 1e-4, momentum: float = 0.9) -> torch.optim.Optimizer:
    """SGD optimizer matching the paper (lr=1e-4 PAPER-STATED; momentum DEFAULT)."""
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_criterion(class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Categorical Crossentropy loss (PAPER-STATED).
    
    If class_weights provided, uses weighted CrossEntropyLoss to handle imbalance.
    """
    return nn.CrossEntropyLoss(weight=class_weights)


def compute_class_weights(dataset, num_classes: int = 2) -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataset.
    
    Returns a tensor of shape (num_classes,) suitable for CrossEntropyLoss weight param.
    """
    # Count samples per class
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dataset:
        class_counts[label] += 1
    
    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(dataset, num_classes: int = 2) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for balanced mini-batches.
    
    Each sample gets a weight inversely proportional to its class frequency.
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dataset:
        class_counts[label] += 1
    
    # Per-sample weights
    sample_weights = np.array([1.0 / class_counts[label] for _, label in dataset], dtype=np.float64)
    sample_weights = torch.from_numpy(sample_weights).double()
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-4,
    momentum: float = 0.9,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    use_weighted_sampler: bool = False,
    train_dataset_for_sampler=None,
    num_classes: int = 2,
) -> dict:
    """Train a model for a number of epochs.

    Returns a history dict with train loss/accuracy and (if val given) val metrics.
    
    Args:
        class_weights: Optional tensor of class weights for weighted CrossEntropyLoss.
        use_weighted_sampler: If True, uses WeightedRandomSampler on train_dataset_for_sampler.
        train_dataset_for_sampler: Dataset to compute sample weights from (required if use_weighted_sampler=True).
        num_classes: Number of classes (for sampler weight computation).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = get_criterion(class_weights.to(device) if class_weights is not None else None)
    optimizer = get_optimizer(model, lr=lr, momentum=momentum)

    # If using weighted sampler, recreate the loader with the sampler
    if use_weighted_sampler and train_dataset_for_sampler is not None:
        sampler = get_weighted_sampler(train_dataset_for_sampler, num_classes)
        train_loader = DataLoader(
            train_dataset_for_sampler,
            batch_size=train_loader.batch_size,
            sampler=sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
        )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    os.makedirs(checkpoint_dir, exist_ok=True) if checkpoint_dir else None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            # Handle ensemble output (dict with "logits" key) or InceptionV3 tuple output (logits, aux_logits)
            if isinstance(outputs, dict):
                outputs = outputs["logits"]
            elif isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_loss, val_acc = 0.0, 0.0
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    # Handle ensemble output (dict with "logits" key) or InceptionV3 tuple output
                    if isinstance(outputs, dict):
                        outputs = outputs["logits"]
                    elif isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)
                    v_running += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    v_total += labels.size(0)
                    v_correct += predicted.eq(labels).sum().item()
            val_loss = v_running / v_total
            val_acc = v_correct / v_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )

        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_path)

    return history


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    extra: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
