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
    """Compute inverse-frequency class weights without decoding any images.

    Works with ``ImageFolder`` and ``Subset`` (what ``load_real_dataset``
    returns). Reading labels from ``targets`` avoids a full image-decoding
    pass over the training set, which used to dominate startup time.

    Returns a tensor of shape (num_classes,) suitable for CrossEntropyLoss weight param.
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)

    targets = _dataset_targets(dataset)
    if targets is not None:
        for label in targets:
            class_counts[int(label)] += 1
    else:
        # Generic fallback: iterate items (slow, but only for exotic datasets).
        for _, label in dataset:
            class_counts[int(label)] += 1

    # Guard against missing classes to avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def _dataset_targets(dataset):
    """Return a list of integer labels if the dataset exposes them cheaply."""
    # torchvision Subset wrapping an ImageFolder
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base = dataset.dataset
        targets = getattr(base, "targets", None)
        if targets is not None:
            return [targets[i] for i in dataset.indices]
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return list(targets)
    # Subset of a Subset (defensive, rare)
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return _dataset_targets(dataset.dataset)
    return None


def get_weighted_sampler(dataset, num_classes: int = 2) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for balanced mini-batches.

    Each sample gets a weight inversely proportional to its class frequency.
    Labels are read from ``targets`` when available (no image decoding).
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)
    targets = _dataset_targets(dataset)
    if targets is not None:
        for label in targets:
            class_counts[int(label)] += 1
        class_counts = np.maximum(class_counts, 1)
        sample_weights = np.array([1.0 / class_counts[int(t)] for t in targets], dtype=np.float64)
    else:
        for _, label in dataset:
            class_counts[int(label)] += 1
        class_counts = np.maximum(class_counts, 1)
        sample_weights = np.array([1.0 / class_counts[int(label)] for _, label in dataset], dtype=np.float64)

    sample_weights = torch.from_numpy(sample_weights).double()
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def _make_grad_scaler(enabled: bool):
    """Create a GradScaler that works across torch versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


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
    use_amp: bool = True,
) -> dict:
    """Train a model for a number of epochs.

    Returns a history dict with train loss/accuracy and (if val given) val metrics.
    
    Args:
        class_weights: Optional tensor of class weights for weighted CrossEntropyLoss.
        use_weighted_sampler: If True, uses WeightedRandomSampler on train_dataset_for_sampler.
        train_dataset_for_sampler: Dataset to compute sample weights from (required if use_weighted_sampler=True).
        num_classes: Number of classes (for sampler weight computation).
        use_amp: Use CUDA automatic mixed precision (fp16) when available. ~1.5-2x faster on T4.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = get_criterion(class_weights.to(device) if class_weights is not None else None)
    optimizer = get_optimizer(model, lr=lr, momentum=momentum)

    # Mixed precision only makes sense on CUDA.
    use_amp = bool(use_amp) and device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)

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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                # Handle ensemble output (dict with "logits" key) or InceptionV3 tuple output (logits, aux_logits)
                if isinstance(outputs, dict):
                    outputs = outputs["logits"]
                elif isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
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
    """Load a model from either a training checkpoint or a raw state_dict.

    ``train_model`` writes per-epoch checkpoints (a dict with
    ``model_state_dict``), while the CLI writes ``final_model.pt`` as a plain
    ``state_dict``. Accept both so ``--checkpoint`` works with either file.
    """
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        # Raw state_dict saved via torch.save(model.state_dict(), path)
        model.load_state_dict(checkpoint)
        checkpoint = {"model_state_dict": checkpoint}
    return checkpoint
