"""Read the bundled prediction artifact and score weight configurations.

`results/test_predictions.npz` holds per-member softmax outputs for all 624
test images. Reading it costs milliseconds, which is what lets the Playground
page recompute accuracy and a confusion matrix on every slider move instead
of running a forward pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from pneumonia.service import DEFAULT_STAGE2_WEIGHTS, PAPER_STAGE1_WEIGHTS

DEFAULT_NPZ_PATH = os.path.join("results", "test_predictions.npz")


@dataclass(frozen=True)
class PredictionTable:
    """Per-member probabilities plus ground truth for the bundled test set."""

    stage1: dict[str, np.ndarray]
    stage2: dict[str, np.ndarray]
    y_true: np.ndarray
    y_true_binary: np.ndarray
    filenames: np.ndarray
    class_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.y_true.shape[0])


def load_prediction_table(path: str = DEFAULT_NPZ_PATH) -> PredictionTable:
    """Load the artifact written by scripts/dump_predictions.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction artifact not found: {path}. Run scripts/dump_predictions.py "
            "to generate it (needs the seven checkpoints and the test set)."
        )

    with np.load(path, allow_pickle=False) as data:
        stage1 = {k.split("/", 1)[1]: data[k] for k in data.files if k.startswith("stage1/")}
        stage2 = {k.split("/", 1)[1]: data[k] for k in data.files if k.startswith("stage2/")}
        if not stage1:
            raise ValueError(f"{path} contains no stage1/* members")

        return PredictionTable(
            stage1=stage1,
            stage2=stage2,
            y_true=data["y_true"],
            y_true_binary=data["y_true_binary"],
            filenames=data["filenames"],
            class_names=tuple(str(c) for c in data["class_names"]),
        )


def _member_weights(members: dict[str, np.ndarray],
                    weights: dict[str, float]) -> dict[str, np.ndarray]:
    """Renormalise weights over the members that are actually present."""
    present = [n for n in members if n in weights]
    if not present:
        raise ValueError("no member in weights has probabilities")
    total = sum(float(weights[n]) for n in present)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {n: np.float32(weights[n] / total) for n in present}


def score(table: PredictionTable, stage1_weights: dict[str, float] | None = None,
          stage2_weights: dict[str, float] | None = None,
          threshold: float = 0.5) -> dict:
    """Score one configuration against the test set, exactly as the pipeline does."""
    s1_weights = dict(stage1_weights or PAPER_STAGE1_WEIGHTS)
    s2_weights = dict(stage2_weights or DEFAULT_STAGE2_WEIGHTS)

    s1_norm = _member_weights(table.stage1, s1_weights)
    stage1_blend = sum(
        table.stage1[name] * weight for name, weight in s1_norm.items()
    ).astype(np.float32)

    p_pneumonia = stage1_blend[:, 1]
    routed = p_pneumonia >= threshold

    predictions = np.zeros(table.y_true.shape[0], dtype=np.int64)
    if table.stage2:
        s2_norm = _member_weights(table.stage2, s2_weights)
        stage2_blend = sum(
            table.stage2[name] * weight for name, weight in s2_norm.items()
        ).astype(np.float32)
        # Stage 2 decides the subtype for routed images; Stage 2 argmax is
        # 0 = BACTERIAL, 1 = VIRAL, and pipeline class order is
        # NORMAL=0, BACTERIAL=1, VIRAL=2.
        predictions[routed] = 1 + stage2_blend[routed].argmax(axis=1)

    return _metrics(predictions, table, int(routed.sum()))


def _metrics(predictions: np.ndarray, table: PredictionTable,
             routed_count: int) -> dict:
    y_true = table.y_true
    class_names = table.class_names
    n_classes = len(class_names)

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, predictions):
        confusion[int(true), int(pred)] += 1

    per_class = {}
    for idx, name in enumerate(class_names):
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        support = int(confusion[idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    correct = int((predictions == y_true).sum())
    total = int(y_true.shape[0])

    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "routed_to_stage2": routed_count,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "class_order": list(class_names),
    }


def member_names(table: PredictionTable) -> dict[str, list[str]]:
    """Member names per stage, in the order the models were trained."""
    stage1_order = ["pneumonet_v1", "pneumonet_v2", "vgg16", "inception_v3"]
    stage2_order = ["pneumonet_v1", "mobilenet_v2", "inception_v3"]
    return {
        "stage1": [n for n in stage1_order if n in table.stage1] or sorted(table.stage1),
        "stage2": [n for n in stage2_order if n in table.stage2] or sorted(table.stage2),
    }
