"""Weighted ensemble models.

Two ensembles per the paper (Section 3.1 / 3.7, PAPER-STATED):
- Stage 1 Detection (Normal vs. Pneumonia): VGG-16, Inception-v3,
  PneumoNet-v1, PneumoNet-v2.
- Stage 2 Classification (Bacterial vs. Viral): Xception, MobileNet-v2,
  PneumoNet-v1.

Aggregation (INFERRED from paper narrative): weighted average of the
base models' softmax class probabilities; final label = argmax. The
paper mentions a genetic-algorithm/grid-search for optimal weights but
does not publish an explicit formula, so weighted average is the default.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .models import PneumoNetV1, PneumoNetV2
from .transfer import TransferLearningModel


class WeightedEnsemble(nn.Module):
    """Weighted average of base model softmax probabilities."""

    def __init__(self, models: list[nn.Module], weights: list[float] | None = None, num_classes: int = 2):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        if len(weights) != len(models):
            raise ValueError("Number of weights must equal number of models")
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                prob = torch.softmax(logits, dim=1)
            probs.append(prob)
        probs = torch.stack(probs, dim=0)  # (n_models, batch, num_classes)
        w = self.weights.view(-1, 1, 1)
        weighted = (probs * w).sum(dim=0)  # (batch, num_classes)
        pred = weighted.argmax(dim=1)
        return {"logits": weighted, "probs": weighted, "pred": pred, "member_probs": probs}


class DetectionEnsemble(WeightedEnsemble):
    """Stage 1: Normal vs. Pneumonia (4 models)."""

    def __init__(self, num_classes: int = 2, weights: list[float] | None = None):
        models = [
            PneumoNetV1(num_classes=num_classes),
            PneumoNetV2(num_classes=num_classes),
            TransferLearningModel("vgg16", num_classes=num_classes),
            TransferLearningModel("inception_v3", num_classes=num_classes),
        ]
        if weights is None:
            # Paper-stated example weights for the best detection combo
            # (Inception-v3:4, VGG-16:5, PneumoNet-v2:3, PneumoNet-v1:1),
            # reordered to match this model list [PneumoNet-v1, PneumoNet-v2,
            # VGG-16, Inception-v3] -> [1, 3, 5, 4] normalized.
            weights = [1.0, 3.0, 5.0, 4.0]
        super().__init__(models=models, weights=weights, num_classes=num_classes)


class ClassificationEnsemble(WeightedEnsemble):
    """Stage 2: Bacterial vs. Viral (3 models)."""

    def __init__(self, num_classes: int = 2, weights: list[float] | None = None):
        # Xception -> INFERRED: Inception-v3 backbone; MobileNet-v2; PneumoNet-v1
        models = [
            TransferLearningModel("xception", num_classes=num_classes),
            TransferLearningModel("mobilenet_v2", num_classes=num_classes),
            PneumoNetV1(num_classes=num_classes),
        ]
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        super().__init__(models=models, weights=weights, num_classes=num_classes)


class TwoStagePipeline(nn.Module):
    """Two-stage inference: detect pneumonia, then classify bacterial vs. viral.

    Stage 1 (DetectionEnsemble): Normal vs. Pneumonia.
      - If Normal -> output "Normal", stop.
      - If Pneumonia -> run Stage 2 (ClassificationEnsemble) -> Bacterial / Viral.
    """

    def __init__(
        self,
        detection: DetectionEnsemble,
        classification: ClassificationEnsemble,
        class_names: tuple[str, ...] = ("Normal", "Bacterial", "Viral"),
    ):
        super().__init__()
        self.detection = detection
        self.classification = classification
        self.class_names = class_names

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        det = self.detection(x)
        det_pred = det["pred"]  # 0 = Normal, 1 = Pneumonia
        batch = x.shape[0]
        final = torch.full((batch,), -1, dtype=torch.long, device=x.device)
        # For simplicity in batched form: run classification on all, then mask.
        cls = self.classification(x)
        cls_pred = cls["pred"]  # 0 = Bacterial, 1 = Viral
        # Map: detection Normal (0) -> class_names[0]; else classification result.
        for i in range(batch):
            if int(det_pred[i]) == 0:
                final[i] = 0  # Normal
            else:
                final[i] = 1 + int(cls_pred[i])  # 1 = Bacterial, 2 = Viral
        return {
            "detection_logits": det["logits"],
            "detection_probs": det["probs"],
            "detection_pred": det_pred,
            "classification_probs": cls["probs"],
            "classification_pred": cls_pred,
            "pred": final,
        }
