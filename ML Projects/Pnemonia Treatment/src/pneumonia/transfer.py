"""Transfer-learning model wrappers.

Wraps torchvision pre-trained backbones (VGG-16, VGG-19, Inception-v3,
Xception is not in torchvision; we use the closest available or a small
custom Xception-like head where needed), replacing the final
classification head with a 2-unit Softmax head for binary pneumonia
tasks.

Paper Section 3.5 lists baselines: VGG-16, VGG-19, Inception-v3,
Xception, ResNet-50, ResNet-101, DenseNet-121, DenseNet-169,
MobileNet-v2 (PAPER-STATED). torchvision provides all except Xception;
for Xception we fall back to Inception-v3 weights with a custom head and
document this as an INFERRED engineering substitution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class TransferLearningModel(nn.Module):
    """Generic wrapper around a torchvision backbone with a new 2-unit head.

    Parameters
    ----------
    name : str
        One of: "vgg16", "vgg19", "inception_v3", "resnet50", "resnet101",
        "densenet121", "densenet169", "mobilenet_v2", "xception" (INFERRED sub).
    num_classes : int
        Output classes (2 for binary tasks).
    pretrained : bool
        Use ImageNet pre-trained weights (DEFAULT engineering choice).
    freeze_backbone : bool
        If True, freeze all backbone parameters (feature-extraction mode).
    """

    _BACKBONES = {
        "vgg16": tv_models.vgg16,
        "vgg19": tv_models.vgg19,
        "inception_v3": tv_models.inception_v3,
        "resnet50": tv_models.resnet50,
        "resnet101": tv_models.resnet101,
        "densenet121": tv_models.densenet121,
        "densenet169": tv_models.densenet169,
        "mobilenet_v2": tv_models.mobilenet_v2,
    }

    def __init__(
        self,
        name: str = "vgg16",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.name = name.lower()
        self.num_classes = num_classes

        if self.name == "xception":
            # INFERRED: torchvision has no Xception; use Inception-v3 backbone
            # with a custom 2-unit head and document the substitution.
            backbone = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.DEFAULT if pretrained else None)
            # Must set AuxLogits=None (not just aux_logits=False) to actually
            # remove the aux branch; otherwise it fails at 224x224 input.
            backbone.aux_logits = False
            backbone.AuxLogits = None
            in_f = backbone.fc.in_features
            backbone.fc = nn.Linear(in_f, num_classes)
            self.backbone = backbone
        elif self.name in self._BACKBONES:
            constructor = self._BACKBONES[self.name]
            weights = None
            if pretrained:
                if self.name == "vgg16":
                    weights = tv_models.VGG16_Weights.DEFAULT
                elif self.name == "vgg19":
                    weights = tv_models.VGG19_Weights.DEFAULT
                elif self.name == "inception_v3":
                    weights = tv_models.Inception_V3_Weights.DEFAULT
                elif self.name == "resnet50":
                    weights = tv_models.ResNet50_Weights.DEFAULT
                elif self.name == "resnet101":
                    weights = tv_models.ResNet101_Weights.DEFAULT
                elif self.name == "densenet121":
                    weights = tv_models.DenseNet121_Weights.DEFAULT
                elif self.name == "densenet169":
                    weights = tv_models.DenseNet169_Weights.DEFAULT
                elif self.name == "mobilenet_v2":
                    weights = tv_models.MobileNet_V2_Weights.DEFAULT
            backbone = constructor(weights=weights)
            if self.name.startswith("vgg"):
                in_f = backbone.classifier[-1].in_features
                backbone.classifier[-1] = nn.Linear(in_f, num_classes)
            elif self.name in ("resnet50", "resnet101"):
                in_f = backbone.fc.in_features
                backbone.fc = nn.Linear(in_f, num_classes)
            elif self.name.startswith("densenet"):
                in_f = backbone.classifier.in_features
                backbone.classifier = nn.Linear(in_f, num_classes)
            elif self.name == "mobilenet_v2":
                in_f = backbone.classifier[-1].in_features
                backbone.classifier[-1] = nn.Linear(in_f, num_classes)
            elif self.name == "inception_v3":
                # Must set AuxLogits=None (not just aux_logits=False) so the aux
                # branch is removed; otherwise it fails at 224x224 input.
                # Applied regardless of `pretrained` so the architecture always
                # matches (no AuxLogits, num_classes head) when loading ckpts.
                backbone.aux_logits = False
                backbone.AuxLogits = None
                in_f = backbone.fc.in_features
                backbone.fc = nn.Linear(in_f, num_classes)
            self.backbone = backbone
        else:
            raise ValueError(f"Unknown backbone: {name}")

        self._maybe_freeze(freeze_backbone)

    def _maybe_freeze(self, freeze: bool) -> None:
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # keep the new head trainable
            for p in self._head_params():
                p.requires_grad = True

    def _head_params(self):
        if self.name.startswith("vgg"):
            return self.backbone.classifier[-1].parameters()
        if self.name in ("resnet50", "resnet101"):
            return self.backbone.fc.parameters()
        if self.name.startswith("densenet"):
            return self.backbone.classifier.parameters()
        if self.name == "mobilenet_v2":
            return self.backbone.classifier[-1].parameters()
        # inception_v3 / xception
        return self.backbone.fc.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inception-v3 expects 299x299 natively but works with 224x224 with a warning;
        # we keep 224x224 for pipeline consistency (INFERRED).
        return self.backbone(x)


def get_transfer_model(name: str, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False) -> TransferLearningModel:
    """Factory for transfer-learning models."""
    return TransferLearningModel(
        name=name, num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone
    )
