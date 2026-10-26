"""Custom CNN architectures: PneumoNet-v1 and PneumoNet-v2.

Architecture details are taken from the paper, Sections 3.3 and 3.4
(pp. 4-5). Where the paper text is ambiguous (e.g. filter kernel sizes
that appear as OCR artifacts such as "1x3"), the most sensible
interpretation consistent with the surrounding description is used and
marked INFERRED.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBN(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, pool: nn.Module | None = None):
        super().__init__()
        pad = k // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool is not None:
            layers.append(pool)
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PneumoNetV1(nn.Module):
    """PneumoNet-v1 custom CNN for pneumonia detection.

    Paper Section 3.3 (PAPER-STATED):
      - Input 224x224x3.
      - Conv2d 128 filters 8x8 stride 3 -> BN.
      - Conv2d 256 filters 5x5 stride 1 -> BN -> MaxPool2d 3x3.
      - Three conv layers: 256 filters (3x3 INFERRED), 256 filters 1x1, 256 filters 1x1,
        each stride 1, BN after each.
      - Four conv layers: 512 filters 3x3, BN + MaxPool2d 2x2 after each.
      - Flatten -> Dense 1024 ReLU -> Dense 1024 ReLU -> Dropout 0.5 -> Dense 2 Softmax.
      - Loss: Categorical Crossentropy. Optimizer: SGD, lr=1e-4.

    Notes (INFERRED / DEFAULT):
      - "1x3" in the paper is interpreted as a 3x3 kernel (OCR artifact).
      - Grayscale CXR images are replicated to 3 channels to match the 3-channel input spec.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # stem: 128 filters 8x8 stride 3
            _ConvBN(in_channels, 128, k=8, stride=3),
            # 256 filters 5x5 stride 1 -> BN -> MaxPool 3x3
            _ConvBN(128, 256, k=5, stride=1, pool=nn.MaxPool2d(kernel_size=3)),
            # three conv layers: 256 (3x3), 256 (1x1), 256 (1x1), stride 1, BN each
            _ConvBN(256, 256, k=3, stride=1),
            _ConvBN(256, 256, k=1, stride=1),
            _ConvBN(256, 256, k=1, stride=1),
            # four conv layers: 512 filters 3x3, BN + MaxPool 2x2 after each
            _ConvBN(256, 512, k=3, stride=1, pool=nn.MaxPool2d(kernel_size=2)),
            _ConvBN(512, 512, k=3, stride=1, pool=nn.MaxPool2d(kernel_size=2)),
            _ConvBN(512, 512, k=3, stride=1, pool=nn.MaxPool2d(kernel_size=2)),
            _ConvBN(512, 512, k=3, stride=1, pool=nn.MaxPool2d(kernel_size=2)),
            nn.AdaptiveAvgPool2d((7, 7)),  # DEFAULT: force fixed 7x7 trunk output regardless of conv/pool reduction
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class PneumoNetV2(nn.Module):
    """PneumoNet-v2 custom CNN for pneumonia detection.

    Paper Section 3.4 (PAPER-STATED):
      - Input 224x224x3.
      - Conv2d 64 filters 3x3 stride 1 -> BN.
      - Three conv layers: 128 filters 3x3 stride 1, BN each, MaxPool2d 2x2
        after first and third.
      - Three conv layers: 256, 256, 512 filters 3x3, BN + MaxPool2d 2x2 after each.
      - Flatten -> Dense 1024 -> Dropout 0.5 -> Dense 1024 -> Dropout 0.5 -> Dense 2.
      - Lower learning rate than PneumoNet-v1 (paper-stated). Optimizer: SGD.

    Notes (INFERRED):
      - Grayscale CXR replicated to 3 channels.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBN(in_channels, 64, k=3, stride=1),
            _ConvBN(64, 128, k=3, stride=1, pool=nn.MaxPool2d(kernel_size=2)),
            _ConvBN(128, 128, k=3, stride=1),
            _ConvBN(128, 128, k=3, stride=1, pool=nn.MaxPool2d(kernel_size=2)),
            _ConvBN(128, 256, k=3, stride=1, pool=nn.MaxPool2d(kernel_size=2)),
            _ConvBN(256, 256, k=3, stride=1, pool=nn.MaxPool2d(kernel_size=2)),
            _ConvBN(256, 512, k=3, stride=1, pool=nn.MaxPool2d(kernel_size=2)),
            nn.AdaptiveAvgPool2d((7, 7)),  # DEFAULT: force fixed 7x7 trunk output
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def _infer_feat_size(model: nn.Module, in_channels: int = 3, img_size: int = 224) -> int:
    """Infer flattened feature size after the conv trunk (for sanity checks)."""
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, img_size, img_size)
        out = model.features(dummy)
        return int(out.view(1, -1).shape[1])


if __name__ == "__main__":  # pragma: no cover - quick manual check
    v1 = PneumoNetV1()
    v2 = PneumoNetV2()
    x = torch.zeros(2, 3, 224, 224)
    print("PneumoNetV1 out:", v1(x).shape)
    print("PneumoNetV2 out:", v2(x).shape)
    print("V1 feat size:", _infer_feat_size(v1))
    print("V2 feat size:", _infer_feat_size(v2))
