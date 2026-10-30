"""Inference service: load the seven checkpoints once, predict, re-blend.

This is the layer the web app talks to. It owns no HTTP and no file serving —
just model loading, the two-stage forward pass, and instant re-weighting of a
cached prediction.

Two things drive the design:

1. **Load once.** The seven checkpoints are ~1.05 GiB on disk and VGG-16 alone
   is 512 MiB, so construction happens a single time and the instance is reused.
2. **Cache member probabilities.** ``predict()`` returns the per-member softmax
   outputs alongside the blended verdict. ``reblend()`` then recomputes a
   prediction from those cached arrays in microseconds, which is what lets the
   Playground page move weight sliders without touching the GPU.

Re-blending accumulates in float32 on purpose. Accumulating in float64 shifts
borderline images across the Stage 1 routing threshold and inflates end-to-end
accuracy by several points; see ``tests/test_service.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from .checkpoints import CHECKPOINTS, checkpoint_path
from .data import get_transforms
from .models import PneumoNetV1, PneumoNetV2
from .transfer import TransferLearningModel

#: Ground-truth order the pipeline emits. Not ImageFolder's alphabetical order.
PIPELINE_CLASSES = ("NORMAL", "BACTERIAL", "VIRAL")

#: Paper's published Stage 1 weights (Section 4.1). Keys are short names.
PAPER_STAGE1_WEIGHTS = {"vgg16": 5, "inception_v3": 4, "pneumonet_v2": 3, "pneumonet_v1": 1}

#: The paper does not publish Stage 2 weights, so they default to equal.
DEFAULT_STAGE2_WEIGHTS = {"pneumonet_v1": 1.0, "mobilenet_v2": 1.0, "inception_v3": 1.0}

#: Short name -> checkpoint config name, per stage.
STAGE1_MEMBERS = {"pneumonet_v1": "stage1_pneumonet_v1", "pneumonet_v2": "stage1_pneumonet_v2",
                  "vgg16": "stage1_vgg16", "inception_v3": "stage1_inception"}
STAGE2_MEMBERS = {"pneumonet_v1": "stage2_pneumonet_v1", "mobilenet_v2": "stage2_mobilenet",
                  "inception_v3": "stage2_xception"}

#: Input size the checkpoints were trained at.
IMAGE_SIZE = 224

#: Stage 1 routes to Stage 2 at or above this P(pneumonia). Matches argmax.
DEFAULT_THRESHOLD = 0.5


def blend_probs(probs_by_name: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of member probabilities, renormalised over what is present.

    Members absent from ``weights`` are dropped and the remainder rescaled to
    sum to 1, so ablations do not need to re-run inference.
    """
    names = [n for n in probs_by_name if n in weights]
    if not names:
        raise ValueError("no member in weights has probabilities")
    w = np.array([weights[n] for n in names], dtype=np.float32)
    w = w / w.sum()
    return sum(weight * probs_by_name[name] for weight, name in zip(w, names))


def build_model(short_name: str, weights_path: str, num_classes: int = 2) -> torch.nn.Module:
    """Construct one backbone and load its checkpoint.

    ``weights_path`` may be a raw state_dict or a training checkpoint holding
    ``model_state_dict``; both are accepted.
    """
    if short_name == "pneumonet_v1":
        model: torch.nn.Module = PneumoNetV1(num_classes=num_classes)
    elif short_name == "pneumonet_v2":
        model = PneumoNetV2(num_classes=num_classes)
    else:
        model = TransferLearningModel(name=short_name, num_classes=num_classes, pretrained=False)

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@dataclass
class Prediction:
    """One image's verdict, with the cached probabilities behind it."""

    label: str
    confidence: float
    class_probs: dict[str, float]
    routed: bool
    stage1: dict[str, np.ndarray] = field(repr=False)
    stage2: dict[str, np.ndarray] | None = field(default=None, repr=False)
    stage1_probs: np.ndarray = field(default=None, repr=False)
    stage2_probs: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "class_probs": self.class_probs,
            "routed": self.routed,
            "stage1_members": {k: float(v[1]) for k, v in self.stage1.items()},
            "stage2_members": ({k: float(v[1]) for k, v in self.stage2.items()}
                               if self.stage2 is not None else None),
        }


class PipelineService:
    """Loads the seven checkpoints and serves two-stage predictions.

    Construct one instance per process. ``load()`` is idempotent but not
    cheap, so call it once at startup rather than per request.
    """

    def __init__(self, checkpoint_root: str | None = None, device: str | None = None):
        self._root_override = checkpoint_root
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._stage1: dict[str, torch.nn.Module] = {}
        self._stage2: dict[str, torch.nn.Module] = {}
        self._transform = get_transforms(image_size=IMAGE_SIZE, augment=False)[1]
        self._loaded = False

    # -- setup ---------------------------------------------------------

    def _resolve(self, mapping: dict[str, str]) -> dict[str, str]:
        resolved = {}
        for short_name, config_name in mapping.items():
            path = checkpoint_path(*CHECKPOINTS[config_name])
            if not os.path.exists(path):
                stage = "stage1" if config_name.startswith("stage1") else "stage2"
                raise FileNotFoundError(
                    f"Missing checkpoint for {stage}/{short_name}: {path}. "
                    "See docs/SETUP.md for where the seven .pt files go."
                )
            resolved[short_name] = path
        return resolved

    def load(self) -> "PipelineService":
        """Load all seven models onto the device. Returns self for chaining."""
        if self._loaded:
            return self
        for short_name, path in self._resolve(STAGE1_MEMBERS).items():
            self._stage1[short_name] = build_model(short_name, path).to(self.device)
        for short_name, path in self._resolve(STAGE2_MEMBERS).items():
            self._stage2[short_name] = build_model(short_name, path).to(self.device)
        self._loaded = True
        return self

    def health(self) -> dict:
        """Readiness report for the app's health page."""
        import torchvision

        found = {}
        # Stages are reported separately: short names repeat across stages
        # (pneumonet_v1 and inception_v3 appear in both), so a single merged
        # dict would silently collapse them and hide four checkpoints.
        for stage, mapping in (("stage1", STAGE1_MEMBERS), ("stage2", STAGE2_MEMBERS)):
            for short_name, config_name in mapping.items():
                path = checkpoint_path(*CHECKPOINTS[config_name])
                found[f"{stage}/{short_name}"] = {
                    "path": path,
                    "present": os.path.exists(path),
                }

        return {
            "ready": all(v["present"] for v in found.values()),
            "device": str(self.device),
            "loaded": self._loaded,
            "checkpoints": found,
            "versions": {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "numpy": np.__version__,
            },
        }

    # -- inference -----------------------------------------------------

    def _preprocess(self, image) -> torch.Tensor:
        """Convert a PIL image (or path) to a normalised 1x3x224x224 batch."""
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self._transform(image).unsqueeze(0)

    def _forward_stage(self, members: dict[str, torch.nn.Module],
                       x: torch.Tensor) -> dict[str, np.ndarray]:
        """Run every member of one stage and return per-member (1, 2) arrays."""
        probs = {}
        with torch.no_grad():
            for name, model in members.items():
                logits = model(x)
                probs[name] = torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float32)
        return probs

    def predict(self, image, stage1_weights: dict[str, float] | None = None,
                stage2_weights: dict[str, float] | None = None,
                threshold: float = DEFAULT_THRESHOLD) -> Prediction:
        """Run the full two-stage pipeline on one image.

        ``stage1_weights``/``stage2_weights`` default to the paper's published
        Stage 1 weights and equal Stage 2 weights.
        """
        if not self._loaded:
            self.load()

        s1_weights = dict(stage1_weights or PAPER_STAGE1_WEIGHTS)
        s2_weights = dict(stage2_weights or DEFAULT_STAGE2_WEIGHTS)

        x = self._preprocess(image).to(self.device)
        stage1 = self._forward_stage(self._stage1, x)

        stage1_blend = blend_probs(stage1, s1_weights)
        p_pneumonia = float(stage1_blend[1])
        routed = p_pneumonia >= threshold

        stage2 = None
        stage2_blend = None
        if routed:
            stage2 = self._forward_stage(self._stage2, x)
            stage2_blend = blend_probs(stage2, s2_weights)

        return self._compose(stage1, stage1_blend, stage2, stage2_blend, routed,
                             s1_weights, s2_weights, threshold)

    def _compose(self, stage1, stage1_blend, stage2, stage2_blend, routed,
                 s1_weights, s2_weights, threshold) -> Prediction:
        """Turn cached per-member probabilities into a three-class verdict.

        The label follows the cascade decision, not a three-way argmax. Stage 1
        routes at the threshold, and once routed Stage 2's own choice decides
        the subtype. An argmax across all three classes can disagree with the
        routing decision and hand back NORMAL for an image Stage 1 already
        flagged as pneumonia, which contradicts the pipeline the metrics were
        measured on.
        """
        p_normal = float(stage1_blend[0])
        if not routed:
            class_probs = {"NORMAL": p_normal, "BACTERIAL": 0.0, "VIRAL": 0.0}
            label = "NORMAL"
        else:
            p_bacterial = float(stage1_blend[1] * stage2_blend[0])
            p_viral = float(stage1_blend[1] * stage2_blend[1])
            class_probs = {"NORMAL": p_normal, "BACTERIAL": p_bacterial, "VIRAL": p_viral}
            # Stage 2 decides the subtype; Stage 1 only decided we are here.
            label = "BACTERIAL" if stage2_blend[0] >= stage2_blend[1] else "VIRAL"

        return Prediction(
            label=label,
            confidence=class_probs[label],
            class_probs=class_probs,
            routed=routed,
            stage1=stage1,
            stage2=stage2,
            stage1_probs=stage1_blend,
            stage2_probs=stage2_blend,
        )

    def reblend(self, prediction: Prediction, stage1_weights: dict[str, float],
                stage2_weights: dict[str, float],
                threshold: float = DEFAULT_THRESHOLD) -> Prediction:
        """Recompute a verdict from a prediction's cached member probabilities.

        No model runs. This is the fast path behind the Playground sliders, and
        it is why ``predict()`` caches the member arrays rather than only the
        blended result.
        """
        s1_weights = dict(stage1_weights or PAPER_STAGE1_WEIGHTS)
        s2_weights = dict(stage2_weights or DEFAULT_STAGE2_WEIGHTS)

        stage1_blend = blend_probs(prediction.stage1, s1_weights)
        p_pneumonia = float(stage1_blend[1])
        routed = p_pneumonia >= threshold

        stage2, stage2_blend = None, None
        if routed and prediction.stage2 is not None:
            stage2_blend = blend_probs(prediction.stage2, s2_weights)
            stage2 = prediction.stage2
        elif routed:
            # Stage 2 was never run (image was Normal at the original weights),
            # so a re-blend cannot classify it. Surface that rather than guess.
            raise ValueError(
                "cannot reblend to a routing that needs Stage 2: the original "
                "prediction was Normal, so Stage 2 was never run"
            )

        return self._compose(prediction.stage1, stage1_blend, stage2, stage2_blend, routed,
                             s1_weights, s2_weights, threshold)
