"""Pneumonia detection and classification package.

Implements the method from:
"Enhancing Paediatric Pneumonia Detection and Classification Using
Customized CNNs and Transfer Learning Based Ensemble Models"
(Int. Res. J. Multidisc. Technovation, 6(6) (2024) 38-53).
"""

from .models import PneumoNetV1, PneumoNetV2  # noqa: F401
from .transfer import TransferLearningModel  # noqa: F401
from .ensemble import DetectionEnsemble, ClassificationEnsemble  # noqa: F401
from .data import PneumoniaDataset, make_synthetic_fixture, get_transforms  # noqa: F401
from .train import train_model, save_checkpoint, load_checkpoint  # noqa: F401
from .evaluate import evaluate_model, compute_metrics  # noqa: F401

__version__ = "0.1.0"
