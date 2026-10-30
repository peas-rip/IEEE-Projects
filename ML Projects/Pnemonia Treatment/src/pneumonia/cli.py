"""Command-line interface for train / evaluate / infer.

Usage examples:
    python -m pneumonia train --config configs/default.yaml --data synthetic
    python -m pneumonia evaluate --config configs/default.yaml --data synthetic --checkpoint results/checkpoint_epoch10.pt
    python -m pneumonia infer --image path/to/xray.png --checkpoint results/final_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import yaml
import torch
import numpy as np

from .data import DatasetConfig, make_synthetic_fixture, build_dataloaders, load_real_dataset
from .models import PneumoNetV1, PneumoNetV2
from .ensemble import DetectionEnsemble, ClassificationEnsemble, TwoStagePipeline
from .train import train_model, save_checkpoint
from .evaluate import evaluate_model


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_cfg_from_config(config: dict) -> DatasetConfig:
    data_cfg = config.get("data", {})
    return DatasetConfig(
        image_size=int(data_cfg.get("image_size", 224)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        seed=int(data_cfg.get("seed", 42)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        grayscale_to_rgb=bool(data_cfg.get("grayscale_to_rgb", True)),
        augment=bool(data_cfg.get("augment", True)),
    )


def _build_model(model_name: str, num_classes: int, pretrained: bool = True, ensemble_weights: list[float] | None = None):
    """Build a model by name (custom CNNs, transfer-learning backbones, or ensembles)."""
    if model_name == "pneumonet_v1":
        return PneumoNetV1(num_classes=num_classes)
    if model_name == "pneumonet_v2":
        return PneumoNetV2(num_classes=num_classes)
    if model_name == "detection_ensemble":
        from .ensemble import DetectionEnsemble
        return DetectionEnsemble(num_classes=num_classes, weights=ensemble_weights)
    if model_name == "classification_ensemble":
        from .ensemble import ClassificationEnsemble
        return ClassificationEnsemble(num_classes=num_classes, weights=ensemble_weights)
    # Transfer-learning models
    from .transfer import TransferLearningModel

    return TransferLearningModel(name=model_name, num_classes=num_classes, pretrained=pretrained)


def cmd_train(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    ensemble_cfg = config.get("ensemble", {})
    cfg = _build_cfg_from_config(config)
    if getattr(args, "num_workers", None) is not None:
        cfg.num_workers = int(args.num_workers)
    use_amp = bool(getattr(args, "amp", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  (amp={use_amp and device.type == 'cuda'}, workers={cfg.num_workers})")

    # Data
    if args.data == "synthetic":
        root = args.synthetic_root or "data/synthetic"
        n = int(data_cfg.get("synthetic_samples", 128))
        img_sz = int(data_cfg.get("synthetic_image_size", 64))
        paths, labels = make_synthetic_fixture(
            root=root, n_samples=n, image_size=img_sz, seed=cfg.seed, n_classes=2
        )
        train_loader, test_loader = build_dataloaders(paths, labels, cfg)
        train_dataset = None  # synthetic uses custom fixture
    else:
        train_loader, test_loader = load_real_dataset(args.data, cfg)
        # Reuse the loader's dataset instead of building ImageFolder a second time.
        # The underlying Subset exposes `.targets`, so class weights need no image decoding.
        train_dataset = train_loader.dataset

    # Model
    model_name = train_cfg.get("model", "pneumonet_v1")
    num_classes = int(train_cfg.get("num_classes", 2))

    # Ensemble weights from config (for detection_ensemble or classification_ensemble)
    ensemble_weights = None
    if model_name == "detection_ensemble":
        ensemble_weights = ensemble_cfg.get("detection_weights", None)
    elif model_name == "classification_ensemble":
        ensemble_weights = ensemble_cfg.get("classification_weights", None)

    model = _build_model(model_name, num_classes, pretrained=True, ensemble_weights=ensemble_weights)

    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("learning_rate", 1e-4))
    momentum = float(train_cfg.get("momentum", 0.9))
    ckpt_dir = args.checkpoint_dir or "results/checkpoints"

    # Class weighting options
    use_class_weights = bool(train_cfg.get("use_class_weights", False))
    use_weighted_sampler = bool(train_cfg.get("use_weighted_sampler", False))

    class_weights = None
    if use_class_weights:
        from .train import compute_class_weights
        # Need the underlying dataset to compute class weights
        if args.data == "synthetic":
            # For synthetic, we can compute from the fixture
            paths, labels = make_synthetic_fixture(
                root=args.synthetic_root or "data/synthetic",
                n_samples=int(data_cfg.get("synthetic_samples", 128)),
                image_size=int(data_cfg.get("synthetic_image_size", 64)),
                seed=cfg.seed, n_classes=num_classes
            )
            class_counts = np.bincount(labels, minlength=num_classes)
            total = class_counts.sum()
            class_weights = torch.tensor(total / (num_classes * class_counts), dtype=torch.float32)
        elif train_dataset is not None:
            class_weights = compute_class_weights(train_dataset, num_classes)

    history = train_model(
        model,
        train_loader,
        val_loader=test_loader,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        device=device,
        checkpoint_dir=ckpt_dir,
        class_weights=class_weights,
        use_weighted_sampler=use_weighted_sampler,
        train_dataset_for_sampler=train_dataset if use_weighted_sampler else None,
        num_classes=num_classes,
        use_amp=use_amp,
    )

    # Save final model
    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    ensemble_cfg = config.get("ensemble", {})
    cfg = _build_cfg_from_config(config)
    if getattr(args, "num_workers", None) is not None:
        cfg.num_workers = int(args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data == "synthetic":
        root = args.synthetic_root or "data/synthetic"
        n = int(data_cfg.get("synthetic_samples", 128))
        img_sz = int(data_cfg.get("synthetic_image_size", 64))
        # For evaluation we want no augmentation; rebuild with augment=False
        cfg.augment = False
        paths, labels = make_synthetic_fixture(
            root=root, n_samples=n, image_size=img_sz, seed=cfg.seed, n_classes=2
        )
        _, test_loader = build_dataloaders(paths, labels, cfg)
    else:
        # Evaluate on the FULL directory passed via --data (held-out test set).
        # load_real_dataset would do an 80/20 split of it; for held-out test dirs
        # we want every image (build_eval_loader).
        from .data import build_eval_loader
        test_loader = build_eval_loader(args.data, cfg)
        print(f"Evaluating on all {sum(len(f) for _, _, f in os.walk(args.data))} files in {args.data}")

    num_classes = int(eval_cfg.get("num_classes", 2))
    model_name = eval_cfg.get("model", "pneumonet_v1")

    # Ensemble weights from config
    ensemble_weights = None
    if model_name == "detection_ensemble":
        ensemble_weights = ensemble_cfg.get("detection_weights", None)
    elif model_name == "classification_ensemble":
        ensemble_weights = ensemble_cfg.get("classification_weights", None)

    model = _build_model(model_name, num_classes, pretrained=False, ensemble_weights=ensemble_weights)

    # Load checkpoint if provided
    if args.checkpoint:
        from .train import load_checkpoint

        load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        # load raw state dict
        state = torch.load(args.model_weights, map_location="cpu")
        model.load_state_dict(state)

    metrics = evaluate_model(model, test_loader, device=device)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    from PIL import Image
    from .data import get_transforms

    # Architecture, input size and label names can come from a config file so
    # every one of the seven checkpoints is usable, not just PneumoNet-v1.
    config = _load_config(args.config) if getattr(args, "config", None) else {}
    eval_cfg = config.get("evaluation", {})
    data_cfg = config.get("data", {})

    model_name = args.model or eval_cfg.get("model") or "pneumonet_v1"
    num_classes = int(eval_cfg.get("num_classes", 2))
    image_size = args.image_size or int(data_cfg.get("image_size", 224))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_name, num_classes, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    transform = get_transforms(image_size=image_size, augment=False)[1]
    image = Image.open(args.image).convert("L")
    image = Image.merge("RGB", (image, image, image))
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = probs.argmax(1).item()

    if args.labels:
        class_names = [name.strip() for name in args.labels.split(",")]
    elif num_classes == 3:
        class_names = ["NORMAL", "BACTERIAL", "VIRAL"]
    else:
        class_names = ["Normal", "Pneumonia"]
    if len(class_names) != num_classes:
        print(f"ERROR: {len(class_names)} labels given but the model has {num_classes} classes")
        return 1

    print(f"Image: {args.image}")
    print(f"Model: {model_name}  ({num_classes} classes, {image_size}px)")
    print(f"Predicted class: {class_names[pred]}")
    print(f"Probabilities: {probs.cpu().numpy().tolist()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pneumonia", description="Pneumonia detection & classification (paper reimplementation)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", required=True, help="Path to YAML config")
    p_train.add_argument("--data", default="synthetic", help="Dataset path or 'synthetic'")
    p_train.add_argument("--synthetic-root", default="data/synthetic", help="Root for synthetic fixture")
    p_train.add_argument("--checkpoint-dir", default="results/checkpoints", help="Checkpoint output directory")
    p_train.add_argument("--num-workers", type=int, default=None, help="Override data loader workers (default: config value)")
    p_train.add_argument("--amp", dest="amp", action="store_true", default=True, help="Use CUDA mixed precision (default: on)")
    p_train.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a model")
    p_eval.add_argument("--config", required=True, help="Path to YAML config")
    p_eval.add_argument("--data", default="synthetic", help="Dataset path or 'synthetic'")
    p_eval.add_argument("--synthetic-root", default="data/synthetic", help="Root for synthetic fixture")
    p_eval.add_argument("--model-weights", help="Path to model state_dict (.pt)")
    p_eval.add_argument("--checkpoint", help="Training checkpoint with epoch/optimizer (.pt)")
    p_eval.add_argument("--num-workers", type=int, default=None, help="Override data loader workers (default: config value)")
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = sub.add_parser("infer", help="Infer on a single image")
    p_infer.add_argument("--image", required=True, help="Path to input chest X-ray image")
    p_infer.add_argument("--checkpoint", required=True, help="Path to model state_dict (.pt)")
    p_infer.add_argument("--config", default=None,
                         help="YAML config supplying the architecture and input size "
                              "(e.g. configs/stage1_vgg16.yaml)")
    p_infer.add_argument("--model", default=None,
                         help="Model name override: pneumonet_v1, pneumonet_v2, vgg16, "
                              "inception_v3, mobilenet_v2, resnet50, ...")
    p_infer.add_argument("--image-size", type=int, default=None,
                         help="Input size override (default: config value or 224)")
    p_infer.add_argument("--labels", default=None,
                         help="Comma-separated class names, e.g. 'Normal,Pneumonia'")
    p_infer.set_defaults(func=cmd_infer)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
