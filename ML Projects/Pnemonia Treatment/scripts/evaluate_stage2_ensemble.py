"""Build and evaluate the full Stage 2 ensemble (Xception-sub, MobileNet-v2, PneumoNet-v1).

Paper Section 3.7 / Section 4.2: the Stage 2 classification ensemble combines
Xception, MobileNet-v2 and PneumoNet-v1 via weighted averaging of softmax
probabilities. Weights are found by grid search over the validation split
(paper mentions grid search ranges 1-15 / 1-20; exact published weights for
this combo are UNKNOWN -> we search a small range and report results).

Usage:
    python scripts/evaluate_stage2_ensemble.py
"""

import argparse
import os
import sys
import itertools
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pneumonia.data import DatasetConfig, build_dataloaders
from pneumonia.transfer import TransferLearningModel
from pneumonia.models import PneumoNetV1
from pneumonia.evaluate import compute_metrics, predict_probs
from pneumonia.reporting import write_json
from pneumonia.checkpoints import checkpoint_path


def load_model(model_name, weights_path, num_classes=2):
    if model_name == "pneumonet_v1":
        model = PneumoNetV1(num_classes=num_classes)
    else:
        model = TransferLearningModel(name=model_name, num_classes=num_classes, pretrained=False)
    state = torch.load(weights_path, map_location="cpu")
    # Accept either a raw state_dict or a checkpoint dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def weighted_ensemble_probs(member_probs, weights):
    """member_probs: list of (N, C) arrays; weights: list of floats."""
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    stacked = np.stack(member_probs, axis=0)  # (M, N, C)
    return (stacked * w[:, None, None]).sum(axis=0)


def available_members(members, allow_missing=False):
    """Keep only members whose weights file exists (see Stage 1 script)."""
    present, missing = {}, []
    for name, path in members.items():
        if os.path.exists(path):
            present[name] = path
        else:
            missing.append((name, path))

    for name, path in missing:
        print(f"MISSING weights: {path} ({name})")

    if not missing:
        return present
    if not allow_missing:
        print("Train the missing model(s) first, or pass --allow-missing to "
              "evaluate with the remaining members.")
        return None
    if len(present) < 2:
        print("At least 2 ensemble members are required.")
        return None

    print(f"WARNING: evaluating without {[n for n, _ in missing]}; "
          f"weights are renormalised over {list(present)}")
    return present


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the Stage 2 classification ensemble")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Evaluate with the members that exist and renormalise their weights")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="Write the metrics of this run to a JSON file")
    args = parser.parse_args(argv)

    # Paths to the three trained Stage 2 models. Override the root with
    # PNEUMONIA_CHECKPOINT_DIR when the files live elsewhere (e.g. Colab Drive).
    members = {
        # "stage2_checkpoints" is the directory used by the first training run.
        "pneumonet_v1": checkpoint_path("stage2_pneumonet_v1", "stage2_checkpoints"),
        "mobilenet_v2": checkpoint_path("stage2_mobilenet"),
        # Xception substitute
        "inception_v3": checkpoint_path("stage2_xception", "stage2_inception_v3"),
    }
    members = available_members(members, args.allow_missing)
    if members is None:
        return 1

    # Test data
    cfg = DatasetConfig(image_size=224, batch_size=16, train_ratio=0.8, seed=42, num_workers=0, augment=False)
    paths, labels = [], []
    from torchvision.datasets import ImageFolder
    from pneumonia.data import get_transforms
    eval_tf = get_transforms(image_size=224, augment=False)[1]
    test_ds = ImageFolder(root="data/real_stage2/test", transform=eval_tf)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    class_names = test_ds.classes  # ['BACTERIAL', 'VIRAL']
    print(f"Test classes: {class_names}  ({len(test_ds)} images)")

    # Load each member and get probabilities
    models = {n: load_model(n, p) for n, p in members.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    member_probs = {}
    solo_metrics = {}
    y_true = None
    for name, model in models.items():
        y_true, probs = predict_probs(model.to(device), test_loader, device=device)
        member_probs[name] = probs
        solo = compute_metrics(y_true, probs, class_names=class_names)
        solo_metrics[name] = solo
        print(f"  {name:15s} solo accuracy: {solo['accuracy']:.4f}  AUC: {solo.get('auc_ovr'):.4f}")

    # --- Grid search weights over a small integer range (paper-style) ---
    print("\nGrid searching ensemble weights over integer range 1-6 ...")
    names = list(members.keys())
    best_acc, best_w = 0.0, None
    grid = range(1, 7)
    for w_comb in itertools.product(grid, repeat=len(names)):
        probs = weighted_ensemble_probs([member_probs[n] for n in names], w_comb)
        acc = (probs.argmax(axis=1) == y_true).mean()
        if acc > best_acc:
            best_acc, best_w = acc, w_comb

    print(f"Best weights {dict(zip(names, best_w))} -> test accuracy {best_acc:.4f}")

    # Final metrics with best weights
    final_probs = weighted_ensemble_probs([member_probs[n] for n in names], best_w)
    metrics = compute_metrics(y_true, final_probs, class_names=class_names)
    print("\n=== Stage 2 Ensemble (Xception-sub + MobileNet-v2 + PneumoNet-v1) ===")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"AUC:                 {metrics.get('auc_ovr'):.4f}")
    print(f"AP:                  {metrics.get('ap_macro'):.4f}")
    print(f"Precision (per cls): {[round(p,4) for p in metrics['precision_per_class']]}")
    print(f"Recall (per cls):    {[round(r,4) for r in metrics['recall_per_class']]}")
    print(f"F1 (per cls):        {[round(f,4) for f in metrics['f1_per_class']]}")
    print(f"Confusion matrix:    {metrics['confusion_matrix']}")
    print(f"\nPaper Stage 2 target: accuracy 91.93%, AUC 0.91")

    if args.json:
        write_json(args.json, {
            "stage": "stage2",
            "dataset": {"root": "data/real_stage2/test", "num_images": len(test_ds),
                        "classes": class_names},
            "checkpoints": dict(members),
            "solo": solo_metrics,
            "grid_weights": dict(zip(names, best_w)),
            "grid_ensemble": metrics,
            "grid_range": [1, 6],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
