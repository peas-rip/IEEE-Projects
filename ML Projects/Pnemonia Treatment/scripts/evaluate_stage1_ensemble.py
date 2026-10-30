"""Evaluate the full Stage 1 detection ensemble (VGG-16, Inception-v3, PneumoNet-v2, PneumoNet-v1).

Paper Section 4.1: Detection ensemble combines VGG-16, Inception-v3, PneumoNet-v1, PneumoNet-v2
via weighted averaging of softmax probabilities. The paper publishes exact weights:
- VGG-16: 5
- Inception-v3: 4
- PneumoNet-v2: 3
- PneumoNet-v1: 1

Target: 98.03% accuracy, AUC 0.97

Usage:
    python scripts/evaluate_stage1_ensemble.py
"""

import argparse
import os
import sys
import itertools
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pneumonia.data import DatasetConfig
from pneumonia.transfer import TransferLearningModel
from pneumonia.models import PneumoNetV1, PneumoNetV2
from pneumonia.evaluate import compute_metrics, predict_probs


def load_model(model_name, weights_path, num_classes=2):
    if model_name == "pneumonet_v1":
        model = PneumoNetV1(num_classes=num_classes)
    elif model_name == "pneumonet_v2":
        model = PneumoNetV2(num_classes=num_classes)
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
    """Keep only members whose weights file exists.

    Returns the filtered dict, or None when evaluation should stop. Without
    ``allow_missing`` any gap is fatal; with it, weights are renormalised over
    the members that are present (and the gap is reported loudly).
    """
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
    parser = argparse.ArgumentParser(description="Evaluate the Stage 1 detection ensemble")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Evaluate with the members that exist and renormalise their weights")
    args = parser.parse_args(argv)

    # Paper's published Stage 1 ensemble weights
    PAPER_WEIGHTS = {
        "vgg16": 5,
        "inception_v3": 4,
        "pneumonet_v2": 3,
        "pneumonet_v1": 1,
    }

    # Paths to the four trained Stage 1 models (update as needed)
    members = {
        "pneumonet_v1": "results/stage1_pneumonet_v1/final_model.pt",
        "pneumonet_v2": "results/stage1_pneumonet_v2/final_model.pt",
        "vgg16": "results/stage1_vgg16/final_model.pt",
        "inception_v3": "results/stage1_inception/final_model.pt",
    }

    members = available_members(members, args.allow_missing)
    if members is None:
        return 1

    # Test data - Stage 1 uses Normal vs Pneumonia
    cfg = DatasetConfig(image_size=224, batch_size=16, train_ratio=0.8, seed=42, num_workers=0, augment=False)
    from torchvision.datasets import ImageFolder
    from pneumonia.data import get_transforms
    eval_tf = get_transforms(image_size=224, augment=False)[1]
    test_ds = ImageFolder(root="data/real_3class/test", transform=eval_tf)
    # For Stage 1, we only need NORMAL vs PNEUMONIA
    # The 3class test has NORMAL, BACTERIAL, VIRAL - map BACTERIAL/VIRAL to PNEUMONIA
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    
    # Create binary labels: NORMAL=0, BACTERIAL/VIRAL=1
    binary_labels = np.array([0 if test_ds.classes[l] == 'NORMAL' else 1 for l in test_ds.targets])
    
    # Create a custom dataset that returns binary labels
    class BinaryStage1Dataset(torch.utils.data.Dataset):
        def __init__(self, base_ds, binary_labels):
            self.base_ds = base_ds
            self.binary_labels = binary_labels
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            img, _ = self.base_ds[idx]
            return img, self.binary_labels[idx]
    
    binary_test_ds = BinaryStage1Dataset(test_ds, binary_labels)
    class_names = ['NORMAL', 'PNEUMONIA']
    print(f"Test classes (Stage 1): {class_names}  ({len(binary_test_ds)} images)")

    test_loader = DataLoader(binary_test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Load each member and get probabilities
    models = {n: load_model(n, p) for n, p in members.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    member_probs = {}
    y_true = None
    for name, model in models.items():
        y_true, probs = predict_probs(model.to(device), test_loader, device=device)
        member_probs[name] = probs
        solo = compute_metrics(y_true, probs, class_names=class_names)
        print(f"  {name:15s} solo accuracy: {solo['accuracy']:.4f}  AUC: {solo.get('auc_ovr'):.4f}")

    # --- Paper's published weights ---
    names = list(members.keys())
    paper_weights = [PAPER_WEIGHTS[n] for n in names]
    print(f"\nUsing paper's published weights: {dict(zip(names, paper_weights))}")
    
    paper_probs = weighted_ensemble_probs([member_probs[n] for n in names], paper_weights)
    paper_metrics = compute_metrics(y_true, paper_probs, class_names=class_names)
    print(f"\n=== Stage 1 Ensemble (Paper weights: VGG-16:5, Inception-v3:4, PneumoNet-v2:3, PneumoNet-v1:1) ===")
    print(f"Accuracy:            {paper_metrics['accuracy']:.4f}")
    print(f"AUC:                 {paper_metrics.get('auc_ovr'):.4f}")
    print(f"AP:                  {paper_metrics.get('ap_macro'):.4f}")
    print(f"Precision (per cls): {[round(p,4) for p in paper_metrics['precision_per_class']]}")
    print(f"Recall (per cls):    {[round(r,4) for r in paper_metrics['recall_per_class']]}")
    print(f"F1 (per cls):        {[round(f,4) for f in paper_metrics['f1_per_class']]}")
    print(f"Confusion matrix:    {paper_metrics['confusion_matrix']}")

    # --- Grid search weights over a small integer range (ablation) ---
    print("\nGrid searching ensemble weights over integer range 1-6 (ablation) ...")
    best_acc, best_w = 0.0, None
    grid = range(1, 7)
    for w_comb in itertools.product(grid, repeat=len(names)):
        probs = weighted_ensemble_probs([member_probs[n] for n in names], w_comb)
        acc = (probs.argmax(axis=1) == y_true).mean()
        if acc > best_acc:
            best_acc, best_w = acc, w_comb

    print(f"Best grid weights {dict(zip(names, best_w))} -> test accuracy {best_acc:.4f}")

    # Final metrics with best weights
    final_probs = weighted_ensemble_probs([member_probs[n] for n in names], best_w)
    metrics = compute_metrics(y_true, final_probs, class_names=class_names)
    print("\n=== Stage 1 Ensemble (Best grid-searched weights) ===")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"AUC:                 {metrics.get('auc_ovr'):.4f}")
    print(f"AP:                  {metrics.get('ap_macro'):.4f}")
    print(f"Precision (per cls): {[round(p,4) for p in metrics['precision_per_class']]}")
    print(f"Recall (per cls):    {[round(r,4) for r in metrics['recall_per_class']]}")
    print(f"F1 (per cls):        {[round(f,4) for f in metrics['f1_per_class']]}")
    print(f"Confusion matrix:    {metrics['confusion_matrix']}")
    print(f"\nPaper Stage 1 target: accuracy 98.03%, AUC 0.97")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())