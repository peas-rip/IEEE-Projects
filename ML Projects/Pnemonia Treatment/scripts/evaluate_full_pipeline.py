#!/usr/bin/env python3
"""Evaluate the full two-stage pipeline on test data.

This evaluates the complete paper workflow:
1. Stage 1: Normal vs Pneumonia (ensemble)
2. If Pneumonia → Stage 2: Bacterial vs Viral (ensemble)
3. Final 3-class prediction: Normal, Bacterial, Viral

Usage:
    python scripts/evaluate_full_pipeline.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pneumonia.ensemble import DetectionEnsemble, ClassificationEnsemble, TwoStagePipeline
from pneumonia.transfer import TransferLearningModel
from pneumonia.models import PneumoNetV1, PneumoNetV2
from pneumonia.data import DatasetConfig, get_transforms
from pneumonia.evaluate import compute_metrics, predict_probs
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def load_model(model_name, weights_path, num_classes=2):
    """Load a model from checkpoint."""
    if model_name == "pneumonet_v1":
        model = PneumoNetV1(num_classes=num_classes)
    elif model_name == "pneumonet_v2":
        model = PneumoNetV2(num_classes=num_classes)
    else:
        model = TransferLearningModel(name=model_name, num_classes=num_classes, pretrained=False)
    
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    # Paper's published Stage 1 ensemble weights
    PAPER_WEIGHTS_STAGE1 = {
        "vgg16": 5,
        "inception_v3": 4,
        "pneumonet_v2": 3,
        "pneumonet_v1": 1,
    }

    # Stage 1 model paths (Normal vs Pneumonia)
    stage1_members = {
        "pneumonet_v1": "results/stage1_pneumonet_v1/final_model.pt",
        "pneumonet_v2": "results/stage1_pneumonet_v2/final_model.pt",
        "vgg16": "results/stage1_vgg16/final_model.pt",
        "inception_v3": "results/stage1_inception/final_model.pt",
    }

    # Stage 2 model paths (Bacterial vs Viral)
    stage2_members = {
        "pneumonet_v1": "results/stage2_pneumonet_v1/final_model.pt",
        "mobilenet_v2": "results/stage2_mobilenet/final_model.pt",
        "inception_v3": "results/stage2_xception/final_model.pt",  # Xception substitute
    }

    # Check all models exist
    all_members = {**stage1_members, **stage2_members}
    for name, path in all_members.items():
        if not os.path.exists(path):
            print(f"MISSING weights: {path} (train this model first)")
            return 1

    # Load test data - use 3-class test for full pipeline evaluation
    cfg = DatasetConfig(image_size=224, batch_size=16, train_ratio=0.8, seed=42, num_workers=0, augment=False)
    eval_tf = get_transforms(image_size=224, augment=False)[1]
    
    # Test directory should have 3 classes: NORMAL, BACTERIAL, VIRAL
    test_root = "data/real_3class/test"
    if not os.path.exists(test_root):
        # Try alternative: stage1/test (has NORMAL, PNEUMONIA) + stage2/test
        test_root = "data/stage1/test"
        print(f"Using Stage 1 test data: {test_root}")
        use_binary_stage1 = True
    else:
        use_binary_stage1 = False
        print(f"Using 3-class test data: {test_root}")

    test_ds = ImageFolder(root=test_root, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    class_names = test_ds.classes
    print(f"Test classes: {class_names}  ({len(test_ds)} images)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Stage 1 models
    print("\nLoading Stage 1 models...")
    stage1_models = {}
    for name, path in stage1_members.items():
        stage1_models[name] = load_model(name, path, num_classes=2).to(device)
        print(f"  Loaded {name}")

    # Load Stage 2 models
    print("\nLoading Stage 2 models...")
    stage2_models = {}
    for name, path in stage2_members.items():
        stage2_models[name] = load_model(name, path, num_classes=2).to(device)
        print(f"  Loaded {name}")

    # Create ensembles with paper weights
    detection = DetectionEnsemble(num_classes=2, weights=[
        PAPER_WEIGHTS_STAGE1["pneumonet_v1"],
        PAPER_WEIGHTS_STAGE1["pneumonet_v2"],
        PAPER_WEIGHTS_STAGE1["vgg16"],
        PAPER_WEIGHTS_STAGE1["inception_v3"],
    ])
    detection.models = torch.nn.ModuleList([stage1_models["pneumonet_v1"],
                                              stage1_models["pneumonet_v2"],
                                              stage1_models["vgg16"],
                                              stage1_models["inception_v3"]])
    detection = detection.to(device)
    detection.eval()

    # Stage 2: use equal weights (paper doesn't publish)
    classification = ClassificationEnsemble(num_classes=2, weights=[1.0, 1.0, 1.0])
    classification.models = torch.nn.ModuleList([stage2_models["inception_v3"],
                                                   stage2_models["mobilenet_v2"],
                                                   stage2_models["pneumonet_v1"]])
    classification = classification.to(device)
    classification.eval()

    # Full pipeline
    pipeline = TwoStagePipeline(detection, classification).to(device)
    pipeline.eval()

    # Run inference
    print("\nRunning full pipeline inference...")
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            result = pipeline(images)
            preds = result["pred"].cpu().numpy()
            
            # Get probabilities for 3 classes
            # Pipeline returns: 0=Normal, 1=Bacterial, 2=Viral
            det_probs = result["detection_probs"].cpu().numpy()  # (N, 2)
            cls_probs = result["classification_probs"].cpu().numpy()  # (N, 2)
            
            # Combine into 3-class probs
            probs_3class = np.zeros((len(preds), 3))
            probs_3class[:, 0] = det_probs[:, 0]  # P(Normal)
            # For Pneumonia cases, split between Bacterial/Viral
            pneu_mask = preds != 0
            probs_3class[pneu_mask, 1] = det_probs[pneu_mask, 1] * cls_probs[pneu_mask, 0]  # P(Bacterial)
            probs_3class[pneu_mask, 2] = det_probs[pneu_mask, 1] * cls_probs[pneu_mask, 1]  # P(Viral)
            
            all_preds.append(preds)
            all_probs.append(probs_3class)
            all_targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_probs = np.vstack(all_probs)
    y_true = np.concatenate(all_targets)

    # Map true labels to 3-class
    if use_binary_stage1:
        # Stage 1 test has NORMAL=0, PNEUMONIA=1
        # Need to map to 3-class for proper evaluation
        print("Warning: Using binary Stage 1 test data. Full 3-class evaluation not possible.")
        class_names_3 = ["NORMAL", "PNEUMONIA"]
    else:
        class_names_3 = ["NORMAL", "BACTERIAL", "VIRAL"]
        # Verify mapping
        expected = ["NORMAL", "BACTERIAL", "VIRAL"]
        if list(class_names) != expected:
            print(f"Warning: Test classes {class_names} don't match expected {expected}")

    # Compute metrics
    print("\n" + "="*60)
    print("  FULL TWO-STAGE PIPELINE RESULTS")
    print("="*60)
    
    metrics = compute_metrics(y_true, y_probs, class_names=class_names_3)
    
    print(f"Overall Accuracy:    {metrics['accuracy']:.4f}")
    print(f"AUC (OvR):           {metrics.get('auc_ovr', 'N/A'):.4f}")
    print(f"AP (macro):          {metrics.get('ap_macro', 'N/A'):.4f}")
    print(f"Precision (per cls): {[round(p,4) for p in metrics['precision_per_class']]}")
    print(f"Recall (per cls):    {[round(r,4) for r in metrics['recall_per_class']]}")
    print(f"F1 (per cls):        {[round(f,4) for f in metrics['f1_per_class']]}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    # Per-stage breakdown
    print("\n" + "-"*60)
    print("  STAGE 1 BREAKDOWN (Normal vs Pneumonia)")
    print("-"*60)
    
    # Stage 1: binary labels
    y_true_binary = (y_true != 0).astype(int)  # Normal=0, Pneumonia=1
    y_pred_binary = (y_pred != 0).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    print(f"Stage 1 Accuracy:  {accuracy_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Stage 1 Precision: {precision_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Stage 1 Recall:    {recall_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Stage 1 F1:        {f1_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Stage 1 CM:\n{confusion_matrix(y_true_binary, y_pred_binary)}")

    # Stage 2: only on Pneumonia cases
    pneu_mask = y_true_binary == 1
    if pneu_mask.sum() > 0:
        print("\n" + "-"*60)
        print("  STAGE 2 BREAKDOWN (Bacterial vs Viral | Pneumonia)")
        print("-"*60)
        y_true_cls = y_true[pneu_mask] - 1  # 0=Bacterial, 1=Viral
        y_pred_cls = y_pred[pneu_mask] - 1
        print(f"Stage 2 Accuracy:  {accuracy_score(y_true_cls, y_pred_cls):.4f}")
        print(f"Stage 2 Precision: {precision_score(y_true_cls, y_pred_cls):.4f}")
        print(f"Stage 2 Recall:    {recall_score(y_true_cls, y_pred_cls):.4f}")
        print(f"Stage 2 F1:        {f1_score(y_true_cls, y_pred_cls):.4f}")
        print(f"Stage 2 CM:\n{confusion_matrix(y_true_cls, y_pred_cls)}")
        print(f"Samples: {pneu_mask.sum()} Pneumonia cases")

    print("\n" + "="*60)
    print("  PAPER TARGETS")
    print("="*60)
    print("Stage 1 (Detection):  Accuracy 98.03%, AUC 0.97")
    print("Stage 2 (Classif.):   Accuracy 91.93%, AUC 0.91")
    print("Full Pipeline:        3-class classification")

    return 0


if __name__ == "__main__":
    sys.exit(main())