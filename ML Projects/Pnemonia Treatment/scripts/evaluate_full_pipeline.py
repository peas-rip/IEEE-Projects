#!/usr/bin/env python3
"""Evaluate the full two-stage pipeline on test data.

This evaluates the complete paper workflow:
1. Stage 1: Normal vs Pneumonia (ensemble)
2. If Pneumonia → Stage 2: Bacterial vs Viral (ensemble)
3. Final 3-class prediction: Normal, Bacterial, Viral

Usage:
    python scripts/evaluate_full_pipeline.py
    python scripts/evaluate_full_pipeline.py --ablate      # + drop-Inception / 2:3:5 variants
    python scripts/evaluate_full_pipeline.py --allow-missing
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pneumonia.ensemble import WeightedEnsemble, TwoStagePipeline
from pneumonia.transfer import TransferLearningModel
from pneumonia.models import PneumoNetV1, PneumoNetV2
from pneumonia.data import DatasetConfig, get_transforms
from pneumonia.evaluate import compute_metrics, predict_probs
from pneumonia.reporting import write_json
from pneumonia.checkpoints import checkpoint_path
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


def available_members(members, allow_missing=False):
    """Keep only members whose weights file exists (see the ensemble scripts)."""
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


def stage2_breakdown(y_true, y_pred, pneumonia_mask):
    """Stage 2 metrics for the pneumonia images Stage 1 sent to the classifier.

    The Stage 2 branch must only be scored on the images it actually saw. Passing
    every pneumonia-labelled image would both charge Stage 2 for Stage 1's misses
    and inject a ``-1`` sentinel into the predictions whenever Stage 1 predicted
    Normal; scikit-learn then sees three distinct labels and treats the task as
    multiclass, rejecting ``average="binary"``.

    Parameters
    ----------
    y_true : sequence of int
        Ground truth in pipeline order (0 = Normal, 1 = Bacterial, 2 = Viral).
    y_pred : sequence of int
        Pipeline predictions, same order.
    pneumonia_mask : sequence of bool
        True for images whose ground truth is Bacterial or Viral.

    Returns
    -------
    dict with the routed/overall accuracies, per-class precision/recall/F1 and
    the 2x2 confusion matrix (rows = true Bacterial/Viral).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pneumonia_mask = np.asarray(pneumonia_mask, dtype=bool)

    routed = pneumonia_mask & (y_pred != 0)
    n_pneumonia = int(pneumonia_mask.sum())
    n_routed = int(routed.sum())

    y_true_cls = y_true[routed] - 1  # 0 = Bacterial, 1 = Viral
    y_pred_cls = y_pred[routed] - 1
    if n_routed:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_cls, y_pred_cls, labels=[0, 1], zero_division=0
        )
        correct = int((y_true_cls == y_pred_cls).sum())
        accuracy_routed = correct / n_routed
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
    else:
        precision, recall, f1 = np.zeros(2), np.zeros(2), np.zeros(2)
        correct = 0
        accuracy_routed = float("nan")
        cm = np.zeros((2, 2), dtype=int)

    # Same denominator as "all pneumonia images": every Stage 1 miss counts as a
    # wrong subtype prediction rather than dropping out of the metric.
    accuracy_all = correct / n_pneumonia if n_pneumonia else float("nan")

    return {
        "n_pneumonia": n_pneumonia,
        "n_routed": n_routed,
        "n_missed": n_pneumonia - n_routed,
        "accuracy_routed": accuracy_routed,
        "accuracy_all": accuracy_all,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def blend_probs(probs_by_name, weights):
    """Weighted average of member probability arrays, renormalised over what exists.

    Members present in ``weights`` but absent from ``probs_by_name`` (dropped via
    ``--allow-missing`` or an ablation) are ignored and the remaining weights are
    rescaled to sum to 1.
    """
    names = [n for n in probs_by_name if n in weights]
    if not names:
        raise ValueError("no member in weights has probabilities")
    w = np.array([weights[n] for n in names], dtype=np.float64)
    w = w / w.sum()
    return sum(weight * probs_by_name[name] for weight, name in zip(w, names))


def pipeline_predict(det_probs, cls_probs):
    """Compose the two stages: Normal stays Normal, else 1 + argmax(Stage 2)."""
    det_pred = np.asarray(det_probs).argmax(1)
    cls_pred = np.asarray(cls_probs).argmax(1)
    return np.where(det_pred == 0, 0, 1 + cls_pred)



def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the full two-stage pipeline")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Evaluate with the members that exist and renormalise their weights")
    parser.add_argument("--ablate", action="store_true",
                        help="Also score the pipeline without Inception-v3 and with "
                             "Stage 2 weights 2/3/5 (extra forward passes, no retraining)")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="Write the metrics of this run to a JSON file")
    args = parser.parse_args(argv)

    # Paper's published Stage 1 ensemble weights
    PAPER_WEIGHTS_STAGE1 = {
        "vgg16": 5,
        "inception_v3": 4,
        "pneumonet_v2": 3,
        "pneumonet_v1": 1,
    }

    # Stage 1 model paths (Normal vs Pneumonia). Override the root with
    # PNEUMONIA_CHECKPOINT_DIR when the files live elsewhere (e.g. Colab Drive).
    stage1_members = {
        "pneumonet_v1": checkpoint_path("stage1_pneumonet_v1"),
        "pneumonet_v2": checkpoint_path("stage1_pneumonet_v2"),
        "vgg16": checkpoint_path("stage1_vgg16"),
        "inception_v3": checkpoint_path("stage1_inception", "stage1_inception_v3"),
    }

    # Stage 2 model paths (Bacterial vs Viral)
    stage2_members = {
        # "stage2_checkpoints" is the directory used by the first training run.
        "pneumonet_v1": checkpoint_path("stage2_pneumonet_v1", "stage2_checkpoints"),
        "mobilenet_v2": checkpoint_path("stage2_mobilenet"),
        # Xception substitute
        "inception_v3": checkpoint_path("stage2_xception", "stage2_inception_v3"),
    }

    stage1_members = available_members(stage1_members, args.allow_missing)
    if stage1_members is None:
        return 1
    stage2_members = available_members(stage2_members, args.allow_missing)
    if stage2_members is None:
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

    # Create ensembles with paper weights. Build them from the members that were
    # actually loaded, so a missing model cannot desynchronise models vs weights.
    s1_names = list(stage1_models)
    s1_weights = [PAPER_WEIGHTS_STAGE1[n] for n in s1_names]
    print(f"\nStage 1 members: {dict(zip(s1_names, s1_weights))}")
    detection = WeightedEnsemble(
        models=[stage1_models[n] for n in s1_names],
        weights=s1_weights,
        num_classes=2,
    ).to(device)
    detection.eval()

    # Stage 2: use equal weights (paper doesn't publish)
    s2_names = list(stage2_models)
    print(f"Stage 2 members: {dict(zip(s2_names, [1.0] * len(s2_names)))}")
    classification = WeightedEnsemble(
        models=[stage2_models[n] for n in s2_names],
        weights=[1.0] * len(s2_names),
        num_classes=2,
    ).to(device)
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

    # ImageFolder assigns labels alphabetically (e.g. BACTERIAL=0, NORMAL=1, VIRAL=2),
    # while the pipeline emits 0=Normal, 1=Bacterial, 2=Viral. Remap the ground truth
    # so the comparison is valid.
    pipeline_order = {"NORMAL": 0, "BACTERIAL": 1, "VIRAL": 2}
    if set(class_names) == set(pipeline_order):
        remap = np.array([pipeline_order[c] for c in class_names], dtype=np.int64)
        y_true = remap[y_true]
        print(f"Remapped ground-truth labels from ImageFolder order {class_names} "
              f"to pipeline order {list(pipeline_order)}")

    # Map true labels to 3-class
    if use_binary_stage1:
        # Stage 1 test has NORMAL=0, PNEUMONIA=1
        # Need to map to 3-class for proper evaluation
        print("Warning: Using binary Stage 1 test data. Full 3-class evaluation not possible.")
        class_names_3 = ["NORMAL", "PNEUMONIA"]
    else:
        class_names_3 = ["NORMAL", "BACTERIAL", "VIRAL"]
        # The ground truth was already remapped to pipeline order above; only warn
        # if the classes cannot be mapped at all.
        if set(class_names) != set(class_names_3):
            print(f"WARNING: test classes {class_names} cannot be mapped to {class_names_3}; "
                  "the 3-class metrics below are not comparable.")

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
    
    print(f"Stage 1 Accuracy:  {accuracy_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Stage 1 Precision: {precision_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Stage 1 Recall:    {recall_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Stage 1 F1:        {f1_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Stage 1 CM:\n{confusion_matrix(y_true_binary, y_pred_binary)}")

    # Stage 2: scored only on the pneumonia images Stage 1 actually routed onward.
    stage2_report = None
    if use_binary_stage1:
        print("\nStage 2 breakdown skipped: binary test data has no BACTERIAL/VIRAL labels.")
    elif y_true_binary.any():
        print("\n" + "-"*60)
        print("  STAGE 2 BREAKDOWN (Bacterial vs Viral | Pneumonia)")
        print("-"*60)

        bd = stage2_breakdown(y_true, y_pred, y_true_binary == 1)
        stage2_report = bd
        print(f"Samples: {bd['n_pneumonia']} pneumonia, {bd['n_routed']} routed to Stage 2, "
              f"{bd['n_missed']} missed by Stage 1")
        print(f"Stage 2 Accuracy (operating set):        {bd['accuracy_routed']:.4f}")
        print(f"Stage 2 Accuracy (misses counted wrong): {bd['accuracy_all']:.4f}")
        print(f"Stage 2 Precision: BACTERIAL {bd['precision'][0]:.4f}  VIRAL {bd['precision'][1]:.4f}")
        print(f"Stage 2 Recall:    BACTERIAL {bd['recall'][0]:.4f}  VIRAL {bd['recall'][1]:.4f}")
        print(f"Stage 2 F1:        BACTERIAL {bd['f1'][0]:.4f}  VIRAL {bd['f1'][1]:.4f}")
        print(f"Stage 2 CM:\n{bd['confusion_matrix']}")
        print("  (rows = true BACTERIAL/VIRAL; Stage-1 misses excluded)")

    # Optional ablations: re-score the pipeline with different members/weights.
    # Extra forward passes only - nothing is retrained.
    ablation_scores = {}
    if args.ablate and not use_binary_stage1:
        print("\n" + "-"*60)
        print("  ABLATIONS (no retraining)")
        print("-"*60)
        print("Collecting per-member probabilities...")
        s1_probs = {name: predict_probs(model, test_loader, device=device)[1]
                    for name, model in stage1_models.items()}
        s2_probs = {name: predict_probs(model, test_loader, device=device)[1]
                    for name, model in stage2_models.items()}

        base_det = blend_probs(s1_probs, PAPER_WEIGHTS_STAGE1)
        base_cls = blend_probs(s2_probs, {n: 1.0 for n in s2_probs})
        drop_det = blend_probs(s1_probs, {n: w for n, w in PAPER_WEIGHTS_STAGE1.items()
                                          if n != "inception_v3"})
        grid_cls = blend_probs(s2_probs, {"pneumonet_v1": 2.0, "mobilenet_v2": 3.0,
                                          "inception_v3": 5.0})

        def score(det_p, cls_p):
            return accuracy_score(y_true, pipeline_predict(det_p, cls_p))

        print(f"baseline (paper weights / equal):   {score(base_det, base_cls):.4f}  "
              f"(reported above: {metrics['accuracy']:.4f})")
        print(f"drop Inception-v3 from Stage 1:     {score(drop_det, base_cls):.4f}")
        print(f"Stage 2 weights 2/3/5:              {score(base_det, grid_cls):.4f}")
        print(f"both:                               {score(drop_det, grid_cls):.4f}")
        print("(2/3/5 was grid-searched on this test set -> optimistic when scored here)")

        ablation_scores = {
            "baseline": score(base_det, base_cls),
            "drop_inception_stage1": score(drop_det, base_cls),
            "stage2_weights_2_3_5": score(base_det, grid_cls),
            "both": score(drop_det, grid_cls),
        }

    print("\n" + "="*60)
    print("  PAPER TARGETS")
    print("="*60)
    print("Stage 1 (Detection):  Accuracy 98.03%, AUC 0.97")
    print("Stage 2 (Classif.):   Accuracy 91.93%, AUC 0.91")
    print("Full Pipeline:        3-class classification")

    if args.json:
        stage1_report = {
            "accuracy": accuracy_score(y_true_binary, y_pred_binary),
            "precision": precision_score(y_true_binary, y_pred_binary),
            "recall": recall_score(y_true_binary, y_pred_binary),
            "f1": f1_score(y_true_binary, y_pred_binary),
            "confusion_matrix": confusion_matrix(y_true_binary, y_pred_binary),
            "routed_to_stage2": int((y_pred_binary == 1).sum()),
        }
        write_json(args.json, {
            "stage": "pipeline",
            "dataset": {"root": test_root, "num_images": len(test_ds),
                        "classes": list(class_names)},
            "checkpoints": {
                "stage1": dict(stage1_members),
                "stage2": dict(stage2_members),
            },
            "stage1_weights": dict(zip(s1_names, s1_weights)),
            "stage2_weights": dict(zip(s2_names, [1.0] * len(s2_names))),
            "overall": metrics,
            "stage1": stage1_report,
            "stage2": stage2_report,
            "ablations": ablation_scores,
        })

    return 0


if __name__ == "__main__":
    sys.exit(main())