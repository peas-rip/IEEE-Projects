#!/usr/bin/env python3
"""Unified training & evaluation pipeline for the two-stage pneumonia detection system.

This script orchestrates the complete paper workflow:
1. Train 4 Stage 1 models (Normal vs Pneumonia)
2. Train 3 Stage 2 models (Bacterial vs Viral) 
3. Evaluate Stage 1 ensemble with paper weights
4. Evaluate Stage 2 ensemble with grid search
5. Run full two-stage pipeline evaluation

Usage:
    python scripts/run_full_pipeline.py --data-dir /path/to/data --output-dir results
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_cmd(cmd, description):
    """Run command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full two-stage pneumonia pipeline")
    parser.add_argument("--data-dir", default="/content/data", help="Root data directory")
    parser.add_argument("--output-dir", default="results", help="Output directory for checkpoints")
    parser.add_argument("--stage", choices=["all", "stage1", "stage2", "eval"], default="all",
                        help="Which stage to run")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    stage1_train = os.path.join(args.data_dir, "stage1", "train")
    stage2_train = os.path.join(args.data_dir, "stage2", "train")
    stage1_test = os.path.join(args.data_dir, "stage1", "test")
    stage2_test = os.path.join(args.data_dir, "stage2", "test")

    # Verify data exists
    for path in [stage1_train, stage2_train]:
        if not os.path.exists(path):
            print(f"ERROR: Data not found at {path}")
            print("Run scripts/prepare_dataset.py first (see README: 'Prepare the data')")
            return 1

    # Stage 1 model configs
    stage1_models = [
        ("pneumonet_v1", "configs/stage1_pneumonet_v1.yaml"),
        ("pneumonet_v2", "configs/stage1_pneumonet_v2.yaml"),
        ("vgg16", "configs/stage1_vgg16.yaml"),
        ("inception_v3", "configs/stage1_inception.yaml"),
    ]

    # Stage 2 model configs
    stage2_models = [
        ("pneumonet_v1", "configs/stage2_pneumonet_v1.yaml"),
        ("mobilenet_v2", "configs/stage2_mobilenet.yaml"),
        ("inception_v3", "configs/stage2_xception.yaml"),
    ]

    python_exe = sys.executable
    base_cmd = [python_exe, "-m", "pneumonia", "train"]

    success = True

    # ===== STAGE 1 TRAINING =====
    if args.stage in ["all", "stage1"]:
        print("\n" + "="*60)
        print("  STAGE 1: DETECTION (Normal vs Pneumonia)")
        print("="*60)

        for model_name, config in stage1_models:
            ckpt_dir = os.path.join(args.output_dir, f"stage1_{model_name}")
            cmd = base_cmd + [
                "--config", config,
                "--data", stage1_train,
                "--checkpoint-dir", ckpt_dir
            ]
            if not run_cmd(cmd, f"Train Stage 1: {model_name}"):
                success = False
                break

    # ===== STAGE 2 TRAINING =====
    if success and args.stage in ["all", "stage2"]:
        print("\n" + "="*60)
        print("  STAGE 2: CLASSIFICATION (Bacterial vs Viral)")
        print("="*60)

        for model_name, config in stage2_models:
            ckpt_dir = os.path.join(args.output_dir, f"stage2_{model_name}")
            cmd = base_cmd + [
                "--config", config,
                "--data", stage2_train,
                "--checkpoint-dir", ckpt_dir
            ]
            if not run_cmd(cmd, f"Train Stage 2: {model_name}"):
                success = False
                break

    # ===== ENSEMBLE EVALUATION =====
    if success and args.stage in ["all", "eval"]:
        print("\n" + "="*60)
        print("  ENSEMBLE EVALUATION")
        print("="*60)

        # Stage 1 ensemble eval
        if not run_cmd(
            [python_exe, "scripts/evaluate_stage1_ensemble.py"],
            "Evaluate Stage 1 Ensemble (paper weights)"
        ):
            success = False

        # Stage 2 ensemble eval
        if success and not run_cmd(
            [python_exe, "scripts/evaluate_stage2_ensemble.py"],
            "Evaluate Stage 2 Ensemble (grid search)"
        ):
            success = False

    # ===== FULL PIPELINE EVALUATION =====
    if success and args.stage in ["all", "eval"]:
        print("\n" + "="*60)
        print("  FULL TWO-STAGE PIPELINE EVALUATION")
        print("="*60)
        
        if not run_cmd(
            [python_exe, "scripts/evaluate_full_pipeline.py"],
            "Evaluate Full Two-Stage Pipeline"
        ):
            success = False

    if success:
        print("\n" + "="*60)
        print("  ✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("  ✗ PIPELINE FAILED")
        print("="*60)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())