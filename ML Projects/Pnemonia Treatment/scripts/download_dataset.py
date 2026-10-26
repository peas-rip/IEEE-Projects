#!/usr/bin/env python3
"""Download and prepare Kermany pneumonia dataset for two-stage training.

Usage:
    python scripts/download_dataset.py --kaggle-username USER --kaggle-key KEY
    python scripts/download_dataset.py --data-dir /path/to/data  # if already downloaded
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def download_kaggle_dataset(username: str, key: str, download_dir: str = "/content/data"):
    """Download dataset from Kaggle using API."""
    import kaggle
    
    # Set credentials
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    
    print("Downloading chest-xray-pneumonia dataset from Kaggle...")
    kaggle.api.dataset_download_files(
        'paultimothymooney/chest-xray-pneumonia',
        path=download_dir,
        unzip=True
    )
    print(f"Dataset downloaded to {download_dir}")


def reorganize_dataset(src_root: str, dst_stage1: str, dst_stage2: str):
    """Reorganize dataset for two-stage pipeline.
    
    Original structure:
        chest_xray/
            train/
                NORMAL/
                PNEUMONIA/
                    BACTERIA/
                    VIRUS/
            test/
                NORMAL/
                PNEUMONIA/
                    BACTERIA/
                    VIRUS/
    
    Stage 1 (Detection): Normal vs Pneumonia
        stage1/
            train/
                NORMAL/
                PNEUMONIA/  (merged BACTERIA + VIRUS)
            test/
                NORMAL/
                PNEUMONIA/
    
    Stage 2 (Classification): Bacterial vs Viral
        stage2/
            train/
                BACTERIAL/
                VIRAL/
            test/
                BACTERIAL/
                VIRAL/
    """
    src = Path(src_root)
    dst1 = Path(dst_stage1)
    dst2 = Path(dst_stage2)
    
    # Create directory structure
    for split in ['train', 'test']:
        for cls in ['NORMAL', 'PNEUMONIA']:
            (dst1 / split / cls).mkdir(parents=True, exist_ok=True)
        for cls in ['BACTERIAL', 'VIRAL']:
            (dst2 / split / cls).mkdir(parents=True, exist_ok=True)
    
    print("Reorganizing dataset...")
    
    for split in ['train', 'test']:
        # Stage 1: Normal
        normal_src = src / split / 'NORMAL'
        normal_dst = dst1 / split / 'NORMAL'
        if normal_src.exists():
            count = 0
            for f in normal_src.iterdir():
                if f.is_file():
                    shutil.copy2(f, normal_dst / f.name)
                    count += 1
            print(f"  Stage1 {split}/NORMAL: {count} images")
        
        # Stage 1: Pneumonia (merge BACTERIA + VIRUS)
        pneu_dst = dst1 / split / 'PNEUMONIA'
        pneu_count = 0
        for subcls in ['BACTERIA', 'VIRUS']:
            sub_src = src / split / 'PNEUMONIA' / subcls
            if sub_src.exists():
                for f in sub_src.iterdir():
                    if f.is_file():
                        shutil.copy2(f, pneu_dst / f.name)
                        pneu_count += 1
        print(f"  Stage1 {split}/PNEUMONIA: {pneu_count} images")
        
        # Stage 2: Bacterial
        bact_src = src / split / 'PNEUMONIA' / 'BACTERIA'
        bact_dst = dst2 / split / 'BACTERIAL'
        if bact_src.exists():
            count = 0
            for f in bact_src.iterdir():
                if f.is_file():
                    shutil.copy2(f, bact_dst / f.name)
                    count += 1
            print(f"  Stage2 {split}/BACTERIAL: {count} images")
        
        # Stage 2: Viral
        viral_src = src / split / 'PNEUMONIA' / 'VIRUS'
        viral_dst = dst2 / split / 'VIRAL'
        if viral_src.exists():
            count = 0
            for f in viral_src.iterdir():
                if f.is_file():
                    shutil.copy2(f, viral_dst / f.name)
                    count += 1
            print(f"  Stage2 {split}/VIRAL: {count} images")
    
    # Verify
    print("\nVerification:")
    for name, path in [('Stage 1', dst1), ('Stage 2', dst2)]:
        print(f"\n{name}:")
        for split in ['train', 'test']:
            for cls in sorted((path / split).iterdir()):
                count = len([f for f in cls.iterdir() if f.is_file()])
                print(f"  {split}/{cls.name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare pneumonia dataset")
    parser.add_argument('--kaggle-username', help='Kaggle username')
    parser.add_argument('--kaggle-key', help='Kaggle API key')
    parser.add_argument('--data-dir', default='/content/data', help='Data directory')
    parser.add_argument('--skip-download', action='store_true', help='Skip download, only reorganize')
    args = parser.parse_args()
    
    src_root = os.path.join(args.data_dir, 'chest_xray')
    dst_stage1 = os.path.join(args.data_dir, 'stage1')
    dst_stage2 = os.path.join(args.data_dir, 'stage2')
    
    if not args.skip_download:
        if not args.kaggle_username or not args.kaggle_key:
            print("Error: --kaggle-username and --kaggle-key required for download")
            sys.exit(1)
        download_kaggle_dataset(args.kaggle_username, args.kaggle_key, args.data_dir)
    
    if not os.path.exists(src_root):
        print(f"Error: Source dataset not found at {src_root}")
        sys.exit(1)
    
    reorganize_dataset(src_root, dst_stage1, dst_stage2)
    print("\nDone! Dataset ready for training.")
    print(f"  Stage 1 (Detection): {dst_stage1}")
    print(f"  Stage 2 (Classification): {dst_stage2}")


if __name__ == "__main__":
    main()