#!/usr/bin/env python3
"""Prepare a local chest X-ray dataset for the two-stage pneumonia pipeline.

This is a Colab/local-only helper: it does not contact any external service.
Point it at an already-downloaded copy of the Kermany "chest_xray" folder
(or a .zip of it) and it produces the Stage 1 and Stage 2 layouts.

Supported source layouts (both are handled):
  1. Nested sub-folders (original archive):
        chest_xray/train/PNEUMONIA/BACTERIA/*.jpeg
        chest_xray/train/PNEUMONIA/VIRUS/*.jpeg
  2. Flat PNEUMONIA folder with labelled filenames:
        chest_xray/train/PNEUMONIA/BACTERIA_1234.jpeg
        chest_xray/train/PNEUMONIA/VIRUS_5678.jpeg

Output layouts:
  data/stage1/{train,test}/{NORMAL,PNEUMONIA}
  data/stage2/{train,test}/{BACTERIAL,VIRAL}

Usage:
    python scripts/prepare_dataset.py --source /content/chest_xray --dst data
    python scripts/prepare_dataset.py --source /content/drive/MyDrive/chest_xray.zip --dst data
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _classify_pneumonia(path: Path) -> str | None:
    """Return BACTERIAL / VIRAL for a pneumonia image, or None if unknown."""
    name = path.name.lower()
    if "bacteria" in name:
        return "BACTERIAL"
    if "virus" in name or "viral" in name:
        return "VIRAL"
    return None


def _extract_if_zip(source: Path, workdir: Path) -> Path:
    if source.is_file() and source.suffix.lower() == ".zip":
        workdir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {source} -> {workdir} ...")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(workdir)
        # Find the chest_xray folder inside the extracted tree
        for root, dirs, _ in os.walk(workdir):
            if "chest_xray" in [d.lower() for d in dirs]:
                for d in dirs:
                    if d.lower() == "chest_xray":
                        return Path(root) / d
        return workdir
    return source


def _locate_chest_xray(source: Path) -> Path:
    """Accept either the chest_xray folder itself or its parent."""
    if (source / "train").is_dir() and (source / "test").is_dir():
        return source
    if (source / "chest_xray").is_dir():
        return source / "chest_xray"
    # Last resort: search one level down
    for child in source.iterdir():
        if child.is_dir() and (child / "train").is_dir():
            return child
    raise FileNotFoundError(
        f"Could not find a 'train'/'test' layout under {source}. "
        "Pass the chest_xray folder (or a zip of it)."
    )


def _copy(src: Path, dst: Path) -> int:
    if not _is_image(src):
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / src.name)
    return 1


def _collect_pneumonia(pneu_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (bacterial_files, viral_files) for one split's PNEUMONIA folder."""
    bacterial: list[Path] = []
    viral: list[Path] = []

    if not pneu_dir.is_dir():
        return bacterial, viral

    for path in pneu_dir.rglob("*"):
        if not _is_image(path):
            continue
        # Nested sub-folder form: parent folder name carries the label.
        parent = path.parent.name.lower()
        if parent in {"bacteria", "bacterial"}:
            bacterial.append(path)
            continue
        if parent in {"virus", "viral"}:
            viral.append(path)
            continue
        # Flat form: filename carries the label.
        label = _classify_pneumonia(path)
        if label == "BACTERIAL":
            bacterial.append(path)
        elif label == "VIRAL":
            viral.append(path)

    return bacterial, viral


def reorganize(source: Path, dst: Path, splits: list[str]) -> None:
    stage1 = dst / "stage1"
    stage2 = dst / "stage2"

    for split in splits:
        split_dir = source / split
        if not split_dir.is_dir():
            print(f"  [skip] {split}: not found in source")
            continue

        # ---- Stage 1: NORMAL vs PNEUMONIA ---------------------------------
        normal_dir = split_dir / "NORMAL"
        normal_count = 0
        if normal_dir.is_dir():
            for f in normal_dir.iterdir():
                normal_count += _copy(f, stage1 / split / "NORMAL")

        bacterial, viral = _collect_pneumonia(split_dir / "PNEUMONIA")
        pneumonia_count = 0
        for f in bacterial + viral:
            pneumonia_count += _copy(f, stage1 / split / "PNEUMONIA")

        # ---- Stage 2: BACTERIAL vs VIRAL ----------------------------------
        for f in bacterial:
            _copy(f, stage2 / split / "BACTERIAL")
        for f in viral:
            _copy(f, stage2 / split / "VIRAL")

        print(
            f"  {split}: NORMAL={normal_count}  PNEUMONIA={pneumonia_count}  "
            f"(BACTERIAL={len(bacterial)}, VIRAL={len(viral)})"
        )


def verify(dst: Path, splits: list[str]) -> None:
    print("\nVerification:")
    for stage, classes in (("stage1", ("NORMAL", "PNEUMONIA")), ("stage2", ("BACTERIAL", "VIRAL"))):
        print(f"\n{stage}:")
        for split in splits:
            row = []
            for cls in classes:
                d = dst / stage / split / cls
                count = len([p for p in d.iterdir() if _is_image(p)]) if d.is_dir() else 0
                row.append(f"{cls}={count}")
            print(f"  {split}: " + "  ".join(row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Kermany chest X-ray dataset (local, no cloud services)")
    parser.add_argument("--source", required=True, help="chest_xray folder or .zip of it")
    parser.add_argument("--dst", default="data", help="Destination data directory (default: data)")
    parser.add_argument("--splits", default="train,test", help="Comma separated splits (default: train,test)")
    parser.add_argument("--clean", action="store_true", help="Delete existing stage1/stage2 before writing")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if not source.exists():
        print(f"ERROR: source not found: {source}")
        return 1

    if args.clean:
        for name in ("stage1", "stage2"):
            target = dst / name
            if target.exists():
                print(f"Removing {target}")
                shutil.rmtree(target)

    source = _extract_if_zip(source, dst / "_extracted")
    source = _locate_chest_xray(source)
    print(f"Source: {source}")
    print(f"Destination: {dst}")

    reorganize(source, dst, splits)
    verify(dst, splits)
    print("\nDone. Stage 1 and Stage 2 folders are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
