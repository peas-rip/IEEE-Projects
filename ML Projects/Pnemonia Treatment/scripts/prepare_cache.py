#!/usr/bin/env python3
"""Pre-resize an ImageFolder-style dataset into a small on-disk cache.

The Kermany chest X-rays are large (often 1000-2500 px). Decoding them every
epoch is the main CPU bottleneck of training. Building a 224x224 cache once
removes that cost for the rest of the session.

Usage:
    python scripts/prepare_cache.py --src data/stage1/train --dst /content/cache224/stage1/train
    python scripts/prepare_cache.py --src data --dst /content/cache224   # whole tree

The output keeps the same sub-folder structure, so it can be passed straight
to `--data` for training or evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}


def _list_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def _resize_one(job: tuple[str, str, int, int]) -> int:
    src, dst, size, quality = job
    try:
        from PIL import Image
        with Image.open(src) as img:
            img = img.convert("L")
            img = img.resize((size, size), Image.BILINEAR)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, format="JPEG", quality=quality)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  [warn] {src}: {exc}")
        return 0


def build_cache(src: Path, dst: Path, size: int, quality: int, workers: int, force: bool) -> tuple[int, int]:
    images = _list_images(src)
    if not images:
        print(f"No images found under {src}")
        return 0, 0

    jobs = []
    skipped = 0
    for path in images:
        rel = path.relative_to(src)
        out = dst / rel.with_suffix(".jpg")
        if out.exists() and not force:
            skipped += 1
            continue
        jobs.append((str(path), str(out), size, quality))

    print(f"{src}: {len(images)} images ({len(jobs)} to build, {skipped} already cached)")
    if not jobs:
        return 0, skipped

    written = 0
    if workers <= 1:
        for job in jobs:
            written += _resize_one(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done in pool.map(_resize_one, jobs, chunksize=32):
                written += done
                if written % 500 == 0:
                    print(f"  ... {written}/{len(jobs)}")
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-resize dataset into a fast 224px cache")
    parser.add_argument("--src", required=True, help="Source dataset root (ImageFolder layout)")
    parser.add_argument("--dst", required=True, help="Destination cache root")
    parser.add_argument("--size", type=int, default=224, help="Target square size (default: 224)")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality (default: 92)")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Worker processes")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the cache exists")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.exists():
        print(f"ERROR: source not found: {src}")
        return 1

    written, skipped = build_cache(src, dst, args.size, args.quality, args.workers, args.force)
    total = len(_list_images(dst)) if dst.exists() else 0
    print(f"\nDone. Wrote {written}, skipped {skipped}. Cache now holds {total} images at {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
