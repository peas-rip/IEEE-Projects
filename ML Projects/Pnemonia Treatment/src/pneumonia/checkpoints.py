"""Locate the seven trained checkpoints on disk.

The checkpoints are large (VGG-16 alone is 512 MiB, ~1.05 GiB for all seven)
and are **not** committed to this repository. Every evaluation script, and the
web app, resolves them through this module so nothing can drift apart.

Search order for a member named ``<name>``:

1. ``$PNEUMONIA_CHECKPOINT_DIR/<name>/final_model.pt`` — explicit override;
   when this is set, nothing else is consulted.
2. ``checkpoints/<name>/final_model.pt`` — the canonical shipping layout.
3. ``Models/<name>/final_model.pt`` — alternate layout used by earlier copies.
4. ``results/<name>/final_model.pt`` — legacy training output.

Within a root, ``legacy_names`` are also tried (e.g. ``stage2_checkpoints``
for ``stage2_pneumonet_v1``).

Two different PneumoNet-v2 runs exist in the wild. ``STAGE1_PNEUMONET_V2_SHA``
is the hash of the one that reproduces the published figures; use
:func:`verify` to check a checkout against it.
"""

from __future__ import annotations

import hashlib
import os

#: Order matters: the first root that contains the file wins.
ROOT_CANDIDATES = ("checkpoints", "Models", "results")

#: Explicit override. When set, no other root is searched.
ENV_VAR = "PNEUMONIA_CHECKPOINT_DIR"

FILENAME = "final_model.pt"

#: All seven models the pipeline needs, mapped to the directories they may
#: appear under (first entry is canonical).
CHECKPOINTS: dict[str, tuple[str, ...]] = {
    "stage1_pneumonet_v1": ("stage1_pneumonet_v1",),
    "stage1_pneumonet_v2": ("stage1_pneumonet_v2",),
    "stage1_vgg16": ("stage1_vgg16",),
    "stage1_inception": ("stage1_inception", "stage1_inception_v3"),
    "stage2_pneumonet_v1": ("stage2_pneumonet_v1", "stage2_checkpoints"),
    "stage2_mobilenet": ("stage2_mobilenet",),
    "stage2_xception": ("stage2_xception", "stage2_inception_v3"),
}

#: SHA-256 (first 16 hex chars) of the checkpoint that reproduces the
#: published metrics. Recorded during Phase 0 verification on 2026-10-30.
#: Only PneumoNet-v2 is listed because it is the one model for which a second,
#: non-matching training run was found on disk.
STAGE1_PNEUMONET_V2_SHA = "a2e5863a9ce9"


def root_candidates() -> list[str]:
    """Roots to search, in priority order."""
    override = os.environ.get(ENV_VAR)
    return [override] if override else list(ROOT_CANDIDATES)


def checkpoint_path(name: str, *legacy_names: str) -> str:
    """Return the first existing path for ``name``, else the canonical path.

    ``legacy_names`` are alternative directory names kept for backward
    compatibility. When nothing exists, the canonical path under the top
    priority root is returned so error messages name the expected location.
    """
    directories = (name,) + legacy_names
    fallback = None
    for root in root_candidates():
        for directory in directories:
            candidate = os.path.join(root, directory, FILENAME)
            if fallback is None:
                fallback = candidate
            if os.path.exists(candidate):
                return candidate
    return fallback or os.path.join(name, FILENAME)


def inventory() -> dict[str, str | None]:
    """Map every known checkpoint to its resolved path, or None if missing."""
    return {
        name: (path if os.path.exists(path) else None)
        for name, directories in CHECKPOINTS.items()
        for path in [checkpoint_path(*directories)]
    }


def describe_inventory() -> str:
    """Human-readable readiness report for the console and the web health page."""
    lines = []
    for name, path in inventory().items():
        if path is None:
            expected = os.path.join(root_candidates()[0], CHECKPOINTS[name][0], FILENAME)
            lines.append(f"  MISSING  {name:22s} -> {expected}")
        else:
            lines.append(f"  ok       {name:22s} -> {path}")
    return "\n".join(lines)


def sha256_head(path: str, length: int = 16) -> str:
    """Return the first ``length`` hex chars of a file's SHA-256."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def verify(name: str = "stage1_pneumonet_v2") -> bool:
    """Check a resolved checkpoint against its known hash. None if unhashed."""
    path = inventory().get(name)
    if path is None:
        return False
    if name == "stage1_pneumonet_v2":
        return sha256_head(path) == STAGE1_PNEUMONET_V2_SHA
    return True


def describe_missing(paths: dict[str, str]) -> str:
    """Build a multi-line "here is what is missing and where it was expected" note."""
    return "\n".join(f"  {name}: {path}" for name, path in sorted(paths.items()))
