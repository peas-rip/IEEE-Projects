"""Locate the seven trained checkpoints on disk.

The checkpoints are ~100 MB each and are **not** shipped with this repository.
Every evaluation script resolves them through this module so the three scripts
cannot drift apart.

Resolution order for a member named ``<name>``:

1. ``$PNEUMONIA_CHECKPOINT_DIR/<name>/final_model.pt``
2. ``results/<name>/final_model.pt``
3. any legacy directory names the caller passes as fallbacks

On Google Colab the files live on Drive, so set the root explicitly::

    export PNEUMONIA_CHECKPOINT_DIR=/content/drive/MyDrive/PneumoniaProject/checkpoints

Nothing is downloaded automatically — if a checkpoint is missing, the message
tells you which path was tried.
"""

from __future__ import annotations

import os

DEFAULT_ROOT = "results"
ENV_VAR = "PNEUMONIA_CHECKPOINT_DIR"
FILENAME = "final_model.pt"


def checkpoint_root() -> str:
    """Return the directory that contains one sub-directory per model."""
    return os.environ.get(ENV_VAR) or DEFAULT_ROOT


def checkpoint_path(name: str, *legacy_names: str) -> str:
    """Return the path for ``name``, preferring one that already exists.

    ``legacy_names`` are alternative directory names kept for backward
    compatibility with earlier runs (e.g. ``stage2_checkpoints``). If no
    candidate exists, the first (canonical) path is returned so the caller's
    "missing checkpoint" message names the expected location.
    """
    root = checkpoint_root()
    candidates = [
        os.path.join(root, directory, FILENAME)
        for directory in (name,) + legacy_names
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def describe_missing(paths: dict[str, str]) -> str:
    """Build a multi-line "here is what is missing and where it was expected" note."""
    lines = [f"  {name}: {path}" for name, path in sorted(paths.items())]
    return "\n".join(lines)
