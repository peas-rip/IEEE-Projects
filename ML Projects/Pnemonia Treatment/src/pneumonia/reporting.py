"""Serialise evaluation results to JSON.

Every evaluation script accepts ``--json PATH`` so a run can be archived and the
numbers in ``docs/RESULTS.md`` can be regenerated instead of retyped.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np


def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays and tuples into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path, payload):
    """Write ``payload`` to ``path`` with a UTC timestamp, creating directories."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    document = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    document.update(to_jsonable(payload))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    print(f"[JSON] wrote {path}")
