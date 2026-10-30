"""Tests for JSON result serialisation."""

import json

import numpy as np

from pneumonia.reporting import to_jsonable, write_json


def test_to_jsonable_converts_numpy_scalars_and_arrays():
    payload = {
        "accuracy": np.float32(0.7853),
        "count": np.int64(624),
        "flag": np.bool_(True),
        "cm": np.array([[152, 6, 76], [3, 207, 32], [2, 15, 131]]),
        "per_class": (np.float64(0.5), np.float64(0.25)),
    }

    got = to_jsonable(payload)

    assert got["accuracy"] == float(np.float32(0.7853))
    assert got["count"] == 624 and isinstance(got["count"], int)
    assert got["flag"] is True
    assert got["cm"] == [[152, 6, 76], [3, 207, 32], [2, 15, 131]]
    assert got["per_class"] == [0.5, 0.25]


def test_to_jsonable_recurses_into_nested_structures():
    got = to_jsonable({"a": [{"b": np.arange(2)}]})
    assert got == {"a": [{"b": [0, 1]}]}


def test_write_json_creates_directories_and_records_timestamp(tmp_path):
    target = tmp_path / "nested" / "run.json"
    write_json(target, {"accuracy": np.float64(0.5)})

    document = json.loads(target.read_text(encoding="utf-8"))

    assert document["accuracy"] == 0.5
    assert "generated_at" in document
    # Must be serialisable exactly as written (no numpy objects leaking through).
    assert json.loads(json.dumps(document)) == document


def test_write_json_overwrites_previous_run(tmp_path):
    target = tmp_path / "run.json"
    write_json(target, {"accuracy": 0.1})
    write_json(target, {"accuracy": 0.2})

    assert json.loads(target.read_text(encoding="utf-8"))["accuracy"] == 0.2
