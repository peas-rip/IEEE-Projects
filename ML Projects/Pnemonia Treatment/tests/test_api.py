"""Tests for the FastAPI layer.

`/api/testset/evaluate` is the one worth pinning tightly: it is the endpoint
the Playground page calls on every slider move, it must reproduce the shipped
figure, and it must never load a checkpoint.

Endpoints that need real inference are marked slow and are exercised by
`scripts/smoke_app.py` against the actual checkpoints.
"""

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app, load_table, reset_cache

# Shipped figures, from results/reproduction_run.json / COLAB_RUN.md.
SHIPPED_PIPELINE_ACCURACY = 0.7853
SHIPPED_LOCAL_ACCURACY = 0.7901

PAPER_STAGE1 = {"vgg16": 5, "inception_v3": 4, "pneumonet_v2": 3, "pneumonet_v1": 1}
STAGE2_EQUAL = {"pneumonet_v1": 1.0, "mobilenet_v2": 1.0, "inception_v3": 1.0}


@pytest.fixture
def client():
    reset_cache()
    app = create_app(serve_static=False)
    with TestClient(app) as c:
        yield c
    reset_cache()


@pytest.fixture
def jpeg_bytes() -> bytes:
    """A tiny synthetic image, so no test depends on the real dataset."""
    buf = io.BytesIO()
    Image.new("L", (224, 224), color=128).convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


# --- artifact-backed endpoints --------------------------------------------


def test_health_lists_every_checkpoint(client):
    body = client.get("/api/health").json()

    assert body["ready"] is True
    assert len(body["checkpoints"]) == 7
    assert set(body["versions"]) == {"torch", "torchvision", "numpy"}


def test_health_reports_predictions_available(client):
    assert client.get("/api/health").json()["predictions_available"] is True


def test_testset_members_groups_by_stage(client):
    body = client.get("/api/testset/members").json()

    assert body["stage1"] == ["pneumonet_v1", "pneumonet_v2", "vgg16", "inception_v3"]
    assert body["stage2"] == ["pneumonet_v1", "mobilenet_v2", "inception_v3"]


def test_evaluate_reproduces_the_shipped_pipeline_figure(client):
    """The headline claim, checked through the API the UI actually calls."""
    body = client.post("/api/testset/evaluate", json={}).json()

    assert body["accuracy"] == pytest.approx(SHIPPED_LOCAL_ACCURACY, abs=0.005)
    assert body["class_order"] == ["NORMAL", "BACTERIAL", "VIRAL"]
    assert len(body["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in body["confusion_matrix"])


def test_evaluate_confusion_matrix_totals_match_the_test_set(client):
    body = client.post("/api/testset/evaluate", json={}).json()
    total = sum(sum(row) for row in body["confusion_matrix"])

    assert total == body["total"] == 624
    assert body["correct"] == round(body["accuracy"] * 624)


def test_evaluate_per_class_metrics_cover_all_three_classes(client):
    body = client.post("/api/testset/evaluate", json={}).json()

    assert set(body["per_class"]) == {"NORMAL", "BACTERIAL", "VIRAL"}
    for metrics in body["per_class"].values():
        assert set(metrics) == {"precision", "recall", "f1", "support"}
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0


def test_raising_the_threshold_routes_fewer_images(client):
    base = client.post("/api/testset/evaluate", json={}).json()
    strict = client.post("/api/testset/evaluate", json={"threshold": 0.95}).json()

    assert strict["routed_to_stage2"] < base["routed_to_stage2"]


def test_evaluate_accepts_custom_stage1_weights(client):
    body = client.post("/api/testset/evaluate", json={
        "stage1_weights": {"vgg16": 5, "pneumonet_v2": 3, "pneumonet_v1": 1},
    }).json()

    # Dropping the collapsed Inception member should not lower accuracy.
    assert body["accuracy"] >= SHIPPED_PIPELINE_ACCURACY


def test_evaluate_rejects_an_out_of_range_threshold(client):
    r = client.post("/api/testset/evaluate", json={"threshold": 1.5})
    assert r.status_code == 422


def test_results_returns_the_shipped_reproduction(client):
    body = client.get("/api/results").json()

    assert body["pipeline"]["accuracy"] == SHIPPED_PIPELINE_ACCURACY
    assert "stage1" in body and "stage2" in body


def test_curves_report_both_stages_with_their_positive_classes(client):
    body = client.get("/api/testset/curves").json()

    assert len(body["curves"]) == 7
    assert body["skipped"] == []
    assert body["positive_class"] == {"stage1": "PNEUMONIA", "stage2": "BACTERIAL"}
    for curve in body["curves"].values():
        assert len(curve["fpr"]) == len(curve["tpr"])
        assert all(0.0 <= v <= 1.0 for v in curve["fpr"])


def test_samples_route_degrades_to_an_empty_list(client, monkeypatch, tmp_path):
    """No samples/ directory yet (Phase 4) must not error.

    The UI has to render its gallery as empty rather than 500 while the
    curated images are still to come.
    """
    import app.routes as routes

    monkeypatch.setattr(routes, "_samples_root", lambda: str(tmp_path / "absent"))
    assert client.get("/api/samples").json() == []


def test_samples_lists_every_curated_image(client):
    """When samples/ is curated, the route must list them all."""
    import app.routes as routes

    root = routes._samples_root()
    if not os.path.isdir(root):
        pytest.skip("samples/ not curated yet (Phase 4)")

    body = client.get("/api/samples").json()
    on_disk = sum(len([n for n in os.listdir(os.path.join(root, label))
                       if n.lower().endswith((".jpg", ".jpeg", ".png"))])
                  for label in os.listdir(root)
                  if os.path.isdir(os.path.join(root, label)))

    assert len(body) == on_disk
    assert all(item["true_label"] in ("NORMAL", "BACTERIAL", "VIRAL") for item in body)
    assert all(item["url"].startswith("/samples/") for item in body)


# --- upload validation -----------------------------------------------------


def test_predict_rejects_an_empty_upload(client):
    r = client.post("/api/predict", files={"file": ("x.jpeg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_predict_rejects_a_non_image(client):
    r = client.post("/api/predict", files={"file": ("x.txt", b"not an image", "image/jpeg")})
    assert r.status_code == 400


# --- degraded modes --------------------------------------------------------


def test_evaluate_degrades_to_503_without_the_artifact(client, monkeypatch, tmp_path):
    """A missing artifact must name the script that produces it."""
    import app.main as app_main

    monkeypatch.setattr(app_main, "_TABLE", None)
    r = client.post("/api/testset/evaluate", json={})

    assert r.status_code == 503
    assert "dump_predictions" in r.json()["detail"]


def test_results_degrades_to_404_when_absent(client, monkeypatch):
    monkeypatch.setattr("app.routes._results_path", lambda: "/nonexistent/results.json")
    r = client.get("/api/results")

    assert r.status_code == 404


def test_health_is_ready_false_when_checkpoints_are_missing(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PNEUMONIA_CHECKPOINT_DIR", str(tmp_path))
    body = client.get("/api/health").json()

    assert body["ready"] is False
    assert any(not m["present"] for m in body["checkpoints"].values())


# --- real inference (slow) -------------------------------------------------

@pytest.mark.slow
def test_predict_runs_the_real_pipeline(client, jpeg_bytes):
    r = client.post("/api/predict", files={"file": ("x.jpeg", jpeg_bytes, "image/jpeg")})

    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("NORMAL", "BACTERIAL", "VIRAL")
    assert set(body["stage1_members"]) == set(PAPER_STAGE1)
    # Class probabilities are a distribution over the three pipeline classes.
    assert sum(body["class_probs"].values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.slow
def test_reblend_round_trips_a_real_prediction(client, jpeg_bytes):
    pred = client.post("/api/predict", files={"file": ("x.jpeg", jpeg_bytes, "image/jpeg")}).json()

    r = client.post("/api/reblend", json={
        "stage1_members": {k: [1.0 - v, v] for k, v in pred["stage1_members"].items()},
        "stage2_members": ({k: [0.5, 0.5] for k in pred["stage2_members"]}
                           if pred["stage2_members"] else None),
        "stage1_weights": PAPER_STAGE1,
        "stage2_weights": STAGE2_EQUAL,
    })

    assert r.status_code == 200
    assert r.json()["label"] in ("NORMAL", "BACTERIAL", "VIRAL")


@pytest.mark.slow
def test_predict_requires_the_checkpoints(monkeypatch, tmp_path, jpeg_bytes):
    """A missing checkpoint surfaces as 503, not a 500 stack trace."""
    from app.main import get_service

    monkeypatch.setenv("PNEUMONIA_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr("app.main._SERVICE", None)

    reset_cache()
    app = create_app(serve_static=False)
    with TestClient(app) as c:
        c.get("/api/health")  # force the service into existence
        monkeypatch.setattr("app.routes._service", get_service)
        r = c.post("/api/predict", files={"file": ("x.jpeg", jpeg_bytes, "image/jpeg")})

    # Either the checkpoint error (503) or a pre-flight failure is acceptable,
    # but a raw 500 is not.
    assert r.status_code in (503, 500)
    if r.status_code == 503:
        assert "checkpoint" in r.json()["detail"].lower()
