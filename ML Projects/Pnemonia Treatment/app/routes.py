"""HTTP endpoints for the web app.

Thin by design: parse and validate, call into `pneumonia.service` (real
inference) or `app.predictions` (the bundled artifact), and return DTOs.
No model arithmetic happens here.
"""

from __future__ import annotations

import io
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from pneumonia.service import (
    DEFAULT_STAGE2_WEIGHTS,
    DEFAULT_THRESHOLD,
    PAPER_STAGE1_WEIGHTS,
    PipelineService,
    Prediction,
    blend_probs,
)

from .predictions import PredictionTable, load_prediction_table, member_names, score
from .schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HealthMember,
    HealthResponse,
    PredictResponse,
    ReblendRequest,
    SampleImage,
)

MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # a 224px CXR is well under this

router = APIRouter(prefix="/api")


def _service() -> PipelineService:
    from .main import get_service

    return get_service()


def _table_available() -> bool:
    from .main import get_table

    return get_table() is not None


def _require_table() -> PredictionTable:
    """The bundled artifact, or a 503 naming the script that produces it."""
    from .main import get_table

    table = get_table()
    if table is None:
        raise HTTPException(
            status_code=503,
            detail="prediction artifact not available; run scripts/dump_predictions.py",
        )
    return table


def _as_prediction(req: ReblendRequest) -> Prediction:
    """Rebuild a Prediction from the member probabilities the client echoes back."""
    stage1 = {k: np.asarray(v, dtype=np.float32) for k, v in req.stage1_members.items()}
    stage2 = None
    if req.stage2_members is not None:
        stage2 = {k: np.asarray(v, dtype=np.float32) for k, v in req.stage2_members.items()}

    s1_weights = req.stage1_weights.as_dict(PAPER_STAGE1_WEIGHTS)
    s2_weights = req.stage2_weights.as_dict(DEFAULT_STAGE2_WEIGHTS)

    stage1_blend = blend_probs(stage1, s1_weights)
    p_pneumonia = float(stage1_blend[1])
    routed = p_pneumonia >= req.threshold

    stage2_blend = None
    if routed and stage2 is not None:
        stage2_blend = blend_probs(stage2, s2_weights)

    return Prediction(
        label="",
        confidence=0.0,
        class_probs={},
        routed=routed,
        stage1=stage1,
        stage2=stage2,
        stage1_probs=stage1_blend,
        stage2_probs=stage2_blend,
    )


def _predict_response(pred: Prediction) -> PredictResponse:
    return PredictResponse(
        label=pred.label,
        confidence=pred.confidence,
        class_probs={k: float(v) for k, v in pred.class_probs.items()},
        routed=pred.routed,
        stage1_members={k: float(v[1]) for k, v in pred.stage1.items()},
        stage2_members=(
            {k: float(v[1]) for k, v in pred.stage2.items()} if pred.stage2 else None
        ),
        stage1_probs=pred.stage1_probs.tolist(),
        stage2_probs=pred.stage2_probs.tolist() if pred.stage2_probs is not None else None,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness report. Safe to call before any model is loaded."""
    raw = _service().health()
    return HealthResponse(
        ready=raw["ready"],
        device=raw["device"],
        loaded=raw["loaded"],
        checkpoints={k: HealthMember(**v) for k, v in raw["checkpoints"].items()},
        versions=raw["versions"],
        predictions_available=_table_available(),
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(file: Annotated[UploadFile, File(...)]) -> PredictResponse:
    """Run the full two-stage pipeline on an uploaded chest X-ray."""
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large")
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="not an image") from exc

    try:
        pred = _service().predict(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _predict_response(pred)


@router.post("/reblend", response_model=PredictResponse)
def reblend(req: ReblendRequest) -> PredictResponse:
    """Recompute a verdict from member probabilities cached by predict().

    No model runs, which is why weight sliders feel instant.
    """
    partial = _as_prediction(req)

    if partial.routed and partial.stage2 is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "this configuration routes to Stage 2 but no Stage 2 members were "
                "sent; re-run /api/predict to obtain them"
            ),
        )

    s1_weights = req.stage1_weights.as_dict(PAPER_STAGE1_WEIGHTS)
    s2_weights = req.stage2_weights.as_dict(DEFAULT_STAGE2_WEIGHTS)

    try:
        pred = _service().reblend(partial, s1_weights, s2_weights, req.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _predict_response(pred)


def _samples_root() -> str:
    """Curated demo images live next to the app package."""
    import os

    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")


@router.get("/samples", response_model=list[SampleImage])
def samples() -> list[SampleImage]:
    """Curated demo images, so a fresh install has something to click on."""
    import os

    root = _samples_root()
    if not os.path.isdir(root):
        return []

    out: list[SampleImage] = []
    for label in sorted(os.listdir(root)):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        for name in sorted(os.listdir(label_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append(SampleImage(name=name, url=f"/samples/{label}/{name}",
                                       true_label=label))
    return out


@router.get("/testset/members")
def testset_members() -> dict[str, list[str]]:
    """Member names per stage, so the UI can label its sliders correctly."""
    return member_names(_require_table())


@router.post("/testset/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Score a weight/threshold configuration against the bundled test set."""
    table = _require_table()

    try:
        result = score(
            table,
            stage1_weights=req.stage1_weights.as_dict(PAPER_STAGE1_WEIGHTS),
            stage2_weights=req.stage2_weights.as_dict(DEFAULT_STAGE2_WEIGHTS),
            threshold=req.threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EvaluateResponse(**result)


def _results_path() -> str:
    """Where the shipped reproduction run lives."""
    import os

    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "results", "reproduction_run.json")


@router.get("/results")
def results() -> dict:
    """The shipped reproduction run, for the Results page."""
    import json
    import os

    path = _results_path()
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="results/reproduction_run.json not found")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@router.get("/testset/curves")
def curves(limit: Annotated[int, Query(ge=1, le=624)] = 624) -> dict:
    """Per-member ROC curves for the test set.

    The two stages have different positive classes, so they cannot share one
    label array. Stage 1 is NORMAL vs PNEUMONIA and its positive class is
    PNEUMONIA (y_true_binary == 1). Stage 2 is BACTERIAL vs VIRAL and its
    positive class is BACTERIAL (y_true == 1), and it is only defined over
    images that are bacterial or viral in the first place. Scoring Stage 2
    against the pneumonia mask would label it with the wrong task.
    """
    from sklearn.metrics import roc_curve

    table = _require_table()

    def _curve(labels: np.ndarray, scores: np.ndarray) -> dict | None:
        """One ROC curve, or None when a class is absent from the subset."""
        if len(set(labels.tolist())) < 2:
            return None
        fpr, tpr, _ = roc_curve(labels, scores)
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    out: dict[str, dict] = {}
    skipped: list[str] = []

    for name, probs in table.stage1.items():
        curve = _curve(table.y_true_binary, probs[:, 1])
        if curve is None:
            skipped.append(f"stage1/{name}")
        else:
            out[f"stage1/{name}"] = curve

    # Stage 2 positives are BACTERIAL (pipeline index 1); restrict to the
    # two pneumonia subtypes so NORMAL images cannot enter the metric.
    is_subtype = table.y_true != 0
    stage2_labels = (table.y_true[is_subtype] == 1).astype(np.int64)
    for name, probs in table.stage2.items():
        curve = _curve(stage2_labels, probs[is_subtype, 0])
        if curve is None:
            skipped.append(f"stage2/{name}")
        else:
            out[f"stage2/{name}"] = curve

    return {
        "curves": out,
        "limit": limit,
        "positive_class": {"stage1": "PNEUMONIA", "stage2": "BACTERIAL"},
        "skipped": skipped,
    }
