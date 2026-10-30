"""Request/response schemas for the web app.

Kept separate from the routes so the API contract is readable in one place and
the model layer never sees a Pydantic type.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from pneumonia.service import PAPER_STAGE1_WEIGHTS, DEFAULT_STAGE2_WEIGHTS


class MemberWeights(BaseModel):
    """Ensemble weights, keyed by short member name."""

    model_config = {"extra": "allow"}

    def as_dict(self, defaults: dict[str, float]) -> dict[str, float]:
        """Merge submitted weights over the defaults.

        Anything the client omits keeps its default, so the frontend can send
        only the sliders the user actually moved.
        """
        merged = dict(defaults)
        merged.update({k: float(v) for k, v in self.model_extra.items()})
        return merged


class ReblendRequest(BaseModel):
    """Recompute a verdict from member probabilities cached by predict()."""

    stage1_members: dict[str, list[float]] = Field(
        ..., description="Stage 1 per-member [P(normal), P(pneumonia)] pairs"
    )
    stage2_members: dict[str, list[float]] | None = Field(
        default=None, description="Stage 2 per-member [P(bacterial), P(viral)] pairs"
    )
    stage1_weights: MemberWeights = Field(default_factory=MemberWeights)
    stage2_weights: MemberWeights = Field(default_factory=MemberWeights)
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Stage 1 routing threshold")


class EvaluateRequest(BaseModel):
    """Score a weight/threshold configuration against the bundled test set."""

    stage1_weights: MemberWeights = Field(default_factory=MemberWeights)
    stage2_weights: MemberWeights = Field(default_factory=MemberWeights)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class EvaluateResponse(BaseModel):
    accuracy: float
    total: int
    correct: int
    routed_to_stage2: int
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float]]
    class_order: list[str]


class HealthMember(BaseModel):
    path: str
    present: bool


class HealthResponse(BaseModel):
    ready: bool
    device: str
    loaded: bool
    checkpoints: dict[str, HealthMember]
    versions: dict[str, str]
    predictions_available: bool


class SampleImage(BaseModel):
    name: str
    url: str
    true_label: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    class_probs: dict[str, float]
    routed: bool
    stage1_members: dict[str, float]
    stage2_members: dict[str, float] | None
    stage1_probs: list[float]
    stage2_probs: list[float] | None


class ErrorDetail(BaseModel):
    detail: str


API_ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorDetail, "description": "Malformed request"},
    404: {"model": ErrorDetail, "description": "Not found"},
    503: {"model": ErrorDetail, "description": "Checkpoints not installed"},
}

DEFAULT_STAGE1_WEIGHTS = dict(PAPER_STAGE1_WEIGHTS)
