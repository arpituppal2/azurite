"""The public decision contract.

The contract is deliberately independent of the model. Validation happens here,
before any tensor is allocated; an invalid request can never return a label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .select_read import SelectRead


class InvalidRequest(ValueError):
    """The request violates the decision contract."""


@dataclass(frozen=True)
class Alternative:
    id: str
    label: str
    description: str | None = None


@dataclass(frozen=True)
class DecisionRequest:
    context: str
    question: str
    alternatives: Sequence[Alternative]
    evidence_required: bool = False
    abstain_allowed: bool = True


@dataclass(frozen=True)
class EvidenceSpan:
    doc_id: str
    byte_start: int
    byte_end: int
    text: str


@dataclass
class DecisionResult:
    choice_id: str | None
    choice_label: str | None
    abstained: bool
    abstain_reason: str | None
    scores: dict[str, float]
    calibrated: bool
    evidence: list[EvidenceSpan]
    model_id: str
    checkpoint_sha256: str
    calibration_id: str | None
    latency_ms: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice_id": self.choice_id,
            "choice_label": self.choice_label,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "scores": dict(self.scores),
            "calibrated": self.calibrated,
            "evidence": [
                {"doc_id": e.doc_id, "byte_start": e.byte_start,
                 "byte_end": e.byte_end, "text": e.text}
                for e in self.evidence
            ],
            "model_id": self.model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "calibration_id": self.calibration_id,
            "latency_ms": self.latency_ms,
            "diagnostics": dict(self.diagnostics),
        }


def _validate(request: DecisionRequest) -> None:
    if not isinstance(request.context, str) or not request.context.strip():
        raise InvalidRequest("context must be a non-empty string")
    if not isinstance(request.question, str):
        raise InvalidRequest("question must be a string")
    if not request.alternatives:
        raise InvalidRequest("alternatives must be non-empty")
    ids = [a.id for a in request.alternatives]
    if len(set(ids)) != len(ids):
        raise InvalidRequest("alternative ids must be unique")
    for alt in request.alternatives:
        if not str(alt.id).strip() or not str(alt.label).strip():
            raise InvalidRequest("alternative id and label must be non-empty")


def decide(
    request: DecisionRequest,
    engine: SelectRead,
    *,
    model_id: str = "azurite-select-read",
    checkpoint_sha256: str = "",
    calibration_id: str | None = None,
) -> DecisionResult:
    """Validate, run the engine, and guarantee the result invariant."""
    _validate(request)
    output = engine.decide(
        request.context,
        request.question,
        [a.label for a in request.alternatives],
        evidence_required=request.evidence_required,
        abstain_allowed=request.abstain_allowed,
    )
    scores = {
        alt.id: float(output.scores[i]) if i < len(output.scores) else 0.0
        for i, alt in enumerate(request.alternatives)
    }
    if output.abstained:
        choice_id = None
        choice_label = None
        reason = output.abstain_reason or "insufficient_evidence"
    else:
        if not (0 <= output.choice < len(request.alternatives)):
            raise RuntimeError("engine returned an out-of-range choice")
        alt = request.alternatives[output.choice]
        choice_id, choice_label, reason = alt.id, alt.label, None
    evidence = [
        EvidenceSpan(
            doc_id="context",
            byte_start=c.byte_start,
            byte_end=c.byte_end,
            text=c.text,
        )
        for c in output.selected
    ]
    if request.evidence_required and not evidence and not output.abstained:
        # The engine is supposed to handle this, but the contract is enforced
        # one more time at the boundary.
        return DecisionResult(
            choice_id=None, choice_label=None, abstained=True,
            abstain_reason="insufficient_evidence", scores=scores,
            calibrated=False, evidence=[], model_id=model_id,
            checkpoint_sha256=checkpoint_sha256, calibration_id=calibration_id,
            latency_ms=0.0, diagnostics={"reader_tokens": output.reader_tokens},
        )
    return DecisionResult(
        choice_id=choice_id,
        choice_label=choice_label,
        abstained=output.abstained,
        abstain_reason=reason,
        scores=scores,
        calibrated=False,
        evidence=evidence,
        model_id=model_id,
        checkpoint_sha256=checkpoint_sha256,
        calibration_id=calibration_id,
        latency_ms=0.0,
        diagnostics={"reader_tokens": output.reader_tokens},
    )
