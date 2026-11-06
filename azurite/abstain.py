"""Split-conformal abstention and temperature calibration.

The conformal threshold is fitted on a calibration partition disjoint from the
test rows. It is not a softmax threshold picked by looking at the locked suite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def conformal_threshold(
    scores: list[float], correct: list[bool] | None = None, *, coverage: float = 0.90
) -> float:
    """Return a max-score threshold that answers about ``coverage`` of rows.

    Coverage here is the fraction of decisions answered, matching
    ``docs/PRODUCT_SPEC.md`` (selective accuracy at a stated coverage) and the
    parity gate's "abstention 90% +/-2 pp". Nonconformity is ``1 - score``; the
    threshold is the ``(1 - coverage)`` empirical quantile on the score scale,
    so ``abstain(score)`` accepts roughly the top ``coverage`` fraction.

    ``correct`` is accepted for call-site compatibility and audit logging. It
    does not move the threshold: an earlier version forced every incorrect row
    to nonconformity 1.0 and compared that quantile against raw scores, which
    drove the threshold to 1.0 (answer nothing) whenever calibration accuracy
    was below the coverage target, and also mixed the two scales.
    """
    if correct is not None and len(scores) != len(correct):
        raise ValueError("scores and correct must have the same length")
    if not scores:
        raise ValueError("cannot fit conformal threshold on zero rows")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be in (0, 1)")
    ordered = sorted(float(score) for score in scores)
    n = len(ordered)
    rank = min(n - 1, max(0, math.floor((1.0 - coverage) * (n + 1))))
    return float(min(1.0, max(0.0, ordered[rank])))


@dataclass
class ConformalCalibrator:
    threshold: float
    coverage: float
    n: int

    @classmethod
    def fit(
        cls, scores: list[float], correct: list[bool] | None = None, *, coverage: float = 0.90
    ) -> "ConformalCalibrator":
        return cls(
            threshold=conformal_threshold(scores, correct, coverage=coverage),
            coverage=coverage,
            n=len(scores),
        )

    def abstain(self, score: float) -> bool:
        return float(score) < self.threshold

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "coverage": self.coverage, "n": self.n}
