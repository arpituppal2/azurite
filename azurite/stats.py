"""Paired statistics used by the decision comparisons.

One owner for the three quantities that appear in every verdict:

* a 95% Wilson score interval for a single accuracy;
* an exact, two-sided McNemar test on paired correctness vectors;
* a one-sided 95% paired bootstrap lower bound for an accuracy difference.

The formulas are deliberately small and dependency-free apart from NumPy.  A
second copy of any of them is a second definition of the result, so experiment
scripts and thesis generators import from here rather than re-deriving them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# 97.5th percentile of the standard normal; kept as a named constant so the
# interval cannot drift between callers.
Z = 1.959963984540054
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_PERCENTILE = 5.0


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials.

    Returns ``(nan, nan)`` for an empty sample.  The result is clamped to
    ``[0, 1]`` so a caller formatting an interval never prints a bound outside
    the probability range.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a: Sequence[int] | np.ndarray, b: Sequence[int] | np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired 0/1 correctness vectors.

    ``a`` and ``b`` are aligned by row.  Returns the discordant counts and the
    exact binomial p-value; a pair with no discordant rows is p = 1.
    """
    left = np.asarray(a, dtype=np.int8)
    right = np.asarray(b, dtype=np.int8)
    if left.shape != right.shape:
        raise ValueError(f"paired vectors differ in shape: {left.shape} != {right.shape}")
    wins = int(((left == 1) & (right == 0)).sum())
    losses = int(((left == 0) & (right == 1)).sum())
    discordant = wins + losses
    if discordant == 0:
        return {"wins": wins, "losses": losses, "p": 1.0}
    tail = sum(
        math.comb(discordant, i) for i in range(0, min(wins, losses) + 1)
    ) / (2 ** discordant)
    return {"wins": wins, "losses": losses, "p": min(1.0, 2.0 * tail)}


def bootstrap_lower(
    a: Sequence[int] | np.ndarray,
    b: Sequence[int] | np.ndarray,
    seed: int = 0,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    percentile: float = BOOTSTRAP_PERCENTILE,
) -> float:
    """One-sided paired bootstrap lower bound for ``mean(a - b)``.

    Rows are resampled together, preserving the pairing.  The default is the
    5th percentile of 10,000 resamples with seed 0, which is the protocol the
    stored verdicts were produced under.
    """
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError(f"paired vectors differ in shape: {left.shape} != {right.shape}")
    n = len(left)
    if n == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(resamples, n))
    means = (left - right)[draws].mean(axis=1)
    return float(np.percentile(means, percentile))
