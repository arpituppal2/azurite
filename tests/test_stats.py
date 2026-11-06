"""The paired statistics are a single owner; these are its invariants.

The stored Jev verdicts quote Wilson bounds, exact McNemar p-values and a paired
bootstrap lower bound. If any of the three drifts, every table and verdict
downstream is wrong in the same direction, so each is checked against a value
computed independently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from azurite.stats import bootstrap_lower, mcnemar_exact, wilson


def test_wilson_matches_stored_len_a2c_interval() -> None:
    lo, hi = wilson(2_683, 3_080)
    assert lo == pytest.approx(0.8588060714377721, abs=1e-12)
    assert hi == pytest.approx(0.8824771723753517, abs=1e-12)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_empty_sample_is_nan_not_a_division_error() -> None:
    lo, hi = wilson(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_mcnemar_exact_matches_stored_len_a2c_vs_old_a0() -> None:
    wins, losses = 149, 188
    a = np.asarray([1] * wins + [0] * losses, dtype=np.int8)
    b = np.asarray([0] * wins + [1] * losses, dtype=np.int8)
    result = mcnemar_exact(a, b)
    assert result["wins"] == wins
    assert result["losses"] == losses
    assert result["p"] == pytest.approx(0.038295325895522966, abs=1e-18)


def test_mcnemar_with_no_discordance_is_one() -> None:
    a = np.asarray([1, 0, 1, 1], dtype=np.int8)
    assert mcnemar_exact(a, a) == {"wins": 0, "losses": 0, "p": 1.0}


def test_bootstrap_constant_difference_is_exact() -> None:
    ones = np.ones(100, dtype=np.int8)
    zeros = np.zeros(100, dtype=np.int8)
    assert bootstrap_lower(ones, zeros, seed=0) == 1.0
    assert bootstrap_lower(ones, ones, seed=0) == 0.0


def test_bootstrap_rejects_unpaired_shapes() -> None:
    with pytest.raises(ValueError):
        bootstrap_lower([1, 0], [1])
