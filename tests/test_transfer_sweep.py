"""The paired test is the only part of the sweep that can be wrong silently.

A wrong p-value does not crash anything; it just produces a confident-looking
number beside a difference that is not there. These are hand-computable cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from transfer_sweep import mcnemar_exact  # noqa: E402


def test_no_discordant_pairs_is_p_one() -> None:
    assert mcnemar_exact(0, 0) == 1.0


def test_symmetric_agreement_is_p_one() -> None:
    """Equal disagreement in both directions is exactly the null."""
    assert mcnemar_exact(10, 10) == pytest.approx(1.0)
    assert mcnemar_exact(100, 100) == pytest.approx(1.0)


def test_five_of_five_one_way() -> None:
    # b=0, c=5: 2 * P(X=0 | n=5, p=.5) = 2/32
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)


def test_ten_of_ten_one_way() -> None:
    # 2 * (1/2)^10 = 2/1024
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)


def test_is_symmetric_in_its_arguments() -> None:
    assert mcnemar_exact(4, 27) == pytest.approx(mcnemar_exact(27, 4))


def test_is_monotone_in_asymmetry() -> None:
    """The more lopsided the discordant pairs, the smaller the p."""
    values = [mcnemar_exact(b, 30) for b in (15, 10, 5, 1)]
    assert values == sorted(values, reverse=True)


def test_stays_a_probability() -> None:
    for b in range(0, 40, 3):
        for c in range(0, 40, 7):
            p = mcnemar_exact(b, c)
            assert 0.0 <= p <= 1.0
