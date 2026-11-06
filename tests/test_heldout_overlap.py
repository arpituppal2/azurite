"""Controls for the held-out duplication check.

A duplication check that reports 0% is worthless unless it can report 100% when
duplication is present. Both directions are asserted here on synthetic streams
before the real number is believed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from heldout_overlap import measure  # noqa: E402


def _stream(n_ids: int, seed: int) -> bytes:
    """Random ids inside the text alphabet, as raw uint16 bytes."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(4, 260, size=n_ids).astype("<u2")
    return ids.tobytes()


def test_duplicated_heldout_is_detected() -> None:
    """Held-out text copied from the training region must be found."""
    train = _stream(200_000, seed=1)
    # held-out = the first 20k bytes of training, i.e. 100% duplicated
    held = train[:20_000]
    m = measure(train, held, windows=50, k=32, seed=0)
    assert m["rate_full"] == 1.0, m


def test_independent_heldout_is_not_detected() -> None:
    """An independent random stream must not produce false positives."""
    train = _stream(200_000, seed=1)
    held = _stream(20_000, seed=2)
    m = measure(train, held, windows=50, k=32, seed=0)
    assert m["rate_full"] == 0.0, m
    assert m["rate_incl_fragment"] == 0.0, m


def test_fragment_detection_catches_partial_overlap() -> None:
    """A window whose first quarter is shared text is reported as a fragment,
    not as a clean miss — the difference between 'unseen' and 'partly seen'.

    Each held-out window is 32 shared ids followed by 96 unique ones, so the
    quarter-window probe (k//4 = 32 ids) matches exactly and the full window
    cannot. Starts are given explicitly: a random start would land mid-window
    and the probe's 64 bytes would straddle the boundary.
    """
    train = _stream(200_000, seed=1)
    rng = np.random.default_rng(3)
    k = 128
    shared = train[1000:1064]  # 32 ids
    unique = rng.integers(4, 260, size=(20 * (k - 32))).astype("<u2").tobytes()
    held = b"".join(shared + unique[i * 192 : (i + 1) * 192] for i in range(20))
    starts = np.arange(20) * (k * 2)
    m = measure(train, held, windows=len(starts), k=k, seed=0, starts=starts)
    assert m["rate_full"] == 0.0, m
    assert m["hits_fragment"] == 20, m
