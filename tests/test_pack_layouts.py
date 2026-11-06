"""Random pack layouts must always fit the 64-chunk window.

The builder repeatedly produced plausible packs that broke one invariant at a
time: targets truncated at late positions, gold spans pointing at distractors,
and separator bytes pushing a span past the end of the window. A property test
over mixed document counts and lengths catches the whole class instead of the
one layout that happened to fail.
"""

from __future__ import annotations

import random

from azurite.selector import chunk_text
from scripts.build_long_context import compose_pack, length_bounds

FILLERS = [f"filler {i} " + "y" * 900 for i in range(200)]


def test_random_layouts_always_fit_and_keep_gold() -> None:
    rng = random.Random(0)
    low, high = length_bounds(64)
    for _ in range(200):
        n = rng.choice([1, 1, 3, 9])
        content = ["MARKER%d " % i + "z" * rng.randrange(50, 4_000) for i in range(n)]
        positions = [rng.choice(["early", "middle", "late"]) for _ in content]
        text, spans = compose_pack(content, FILLERS, positions=positions, gold=(0,), chunks_per_pack=64)
        size = len(text.encode("utf-8"))
        assert low <= size <= high, size
        assert len(chunk_text(text)) == 64
        assert spans and spans[0]["chunks"]
        # The marker may straddle two overlapping chunks, so check the span's
        # byte range rather than one chunk's text.
        raw = text.encode("utf-8")
        start = spans[0]["start"]
        assert raw[start : start + 7] == b"MARKER0"
