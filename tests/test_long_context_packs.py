"""Pack invariants: 64 chunks, target present, gold by construction, no question.

Builder defects produced plausible-looking recall numbers before they were
caught: 9-chunk packs made top-8 selection ~89% recall by construction, and the
middle/late placement truncated the target document out of the pack entirely.
The no-question battery additionally must not contain the question string, or
BM25 scores at the ceiling and the "+3 pp" selector gate has no headroom.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from azurite.selector import chunk_text
from scripts.build_long_context import compose_pack

FILLERS = [f"filler {i} " + "y" * 900 for i in range(200)]
CONTENT = ["TARGETDOC " + "x" * 400]
DEV = Path("results/parity/long_context_dev_noquestion")


def test_pack_holds_64_chunks_and_keeps_the_target() -> None:
    for position in ("early", "middle", "late"):
        text, spans = compose_pack(CONTENT, FILLERS, positions=[position], chunks_per_pack=64)
        chunks = chunk_text(text)
        assert len(chunks) == 64
        assert "TARGETDOC" in text, f"target cut from a {position} pack"
        span = spans[0]
        assert span["chunks"]
        # A document spans one or two overlapping 2,048-byte chunks; what must
        # hold is that the span's first chunk actually contains it.
        assert "TARGETDOC" in chunks[span["chunks"][0]].text


def test_gold_chunk_moves_with_position() -> None:
    golds = [
        compose_pack(CONTENT, FILLERS, positions=[p], chunks_per_pack=64)[1][0]["chunks"][0]
        for p in ("early", "middle", "late")
    ]
    assert golds[0] < golds[1] < golds[2]


def test_distributed_evidence_yields_two_spans() -> None:
    text, spans = compose_pack(
        ["FIRSTHALF " + "a" * 300, "SECONDHALF " + "b" * 300],
        FILLERS,
        positions=["early", "late"],
        gold=(0, 1),
        chunks_per_pack=64,
    )
    assert len(spans) == 2
    assert spans[0]["start"] < spans[1]["start"]
    chunks = chunk_text(text)
    assert "FIRSTHALF" in chunks[spans[0]["chunks"][0]].text
    assert "SECONDHALF" in chunks[spans[1]["chunks"][0]].text


def test_gold_follows_the_offset_not_a_duplicate_paragraph() -> None:
    # A duplicate of the content at the front must not capture the label.
    text, spans = compose_pack(
        CONTENT, [CONTENT[0], *FILLERS], positions=["late"], chunks_per_pack=64
    )
    assert "TARGETDOC" in text
    assert spans[0]["start"] > 30_000


@pytest.mark.skipif(not (DEV / "manifest.json").exists(), reason="dev battery not built")
def test_dev_battery_never_embeds_the_question() -> None:
    manifest = json.loads((DEV / "manifest.json").read_text())
    checked = 0
    for shard in manifest["shards"]:
        with gzip.open(DEV / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                pack = json.loads(line)
                assert pack["target_question"] not in pack["context"]
                assert pack["chunks"] == 64
                assert len(chunk_text(pack["context"])) == 64
                for span in pack["gold_spans"]:
                    assert span["chunks"]
                checked += 1
                if checked >= 200:
                    return
    assert checked > 0
