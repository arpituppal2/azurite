from __future__ import annotations

import pytest
import torch

from azurite.abstain import conformal_threshold
from azurite.api import Alternative, DecisionRequest, InvalidRequest, decide
from azurite.battery import summarize, wilson
from azurite.config import EncoderConfig
from azurite.model import Azurite
from azurite.parity import JEV_REFERENCE, gate
from azurite.select_read import SelectRead, SelectReadOutput
from azurite.selector import BM25ChunkSelector, chunk_text, recall_at_k
from azurite.tokenizers import ByteTokenizer


def tiny_reader() -> Azurite:
    return Azurite(EncoderConfig(
        d_model=32, n_layer=1, n_head=4, d_ff=64,
        vocab_size=260, max_seq=64,
    ))


def test_chunk_offsets_cover_text():
    text = "alpha beta gamma delta epsilon zeta"
    chunks = chunk_text(text, chunk_bytes=16, stride=8)
    assert chunks
    assert chunks[0].byte_start == 0
    assert chunks[-1].byte_end == len(text.encode("utf-8"))
    for c in chunks:
        assert 0 <= c.byte_start < c.byte_end <= len(text.encode("utf-8"))
        assert c.text


def test_bm25_selects_evidence_chunk():
    text = "bank card arrival\n\nrefund for a duplicate charge\n\nloan repayment schedule"
    chunks = chunk_text(text, chunk_bytes=32, stride=16)
    selector = BM25ChunkSelector(chunks)
    picked = selector.select("duplicate charge refund", top_k=2)
    assert picked
    assert any("duplicate" in c.text for c in picked)


def test_recall_at_k():
    chunks = chunk_text("a b c d e f g h i j", chunk_bytes=8, stride=4)
    assert recall_at_k(chunks[:2], [chunks[0].index], k=2) == 1.0
    assert recall_at_k(chunks[:1], [chunks[-1].index], k=1) == 0.0


def test_select_read_contract():
    engine = SelectRead(tiny_reader(), ByteTokenizer(), top_k=2)
    out = engine.decide("alpha beta gamma\n\ndelta epsilon", "which?", ["yes", "no"])
    assert isinstance(out, SelectReadOutput)
    assert len(out.scores) == 2
    assert out.choice in (0, 1)
    assert out.selected


def test_api_rejects_duplicate_ids():
    request = DecisionRequest(
        context="x", question="q",
        alternatives=[Alternative("a", "yes"), Alternative("a", "no")],
    )
    with pytest.raises(InvalidRequest):
        decide(request, engine=object())


def test_api_requires_nonempty_context():
    request = DecisionRequest(
        context=" ", question="q",
        alternatives=[Alternative("a", "yes")],
    )
    with pytest.raises(InvalidRequest):
        decide(request, engine=object())


def test_api_abstention_has_no_choice():
    class Fake:
        def decide(self, *args, **kwargs):
            return SelectReadOutput(
                choice=-1, scores=[0.5, 0.5], abstained=True,
                abstain_reason="insufficient_evidence", selected=[], reader_tokens=0,
            )

    request = DecisionRequest(
        context="x", question="q",
        alternatives=[Alternative("yes", "Yes"), Alternative("no", "No")],
        evidence_required=True,
    )
    result = decide(request, Fake())
    assert result.abstained
    assert result.choice_id is None
    assert result.abstain_reason == "insufficient_evidence"


def test_wilson_and_summary():
    lo, hi = wilson(9, 10)
    assert 0.55 < lo < 0.9 and hi > 0.9
    summary = summarize([
        {"slice": "classification", "gold": 1, "choice": 1, "abstained": False},
        {"slice": "classification", "gold": 0, "choice": 1, "abstained": False},
    ])
    assert summary["n"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["slices"]["classification"]["n"] == 2


def test_conformal_threshold_is_bounded():
    scores = [0.99, 0.9, 0.8, 0.7, 0.6]
    threshold = conformal_threshold(scores, [True] * 5, coverage=0.8)
    assert 0.0 <= threshold <= 1.0
    assert threshold >= 0.2


def test_gate_passes_on_all_conditions():
    slices = {
        name: {"n": 10, "accuracy": 0.95, "wilson95": [0.9, 1.0]}
        for name in JEV_REFERENCE
    }
    summary = {"n": 80, "accuracy": 0.95, "wilson95": [0.9, 1.0], "slices": slices}
    verdict = gate(
        summary, selector_recall=0.95, evidence_precision=0.9,
        context={"niah_32k": 0.96, "multi_hop": 0.85, "worst_position_gap_pp": 2.0},
        m1={"p95_ms": 4000, "peak_rss_gib": 4.5, "swap_delta_mb": 0},
        regression={"accuracy": 0.89},
    )
    assert verdict["passed"], verdict["failures"]


def test_gate_fails_without_measurements():
    slices = {name: {"n": 1, "accuracy": 0.5, "wilson95": [0.1, 0.9]} for name in JEV_REFERENCE}
    summary = {"n": 8, "accuracy": 0.5, "wilson95": [0.1, 0.9], "slices": slices}
    verdict = gate(summary)
    assert not verdict["passed"]
    assert len(verdict["failures"]) >= 4


def test_cli_has_decision_surface():
    from azurite.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "decide", "--context", "x", "--reader", "r.pt", "--alt", "yes", "--alt", "no",
    ])
    assert args.func.__name__ == "command_decide"
    assert args.alt == ["yes", "no"]
