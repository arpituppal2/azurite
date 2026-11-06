"""Regression tests for pack-to-pack evidence contamination.

The old audit only scanned pack evidence against the pretraining stream. A
training pack could therefore contain the exact evidence of a dev/locked pack
and still be reported clean. These tests inject that case and its negative
control, so the detector is proven before a real pack build relies on it.
"""

from __future__ import annotations

from azurite.pack_audit import (
    audit_pack,
    audit_packs,
    build_reference_windows,
    filter_packs,
    pack_evidence_windows,
)
from azurite.tokenizers import ByteTokenizer


TOK = ByteTokenizer()


def pack(pack_id: str, text: str, *, extra: str = "") -> dict:
    context = extra + text
    start = len(extra.encode("utf-8"))
    return {
        "id": pack_id,
        "context": context,
        "gold_spans": [{
            "start": start,
            "end": start + len(text.encode("utf-8")),
            "chunks": [0],
        }],
    }


def test_contaminated_pack_is_rejected_and_clean_pack_accepted() -> None:
    shared = "shared evidence sentence about a card arrival dispute"
    reference = [pack("dev-1", shared), pack("locked-1", "unrelated locked evidence")]
    windows = build_reference_windows(reference, TOK)
    assert windows

    dirty = pack("train-dirty", shared)
    clean = pack("train-clean", "a completely different sentence with no overlap")
    dirty_result = audit_pack(dirty, windows, TOK)
    clean_result = audit_pack(clean, windows, TOK)
    assert not dirty_result.accepted
    assert dirty_result.matched_windows > 0
    assert "overlaps" in (dirty_result.reason or "")
    assert clean_result.accepted
    assert clean_result.matched_windows == 0


def test_distributed_evidence_is_audited_across_both_spans() -> None:
    first = "first half evidence sentence with enough tokens"
    second = "second half evidence sentence with enough tokens"
    reference = [pack("dev-distributed", first), pack("dev-second", second)]
    windows = build_reference_windows(reference, TOK)
    target = {
        "id": "train-distributed",
        "context": first + "\n\n" + second,
        "gold_spans": [
            {"start": 0, "end": len(first.encode()), "chunks": [0]},
            {"start": len(first.encode()) + 2,
             "end": len(first.encode()) + 2 + len(second.encode()), "chunks": [1]},
        ],
    }
    assert len(pack_evidence_windows(target, TOK)) >= 2
    result = audit_pack(target, windows, TOK)
    assert not result.accepted
    assert result.matched_windows >= 2


def test_unauditable_pack_is_rejected_not_called_clean() -> None:
    result = audit_pack({"id": "empty", "context": "", "gold_spans": []}, set(), TOK)
    assert not result.accepted
    assert not result.auditable
    assert "unauditable" in (result.reason or "")


def test_filter_packs_returns_rejected_set_and_failing_gate() -> None:
    shared = "shared evidence sentence about a card arrival dispute"
    reference = [pack("dev-1", shared)]
    dirty = pack("train-dirty", shared)
    clean = pack("train-clean", "independent clean evidence sentence")
    accepted, rejected, report = filter_packs([dirty, clean], reference, TOK)
    assert [p["id"] for p in accepted] == ["train-clean"]
    assert [p["id"] for p in rejected] == ["train-dirty"]
    assert report["gate"]["state"] == "fail"
    assert report["rejected_packs"] == 1
    assert report["unauditable_packs"] == 0
    assert len(report["results"]) == 2


def test_filter_packs_empty_reference_fails_closed() -> None:
    accepted, rejected, report = filter_packs(
        [pack("train-clean", "independent clean evidence sentence")], [], TOK
    )
    assert accepted == []
    assert rejected
    assert report["reference_windows"] == 0
    assert report["gate"]["state"] == "fail"


def test_audit_packs_preserves_one_row_per_pack() -> None:
    shared = "shared evidence sentence about a card arrival dispute"
    reference = build_reference_windows([pack("dev-1", shared)], TOK)
    rows = audit_packs(
        [pack("p1", shared), pack("p2", "another clean sentence for the control")],
        reference,
        TOK,
    )
    assert [r.pack_id for r in rows] == ["p1", "p2"]
    assert rows[0].matched_windows > 0
    assert rows[1].accepted
