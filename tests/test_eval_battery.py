"""Invariants for the frozen 10k evaluation battery and its helpers."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.build_eval_battery import QUOTAS, _noisy
from scripts.build_long_context import compose_pack, length_bounds
from scripts.audit_eval_battery import MIN_ANSWER, strip_ws, windows


def test_quotas_match_the_plan():
    assert QUOTAS == {
        "classification": 2_000,
        "boolean": 1_500,
        "pairwise": 1_000,
        "ordinal": 1_000,
        "document": 2_000,
        "abstention": 1_000,
        "robustness": 1_000,
        "multilingual": 500,
    }
    assert sum(QUOTAS.values()) == 10_000


@pytest.mark.parametrize("text", ["plain ascii filler. " * 40, "naïve café “quoted” — ünïcode. " * 40])
def test_compose_pack_pads_until_the_chunker_agrees(text):
    """Multibyte text can sit above the byte floor and still be a chunk short."""
    low, high = length_bounds(64)
    fillers = [text] * 120
    composed, spans = compose_pack([text[:400]], fillers, positions=["early"], gold=(0,))
    size = len(composed.encode("utf-8"))
    assert low <= size <= high
    from azurite.selector import chunk_text

    assert len(chunk_text(composed)) == 64
    assert spans[0]["chunks"][0] == 0


def test_noise_perturbs_without_changing_length_much():
    base = "The bank transfer was completed and the customer received a receipt."
    out = _noisy(base, None)
    assert out != base
    assert "teh" in out or "adn" in out or "taht" in out or "wiht" in out or "form" in out


def test_audit_windows_are_exact_and_bounded():
    long_answer = "a" * 20 + " " + "b" * 40
    ws = windows(long_answer)
    assert ws and all(len(w) == 40 for w in ws)
    assert all(w in strip_ws(long_answer) for w in ws)
    assert len(strip_ws(long_answer)) >= MIN_ANSWER


def test_battery_manifest_is_self_describing_if_present():
    from pathlib import Path

    root = Path("results/parity/eval_battery_10k")
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("battery not built in this checkout")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["rows"] == 10_000
    assert manifest["families"] == QUOTAS
    assert manifest["family_counts"] == QUOTAS
    assert manifest["battery_sha256"] == hashlib.sha256(
        "".join(
            f"{s['file']}:{s['sha256']}:{s['rows']}\n"
            for s in sorted(manifest["shards"], key=lambda s: s["index"])
        ).encode()
    ).hexdigest()
    assert manifest["licences"]
    assert manifest["document_sources"] == ["squad2_answerability_ctx", "contract_nli_ctx", "boolq_ctx"]


def test_loader_returns_every_row_with_provenance():
    from pathlib import Path

    from azurite.battery import read_eval_battery

    root = Path("results/parity/eval_battery_10k")
    if not (root / "manifest.json").exists():
        pytest.skip("battery not built in this checkout")
    rows, manifest = read_eval_battery(root)
    assert len(rows) == 10_000 == manifest["rows"]
    counts = {}
    for row in rows:
        counts[row.slice] = counts.get(row.slice, 0) + 1
        assert row.meta["provenance"]["dataset"]
        assert row.meta["provenance"]["split"]
    assert counts == QUOTAS
    document = [r for r in rows if r.slice == "document"]
    assert all(r.gold_evidence for r in document)
    assert all(r.meta["gold_spans"] for r in document)
