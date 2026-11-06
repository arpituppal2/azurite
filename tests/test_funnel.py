"""Tests for the retrieve-then-rerank funnel (K-cap lift).

The funnel lets the cross-encoder decide tasks with K > CROSS_ENCODER_MAX_K:
a dual scorer retrieves the top-M candidates per example and the cross head
reranks those M only. The invariants that would silently corrupt such a
comparison — a subset pack that differs from the plain pack on identical
input, a dropped gold that is not counted as a miss, a prediction mapped to
the wrong bank index — are asserted here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from azurite.decisions import (  # noqa: E402
    CandidateBank,
    Example,
    TaskSpec,
    pack_cross,
    pack_cross_subset,
)
from azurite.tokenizers import SEP_ID, ByteTokenizer  # noqa: E402
from transfer_eval import funnel_map_predictions  # noqa: E402

TOK = ByteTokenizer()

SPEC = TaskSpec(
    name="synthetic",
    split_rel="default/test",
    text_fields=("text",),
    label_field="label",
    instruction="Choose one.",
)


def bank(labels: list[str]) -> CandidateBank:
    return CandidateBank(
        task="synthetic",
        labels=labels,
        raw_ids=list(range(len(labels))),
    ).tokenize(TOK)


def examples(n: int, k: int, *, seed: int = 0) -> list[Example]:
    gen = torch.Generator().manual_seed(seed)
    golds = torch.randint(0, k, (n,), generator=gen).tolist()
    return [Example(index=i, context=f"example number {i} with some words", gold=g)
            for i, g in enumerate(golds)]


def test_subset_with_full_keep_matches_plain_cross_exactly() -> None:
    # M == K with identity keep-lists must pack byte-identical rows to
    # pack_cross with shuffle off; otherwise the funnel is a different task.
    b = bank(["alpha", "be", "gamma", "d"])
    rows = examples(8, 4, seed=5)
    keep = [list(range(4)) for _ in rows]
    ref, _ = pack_cross(TOK, SPEC, b, rows, max_seq=256, shuffle=False, seed=0)
    sub, stats = pack_cross_subset(TOK, SPEC, b, rows, keep, max_seq=256)
    assert stats["path"] == "cross-funnel"
    assert stats["m"] == 4
    assert stats["funnel_miss"] == 0
    assert stats["packed_indices"] == [ex.index for ex in rows]
    assert torch.equal(sub.ids, ref.ids)
    assert torch.equal(sub.pad_mask, ref.pad_mask)
    assert torch.equal(sub.spans, ref.spans)
    assert torch.equal(sub.gold, ref.gold)


def test_subset_gold_resolves_to_its_own_candidate_span() -> None:
    b = bank(["alpha", "beta", "gamma", "delta", "epsilon"])
    rows = examples(10, 5, seed=9)
    keep = [[(i + j) % 5 for j in range(3)] for i in range(len(rows))]
    for row, ex in zip(keep, rows):
        if ex.gold not in row:
            row[0] = ex.gold
    batch, stats = pack_cross_subset(TOK, SPEC, b, rows, keep, max_seq=256)
    assert stats["funnel_miss"] == 0
    for r, ex in enumerate(rows):
        span = batch.spans[r, int(batch.gold[r])].tolist()
        assert batch.ids[r, span[0]:span[1]].tolist() == b.token_ids[ex.gold]


def test_unretrieved_gold_is_dropped_and_counted_not_remapped() -> None:
    b = bank(["alpha", "beta", "gamma"])
    rows = [Example(index=0, context="one", gold=2),
            Example(index=1, context="two", gold=0)]
    batch, stats = pack_cross_subset(

        TOK, SPEC, b, rows, [[0, 1], [0, 1]], max_seq=256)
    assert stats["packed"] == 1
    assert stats["funnel_miss"] == 1
    assert stats["packed_indices"] == [1]
    assert batch.gold.tolist() == [0]


def test_subset_rejects_malformed_keep_lists() -> None:
    b = bank(["a", "b", "c"])
    rows = examples(3, 3)
    with pytest.raises(ValueError, match="ragged"):
        pack_cross_subset(TOK, SPEC, b, rows, [[0, 1], [0], [1, 2]], max_seq=64)
    with pytest.raises(ValueError, match="duplicate"):
        pack_cross_subset(TOK, SPEC, b, rows, [[0, 0]] * 3, max_seq=64)
    with pytest.raises(ValueError, match="outside the bank"):
        pack_cross_subset(TOK, SPEC, b, rows, [[0, 3]] * 3, max_seq=64)
    with pytest.raises(ValueError, match="keep-lists"):
        pack_cross_subset(TOK, SPEC, b, rows, [[0, 1]], max_seq=64)
    with pytest.raises(ValueError, match=">= 2"):
        pack_cross_subset(TOK, SPEC, b, rows, [[0]] * 3, max_seq=64)
    with pytest.raises(ValueError, match="exceeds the cross-encoder cap"):
        pack_cross_subset(TOK, SPEC, bank([f"l{i}" for i in range(20)]), examples(2, 20),
                          [list(range(17))] * 2, max_seq=1024)


def test_subset_supports_k_beyond_the_plain_cross_cap() -> None:
    b = bank([f"intent_{i}" for i in range(77)])
    rows = examples(6, 77, seed=4)
    keep = [[(i + j) % 77 for j in range(8)] for i in range(len(rows))]
    for k, ex in zip(keep, rows):
        if ex.gold not in k:
            k[0] = ex.gold
    batch, stats = pack_cross_subset(TOK, SPEC, b, rows, keep, max_seq=1024)
    assert stats["k"] == 77 and stats["m"] == 8
    assert stats["packed"] == 6
    assert batch.spans.shape == (6, 8, 2)


def test_map_predictions_returns_bank_indices_and_none_for_misses() -> None:
    # Each prediction is resolved through the keep-list of its OWN row
    # (selected via packed_indices), not the positionally-zipped keep-list.
    keep_full = [[2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 1, 0]]
    packed_indices = [0, 2, 3]
    subset_pred = [0, 2, 1]
    mapped = funnel_map_predictions(subset_pred, keep_full, packed_indices, 4)
    assert mapped == [2, None, 2, 1]


def test_map_predictions_empty_pack_is_all_miss() -> None:
    assert funnel_map_predictions([], [[0, 1]], [], 1) == [None]


def test_packed_indices_are_positions_not_global_ids() -> None:
    # Regression: stratified draws carry non-contiguous dataset indices, so
    # packed_indices must be positions into the rows/keep lists, or the
    # end-to-end mapping indexes out of range.
    b = bank(["alpha", "beta", "gamma"])
    rows = [Example(index=5, context="a", gold=0),
            Example(index=7, context="b", gold=2),
            Example(index=9, context="c", gold=1)]
    keep = [[0, 2], [0, 1], [1, 2]]
    batch, stats = pack_cross_subset(TOK, SPEC, b, rows, keep, max_seq=256)
    assert stats["packed"] == 2
    assert stats["funnel_miss"] == 1
    assert stats["packed_indices"] == [0, 2]
    mapped = funnel_map_predictions(batch.gold.tolist(), keep,
                                    stats["packed_indices"], len(rows))
    assert mapped == [0, None, 1]
