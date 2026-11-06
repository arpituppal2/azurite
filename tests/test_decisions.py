"""Tests for the decision interface.

The invariants here are the ones whose failure is invisible in an accuracy
number: a span that includes a separator still produces a plausible logit, and a
gold index pointing at the wrong candidate produces a model that trains happily
toward the wrong answer. The earlier harness in this project reported 0.0000 on
every task for exactly that class of reason, so these are asserted rather than
eyeballed.
"""

from __future__ import annotations

import pytest
import torch

from azurite.decisions import (
    CROSS_ENCODER_MAX_K,
    CandidateBank,
    Example,
    TaskSpec,
    pack_cross,
    pack_cross_subset,
    pack_dual,
)
from azurite.tokenizers import CLS_ID, PAD_ID, SEP_ID, ByteTokenizer

TOK = ByteTokenizer()

SPEC = TaskSpec(
    name="synthetic",
    split_rel="default/test",
    text_fields=("text",),
    label_field="label",
    instruction="Choose one.",
)


def bank(labels: list[str], raw_ids: list[object] | None = None) -> CandidateBank:
    return CandidateBank(
        task="synthetic",
        labels=labels,
        raw_ids=raw_ids if raw_ids is not None else list(range(len(labels))),
    ).tokenize(TOK)


def examples(n: int, k: int, *, seed: int = 0) -> list[Example]:
    gen = torch.Generator().manual_seed(seed)
    golds = torch.randint(0, k, (n,), generator=gen).tolist()
    return [Example(index=i, context=f"example number {i} with some words", gold=g) for i, g in enumerate(golds)]


# --------------------------------------------------------------------------
# Label resolution
# --------------------------------------------------------------------------


def test_resolve_maps_raw_labels_to_positions() -> None:
    b = bank(["World", "Sports"], raw_ids=[10, 20])
    assert b.resolve(10) == 0
    assert b.resolve(20) == 1


def test_resolve_raises_on_unknown_label() -> None:
    b = bank(["World"], raw_ids=[0])
    with pytest.raises(KeyError):
        b.resolve(99)


def test_resolve_works_for_string_labels() -> None:
    b = bank(["English", "German"], raw_ids=["English", "German"])
    assert b.resolve("German") == 1


# --------------------------------------------------------------------------
# Cross-encoder packing
# --------------------------------------------------------------------------


def test_every_span_covers_exactly_its_candidate_tokens() -> None:
    b = bank(["alpha", "beta", "gamma"])
    batch, stats = pack_cross(
        TOK, SPEC, b, examples(6, 3), max_seq=256, shuffle=False, seed=0
    )
    assert stats["packed"] == 6
    for row in range(batch.ids.shape[0]):
        for k in range(len(b)):
            start, end = batch.spans[row, k].tolist()
            assert 0 <= start < end <= batch.ids.shape[1]
            got = batch.ids[row, start:end].tolist()
            assert got == b.token_ids[k], f"row {row} cand {k}: {got} != {b.token_ids[k]}"
            # the token after the span must be the separator, not more text
            assert batch.ids[row, end].item() == SEP_ID


def test_spans_never_include_padding() -> None:
    b = bank(["alpha", "beta"])
    batch, _ = pack_cross(TOK, SPEC, b, examples(4, 2), max_seq=128, seed=0)
    for row in range(batch.ids.shape[0]):
        real = int(batch.pad_mask[row].sum())
        for k in range(len(b)):
            _, end = batch.spans[row, k].tolist()
            assert end <= real, f"span {k} ends at {end} past {real} real tokens"


def test_gold_points_at_the_span_containing_the_gold_candidate() -> None:
    b = bank(["alpha", "beta", "gamma"])
    rows = examples(8, 3, seed=3)
    batch, _ = pack_cross(TOK, SPEC, b, rows, max_seq=256, shuffle=True, seed=7)
    for row, ex in enumerate(rows):
        gold_span = batch.spans[row, int(batch.gold[row])].tolist()
        span_tokens = batch.ids[row, gold_span[0] : gold_span[1]].tolist()
        assert span_tokens == b.token_ids[ex.gold], (
            f"row {row}: gold points at {span_tokens}, expected {b.token_ids[ex.gold]}"
        )


def test_shuffle_is_deterministic_for_a_seed() -> None:
    # Labels of *different lengths*: with equal-length labels the spans land in
    # the same places whatever the order, so spans are a useless probe and the
    # permutation only shows up in `gold`.
    b = bank(["alpha", "be", "gamma", "d"])
    rows = examples(5, 4, seed=1)
    first, _ = pack_cross(TOK, SPEC, b, rows, max_seq=256, shuffle=True, seed=11)
    second, _ = pack_cross(TOK, SPEC, b, rows, max_seq=256, shuffle=True, seed=11)
    assert torch.equal(first.gold, second.gold)
    assert torch.equal(first.spans, second.spans)
    third, _ = pack_cross(TOK, SPEC, b, rows, max_seq=256, shuffle=True, seed=12)
    assert not torch.equal(first.gold, third.gold), "seed did not change the order"
    # unshuffled, the gold position is the gold label index
    plain, _ = pack_cross(TOK, SPEC, b, rows, max_seq=256, shuffle=False)
    assert plain.gold.tolist() == [ex.gold for ex in rows]


def test_unpackable_examples_are_counted_not_substituted() -> None:
    # 4 candidates keeps the fixed overhead (instruction + specials) small
    # enough that one row genuinely fits while the other genuinely does not.
    b = bank([f"cand{i}" for i in range(4)])
    rows = [
        Example(index=0, context="x" * 400, gold=0),  # cannot fit at max_seq=64
        Example(index=1, context="short", gold=1),  # fits
    ]
    batch, stats = pack_cross(TOK, SPEC, b, rows, max_seq=64, shuffle=False, seed=0)
    assert stats["unpackable"] == 1
    assert stats["packed"] == 1
    # the surviving row is the short one; unshuffled, the gold position is the
    # gold label index, which is what makes this check meaningful
    assert batch.gold.tolist() == [b.resolve(1)]


def test_all_unpackable_raises_rather_than_training_on_nothing() -> None:
    b = bank([f"cand{i}" for i in range(4)])
    rows = [Example(index=0, context="x" * 400, gold=0)]
    with pytest.raises(ValueError, match="no packable examples"):
        pack_cross(TOK, SPEC, b, rows, max_seq=64, seed=0)


def test_cross_encoder_refuses_too_many_candidates() -> None:
    b = bank([f"label_{i}" for i in range(CROSS_ENCODER_MAX_K + 1)])
    with pytest.raises(ValueError, match="dual encoder|pack_dual"):
        pack_cross(TOK, SPEC, b, examples(2, CROSS_ENCODER_MAX_K + 1), max_seq=1024)


def test_pair_fields_get_separate_context_shares() -> None:
    pair_spec = TaskSpec(
        name="synthetic_pair",
        split_rel="default/test",
        text_fields=("premise", "hypothesis"),
        label_field="label",
        instruction="Choose one.",
    )
    b = bank(["alpha", "beta"])
    premise = "premise " * 40
    hypothesis = "hypothesis words"
    rows = [
        Example(
            index=0,
            context=premise + "\n" + hypothesis,
            gold=1,
            parts=(premise, hypothesis),
        )
    ]
    batch, stats = pack_cross(
        TOK, pair_spec, b, rows, max_seq=64, shuffle=False, seed=0
    )
    assert stats["field_aware"] is True
    assert stats["context_budget"] > 32
    assert stats["field_budget"] == (stats["context_budget"] - 1) // 2
    real = batch.ids[0][batch.pad_mask[0]].tolist()
    seps = [i for i, token in enumerate(real) if token == SEP_ID]
    hyp = TOK.encode(hypothesis)
    assert hyp == real[seps[1] + 1 : seps[1] + 1 + len(hyp)]
    assert len(real) <= 64


def test_single_field_packing_keeps_the_half_window_cap() -> None:
    b = bank(["alpha", "beta"])
    _, stats = pack_cross(TOK, SPEC, b, examples(2, 2), max_seq=256, seed=0)
    assert stats["field_aware"] is False
    assert stats["context_budget"] == 128
    assert stats["field_budget"] is None


def test_context_is_truncated_not_the_candidates() -> None:
    b = bank(["alpha", "beta"])
    long_ctx = "word " * 400
    batch, stats = pack_cross(
        TOK,
        SPEC,
        b,
        [Example(index=0, context=long_ctx, gold=1)],
        max_seq=256,
        shuffle=False,
        seed=0,
    )
    assert stats["packed"] == 1
    # both candidates survive in full
    for k in range(len(b)):
        start, end = batch.spans[0, k].tolist()
        assert batch.ids[0, start:end].tolist() == b.token_ids[k]


# --------------------------------------------------------------------------
# Dual-encoder packing
# --------------------------------------------------------------------------


def test_dual_batch_shapes_and_gold() -> None:
    b = bank([f"label_{i}" for i in range(77)])
    b.tokenize(TOK)
    rows = examples(12, 77, seed=2)
    batch, stats = pack_dual(TOK, SPEC, b, rows, max_seq=64)
    assert stats["path"] == "dual"
    assert batch.context_ids.shape[0] == 12
    assert batch.context_mask.dtype == torch.bool
    assert batch.gold.tolist() == [ex.gold for ex in rows]
    # padding is masked out
    for row in range(batch.context_ids.shape[0]):
        real = int(batch.context_mask[row].sum())
        assert (batch.context_ids[row, real:] == PAD_ID).all()


def test_dual_supports_large_k_without_truncating_the_candidate_set() -> None:
    b = bank([f"intent_{i}" for i in range(150)])
    b.tokenize(TOK)
    batch, stats = pack_dual(TOK, SPEC, b, examples(3, 150), max_seq=64)
    assert len(b) == 150
    assert stats["packed"] == 3



def test_cls_and_sep_frame_the_context() -> None:
    b = bank(["a", "b"])
    batch, _ = pack_dual(TOK, SPEC, b, [Example(0, "hello", 0)], max_seq=64)
    ids = batch.context_ids[0][batch.context_mask[0]].tolist()
    assert ids[0] == CLS_ID
    assert ids[-1] == SEP_ID
    assert ids[1:-1] == TOK.encode("hello")


def test_field_aware_dual_leaves_fitting_pairs_byte_identical() -> None:
    pair_spec = TaskSpec(
        name="synthetic_pair",
        split_rel="default/test",
        text_fields=("premise", "hypothesis"),
        label_field="label",
        instruction="Choose one.",
    )
    b = bank(["alpha", "beta"])
    premise, hypothesis = "short premise", "short hypothesis"
    rows = [Example(0, premise + "\n" + hypothesis, 1, (premise, hypothesis))]
    plain, plain_stats = pack_dual(TOK, pair_spec, b, rows, max_seq=64)
    aware, aware_stats = pack_dual(
        TOK, pair_spec, b, rows, max_seq=64, field_aware=True
    )
    assert torch.equal(plain.context_ids, aware.context_ids)
    assert torch.equal(plain.context_mask, aware.context_mask)
    assert plain_stats["split_rows"] == 0
    assert aware_stats["split_rows"] == 0
    assert aware_stats["field_aware"] is True


def test_field_aware_dual_keeps_the_hypothesis_when_the_pair_overflows() -> None:
    """The old flat packer truncates from the rear, so a long premise can
    delete the hypothesis entirely. A split row must always contain it."""
    pair_spec = TaskSpec(
        name="synthetic_pair",
        split_rel="default/test",
        text_fields=("premise", "hypothesis"),
        label_field="label",
        instruction="Choose one.",
    )
    b = bank(["alpha", "beta"])
    premise = "premise " * 40
    hypothesis = "distinctive hypothesis words"
    rows = [Example(0, premise + "\n" + hypothesis, 1, (premise, hypothesis))]
    plain, _ = pack_dual(TOK, pair_spec, b, rows, max_seq=64)
    aware, stats = pack_dual(
        TOK, pair_spec, b, rows, max_seq=64, field_aware=True
    )
    hyp = TOK.encode(hypothesis)
    plain_ids = plain.context_ids[0][plain.context_mask[0]].tolist()
    assert not any(
        plain_ids[i : i + len(hyp)] == hyp for i in range(len(plain_ids))
    ), "the control unexpectedly retained the hypothesis"
    ids = aware.context_ids[0][aware.context_mask[0]].tolist()
    assert ids[0] == CLS_ID and ids[-1] == SEP_ID
    assert any(ids[i : i + len(hyp)] == hyp for i in range(len(ids)))
    assert stats["split_rows"] == 1
    assert stats["fields_dropped"] == 0
