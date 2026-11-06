"""Behavioral tests for the task-7 decision tuner.

These exercise the controls the old script lacked: candidate dropout with a
protected gold, paraphrase and hard-negative construction, a disjoint
train-split validation carve, reader freeze/unfreeze, per-row validation records,
and resumable state. The tiny model keeps the tests CPU-only.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from azurite.config import EncoderConfig
from azurite.decision_tuning import (
    activate_reader,
    build_optimizer,
    hard_negative_indices,
    label_variants,
    load_train_validation,
    load_training_state,
    paraphrase_labels,
    save_head_npz,
    save_training_state,
    select_candidate_indices,
    set_reader_trainable,
    split_train_validation,
    train_batch,
    validation_gate,
)
from azurite.decisions import CandidateBank, Example, TaskSpec
from azurite.model import Azurite, AzuriteDual
from azurite.tokenizers import ByteTokenizer


TOK = ByteTokenizer()
SPEC = TaskSpec(
    name="synthetic",
    split_rel="default/test",
    text_fields=("text",),
    label_field="label",
    instruction="Choose one.",
)


def tiny_dual() -> AzuriteDual:
    cfg = EncoderConfig(
        d_model=32, n_layer=1, n_head=4, d_ff=64,
        vocab_size=260, max_seq=64,
    )
    return AzuriteDual(Azurite(cfg))


def bank(labels: list[str]) -> CandidateBank:
    return CandidateBank("synthetic", labels, list(range(len(labels)))).tokenize(TOK)


def examples(n: int = 8, *, labels: int = 4, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    return [
        Example(
            index=i,
            context=f"example {i} mentions card and account",
            gold=rng.randrange(labels),
        )
        for i in range(n)
    ]


def test_label_paraphrases_keep_order_and_are_deterministic() -> None:
    labels = ["card_arrival", "card fee", "loan"]
    assert label_variants(labels[0])[0] == "card_arrival"
    assert "card arrival" in label_variants(labels[0])
    assert paraphrase_labels(labels, random.Random(0), probability=0.0) == labels
    first = paraphrase_labels(labels, random.Random(7), probability=1.0)
    second = paraphrase_labels(labels, random.Random(7), probability=1.0)
    assert first == second
    assert len(first) == len(labels)
    for original, variant in zip(labels, first):
        assert variant in label_variants(original)


def test_candidate_dropout_never_drops_gold_and_can_drop_others() -> None:
    labels = ["alpha", "beta", "gamma", "delta", "epsilon"]
    for seed in range(25):
        kept = select_candidate_indices(
            "alpha", labels, 3, random.Random(seed),
            dropout=1.0, hard_negatives=0, max_candidates=5,
        )
        assert 3 in kept
        assert len(kept) >= 2
    kept = select_candidate_indices(
        "alpha", labels, 0, random.Random(1),
        dropout=0.0, hard_negatives=0, max_candidates=5,
    )
    assert kept == list(range(5))


def test_hard_negatives_are_lexically_relevant_and_exclude_gold() -> None:
    labels = ["card arrival", "card fee", "loan repayment", "atm support"]
    hard = hard_negative_indices("my card card fee", labels, 3, k=2)
    assert 3 not in hard
    assert hard
    assert hard[0] in (0, 1)


def test_candidate_selection_is_seeded_and_keeps_hard_negatives() -> None:
    labels = ["card arrival", "card fee", "loan repayment", "atm support"]
    a = select_candidate_indices(
        "card fee", labels, 0, random.Random(3),
        dropout=1.0, hard_negatives=2, max_candidates=2,
    )
    b = select_candidate_indices(
        "card fee", labels, 0, random.Random(3),
        dropout=1.0, hard_negatives=2, max_candidates=2,
    )
    assert a == b
    assert 0 in a and 1 in a


def test_train_validation_split_is_disjoint() -> None:
    rows = examples(20)
    train, validation = split_train_validation(rows, fraction=0.25, seed=4)
    assert train and validation
    assert len(train) + len(validation) == len(rows)
    assert {ex.index for ex in train}.isdisjoint({ex.index for ex in validation})
    assert len(validation) == 5


def test_load_train_validation_uses_only_train_split(monkeypatch) -> None:
    calls: list[str] = []
    rows = examples(10)

    def fake_load(spec, candidate, *, split="test", limit=None, offset=0):
        calls.append(split)
        return rows

    import azurite.decision_tuning as tuning

    monkeypatch.setattr(tuning, "load_examples", fake_load)
    train, validation = load_train_validation(SPEC, bank(["a", "b"]), fraction=0.2, seed=0)
    assert calls == ["train"]
    assert train and validation


def test_reader_freeze_and_warmup_unfreeze() -> None:
    dual = tiny_dual()
    opt = build_optimizer(dual, reader_mode="warmup", head_lr=1e-3, reader_lr=3e-5)
    assert [g["name"] for g in opt.param_groups] == ["head", "reader"]
    assert all(not p.requires_grad for p in dual.encoder.parameters())
    assert opt.param_groups[1]["lr"] == 0.0
    changed = activate_reader(
        dual, opt, step=5, freeze_steps=5, reader_mode="warmup", reader_lr=3e-5
    )
    assert changed > 0
    assert all(p.requires_grad for p in dual.encoder.parameters())
    assert opt.param_groups[1]["lr"] == 3e-5


def test_freeze_has_no_reader_group_and_unfreeze_has_gradients() -> None:
    frozen = tiny_dual()
    frozen_opt = build_optimizer(frozen, reader_mode="freeze")
    assert [g["name"] for g in frozen_opt.param_groups] == ["head"]
    assert all(not p.requires_grad for p in frozen.encoder.parameters())

    unfrozen = tiny_dual()
    unfrozen_opt = build_optimizer(unfrozen, reader_mode="unfreeze")
    assert [g["name"] for g in unfrozen_opt.param_groups] == ["head", "reader"]
    assert all(p.requires_grad for p in unfrozen.encoder.parameters())
    assert set_reader_trainable(unfrozen, False) == len(list(unfrozen.encoder.parameters()))


def test_training_batch_backpropagates_with_candidate_dropout() -> None:
    dual = tiny_dual()
    dual.train()
    rows = [
        Example(0, "card arrival fee", 0),
        Example(1, "loan repayment", 2),
        Example(2, "atm card support", 3),
    ]
    loss, stats = train_batch(
        dual, TOK, SPEC, bank(["card arrival", "card fee", "loan", "atm"]),
        rows, rng=random.Random(1), candidate_dropout=0.5,
        hard_negatives=1, max_candidates=3, paraphrase_probability=0.5,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["kept_mean"] >= 2
    assert stats["tokens"] > 0
    assert dual.proj.weight.grad is not None
    assert torch.isfinite(dual.proj.weight.grad).all()


def test_validation_gate_requires_every_family_and_evidence() -> None:
    records = [
        {"family": "classification", "gold": 0, "choice": 0, "bm25_choice": 1},
        {"family": "classification", "gold": 0, "choice": 0, "bm25_choice": 1},
        {"family": "classification", "gold": 1, "choice": 1, "bm25_choice": 0},
        {"family": "classification", "gold": 1, "choice": 0, "bm25_choice": 0},
        {"family": "boolean", "gold": 0, "choice": 0, "bm25_choice": 0},
        {"family": "boolean", "gold": 0, "choice": 0, "bm25_choice": 1},
        {"family": "boolean", "gold": 1, "choice": 1, "bm25_choice": 0},
        {"family": "boolean", "gold": 1, "choice": 1, "bm25_choice": 1},
    ]
    result = validation_gate(records, evidence_precision=0.85)
    assert result["passed"], result["failures"]
    assert set(result["families"]) == {"classification", "boolean"}

    failed = validation_gate(records, evidence_precision=None)
    assert not failed["passed"]
    assert any("evidence" in failure for failure in failed["failures"])

    weak = list(records)
    # Make the BM25 control perfect on this family while the model drops one
    # row: the family must fail even though it still beats majority.
    weak[4] = {"family": "boolean", "gold": 0, "choice": 0, "bm25_choice": 0}
    weak[5] = {"family": "boolean", "gold": 0, "choice": 0, "bm25_choice": 0}
    weak[6] = {"family": "boolean", "gold": 1, "choice": 1, "bm25_choice": 1}
    weak[7] = {"family": "boolean", "gold": 1, "choice": 0, "bm25_choice": 1}
    failed = validation_gate(weak, evidence_precision=0.85)
    assert not failed["passed"]
    assert any("BM25" in failure for failure in failed["failures"])


def test_state_round_trip_preserves_weights_optimizer_and_rng(tmp_path) -> None:
    dual = tiny_dual()
    opt = build_optimizer(dual, reader_mode="freeze")
    rng = random.Random(11)
    state = tmp_path / "decision.state.pt"
    save_training_state(
        state, dual, opt, step=3, tokens_seen=123, args={"mode": "test"}, rng=rng
    )
    fresh = tiny_dual()
    fresh_opt = build_optimizer(fresh, reader_mode="freeze")
    payload, new_rng = load_training_state(state, fresh, fresh_opt)
    assert payload["step"] == 3
    assert payload["tokens_seen"] == 123
    assert torch.equal(fresh.proj.weight, dual.proj.weight)
    assert new_rng.random() == rng.random()


def test_head_npz_round_trips(tmp_path) -> None:
    dual = tiny_dual()
    path = save_head_npz(tmp_path / "head.npz", dual)
    data = np.load(path, allow_pickle=False)
    assert data["proj"].shape == (32, 32)
    assert data["log_scale"].shape == (1,)


def test_2d_candidate_mask_only_masks_requested_rows() -> None:
    dual = tiny_dual()
    ids = torch.randint(4, 100, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)
    cand_ids = torch.randint(4, 100, (3, 4))
    cand_mask = torch.ones(3, 4, dtype=torch.bool)
    valid = torch.tensor([[True, True, False], [False, True, True]])
    logits = dual(ids, mask, cand_ids, cand_mask, valid)
    assert torch.isinf(logits[0, 2]) and logits[0, 2] < 0
    assert torch.isinf(logits[1, 0]) and logits[1, 0] < 0
    assert torch.isfinite(logits[0, :2]).all()
    assert torch.isfinite(logits[1, 1:]).all()


def test_script_exposes_task7_controls() -> None:
    from scripts.train_decision import main as _main  # noqa: F401
    import scripts.train_decision as script

    source = script.__doc__ or ""
    assert "candidate dropout" in source
    assert "paraphrase" in source
    assert "hard negatives" in source
    assert "resumable" in source
