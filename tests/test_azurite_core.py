"""Tests for the azurite core: accounting, masking, and tokenizer arms.

These are the properties that, if silently false, would invalidate a scaling
curve rather than produce a visibly wrong number:

  * the analytic parameter count must equal the instantiated count, or
    "matched scale" is a claim rather than a fact
  * padding must not influence real positions, or every rung is trained on a
    different effective input distribution at each batch size
  * invalid candidates must never win, or a 150-option task's accuracy is
    partly a function of the masking bug (v1 shipped exactly this class of bug)
  * both tokenizer arms must round-trip exactly, or the byte arm's sequences and
    the subword arm's sequences are not describing the same text
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import pytest
import torch

from azurite.config import MAX_SEQ, RUNG_DIMS, VOCAB_SIZE, rung
from azurite.model import Azurite, AzuriteDecision, DecisionBatch, RoPE, parameter_counts
from azurite.tokenizers import (
    CLS_ID,
    PAD_ID,
    SEP_ID,
    ByteTokenizer,
    BpeTokenizer,
)

ROOT = Path(__file__).resolve().parent.parent
RUNGS = sorted(RUNG_DIMS)
RUNG_ORDER = ["tiny", "small", "mid", "large"]


def corpus_files(limit: int = 2) -> list[Path]:
    files = sorted(glob.glob(str(ROOT / "data/corpus/*.jsonl.gz")))
    if not files:
        files = sorted(glob.glob(str(ROOT / "data/corpus/*.jsonl")))
    return [Path(f) for f in files[:limit]]


# --------------------------------------------------------------------------
# Parameter accounting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", RUNG_ORDER)
@pytest.mark.parametrize("arm", ["byte", "bpe"])
def test_analytic_counts_match_instantiated(name: str, arm: str) -> None:
    cfg = rung(name, arm)
    model = Azurite(cfg)
    counted = parameter_counts(cfg)
    assert model.num_parameters() == counted["pretrain_total"], (
        f"{name}/{arm}: instantiated {model.num_parameters():,} vs "
        f"analytic {counted['pretrain_total']:,}"
    )
    decision = AzuriteDecision(Azurite(cfg))
    instantiated = sum(p.numel() for p in decision.parameters() if p.requires_grad)
    # The decision head shares the encoder's tied embedding, so the total is the
    # encoder plus the head's own two matrices (the score bias is counted).
    assert instantiated == counted["decision_total"], (
        f"{name}/{arm}: decision {instantiated:,} vs "
        f"analytic {counted['decision_total']:,}"
    )


def test_rungs_increase_in_size() -> None:
    sizes = [parameter_counts(rung(n, "byte"))["pretrain_total"] for n in RUNG_ORDER]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes), sizes


def test_bpe_vocabulary_costs_real_parameters() -> None:
    """The arms must not differ by a rounding error in embedding size: at 32k
    vocabulary the embedding matrix is a real fraction of the smallest rungs,
    and the study compares arms, so that difference is reported, not hidden."""
    byte_cfg = rung("tiny", "byte")
    bpe_cfg = rung("tiny", "bpe")
    delta = (
        parameter_counts(bpe_cfg)["pretrain_total"]
        - parameter_counts(byte_cfg)["pretrain_total"]
    )
    expected = (VOCAB_SIZE["bpe"] - VOCAB_SIZE["byte"]) * byte_cfg.d_model
    assert delta == expected


# --------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------


def test_rope_preserves_norm_and_varies_with_position() -> None:
    rope = RoPE(d_head=32, max_seq=64)
    x = torch.randn(1, 2, 8, 32)
    y = rope.rotate(x)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)
    # Same vector at two positions must rotate differently; a constant result
    # means the table is not being applied.
    a = torch.ones(1, 1, 1, 32)
    two = rope.rotate(a.expand(1, 1, 2, 32))
    assert not torch.allclose(two[0, 0, 0], two[0, 0, 1])


def test_nn_module_apply_still_reaches_every_module() -> None:
    """Guard against the bug this file already caught once: a method named
    `apply` on any submodule shadows `nn.Module.apply`, so `Azurite(cfg)`
    raises instead of initialising, and the failure looks like a crash rather
    than an untrained model."""
    cfg = rung("tiny", "byte")
    seen = set()
    Azurite(cfg).apply(lambda m: seen.add(type(m).__name__))
    assert {"RoPE", "Attention", "SwiGLU", "RMSNorm", "Azurite"} <= seen


def test_rope_table_covers_max_seq_for_both_arms() -> None:
    for arm, seq in MAX_SEQ.items():
        rope = RoPE(d_head=32, max_seq=seq)
        assert rope.cos.shape[0] == seq, (arm, rope.cos.shape)


# --------------------------------------------------------------------------
# Padding invariance
# --------------------------------------------------------------------------


def test_padding_matches_truncation_exactly() -> None:
    """The property that matters: running 20 tokens with 5 of them padded must
    equal running the 15 real tokens alone, at those 15 positions.

    Note what this test deliberately does NOT compare: a full-length run whose
    last 5 slots hold real tokens against a padded run. Those differ for a
    legitimate reason — bidirectional attention lets position 0 read position
    19 when position 19 is real and stops when it is masked — so asserting
    equality there would be asserting that attention is not bidirectional.
    """
    torch.manual_seed(0)
    cfg = rung("tiny", "byte")
    model = Azurite(cfg).eval()
    keep = 15
    ids = torch.randint(5, cfg.vocab_size, (1, 20))
    padded = ids.clone()
    padded[0, keep:] = PAD_ID
    mask = torch.ones_like(padded, dtype=torch.bool)
    mask[0, keep:] = False
    with_pad = model(padded, mask)
    truncated = model(ids[:, :keep], torch.ones_like(ids[:, :keep], dtype=torch.bool))
    assert torch.allclose(with_pad[:, :keep], truncated, atol=1e-4), (
        "padding leaked into real positions"
    )


def test_padded_slots_can_hold_arbitrary_ids() -> None:
    torch.manual_seed(1)
    cfg = rung("tiny", "byte")
    model = Azurite(cfg).eval()
    ids = torch.randint(5, cfg.vocab_size, (2, 16))
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[1, 10:] = False
    first = model(ids, mask)
    other = ids.clone()
    other[1, 10:] = torch.randint(5, cfg.vocab_size, (6,))
    second = model(other, mask)
    assert torch.allclose(first[1, :10], second[1, :10], atol=1e-4)


# --------------------------------------------------------------------------
# Decision head
# --------------------------------------------------------------------------


def make_batch(k: int = 5, invalid: tuple[int, ...] = (3, 4), t: int = 24) -> DecisionBatch:
    ids = torch.randint(5, 100, (1, t))
    pad_mask = torch.ones(1, t, dtype=torch.bool)
    spans = torch.zeros(1, k, 2, dtype=torch.long)
    for i in range(k):
        start = 4 + i * 4
        spans[0, i] = torch.tensor([start, min(start + 3, t)])
    cand_mask = torch.ones(1, k, dtype=torch.bool)
    for i in invalid:
        cand_mask[0, i] = False
    return DecisionBatch(
        ids=ids, pad_mask=pad_mask, spans=spans, cand_mask=cand_mask, gold=torch.tensor([0])
    )


def test_invalid_candidates_get_negative_infinity() -> None:
    torch.manual_seed(2)
    model = AzuriteDecision(Azurite(rung("tiny", "byte"))).eval()
    logits = model(make_batch(invalid=(3, 4)))
    assert torch.isinf(logits[0, 3]) and logits[0, 3] < 0
    assert torch.isinf(logits[0, 4]) and logits[0, 4] < 0
    assert torch.isfinite(logits[0, :3]).all()


def test_invalid_candidates_never_win_across_random_inits() -> None:
    for seed in range(5):
        torch.manual_seed(seed)
        model = AzuriteDecision(Azurite(rung("tiny", "byte"))).eval()
        probs = model.probabilities(make_batch(invalid=(3, 4)))
        assert probs[0, 3] == 0 and probs[0, 4] == 0
        assert int(probs.argmax()) not in (3, 4)
    assert math.isclose(float(model.probabilities(make_batch()).sum()), 1.0, rel_tol=1e-5)


def test_span_pooling_averages_exactly_the_span() -> None:
    model = AzuriteDecision(Azurite(rung("tiny", "byte"))).eval()
    hidden = torch.zeros(1, 6, 8)
    hidden[0, 1:4] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    spans = torch.tensor([[[1, 4], [0, 1]]])
    cand_mask = torch.ones(1, 2, dtype=torch.bool)
    pooled = model._pool_spans(hidden, spans, cand_mask)
    assert torch.allclose(pooled[0, 0], torch.tensor([1.0, 2, 3, 4, 5, 6, 7, 8]))
    assert torch.allclose(pooled[0, 1], torch.zeros(8))


def test_loss_ignores_invalid_candidates() -> None:
    torch.manual_seed(3)
    model = AzuriteDecision(Azurite(rung("tiny", "byte")))
    batch = make_batch(invalid=(3, 4))
    logits = model(batch)
    loss = AzuriteDecision.loss(logits, batch.gold)
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------
# Masked-LM objective
# --------------------------------------------------------------------------


def test_mlm_loss_finite_and_masks_at_least_one_position_per_row() -> None:
    torch.manual_seed(4)
    cfg = rung("tiny", "byte")
    model = Azurite(cfg)
    ids = torch.randint(5, cfg.vocab_size, (4, 24))
    ids[3, 10:] = PAD_ID
    loss, stats = model.mlm_loss(ids, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    assert stats["masked_positions"] > 0
    gene = torch.Generator().manual_seed(0)
    ids_all_pad = torch.full((2, 8), PAD_ID)
    loss_pad, stats_pad = model.mlm_loss(ids_all_pad, generator=gene)
    # A batch with nothing to predict must contribute zero, not NaN. It used to
    # be NaN (cross-entropy over an all-ignore_index target), which destroys
    # every parameter on the backward pass and looks like divergence.
    assert torch.isfinite(loss_pad)
    assert float(loss_pad.detach()) == 0.0
    loss_pad.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert stats_pad["masked_positions"] == 0


# --------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------


def test_byte_tokenizer_round_trips_corpus_lines() -> None:
    tok = ByteTokenizer()
    import gzip

    checked = 0
    for path in corpus_files(limit=1):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                text = rec.get("text", "")[:400]
                ids = tok.encode(text)
                assert all(0 <= i < tok.vocab_size for i in ids)
                assert tok.decode(ids) == text
                checked += 1
                if checked >= 200:
                    break
        break
    assert checked >= 50, f"only checked {checked} lines"


def test_byte_tokenizer_handles_non_ascii_and_controls() -> None:
    tok = ByteTokenizer()
    for text in ["", "café", "日本語のテキスト", "emoji 🌍 ok", "null\x00byte", "\n\t"]:
        assert tok.decode(tok.encode(text)) == text


def test_bpe_tokenizer_trains_on_local_text_and_round_trips(tmp_path: Path) -> None:
    lines = []
    for i in range(400):
        lines.append(
            json.dumps(
                {
                    "text": (
                        f"Document {i}: the quick brown fox jumps over the lazy dog "
                        f"while catalogue titles and intent labels repeat phrases. "
                    )
                    * 3
                }
            )
        )
    src = tmp_path / "sample.jsonl"
    src.write_text("\n".join(lines) + "\n")
    tok, record = BpeTokenizer.train(
        [src], vocab_size=512, out_path=tmp_path / "bpe.json", sample_bytes=200_000
    )
    assert record["documents"] > 0
    assert record["sha256_of_training_text"]
    assert tok.vocab_size == 512
    probe = "the quick brown fox jumps over the lazy dog"
    assert tok.decode(tok.encode(probe)) == probe
    reloaded = BpeTokenizer.load(tmp_path / "bpe.json")
    assert reloaded.encode(probe) == tok.encode(probe)
    assert (tmp_path / "bpe.json.record.json").exists()


def test_specials_are_shared_between_arms() -> None:
    tok = ByteTokenizer()
    assert tok.encode("abc", add_specials=True)[0] == CLS_ID
    assert tok.encode("abc", add_specials=True)[-1] == SEP_ID
