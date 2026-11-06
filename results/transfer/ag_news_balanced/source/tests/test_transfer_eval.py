"""The transfer comparison is only interpretable if the two arms differ in
exactly one thing: the initial weights.

Two properties make that true, and both are asserted here rather than assumed:

1. the example schedule depends only on (rows, batch, steps, seed), so the
   pretrained and random arms consume the same examples in the same order;
2. the random arm's initialisation is a function of the seed, not of whatever
   global RNG state happened to exist when the process started — which is how
   the first version of the harness worked, making each run irreproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from azurite.config import rung  # noqa: E402
from transfer_eval import batch_schedule, build_model, draw_provenance, evaluate, state_sha256  # noqa: E402
from azurite.decisions import CandidateBank, Example, TaskSpec, pack_cross
from azurite.tokenizers import ByteTokenizer


def test_evidence_draw_matches_shuffled_packing() -> None:
    spec = TaskSpec("synthetic", "test", ("text",), "label", "Choose one.")
    bank = CandidateBank("synthetic", ["World", "Sports", "Business", "Sci/Tech"], list(range(4)))
    rows = [Example(i, f"news example {i}", i % 4) for i in range(12)]
    batch, stats = pack_cross(ByteTokenizer(), spec, bank, rows, max_seq=256, seed=2)
    proof = draw_provenance(rows, bank, spec.name, 2, "cross")
    assert stats["unpackable"] == 0
    assert proof["class_counts"] == {str(i): 3 for i in range(4)}
    assert batch.gold.tolist() == [r["packed_gold"] for r in proof["rows"]]
    assert [r["candidate_order"][g] for r, g in zip(proof["rows"], batch.gold.tolist())] == [r.gold for r in rows]


def test_probability_evidence_does_not_change_metrics_or_rng() -> None:
    spec = TaskSpec("synthetic", "test", ("text",), "label", "Choose one.")
    bank = CandidateBank("synthetic", ["yes", "no"], [0, 1])
    batch, _ = pack_cross(ByteTokenizer(), spec, bank, [Example(0, "hello", 0)], max_seq=256)
    torch.manual_seed(0)
    model, _ = build_model(rung("tiny", "byte"), "cross", None, "cpu")
    rng = torch.get_rng_state().clone()
    plain = evaluate(model, batch, "cpu", keep_predictions=True)
    evidence = evaluate(model, batch, "cpu", keep_predictions=True, keep_probabilities=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert {k: v for k, v in evidence.items() if k != "probabilities"} == plain
    assert torch.allclose(torch.tensor(evidence["probabilities"]).sum(-1), torch.ones(1))
    assert torch.tensor(evidence["probabilities"]).argmax(-1).tolist() == plain["predictions"]
    assert state_sha256(model.state_dict()) == state_sha256(dict(reversed(list(model.state_dict().items()))))


def test_schedule_is_identical_for_the_same_seed() -> None:
    a = list(batch_schedule(100, 8, 40, seed=0))
    b = list(batch_schedule(100, 8, 40, seed=0))
    assert len(a) == 40
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_schedule_depends_on_the_seed() -> None:
    a = list(batch_schedule(100, 8, 10, seed=0))
    b = list(batch_schedule(100, 8, 10, seed=1))
    assert any(not torch.equal(x, y) for x, y in zip(a, b))


def test_schedule_covers_every_row_once_per_epoch() -> None:
    """An epoch that skipped rows would silently shrink the training set, and
    disproportionately so for whichever rows sorted last."""
    seen = torch.cat(list(batch_schedule(96, 8, 12, seed=3)))
    assert sorted(seen.tolist()) == list(range(96))


def test_schedule_reshuffles_between_epochs() -> None:
    schedule = list(batch_schedule(64, 8, 16, seed=5))
    first = torch.cat(schedule[:8]).tolist()
    second = torch.cat(schedule[8:]).tolist()
    assert first != second, "epoch 2 repeated epoch 1's order"


def test_random_init_is_a_function_of_the_seed() -> None:
    cfg = rung("tiny", "byte")
    torch.manual_seed(0)
    a, _ = build_model(cfg, "cross", None, "cpu")
    torch.manual_seed(0)
    b, _ = build_model(cfg, "cross", None, "cpu")
    for (ka, va), (kb, vb) in zip(
        a.encoder.state_dict().items(), b.encoder.state_dict().items()
    ):
        assert ka == kb
        assert torch.equal(va, vb), f"{ka} differs between two seeded inits"

    torch.manual_seed(1)
    c, _ = build_model(cfg, "cross", None, "cpu")
    assert not torch.equal(
        a.encoder.state_dict()["embed.weight"], c.encoder.state_dict()["embed.weight"]
    )
