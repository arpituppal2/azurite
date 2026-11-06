"""The Jev protocol engine: packing, one real arm run, and the P0 decision.

These are the parts a thin CLI must not re-implement. The packing check pins the
message-only arm and the complete-example arm; the inference check runs the real
QueryDual forward on a tiny config and asserts the record shape the verdict
renderer consumes.
"""

from __future__ import annotations

import torch

from azurite.config import EncoderConfig
from azurite.jev import Arm, Packed, decide, pack_contexts, run_arm
from azurite.model import Azurite
from azurite.query import QueryDual


class FakeTokenizer:
    def encode(self, text: str, add_specials: bool = False) -> list[int]:
        ids = [5 + (ord(ch) % 200) for ch in text]
        return ([2] + ids + [3]) if add_specials else ids


def _rows() -> list[dict]:
    return [
        {
            "id": "r0",
            "i": 0,
            "text": "my card was declined",
            "gold": "declined",
            "gold_index": 0,
            "examples": [{"id": 10, "message": "card declined at shop", "label": "declined"}],
        },
        {
            "id": "r1",
            "i": 1,
            "text": "where is my refund",
            "gold": "refund",
            "gold_index": 1,
            "examples": [{"id": 11, "message": "refund not shown", "label": "refund"}],
        },
    ]


def test_pack_contexts_separates_message_only_from_examples() -> None:
    tok = FakeTokenizer()
    arms = [Arm("a0", "query", 0), Arm("a2c", "query", 24, budget=1_020)]
    packed = pack_contexts(_rows(), arms, tok)
    assert packed.contexts["a0"] == [row["text"] for row in _rows()]
    assert packed.effective["a0"] == [0, 0]
    assert all("Labeled examples:" in text for text in packed.contexts["a2c"])
    assert packed.effective["a2c"] == [1, 1]
    assert packed.ids["a0"].shape[0] == 2
    assert packed.ids["a2c"].shape[0] == 2


def test_run_arm_returns_the_record_shape_verdicts_expect() -> None:
    cfg = EncoderConfig(
        d_model=16, n_layer=1, n_head=2, d_ff=32, vocab_size=260, max_seq=64
    )
    model = QueryDual(Azurite(cfg)).eval()
    rows = _rows()
    packed = Packed(
        messages=[row["text"] for row in rows],
        msg_ids=torch.tensor([[2, 10, 11, 3], [2, 12, 13, 3]]),
        msg_mask=torch.ones(2, 4, dtype=torch.bool),
        contexts={"a0": [row["text"] for row in rows]},
        effective={"a0": [0, 0]},
        ids={"a0": torch.tensor([[2, 10, 11, 3], [2, 12, 13, 3]])},
        masks={"a0": torch.ones(2, 4, dtype=torch.bool)},
    )
    cand_ids = torch.tensor([[5, 6, 0], [7, 8, 0], [9, 10, 0]])
    cand_mask = torch.tensor([[True, True, False]] * 3)
    summary, records = run_arm(
        model,
        kind="query",
        arm=Arm("a0", "query", 0),
        packed=packed,
        rows=rows,
        cand_ids=cand_ids,
        cand_mask=cand_mask,
        batch=2,
        device=torch.device("cpu"),
        head_sha="deadbeef",
        head_path="test",
    )
    assert summary["n"] == 2 and len(records) == 2
    assert set(records[0]) >= {
        "arm", "id", "i", "gold_index", "pred", "correct",
        "context_sha256", "head_sha256",
    }
    assert summary["correct"] == sum(row["correct"] for row in records)
    assert 0.0 <= summary["wilson95"][0] <= summary["wilson95"][1] <= 1.0


def test_decide_applies_the_p0_rule() -> None:
    base_summary = {"accuracy": 0.50, "n": 100, "correct": 50}
    base_pair = {"mcnemar": {"p": 0.5}, "bootstrap_lower_95": -0.10}
    missed = decide(
        primary_arm="a",
        summaries={"a": base_summary},
        paired={"a": base_pair},
        rule="test",
    )
    assert missed["verdict"] == "P0 missed"
    assert not missed["point_estimate_meets_bar"]

    met = decide(
        primary_arm="a",
        summaries={"a": {**base_summary, "accuracy": 0.93}},
        paired={"a": base_pair},
        rule="test",
    )
    assert met["verdict"] == "P0 met (equal or above)"
    assert met["noninferior_at_minus_2pp"] is False

    exceeded = decide(
        primary_arm="a",
        summaries={"a": {**base_summary, "accuracy": 0.93}},
        paired={"a": {"mcnemar": {"p": 0.01}, "bootstrap_lower_95": 0.01}},
        rule="test",
    )
    assert exceeded["verdict"] == "P0 exceeded"
