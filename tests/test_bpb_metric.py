"""The gate is bits per byte; a BPE token is not a byte.

`evaluate()` used to divide nats-per-token by ln2 and call it bits/byte. That is
exact for the byte arm (one token is one byte) and wrong by ~3.6x for the BPE
arm, which would have made the reader's held-out gate compare a bits-per-token
number against a byte bigram.
"""

from __future__ import annotations

import numpy as np
import torch

from scripts.pretrain import evaluate, token_byte_lengths


class _Stub:
    """One nat per masked position; 30% of positions masked, as in pretraining."""

    def eval(self):
        return self

    def train(self):
        return self

    def mlm_loss(self, ids, generator=None):
        return torch.tensor(1.0), {"masked_positions": float(ids.numel()) * 0.3}


def test_byte_arm_bpb_equals_bits_per_token() -> None:
    stream = np.tile(np.arange(4, 260, dtype=np.uint16), 3)  # byte arm: one byte per id
    res = evaluate(_Stub(), stream, 0, 64, 4, "cpu", seed=7, arm="byte")
    assert abs(res["bytes_per_token"] - 1.0) < 1e-9
    assert abs(res["bits_per_byte"] - res["bits_per_token"]) < 1e-9
    assert res["metric_schema"] == 2


def test_bpe_arm_divides_by_measured_bytes_per_token() -> None:
    table = token_byte_lengths("bpe")
    wide = int(np.flatnonzero(table >= 3)[0])
    stream = np.full(400, wide, dtype=np.uint16)
    res = evaluate(_Stub(), stream, 0, 64, 4, "cpu", seed=7, arm="bpe")
    assert res["bytes_per_token"] >= 3
    expected = res["bits_per_token"] / res["bytes_per_token"]
    assert abs(res["bits_per_byte"] - expected) < 1e-9


def test_token_byte_table_matches_encode() -> None:
    tok_table = token_byte_lengths("bpe")
    from azurite.tokenizers import load as load_tokenizer

    tok = load_tokenizer("bpe")
    text = "Azurite reads a document and answers."
    ids = tok.encode(text)
    assert sum(int(tok_table[i]) for i in ids) == len(text.encode("utf-8"))
