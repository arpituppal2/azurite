#!/usr/bin/env python3
"""Bits/byte baselines for the L1 gate: unigram and bigram over raw bytes.

Why this exists: an L1 loss curve that goes down is not evidence of learning. A
context-free byte model gets roughly 4.5-5 bits/byte on English text and a
bigram gets a bit under that; if the trained model does not beat the bigram, it
has learned nothing about context and there is no reason to spend 300 GPU-hours
scaling it. This is the cheapest falsifier in the project.

Both baselines are fit on the **training** region of the same token stream the
model reads, and scored on the same held-out tail, so the comparison differs only
in the model.

Usage:
    .venv/bin/python3 scripts/bigram_baseline.py --tokens 100e6 --arm byte
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results" / "pretrain"
TOKENS = PROJECT / "data" / "tokens"
HELDOUT_FRACTION = 0.02
N_BYTES = 256
SEP_ID = 3  # document boundary written by scripts/pretrain.py prepare()


def bits_per_byte_from_logp(logp: np.ndarray) -> float:
    return float(np.mean(-logp) / math.log(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=100e6)
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--count-tokens", type=float, default=20e6)
    ap.add_argument("--eval-tokens", type=float, default=2e6)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    path = TOKENS / f"{args.arm}_{int(args.tokens)}.u16"
    if not path.exists():
        raise SystemExit(f"no token stream at {path}; run scripts/pretrain.py --prepare")
    stream = np.memmap(path, dtype=np.uint16, mode="r")
    n_total = len(stream)
    n_train = int(n_total * (1 - HELDOUT_FRACTION))

    # Symbols are the 256 byte values plus <sep> at index 0. <sep> is a real
    # stream symbol now (the document boundary written by prepare()), so the
    # baseline and the model must see the same alphabet or the bits/byte
    # comparison would be between two different sequences. It is ~0.02% of
    # positions, which is why it cannot move the baseline, but the denominator
    # is stated rather than assumed.
    n_sym = N_BYTES + 1

    def to_symbols(arr, what: str) -> np.ndarray:
        a = np.asarray(arr, dtype=np.int64)
        bad = (a != SEP_ID) & ((a < 4) | (a >= 4 + N_BYTES))
        if bad.any():
            raise SystemExit(f"{what} contains ids outside the byte alphabet")
        return np.where(a == SEP_ID, 0, a - 3)

    count_n = min(int(args.count_tokens), n_train)
    train = to_symbols(stream[:count_n], "token stream")

    unigram = np.bincount(train, minlength=n_sym).astype(np.float64)
    uni_p = (unigram + args.smoothing) / (unigram.sum() + args.smoothing * n_sym)

    pairs = train[:-1] * n_sym + train[1:]
    bigram = np.bincount(pairs, minlength=n_sym * n_sym).reshape(n_sym, n_sym)
    bigram = bigram.astype(np.float64) + args.smoothing
    bi_p = bigram / bigram.sum(axis=1, keepdims=True)

    # score on the held-out tail
    eval_n = min(int(args.eval_tokens), n_total - n_train)
    held = to_symbols(stream[n_train : n_train + eval_n], "held-out slice")

    uni_logp = np.log(uni_p[held])
    bi_logp = np.log(bi_p[held[:-1], held[1:]])
    heldout_bperplexity = 2 ** bits_per_byte_from_logp(bi_logp)

    result = {
        "arm": args.arm,
        "token_stream": str(path.relative_to(PROJECT)),
        "count_tokens": int(count_n),
        "eval_tokens": int(eval_n),
        "smoothing": args.smoothing,
        "unigram_bits_per_byte": round(bits_per_byte_from_logp(uni_logp), 4),
        "bigram_bits_per_byte": round(bits_per_byte_from_logp(bi_logp), 4),
        "bigram_cross_entropy_nats": round(float(np.mean(-bi_logp)), 4),
        "bigram_perplexity_over_bytes": round(heldout_bperplexity, 2),
        "note": (
            "bigram and unigram are fit on the training region only; the score is "
            "on the held-out tail, the same slice scripts/pretrain.py reports. "
            "Alphabet is 256 bytes plus <sep>, which is 0.02% of positions."
        ),
    }

    out = Path(args.out) if args.out else RESULTS / f"bigram_{args.arm}_{int(args.tokens)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")

    print(json.dumps(result, indent=1))
    print(f"\nwrote {out.resolve().relative_to(PROJECT)}")
    print(
        f"\nL1 gate: a trained model must report bits/byte below "
        f"{result['bigram_bits_per_byte']:.4f} to have beaten the bigram."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
