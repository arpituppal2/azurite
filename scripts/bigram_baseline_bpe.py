#!/usr/bin/env python3
"""Byte bigram baseline over the SAME text the BPE reader reads.

The plan's gate is stated in bits per byte: held-out bpb must beat the bigram by
at least 0.05. `scripts/bigram_baseline.py` can only read a byte-arm memmap, so
it cannot score the BPE stream. This fits a byte bigram on text decoded from the
BPE stream's training region and scores it on text decoded from that stream's
held-out tail, so the model and the baseline see the same documents.

Paired with `pretrain.evaluate()`'s byte accounting, both numbers are bits per
UTF-8 byte of the same text.

    python scripts/bigram_baseline_bpe.py --stream data/tokens/bpe_6200000000.u16 \
        --count-tokens 10e6 --eval-tokens 1e6 \
        --out results/parity/bigram_bpe_6200000000.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.tokenizers import SEP_ID, load as load_tokenizer  # noqa: E402

HELDOUT_FRACTION = 0.02
N_BYTES = 256


def decode_documents(tok, ids: np.ndarray):
    """Yield the UTF-8 bytes of each document; <sep> never enters the text."""
    prev = 0
    for cut in np.flatnonzero(ids == SEP_ID):
        if cut > prev:
            yield tok.decode(ids[prev:cut].tolist()).encode("utf-8")
        prev = cut + 1
    if prev < len(ids):
        yield tok.decode(ids[prev:].tolist()).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--count-tokens", type=float, default=10e6)
    ap.add_argument("--eval-tokens", type=float, default=1e6)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--out", default="results/parity/bigram_bpe.json")
    args = ap.parse_args()

    path = Path(args.stream)
    if not path.is_absolute():
        path = PROJECT / path
    if not path.exists():
        raise SystemExit(f"no token stream at {path}")
    tok = load_tokenizer("bpe")
    stream = np.memmap(path, dtype=np.uint16, mode="r")
    n_total = len(stream)
    n_train = int(n_total * (1 - HELDOUT_FRACTION))

    count_n = min(int(args.count_tokens), n_train)
    eval_n = min(int(args.eval_tokens), n_total - n_train)

    uni = np.zeros(N_BYTES, dtype=np.float64)
    bi = np.zeros(N_BYTES * N_BYTES, dtype=np.float64)
    train_bytes = 0
    train_docs = 0
    for doc in decode_documents(tok, np.asarray(stream[:count_n], dtype=np.int64)):
        if len(doc) < 2:
            continue
        arr = np.frombuffer(doc, dtype=np.uint8)
        uni += np.bincount(arr, minlength=N_BYTES)
        bi += np.bincount(arr[:-1].astype(np.int64) * N_BYTES + arr[1:], minlength=N_BYTES * N_BYTES)
        train_bytes += len(arr)
        train_docs += 1
    bi = bi.reshape(N_BYTES, N_BYTES)

    s = args.smoothing
    uni_p = (uni + s) / (uni.sum() + s * N_BYTES)
    bi_p = (bi + s) / (bi.sum(axis=1, keepdims=True) + s * N_BYTES)

    held_start = n_train
    held = np.asarray(stream[held_start:held_start + eval_n], dtype=np.int64)
    uni_logp = 0.0
    bi_logp = 0.0
    scored = 0
    held_docs = 0
    held_bytes = 0
    for doc in decode_documents(tok, held):
        if len(doc) < 2:
            continue
        arr = np.frombuffer(doc, dtype=np.uint8)
        uni_logp += float(np.log(uni_p[arr]).sum())
        bi_logp += float(np.log(bi_p[arr[:-1], arr[1:]]).sum())
        scored += len(arr) - 1
        held_bytes += len(arr)
        held_docs += 1

    # The unigram covers every byte, including each document's first; the bigram
    # only covers pairs. Each gets its own denominator.
    bigram_bpb = -bi_logp / max(1, scored) / math.log(2)
    unigram_bpb = -uni_logp / max(1, held_bytes) / math.log(2)

    result = {
        "stream": str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path),
        # l1_report.find_baseline matches a run's token_stream by file name, and
        # only looks under results/pretrain/. Same key, same naming convention.
        "token_stream": str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path),
        "arm": "bpe",
        "tokenizer": "azurite_bpe_32004.json",
        "count_tokens": int(count_n),
        "count_bytes": int(train_bytes),
        "count_documents": train_docs,
        "train_bytes_per_token": round(train_bytes / max(1, count_n), 4),
        "eval_tokens": int(len(held)),
        "eval_documents": held_docs,
        "eval_bytes": int(held_bytes),
        "eval_scored_bytes": int(scored),
        "eval_bytes_per_token": round(held_bytes / max(1, len(held)), 4),
        "smoothing": s,
        "unigram_bits_per_byte": round(unigram_bpb, 4),
        "bigram_bits_per_byte": round(bigram_bpb, 4),
        "gate": {"criterion": "held-out model bpb <= bigram bpb - 0.05",
                 "threshold_bits_per_byte": round(bigram_bpb - 0.05, 4)},
        "note": ("bigram fit on the training region only and scored on the held-out tail of the same BPE stream; "
                 "bytes are UTF-8 bytes of the decoded text, so this is comparable with pretrain.evaluate() "
                 "metric_schema 2"),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
