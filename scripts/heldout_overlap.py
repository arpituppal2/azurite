#!/usr/bin/env python3
"""Is the held-out tail actually unseen text?

The L1 gate compares the model's held-out bits/byte against a bigram model fit
on the training region. That comparison is only meaningful if the held-out
region does not contain text the model already trained on. Splitting by
*position* does not guarantee that: FineWeb, FineWeb-Edu and Wikipedia are
crawled corpora, and the same page can appear in several shards. A random window
of the held-out region can therefore be a verbatim duplicate of training text,
and a model that memorised it would look better than it is.

This measures that directly, with exact substring search rather than a proxy:
sample windows from the held-out region and search for each one verbatim in the
training region.

It searches the **uint16 memmap as raw bytes**, so no decoding happens and
`<sep>` boundaries cannot be accidentally normalised away. A window of k ids
occupies exactly 2k bytes of that view.

What it cannot see: paraphrase, translation, reformatting, or the same article
with different markup. A zero here means no verbatim duplication at the window
length tested — it is a lower bound on contamination, not a clean bill.

Usage:
    .venv/bin/python3 scripts/heldout_overlap.py
    .venv/bin/python3 scripts/heldout_overlap.py --windows 2000 --len 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
TOKENS = PROJECT / "data" / "tokens"
RESULTS = PROJECT / "results" / "pretrain"
HELDOUT_FRACTION = 0.02


def measure(
    train: bytes,
    held: bytes,
    windows: int,
    k: int,
    seed: int,
    starts: np.ndarray | None = None,
) -> dict:
    """Fraction of sampled held-out windows that occur verbatim in `train`.

    Separated from `main` so the two controls can be run on synthetic bytes:
    a stream whose held-out half copies training text must report 100%, and an
    independent stream must report 0%. A search that cannot reproduce the
    positive case would report 0% on real data and mean nothing.
    """
    if len(held) < 2 * (k + 1):
        raise ValueError("held-out region shorter than one window")
    if starts is None:
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, len(held) - 2 * k - 1, size=windows)
    windows = len(starts)
    q = max(8, k // 4)
    hits_full = 0
    hits_half = 0
    examples: list[dict] = []
    for i, s in enumerate(starts):
        if train.find(held[s : s + 2 * k]) >= 0:
            hits_full += 1
            if len(examples) < 5:
                examples.append({"window_index": int(i), "offset_bytes": int(s), "len_ids": k})
        elif train.find(held[s : s + 2 * q]) >= 0:
            hits_half += 1
    return {
        "hits_full": hits_full,
        "hits_fragment": hits_half,
        "rate_full": hits_full / windows,
        "rate_incl_fragment": (hits_full + hits_half) / windows,
        "examples": examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=100e6)
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--windows", type=int, default=1000)
    ap.add_argument("--len", type=int, default=128, help="window length in ids")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    path = TOKENS / f"{args.arm}_{int(args.tokens)}.u16"
    if not path.exists():
        raise SystemExit(f"no token stream at {path}; run scripts/pretrain.py --prepare")

    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n_ids = len(raw) // 2
    n_train = int(n_ids * (1 - HELDOUT_FRACTION))
    held_mask = slice(n_train * 2, len(raw))

    print(f"stream {n_ids:,} ids; train {n_train:,}; held-out {n_ids - n_train:,}", flush=True)
    t0 = time.perf_counter()
    # The training region as an immutable bytes object: bytes.find is the
    # linear-time C search, and it is faster than any per-window scan in Python.
    train = bytes(raw[: n_train * 2])
    print(f"training region materialised: {len(train) / 2**20:.0f} MiB "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)
    held = bytes(raw[held_mask])
    print(f"held-out region: {len(held) / 2**20:.1f} MiB", flush=True)

    k = args.len
    t0 = time.perf_counter()
    try:
        m = measure(train, held, args.windows, k, args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc))
    elapsed = time.perf_counter() - t0
    hits_full, hits_half = m["hits_full"], m["hits_fragment"]
    full_rate, frag_rate = m["rate_full"], m["rate_incl_fragment"]
    examples = m["examples"]
    result = {
        "token_stream": str(path.relative_to(PROJECT)),
        "train_ids": n_train,
        "heldout_ids": n_ids - n_train,
        "windows": args.windows,
        "window_ids": k,
        "window_bytes": 2 * k,
        "seed": args.seed,
        "verbatim_hits_full_window": hits_full,
        "verbatim_rate_full_window": round(full_rate, 5),
        "fragment_hits_quarter_window": hits_half,
        "verbatim_rate_incl_fragment": round(frag_rate, 5),
        "seconds": round(elapsed, 1),
        "examples": examples,
        "method": (
            "exact substring search of sampled held-out windows in the raw uint16 "
            "training region viewed as bytes; lower bound on duplication"
        ),
        "cannot_detect": "paraphrase, translation, reformatting, markup changes",
    }
    out = Path(args.out) if args.out else RESULTS / f"heldout_overlap_{args.arm}_{int(args.tokens)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")

    print()
    print(f"full {k}-id windows found verbatim in training: {hits_full}/{args.windows} "
          f"= {full_rate:.2%}")
    print(f"including {k // 4}-id fragments: {hits_full + hits_half}/{args.windows} "
          f"= {frag_rate:.2%}")
    print(f"wrote {out.resolve().relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
