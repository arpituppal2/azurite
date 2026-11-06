#!/usr/bin/env python3
"""Score a checkpoint on *every* held-out byte, in a fixed order.

Why this exists
---------------
`pretrain.py --eval-only` draws N random windows from the held-out tail and, per
window, masks a fresh random 30% of positions. Two independent random quantities
therefore enter every number it reports:

  1. which windows          (rng.integers over the held-out span)
  2. which positions masked (rng_mask, seeded separately)

Seeding makes that *reproducible* but not *accurate*: a fixed seed is one
arbitrary draw of a small sample. Measured on the large rung's step-10000
checkpoint, the same frozen weights scored 2.168, 2.286, 2.313, 2.383, 2.393 and
2.520 across three seeds and two window counts — a 0.35 bit/byte spread. Every
rung on the ladder was scored with 64 windows, where the spread is widest, so
the reported curve (tiny 1.3425 → small 1.1175 → mid 1.2161 → large 2.8953) sits
entirely inside its own noise floor. The "inversion" at large and the
"small beats mid" ordering were not measurable with that protocol.

This script removes both random quantities by removing the sampling:

  - windows are non-overlapping and tile the held-out region exactly once, from
    `n_train` to the end of the stream, in index order;
  - the mask generator for a batch is seeded by the batch's *index*, not by a
    free parameter. The masking is still random 30% per position (that is the
    model's training distribution), but which mask a given byte receives is a
    pure function of its position, so two files scored here are identical
    byte-for-byte on the same inputs.

The protocol is then fully described by `(seq, mask_prob, batch)` — there is no
seed to tune and no way to shop for a favourable draw.

Coverage matters at this scale: the held-out tail is 2% of the stream, so a
100M-byte stream has 2,000,000 held-out bytes → 1,953 windows at seq 1024. Full
coverage scores ~600k masked positions instead of 19.6k, which is a 30x increase
in evidence and drops the standard error by ~5.5x.

Fold spread is reported alongside the mean. A single number hides that the
held-out text is heterogeneous — code, prose and math differ in intrinsic
entropy — so the folds make it visible whether a rung's advantage is uniform or
concentrated in one region.

Usage:
    python3 scripts/eval_full.py --run results/pretrain/l1_large_byte_x5 --rung large
    python3 scripts/eval_full.py --ckpt runs/a/ckpt.pt --rung small --seq 1024 --folds 8

Writes `heldout_full.json` into the run directory (or `--out`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import rung as rung_cfg  # noqa: E402
from azurite.env import configure_mps, select_device  # noqa: E402
from azurite.model import Azurite, parameter_counts  # noqa: E402

HELDOUT_FRACTION = 0.02  # must match scripts/pretrain.py
MASK_PROB = 0.30          # must match Azurite.mlm_loss default


def resolve_stream(run_dir: Path | None, explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else PROJECT / p
    if run_dir is not None and (run_dir / "run.json").exists():
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        p = Path(meta["token_stream"])
        if not p.is_absolute():
            p = PROJECT / p
        if p.exists():
            return p
    raise SystemExit("could not resolve the token stream; pass --stream")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="run directory containing ckpt.pt")
    ap.add_argument("--ckpt", default="", help="explicit checkpoint path")
    ap.add_argument("--rung", default="tiny")
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--stream", default="")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N windows (smoke tests only)")
    ap.add_argument("--mem-cap-gib", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve() if args.run else None
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    elif run_dir is not None:
        ckpt_path = run_dir / "ckpt.pt"
    else:
        raise SystemExit("pass --run or --ckpt")
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}")

    if args.mem_cap_gib:
        configure_mps(args.mem_cap_gib)
    device = select_device()

    stream_path = resolve_stream(run_dir, args.stream)
    stream = np.memmap(stream_path, dtype=np.uint16, mode="r")
    n_total = len(stream)
    n_train = int(n_total * (1 - HELDOUT_FRACTION))
    # Prefer the corpus's own document-level boundary over the 2% rule: the two
    # differ by up to one document, and scoring a document that was trained on
    # would make the held-out number optimistic in a way nothing else catches.
    _split = stream_path.with_suffix(".split.json")
    if _split.exists():
        _rec = json.loads(_split.read_text(encoding="utf-8"))
        if 0 < int(_rec["n_train"]) < n_total:
            n_train = int(_rec["n_train"])
            print(f"document-level split: n_train={n_train:,}", flush=True)

    cfg = rung_cfg(args.rung, args.arm)
    counts = parameter_counts(cfg)
    model = Azurite(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    step = int(state.get("step", -1))
    model.eval()

    seq = args.seq
    n_windows = (n_total - n_train - 1) // seq
    if args.limit:
        n_windows = min(n_windows, args.limit)

    # Per-window (nats, masked) so folds are a pure regrouping of the same sum.
    per_window: list[tuple[float, int]] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_windows):
            s = n_train + i * seq
            ids = torch.from_numpy(
                stream[s:s + seq].astype(np.int64)
            )[None].to(device)
            # Deterministic mask: a function of position in the held-out region,
            # not of a tunable seed. The generator still advances internally, so
            # the 30% pattern is genuinely random per position — but which byte
            # receives which mask depends only on where that byte sits.
            gen = torch.Generator(device=device).manual_seed(1_000_003 + i)
            loss, stats = model.mlm_loss(ids, mask_prob=MASK_PROB, generator=gen)
            n = stats["masked_positions"]
            if n:
                per_window.append((float(loss) * n, int(n)))
    elapsed = time.perf_counter() - t0

    total_nats = sum(a for a, _ in per_window)
    total_tokens = sum(b for _, b in per_window)
    bpb = (total_nats / max(1, total_tokens)) / math.log(2)

    # Fold spread: contiguous equal slices of the scored region.
    folds = []
    k = max(1, len(per_window) // args.folds)
    for f in range(args.folds):
        chunk = per_window[f * k:(f + 1) * k] if f < args.folds - 1 else per_window[f * k:]
        if not chunk:
            continue
        nats = sum(a for a, _ in chunk)
        toks = sum(b for _, b in chunk)
        folds.append({
            "fold": f,
            "windows": len(chunk),
            "bits_per_byte": round((nats / toks) / math.log(2), 6) if toks else None,
        })
    fold_vals = [f["bits_per_byte"] for f in folds if f["bits_per_byte"] is not None]

    # Standard error of the mean over windows, from the per-window dispersion.
    means = [a / b for a, b in per_window if b]
    if len(means) > 1:
        arr = np.asarray(means, dtype=np.float64)
        se = float(arr.std(ddof=1) / math.sqrt(len(arr))) / math.log(2)
    else:
        se = float("nan")

    rec = {
        "protocol": "exhaustive-nonoverlapping",
        "checkpoint": str(ckpt_path),
        "step": step,
        "tokens_seen": int(state.get("tokens_seen", -1)),
        "rung": args.rung,
        "arm": args.arm,
        "params": counts["pretrain_total"],
        "seq": seq,
        "mask_prob": MASK_PROB,
        "heldout_span": [n_train, n_total],
        "windows_scored": len(per_window),
        "coverage_fraction": round(len(per_window) / max(1, n_windows), 4),
        "scored_tokens": total_tokens,
        "bits_per_byte": round(bpb, 6),
        "stderr": round(se, 6) if se == se else None,
        "folds": folds,
        "fold_min": min(fold_vals) if fold_vals else None,
        "fold_max": max(fold_vals) if fold_vals else None,
        "fold_spread": round(max(fold_vals) - min(fold_vals), 6) if fold_vals else None,
        "seconds": round(elapsed, 1),
        "measured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": (
            "No sampling: windows tile the held-out region once, in order, and "
            "each window's mask is seeded by its index. Two scorers agree "
            "bit-for-bit; there is no seed to choose and no favourable draw to "
            "find. Reproducible != accurate, which is why the old 64-window "
            "protocol was replaced rather than merely seeded."
        ),
    }

    out = Path(args.out) if args.out else (
        (run_dir / "heldout_full.json") if run_dir else Path("heldout_full.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")

    print(f"step {step:,}  {rec['params']:,} params  seq {seq}")
    print(f"scored {total_tokens:,} masked positions over {len(per_window):,} "
          f"windows ({rec['coverage_fraction'] * 100:.1f}% coverage) in {elapsed:.0f}s")
    print(f"bits/byte {bpb:.4f}  (fold spread "
          f"{rec['fold_min']:.4f}..{rec['fold_max']:.4f})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
