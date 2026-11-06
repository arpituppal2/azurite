#!/usr/bin/env python3
"""Score every ladder rung under the one exhaustive protocol, into one table.

The ladder's numbers were originally produced by `pretrain.py --eval-only` with
64 random windows: 0.66% coverage of a 2M-byte held-out region, and a fresh
random mask per window. Measured on one frozen checkpoint, six draws of that
protocol spanned 2.168..2.520 bits/byte — 0.35, which is larger than every gap
the ladder claims to measure. Those numbers cannot support a scaling law.

This driver replaces them. Each rung is scored by `scripts/eval_full.py`, which
tiles the whole held-out region once in a fixed order and derives each window's
mask from its position. Repeatability is ~0.002 bits/byte (residual CUDA kernel
non-determinism), which is two orders of magnitude tighter than the old draws.

Two sequence lengths are scored per rung where the checkpoint exists at both:

  - `seq=1024` for every rung, so the comparison is like-for-like even though
    tiny/small were trained at 256. This is a train/test length mismatch and it
    is labelled as one; a model asked to predict at 4x its training window is
    being measured outside its competence.
  - `seq=256`, the length tiny/small actually trained at, so the matched
    comparison is available for the rungs where 1024 would be unfair.

The table reports both rather than picking the flattering one, because the
choice of eval length reverses the ranking (tiny 1.1781 at 256 vs 1.6661 at
1024) and silently choosing either would be the same error as the old protocol.

Usage:
    python3 scripts/score_curve.py                 # default rung set
    python3 scripts/score_curve.py --seqs 1024
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PRETRAIN = PROJECT / "results" / "pretrain"
STREAM = "data/tokens/byte_100000000.u16"

# label -> (run dir, rung, params come from the checkpoint itself)
RUNGS = [
    ("tiny", "l1_tiny_byte", "tiny"),
    ("small", "l1_small_byte", "small"),
    ("mid", "l1_mid_byte", "mid"),
    ("large", "l1_large_byte", "large"),
    ("large_x5@163.8M", "_snap_x5", "large"),
]


def score(run_rel: str, rung: str, seq: int, tag: str, timeout: int = 3600) -> dict:
    out = PRETRAIN / run_rel / f"full_seq{seq}.json"
    if out.exists():
        rec = json.loads(out.read_text(encoding="utf-8"))
        rec["cached"] = True
        return rec
    cmd = [
        sys.executable, str(PROJECT / "scripts" / "eval_full.py"),
        "--run", str(PRETRAIN / run_rel), "--rung", rung,
        "--stream", STREAM, "--seq", str(seq),
        "--out", str(out),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        return {"label": tag, "seq": seq, "error": proc.stderr.strip().splitlines()[-6:]}
    rec = json.loads(out.read_text(encoding="utf-8"))
    rec["label"] = tag
    rec["cached"] = False
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", type=int, default=[1024, 256])
    ap.add_argument("--out", default=str(PRETRAIN / "curve_full.json"))
    args = ap.parse_args()

    rows = []
    t0 = time.perf_counter()
    for seq in args.seqs:
        for label, run_rel, rung in RUNGS:
            if not (PRETRAIN / run_rel / "ckpt.pt").exists():
                print(f"skip {label} seq={seq}: no checkpoint at {run_rel}/ckpt.pt",
                      flush=True)
                continue
            rec = score(run_rel, rung, seq, label)
            if "error" in rec:
                print(f"FAIL {label} seq={seq}: {rec['error']}", flush=True)
                continue
            rec["label"] = label
            rec["seq"] = seq
            rows.append(rec)
            print(f"{label:16s} seq={seq:<5} {rec['params']:>12,} params  "
                  f"step {rec['step']:>7,}  bpb {rec['bits_per_byte']:.4f}  "
                  f"[{rec['fold_min']:.3f}..{rec['fold_max']:.3f}]  "
                  f"{rec['scored_tokens']:,} pos", flush=True)

    summary = {
        "protocol": "exhaustive-nonoverlapping",
        "mask_prob": 0.30,
        "repeatability": "+/- 0.002 bits/byte (CUDA kernel order, measured A/B)",
        "superseded": "64-window random draws, spread 0.168..0.520 on one checkpoint",
        "seconds": round(time.perf_counter() - t0, 1),
        "measured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
