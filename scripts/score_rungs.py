#!/usr/bin/env python3
"""Score checkpoints under one fixed, deterministic evaluation protocol.

The ladder's numbers were produced across two protocols: the local runs drew
their eval windows from a fixed seed that ignored `--eval-seed`, and the mid run
used a different window length. Bits-per-byte measured on different window
lengths is not the same quantity, so a curve built from both is not a curve.

This script re-scores everything one way:

    seq 256, 64 windows, seed 1234   — run TWICE, to prove reproducibility
    seq 1024, 64 windows, seed 1234  — to show how much window length moves it

The second protocol exists because the difference between them is itself a
result: if a 33.7M model scored at seq 1024 is worse than a 14.3M model scored
at seq 256, the comparison may be measuring window length rather than capacity.

Usage:
    python3 scripts/score_rungs.py --spec l1_mid_byte:mid --spec _ctrl_lap_tiny:tiny
    # writes <dir>/score.json for each spec
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results" / "pretrain"

# (label, seq, windows, seed)
PROTOCOLS = [
    ("seq256_run1", 256, 64, 1234),
    ("seq256_run2", 256, 64, 1234),
    ("seq1024", 1024, 64, 1234),
]


def evaluate(rung: str, out_rel: str, seq: int, windows: int, seed: int) -> dict:
    # `pretrain.py` resolves `--out` relative to the process cwd, not to
    # `results/pretrain/`. Passing the bare directory name made it look for
    # `./_ctrl_lap_tiny/ckpt.pt` and report "no checkpoint" for two checkpoints
    # that were sitting right where they belonged.
    run_rel = f"results/pretrain/{out_rel}"
    cmd = [
        sys.executable, str(PROJECT / "scripts" / "pretrain.py"),
        "--rung", rung, "--arm", "byte", "--tokens", "100e6",
        "--out", run_rel, "--eval-only",
        "--seq", str(seq), "--eval-windows", str(windows), "--eval-seed", str(seed),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip().splitlines()[-8:]}
    held = PROJECT / run_rel / "heldout.json"
    if not held.exists():
        return {"error": "no heldout.json"}
    return json.loads(held.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True,
                    help="result-directory-name:rung, repeatable")
    ap.add_argument("--out", default="", help="optional aggregate json path")
    args = ap.parse_args()

    aggregate: dict = {"scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rungs": {}}

    for spec in args.spec:
        if ":" not in spec:
            print(f"bad spec {spec!r}; want dir:rung", flush=True)
            return 2
        out_rel, rung = spec.split(":", 1)
        if not (RESULTS / out_rel / "ckpt.pt").exists():
            print(f"skip {out_rel}: no ckpt.pt", flush=True)
            continue

        record = {"rung": rung, "dir": out_rel, "protocols": {}}
        for label, seq, windows, seed in PROTOCOLS:
            rec = evaluate(rung, out_rel, seq, windows, seed)
            record["protocols"][label] = rec
            bpb = rec.get("bits_per_byte")
            print(f"{out_rel:22s} {label:12s} {bpb if bpb is not None else rec}", flush=True)

        a = record["protocols"]["seq256_run1"].get("bits_per_byte")
        b = record["protocols"]["seq256_run2"].get("bits_per_byte")
        record["reproducible"] = (a is not None and a == b)
        if a is not None and b is not None:
            print(f"{out_rel:22s} identical-seed check: {a!r} == {b!r} -> {record['reproducible']}",
                  flush=True)

        (RESULTS / out_rel / "score.json").write_text(
            json.dumps(record, indent=1) + "\n", encoding="utf-8")
        aggregate["rungs"][out_rel] = record

    if args.out:
        (PROJECT / args.out).write_text(json.dumps(aggregate, indent=1) + "\n",
                                        encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
