"""Recovery v1: ONE bounded fix attempt — banking77 dual s0 at 2x steps (756).

Diagnosis (from pulled histories): banking77-dual learns and is still rising
at step 377 — truncated by budget, not collapsed. Emotion both-paths pins NLL
exactly at ln6 (collapse signature; no fix attempted — dead-end family).
Financial_phrasebank plateaus (underfit, flat — more steps unlikely to add
14pp). Same recipe otherwise. Per-cell timeout 300 s.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WS = Path("/marimo/azurite_ws")
CKPT = WS / "results/pretrain/l1_tiny_byte/ckpt.pt"
OUT = WS / "results/transfer/recovery_v1"
CELLS = [("banking77_dual_s0_x2", 756, 2000, 500)]
PER_CELL_TIMEOUT = 300


def main() -> int:
    assert CKPT.exists(), f"anchor ckpt missing: {CKPT}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CKPT_CHECK.txt").write_text(f"{CKPT} exists\n")
    recs = []
    for name, steps, examples, test in CELLS:
        cell = OUT / name
        ev = cell / "evidence"
        ev.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(WS / "scripts/transfer_eval.py"),
               "--task", "banking77", "--arm", "byte", "--rung", "tiny",
               "--init", "pretrained", "--ckpt", str(CKPT),
               "--path", "dual", "--steps", str(steps), "--batch", "16",
               "--lr", "5e-4", "--seed", "0", "--split-seed", "0",
               "--stratify", "--cross-shuffle", "off",
               "--examples", str(examples), "--test-examples", str(test),
               "--device", "cuda",
               "--evidence-dir", str(ev), "--out", str(cell / "result.json")]
        t0 = time.perf_counter()
        try:
            pr = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=PER_CELL_TIMEOUT)
            rc = pr.returncode
            tail = (pr.stdout + pr.stderr)[-800:]
        except subprocess.TimeoutExpired:
            rc, tail = -1, "TIMEOUT"
        wall = round(time.perf_counter() - t0, 1)
        acc = None
        rp = cell / "result.json"
        if rp.exists():
            try:
                acc = json.loads(rp.read_text())["final"]["accuracy"]
            except Exception:
                acc = "UNPARSEABLE"
        (cell / "CELL.log").write_text(f"rc={rc} wall={wall} acc={acc}\n{tail}\n")
        recs.append({"cell": name, "rc": rc, "wall_s": wall, "acc": acc})
        print(f"{name}: rc={rc} wall={wall} acc={acc}", flush=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(recs, indent=1) + "\n")
    print(f"DONE {len(recs)} cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
