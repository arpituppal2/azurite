"""Superiority v1: fresh pretrained cross+dual cells on tasks with recorded Jev baselines.

Usage: run_superiority_v1.py BATCH (1 or 2).
Batch 1: emotion + financial_phrasebank. Batch 2: prompt_injections + banking77.
Recipe mirrors Execution-1 (stratified, split-seed 0, cross-shuffle off,
batch 16, lr 5e-4, pretrained anchor ckpt); per-task train/test budgets sized
to pool (recorded in SUMMARY). Seed 0 only; seed variation documented elsewhere.
Per-cell timeout 180 s.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WS = Path("/marimo/azurite_ws")
CKPT = WS / "results/pretrain/l1_tiny_byte/ckpt.pt"
OUT = WS / "results/transfer/superiority_v1"

BUDGETS = {
    "emotion": {"examples": 2000, "test": 500, "steps": 378},
    "financial_phrasebank": {"examples": 1200, "test": 500, "steps": 378},
    "prompt_injections": {"examples": 546, "test": 116, "steps": 378},
    "banking77": {"examples": 2000, "test": 500, "steps": 378},
}
BATCHES = {
    "1": ["emotion", "financial_phrasebank"],
    "2": ["prompt_injections", "banking77"],
}
PER_CELL_TIMEOUT = 180

# Recorded Jev points (envelope v0 BASELINES provenance; NOT re-measured here —
# live Jev queries would spend founder prepaid credits).
JEV_RECORDED = {
    "emotion": 0.585,
    "financial_phrasebank": 0.73,
    "prompt_injections": 0.741,
    "banking77": 0.797,
}


def main() -> int:
    batch = sys.argv[1] if len(sys.argv) > 1 else "1"
    tasks = BATCHES[batch]
    assert CKPT.exists(), f"anchor ckpt missing: {CKPT}"
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for task in tasks:
        b = BUDGETS[task]
        for path in ("cross", "dual"):
            cell = OUT / f"{task}_{path}_s0"
            ev = cell / "evidence"
            ev.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(WS / "scripts/transfer_eval.py"),
                   "--task", task, "--arm", "byte", "--rung", "tiny",
                   "--init", "pretrained", "--ckpt", str(CKPT),
                   "--path", path, "--steps", str(b["steps"]), "--batch", "16",
                   "--lr", "5e-4", "--seed", "0", "--split-seed", "0",
                   "--stratify", "--cross-shuffle", "off",
                   "--examples", str(b["examples"]),
                   "--test-examples", str(b["test"]),
                   "--device", "cuda",
                   "--evidence-dir", str(ev), "--out", str(cell / "result.json")]
            t0 = time.perf_counter()
            try:
                pr = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=PER_CELL_TIMEOUT)
                rc = pr.returncode
                tail = (pr.stdout + pr.stderr)[-600:]
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
            recs.append({"cell": f"{task}_{path}_s0", "rc": rc,
                         "wall_s": wall, "acc": acc, "budget": b,
                         "jev_recorded": JEV_RECORDED[task]})
            print(f"{task}_{path}: rc={rc} wall={wall} acc={acc} "
                  f"(jev recorded {JEV_RECORDED[task]})", flush=True)
    (OUT / f"SUMMARY_batch{batch}.json").write_text(json.dumps(recs, indent=1) + "\n")
    print(f"DONE batch {batch}: {len(recs)} cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
