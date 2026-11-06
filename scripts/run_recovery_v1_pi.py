"""Recovery v1 preserved-win checks: re-run prompt_injections cross+dual s0.

Identical CLI recipe to superiority_v1 cells (task prompt_injections, arm
byte, rung tiny, init pretrained, steps 378, batch 16, lr 5e-4, seed 0,
split-seed 0, stratify, cross-shuffle off, examples 2000, test-examples 500).
Win preserved iff Wilson LB (95%%) of rerun acc still exceeds recorded Jev
point 0.741. Per-cell timeout 180 s.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WS = Path("/marimo/azurite_ws")
CKPT = WS / "results/pretrain/l1_tiny_byte/ckpt.pt"
OUT = WS / "results/transfer/recovery_v1/preserved_wins"
CELLS = ["cross", "dual"]
PER_CELL_TIMEOUT = 180


def main() -> int:
    assert CKPT.exists(), f"anchor ckpt missing: {CKPT}"
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for path in CELLS:
        name = f"pi_{path}_s0_check"
        cell = OUT / name
        ev = cell / "evidence"
        ev.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(WS / "scripts/transfer_eval.py"),
               "--task", "prompt_injections", "--arm", "byte", "--rung", "tiny",
               "--init", "pretrained", "--ckpt", str(CKPT),
               "--path", path, "--steps", "378", "--batch", "16",
               "--lr", "5e-4", "--seed", "0", "--split-seed", "0",
               "--stratify", "--cross-shuffle", "off",
               "--examples", "2000", "--test-examples", "500",
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
