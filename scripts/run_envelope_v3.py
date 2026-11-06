"""Envelope-v3 bounded runs: see results/transfer/envelope_v3/PREREG_V3.json.

4 cells (emotion s1/s2 x cross/dual), identical recipe, per-cell timeout
180 s. SINGLE LAUNCH ONLY — do not re-run; the evidence guard refuses
overwrites and second-pass logs destroy first-run wall timings.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WS = Path("/marimo/azurite_ws")
CKPT = WS / "results/pretrain/l1_tiny_byte/ckpt.pt"
OUT = WS / "results/transfer/envelope_v3/cells"
CELLS = [
    ("emotion", "cross", 1),
    ("emotion", "dual", 1),
    ("emotion", "cross", 2),
    ("emotion", "dual", 2),
]
PER_CELL_TIMEOUT = 180


def main() -> int:
    assert CKPT.exists(), f"anchor ckpt missing: {CKPT}"
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for task, path, seed in CELLS:
        name = f"{task}_{path}_s{seed}"
        cell = OUT / name
        ev = cell / "evidence"
        ev.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(WS / "scripts/transfer_eval.py"),
               "--task", task, "--arm", "byte", "--rung", "tiny",
               "--init", "pretrained", "--ckpt", str(CKPT),
               "--path", path, "--steps", "378", "--batch", "16",
               "--lr", "5e-4", "--seed", str(seed), "--split-seed", "0",
               "--stratify", "--cross-shuffle", "off",
               "--examples", "2000", "--test-examples", "500",
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
        recs.append({"cell": name, "rc": rc, "wall_s": wall, "acc": acc})
        print(f"{name}: rc={rc} wall={wall} acc={acc}", flush=True)
    (OUT.parent / "SUMMARY_cells.json").write_text(json.dumps(recs, indent=1) + "\n")
    print(f"DONE {len(recs)} cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
