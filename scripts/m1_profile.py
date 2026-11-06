#!/usr/bin/env python3
"""Profile Select+Read latency/RSS/swap. Inference only; no training."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.env import pageouts, swap_used_mb  # noqa: E402
from azurite.select_read import SelectRead  # noqa: E402


def rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / 2**30
    return value / 2**20


def machine_name() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
        return out or platform.processor() or "macOS"
    return platform.processor() or platform.machine()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizers/azurite_bpe_32004.json")
    ap.add_argument("--selector", default=None)
    ap.add_argument("--head", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--lengths", default="1024,4096,16384,32768")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/parity/M1_PROFILE.json")
    args = ap.parse_args()
    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    engine = SelectRead.from_checkpoint(
        args.checkpoint, tokenizer_path=args.tokenizer,
        selector_path=args.selector, head_path=args.head, device=device,
    )
    swap0, page0 = swap_used_mb(), pageouts()
    records = []
    for length in [int(x) for x in args.lengths.split(",") if x]:
        # Approximate token length with repeated words; the profile is about
        # scaling, not content. The selector chunking sees the same text.
        text = ("evidence " * max(1, length // 2))[: max(1, length * 2)]
        latencies = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            engine.decide(text, "Which alternative is supported?", ["yes", "no"])
            latencies.append((time.perf_counter() - t0) * 1000.0)
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        records.append({"tokens": length, "p50_ms": round(p50, 1), "p95_ms": round(p95, 1),
                        "runs": len(latencies)})
    report = {
        "machine": machine_name(),
        "device": device,
        "torch": torch.__version__,
        "peak_rss_gib": round(rss_gib(), 2),
        "swap_delta_mb": round(swap_used_mb() - swap0, 1),
        "pageout_delta": pageouts() - page0,
        "lengths": records,
        "gate_eligible": "M1" in machine_name(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
