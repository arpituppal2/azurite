#!/usr/bin/env python3
"""External baseline: pplx-decider-flash-v1.2 through its own decision API.

Runs the competitor's intended inference path (`pplx_decider.Decider`), not a
text-generation shim, on the SAME held-out rows and candidate labels used by the
Azurite/ModernBERT funnel cells, so the numbers are directly comparable in
task terms.

Comparability caveat, stated at the top because it decides how to read the
result: this is a ZERO-SHOT evaluation. The models it is compared against were
fine-tuned on 2,000 training rows of the same task. The competitor is not
adapted at all — it is being asked to decide with only the option descriptions
in the prompt. That is the honest end-product comparison (what you get if you
buy it), but it is not an adaptation-matched one, and it must never be reported
as if it were.

Usage:
    python scripts/run_pplx_baseline.py --task banking77 --split-seed 123 \\
        --test-examples 1000 --out results/transfer/envelope_v12/pplx_banking77_s123.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.decisions import TASKS, candidate_bank, stratified_examples  # noqa: E402

CHECKPOINT = ("/home/marimo/.cache/huggingface/hub/models--perplexity-ai--"
              "pplx-decider-flash-v1.2/snapshots/9ff7cf9838e59df461c276b7699d56b244d2af58")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="banking77")
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--test-examples", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (smoke tests)")
    ap.add_argument("--ckpt", default=CHECKPOINT)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.ckpt) / "source" / "src"))
    from pplx_decider import Decider

    spec = TASKS[args.task]
    bank = candidate_bank(spec)
    rows = stratified_examples(spec, bank, "test", args.test_examples,
                               args.split_seed + 1)
    if args.limit:
        rows = rows[: args.limit]
    criteria = {label: None for label in bank.labels}

    decider = Decider.from_pretrained(Path(args.ckpt), device="cuda")
    print(f"loaded decider; task={args.task} K={len(bank)} n={len(rows)}", flush=True)

    latencies: list[float] = []
    preds: list[int] = []
    confidences: list[float] = []
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        reqs = [{"state": r.context,
                 "questions": {"q": {"type": "choice",
                                     "instructions": spec.instruction,
                                     "criteria": criteria}}}
                for r in chunk]
        t0 = time.perf_counter()
        out = decider.decide(reqs)
        if isinstance(out, dict):
            out = [out]
        latencies.append((time.perf_counter() - t0) / len(chunk))
        for r, resp in zip(chunk, out):
            ans = resp["answers"]["q"]
            choice = ans["choice"]
            preds.append(bank.labels.index(choice))
            confidences.append(float(ans.get("confidence", 0.0)))
        done = min(len(rows), start + args.batch)
        if start % (args.batch * 10) == 0:
            acc = sum(1 for p, r in zip(preds, rows) if p == r.gold) / len(preds)
            print(f"  {done}/{len(rows)}  running acc {acc:.4f}", flush=True)

    gold = [r.gold for r in rows]
    correct = sum(1 for p, g in zip(preds, gold) if p == g)
    acc = correct / len(gold)
    from math import sqrt
    z = 1.96
    n = len(gold)
    d = 1 + z * z / n
    c = acc + z * z / (2 * n)
    s = z * sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))
    lb = (c - s) / d
    result = {
        "task": args.task, "model": "perplexity-ai/pplx-decider-flash-v1.2",
        "api": "pplx_decider.Decider (its own decision inference path)",
        "adaptation": "ZERO-SHOT (no fine-tuning on this task; competitors were fine-tuned on 2000 rows)",
        "split_seed": args.split_seed, "test_split_seed": args.split_seed + 1,
        "n": n, "k": len(bank),
        "accuracy": round(acc, 4),
        "wilson_lb": round(lb, 4),
        "mean_confidence": round(statistics.fmean(confidences), 4),
        "latency_s_per_request_mean": round(statistics.fmean(latencies), 4),
        "latency_s_per_request_p50": round(statistics.median(latencies), 4),
        "predictions": preds, "gold": gold,
    }
    out = Path(args.out) if args.out else (PROJECT / "results" / "transfer" /
                                           "envelope_v12" /
                                           f"pplx_{args.task}_s{args.split_seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"accuracy {acc:.4f}  Wilson LB {lb:.4f}  n={n}  "
          f"{result['latency_s_per_request_mean']:.3f}s/request", flush=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
