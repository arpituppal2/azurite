#!/usr/bin/env python3
"""Aggregate the E4 per-width runs into a summary, decision and manifest.

Reads only stored result records.  Verifies that every run names the same
pre-registration hash and the same source checkpoint, recomputes each accuracy
from its own per-example correctness bits, and applies the pre-registered
non-inferiority rule:

    holds_task = one-sided 95% bootstrap lower bound of (acc32 - accK) > -5.0 pp

The bootstrap is paired by test row (same rows, same gold), 10,000 resamples,
seed 0, percentile.  Exact McNemar tests are reported for the secondary
comparisons.

Usage:
    python3 scripts/e4_summary.py --root results/e4_candidate_sweep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.stats import bootstrap_lower, mcnemar_exact, wilson  # noqa: E402

TASKS = ("banking77", "clinc_oos", "massive_intent")
WIDTHS = {"banking77": [2, 8, 32, 77], "clinc_oos": [2, 8, 32, 151],
          "massive_intent": [2, 8, 32, 60]}
PUBLISHED = {"banking77": 0.8845, "clinc_oos": 0.8850, "massive_intent": 0.8200}
MARGIN = -5.0
BOOT = 10_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits(hexstr: str, n: int) -> np.ndarray:
    raw = bytes.fromhex(hexstr)
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        out[i] = (raw[i // 8] >> (i % 8)) & 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/e4_candidate_sweep")
    args = ap.parse_args()
    root = Path(args.root)
    prereg = root / "PRE_REGISTRATION.md"
    if not prereg.exists():
        raise SystemExit("missing PRE_REGISTRATION.md")
    prereg_sha = sha256(prereg)

    cells: dict[str, dict[int, dict]] = {}
    ckpts = set()
    for task in TASKS:
        cells[task] = {}
        for w in WIDTHS[task]:
            rp = root / task / f"w{w}" / "result.json"
            if not rp.exists():
                raise SystemExit(f"missing {rp}")
            rec = json.loads(rp.read_text())
            if rec.get("prereg_sha256") not in (None, prereg_sha):
                raise SystemExit(f"{rp}: pre-registration hash mismatch")
            if rec["protocol"] != "E4-width-training" or rec["width"] != w:
                raise SystemExit(f"{rp}: wrong protocol or width")
            if rec["task"] != task:
                raise SystemExit(f"{rp}: wrong task")
            ckpts.add(rec["source_checkpoint"]["sha256"])
            correct = bits(rec["correct_bits_hex"], rec["n"])
            if int(correct.sum()) != int(rec["correct"]):
                raise SystemExit(f"{rp}: correctness bits do not sum to recorded correct")
            if not math.isclose(correct.mean(), rec["accuracy"], abs_tol=1e-12):
                raise SystemExit(f"{rp}: accuracy does not recompute")
            pe = rp.parent / "per_example.jsonl"
            if not pe.exists():
                raise SystemExit(f"missing {pe} (run scripts/e4_per_example.py first)")
            if len(pe.read_text().splitlines()) != rec["n"]:
                raise SystemExit(f"{pe}: line count does not match n={rec['n']}")
            cells[task][w] = {"record": rec, "correct": correct,
                              "path": str(rp), "sha256": sha256(rp),
                              "per_example_sha256": sha256(pe)}
    if len(ckpts) != 1:
        raise SystemExit(f"runs used different checkpoints: {ckpts}")

    summary = {"protocol": "E4", "prereg_sha256": prereg_sha,
               "margin_pp": MARGIN, "bootstrap": {"n": BOOT, "seed": 0},
               "checkpoint_sha256": ckpts.pop(), "tasks": {}, "generated_utc": None}
    holds = 0
    for task in TASKS:
        k = WIDTHS[task][-1]
        kv = cells[task][k]
        acc_k = float(kv["correct"].mean())
        task_out = {"k": k, "published_k_accuracy": PUBLISHED[task],
                    "widths": {}, "primary": {}}
        for w in WIDTHS[task]:
            c = cells[task][w]
            kk = int(c["correct"].sum())
            lo, hi = wilson(kk, len(c["correct"]))
            task_out["widths"][str(w)] = {
                "accuracy": float(kk / len(c["correct"])),
                "correct": kk,
                "n": int(len(c["correct"])),
                "wilson95": [lo, hi],
                "nll": c["record"]["nll"],
                "chance": 1.0 / w,
                "train_seconds": c["record"]["train_seconds"],
                "full_k_accuracy": c["record"]["full_k"]["accuracy"],
                "result_sha256": c["sha256"],
                "per_example_sha256": c["per_example_sha256"],
                "path": c["path"],
            }
        v32 = cells[task][32]["correct"]
        diff = float(v32.mean() - acc_k)
        lower = bootstrap_lower(v32, kv["correct"])
        mc = mcnemar_exact(v32, kv["correct"])
        mc2 = mcnemar_exact(cells[task][2]["correct"], kv["correct"])
        mc8 = mcnemar_exact(cells[task][8]["correct"], cells[task][32]["correct"])
        verdict = "holds" if lower > MARGIN else "falsified"
        holds += int(verdict == "holds")
        task_out["primary"] = {
            "width": 32, "acc32": float(v32.mean()), "accK": acc_k,
            "delta_pp": 100 * diff,
            "lower95_pp": 100 * lower,
            "mcnemar": mc,
            "verdict": verdict,
        }
        task_out["secondary"] = {
            "width2_vs_K": {"delta_pp": 100 * float(cells[task][2]["correct"].mean() - acc_k),
                            "mcnemar": mc2},
            "width8_vs_32": {"delta_pp": 100 * float(cells[task][8]["correct"].mean()
                                                     - v32.mean()),
                             "mcnemar": mc8},
        }
        task_out["harness"] = {
            "accK": acc_k,
            "published": PUBLISHED[task],
            "abs_delta_pp": 100 * abs(acc_k - PUBLISHED[task]),
            "reproduces_within_0p2pp": abs(acc_k - PUBLISHED[task]) <= 0.002,
        }
        summary["tasks"][task] = task_out

    summary["task_count"] = len(TASKS)
    summary["holds_count"] = holds
    summary["aggregate_verdict"] = "holds" if holds == len(TASKS) else "falsified"
    import datetime
    summary["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    (root / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = ["# E4 — the candidate-set sweep", "",
             f"Pre-registration SHA-256: `{prereg_sha}`  ",
             f"Checkpoint SHA-256: `{summary['checkpoint_sha256']}`  ",
             f"Decision rule: accuracy at |C|=32 within {abs(MARGIN):.1f} points of "
             f"|C|=K (paired bootstrap, one-sided 95% lower bound).", "",
             "| task | K | acc@2 | acc@8 | acc@32 | acc@K | Δ(32−K) pp | lower95 pp | McNemar p | verdict |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for task in TASKS:
        t = summary["tasks"][task]
        wd = t["widths"]
        p = t["primary"]
        lines.append(
            f"| {task} | {t['k']} | {100*wd['2']['accuracy']:.2f} | "
            f"{100*wd['8']['accuracy']:.2f} | {100*wd['32']['accuracy']:.2f} | "
            f"{100*wd[str(t['k'])]['accuracy']:.2f} | {p['delta_pp']:+.2f} | "
            f"{p['lower95_pp']:+.2f} | {p['mcnemar']['p']:.2e} | {p['verdict']} |")
    lines += ["", f"**Aggregate:** {summary['aggregate_verdict']} "
                  f"({holds}/{len(TASKS)} tasks hold).", "",
              "Every arm has a `per_example.jsonl` with `(i, gold, pred, correct)`; "
              "`scripts/e4_per_example.py` rebuilds the deterministic test split, "
              "checks its label list against `test_gold_sha256` and checks the "
              "correctness sum against `result.json` before writing it. The "
              "`gold_bits_hex` field in `result.json` is a legacy parity checksum "
              "(one bit per gold label), not a label vector; it is not used for "
              "anything and the label list is pinned by the digest instead.", "",
              "Harness check (width-K arm vs published suite accuracy):"]

    for task in TASKS:
        h = summary["tasks"][task]["harness"]
        lines.append(f"- {task}: trained {100*h['accK']:.2f}% vs published "
                     f"{100*h['published']:.2f}% (|Δ| {h['abs_delta_pp']:.2f} pp, "
                     f"{'ok' if h['reproduces_within_0p2pp'] else 'DISCREPANCY'})")
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({t: summary["tasks"][t]["primary"] for t in TASKS}, indent=1))
    print("aggregate", summary["aggregate_verdict"], f"{holds}/{len(TASKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
