#!/usr/bin/env python3
"""Majority-class accuracy per task, from the frozen test splits.

Every decision number in this project is only meaningful next to the prior it has
to beat, and for an unbalanced task the prior is not 1/K: `financial_phrasebank`
is 61.2% on three options, `banking77` is 1.3% on seventy-seven. Computed from the
same `load_examples` the training path uses, so the split being scored is the same
one the prior is measured on.

Usage:
    .venv/bin/python3 scripts/task_priors.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.decisions import TASKS, candidate_bank, load_examples  # noqa: E402
from azurite.paths import rel  # noqa: E402

OUT = PROJECT / "results" / "transfer" / "PRIORS.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100_000)
    args = ap.parse_args()

    out = {}
    print(f"{'task':24s} {'K':>4s} {'test rows':>9s} {'majority':>9s} {'1/K':>7s}")
    for name in sorted(TASKS):
        spec = TASKS[name]
        bank = candidate_bank(spec)
        rows = load_examples(spec, bank, split="test", limit=args.limit)
        if not rows:
            # Printed, never skipped silently: a task missing from this table would
            # otherwise read as a task that does not exist rather than one whose
            # split could not be loaded.
            print(f"{name:24s} {len(bank):>4d} {'—':>9s} {'no test split':>9s}")
            out[name] = {"k": len(bank), "test_rows": 0, "error": "no test split"}
            continue
        counts = Counter(r.gold for r in rows)
        majority = max(counts.values()) / len(rows)
        out[name] = {
            "k": len(bank),
            "test_rows": len(rows),
            "majority_accuracy": round(majority, 4),
            "uniform_accuracy": round(1 / len(bank), 4),
        }
        print(
            f"{name:24s} {len(bank):>4d} {len(rows):>9d} {majority:>8.1%} "
            f"{1 / len(bank):>6.1%}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
