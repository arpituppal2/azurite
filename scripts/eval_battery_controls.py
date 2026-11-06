#!/usr/bin/env python3
"""Control baselines for the frozen 10k battery. No model, no GPU.

These are floors, not results: the most frequent gold label per family and the
mean uniform-guess rate over the alternatives actually offered. Task 9's model
numbers are only meaningful next to these, and running them proves the harness
reads and scores every row of the frozen battery.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.battery import read_eval_battery, summarize, wilson  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="results/parity/eval_battery_10k")
    ap.add_argument("--out", default="results/parity/eval_battery_controls.json")
    args = ap.parse_args()

    rows, manifest = read_eval_battery(args.battery)
    by_family: dict[str, list] = {}
    for row in rows:
        by_family.setdefault(row.slice, []).append(row)

    families: dict[str, dict] = {}
    records: list[dict] = []
    for family, items in sorted(by_family.items()):
        majority = Counter(r.gold for r in items).most_common(1)[0][0]
        correct = sum(1 for r in items if r.gold == majority)
        chance = sum(1 / len(r.alternatives) for r in items) / len(items)
        lo, hi = wilson(correct, len(items))
        families[family] = {
            "n": len(items),
            "majority_class": majority,
            "majority_accuracy": correct / len(items),
            "majority_wilson95": [lo, hi],
            "uniform_chance": chance,
            "k_median": sorted(len(r.alternatives) for r in items)[len(items) // 2],
        }
        for r in items:
            records.append({"id": r.id, "slice": family, "gold": r.gold, "choice": majority, "abstained": False})

    summary = summarize(records)
    report = {
        "battery": str(args.battery),
        "battery_sha256": manifest.get("battery_sha256"),
        "kind": "control_baseline",
        "note": "majority-class and uniform-guess floors; not a model result and never a parity number",
        "rows": len(rows),
        "families": families,
        "aggregate": {
            "n": summary["n"],
            "majority_accuracy": summary["accuracy"],
            "wilson95": summary["wilson95"],
            "uniform_chance": sum(f["uniform_chance"] * f["n"] for f in families.values()) / len(rows),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=1))
    print(json.dumps({f: {"majority": round(v["majority_accuracy"], 4), "chance": round(v["uniform_chance"], 4)}
                      for f, v in families.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
