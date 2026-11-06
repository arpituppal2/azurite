#!/usr/bin/env python3
"""Summarize the full-split Jev-overlap runs into one comparable table.

Each task directory holds the unchanged `transfer_eval.py` record: per-example
canonical gold and prediction vectors, the final metric block, and the exact
command line. This script reads those records, recomputes the headline metric
from the retained predictions, and prints the gap to the cited Jev point.

The Jev points are published cross-harness numbers, not paired runs. A positive
gap is a point above a cited number; the Wilson interval describes test-row
sampling only, not checkpoint or seed variance.

Usage:
    .venv/bin/python scripts/compare_jev_full.py results/jev_full_20261030 ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_jev_overlap import JEV

ROOT = Path(__file__).resolve().parent.parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def f1(gold: list[int], pred: list[int], positive: int = 1) -> float:
    tp = sum(1 for g, p in zip(gold, pred) if g == positive and p == positive)
    fp = sum(1 for g, p in zip(gold, pred) if g != positive and p == positive)
    fn = sum(1 for g, p in zip(gold, pred) if g == positive and p != positive)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def metric_for(task: str, payload: dict) -> tuple[str, float, tuple[float, float] | None]:
    final = payload["final"]
    gold = final.get("canonical_gold") or final.get("gold") or []
    pred = final.get("canonical_predictions") or final.get("predictions") or []
    n = len(gold)
    jev = JEV.get(task, {})
    metric = jev.get("metric", "accuracy")
    if metric.lower() == "f1":
        value = f1(gold, pred)
        return ("F1", value, None)
    if metric.lower() == "macro-f1":
        labels = sorted(set(gold) | set(pred))
        value = sum(f1(gold, pred, positive=c) for c in labels) / len(labels)
        return ("macro-F1", value, None)
    k = sum(1 for g, p in zip(gold, pred) if g == p)
    value = k / n if n else float("nan")
    return ("accuracy", value, wilson(k, n))


def verdict(value: float, jev: float, ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "above" if value >= jev else "below"
    if ci[0] > jev:
        return "clear above"
    if ci[1] < jev:
        return "clear below"
    return "tie (interval covers point)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    rows = []
    seen = set()
    for d in args.dirs:
        root = Path(d)
        if not root.exists():
            continue
        for cell in sorted(root.iterdir()):
            result = cell / "result.json"
            if not result.is_file() or cell.name in seen:
                continue
            payload = json.loads(result.read_text())
            task = payload.get("task", cell.name)
            if task in seen:
                continue
            seen.add(task)
            metric, value, ci = metric_for(task, payload)
            jev = JEV.get(task, {})
            jev_score = jev.get("score")
            gap = None if jev_score is None else 100 * (value - jev_score)
            rows.append({
                "task": task,
                "n": len(payload["final"].get("gold") or []),
                "metric": metric,
                "value": round(value, 4),
                "ci": [round(x, 4) for x in ci] if ci else None,
                "jev": jev_score,
                "gap_pp": None if gap is None else round(gap, 2),
                "verdict": verdict(value, jev_score, ci) if jev_score is not None else "no cited point",
                "train": payload.get("train_examples"),
                "steps": payload.get("steps"),
                "max_seq": payload.get("evidence", {}).get("pack", {}).get("max_seq"),
                "seconds": payload.get("seconds"),
                "result": str(result),
            })

    if not rows:
        print("no completed results")
        return 1

    order = {k: i for i, k in enumerate(JEV)}
    rows.sort(key=lambda r: order.get(r["task"], 999))
    print("| task | n | metric | Azurite | 95% Wilson | Jev | gap pp | verdict | train | steps |")
    print("|---|---:|---|---:|---|---:|---:|---|---:|---:|")
    for r in rows:
        ci = f"{r['ci'][0]*100:.2f}-{r['ci'][1]*100:.2f}" if r["ci"] else "—"
        gap = f"{r['gap_pp']:+.2f}" if r["gap_pp"] is not None else "—"
        jev = f"{r['jev']*100:.1f}" if r["jev"] is not None else "—"
        print(f"| {r['task']} | {r['n']} | {r['metric']} | {r['value']*100:.2f} | "
              f"{ci} | {jev} | {gap} | {r['verdict']} | {r['train']} | {r['steps']} |")

    wins = sum(1 for r in rows if r["verdict"] in {"clear above", "above"})
    ties = sum(1 for r in rows if r["verdict"].startswith("tie"))
    losses = sum(1 for r in rows if r["verdict"] in {"clear below", "below"})
    print(f"\n{wins} above, {ties} ties, {losses} below, {len(rows)} completed")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
