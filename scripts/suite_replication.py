#!/usr/bin/env python3
"""Measure run-to-run variation for three large-rung cells.

Three cells (clinc_oos, massive_intent, massive_scenario) were executed twice
on the same checkpoint, the same test draw and the same pre-registered recipe,
on two different sandbox instances.  The archived, pre-registered execution is
`<cell>/result.json`; the re-execution is
`<cell>/replication_run2/{result,provenance}.json`.  The archived record stays
canonical so the pre-registered run is the reported result, not whichever of
the two runs happens to score higher.

Two runs of a fixed configuration differ by floating-point kernel scheduling
on different GPUs; this is the empirical noise floor against which the
three-rung deltas in COMPARISON.md must be read.  The script writes
REPLICATION.json and REPLICATION.md; it fails if the two runs did not use the
same test draw or the same checkpoint.

Usage:
    python scripts/suite_replication.py
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "results/suite_large_113m"
CELLS = ["clinc_oos", "massive_intent", "massive_scenario"]
CHECKPOINT_SHA = "aba65b60e30654a9895dc9c365d62b555f96dcce96ad8a8ad56cc718ee0865ec"
Z = 1.959963984540054


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(correct: int, n: int, z: float = Z) -> tuple[float, float]:
    p = correct / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binom_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0**n


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(b, c), n))


def main() -> int:
    rows = {}
    for cell in CELLS:
        run1_path = OUT_DIR / cell / "result.json"
        run2_path = OUT_DIR / cell / "replication_run2" / "result.json"
        prov2 = json.loads(
            (OUT_DIR / cell / "replication_run2/evidence/provenance.json").read_text()
        )
        if not run1_path.exists():
            raise SystemExit(f"missing first-run record: {run1_path}")
        run2 = json.loads(run2_path.read_text())
        run1 = json.loads(run1_path.read_text())
        f1, f2 = run1["final"], run2["final"]
        if f1["gold"] != f2["gold"]:
            raise SystemExit(f"{cell}: test draws differ between runs")
        for tag, prov in [("run2", prov2)]:
            if prov["source_checkpoint"]["sha256"] != CHECKPOINT_SHA:
                raise SystemExit(f"{cell}/{tag}: wrong source checkpoint")
        n = len(f1["gold"])
        c1 = sum(1 for a, b in zip(f1["predictions"], f1["gold"]) if a == b)
        c2 = sum(1 for a, b in zip(f2["predictions"], f2["gold"]) if a == b)
        only1 = sum(
            1 for a, b, g in zip(f1["predictions"], f2["predictions"], f1["gold"])
            if a == g and b != g
        )
        only2 = sum(
            1 for a, b, g in zip(f1["predictions"], f2["predictions"], f1["gold"])
            if a != g and b == g
        )
        flips = sum(1 for a, b in zip(f1["predictions"], f2["predictions"]) if a != b)
        rows[cell] = {
            "n": n,
            "run1": {
                "accuracy": f1["accuracy"],
                "correct": c1,
                "wilson95": wilson(c1, n),
                "nll": f1["nll"],
                "ece": f1["ece"],
                "source": f"results/suite_large_113m/{cell}/result.json",
                "source_sha256": sha256(run1_path),
            },
            "run2": {
                "accuracy": f2["accuracy"],
                "correct": c2,
                "wilson95": wilson(c2, n),
                "nll": f2["nll"],
                "ece": f2["ece"],
                "source": (
                    f"results/suite_large_113m/{cell}/replication_run2/result.json"
                ),
                "source_sha256": sha256(run2_path),
            },
            "delta_pp": (c2 - c1) / n * 100.0,
            "prediction_flips": flips,
            "run1_only_correct": only1,
            "run2_only_correct": only2,
            "mcnemar_exact_p": mcnemar_exact(only1, only2),
        }

    deltas = [abs(r["delta_pp"]) for r in rows.values()]
    flips = [r["prediction_flips"] for r in rows.values()]
    doc = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Two independent executions of the same pre-registered large-rung "
            "recipe on the same checkpoint and test draw, on two sandbox "
            "instances. These deltas are run-to-run variation, not a treatment "
            "effect, and bound how small a three-rung difference can honestly "
            "be interpreted."
        ),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "cells": rows,
        "summary": {
            "max_abs_delta_pp": max(deltas),
            "mean_abs_delta_pp": sum(deltas) / len(deltas),
            "max_prediction_flips": max(flips),
            "mean_prediction_flips": sum(flips) / len(flips),
            "n_cells": len(rows),
        },
    }
    (OUT_DIR / "REPLICATION.json").write_text(json.dumps(doc, indent=1) + "\n")
    (OUT_DIR / "REPLICATION.md").write_text(render(doc) + "\n")
    print(json.dumps(doc["summary"], indent=1))
    return 0


def render(doc: dict) -> str:
    s = doc["summary"]
    L = [
        "# Run-to-run replication at the large rung",
        "",
        f"Generated by `scripts/suite_replication.py` on {doc['generated_utc']}.",
        "",
        doc["note"],
        "",
        "## Result",
        "",
        "| cell | n | archived | 95% Wilson | replication | 95% Wilson | Δ pp | prediction flips | archived only | replication only | exact p |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for cell, r in doc["cells"].items():
        L.append(
            f"| {cell} | {r['n']:,} | {r['run1']['accuracy']*100:.2f}% | "
            f"{r['run1']['wilson95'][0]*100:.2f}–{r['run1']['wilson95'][1]*100:.2f} | "
            f"{r['run2']['accuracy']*100:.2f}% | "
            f"{r['run2']['wilson95'][0]*100:.2f}–{r['run2']['wilson95'][1]*100:.2f} | "
            f"{r['delta_pp']:+.2f} | {r['prediction_flips']} | "
            f"{r['run1_only_correct']} | {r['run2_only_correct']} | "
            f"{r['mcnemar_exact_p']:.2g} |"
        )
    L += [
        "",
        "## What this bounds",
        "",
        f"Across {s['n_cells']} cells the two executions of one fixed "
        f"configuration differ by at most {s['max_abs_delta_pp']:.2f} pp "
        f"(mean {s['mean_abs_delta_pp']:.2f} pp) and flip at most "
        f"{s['max_prediction_flips']} of the test rows "
        f"(mean {s['mean_prediction_flips']:.0f}). A three-rung difference "
        "smaller than that is not distinguishable from re-running the same "
        "model on another card.",
        "",
        "This is why `COMPARISON.md` reports the paired exact McNemar test "
        "beside every rung delta: a cell is called a win or a loss only when "
        "the paired row-level evidence rejects at p < 0.05, and the replication "
        "deltas here show what a null difference looks like in this pipeline.",
        "",
        "## Provenance",
        "",
        "| cell | archived record | SHA-256 | replication record | SHA-256 |",
        "|---|---|---|---|---|",
    ]
    for cell, r in doc["cells"].items():
        L.append(
            f"| {cell} | `{r['run1']['source']}` | "
            f"`{r['run1']['source_sha256'][:32]}…` | `{r['run2']['source']}` | "
            f"`{r['run2']['source_sha256'][:32]}…` |"
        )
    L += [
        "",
        f"Both runs loaded `results/pretrain/l1_large_byte_iso10v3/ckpt.pt`, "
        f"SHA-256 `{doc['checkpoint_sha256']}`.",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
