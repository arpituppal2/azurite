"""Parity gate against fixed Jev reference points. No Jev API calls live here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .battery import summarize, wilson

# Published Jev points used only as fixed reference values. Cells without a
# published point are marked "assumed" and are not presented as measurements.
JEV_REFERENCE = {
    "classification": {"value": 0.8053, "source": "published 8-cell mean (Banking77 78.8)"},
    "boolean": {"value": 0.8500, "source": "assumed; no published Jev BoolQ point"},
    "pairwise": {"value": 0.8500, "source": "assumed; no published Jev pairwise point"},
    "ordinal": {"value": 0.8000, "source": "assumed; no published Jev ordinal point"},
    "document": {"value": 0.8000, "source": "assumed; no published Jev document point"},
    "abstention": {"value": 0.8500, "source": "assumed; no published Jev abstention point"},
    "robustness": {"value": 0.8000, "source": "assumed; no published Jev robustness point"},
    "multilingual": {"value": 0.8600, "source": "published Jev XNLI"},
}

JEV_AGGREGATE = sum(
    JEV_REFERENCE[s]["value"] * n for s, n in
    {"classification": 2000, "boolean": 1500, "pairwise": 1000, "ordinal": 1000,
     "document": 2000, "abstention": 1000, "robustness": 1000, "multilingual": 500}.items()
) / 10_000

MARGIN = 0.03
SLICE_FLOOR = 0.05
MIN_SLICES = 6


def _metric(block: dict, key: str, default: float) -> float:
    """Numeric metric, treating only an explicit JSON null as unmeasured.

    `block.get(key, default)` returns None for a stored null, and `x or default`
    would turn a legitimate 0 (0 ms latency, 0 MB swap) into a failure. This
    keeps 0 and defaults only on null/missing.
    """
    value = block.get(key)
    return default if value is None else float(value)


def gate(
    summary: dict,
    *,
    selector_recall: float | None = None,
    evidence_precision: float | None = None,
    context: dict | None = None,
    m1: dict | None = None,
    regression: dict | None = None,
) -> dict:
    failures: list[str] = []
    lo, _ = summary["wilson95"]
    if lo <= JEV_AGGREGATE + MARGIN:
        failures.append(
            f"aggregate lower bound {lo:.4f} <= Jev {JEV_AGGREGATE:.4f} + {MARGIN:.2f}"
        )
    at_or_above = 0
    for name, ref in JEV_REFERENCE.items():
        cell = summary["slices"].get(name)
        if cell is None:
            failures.append(f"missing slice {name}")
            continue
        if cell["accuracy"] >= ref["value"]:
            at_or_above += 1
        if cell["accuracy"] < ref["value"] - SLICE_FLOOR:
            failures.append(
                f"slice {name} {cell['accuracy']:.4f} below Jev {ref['value']:.4f} - {SLICE_FLOOR:.2f}"
            )
    if at_or_above < MIN_SLICES:
        failures.append(f"only {at_or_above}/8 slices at or above Jev; need {MIN_SLICES}")
    if selector_recall is None or selector_recall < 0.90:
        failures.append(f"selector recall {selector_recall} < 0.90")
    if evidence_precision is None or evidence_precision < 0.80:
        failures.append(f"evidence precision {evidence_precision} < 0.80")
    if context is None:
        failures.append("context metrics missing")
    else:
        if _metric(context, "niah_32k", 0.0) < 0.95:
            failures.append("32k needle-in-a-haystack < 0.95")
        if _metric(context, "multi_hop", 0.0) < 0.80:
            failures.append("multi-hop < 0.80")
        if _metric(context, "worst_position_gap_pp", 99.0) > 5.0:
            failures.append("worst position gap > 5 pp")
    if m1 is None:
        failures.append("M1 profile missing")
    else:
        if _metric(m1, "p95_ms", 1e9) > 5_000:
            failures.append("M1 p95 > 5000 ms")
        if _metric(m1, "peak_rss_gib", 1e9) > 5.0:
            failures.append("M1 RSS > 5 GiB")
        if _metric(m1, "swap_delta_mb", 1e9) > 16:
            failures.append("M1 swap delta > 16 MB")
    if regression is None:
        failures.append("Banking77 regression missing")
    elif _metric(regression, "accuracy", 0.0) < 0.8906 - 0.005:
        failures.append("Banking77 regression below 89.06 - 0.50 pp")
    return {
        "passed": not failures,
        "failures": failures,
        "jev_reference": JEV_REFERENCE,
        "jev_aggregate": JEV_AGGREGATE,
        "margin": MARGIN,
        "slice_floor": SLICE_FLOOR,
        "selector_recall": selector_recall,
        "evidence_precision": evidence_precision,
    }


def write_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
