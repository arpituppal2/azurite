#!/usr/bin/env python3
"""Annotate comparator points with their evidence class.

The registry previously used `comparable: false` for every point, which is
correct but does not say *why*. This pass adds the distinction the mission now
requires:

  published_reference  a number published by a vendor or third party, not rerun
  local_reconstruction derived from stored public per-example outputs, not a
                       live model call
  reproduced_live      same rows, same pinned model version, live call; none
                       exists in this checkout

The annotation is additive. No value, n, source, or note is changed.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "research" / "benchmark_registry.json"


def datasets(family: dict):
    for dataset in family.get("datasets", []):
        yield dataset
    for sub in family.get("subfamilies", []):
        for dataset in sub.get("datasets", []):
            yield dataset


def classify(point: dict) -> str:
    source = str(point.get("source", "")).lower()
    version = str(point.get("version", "")).lower()
    if "local reconstruction" in source:
        return "local_reconstruction"
    if version.startswith("reconstructed"):
        return "local_reconstruction"
    return "published_reference"


def main() -> int:
    registry = json.loads(PATH.read_text())
    counts = {"published_reference": 0, "local_reconstruction": 0, "reproduced_live": 0}
    classes: dict[str, list[dict]] = {key: [] for key in counts}
    for family in registry["families"]:
        for dataset in datasets(family):
            for point in dataset.get("comparator", {}).get("points", []):
                klass = classify(point)
                point["comparison_class"] = klass
                point["reproduced"] = False
                counts[klass] += 1
                classes[klass].append({
                    "family": family["id"],
                    "dataset": dataset["id"],
                    "value": point["value"],
                    "n": point["n"],
                    "source": point["source"],
                    "version": point["version"],
                })
    registry["comparator_classes"] = {
        "definitions": {
            "published_reference": (
                "A vendor or third-party published number. It names a source, "
                "version, split, and n; it is not a row produced by this study."
            ),
            "local_reconstruction": (
                "A stored per-example comparator vector reconstructed locally. "
                "It is not a live model call and is not a reproduced run."
            ),
            "reproduced_live": (
                "Same rows, same pinned model version, same frozen request, live "
                "call, with request and response hashes. None exists here."
            ),
        },
        "counts": counts,
        "live_reproduction_claim": (
            "Zero. No point in this registry is a live reproduced Jev run. "
            "The Banking77 0.924 cell is a local reconstruction, not a run."
        ),
        "points": classes,
    }
    PATH.write_text(json.dumps(registry, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print("wrote", PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
