"""The frozen 10k parity battery: schema, validation, and summaries.

A battery row is deliberately model-agnostic. The same row can be scored by
Azurite, a lexical baseline, or a live Jev call later, without changing labels.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

SLICE_QUOTAS = {
    "classification": 2_000,
    "boolean": 1_500,
    "pairwise": 1_000,
    "ordinal": 1_000,
    "document": 2_000,
    "abstention": 1_000,
    "robustness": 1_000,
    "multilingual": 500,
}
TOTAL = sum(SLICE_QUOTAS.values())


@dataclass
class BatteryRow:
    id: str
    slice: str
    context: str
    question: str
    alternatives: list[str]
    gold: int
    source: str
    licence: str = "unresolved"
    gold_evidence: tuple[int, int] | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["gold_evidence"] = list(self.gold_evidence) if self.gold_evidence else None
        return out


def validate(rows: Sequence[BatteryRow], *, expected_total: int = TOTAL) -> dict:
    errors: list[str] = []
    ids: set[str] = set()
    counts = {k: 0 for k in SLICE_QUOTAS}
    for i, row in enumerate(rows):
        if row.id in ids:
            errors.append(f"row {i}: duplicate id {row.id!r}")
        ids.add(row.id)
        if row.slice not in counts:
            errors.append(f"row {i}: unknown slice {row.slice!r}")
            continue
        counts[row.slice] += 1
        if not row.context.strip():
            errors.append(f"row {i}: empty context")
        if not row.alternatives:
            errors.append(f"row {i}: no alternatives")
        if not 0 <= row.gold < len(row.alternatives):
            errors.append(f"row {i}: gold {row.gold} outside {len(row.alternatives)} alternatives")
        if len(set(row.alternatives)) != len(row.alternatives):
            errors.append(f"row {i}: duplicate alternatives")
    if len(rows) != expected_total:
        errors.append(f"expected {expected_total} rows, got {len(rows)}")
    for slice_name, quota in SLICE_QUOTAS.items():
        if counts.get(slice_name, 0) != quota:
            errors.append(f"slice {slice_name}: expected {quota}, got {counts.get(slice_name, 0)}")
    return {"ok": not errors, "errors": errors, "counts": counts, "n": len(rows)}


def write_battery(rows: Sequence[BatteryRow], path: str | Path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "total": len(rows),
        "quotas": SLICE_QUOTAS,
        "rows": [r.to_dict() for r in rows],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=9) as fh:
        fh.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    manifest = {"path": str(path), "rows": len(rows), "sha256": digest, "bytes": len(raw)}
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    return manifest


def read_battery(path: str | Path) -> tuple[list[BatteryRow], dict]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = [
        BatteryRow(
            id=str(r["id"]), slice=str(r["slice"]), context=str(r["context"]),
            question=str(r["question"]), alternatives=list(r["alternatives"]),
            gold=int(r["gold"]), source=str(r.get("source", "")),
            licence=str(r.get("licence", "unresolved")),
            gold_evidence=tuple(r["gold_evidence"]) if r.get("gold_evidence") else None,
            meta=dict(r.get("meta", {})),
        )
        for r in payload["rows"]
    ]
    return rows, payload


def read_eval_battery(path: str | Path) -> tuple[list[BatteryRow], dict]:
    """Load a frozen evaluation battery from a sharded directory or a single file.

    The 10k battery stores one JSON row per line across `jsonl.gz` shards with a
    merged manifest; the earlier battery stored one gzip'd JSON payload. Both
    map onto the same BatteryRow so scoring code does not care which it got.
    """
    p = Path(path)
    if p.is_file():
        return read_battery(p)
    manifest = json.loads((p / "manifest.json").read_text())
    rows: list[BatteryRow] = []
    for shard in sorted(manifest["shards"], key=lambda s: s["index"]):
        with gzip.open(p / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                spans = r.get("gold_spans") or []
                evidence = (int(spans[0]["start"]), int(spans[0]["end"])) if spans else None
                provenance = r.get("provenance") or {}
                rows.append(BatteryRow(
                    id=str(r["id"]), slice=str(r["family"]), context=str(r["context"]),
                    question=str(r["question"]), alternatives=list(r["alternatives"]),
                    gold=int(r["gold"]), source=str(provenance.get("dataset", "")),
                    licence=str(r.get("licence", "unresolved")),
                    gold_evidence=evidence,
                    meta={
                        **dict(r.get("meta") or {}),
                        "family": r["family"], "task": r["task"],
                        "gold_label": r["gold_label"], "answer": r.get("answer"),
                        "provenance": provenance, "gold_spans": spans,
                    },
                ))
    if len(rows) != int(manifest.get("rows", len(rows))):
        raise ValueError(f"manifest says {manifest.get('rows')} rows, read {len(rows)}")
    return rows, manifest


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(records: Iterable[dict]) -> dict:
    rows = list(records)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r["slice"]), []).append(r)
    slices = {}
    for name, items in sorted(by.items()):
        n = len(items)
        correct = sum(1 for r in items if not r.get("abstained") and int(r.get("choice", -1)) == int(r["gold"]))
        answered = sum(1 for r in items if not r.get("abstained"))
        lo, hi = wilson(correct, n)
        slices[name] = {
            "n": n, "correct": correct, "accuracy": correct / n if n else 0.0,
            "wilson95": [lo, hi], "answered": answered,
            "coverage": answered / n if n else 0.0,
            "selective_accuracy": correct / answered if answered else 0.0,
            "mean_latency_ms": sum(float(r.get("latency_ms", 0.0)) for r in items) / n if n else 0.0,
        }
    n = len(rows)
    correct = sum(1 for r in rows if not r.get("abstained") and int(r.get("choice", -1)) == int(r["gold"]))
    lo, hi = wilson(correct, n)
    return {
        "n": n, "correct": correct, "accuracy": correct / n if n else 0.0,
        "wilson95": [lo, hi], "slices": slices,
        "answered": sum(1 for r in rows if not r.get("abstained")),
    }
