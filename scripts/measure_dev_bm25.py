#!/usr/bin/env python3
"""Re-measure BM25 recall@8 on the frozen dev packs, with no model involved.

Writes both the summary and a per-row record so the gate number can be audited
row by row. Scoring is identical to `score_pack` in `scripts/build_long_context.py`:
BM25 over 64 chunks, top 8, any-overlap is the gate metric and strict first-chunk
recall is reported alongside it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from azurite.selector import chunk_text  # noqa: E402
from build_long_context import BM25ChunkSelector, TOP_K  # noqa: E402

BATTERY = PROJECT / "results/parity/long_context_dev_noquestion"
OUT = PROJECT / "results/parity/bm25_dev_remeasure.json"
ROWS_OUT = PROJECT / "results/parity/bm25_dev_rows.jsonl.gz"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=str(BATTERY))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rows-out", default=str(ROWS_OUT))
    args = ap.parse_args()

    battery = Path(args.battery)
    manifest = json.loads((battery / "manifest.json").read_text())
    n = any_hit = strict = 0
    by_position: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_source: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    lines: list[bytes] = []
    for shard in sorted(manifest["shards"], key=lambda s: s["index"]):
        with gzip.open(battery / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                chunks = chunk_text(row["context"])
                picked = [c.index for c in BM25ChunkSelector(chunks).select(row["target_question"], top_k=TOP_K)]
                picked_set = set(picked)
                spans = row["gold_spans"]
                hit = all(any(i in picked_set for i in span["chunks"]) for span in spans)
                first = all(span["chunks"][0] in picked_set for span in spans)
                n += 1
                any_hit += int(hit)
                strict += int(first)
                by_position[row["position"]][0] += int(hit)
                by_position[row["position"]][1] += int(first)
                by_position[row["position"]][2] += 1
                by_source[row["source"]][0] += int(hit)
                by_source[row["source"]][1] += int(first)
                by_source[row["source"]][2] += 1
                record = {
                    "id": row["id"],
                    "source": row["source"],
                    "position": row["position"],
                    "distractors": row.get("distractors"),
                    "distributed": row.get("distributed"),
                    "any_hit": bool(hit),
                    "strict_hit": bool(first),
                    "gold_chunks": [c for span in spans for c in span["chunks"]],
                    "gold_spans": [list(span["chunks"]) for span in spans],
                    "picked_chunks": sorted(picked),
                }
                lines.append((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())

    rows_path = Path(args.rows_out)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
            for line in lines:
                gz.write(line)
    rows_sha = sha256_file(rows_path)

    payload = {
        "battery": display_path(battery),
        "battery_manifest_sha256": sha256_file(battery / "manifest.json"),
        "battery_shard_sha256": {s["file"]: s["sha256"] for s in manifest["shards"]},
        "packs": n,
        "rows": {
            "path": display_path(rows_path),
            "count": len(lines),
            "bytes": rows_path.stat().st_size,
            "sha256": rows_sha,
        },
        "selector": "BM25",
        "top_k": TOP_K,
        "selector_recall": any_hit / n,
        "selector_recall_strict": strict / n,
        "by_position": {
            name: {"any": a / c, "strict": s / c, "n": c}
            for name, (a, s, c) in sorted(by_position.items())
        },
        "by_source": {
            name: {"any": a / c, "strict": s / c, "n": c}
            for name, (a, s, c) in sorted(by_source.items())
        },
        "gate": {
            "criterion": "any-overlap recall@8 >= 0.85 on dev packs",
            "threshold": 0.85,
            "metric": "every gold span touched by at least one of the top-8 chunks",
            "rows": n,
            "derived_from": "per-row picked_chunks/gold_spans/any_hit",
            "state": "pass" if any_hit / n >= 0.85 else "fail",
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "packs": n, "recall": payload["selector_recall"],
        "strict": payload["selector_recall_strict"], "gate": payload["gate"]["state"],
        "rows_sha256": rows_sha,
    }, indent=1))
    return 0 if payload["gate"]["state"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
