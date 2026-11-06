#!/usr/bin/env python3
"""Measure BM25 recall@8 on a frozen pack battery, with per-row records."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from build_long_context import score_pack  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", required=True)
    args = ap.parse_args()
    battery = Path(args.battery)
    freeze = json.loads(Path(args.freeze).read_text())
    if not freeze.get("manifest_sha256"):
        raise SystemExit("freeze record has no SHA")
    manifest = json.loads((battery / "manifest.json").read_text())
    rows = []
    by_pos = defaultdict(lambda: [0, 0, 0])
    by_dist = defaultdict(lambda: [0, 0, 0])
    for shard in manifest["shards"]:
        with gzip.open(battery / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                pack = json.loads(line)
                any_hit, strict_hit = score_pack(pack)
                rec = {"id": pack.get("id"), "source": pack.get("source"),
                       "position": pack.get("position"),
                       "distractors": pack.get("distractors"),
                       "distributed": pack.get("distributed"),
                       "gold_spans": pack.get("gold_spans"),
                       "bm25_any": bool(any_hit), "bm25_strict": bool(strict_hit)}
                rows.append(rec)
                pos = by_pos[rec["position"]]; pos[0] += int(any_hit); pos[1] += int(strict_hit); pos[2] += 1
                dist = by_dist[str(rec["distractors"])]; dist[0] += int(any_hit); dist[1] += int(strict_hit); dist[2] += 1
    n = len(rows)
    any_hits = sum(r["bm25_any"] for r in rows)
    strict = sum(r["bm25_strict"] for r in rows)
    summary = {
        "battery": str(battery),
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "n": n,
        "top_k": 8,
        "selector": "BM25",
        "bm25_recall": any_hits / n,
        "bm25_recall_strict": strict / n,
        "by_position": {k: {"any": v[0] / v[2], "strict": v[1] / v[2], "n": v[2]}
                        for k, v in sorted(by_pos.items())},
        "by_distractors": {k: {"any": v[0] / v[2], "strict": v[1] / v[2], "n": v[2]}
                           for k, v in sorted(by_dist.items())},
        "gate_85": any_hits / n >= 0.85,
        "gate_90": any_hits / n >= 0.90,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
    Path(args.rows).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rows).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
