#!/usr/bin/env python3
"""Freeze the 10k battery's per-row index and bind it to the training stream.

The battery shards themselves are never modified. This script only reads them
and writes three new artifacts next to the battery:

  results/parity/eval_battery_row_index.jsonl.gz
      one line per battery row: id, family, task, shard, position in shard,
      canonical row sha256, gold, gold label, provenance, licence, and the
      context/question sizes that a re-run can compare.
  results/parity/eval_battery_row_index.record.json
      rows, family counts, file bytes and sha256, and the canonical method.
  results/parity/eval_battery_stream_binding.json
      battery sha256 / freeze sha256 / manifest sha256, the per-row index sha,
      and the *actual* BPE training stream it is frozen against (sha256, token
      counts, held-out boundary, split rule).

Determinism: rows are read in shard and line order, JSON is written with sorted
keys and fixed separators, and the gzip member uses mtime=0. Re-running against
the same shards produces the same bytes and the same sha256.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

BATTERY = PROJECT / "results" / "parity" / "eval_battery_10k"
STREAM = PROJECT / "data" / "tokens" / "bpe_6200000000.clean.u16"
SPLIT = PROJECT / "data" / "tokens" / "bpe_6200000000.clean.split.json"
AUDIT = PROJECT / "results" / "parity" / "eval_battery_source_docs.clean.json"

QUOTAS = {
    "classification": 2_000,
    "boolean": 1_500,
    "pairwise": 1_000,
    "ordinal": 1_000,
    "document": 2_000,
    "abstention": 1_000,
    "robustness": 1_000,
    "multilingual": 500,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def row_sha256(row: dict) -> str:
    raw = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def canonical_line(entry: dict) -> bytes:
    return (json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def battery_rows(root: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    index: list[dict] = []
    for shard in sorted(manifest["shards"], key=lambda s: s["index"]):
        with gzip.open(root / shard["file"], "rt", encoding="utf-8") as fh:
            position = 0
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                provenance = row.get("provenance") or {}
                context = str(row.get("context", ""))
                index.append({
                    "id": str(row["id"]),
                    "family": str(row["family"]),
                    "task": str(row["task"]),
                    "shard": str(shard["file"]),
                    "shard_index": int(shard["index"]),
                    "position_in_shard": position,
                    "row_sha256": row_sha256(row),
                    "gold": int(row["gold"]),
                    "gold_label": str(row["gold_label"]),
                    "dataset": str(provenance.get("dataset", "")),
                    "split": str(provenance.get("split", "")),
                    "source_row": provenance.get("row"),
                    "licence": str(row.get("licence", "")),
                    "context_bytes": len(context.encode("utf-8")),
                    "question_chars": len(str(row.get("question", ""))),
                    "alternatives": len(row.get("alternatives") or []),
                    "gold_spans": len(row.get("gold_spans") or []),
                })
                position += 1
        if position != int(shard["rows"]):
            raise SystemExit(f"{shard['file']}: read {position} rows, manifest says {shard['rows']}")
    return manifest, index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=str(BATTERY))
    ap.add_argument("--stream", default=str(STREAM))
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--audit", default=str(AUDIT))
    ap.add_argument("--index", default="results/parity/eval_battery_row_index.jsonl.gz")
    ap.add_argument("--out", default="results/parity/eval_battery_stream_binding.json")
    args = ap.parse_args()

    root = Path(args.battery)
    manifest, index = battery_rows(root)
    battery_sha = manifest.get("battery_sha256")
    freeze_path = root / "FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("battery_sha256") != battery_sha:
        raise SystemExit("FREEZE.json and manifest.json disagree on battery sha256")
    if len(index) != int(manifest["rows"]):
        raise SystemExit(f"index has {len(index)} rows, manifest says {manifest['rows']}")
    if len({e["id"] for e in index}) != len(index):
        raise SystemExit("duplicate ids in battery row index")
    counts = Counter(e["family"] for e in index)
    if dict(counts) != QUOTAS:
        raise SystemExit(f"family counts {dict(counts)} do not match quotas {QUOTAS}")

    index_path = PROJECT / args.index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
            for entry in index:
                gz.write(canonical_line(entry))
    index_sha = sha256_file(index_path)
    index_bytes = index_path.stat().st_size
    index_digest = hashlib.sha256("".join(e["row_sha256"] for e in index).encode()).hexdigest()

    split = json.loads(Path(args.split).read_text())
    stream_path = Path(args.stream)
    stream_bytes = stream_path.stat().st_size
    if stream_bytes % 2:
        raise SystemExit("stream is not a whole number of uint16 tokens")
    stream_tokens = stream_bytes // 2
    if int(split["n_total"]) != stream_tokens:
        raise SystemExit(f"split says {split['n_total']} tokens, file has {stream_tokens}")
    if int(split["n_train"]) + int(split["heldout_bytes"]) != stream_tokens:
        raise SystemExit("split boundary does not add up to the stream length")
    stream_sha = sha256_file(stream_path)

    audit = json.loads(Path(args.audit).read_text()) if Path(args.audit).exists() else None
    audit_path = Path(args.audit)
    audit_body = (audit or {}).get("audit") or {}
    record = {
        "schema": 1,
        "purpose": "task 8 freeze provenance: per-row identity, battery hash, and the exact pretraining stream",
        "battery": {
            "path": str(root.relative_to(PROJECT)),
            "rows": len(index),
            "shards": len(manifest["shards"]),
            "families": dict(sorted(counts.items())),
            "quotas": QUOTAS,
            "battery_sha256": battery_sha,
            "battery_sha256_method": manifest.get("battery_sha256_method"),
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "freeze_path": str(freeze_path.relative_to(PROJECT)),
            "freeze_sha256": sha256_file(freeze_path),
            "gzip_bytes": sum(int(s["gzip_bytes"]) for s in manifest["shards"]),
        },
        "row_index": {
            "path": str(index_path.relative_to(PROJECT)),
            "rows": len(index),
            "bytes": index_bytes,
            "sha256": index_sha,
            "row_digest_sha256": index_digest,
            "row_digest_method": "sha256 of the concatenated per-row canonical sha256 strings in shard order",
            "fields": [
                "id", "family", "task", "shard", "shard_index", "position_in_shard",
                "row_sha256", "gold", "gold_label", "dataset", "split", "source_row",
                "licence", "context_bytes", "question_chars", "alternatives", "gold_spans",
            ],
        },
        "stream": {
            "path": str(stream_path.relative_to(PROJECT)),
            "bytes": stream_bytes,
            "tokens": stream_tokens,
            "sha256": stream_sha,
            "training_tokens": int(split["n_train"]),
            "heldout_tokens": int(split["heldout_bytes"]),
            "heldout_fraction": float(split["heldout_fraction"]),
            "excluded_documents": int(split["excluded_documents"]),
            "split_rule": split["split_rule"],
            "split_path": str(Path(args.split).relative_to(PROJECT)),
            "split_sha256": sha256_file(Path(args.split)),
            "note": (
                "The battery is frozen against the corrected clean stream, not the earlier "
                "e8da0a60 stream whose byte boundary crossed a document."
            ),
        },
        "battery_source_audit": None if audit is None else {
            "path": str(audit_path.relative_to(PROJECT)),
            "sha256": sha256_file(audit_path),
            "stream_sha256": audit.get("new_stream_sha256"),
            "patterns": audit_body.get("patterns"),
            "whole_stream_hit_source_docs": audit_body.get("whole_stream_hit_source_docs"),
            "whole_stream_hit_positions": audit_body.get("whole_stream_hit_positions"),
            "control_found": audit_body.get("control_found"),
            "gate": audit.get("audit", {}).get("gate"),
            "scope": "battery source documents only; per-row dev/locked pack findings are a separate artifact",
        },
        "limits": [
            "The row index proves identity and counts; it does not re-run the battery builder.",
            "The stream sha256 is computed from the file on disk, not copied from a prior record.",
            "This binding does not by itself prove the seeded build regenerates; see eval_battery_regeneration.json.",
        ],
    }
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=1) + "\n"
    out.write_text(payload, encoding="utf-8")
    (out.with_suffix(out.suffix + ".sha256")).write_text(
        hashlib.sha256(payload.encode()).hexdigest() + "\n", encoding="utf-8"
    )
    (index_path.with_suffix(index_path.suffix + ".sha256")).write_text(index_sha + "\n", encoding="utf-8")
    index_record = {
        "path": str(index_path.relative_to(PROJECT)),
        "rows": len(index),
        "bytes": index_bytes,
        "sha256": index_sha,
        "row_digest_sha256": index_digest,
        "families": dict(sorted(counts.items())),
        "battery_sha256": battery_sha,
        "freeze_sha256": sha256_file(freeze_path),
    }
    index_record_path = index_path.with_suffix("").with_suffix(".record.json")
    index_record_path.write_text(json.dumps(index_record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(index),
        "families": dict(sorted(counts.items())),
        "row_index_sha256": index_sha,
        "stream_sha256": stream_sha,
        "binding": str(out.relative_to(PROJECT)),
        "binding_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
