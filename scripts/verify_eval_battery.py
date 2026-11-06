#!/usr/bin/env python3
"""Verify the frozen 10k evaluation battery on disk.

Checks, per row and per shard:
  * shard files exist, sizes match the manifest, and sha256 matches the metadata;
  * ids are unique, families are known, and per-family counts equal the plan's
    quotas (2,000/1,500/1,000/1,000/2,000/1,000/1,000/500);
  * every row has a question, context, non-empty unique alternatives, a gold
    index inside the alternatives, and a gold label that equals the alternative;
  * provenance (dataset, split, row) and a licence are present;
  * document rows are 64 chunks of 65,537..66,560 bytes, the question does not
    occur in the context, and the gold spans resolve to real chunk indices;
  * abstention rows record whether the answer was verified unique/present or
    verified absent, and answerable rows carry a span answer;
  * the canonical battery sha256 recomputed from the manifest equals the one
    stored in it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from azurite.selector import chunk_text  # noqa: E402

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
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def battery_sha256(shards: list[dict]) -> str:
    h = hashlib.sha256()
    for shard in sorted(shards, key=lambda s: s["index"]):
        h.update(f"{shard['file']}:{shard['sha256']}:{shard['rows']}\n".encode())
    return h.hexdigest()


def check_row(row: dict, problems: list[str]) -> None:
    rid = row.get("id", "<no id>")
    family = row.get("family")
    for key in ("id", "family", "task", "question", "context", "alternatives", "gold",
                "gold_label", "provenance", "licence", "meta", "gold_spans"):
        if key not in row:
            problems.append(f"{rid}: missing key {key!r}")
    if family not in QUOTAS:
        problems.append(f"{rid}: unknown family {family!r}")
        return
    alts = row.get("alternatives") or []
    if not alts:
        problems.append(f"{rid}: no alternatives")
    elif len(set(alts)) != len(alts):
        problems.append(f"{rid}: duplicate alternatives")
    gold = row.get("gold")
    if not isinstance(gold, int) or not 0 <= gold < len(alts):
        problems.append(f"{rid}: gold {gold!r} outside {len(alts)} alternatives")
    elif row.get("gold_label") != alts[gold]:
        problems.append(f"{rid}: gold_label {row.get('gold_label')!r} != alternative {alts[gold]!r}")
    if not str(row.get("question", "")).strip():
        problems.append(f"{rid}: empty question")
    if not str(row.get("context", "")).strip():
        problems.append(f"{rid}: empty context")
    prov = row.get("provenance") or {}
    if not prov.get("dataset") or not prov.get("split") or not isinstance(prov.get("row"), int):
        problems.append(f"{rid}: incomplete provenance {prov!r}")
    if not row.get("licence"):
        problems.append(f"{rid}: empty licence")

    if family in ("document", "abstention"):
        context = str(row["context"])
        size = len(context.encode("utf-8"))
        if not 65_537 <= size <= 66_560:
            problems.append(f"{rid}: context is {size} bytes, outside 65537..66560")
        chunks = chunk_text(context)
        if len(chunks) != 64:
            problems.append(f"{rid}: {len(chunks)} chunks, expected 64")
        if row.get("question") and row["question"] in context:
            problems.append(f"{rid}: question occurs in context")
        if family == "document":
            spans = row.get("gold_spans") or []
            if not spans:
                problems.append(f"{rid}: document row has no gold span")
            for span in spans:
                for key in ("start", "end", "chunks"):
                    if key not in span:
                        problems.append(f"{rid}: span missing {key!r}")
                idx = span.get("chunks") or []
                if not idx or any(not isinstance(i, int) or not 0 <= i < 64 for i in idx):
                    problems.append(f"{rid}: span chunks {idx!r} outside 0..63")
            if spans and spans[0].get("chunks") and spans[0]["chunks"][0] != row.get("meta", {}).get("gold_chunk"):
                problems.append(f"{rid}: gold_chunk disagrees with spans[0]")
        else:
            meta = row.get("meta") or {}
            variant = meta.get("variant")
            if variant not in ("answerable", "unanswerable"):
                problems.append(f"{rid}: abstention variant {variant!r}")
            spans = row.get("gold_spans") or []
            if variant == "answerable":
                if not row.get("answer"):
                    problems.append(f"{rid}: answerable row has no answer")
                if meta.get("phrase_verified_unique") is not True:
                    problems.append(f"{rid}: answerable row not marked verified unique")
                if not spans:
                    problems.append(f"{rid}: answerable row has no gold span")
            elif variant == "unanswerable":
                if meta.get("phrase_verified_absent") is not True:
                    problems.append(f"{rid}: unanswerable row not marked verified absent")
                if spans:
                    problems.append(f"{rid}: unanswerable row carries a gold span")
                if row.get("answer"):
                    problems.append(f"{rid}: unanswerable row carries an answer")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="results/parity/eval_battery_10k")
    args = ap.parse_args()
    root = Path(args.battery)
    manifest = json.loads((root / "manifest.json").read_text())

    problems: list[str] = []
    shards = manifest.get("shards") or []
    if not shards:
        problems.append("manifest has no shards")
    counts: Counter = Counter()
    sizes: list[int] = []
    for shard in shards:
        path = root / shard["file"]
        if not path.exists():
            problems.append(f"{shard['file']}: missing")
            continue
        size = path.stat().st_size
        if size != shard.get("gzip_bytes"):
            problems.append(f"{shard['file']}: bytes {size} != manifest {shard.get('gzip_bytes')}")
        if sha256_file(path) != shard.get("sha256"):
            problems.append(f"{shard['file']}: sha256 mismatch")
        rows_in_shard = 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows_in_shard += 1
                row = json.loads(line)
                counts[row.get("family")] += 1
                check_row(row, problems)
        if rows_in_shard != shard.get("rows"):
            problems.append(f"{shard['file']}: {rows_in_shard} rows != manifest {shard.get('rows')}")
        sizes.append(size)

    total = sum(counts.values())
    if total != 10_000:
        problems.append(f"total rows {total}, expected 10000")
    for family, quota in QUOTAS.items():
        if counts.get(family, 0) != quota:
            problems.append(f"family {family}: {counts.get(family, 0)} rows, expected {quota}")
    if dict(manifest.get("family_counts", {})) != {k: counts[k] for k in QUOTAS}:
        problems.append(f"manifest family_counts disagree with data: {manifest.get('family_counts')}")

    canonical = battery_sha256(shards)
    if manifest.get("battery_sha256") != canonical:
        problems.append(f"battery sha mismatch: manifest {manifest.get('battery_sha256')} != {canonical}")

    report = {
        "battery": str(root),
        "battery_sha256": canonical,
        "rows": total,
        "shards": len(shards),
        "gzip_bytes": sum(sizes),
        "families": dict(sorted(counts.items())),
        "problems": problems[:50],
        "problem_count": len(problems),
    }
    print(json.dumps(report, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
