#!/usr/bin/env python3
"""Recompute corpus records from the shard files, without re-downloading.

Why this is a repair and not a re-pull: the shards on disk are correct. What
was wrong was what the manifest said about them. The writer counted only the
text payload while the verifier measured the decompressed file, so every shard
disagreed by the size of its own JSON syntax and escaping -- 0.76% systematically,
which is small enough to look like noise and large enough to be wrong.

This walks each recorded shard once and re-derives, from the file itself:

    docs, bytes_uncompressed (whole decompressed file), bytes_text (text only),
    bytes_compressed, sha256

and rewrites the corpus entries. It records in the manifest that it ran, so the
provenance of the numbers stays traceable.

Usage:
    .venv/bin/python3 scripts/repair_corpus_records.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import zlib
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "data" / "MANIFEST.json"


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def measure(shard: Path) -> dict:
    docs = 0
    text_bytes = 0
    file_bytes = 0
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with shard.open("rb") as fh:
        while True:
            block = fh.read(1 << 22)
            if not block:
                break
            file_bytes += len(dec.decompress(block))
    with gzip.open(shard, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            docs += 1
            text = json.loads(line).get("text", "")
            text_bytes += len(text.encode("utf-8"))
    return {
        "docs": docs,
        "bytes_uncompressed": file_bytes,
        "bytes_text": text_bytes,
        "bytes_compressed": shard.stat().st_size,
        "sha256": sha256_file(shard),
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    changed = []
    for key, rec in sorted(manifest["datasets"].items()):
        if rec.get("kind") != "corpus":
            continue
        print(f"\n{key}: {len(rec.get('shards', []))} shards")
        for shard in rec["shards"]:
            path = PROJECT / shard["path"]
            if not path.exists():
                print(f"  !! {path.name} missing")
                continue
            measured = measure(path)
            delta = measured["bytes_uncompressed"] - shard.get("bytes_uncompressed", 0)
            shard.update(measured)
            shard["seconds"] = shard.get("seconds")
            print(
                f"  {path.name}: {measured['docs']:,} docs, "
                f"file {measured['bytes_uncompressed'] / 1e9:.3f} GB, "
                f"text {measured['bytes_text'] / 1e9:.3f} GB "
                f"(was {delta / 1e6:+.1f} MB)"
            )
            changed.append(key)
        rec["docs"] = sum(s["docs"] for s in rec["shards"])
        rec["bytes_uncompressed"] = sum(s["bytes_uncompressed"] for s in rec["shards"])
        rec["bytes_text"] = sum(s["bytes_text"] for s in rec["shards"])
        rec["bytes_compressed"] = sum(s["bytes_compressed"] for s in rec["shards"])
        rec["records_recomputed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        print(
            f"  -> {rec['docs']:,} docs, {rec['bytes_text'] / 1e9:.2f} GB text / "
            f"{rec['cap_bytes'] / 1e9:.2f} GB cap"
        )

    manifest["corpus_records_recomputed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nrewrote {len(set(changed))} corpus record(s) in {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
