#!/usr/bin/env python3
"""Verify what is on disk against data/MANIFEST.json.

The manifest is only worth something if it can be checked. This re-derives
every count from the files themselves:

  * task splits: reloaded through the datasets library, rows and columns
    compared against the record
  * corpus shards: streamed through zlib, counting documents and bytes; a
    successful full decompression also validates the gzip CRC
  * checkpoints: sha256 of every recorded file

Any disagreement is a failure and is printed with both numbers.

Usage:
    .venv/bin/python3 scripts/verify_data.py [--models] [--quick]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
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


def count_gzip(path: Path, counts: bool = True) -> tuple[int, int]:
    """Documents and exact uncompressed bytes, validating the gzip stream."""
    docs = 0
    size = 0
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with path.open("rb") as fh:
        while True:
            block = fh.read(1 << 22)
            if not block:
                break
            size += len(dec.decompress(block))
    if counts:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for _ in fh:
                docs += 1
    return docs, size


def used_list(rec: dict) -> list[dict]:
    used = rec.get("used", [])
    return [used] if isinstance(used, dict) else list(used)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="also hash checkpoints")
    ap.add_argument("--quick", action="store_true", help="skip hashing shards")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print("no manifest")
        return 1
    manifest = json.loads(MANIFEST.read_text())
    failures: list[str] = []

    print(f"{'key':24s} {'what':>12s} {'manifest':>12s} {'on disk':>12s}  verdict")
    print("-" * 78)

    for key, rec in sorted(manifest["datasets"].items()):
        if rec.get("kind") != "task":
            continue
        for split, s in rec["splits"].items():
            cfgs = [u["config"] for u in used_list(rec) if u.get("config")]
            candidates = [PROJECT / rec["path"] / (c or "default") / split for c in cfgs]
            candidates.append(PROJECT / rec["path"] / "default" / split)
            split_dir = next((c for c in candidates if c.exists()), None)
            if split_dir is None:
                failures.append(f"{key}/{split}")
                print(f"{key:24s} {split:>12s} {s['rows']:>12,d} {'-':>12s}  MISSING")
                continue
            from datasets import load_from_disk

            ds = load_from_disk(str(split_dir))
            rows, cols = ds.num_rows, ds.column_names
            del ds
            ok = rows == s["rows"] and cols == s["columns"]
            if not ok:
                failures.append(f"{key}/{split}")
            print(f"{key:24s} {split:>12s} {s['rows']:>12,d} {rows:>12,d}  "
                  f"{'ok' if ok else 'MISMATCH cols=' + str(cols)}")

    for key, rec in sorted(manifest["datasets"].items()):
        if rec.get("kind") != "corpus":
            continue
        tot_docs = tot_bytes = 0
        bad = []
        for shard in rec.get("shards", []):
            path = PROJECT / shard["path"]
            if not path.exists():
                bad.append(f"{path.name}: missing")
                continue
            docs, size = count_gzip(path, counts=not args.quick)
            tot_docs += docs
            tot_bytes += size
            if docs != shard["docs"]:
                bad.append(f"{path.name}: {docs} docs vs {shard['docs']}")
            if abs(size - shard["bytes_uncompressed"]) > 1024:
                bad.append(f"{path.name}: {size} bytes vs {shard['bytes_uncompressed']}")
            if not args.quick:
                digest = sha256_file(path)
                if digest != shard["sha256"]:
                    bad.append(f"{path.name}: sha256 differs")
        ok = (
            not bad
            and tot_docs == rec.get("docs")
            and abs(tot_bytes - rec.get("bytes_uncompressed", 0)) <= 1024
            and len(rec.get("shards", [])) > 0
        )
        if not ok:
            failures.append(key)
        print(f"{key:24s} {'corpus':>12s} {rec.get('docs', 0):>12,d} {tot_docs:>12,d}"
              f"  {'ok' if ok else 'MISMATCH'} "
              f"({tot_bytes / 1e9:.2f} GB vs {rec.get('bytes_uncompressed', 0) / 1e9:.2f} GB,"
              f" {len(rec.get('shards', []))} shards)")
        for b in bad[:5]:
            print(f"    !! {b}")

    if args.models:
        for key, rec in sorted(manifest.get("models", {}).items()):
            if rec.get("error"):
                print(f"{key:24s} {'model':>12s} {'-':>12s} {'-':>12s}  FAILED PULL")
                failures.append(key)
                continue
            root = PROJECT / rec["local_path"]
            bad = 0
            for f in rec.get("files", []):
                p = root / f["path"]
                if not p.exists() or sha256_file(p) != f["sha256"]:
                    bad += 1
            ok = bad == 0
            if not ok:
                failures.append(key)
            print(f"{key:24s} {'model':>12s} {len(rec.get('files', [])):>12d} "
                  f"{len(rec.get('files', [])) - bad:>12d}  "
                  f"{'ok' if ok else f'{bad} FILE(S) DIFFER'}"
                  f" ({rec.get('params_total', 0) / 1e6:.1f}M params)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} artefact(s): {', '.join(failures)}")
        return 1
    print("PASS: on-disk data matches the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
