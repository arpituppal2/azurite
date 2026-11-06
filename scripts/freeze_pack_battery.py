#!/usr/bin/env python3
"""Freeze a built pack battery and record its SHA before evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--overlap", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    battery = Path(args.battery)
    manifest_path = battery / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shards = []
    for shard in manifest["shards"]:
        path = battery / shard["file"]
        got = sha256_file(path)
        if got != shard["sha256"]:
            raise SystemExit(f"shard hash mismatch: {shard['file']}")
        shards.append({"file": shard["file"], "packs": shard["packs"],
                       "sha256": got, "bytes": path.stat().st_size})
    if manifest.get("packs") != sum(s["packs"] for s in shards):
        raise SystemExit("manifest pack count mismatch")
    overlap = json.loads(Path(args.overlap).read_text())
    if not overlap.get("zero_overlap"):
        raise SystemExit("refusing to freeze: source document overlap is not zero")
    payload = {
        "schema": 1,
        "battery": "locked_32k",
        "path": str(battery),
        "packs": manifest["packs"],
        "chunks_per_pack": manifest["config"]["chunks_per_pack"],
        "split": manifest["config"]["split"],
        "sources": manifest["config"]["sources"],
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "shards": shards,
        "overlap_record": str(args.overlap),
        "document_overlap": {"train": overlap["overlap_train"],
                             "dev": overlap["overlap_dev"]},
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_started": False,
        "note": "This battery was built from official test splits. BM25/neural evaluation must start only after this record exists.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: payload[k] for k in ("packs", "manifest_sha256", "document_overlap", "frozen_at")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
