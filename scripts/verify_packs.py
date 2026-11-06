#!/usr/bin/env python3
"""Verify a pack battery from disk: 64 chunks, gold present, position as labelled.

Sampling reads a handful of shards rather than the whole battery, so it stays
cheap on a 100k-pack build while still covering every shard that gets read.

    .venv/bin/python3 scripts/verify_packs.py --battery results/parity/long_context_dev_noquestion
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.selector import chunk_text  # noqa: E402


def battery_sha256(manifest: dict) -> str:
    """Digest of the battery definition: every shard's name and hash, in order."""
    h = hashlib.sha256()
    for shard in sorted(manifest["shards"], key=lambda s: s["index"]):
        h.update(f"{shard['file']}:{shard['sha256']}:{shard['packs']}\n".encode())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.battery)
    manifest = json.loads((root / "manifest.json").read_text())
    shards = sorted(manifest["shards"], key=lambda s: s["index"])
    if not shards:
        raise SystemExit("manifest has no shards")
    step = max(1, len(shards) // args.shards)
    chosen = shards[::step][: args.shards]
    per_shard = max(1, args.sample // len(chosen))

    problems: list[str] = []
    checked = 0
    positions: dict[str, int] = {}
    for shard in chosen:
        path = root / shard["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != shard["sha256"]:
            problems.append(f"{path.name}: sha256 mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if checked >= args.sample or per_shard <= 0:
                    break
                per_shard -= 1
                pack = json.loads(line)
                checked += 1
                raw = pack["context"].encode("utf-8")
                size = len(raw)
                chunks = chunk_text(pack["context"])
                if len(chunks) != manifest["config"]["chunks_per_pack"]:
                    problems.append(f"{pack['id']}: {len(chunks)} chunks")
                if pack["chunks"] != len(chunks):
                    problems.append(f"{pack['id']}: chunk count field {pack['chunks']}")
                question = pack.get("target_question", "")
                if question and question in pack["context"]:
                    problems.append(f"{pack['id']}: question embedded")
                spans = pack["gold_spans"]
                if not spans:
                    problems.append(f"{pack['id']}: no gold span")
                for span in spans:
                    if not (0 <= span["start"] < span["end"] <= size):
                        problems.append(f"{pack['id']}: span outside text")
                    expected = [
                        c.index for c in chunks
                        if c.byte_start < span["end"] and c.byte_end > span["start"]
                    ]
                    if expected != span["chunks"]:
                        problems.append(f"{pack['id']}: span chunk mismatch {expected} vs {span['chunks']}")
                if pack["gold_chunk"] != spans[0]["chunks"][0]:
                    problems.append(f"{pack['id']}: gold_chunk disagrees with span")
                label = pack["position"]
                positions[label] = positions.get(label, 0) + 1
                if label == "distributed":
                    if len(spans) != 2 or spans[0]["start"] >= spans[1]["start"]:
                        problems.append(f"{pack['id']}: distributed layout")
                elif len(spans) == 1:
                    start = spans[0]["start"]
                    third = size / 3
                    want = {"early": (0, third), "middle": (third, 2 * third), "late": (2 * third, size)}[label]
                    if not (want[0] <= start < want[1]):
                        problems.append(f"{pack['id']}: {label} at {start}/{size}")

    digest = battery_sha256(manifest)
    print(json.dumps({
        "battery": str(root),
        "battery_sha256": digest,
        "packs_in_manifest": manifest["packs"],
        "shards": len(shards),
        "checked": checked,
        "positions": positions,
        "problems": problems[:10],
        "problem_count": len(problems),
    }, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
