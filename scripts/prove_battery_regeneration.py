#!/usr/bin/env python3
"""Prove the frozen 10k battery regenerates from its recorded seed.

Runs the real builder into a scratch directory (never over the frozen battery),
then compares every observable identity the freeze records:

  * canonical battery sha256 (sha256 over sorted shard file:sha:rows lines);
  * each shard's sha256, byte size and row count;
  * per-family counts and total rows;
  * a digest of the per-row canonical sha256 strings, built independently from
    the rebuilt shards, against the frozen per-row index.

The build uses the same seed (0), shard size (1,000) and abstention blocklist
that the frozen build used. The scratch directory is removed unless --keep is
passed; the proof file stores hashes and counts, not the rebuilt battery.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

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


def battery_sha(shards: list[dict]) -> str:
    h = hashlib.sha256()
    for shard in sorted(shards, key=lambda s: s["index"]):
        h.update(f"{shard['file']}:{shard['sha256']}:{shard['rows']}\n".encode())
    return h.hexdigest()


def row_digest(root: Path, shards: list[dict]) -> tuple[str, int, Counter]:
    h = hashlib.sha256()
    total = 0
    counts: Counter = Counter()
    for shard in sorted(shards, key=lambda s: s["index"]):
        with gzip.open(root / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                h.update(hashlib.sha256(raw.encode("utf-8")).hexdigest().encode())
                counts[row["family"]] += 1
                total += 1
    return h.hexdigest(), total, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="results/parity/eval_battery_10k")
    ap.add_argument("--out", default="results/parity/eval_battery_regeneration.json")
    ap.add_argument("--tmp", default="/tmp/azurite_battery_rebuild_seed0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows-per-shard", type=int, default=1_000)
    ap.add_argument(
        "--block-terms", default="results/parity/.eval_battery_blocklist.json",
        help="blocklist the frozen build used for the abstention family",
    )
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    frozen = PROJECT / args.battery
    tmp = Path(args.tmp).resolve()
    if not (str(tmp).startswith("/tmp/") or str(tmp).startswith("/private/tmp/")):
        raise SystemExit("refusing to rebuild outside /tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    cmd = [
        sys.executable, str(PROJECT / "scripts" / "build_eval_battery.py"),
        "--out-dir", str(tmp), "--seed", str(args.seed),
        "--rows-per-shard", str(args.rows_per_shard),
        "--block-terms", str(PROJECT / args.block_terms),
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True)
    seconds = round(time.time() - started, 1)
    (tmp / "build.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (tmp / "build.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stdout[-4000:], file=sys.stderr)
        print(proc.stderr[-4000:], file=sys.stderr)
        payload = {
            "schema": 1,
            "regenerated": False,
            "reason": f"builder exited {proc.returncode}",
            "command": " ".join(cmd),
            "seconds": seconds,
            "stderr_tail": proc.stderr[-2000:],
        }
        out = PROJECT / args.out
        out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        return 1

    frozen_manifest = json.loads((frozen / "manifest.json").read_text())
    rebuilt_manifest = json.loads((tmp / "manifest.json").read_text())
    rebuilt_shards = rebuilt_manifest["shards"]

    shard_checks = []
    for shard in sorted(rebuilt_shards, key=lambda s: s["index"]):
        path = tmp / shard["file"]
        actual = sha256_file(path)
        frozen_shard = next(
            (s for s in frozen_manifest["shards"] if s["index"] == shard["index"]), None
        )
        shard_checks.append({
            "index": shard["index"],
            "file": shard["file"],
            "rows": shard["rows"],
            "bytes": path.stat().st_size,
            "sha256": actual,
            "frozen_sha256": frozen_shard["sha256"] if frozen_shard else None,
            "matches_frozen": bool(frozen_shard) and actual == frozen_shard["sha256"]
            and path.stat().st_size == int(frozen_shard["gzip_bytes"])
            and int(shard["rows"]) == int(frozen_shard["rows"]),
        })

    rebuilt_battery_sha = battery_sha(rebuilt_shards)
    rebuilt_row_digest, rebuilt_rows, rebuilt_counts = row_digest(tmp, rebuilt_shards)
    index_record_path = PROJECT / "results" / "parity" / "eval_battery_row_index.record.json"
    index_record = json.loads(index_record_path.read_text()) if index_record_path.exists() else {}

    checks = {
        "rows": rebuilt_rows == int(frozen_manifest["rows"]),
        "families": dict(rebuilt_counts) == QUOTAS,
        "battery_sha256": rebuilt_battery_sha == frozen_manifest["battery_sha256"],
        "manifest_battery_sha256": rebuilt_manifest.get("battery_sha256") == frozen_manifest["battery_sha256"],
        "all_shards": all(c["matches_frozen"] for c in shard_checks) and len(shard_checks) == len(frozen_manifest["shards"]),
        "row_digest": bool(index_record) and rebuilt_row_digest == index_record.get("row_digest_sha256"),
    }
    ok = all(checks.values())
    payload = {
        "schema": 1,
        "regenerated": ok,
        "checks": checks,
        "command": " ".join(cmd),
        "seconds": seconds,
        "builder_stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
        "frozen": {
            "path": args.battery,
            "battery_sha256": frozen_manifest["battery_sha256"],
            "rows": frozen_manifest["rows"],
            "shards": len(frozen_manifest["shards"]),
            "manifest_sha256": sha256_file(frozen / "manifest.json"),
        },
        "rebuilt": {
            "path": str(tmp),
            "battery_sha256": rebuilt_battery_sha,
            "rows": rebuilt_rows,
            "families": dict(sorted(rebuilt_counts.items())),
            "row_digest_sha256": rebuilt_row_digest,
        },
        "row_digest_from_frozen_index": index_record.get("row_digest_sha256"),
        "shards": shard_checks,
        "seed": args.seed,
        "rows_per_shard": args.rows_per_shard,
        "block_terms": str(args.block_terms),
        "rebuild_retained": bool(args.keep),
        "note": (
            "The rebuilt battery is byte-identical shard-for-shard. The scratch directory "
            "is not part of the repository; this file carries every hash needed to re-run it."
        ) if ok else "Regeneration did not reproduce the frozen battery; see checks and shard table.",
    }
    out = PROJECT / args.out
    text = json.dumps(payload, indent=1) + "\n"
    out.write_text(text, encoding="utf-8")
    (out.with_suffix(out.suffix + ".sha256")).write_text(
        hashlib.sha256(text.encode()).hexdigest() + "\n", encoding="utf-8"
    )
    if not args.keep:
        shutil.rmtree(tmp)
    print(json.dumps({
        "regenerated": ok, "checks": checks, "battery_sha256": rebuilt_battery_sha,
        "seconds": seconds, "out": str(out.relative_to(PROJECT)),
    }, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
