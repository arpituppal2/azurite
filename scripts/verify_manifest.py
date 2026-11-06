#!/usr/bin/env python3
"""Write or verify a SHA-256 manifest for a results directory.

Usage:
    python scripts/verify_manifest.py results/suite_large_113m --write
    python scripts/verify_manifest.py results/suite_large_113m

The manifest is `MANIFEST.sha256`, one `<sha256>  <path>` line per file, with
paths relative to the directory and sorted, the same shape `shasum -a 256 -c`
reads.  The manifest itself, `.DS_Store` and other editor droppings are never
listed.  Verification fails if a file is missing, hashes differently, or
appears in the directory without being listed, so a silently added or removed
artifact is an error rather than an omission.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

IGNORE = {".DS_Store", "MANIFEST.sha256"}
IGNORE_SUFFIX = (".pyc", ".swp", ".tmp", ".part")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        if p.name in IGNORE or p.name.startswith("._"):
            continue
        if p.name.endswith(IGNORE_SUFFIX):
            continue
        files.append(p)
    return files


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        entries[rel] = digest
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    root = Path(args.directory).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    manifest = root / "MANIFEST.sha256"

    if args.write:
        lines = [f"{sha256(p)}  {p.relative_to(root)}" for p in collect(root)]
        manifest.write_text("\n".join(lines) + "\n")
        print(f"wrote {manifest} with {len(lines)} entries")
        return 0

    if not manifest.exists():
        raise SystemExit(f"missing manifest: {manifest}")
    listed = parse_manifest(manifest)
    actual = {str(p.relative_to(root)): p for p in collect(root)}
    problems = []
    for rel, path in actual.items():
        if rel not in listed:
            problems.append(f"unlisted file: {rel}")
            continue
        got = sha256(path)
        if got != listed[rel]:
            problems.append(f"hash mismatch: {rel} (manifest {listed[rel][:16]}, file {got[:16]})")
    for rel in listed:
        if rel not in actual:
            problems.append(f"missing file: {rel}")
    if problems:
        for line in problems:
            print(f"FAIL {line}", file=sys.stderr)
        return 1
    print(f"OK {len(listed)} files verified in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
