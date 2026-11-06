#!/usr/bin/env python3
"""Write the manifest for the protocol-matched Banking77 evidence.

Hashes only files that already exist; it never edits a result. Paths are stored
relative to the repository root so the manifest is verifiable from a fresh
checkout of the evidence directory.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "results/jev_matched/PRE_REGISTRATION.md",
    "results/jev_matched/PRE_REGISTRATION_AMENDMENT_1.md",
    "results/jev_matched/inputs.json",
    "results/jev_matched/summary.json",
    "results/jev_matched/VERDICT.md",
    "results/jev_matched/per_example_no_examples_1024.jsonl",
    "results/jev_matched/per_example_examples24_5120.jsonl",
    "results/jev_matched/per_example_examples24_1024.jsonl",
    "results/jev_matched/v1_replication/jev_matched_eval_v1.py",
    "results/jev_matched/v1_replication/per_example_no_examples_1024.jsonl",
    "results/jev_matched/v1_replication/per_example_examples24_5120.jsonl",
    "results/jev_matched/v1_replication/run_all.log",
    "scripts/jev_matched_inputs.py",
    "scripts/jev_matched_eval.py",
    "scripts/jev_matched_manifest.py",
    "scripts/jev_matched_verdict.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT, type=Path)
    ap.add_argument("--out", default=ROOT / "results/jev_matched/MANIFEST.sha256", type=Path)
    args = ap.parse_args()
    lines = []
    for rel in FILES:
        path = args.root / rel
        if not path.exists():
            raise SystemExit(f"missing evidence file: {rel}")
        lines.append(f"{sha256(path)}  ./{rel}")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
