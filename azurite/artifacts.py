"""Content identity and per-example record verification.

The evaluation, training and verdict paths all need the same two things: a
SHA-256 over a file or a serialised string, and a check that a stored
``per_example_*.jsonl`` still recomputes the counts in its summary. Keeping them
here means the verdict renderers do not import the model stack just to hash a
file, and there is one definition of what "the record verifies" means.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def verify_arm_records(directory: Path, summary: dict) -> None:
    """Recompute every arm from its JSONL and check the stored file hash."""
    for name, arm in summary["arms"].items():
        path = directory / f"per_example_{name}.jsonl"
        rows = read_jsonl(path)
        correct = sum(int(row["correct"]) for row in rows)
        if len(rows) != arm["n"] or correct != arm["correct"]:
            raise SystemExit(f"{name}: recomputed {correct}/{len(rows)} != stored")
        if sha256_file(path) != arm["per_example_sha256"]:
            raise SystemExit(f"{name}: per-example hash mismatch")
