#!/usr/bin/env python3
"""Audit the frozen token stream before any optimizer step. CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="data/tokens/bpe_6200000000.u16")
    ap.add_argument("--expected-tokens", type=float, default=6.2e9)
    ap.add_argument("--out", default="results/parity/stream_audit.json")
    args = ap.parse_args()
    path = Path(args.stream)
    if not path.is_absolute():
        path = PROJECT / path
    if not path.exists():
        print(f"missing stream: {path}", file=sys.stderr)
        return 2
    if path.stat().st_size % 2:
        print("stream byte length is not a whole number of uint16 tokens", file=sys.stderr)
        return 2
    tokens = path.stat().st_size // 2
    if tokens < int(args.expected_tokens * 0.98):
        print(f"stream has {tokens:,} tokens, below 98% of {args.expected_tokens:,.0f}", file=sys.stderr)
        return 2
    record_path = path.with_suffix(".record.json")
    split_path = path.with_suffix(".split.json")
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    split = json.loads(split_path.read_text()) if split_path.exists() else None
    audit = {
        "stream": str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path),
        "bytes": path.stat().st_size, "tokens": tokens,
        "sha256": sha256(path),
        "record": record, "split": split,
        "record_sha256": sha256(record_path) if record_path.exists() else None,
        "split_sha256": sha256(split_path) if split_path.exists() else None,
        "expected_tokens": int(args.expected_tokens),
        "ok": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("tokens", "sha256", "ok")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
