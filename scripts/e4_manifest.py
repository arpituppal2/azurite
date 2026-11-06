#!/usr/bin/env python3
"""Write the E4 artifact manifest (run where the artifacts live).

Covers the pre-registration and its hash, the summary, and for every valid arm
its `result.json`, `per_example.jsonl` and `final.pt`.  The void first launch
under `_invalid_init/` is hashed too, so the record of what was discarded is as
tamper-evident as the record that was kept, but it is listed under its own
prefix and never enters the summary.

Usage:
    python3 scripts/e4_manifest.py --root results/e4_candidate_sweep
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/e4_candidate_sweep")
    args = ap.parse_args()
    root = Path(args.root)

    files = [root / "PRE_REGISTRATION.md", root / "PRE_REGISTRATION.sha256",
             root / "SUMMARY.json", root / "SUMMARY.md"]
    for task in ("banking77", "clinc_oos", "massive_intent"):
        for arm in sorted((root / task).glob("w*")):
            for name in ("result.json", "per_example.jsonl", "final.pt"):
                files.append(arm / name)
    for path in sorted((root / "_invalid_init").rglob("*")):
        if path.is_file():
            files.append(path)

    lines = []
    for path in sorted(set(files)):
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")
        lines.append(f"{sha256(path)}  {path.relative_to(root)}")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(f"wrote {root/'MANIFEST.sha256'} ({len(lines)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
