#!/usr/bin/env python3
"""Audit target packs against dev/locked pack evidence and reject overlaps.

Usage:
    .venv/bin/python scripts/audit_pack_reference.py \
      --reference dev=results/parity/long_context_dev_noquestion \
      --reference locked=results/parity/long_context_locked_noquestion \
      --target train=results/parity/long_context_train_100k \
      --out results/parity/pack_reference_contamination.json

Every target row gets a stored audit record. Contaminated and unauditable packs
are written to ``--rejected-out``; only accepted packs may be used for training.
The command exits 1 when any target pack is rejected.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.pack_audit import audit_packs, build_reference_windows  # noqa: E402
from azurite.tokenizers import load as load_tokenizer  # noqa: E402


def read_packs(path: Path) -> list[dict]:
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text())
        rows: list[dict] = []
        for shard in sorted(manifest["shards"], key=lambda s: s["index"]):
            with gzip.open(path / shard["file"], "rt", encoding="utf-8") as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
        return rows
    suffix = path.suffix.lower()
    if suffix in (".gz", ".gzip"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_named(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected LABEL=DIR, got {value!r}")
    label, raw = value.split("=", 1)
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT / path
    return label, path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", action="append", type=parse_named, required=True,
                    help="dev/locked reference set, LABEL=DIR (repeatable)")
    ap.add_argument("--target", action="append", type=parse_named, required=True,
                    help="target pack set to audit, LABEL=DIR (repeatable)")
    ap.add_argument("--out", default="results/parity/pack_reference_contamination.json")
    ap.add_argument("--rejected-out", default="results/parity/pack_reference_rejected.jsonl")
    ap.add_argument("--tokenizer", default="bpe")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    reference_packs: list[dict] = []
    for label, path in args.reference:
        rows = read_packs(path)
        reference_packs.extend(rows)
        print(f"reference {label}: {len(rows)} packs", flush=True)
    reference_windows = build_reference_windows(reference_packs, tok)
    if not reference_windows:
        raise SystemExit("reference has no auditable evidence windows")

    target_sets: dict[str, dict] = {}
    rejected_rows: list[dict] = []
    any_rejected = False
    for label, path in args.target:
        rows = read_packs(path)
        results = audit_packs(rows, reference_windows, tok)
        accepted = sum(1 for r in results if r.accepted)
        rejected = [r for r in results if not r.accepted]
        any_rejected = any_rejected or bool(rejected)
        target_sets[label] = {
            "path": str(path),
            "packs": len(rows),
            "accepted": accepted,
            "rejected": len(rejected),
            "unauditable": sum(1 for r in results if not r.auditable),
            "matched": sum(1 for r in results if r.matched_windows),
            "results": [r.to_dict() for r in results],
        }
        for result in rejected:
            rejected_rows.append({
                "set": label, "pack_id": result.pack_id,
                "auditable": result.auditable,
                "matched_windows": result.matched_windows,
                "reason": result.reason,
            })
        print(f"target {label}: {accepted} accepted, {len(rejected)} rejected", flush=True)

    report = {
        "schema": 1,
        "reference_sets": [label for label, _ in args.reference],
        "reference_packs": len(reference_packs),
        "reference_windows": len(reference_windows),
        "targets": target_sets,
        "gate": {
            "criterion": "every target pack has auditable evidence and no window overlaps dev/locked evidence",
            "state": "pass" if not any_rejected else "fail",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    rejected_path = Path(args.rejected_out)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    with rejected_path.open("w", encoding="utf-8") as fh:
        for row in rejected_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report["gate"], indent=1))
    print(f"wrote {out} and {rejected_path}")
    return 0 if report["gate"]["state"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
