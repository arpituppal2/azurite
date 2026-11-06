#!/usr/bin/env python3
"""Guarded live Jev run. Refuses to start unless the internal gate passed."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.battery import read_battery  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="results/parity/battery_10k.json.gz")
    ap.add_argument("--gate", default="results/parity/PARITY_GATE.json")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--api-key-env", default="TYPESAFE_API_KEY")
    ap.add_argument("--out", default="results/parity/JEV_LIVE.json")
    args = ap.parse_args()
    if not Path(args.gate).exists():
        print(f"refusing live Jev call: no gate file at {args.gate}", file=sys.stderr)
        return 2
    gate = json.loads(Path(args.gate).read_text())
    if not gate.get("passed"):
        print("refusing live Jev call: internal parity gate has not passed", file=sys.stderr)
        return 2
    key = os.environ.get(args.api_key_env)
    if not key:
        print(f"missing {args.api_key_env}", file=sys.stderr)
        return 2
    rows, _ = read_battery(args.battery)
    records = []
    for row in rows:
        body = json.dumps({
            "model": "jev-1.14.0",
            "input": row.context,
            "question": row.question,
            "options": row.alternatives,
        }).encode("utf-8")
        req = urllib.request.Request(args.endpoint, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
        records.append({"id": row.id, "response": payload, "model": "jev-1.14.0"})
    out = Path(args.out)
    out.write_text(json.dumps(records, indent=1) + "\n")
    print(f"wrote {out} ({len(records)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
