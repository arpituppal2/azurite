#!/usr/bin/env python3
"""Fetch one small remote artifact through the execute-code tunnel.

This is a development helper, not part of the model path.  It is used to bring
the per-row evaluation records back into the repo after they are produced on
the GPU host.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EXEC = PROJECT / ".agents/skills/marimo-pair/scripts/execute-code.sh"
SESSION = PROJECT / ".stage/live_session.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--local", required=True)
    args = ap.parse_args()
    session = json.loads(SESSION.read_text())
    code = (
        "import base64, pathlib\n"
        f"p = pathlib.Path({args.remote!r})\n"
        "print('B64START' + base64.b64encode(p.read_bytes()).decode() + 'B64END')\n"
    )
    env = dict(os.environ, MARIMO_TOKEN=session["token"])
    proc = subprocess.run(
        ["bash", str(EXEC), "--url", session["base"], "-c", code],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr[-2000:] + proc.stdout[-2000:])
    text = proc.stdout
    start = text.index("B64START") + len("B64START")
    end = text.index("B64END", start)
    raw = base64.b64decode(text[start:end])
    out = Path(args.local)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    print(f"{args.remote} -> {out} ({len(raw):,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
