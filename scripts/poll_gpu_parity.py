#!/usr/bin/env python3
"""Poll the GPU host for stream-build and reader-pretrain state. Read-only.

Writes results/parity/gpu/current_state.json so every observation lands in a
stored record rather than only in a terminal scrollback. Never runs an
optimizer, never edits the remote workspace.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EXEC = PROJECT / ".agents/skills/marimo-pair/scripts/execute-code.sh"
SESSION = PROJECT / ".stage/live_session.json"
STATE = PROJECT / "results/parity/gpu/current_state.json"

PROBE = r'''
import json, os, pathlib, time
ws = pathlib.Path("/marimo/azurite_ws")
out = pathlib.Path("/marimo/azurite_ws/results/parity")
def proc(pid):
    p = pathlib.Path(f"/proc/{pid}")
    if not p.exists(): return None
    st = (p/"stat").read_text().split()
    d = {"state": st[2], "utime": int(st[13]), "stime": int(st[14])}
    try:
        d["cmd"] = (p/"cmdline").read_bytes().replace(b"\x00", b" ").decode()[:160]
    except Exception: d["cmd"] = None
    return d
def fill(path, cap=None):
    """Last written token, by binary search.

    The memmap is preallocated, so file size says nothing about progress. A
    linear backward scan from the end reads gigabytes every poll; probing a
    256 KiB window at a candidate offset costs nothing and a run of 32k zero
    tokens inside real text does not happen.
    """
    size = path.stat().st_size
    cap = min(size, cap) if cap else size
    fh = path.open("rb")
    def nz(off):
        fh.seek(off)
        return any(fh.read(262144))
    try:
        if not nz(0):
            return 0
        lo, step = 0, 1 << 20
        while lo + step < cap and nz(lo + step):
            lo += step
            step <<= 1
        hi = min(cap, lo + step)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if nz(mid):
                lo = mid
            else:
                hi = mid
        return hi // 2
    finally:
        fh.close()
s = ws/"data/tokens/bpe_6200000000.u16"
rec = s.with_suffix(".record.json")
launch = out/"pretrain_launch.json"
ck = out/"reader_xlarge_4k"/"checkpoint.json"
log = out/"pretrain.log"
st = {
    "checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "stream_alive": proc(10561) is not None,
    "watch_alive": proc(10694) is not None,
    "stream_tokens": fill(s) if s.exists() else 0,
    "stream_record": rec.exists(),
    "record": json.loads(rec.read_text()) if rec.exists() else None,
    "audit": json.loads((out/"stream_audit.json").read_text())[ "tokens" ] if (out/"stream_audit.json").exists() else None,
    "launch": json.loads(launch.read_text()) if launch.exists() else None,
    "pretrain_alive": (lambda p: proc(p) is not None if p else None)(
        json.loads(launch.read_text())["pid"] if launch.exists() else None),
    "checkpoint": json.loads(ck.read_text()) if ck.exists() else None,
    "log_tail": log.read_text()[-1200:] if log.exists() else None,
    "watcher_log": (out/"pretrain_launch.log").read_text()[-600:] if (out/"pretrain_launch.log").exists() else "",
}
print(json.dumps(st, indent=1))
'''


def main() -> int:
    ses = json.loads(SESSION.read_text())
    proc = subprocess.run(
        ["bash", str(EXEC), "--url", ses["base"], "--token", ses["token"], "-c", PROBE],
        capture_output=True, text=True, timeout=300,
    )
    text = proc.stdout
    start = text.find("{")
    if proc.returncode != 0 or start < 0:
        print(f"poll failed rc={proc.returncode}: {proc.stderr[-400:]}{text[-400:]}", file=sys.stderr)
        return 2
    state = json.loads(text[start:text.rfind("}") + 1])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
    pct = 100.0 * state["stream_tokens"] / 6.2e9
    line = (
        f"{state['checked']} stream={state['stream_tokens']:,} tok ({pct:.1f}%) "
        f"alive={state['stream_alive']} record={state['stream_record']} "
        f"audit={'yes' if state['audit'] else 'no'} "
        f"launch={'yes' if state['launch'] else 'no'} "
        f"ckpt={state['checkpoint']['step'] if state['checkpoint'] else 'none'}"
    )
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
