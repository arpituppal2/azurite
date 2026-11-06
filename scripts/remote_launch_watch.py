#!/usr/bin/env python3
"""Finish the GPU handoff on Molab itself.

Waits for the 122 uploaded stream parts, assembles them, verifies the frozen
SHA-256, runs the 20-step 4k probe, and only then launches the supervised
reader pretrain. Every transition is written under results/parity/gpu so the
record survives the session. No optimizer runs unless the probe passes.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

WS = pathlib.Path("/marimo/azurite_ws")
TOK = WS / "data" / "tokens"
PARTS = TOK.glob(".bpe_6200000000.u16.part*")
FINAL = TOK / "bpe_6200000000.u16"
SHA = "e8da0a60483e45ab194ad861979456abd50d65a7e728feca9de727732dd4deec"
BYTES = 10_186_393_124
N_PARTS = 122
GPU = WS / "results" / "parity" / "gpu"
OUT = WS / "results" / "parity" / "reader_xlarge_4k"


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write(name: str, obj: dict) -> None:
    GPU.mkdir(parents=True, exist_ok=True)
    (GPU / name).write_text(json.dumps(obj, indent=1) + "\n")


def main() -> int:
    GPU.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 2700
    while time.time() < deadline:
        parts = sorted(TOK.glob(".bpe_6200000000.u16.part*"))
        total = sum(p.stat().st_size for p in parts)
        if len(parts) == N_PARTS and total == BYTES:
            time.sleep(5)
            parts = sorted(TOK.glob(".bpe_6200000000.u16.part*"))
            total = sum(p.stat().st_size for p in parts)
            if len(parts) == N_PARTS and total == BYTES:
                break
        time.sleep(5)
    else:
        write("watcher_failure.json", {"at": stamp(), "reason": "timed out waiting for 122 parts",
                                       "parts": len(parts), "bytes": total})
        return 2

    if FINAL.exists():
        FINAL.unlink()
    with FINAL.open("wb") as fh:
        for p in parts:
            fh.write(p.read_bytes())
    h = hashlib.sha256()
    with FINAL.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    record = {"at": stamp(), "parts": len(parts), "bytes": FINAL.stat().st_size,
              "sha256": digest, "matches_frozen": digest == SHA}
    write("stream_verified.json", record)
    print(json.dumps(record), flush=True)
    if digest != SHA:
        return 3
    for p in parts:
        p.unlink()

    probe_path = GPU / "probe_20step.json"
    cmd = [sys.executable, "scripts/bench_4k.py", "--seq", "4096", "--batch", "2",
           "--steps", "20", "--dtype", "bf16", "--out", str(probe_path)]
    started = time.time()
    proc = subprocess.run(cmd, cwd=WS, capture_output=True, text=True, timeout=1800)
    text = proc.stdout + proc.stderr
    (GPU / "probe_20step.log").write_text(text)
    rate = None
    m = re.search(r"([0-9][0-9,.]*)\s*tok/s", text)
    if m:
        rate = float(m.group(1).replace(",", ""))
    probe = {}
    try:
        probe = json.loads(probe_path.read_text())
    except Exception:
        pass
    if rate is None:
        for key in ("tokens_per_s", "tok_per_s", "throughput"):
            if isinstance(probe.get(key), (int, float)):
                rate = float(probe[key])
                break
    record = {"at": stamp(), "returncode": proc.returncode, "seconds": round(time.time() - started, 2),
              "tokens_per_s": rate, "probe": probe, "gate": ">=20000 tok/s",
              "passed": proc.returncode == 0 and rate is not None and rate >= 20000}
    write("probe_verdict.json", record)
    print(json.dumps(record), flush=True)
    if not record["passed"]:
        return 4

    log = (GPU / "supervisor.log").open("a", encoding="utf-8")
    sup = subprocess.Popen(
        [sys.executable, "scripts/gpu_supervisor.py",
         "--out", "results/parity/reader_xlarge_4k",
         "--log-dir", "results/parity/gpu",
         "--segment-seconds", "3600", "--mem-cap-gib", "60",
         "--start-segment-index", "1"],
        cwd=WS, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    launch = {"at": stamp(), "supervisor_pid": sup.pid, "cmd": "scripts/gpu_supervisor.py",
              "out": str(OUT), "probe_tokens_per_s": rate, "stream_sha256": digest}
    write("launch.json", launch)
    print(json.dumps(launch), flush=True)

    bigram_log = (GPU / "bigram.log").open("a", encoding="utf-8")
    bigram = subprocess.Popen(
        [sys.executable, "scripts/bigram_baseline_bpe.py",
         "--stream", str(FINAL),
         "--out", "results/parity/bigram_bpe_6200000000.json"],
        cwd=WS, stdin=subprocess.DEVNULL, stdout=bigram_log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write("bigram_launch.json", {"at": stamp(), "pid": bigram.pid})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
