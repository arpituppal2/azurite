#!/usr/bin/env python3
"""Upload files to a Molab sandbox with parallel base64 chunks.

The kernel endpoint is the only write path that works through the tunnel. The
sequential uploader in `molab_sync.sh` is reliable but caps a 3.2 GB corpus at
about twenty minutes, which does not fit a 30-minute GPU session. This uploader
sends independent chunks concurrently: the network transfer overlaps, while the
kernel writes each chunk to a distinct file and a final request assembles the
file, decompresses it and verifies the local SHA-256.

    MARIMO_TOKEN=... python3 scripts/molab_put_parallel.py --url ... \
        --put data/tokens/byte_corpus_v2.u16:/marimo/azurite_ws/data/tokens/byte_corpus_v2.u16
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 4_000_000  # base64 characters per request; ~3 MB raw


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def post(base: str, token: str, session: str, code: str, timeout: int = 300) -> str:
    # urllib is rejected by the tunnel's bot filter; curl is the client the
    # marimo-pair skill already uses, so shell out rather than reimplement TLS.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as body:
        json.dump({"code": code}, body)
        body_path = body.name
    try:
        last = ""
        for attempt in range(3):
            proc = subprocess.run([
                "curl", "-sN", "-X", "POST", f"{base}/api/kernel/execute",
                "-H", "Content-Type: application/json",
                "-H", f"Marimo-Session-Id: {session}",
                "-H", f"Authorization: Bearer {token}",
                "--data-binary", f"@{body_path}",
            ], capture_output=True, text=True, timeout=timeout)
            last = proc.stdout + proc.stderr
            if proc.returncode == 0 and '"success": true' in proc.stdout:
                return proc.stdout
            time.sleep(1 + attempt)
        raise SystemExit(f"remote error after retries: {last[-1000:]}")
    finally:
        Path(body_path).unlink(missing_ok=True)


def session_for(base: str, token: str) -> str:
    proc = subprocess.run([
        "curl", "-sS", "-m", "30", "-H", f"Authorization: Bearer {token}",
        f"{base}/api/sessions",
    ], capture_output=True, text=True, timeout=45)
    if proc.returncode != 0:
        raise SystemExit(f"session lookup failed: {proc.stderr[-500:]}")
    data = json.loads(proc.stdout)
    if not data:
        raise SystemExit(f"no active session on {base}")
    return next(iter(data))


def put_one(base: str, token: str, session: str, local: Path, remote: str,
            jobs: int, gz_level: int) -> None:
    raw_sha = sha256(local)
    workdir = Path(tempfile.mkdtemp(prefix="molab-put-"))
    gz = workdir / (local.name + ".gz")
    with local.open("rb") as src, gz.open("wb") as dst:
        with gzip.GzipFile(fileobj=dst, mode="wb", compresslevel=gz_level) as z:
            for block in iter(lambda: src.read(1 << 20), b""):
                z.write(block)
    b64 = workdir / (local.name + ".b64")
    with gz.open("rb") as src, b64.open("wb") as dst:
        base64.encode(src, dst)
    chunks = []
    with b64.open("rb") as fh:
        index = 0
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            chunks.append((index, block))
            index += 1
    xfer = f"/marimo/_pxfer_{uuid.uuid4().hex[:12]}"
    print(f"  {local} -> {remote}: {len(chunks)} chunk(s), {gz.stat().st_size:,} B gz", flush=True)

    def send(item: tuple[int, bytes]) -> int:
        index, block = item
        code = (
            "import pathlib\n"
            f"p = pathlib.Path({xfer!r})\n"
            "p.mkdir(parents=True, exist_ok=True)\n"
            f"(p / 'part_{index:05d}.b64').write_bytes({block!r})\n"
            f"print('ok', {index})\n"
        )
        post(base, token, session, code)

    def upload(items: list[tuple[int, bytes]]) -> None:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(send, item) for item in items]
            for future in as_completed(futures):
                future.result()

    upload(chunks)
    verify = (
        "import json, pathlib\n"
        f"p = pathlib.Path({xfer!r})\n"
        "parts = sorted(p.glob('part_*.b64'))\n"
        "print('SIZES', json.dumps([q.stat().st_size for q in parts]))\n"
    )
    expected = [CHUNK] * (len(chunks) - 1) + [len(chunks[-1][1])]
    for attempt in range(3):
        got = post(base, token, session, verify)
        match = re.search(r"SIZES (\[[0-9, ]*\])", got)
        if not match:
            raise SystemExit(f"chunk manifest missing: {got[-500:]}")
        sizes = json.loads(match.group(1))
        bad = [i for i, size in enumerate(sizes) if i >= len(expected) or size != expected[i]]
        if not bad and len(sizes) == len(expected):
            break
        print(f"  repairing {len(bad)} chunk(s): {bad[:12]}", flush=True)
        upload([chunks[i] for i in bad if i < len(chunks)])
    assemble = (
        "import base64, gzip, hashlib, pathlib, shutil\n"
        f"p = pathlib.Path({xfer!r})\n"
        "raw = b''.join(base64.b64decode(q.read_bytes()) for q in sorted(p.glob('part_*.b64')))\n"
        "data = gzip.decompress(raw)\n"
        f"out = pathlib.Path({remote!r})\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_bytes(data)\n"
        "print('bytes', len(data))\n"
        "print('sha256', hashlib.sha256(data).hexdigest())\n"
        f"print('MATCH' if hashlib.sha256(data).hexdigest() == {raw_sha!r} else 'MISMATCH')\n"
        "shutil.rmtree(p, ignore_errors=True)\n"
    )
    out = post(base, token, session, assemble, timeout=600)
    print(f"  {remote}: {out.strip().splitlines()[-3:]}", flush=True)
    if raw_sha not in out:
        raise SystemExit(f"remote hash mismatch for {remote}")
    shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", default=os.environ.get("MARIMO_TOKEN", ""))
    ap.add_argument("--put", action="append", required=True, metavar="LOCAL:REMOTE")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gzip-level", type=int, default=1)
    args = ap.parse_args()
    if not args.token:
        raise SystemExit("--token or MARIMO_TOKEN is required")
    base = args.url.rstrip("/")
    session = session_for(base, args.token)
    print(f"session {session}", flush=True)
    for spec in args.put:
        local_text, remote = spec.split(":", 1)
        local = Path(local_text)
        if not local.is_file():
            raise SystemExit(f"missing {local}")
        put_one(base, args.token, session, local, remote, args.jobs, args.gzip_level)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
