#!/usr/bin/env python3
"""Upload files to a Molab sandbox through the file-explorer multipart endpoint.

`POST /api/files/create` streams a multipart file to disk and is the only fast,
race-free write path through the tunnel. Cloudflare rejects bodies above about
100 MB, so larger files are split into 80 MB parts and concatenated in the
sandbox. The final SHA-256 is checked against the local file.

    MARIMO_TOKEN=... python3 scripts/molab_put_direct.py --url ... \
        --put results/pretrain/l1_large_byte_iso10v3/weights_only.pt:/marimo/azurite_ws/...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PART = 80 * 1024 * 1024
EXEC = Path(__file__).resolve().parent.parent / ".agents/skills/marimo-pair/scripts/execute-code.sh"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def upload_part(base: str, token: str, path: str, name: str, local: Path) -> None:
    proc = subprocess.run([
        "curl", "-sS", "-m", "300", "-X", "POST", f"{base}/api/files/create",
        "-H", f"Authorization: Bearer {token}",
        "-F", f"path={path}", "-F", "type=file", "-F", f"name={name}",
        "-F", f"file=@{local}",
    ], capture_output=True, text=True, timeout=360)
    if proc.returncode != 0 or '"success":true' not in proc.stdout.replace(" ", ""):
        raise SystemExit(f"upload failed for {name}: {proc.returncode} {proc.stdout[-500:]} {proc.stderr[-300:]}")


def remote_python(base: str, token: str, code: str) -> str:
    proc = subprocess.run([
        "bash", str(EXEC), "--url", base, "--token", token, "-c", code,
    ], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit(f"remote code failed: {proc.stderr[-800:]}{proc.stdout[-800:]}")
    return proc.stdout


def put(base: str, token: str, local: Path, remote: str, jobs: int) -> None:
    want = sha256(local)
    remote_path = Path(remote)
    remote_dir = str(remote_path.parent)
    name = remote_path.name
    work = Path(tempfile.mkdtemp(prefix="molab-direct-"))
    try:
        if local.stat().st_size <= PART:
            upload_part(base, token, remote_dir, name, local)
        else:
            # Materialise one part per worker, never the whole file: a 5.2 GB
            # checkpoint split into 80 MB parts used to need 5.2 GB of scratch
            # on a host that had 10 GB free, and the whole transfer died with
            # ENOSPC after 9 parts. Peak scratch is now jobs * PART.
            count = (local.stat().st_size + PART - 1) // PART

            def send(index: int) -> None:
                part = work / f"{name}.part{index:03d}"
                with local.open("rb") as src:
                    src.seek(index * PART)
                    part.write_bytes(src.read(PART))
                try:
                    upload_part(base, token, remote_dir, f".{name}.part{index:03d}", part)
                finally:
                    part.unlink(missing_ok=True)

            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = [pool.submit(send, i) for i in range(count)]
                for future in as_completed(futures):
                    future.result()
            code = (
                "import hashlib, pathlib\n"
                f"d = pathlib.Path({remote_dir!r})\n"
                f"parts = sorted(d.glob('.{name}.part*'))\n"
                f"out = d / {name!r}\n"
                "with out.open('wb') as fh:\n"
                "    for p in parts:\n"
                "        fh.write(p.read_bytes())\n"
                "        p.unlink()\n"
                "h = hashlib.sha256()\n"
                "with out.open('rb') as fh:\n"
                "    for b in iter(lambda: fh.read(1 << 20), b''):\n"
                "        h.update(b)\n"
                "print('SIZE', out.stat().st_size)\n"
                "print('SHA', h.hexdigest())\n"
            )
            out = remote_python(base, token, code)
            if want not in out:
                raise SystemExit(f"remote hash mismatch for {remote}: {out[-500:]}")
        print(f"  {local} -> {remote} ({local.stat().st_size:,} B)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", default=os.environ.get("MARIMO_TOKEN", ""))
    ap.add_argument("--put", action="append", required=True, metavar="LOCAL:REMOTE")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if not args.token:
        raise SystemExit("--token or MARIMO_TOKEN is required")
    base = args.url.rstrip("/")
    for spec in args.put:
        local_text, remote = spec.split(":", 1)
        local = Path(local_text)
        if not local.is_file():
            raise SystemExit(f"missing {local}")
        put(base, args.token, local, remote, args.jobs)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
