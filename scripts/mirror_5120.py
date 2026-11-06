#!/usr/bin/env python3
"""Mirror the 5,120-byte joint fine-tune and head artifacts out of a Molab sandbox.

The sandbox can be reclaimed mid-run. This poller keeps a verified local copy of
every small record and of each rolling training state, so a relaunch resumes from
the last pulled step instead of from zero. It never writes into the sandbox.

    MARIMO_TOKEN=... python3 scripts/mirror_5120.py --url https://sb-XXXX.sb.molab.run --interval 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/marimo/azurite_ws"
HEAD_REMOTE = "results/jev_query_5120"
SMALL = ("launch.json", "run.json", "train.jsonl", "train_meta.json",
         "train_state.json", "PRE_REGISTRATION.md", "PRE_REGISTRATION.sha256",
         "PRE_REGISTRATION_AMENDMENT_1.md", "PRE_REGISTRATION_AMENDMENT_1.sha256")
EVAL_SMALL = ("summary.json", "VERDICT.md", "MANIFEST.sha256",
              "per_example_old_a0.jsonl", "per_example_old_a2c.jsonl",
              "per_example_len_a0.jsonl", "per_example_len_a2c.jsonl")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(base: str, token: str, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    url = f"{base}/api/files/download?path=" + urllib.parse.quote(remote, safe="/")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "User-Agent": "azurite-mirror/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
    except Exception as exc:  # noqa: BLE001 - a missing file is normal before it is written
        tmp.unlink(missing_ok=True)
        print(f"skip {remote}: {exc}", flush=True)
        return False
    os.replace(tmp, local)
    return True


def pull_small(base: str, token: str, remote_dir: str, local_dir: Path,
               names: tuple[str, ...] = SMALL) -> None:
    for name in names:
        remote = f"{REMOTE_ROOT}/{remote_dir}/{name}"
        local = local_dir / name
        before = sha256(local) if local.exists() else None
        if download(base, token, remote, local):
            after = sha256(local)
            if before != after:
                print(f"pulled {remote} ({local.stat().st_size} B)", flush=True)


def pull_blob(base: str, token: str, remote: str, local: Path, want: str, label: str) -> None:
    have = sha256(local) if local.exists() else None
    if want == have:
        return
    print(f"pulling {label} sha={want[:12]}", flush=True)
    if download(base, token, remote, local):
        got = sha256(local)
        if got != want:
            local.unlink()
            raise SystemExit(f"{label} hash mismatch: {got} != {want}")
        print(f"verified {label} ({local.stat().st_size} B) sha={got[:12]}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dest", default=str(ROOT), type=Path)
    args = ap.parse_args()
    token = os.environ.get("MARIMO_TOKEN", "")
    if not token:
        raise SystemExit("MARIMO_TOKEN is not set")
    base = args.url.rstrip("/")
    head_local = args.dest / HEAD_REMOTE
    print(f"mirroring {base} -> {args.dest}", flush=True)

    while True:
        pull_small(base, token, HEAD_REMOTE, head_local)

        state_json = head_local / "train_state.json"
        if state_json.exists():
            try:
                state = json.loads(state_json.read_text())
            except json.JSONDecodeError:
                state = {}
            want = state.get("sha256")
            if want:
                pull_blob(base, token, f"{REMOTE_ROOT}/{HEAD_REMOTE}/train_state.pt",
                          head_local / "train_state.pt", want, "train_state.pt")

        meta_path = head_local / "train_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta = {}
            want = meta.get("head_sha256")
            if want:
                pull_blob(base, token, f"{REMOTE_ROOT}/{HEAD_REMOTE}/final.pt",
                          head_local / "final.pt", want, "final.pt")

        pull_small(base, token, f"{HEAD_REMOTE}/eval", head_local / "eval", EVAL_SMALL)

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
