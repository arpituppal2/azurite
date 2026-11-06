#!/usr/bin/env python3
"""Mirror experiment artifacts out of a disposable Molab sandbox.

The 2026-10-28 reclamation destroyed the v2 corpus, the protocol checkpoint and a
running 1.13B-token ladder job because they existed only on the sandbox disk.This process exists so that failure costs
wall-clock instead of results: it polls the remote `results/` tree, copies changed
files locally, and pulls a checkpoint as soon as `checkpoint.json` names a
complete, hashed checkpoint. A segment marker is no longer required: a sandbox
defect can kill the process between checkpoints, and the last verified weights
must already be off the box. Every pulled checkpoint is checked against the
sha256 in `checkpoint.json` before it replaces the local copy.

Usage:
    MARIMO_TOKEN=... python3 scripts/molab_pull.py \
        --url https://sb-XXXX.sb.molab.run/ --once
    MARIMO_TOKEN=... python3 scripts/molab_pull.py --url ... --interval 60

The token is read from the environment, never from a file or argv, so it does not
appear in `ps`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EXEC = PROJECT / ".agents/skills/marimo-pair/scripts/execute-code.sh"
REMOTE_ROOT = "/marimo/azurite_ws"
DEST = PROJECT / "results/recovered_20261028"
LOCAL_STATE = DEST / ".pull_state.json"
INCLUDE: list[str] = []


def remote(code: str, timeout: float = 120.0) -> str:
    env = os.environ.copy()
    token = env.get("MARIMO_TOKEN", "")
    if not token:
        raise SystemExit("MARIMO_TOKEN is not set")
    proc = subprocess.run(
        ["bash", str(EXEC), "--url", args.url, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"remote execution failed ({proc.returncode}): {proc.stderr[-500:]}")
    return proc.stdout


def manifest() -> dict[str, dict]:
    code = f"""
import json, os, pathlib
root = pathlib.Path({REMOTE_ROOT!r})
out = []
for dirpath, dirnames, filenames in os.walk(root / 'results', followlinks=False):
    # `source/data` is a symlink into the 3.3 GB corpus; following it turns one
    # manifest into a walk of every dataset and model cache on the box.
    dirnames[:] = [d for d in dirnames
                   if d not in ('__pycache__', '.git')
                   and not (pathlib.Path(dirpath) / d).is_symlink()]
    for name in filenames:
        if name.startswith('._') or name.endswith(('.tmp', '.part', '.lock')):
            continue
        p = pathlib.Path(dirpath) / name
        if not p.is_file() or p.is_symlink():
            continue
        rel = str(p.relative_to(root))
        st = p.stat()
        out.append({{'path': rel, 'size': st.st_size, 'mtime': st.st_mtime}})
print(json.dumps(out))
"""
    raw = remote(code).strip().splitlines()
    if not raw:
        return {}
    return {row["path"]: row for row in json.loads(raw[-1])}


def marker_for(path: str, entries: dict[str, dict]) -> dict | None:
    """A completion marker for checkpoint formats that predate checkpoint.json."""
    directory = str(Path(path).parent)
    markers = [f"{directory}/checkpoint.json", f"{directory}/segment.json",
               f"{directory}/summary.json", f"{directory}/result.json"]
    found = [entries[m] for m in markers if m in entries]
    return max(found, key=lambda e: e["mtime"]) if found else None


def archive_boundary(local: Path, row: dict) -> None:
    """Keep the weights at each segment boundary, not only the newest file.

    The normal mirror path is one `ckpt.pt` that every checkpoint replaces. A
    segment boundary is the unit the next pass resumes from, so retain those
    weights under `checkpoints/` and leave the rolling copy alone.
    """
    # Archived segment checkpoints already live under checkpoints/; copying
    # them again would make a second, nested copy of every 1.3 GB file.
    if "checkpoints" in local.parts:
        return
    if row.get("status") not in ("truncated", "complete"):
        return
    sha = row.get("sha256")
    if not sha or not local.exists():
        return
    seg = row.get("segment_index")
    step = row.get("step")
    archive = DEST / "checkpoints" / f"seg_{seg}_step_{step}_{str(sha)[:12]}.pt"
    if archive.exists():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, archive)
    print(f"archived boundary checkpoint {archive} ({archive.stat().st_size:,} B)", flush=True)


def pull_file(rel: str, entry: dict, expected_sha256: str | None = None) -> Path:
    # Never write into the canonical results tree: a live pull once overwrote
    # the pre-loss l1_tiny_byte record at the same relative path.
    local = DEST / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    # Keep `/` literal: the endpoint rejects percent-encoded separators with 403.
    url = f"{args.url.rstrip('/')}/api/files/download?path=" + urllib.parse.quote(
        REMOTE_ROOT + "/" + rel, safe="/")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {os.environ['MARIMO_TOKEN']}",
        # Cloudflare rejects the default Python-urllib user agent with error 1010.
        "User-Agent": "azurite-puller/1.0",
    })
    got = 0
    digest = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as out:
        while True:
            block = resp.read(1024 * 1024)
            if not block:
                break
            out.write(block)
            digest.update(block)
            got += len(block)
    expected = int(entry["size"])
    if got != expected:
        # Append-only logs can grow between the manifest and the download.
        # A short read is still a failure; a longer one is safe for .jsonl/.log.
        if got > expected and rel.endswith((".jsonl", ".log")):
            pass
        else:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"size mismatch for {rel}: got {got}, expected {expected}")
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch for {rel}: got {digest.hexdigest()}, "
            f"checkpoint.json says {expected_sha256}"
        )
    os.replace(tmp, local)
    return local


def main() -> int:
    state: dict[str, dict] = {}
    if LOCAL_STATE.exists():
        try:
            state = json.loads(LOCAL_STATE.read_text())
        except json.JSONDecodeError:
            state = {}
    entries = manifest()
    if INCLUDE:
        entries = {
            rel: entry for rel, entry in entries.items()
            if any(rel.startswith(prefix) for prefix in INCLUDE)
        }
    pulled = 0
    skipped = 0
    verified: dict[str, dict] = {}
    for rel, entry in sorted(entries.items()):
        local = DEST / rel
        is_ckpt = rel.endswith((".pt", ".safetensors"))
        if is_ckpt:
            # Preferred protocol: checkpoint.json is written by pretrain.py on
            # every checkpoint, including mid-segment ones. Pull the small
            # marker first, then the weights it names, and verify the hash.
            meta_rel = str(Path(rel).parent / "checkpoint.json")
            meta = None
            if meta_rel in entries:
                marker = entries[meta_rel]
                marker_local = DEST / meta_rel
                old_marker = state.get(meta_rel, {})
                marker_changed = not (
                    old_marker.get("mtime") == marker["mtime"]
                    and old_marker.get("size") == marker["size"]
                    and marker_local.exists()
                )
                try:
                    if marker_changed:
                        pull_file(meta_rel, marker)
                        state[meta_rel] = {"mtime": marker["mtime"], "size": marker["size"]}
                    meta = json.loads(marker_local.read_text())
                except Exception as exc:
                    print(f"FAILED {meta_rel}: {exc}", file=sys.stderr, flush=True)
                    meta = None
            if meta is not None:
                described = meta.get("checkpoint") or {}
                expected_bytes = described.get("bytes")
                expected_sha = described.get("sha256")
                if described.get("path") not in (None, local.name):
                    skipped += 1
                    continue
                if expected_bytes is not None and int(expected_bytes) != int(entry["size"]):
                    skipped += 1
                    continue
                old = state.get(rel, {})
                if (expected_sha and old.get("sha256") == expected_sha
                        and local.exists() and local.stat().st_size == entry["size"]):
                    skipped += 1
                    verified[rel] = {**described, "step": meta.get("step"),
                                     "tokens_seen": meta.get("tokens_seen"),
                                     "status": meta.get("status"),
                                     "segment_index": meta.get("segment_index")}
                    archive_boundary(local, verified[rel])
                    continue
                try:
                    pull_file(rel, entry, expected_sha256=expected_sha)
                    state[rel] = {"mtime": entry["mtime"], "size": entry["size"],
                                  "sha256": expected_sha}
                    verified[rel] = {**described, "step": meta.get("step"),
                                     "tokens_seen": meta.get("tokens_seen"),
                                     "status": meta.get("status"),
                                     "segment_index": meta.get("segment_index")}
                    archive_boundary(local, verified[rel])
                    pulled += 1
                    print(f"pulled {rel} ({entry['size']:,} B, sha verified)", flush=True)
                except Exception as exc:
                    print(f"FAILED {rel}: {exc}", file=sys.stderr, flush=True)
                continue
            # Legacy fallback for directories that only have segment/summary
            # markers and no checkpoint.json.
            marker = marker_for(rel, entries)
            if marker is None:
                skipped += 1
                continue
            marker_rel = marker["path"]
            marker_state = state.get(marker_rel, {})
            unchanged_marker = (
                marker_state.get("mtime") == marker["mtime"]
                and marker_state.get("size") == marker["size"]
            )
            if unchanged_marker and local.exists() and local.stat().st_size == entry["size"]:
                skipped += 1
                continue
        else:
            old = state.get(rel, {})
            if (old.get("mtime") == entry["mtime"] and old.get("size") == entry["size"]
                    and local.exists() and local.stat().st_size == entry["size"]):
                skipped += 1
                continue
        try:
            pull_file(rel, entry)
            state[rel] = {"mtime": entry["mtime"], "size": entry["size"]}
            pulled += 1
            print(f"pulled {rel} ({entry['size']:,} B)", flush=True)
        except Exception as exc:  # keep the other artifacts flowing
            print(f"FAILED {rel}: {exc}", file=sys.stderr, flush=True)
    LOCAL_STATE.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    if verified:
        (DEST / "MIRROR_VERIFIED.json").write_text(json.dumps({
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoints": verified,
        }, indent=2, sort_keys=True) + "\n")
        # Preserve the pull history; MIRROR_VERIFIED.json only names the latest
        # checkpoint and a long run overwrites it every few minutes.
        history = DEST / "CHECKPOINTS.jsonl"
        seen: set[tuple[str, str]] = set()
        if history.exists():
            for line in history.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    seen.add((str(row.get("path")), str(row.get("sha256"))))
                except json.JSONDecodeError:
                    continue
        with history.open("a", encoding="utf-8") as fh:
            for rel, row in sorted(verified.items()):
                key = (rel, str(row.get("sha256")))
                if key in seen:
                    continue
                fh.write(json.dumps({
                    "path": rel,
                    "sha256": row.get("sha256"),
                    "bytes": row.get("bytes"),
                    "step": row.get("step"),
                    "tokens_seen": row.get("tokens_seen"),
                    "status": row.get("status"),
                    "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }) + "\n")
                seen.add(key)
    print(f"poll {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: "
          f"{pulled} pulled, {skipped} unchanged", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dest", default=str(DEST), help="isolated local mirror root")
    ap.add_argument("--include", action="append", default=[],
                    help="only mirror paths starting with this prefix (repeatable)")
    args = ap.parse_args()
    globals()["DEST"] = Path(args.dest)
    globals()["LOCAL_STATE"] = DEST / ".pull_state.json"
    globals()["INCLUDE"] = args.include
    if args.once:
        raise SystemExit(main())
    while True:
        try:
            main()
        except Exception as exc:
            print(f"poll error: {exc}", file=sys.stderr, flush=True)
        time.sleep(args.interval)
