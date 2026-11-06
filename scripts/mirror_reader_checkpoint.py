#!/usr/bin/env python3
"""Catch the reader mirror up to the newest checkpoint the live run has written.

One command does the whole catch-up:

    python scripts/mirror_reader_checkpoint.py

It reads the Molab session from `.stage/live_session_new.json` (base URL and
token, never argv), then:

1. reads the remote `checkpoint.json` and hard-links `ckpt.pt` to
   `.mirror_snapshots/step_<N>.pt` on the sandbox. A hard link is zero-I/O and
   freezes the inode, so the live run's atomic rename never moves, rewrites or
   deletes the bytes being fetched. `ckpt.pt` itself is never modified.
2. verifies the snapshot's sha256 against the marker before downloading.
3. downloads the stable snapshot, resuming a partial `.part` with HTTP Range,
   then verifies byte size and sha256.
4. rebuilds `checkpoint.json` from the checkpoint payload (not from a marker
   that may have advanced) and writes `MIRROR_VERIFIED.json` next to it.
5. refreshes `READER_REMOTE_STATUS.json`, `READER_LIVE_STATUS.json` and
   `reader_live_observation.json` from that verified mirror plus one fresh
   remote read, including the worst-case compute-at-risk figure.
6. removes the sandbox snapshot only after the local copy verifies.

Running it again is safe: an already-verified mirror for the same step is
re-used (no second 5 GB transfer), records are rewritten consistently, and a
stale `.part` is discarded rather than trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PARITY = PROJECT / "results/parity"
REMOTE_ROOT = "/marimo/azurite_ws"
# The live run writes into reader_restart_25500; the retired reader_xlarge_4k
# directory is kept only as the name of the archive layout inside a mirror.
REMOTE_RUN = REMOTE_ROOT + "/results/parity/reader_restart_25500"
ARCHIVE_LAYOUT = "results/parity/reader_xlarge_4k"
SNAPSHOT_DIR = REMOTE_ROOT + "/.mirror_snapshots"
EXEC = PROJECT / ".agents/skills/marimo-pair/scripts/execute-code.sh"
TOKENS_PER_STEP = 8192  # batch 2 x seq 4096, frozen geometry


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


def remote(base: str, token: str, code: str, timeout: float = 600.0) -> str:
    env = os.environ.copy()
    env["MARIMO_TOKEN"] = token
    proc = subprocess.run(
        ["bash", str(EXEC), "--url", base, "-c", code],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    if proc.returncode != 0:
        raise SystemExit(f"remote execution failed ({proc.returncode}): {proc.stderr[-400:]}")
    return proc.stdout


def snapshot(base: str, token: str) -> dict:
    code = f"""
import hashlib, json, os, shutil
root = {REMOTE_RUN!r}
snapdir = {SNAPSHOT_DIR!r}
os.makedirs(snapdir, exist_ok=True)
for attempt in range(8):
    ck = json.load(open(root + "/checkpoint.json"))
    step = int(ck["step"]); sha = ck["checkpoint"]["sha256"]; size = int(ck["checkpoint"]["bytes"])
    snap = f"{{snapdir}}/step_{{step}}.pt"
    if not os.path.exists(snap):
        try:
            os.link(root + "/ckpt.pt", snap)
        except OSError:
            shutil.copyfile(root + "/ckpt.pt", snap)
    h = hashlib.sha256()
    with open(snap, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    got = h.hexdigest()
    if got == sha:
        print("MIRROR " + json.dumps({{"step": step, "sha256": sha, "bytes": size,
            "snapshot_path": snap, "attempt": attempt, "saved_at": ck.get("saved_at"),
            "segment_index": ck.get("segment_index"), "config": ck.get("config")}}))
        break
    os.unlink(snap)
else:
    print("MIRROR_ERROR marker and hard-linked snapshot never agreed")
"""
    out = remote(base, token, code, timeout=300)
    for line in out.splitlines():
        if line.startswith("MIRROR "):
            return json.loads(line[len("MIRROR "):])
    raise SystemExit(f"no snapshot produced: {out[-400:]}")


def download(base: str, token: str, snap: dict, dest: Path) -> bool:
    """Download the frozen snapshot; return True if bytes were transferred."""
    size, sha = int(snap["bytes"]), snap["sha256"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    url = f"{base.rstrip('/')}/api/files/download?path=" + urllib.parse.quote(
        snap["snapshot_path"], safe="/")
    for attempt in range(6):
        start = part.stat().st_size if part.exists() else 0
        if start > size:
            part.unlink(missing_ok=True)
            start = 0
        if start == size:
            if sha256_file(part) == sha:
                os.replace(part, dest)
                return True
            part.unlink(missing_ok=True)
            start = 0
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "azurite-mirror/1.0",
        })
        if start:
            req.add_header("Range", f"bytes={start}-")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                if start and resp.status != 206:
                    part.unlink(missing_ok=True)
                    continue
                with part.open("ab" if start else "wb") as fh:
                    for block in iter(lambda: resp.read(1 << 22), b""):
                        fh.write(block)
        except Exception as exc:  # noqa: BLE001 - retry transport failures
            print(f"download attempt {attempt + 1} failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(2)
            continue
        if part.stat().st_size == size and sha256_file(part) == sha:
            os.replace(part, dest)
            return True
        print(f"attempt {attempt + 1}: size/sha mismatch, discarding partial", flush=True)
        part.unlink(missing_ok=True)
    raise SystemExit(f"could not download {snap['snapshot_path']} after 6 attempts")


def payload_meta(path: Path, cfg_hint: dict | None) -> dict:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    model = state["model"]
    total = sum(v.numel() for v in model.values() if hasattr(v, "numel"))
    tied = model["mlm.weight"].numel() if ("mlm.weight" in model and state["config"].get("tie_mlm_head")) else 0
    cfg = state["config"]
    meta = {
        "step": int(state["step"]),
        "tokens_seen": int(state["tokens_seen"]),
        "target_tokens": int(state["target_tokens"]),
        "config": cfg,
        "seq": int(state["seq"]),
        "batch": int(state["batch"]),
        "lr": float(state["lr"]),
        "seed": int(state["seed"]),
        "params_state_dict": int(total),
        "params_tied_head": int(tied),
        "params_encoder": int(total - tied),
    }
    del state, model
    if cfg_hint and cfg_hint != cfg:
        print("WARNING payload config differs from the marker config", flush=True)
    return meta


def fresh_read(base: str, token: str, snap_path: str) -> dict:
    code = f"""
import json, os
root = {REMOTE_RUN!r}
ck = json.load(open(root + "/checkpoint.json"))
row = json.loads(open(root + "/train.jsonl").read().strip().splitlines()[-1])
snap = {snap_path!r}
snapdir = {SNAPSHOT_DIR!r}
removed = False
if os.path.exists(snap):
    os.unlink(snap); removed = True
# Superseded snapshots pin old inodes; the verified local copy is the keeper.
for name in list(os.listdir(snapdir)) if os.path.isdir(snapdir) else []:
    path = os.path.join(snapdir, name)
    if name.startswith("step_") and name.endswith(".pt") and path != snap:
        try:
            os.unlink(path)
        except OSError:
            pass
print("READ " + json.dumps({{"checkpoint_step": ck["step"], "checkpoint_sha256": ck["checkpoint"]["sha256"],
    "tokens_seen": ck["tokens_seen"], "log_step": row["step"], "log_tokens": row["tokens_seen"],
    "loss": row["loss"], "tokens_per_s": row["tokens_per_s"], "peak_live_gib": row["peak_live_gib"],
    "swap_mb": row["swap_mb"], "snapshot_removed": removed}}))
"""
    out = remote(base, token, code, timeout=180)
    for line in out.splitlines():
        if line.startswith("READ "):
            return json.loads(line[len("READ "):])
    raise SystemExit(f"no fresh read: {out[-400:]}")


def main() -> int:
    global REMOTE_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=".stage/live_session_new.json")
    ap.add_argument("--dest", default="", help="mirror directory; defaults to reader_live_mirror_step<N>")
    ap.add_argument("--run-dir", default=REMOTE_RUN,
                    help="remote run directory to snapshot (default: the live reader_restart_25500 run)")
    ap.add_argument("--promote-primary", action="store_true",
                    help="make this step the resume-proven primary anchor (only with a resume proof for it)")
    args = ap.parse_args()
    REMOTE_RUN = args.run_dir.rstrip("/")
    session = read_json(PROJECT / args.session)
    base, token = session["base"].rstrip("/"), session["token"]

    snap = snapshot(base, token)
    step, sha, size = int(snap["step"]), snap["sha256"], int(snap["bytes"])
    mirror_dir = PROJECT / (args.dest or f"results/parity/reader_live_mirror_step{step}")
    ckpt = mirror_dir / ARCHIVE_LAYOUT / "ckpt.pt"
    record_path = ckpt.parent / "checkpoint.json"
    verified_path = mirror_dir / "MIRROR_VERIFIED.json"

    if ckpt.exists() and ckpt.stat().st_size == size and sha256_file(ckpt) == sha:
        transferred = False
        if record_path.exists():
            meta = read_json(record_path)
            meta = {k: meta[k] for k in ("step", "tokens_seen", "target_tokens", "config", "seq", "batch", "lr", "seed")}
            meta["params_state_dict"] = read_json(verified_path).get("params_state_dict")
            meta["params_tied_head"] = read_json(verified_path).get("params_tied_head")
            meta["params_encoder"] = read_json(verified_path).get("params_encoder")
        else:
            meta = payload_meta(ckpt, snap.get("config"))
    else:
        transferred = download(base, token, snap, ckpt)
        meta = payload_meta(ckpt, snap.get("config"))

    if meta["step"] != step:
        raise SystemExit(f"payload step {meta['step']} != marker step {step}")
    record = {
        "schema": 2,
        "status": "running",
        "step": step,
        "tokens_seen": meta["tokens_seen"],
        "target_tokens": meta["target_tokens"],
        "fraction": round(meta["tokens_seen"] / max(1, meta["target_tokens"]), 6),
        "segment_index": snap.get("segment_index", -1),
        "checkpoint": {"path": "ckpt.pt", "bytes": size, "sha256": sha},
        "config": meta["config"],
        "seq": meta["seq"], "batch": meta["batch"], "lr": meta["lr"], "seed": meta["seed"],
        "saved_at": snap.get("saved_at"),
        "archived_from": "sandbox hardlink snapshot of ckpt.pt (live file untouched)",
        "snapshot_path": snap["snapshot_path"],
    }
    write_json(record_path, record)
    verified = {
        "kind": "mirror_verified",
        "step": step,
        "tokens_seen": meta["tokens_seen"],
        "bytes": size,
        "sha256": sha,
        "local_path": str(ckpt.relative_to(PROJECT)),
        "method": "hardlink snapshot on the sandbox, HTTP download of the stable inode",
        "live_checkpoint_touched": False,
        "snapshot_removed_after_download": None,
        "download_transferred": transferred,
        "snapshot_attempts": int(snap.get("attempt", 0)) + 1,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params_state_dict": meta["params_state_dict"],
        "params_tied_head": meta["params_tied_head"],
        "params_encoder": meta["params_encoder"],
    }
    write_json(verified_path, verified)

    read = fresh_read(base, token, snap["snapshot_path"])
    verified["snapshot_removed_after_download"] = bool(read["snapshot_removed"])
    write_json(verified_path, verified)
    # Keep a copy beside the weights: the closure verifier resolves the mirror
    # from READER_REMOTE_STATUS.mirror_path and expects the record there.
    write_json(ckpt.parent / "MIRROR_VERIFIED.json", verified)

    now = verified["verified_at"]
    mirror_rel = str(ckpt.parent.relative_to(PROJECT))
    tokens = int(meta["tokens_seen"])
    prev_remote = read_json(PARITY / "READER_REMOTE_STATUS.json")
    primary_step = int(prev_remote.get("last_checkpoint_step") or 0)
    # The verifier pins READER_REMOTE_STATUS to the anchor that carries a real
    # isolated resume proof, and that proof needs GPU work we will not do while
    # the reader is training. So a newer mirror is registered alongside the
    # proven anchor unless --promote-primary says a proof exists for it.
    promote = bool(args.promote_primary) or primary_step == 0
    anchors = [a for a in prev_remote.get("mirror_anchors", []) if int(a.get("step", 0)) != step]
    anchors.append({
        "step": step, "tokens_seen": tokens, "bytes": size, "sha256": sha,
        "path": mirror_rel, "resume_proven": promote,
        "verified_locally": True,
        "note": ("Byte- and sha256-verified local copy taken while the live run continued; "
                 "isolated resume proof not regenerated (requires GPU)."),
    })
    anchors.sort(key=lambda a: int(a["step"]))
    remote_status = {
        **prev_remote,
        "observed_at": now, "status": "LIVE",
        "mirror_anchors": anchors,
        "newest_mirror_anchor_step": step,
        "newest_mirror_anchor_sha256": sha,
        "newest_mirror_anchor_path": mirror_rel,
    }
    if promote:
        remote_status.update({
            "last_checkpoint_step": step, "tokens_seen": tokens,
            "fraction": round(tokens / max(1, meta["target_tokens"]), 6),
            "checkpoint_sha256": sha, "mirror_path": mirror_rel,
        })
    remote_status["note"] = (
        f"Step {step:,} snapshotted by hardlink (live ckpt.pt untouched), downloaded, "
        "sha256 and byte size verified; the live run continues past it. "
        + ("" if promote else f"Primary resume-proven anchor remains step {primary_step:,}."))
    write_json(PARITY / "READER_REMOTE_STATUS.json", remote_status)

    live = read_json(PARITY / "READER_LIVE_STATUS.json")
    live["observed_at"] = now
    live["latest_mirrored_checkpoint"] = {
        "step": step, "tokens_seen": tokens, "target_tokens": meta["target_tokens"],
        "fraction": round(tokens / max(1, meta["target_tokens"]), 6),
        "bytes": size, "sha256": sha, "path": str(ckpt.relative_to(PROJECT)),
        "resume_proven": promote,
    }
    live["mirror_anchors"] = anchors
    live["latest_observed_remote_checkpoint"] = {
        "step": int(read["checkpoint_step"]), "tokens_seen": int(read["tokens_seen"]),
        "sha256": read["checkpoint_sha256"], "mirrored": int(read["checkpoint_step"]) == step,
        "note": "Fresh read after the mirror was verified.",
    }
    gap_steps = max(0, int(read["log_step"]) - step)
    gap_tokens = gap_steps * TOKENS_PER_STEP
    rate = float(read["tokens_per_s"])
    live["forecast"] = {
        "remaining_tokens": 6_200_000_000 - int(read["log_tokens"]),
        "eta_hours_at_measured_rate": round((6_200_000_000 - int(read["log_tokens"])) / rate / 3600, 1),
        "eta_days_at_measured_rate": round((6_200_000_000 - int(read["log_tokens"])) / rate / 86400, 2),
        "newest_anchor_step": step,
        "mirror_gap_steps": gap_steps,
        "mirror_gap_tokens": gap_tokens,
        "mirror_gap_gpu_hours": round(gap_tokens / rate / 3600, 2),
        "worst_case_compute_at_risk_minutes": round(gap_tokens / rate / 60, 1),
        "basis": f"log step {int(read['log_step']):,} / {int(read['log_tokens']):,} tokens / {rate:,} tok/s measured {now}",
    }
    write_json(PARITY / "READER_LIVE_STATUS.json", live)

    obs = read_json(PARITY / "reader_live_observation.json")
    obs.update({
        "observed_at": now,
        "newest_verified_anchor_step": step,
        "newest_verified_anchor_sha256": sha,
        "newest_verified_anchor_path": mirror_rel,
        "newest_verified_anchor_resume_proven": promote,
        "remote_checkpoint_step": int(read["checkpoint_step"]),
        "remote_checkpoint_tokens_seen": int(read["tokens_seen"]),
        "remote_checkpoint_sha256": read["checkpoint_sha256"],
        "log_step": int(read["log_step"]), "log_tokens_seen": int(read["log_tokens"]),
        "loss_last": read["loss"], "tokens_per_s": rate,
        "peak_live_gib": read["peak_live_gib"], "swap_mb": read["swap_mb"],
    })
    if promote:
        obs.update({
            "checkpoint_step": step, "tokens_seen": tokens, "checkpoint_sha256": sha,
            "fraction": round(tokens / 6_200_000_000, 6),
        })
    write_json(PARITY / "reader_live_observation.json", obs)

    print("MIRROR_RESULT " + json.dumps({
        "step": step, "sha256": sha, "bytes": size,
        "local": str(ckpt.relative_to(PROJECT)),
        "transferred": transferred,
        "params_encoder": meta["params_encoder"],
        "live_log_step": int(read["log_step"]),
        "live_checkpoint_step": int(read["checkpoint_step"]),
        "at_risk_minutes": live["forecast"]["worst_case_compute_at_risk_minutes"],
        "snapshot_removed": verified["snapshot_removed_after_download"],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
