#!/usr/bin/env python3
"""Run a declared experiment plan back-to-back on one GPU, with no idle gap.

The plan is a JSON file, so what will be run is auditable before it runs and the
sequence is not reconstructed from shell history afterwards:

    [{"name": "small_ctx256", "rung": "small", "seq": 256,
      "tokens": 142609920, "tag": "ctx256"}, ...]

Each job trains through `train_segments.py` (bounded, resumable, checkpoints every
200 steps) and is then scored by `eval_full.py` (exhaustive, no sampling). Both are
skipped when their outputs already exist, so re-running this command after a
sandbox drop resumes rather than restarts.

Why the state file is written the way it is
-------------------------------------------
An earlier ladder driver set `status="running"` *before* launching the child and
never cleared it, so its queue file reported a rung as running while nothing of
the sort was on the card. A status record that cannot be wrong is worth more than
one that is usually right, so:

- a job is `waiting` while the card is busy with a prior process,
- `starting` between claiming the job and the child existing,
- `running` only after `Popen` returns a live PID,
- and every transition is stamped with wall-clock epoch seconds, so the gap
  between one job's end and the next job's start is *measured*, not asserted.

`GPU_QUEUE_STATE.json` is the single source of truth; a notebook cell reads it.

Two operational rules this enforces, both of which cost real GPU-hours when done
by hand:

1. **No gap between jobs.** The next run starts when the previous process exits,
   not when a human notices.
2. **GPU busied before anything else.** `--wait-pid` blocks until a prior run has
   released the card, so this can be launched while something else is mid-segment
   without either job contending for SM time.

Usage:
    python3 scripts/gpu_queue.py --plan research/queue_iso10.json \
        --stream data/tokens/byte_corpus_v2.u16 --wait-pid 12345
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results" / "pretrain"
LOGS = PROJECT / "logs"
STATE = RESULTS / "GPU_QUEUE_STATE.json"


def iso(epoch: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(epoch if epoch else time.time()))


def write(obj: dict) -> None:
    """Atomic replace: a reader never sees a half-written state file."""
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    tmp.replace(STATE)


def event(state: dict, what: str, **fields) -> None:
    """Append a timestamped transition. The idle-gap table is derived from these."""
    state.setdefault("events", []).append({"ts": round(time.time(), 3), "at": iso(), "event": what, **fields})
    # Bounded: the tail is what a reader needs, and an unbounded log makes the
    # state file expensive to poll from a notebook cell every few seconds.
    state["events"] = state["events"][-200:]


def others_running() -> list[str]:
    """Processes that are using the GPU for training, excluding this queue."""
    out = []
    me = os.getpid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            cl = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if "pretrain.py" in cl or "train_segments.py" in cl:
            out.append(f"{p.name}: {cl[:80]}")
    return out


def card_is_free(wait_pid: int) -> tuple[bool, list[str]]:
    busy = others_running()
    if wait_pid and Path(f"/proc/{wait_pid}").exists():
        busy = [f"pid {wait_pid} (requested via --wait-pid)"] + busy
    return (not busy), busy


def run_job(job: dict, stream: str, arm: str, segment_seconds: float,
            state: dict, prev_end_ts: float) -> dict:
    name = job["name"]
    out_dir = RESULTS / f"l1_{job['rung']}_{arm}_{job['tag']}"
    log = LOGS / f"queue_{name}.log"

    rec = state["jobs"].setdefault(name, {})
    rec.update({"name": name, "dir": out_dir.name, "rung": job["rung"],
                "seq": job["seq"], "tokens_target": int(job["tokens"]),
                "status": "starting", "started": iso(), "started_ts": round(time.time(), 3)})
    rec["idle_s"] = round(rec["started_ts"] - prev_end_ts, 1)
    state["status"] = "starting"
    state["current"] = name
    event(state, "job_start", name=name, idle_s=rec["idle_s"])
    write(state)

    train_started_ts = time.time()
    if not (out_dir / "summary.json").exists():
        cmd = [
            sys.executable, str(PROJECT / "scripts" / "train_segments.py"),
            "--rung", job["rung"], "--arm", arm,
            "--tokens", str(int(job["tokens"])), "--seq", str(job["seq"]),
            "--batch", str(int(job.get("batch", 16))),
            "--stream", stream, "--tag", job["tag"],
            "--segment-seconds", str(segment_seconds),
            "--max-segments", str(int(job.get("max_segments", 600))),
        ]
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=str(PROJECT), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            # Running means a child exists. This line is the reason the status
            # file can be trusted: `starting` cannot be mistaken for `running`.
            rec["pid"] = proc.pid
            rec["status"] = "running"
            state["status"] = "running"
            event(state, "train_popen", name=name, pid=proc.pid)
            write(state)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
                fh.flush()
            rec["exit"] = proc.wait()
        rec["train_seconds"] = round(time.time() - train_started_ts, 1)
        if rec["exit"] != 0:
            rec["status"] = "failed"
            rec["finished"] = iso()
            rec["finished_ts"] = round(time.time(), 3)
            event(state, "job_failed", name=name, exit=rec["exit"])
            state["status"] = "failed"
            return rec
    else:
        rec["train_seconds"] = 0.0
        rec["exit"] = 0
        event(state, "train_skipped", name=name, reason="summary.json exists")

    score = out_dir / "full.json"
    rec["status"] = "scoring"
    state["status"] = "scoring"
    write(state)
    score_started = time.time()
    if not score.exists():
        cmd = [
            sys.executable, str(PROJECT / "scripts" / "eval_full.py"),
            "--run", str(out_dir), "--rung", job["rung"],
            "--stream", stream, "--seq", str(job["seq"]), "--out", str(score),
        ]
        p = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
        if p.returncode != 0:
            rec["status"] = "scored_failed"
            rec["stderr"] = p.stderr.strip().splitlines()[-4:]
            rec["finished"] = iso()
            rec["finished_ts"] = round(time.time(), 3)
            event(state, "score_failed", name=name)
            state["status"] = "failed"
            return rec
        print(p.stdout.strip(), flush=True)
    rec["score_seconds"] = round(time.time() - score_started, 1)

    r = json.loads(score.read_text())
    rec.update({"status": "complete", "bits_per_byte": r["bits_per_byte"],
                "params": r["params"], "seq": job["seq"], "tokens": r["tokens_seen"]})
    rec["finished"] = iso()
    rec["finished_ts"] = round(time.time(), 3)
    rec["wall_seconds"] = round(rec["finished_ts"] - rec["started_ts"], 1)
    event(state, "job_complete", name=name, bits_per_byte=r["bits_per_byte"])
    state["status"] = "running"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--segment-seconds", type=float, default=3000.0)
    ap.add_argument("--wait-pid", type=int, default=0)
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    jobs = plan["jobs"] if isinstance(plan, dict) else plan

    state: dict = {"stream": args.stream, "plan": args.plan, "arm": args.arm,
                   "queue_started": iso(), "queue_started_ts": round(time.time(), 3),
                   "status": "starting", "current": None, "jobs": {}, "events": []}
    if STATE.exists():
        try:
            old = json.loads(STATE.read_text(encoding="utf-8"))
            state["jobs"] = old.get("jobs", {})
            state["events"] = old.get("events", [])
        except json.JSONDecodeError:
            pass
    event(state, "queue_start", plan=args.plan, wait_pid=args.wait_pid or None)
    write(state)

    # --- wait for the card, labelled honestly ---------------------------------
    free, busy = card_is_free(args.wait_pid)
    while not free:
        state["status"] = "waiting"
        state["waiting_on"] = busy[:3]
        if "wait_began" not in state:
            state["wait_began"] = iso()
            state["wait_began_ts"] = round(time.time(), 3)
            event(state, "wait_begin", on=busy[:2])
        write(state)
        time.sleep(args.poll_seconds)
        free, busy = card_is_free(args.wait_pid)

    state["card_free_at"] = iso()
    state["card_free_ts"] = round(time.time(), 3)
    state["gpu_wait_s"] = round(state["card_free_ts"] - state["queue_started_ts"], 1)
    state["status"] = "starting"
    state["waiting_on"] = []
    event(state, "card_free")
    write(state)
    print(f"card free at {state['card_free_at']} after {state['gpu_wait_s']:.0f}s waiting", flush=True)

    prev_end_ts = state["card_free_ts"]
    t0 = time.perf_counter()
    for job in jobs:
        done = state["jobs"].get(job["name"], {})
        if done.get("status") == "complete":
            print(f"[{job['name']}] complete — next", flush=True)
            prev_end_ts = done.get("finished_ts", prev_end_ts)
            continue
        print(f"\n=== {job['name']}: {job['rung']} seq {job['seq']} "
              f"{int(job['tokens']):,} tokens ===", flush=True)
        rec = run_job(job, args.stream, args.arm, args.segment_seconds, state, prev_end_ts)
        state["elapsed_s"] = round(time.perf_counter() - t0, 1)
        write(state)
        print(json.dumps({k: rec.get(k) for k in
                          ("name", "status", "bits_per_byte", "train_seconds", "idle_s")}), flush=True)
        if rec["status"] != "complete":
            print("stopping the queue so the failure stays visible", flush=True)
            return 1
        prev_end_ts = rec["finished_ts"]

    state["status"] = "complete"
    state["finished"] = iso()
    state["finished_ts"] = round(time.time(), 3)
    event(state, "plan_complete")
    write(state)
    print(f"\nplan complete in {state['elapsed_s'] / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
