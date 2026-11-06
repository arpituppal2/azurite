#!/usr/bin/env python3
"""Run one ladder rung as a sequence of bounded, resumable segments.

Why this exists: a Molab sandbox has a 90-minute idle drop and a 12-hour ceiling,
and its disk is disposable. So a multi-hour `pretrain.py` invocation is not a
reliable unit of work — the unit has to be a **segment**: a bounded slice that
checkpoints, exits 0, and leaves a record the next segment resumes from.

    pretrain.py --max-seconds N   stops cleanly, saves ckpt.pt, writes
                                  segment.json, and deliberately does NOT write
                                  summary.json (a partial rung must never be
                                  graded as a finished one)
    train_segments.py             runs that in a loop, updating driver.json after
                                  every transition, until summary.json exists

The whole run is therefore restartable: kill it, drop the sandbox, come back —
`--resume` loads the newest checkpoint and continues.

What a drop costs is bounded by `--segment-seconds` plus `--ckpt-every`, not by
the length of the run. With the mid rung at ~2.4 steps/s, a 200-step checkpoint
interval is roughly 80 seconds of lost work.

Usage:
    python3 scripts/train_segments.py --rung mid --tokens 100e6 \\
        --segment-seconds 1800 --max-segments 24
    # background, surviving the API call that launched it:
    nohup python3 scripts/train_segments.py ... > logs/train_mid.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results" / "pretrain"


def write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)  # atomic: a reader never sees a half-written record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="mid")
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--tokens", type=float, default=100e6)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segment-seconds", type=float, default=1800.0)
    ap.add_argument("--max-segments", type=int, default=24)
    ap.add_argument("--start-index", type=int, default=0,
                    help="first segment index; use after a recovered segment so history does not reuse an index")
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--mem-cap-gib", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--stream", default="",
                    help="explicit uint16 memmap, for multi-epoch budgets")
    # A rung trained at a second budget is a different run and must not be
    # written into the first one's directory: `l1_mid_byte` already has a
    # summary.json, and the driver would read that as "already complete".
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    ap.add_argument("--out-dir", default="",
                    help="explicit output directory (tests); default results/pretrain/l1_<rung>_<arm>_<tag>")
    args = ap.parse_args()

    name = f"l1_{args.rung}_{args.arm}" + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    driver_path = out_dir / "driver.json"
    summary_path = out_dir / "summary.json"
    segment_path = out_dir / "segment.json"
    live_path = out_dir / "segment.live.json"
    history_path = out_dir / "segments.jsonl"
    checkpoint_path = out_dir / "checkpoint.json"

    state: dict = {
        "rung": args.rung,
        "arm": args.arm,
        "tokens_target": int(args.tokens),
        "lr": args.lr,
        "warmup": args.warmup,
        "seed": args.seed,
        "segment_seconds": args.segment_seconds,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "running",
        "start_index": args.start_index,
        "segments": [],
    }
    write(driver_path, state)

    t_run = time.perf_counter()
    last_tokens = 0
    if checkpoint_path.exists():
        try:
            last_tokens = int(json.loads(checkpoint_path.read_text(encoding="utf-8"))["tokens_seen"])
        except (json.JSONDecodeError, KeyError, TypeError):
            last_tokens = 0

    for index in range(args.start_index, args.start_index + args.max_segments):
        if summary_path.exists():
            state["status"] = "complete"
            break

        # Resume only when there is state to resume. On the first segment of a
        # genuinely new directory there is none, and pretrain.py starts fresh.
        # If a checkpoint is missing but a log/state remains, `--resume` makes
        # pretrain.py refuse rather than silently restarting the rung at zero.
        has_ckpt = (out_dir / "ckpt.pt").exists()
        has_state = any((out_dir / n).exists() for n in
                        ("checkpoint.json", "train.jsonl", "run.json", "segment.json"))
        resume = has_ckpt or has_state

        state["current_segment"] = index
        state["status"] = "running"
        state["started_segment_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write(driver_path, state)
        write(live_path, {
            "schema": 2, "status": "running", "index": index,
            "started_at": state["started_segment_at"],
            "resume_tokens": last_tokens,
            "target_tokens": int(args.tokens),
            "segment_seconds": args.segment_seconds,
        })

        cmd = [
            sys.executable,
            str(PROJECT / "scripts" / "pretrain.py"),
            "--rung", args.rung,
            "--arm", args.arm,
            "--tokens", str(args.tokens),
            "--seq", str(args.seq),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--warmup", str(args.warmup),
            "--seed", str(args.seed),
            "--out", str(out_dir),
            "--max-seconds", str(args.segment_seconds),
            "--ckpt-every", str(args.ckpt_every),
            "--log-every", str(args.log_every),
            "--mem-cap-gib", str(args.mem_cap_gib),
            "--segment-index", str(index),
        ]
        if args.stream:
            cmd += ["--stream", args.stream]
        if resume:
            cmd.append("--resume")

        started = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
        elapsed = round(time.perf_counter() - started, 1)

        seg: dict = {
            "index": index,
            "returncode": proc.returncode,
            "seconds": elapsed,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if segment_path.exists():
            try:
                seg.update(json.loads(segment_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        seg["tail"] = [ln for ln in proc.stdout.strip().splitlines() if ln][-5:]
        seg["resume"] = resume
        state["segments"].append(seg)
        state["elapsed_total_s"] = round(time.perf_counter() - t_run, 1)

        # Append-only history: segment.json is the current marker and is
        # overwritten at the start of the next segment, so it cannot be the
        # only record that a segment happened. A sandbox death loses at most
        # the live segment, never the completed ones.
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                **seg,
                "driver_index": index,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }) + "\n")

        if proc.returncode != 0:
            state["status"] = "failed"
            state["error_tail"] = proc.stderr.strip().splitlines()[-15:]
            write(driver_path, state)
            print("\n".join(state["error_tail"]), flush=True)
            return proc.returncode

        observed = int(seg.get("tokens_seen", last_tokens) or last_tokens)
        if not summary_path.exists() and proc.returncode == 0 and observed <= last_tokens:
            state["status"] = "stalled"
            state["error_tail"] = [
                f"segment {index} made no token progress: {last_tokens:,} -> {observed:,}",
            ]
            write(driver_path, state)
            print(state["error_tail"][0], flush=True)
            return 1
        last_tokens = max(last_tokens, observed)

        print(json.dumps({k: seg.get(k) for k in
                          ("index", "seconds", "tokens_seen", "fraction")}), flush=True)
        write(driver_path, state)

        if summary_path.exists():
            state["status"] = "complete"
            break
    else:
        # Loop finished without completing the rung: being explicit beats
        # reporting "complete" for a run that merely ran out of segments.
        state["status"] = "segments-exhausted"

    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state["elapsed_total_s"] = round(time.perf_counter() - t_run, 1)

    # `summary.json` records only the FINAL segment's wall clock, because that
    # is the only process that sees summary.json being written. The honest
    # total lives here, in the driver, so it is stamped onto the summary rather
    # than left for a reader to infer from a number that means something else.
    if summary_path.exists():
        try:
            summ = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summ = None
        if summ is not None:
            summ["driver_elapsed_s"] = state["elapsed_total_s"]
            summ["segments"] = len(state["segments"])
            summ["segment_seconds"] = args.segment_seconds
            summary_path.write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")

    write(driver_path, state)
    print(f"status {state['status']} after {len(state['segments'])} segment(s)", flush=True)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
