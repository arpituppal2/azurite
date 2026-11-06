#!/usr/bin/env python3
"""Watch a live run and score it at fixed token milestones, unattended.

A 500M-token rung takes hours and the sandbox can vanish at any point. Sampling
the trajectory by hand means sitting in front of a terminal; not sampling it
means having only an endpoint, which cannot distinguish "still improving" from
"already saturated" — the exact question the token-starvation test is asking.

So this runs alongside training, and whenever the checkpoint crosses a token
milestone it copies the weights, scores them with the exhaustive protocol, and
deletes the copy. Only JSON survives, so a ten-point trajectory costs a few
megabytes instead of 14 GB of duplicated checkpoints.

The copy is taken with `shutil.copy2` while training may be writing the file.
`torch.save` writes in place, so a copy can catch a half-written file; a
truncated zip fails loudly in `torch.load` and the milestone is retried on the
next pass rather than being recorded as a number. A wrong measurement that looks
plausible is the failure mode this project has already been burned by.

Usage:
    python3 scripts/track_curve.py --run l1_large_byte_x5 --rung large \
        --stream data/tokens/byte_100000000.u16 --every 50e6 --until 500e6
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
PRETRAIN = PROJECT / "results" / "pretrain"


def ckpt_step(path: Path) -> tuple[int, int] | None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None  # mid-write; retry later
    return int(state.get("step", -1)), int(state.get("tokens_seen", -1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory under results/pretrain")
    ap.add_argument("--rung", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--every", type=float, default=50e6)
    ap.add_argument("--until", type=float, default=500e6)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--poll", type=float, default=60.0)
    args = ap.parse_args()

    run_dir = PRETRAIN / args.run
    ckpt = run_dir / "ckpt.pt"
    traj_path = run_dir / "trajectory.json"
    traj = json.loads(traj_path.read_text()) if traj_path.exists() else {"points": []}
    done = {p["milestone"] for p in traj["points"]}

    milestones = []
    m = args.every
    while m <= args.until + 1:
        milestones.append(int(m))
        m += args.every

    staging = PRETRAIN / "_track_tmp"
    staging.mkdir(parents=True, exist_ok=True)

    # A milestone the run already passed before tracking started cannot be
    # measured: its checkpoint was overwritten in place. Scoring the current
    # weights and labelling them "50M" because 50M is the lowest unmet
    # threshold would put a number in the table next to a token count it was
    # not measured at. Two such points were produced before this guard existed
    # and had to be relabelled by hand.
    start = ckpt_step(ckpt) if ckpt.exists() else None
    if start is not None:
        passed = [m for m in milestones if m <= start[1]]
        done.update(passed)
        if passed:
            print(f"skipping {[f'{x/1e6:.0f}M' for x in passed]}: already at "
                  f"{start[1]:,} tokens when tracking began", flush=True)

    print(f"tracking {args.run} at {[f'{x/1e6:.0f}M' for x in milestones]}", flush=True)

    idle_passes = 0
    while True:
        if not ckpt.exists():
            time.sleep(args.poll)
            continue
        got = ckpt_step(ckpt)
        if got is None:
            time.sleep(args.poll)
            continue
        step, tokens = got
        crossed = [mm for mm in milestones if mm not in done and tokens >= mm]
        # Label by the LATEST threshold crossed, not the earliest: a tracker
        # that wakes up at 205M has crossed 50M..200M at once, and only the
        # highest of those describes where the weights actually are.
        crossed = crossed[-1:] if crossed else []
        if not crossed:
            if tokens >= args.until:
                print(f"reached {tokens:,} tokens; tracker done", flush=True)
                return 0
            time.sleep(args.poll)
            idle_passes += 1
            continue

        # Score the checkpoint as it stands now, labelled by the milestone it
        # crossed rather than by its exact token count: the label is the
        # independent variable, and the true count is recorded alongside it.
        shutil.copy2(ckpt, staging / "ckpt.pt")
        out = run_dir / f"traj_{crossed[0] // 1_000_000:03d}M.json"
        cmd = [
            sys.executable, str(PROJECT / "scripts" / "eval_full.py"),
            "--run", str(staging), "--rung", args.rung,
            "--stream", args.stream, "--seq", str(args.seq), "--out", str(out),
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"score failed at {crossed[0]/1e6:.0f}M: "
                  f"{proc.stderr.strip().splitlines()[-3:]}", flush=True)
            time.sleep(args.poll)
            continue
        rec = json.loads(out.read_text())
        rec["milestone"] = crossed[0]
        rec["tokens_at_measure"] = tokens
        traj["points"].append(rec)
        done.update(crossed[:1])
        traj_path.write_text(json.dumps(traj, indent=1) + "\n")
        print(f"{crossed[0]/1e6:5.0f}M  step {step:>7,}  "
              f"bpb {rec['bits_per_byte']:.4f}  (measured at {tokens:,} tokens)",
              flush=True)
        (staging / "ckpt.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
