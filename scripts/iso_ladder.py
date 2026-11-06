#!/usr/bin/env python3
"""Run the ladder at fixed tokens-per-parameter, back to back, with no idle gap.

The bug this fixes
------------------
The first ladder held tokens constant (100M) and varied parameters. That is one
vertical slice through a two-dimensional surface (N, D), and this slice crosses a
regime boundary:

    tiny   2.41M params   D/N = 41.5   capacity-limited
    small  14.26M         D/N =  7.0   near the observed minimum
    mid    33.70M         D/N =  3.0   capacity-rich
    large 113.47M         D/N =  0.88  memorising

The measured consequence is not subtle. At D/N = 0.88 the large rung drove
*train* loss to 0.051 nats (0.074 bits/byte) while its held-out score was 2.895
bits/byte — a 39x gap. A 113M-parameter model has enough capacity to store a
98 MB corpus outright, and with the objective it was given, storing is the
cheapest way to reduce loss. The tiny rung cannot memorise, so it is forced to
learn structure that transfers, which is why it scores 1.342.

Read against Chinchilla (D/N ~ 20 for compute-optimal training), the "inverted"
curve is not a failure of scaling. It is the expected fixed-D slice: loss is
U-shaped in N when D is pinned, because N/D crosses from capacity-limited to
data-limited. Comparing rungs at fixed D compares four models at four different
distances from the compute-optimal line, which is not a scaling law.

What this driver does instead
-----------------------------
Holds D/N constant and varies N. Every rung then sits on one iso-token-per-param
line, so a monotone decrease in held-out bits/byte is a scaling law and its
absence is a different (still reportable) result. D/N is a knob: 20 is the
Chinchilla reference, 10 is the default here because it fits the session budget
and sits between the measured minimum (~7) and the literature value.

Why a queue and not four commands
---------------------------------
A GPU that is not computing is money spent for nothing, and the expensive part
of a ladder is the gap between rungs — waiting for a human to notice a rung
finished. This walks the list itself, skips rungs that already have a
`summary.json`, and starts the next one the moment the previous exits. Each rung
still runs as bounded resumable segments through `train_segments.py`, so a
sandbox drop costs at most one segment, and re-running this command resumes
where it stopped.

Usage:
    python3 scripts/iso_ladder.py --tokens-per-param 10
    python3 scripts/iso_ladder.py --rungs tiny small --tokens-per-param 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import rung as rung_cfg  # noqa: E402
from azurite.model import parameter_counts  # noqa: E402

RESULTS = PROJECT / "results" / "pretrain"
LOGS = PROJECT / "logs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=["tiny", "small", "mid", "large"])
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--tokens-per-param", type=float, default=10.0)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--segment-seconds", type=float, default=3000.0)
    ap.add_argument("--max-segments", type=int, default=600)
    ap.add_argument("--stream", default="data/tokens/byte_100000000.u16")
    ap.add_argument("--tag", default="")
    ap.add_argument("--queue", default=str(RESULTS / "ISO_QUEUE.json"))
    args = ap.parse_args()

    tag = args.tag or f"iso{args.tokens_per_param:g}"
    LOGS.mkdir(parents=True, exist_ok=True)
    queue_path = Path(args.queue)

    state = {"tokens_per_param": args.tokens_per_param, "tag": tag,
             "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rungs": {}}
    if queue_path.exists():
        state = json.loads(queue_path.read_text(encoding="utf-8"))
        state.setdefault("rungs", {})

    plan = []
    for name in args.rungs:
        cfg = rung_cfg(name, args.arm)
        params = parameter_counts(cfg)["pretrain_total"]
        tokens = int(params * args.tokens_per_param)
        plan.append((name, params, tokens))

    print(f"iso-D/N ladder  D/N={args.tokens_per_param:g}  tag={tag}  seq={args.seq} "
          f"batch={args.batch}", flush=True)
    total = sum(t for _, _, t in plan)
    for name, params, tokens in plan:
        print(f"  {name:6s} {params:>12,} params  {tokens:>14,} tokens  "
              f"epochs {tokens / 100e6:5.2f}", flush=True)
    print(f"  total {total:,} tokens", flush=True)

    t_all = time.perf_counter()
    for name, params, tokens in plan:
        out_dir = RESULTS / f"l1_{name}_{args.arm}_{tag}"
        entry = state["rungs"].setdefault(name, {})
        entry.update({"params": params, "tokens": tokens, "dir": out_dir.name})

        if (out_dir / "summary.json").exists():
            print(f"\n[{name}] already complete — next", flush=True)
            entry["status"] = "complete"
            queue_path.write_text(json.dumps(state, indent=1) + "\n")
            continue

        entry["status"] = "running"
        entry["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        queue_path.write_text(json.dumps(state, indent=1) + "\n")

        cmd = [
            sys.executable, str(PROJECT / "scripts" / "train_segments.py"),
            "--rung", name, "--arm", args.arm,
            "--tokens", str(tokens), "--seq", str(args.seq),
            "--batch", str(args.batch), "--stream", args.stream,
            "--tag", tag,
            "--segment-seconds", str(args.segment_seconds),
            "--max-segments", str(args.max_segments),
        ]
        print(f"\n=== {name}: {tokens:,} tokens ({tokens / 100e6:.2f} epochs) ===",
              flush=True)
        log_path = LOGS / f"iso_{name}_{tag}.log"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=str(PROJECT), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
                fh.flush()
            code = proc.wait()

        entry["exit"] = code
        entry["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        entry["status"] = (
            "complete" if (out_dir / "summary.json").exists()
            else ("failed" if code != 0 else "incomplete")
        )
        queue_path.write_text(json.dumps(state, indent=1) + "\n")
        if code != 0:
            print(f"[{name}] exited {code}; stopping the queue so the failure is "
                  "visible rather than silently skipped", flush=True)
            return code

    state["elapsed_s"] = round(time.perf_counter() - t_all, 1)
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    queue_path.write_text(json.dumps(state, indent=1) + "\n")
    print(f"\nqueue done in {state['elapsed_s'] / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
