#!/usr/bin/env python3
"""Run the pretraining ladder rung by rung, resumably, unattended.

Each rung is a separate `scripts/pretrain.py` process. This script exists so a
multi-hour ladder survives interruption: it skips rungs that already have a
`summary.json`, writes `LADDER_STATE.json` after every transition, and can be
re-run after a crash because `pretrain.py --resume` picks up from its last
checkpoint.

Two scheduling choices, made explicit because both are defensible and they are
not the same experiment:

- `--tokens` fixed across rungs (default). Compute scales with parameters only,
  and *tokens per parameter* falls as the rung grows. This measures what a fixed
  data budget buys at each size — the cleanest axis for "does the transferred
  representation improve with model size".
- `--tokens-per-param P`. Tokens grow with parameters (P=20 is Chinchilla-ish),
  so compute grows as parameters². This is the compute-optimal ladder, and it is
  far more expensive locally.

The probe mode measures tokens/s per rung for a few steps and writes the
estimate, so a ladder schedule is priced from measurement on this machine
rather than from an assumption. Local MPS throughput is not transferable from
published GPU-hour counts.

Usage:
    .venv/bin/python3 scripts/run_ladder.py --probe --rungs small mid large
    .venv/bin/python3 scripts/run_ladder.py --rungs small --tokens 100e6
"""

from __future__ import annotations

import argparse
import json
import re
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
STATE = RESULTS / "LADDER_STATE.json"

TRAIN_LINE = re.compile(r"step\s+[\d,]+/[\d,]+\s+loss\s+([\d.]+)\s+([\d,]+) tok/s")


def read_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"rungs": {}}
    return {"rungs": {}}


def write_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    STATE.write_text(json.dumps(state, indent=1) + "\n")


def run(cmd: list[str], log_path: Path) -> int:
    """Run a child, streaming its output to both a log and this process's stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
        return proc.wait()


def probe(rung: str, arm: str, steps: int, batch: int, seq: int, mem_cap: float) -> dict:
    """Measure tokens/s on a short run instead of assuming it."""
    log = LOGS / f"probe_{rung}_{arm}.log"
    cmd = [
        sys.executable, str(PROJECT / "scripts" / "pretrain.py"),
        "--rung", rung, "--arm", arm, "--tokens", str(steps * batch * seq),
        "--seq", str(seq), "--batch", str(batch), "--log-every", "5",
        "--mem-cap-gib", str(mem_cap),
        # One level below `results/pretrain/`, deliberately: `l1_report.py` globs
        # `*/summary.json`, and a 25-step probe also writes a summary.json. Left at
        # the top level, a probe would be graded as a rung and reported as a
        # catastrophic failure of a model that had barely started training.
        "--out", str(RESULTS / "_probes" / f"{rung}_{arm}"),
    ]
    code = run(cmd, log)
    if code != 0:
        return {"rung": rung, "arm": arm, "status": "failed", "exit": code}
    text = log.read_text(encoding="utf-8")
    rates = [float(m.group(2).replace(",", "")) for m in TRAIN_LINE.finditer(text)]
    tail = rates[-5:] if rates else []
    return {
        "rung": rung,
        "arm": arm,
        "status": "ok" if tail else "no-measurement",
        "tokens_per_s": round(sum(tail) / len(tail), 1) if tail else None,
        "steps": steps,
        "batch": batch,
        "seq": seq,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=["small"])
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--tokens", type=float, default=100e6)
    ap.add_argument("--tokens-per-param", type=float, default=None)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mem-cap-gib", type=float, default=6.0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-steps", type=int, default=20)
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    state = read_state()
    state.setdefault("rungs", {})

    if args.probe:
        probes = []
        for rung in args.rungs:
            print(f"\n=== probe {rung} ===", flush=True)
            p = probe(rung, args.arm, args.probe_steps, args.batch, args.seq,
                      args.mem_cap_gib)
            probes.append(p)
            print(json.dumps(p), flush=True)
            state["rungs"].setdefault(rung, {})["probe"] = p
            write_state(state)
        out = RESULTS / f"ladder_probe_{args.arm}.json"
        out.write_text(json.dumps(probes, indent=1) + "\n")
        print(f"\nwrote {out.resolve().relative_to(PROJECT)}")
        for p in probes:
            if p.get("tokens_per_s"):
                hours = args.tokens / p["tokens_per_s"] / 3600
                print(f"  {p['rung']:6s} {p['tokens_per_s']:>9,.0f} tok/s → "
                      f"{hours:5.1f} h for {args.tokens / 1e6:.0f}M tokens")
        return 0

    for name in args.rungs:
        cfg = rung_cfg(name, args.arm)
        tokens = (
            parameter_counts(cfg)["pretrain_total"] * args.tokens_per_param
            if args.tokens_per_param
            else args.tokens
        )
        out_dir = RESULTS / f"l1_{name}_{args.arm}"
        entry = state["rungs"].setdefault(name, {})
        entry.update({"tokens": int(tokens), "arm": args.arm})

        if (out_dir / "summary.json").exists():
            print(f"skip {name}: already complete", flush=True)
            entry["status"] = "complete"
            write_state(state)
            continue

        entry["status"] = "running"
        entry["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_state(state)
        cmd = [
            sys.executable, str(PROJECT / "scripts" / "pretrain.py"),
            "--rung", name, "--arm", args.arm, "--tokens", str(tokens),
            "--seq", str(args.seq), "--batch", str(args.batch),
            "--mem-cap-gib", str(args.mem_cap_gib),
            "--out", str(out_dir), "--resume",
        ]
        print(f"\n=== {name}: {tokens:,.0f} tokens ===", flush=True)
        code = run(cmd, LOGS / f"l1_{name}_{args.arm}.log")
        entry["exit"] = code
        entry["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if code != 0:
            entry["status"] = "failed"
            write_state(state)
            print(f"{name} failed with exit {code}; stopping the ladder", flush=True)
            return code
        entry["status"] = "complete" if (out_dir / "summary.json").exists() else "incomplete"
        write_state(state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
