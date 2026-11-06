#!/usr/bin/env python3
"""Run the pre-registered nine-cell suite for the large rung.

This is the wave runner for `results/suite_large_113m/PRE_REGISTRATION.md`.
It launches one process per task, all concurrently, with the same recipe that
produced `results/suite_dual/`: dual path, 1e-4, 3,000 steps, batch 32, 4,000
train rows and 2,000 test rows (split limits apply), seed 0, native 1,024
context.  It does not tune anything per task and does not run the random arm.

Usage:
    python scripts/run_suite_large.py \
        --ckpt results/pretrain/l1_large_byte_iso10v3/ckpt.pt \
        --out-dir results/suite_large_113m --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TASKS = [
    "ag_news",
    "emotion",
    "banking77",
    "clinc_oos",
    "massive_intent",
    "massive_scenario",
    "xnli",
    "language_id",
    "financial_phrasebank",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--examples", type=int, default=4000)
    ap.add_argument("--test-examples", type=int, default=2000)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint: {ckpt}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_sha = sha256(ckpt)
    meta = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_bytes": ckpt.stat().st_size,
        "protocol": {
            "path": "auto (dual)",
            "lr": args.lr,
            "steps": args.steps,
            "batch": args.batch,
            "seed": 0,
            "split_seed": 0,
            "train_examples": args.examples,
            "test_examples": args.test_examples,
            "max_seq": args.max_seq,
            "device": args.device,
            "threads": args.threads,
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cells": {},
    }
    (out_dir / "CHECKPOINT.json").write_text(json.dumps({
        "path": str(ckpt), "bytes": ckpt.stat().st_size, "sha256": ckpt_sha,
    }, indent=2) + "\n")

    procs: dict[str, subprocess.Popen] = {}
    logs: dict[str, object] = {}
    for task in args.tasks:
        cell = out_dir / task
        evidence = cell / "evidence"
        cell.mkdir(parents=True, exist_ok=True)
        result = cell / "result.json"
        cmd = [
            sys.executable, "-u", str(PROJECT / "scripts" / "transfer_eval.py"),
            "--task", task, "--arm", "byte", "--rung", "large",
            "--path", "auto", "--init", "pretrained", "--ckpt", str(ckpt),
            "--seed", "0", "--examples", str(args.examples),
            "--test-examples", str(args.test_examples),
            "--steps", str(args.steps), "--batch", str(args.batch),
            "--lr", str(args.lr), "--stratify", "--split-seed", "0",
            "--device", args.device, "--threads", str(args.threads),
            "--max-seq", str(args.max_seq),
            "--out", str(result), "--evidence-dir", str(evidence),
        ]
        (cell / "launch.json").write_text(json.dumps({
            "cmd": cmd, "cwd": str(PROJECT),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol": meta["protocol"],
            "checkpoint": {"path": str(ckpt), "sha256": ckpt_sha},
        }, indent=1) + "\n")
        log = (cell / "run.log").open("wb")
        logs[task] = log
        log.write((" ".join(cmd) + "\n\n").encode())
        log.flush()
        procs[task] = subprocess.Popen(cmd, cwd=str(PROJECT), stdout=log,
                                       stderr=subprocess.STDOUT)
        print(f"[suite] started {task} pid={procs[task].pid}", flush=True)

    failed = []
    for task, proc in procs.items():
        rc = proc.wait()
        log = logs[task]
        log.close()  # type: ignore[union-attr]
        cell = out_dir / task
        row = {"task": task, "returncode": rc}
        if rc == 0 and (cell / "result.json").exists():
            result = json.loads((cell / "result.json").read_text())
            final = result.get("final", {})
            row.update({
                "status": "ok", "n": final.get("n"),
                "accuracy": final.get("accuracy"),
                "nll": final.get("nll"), "ece": final.get("ece"),
                "acc_at_50pct_coverage": final.get("acc_at_50pct_coverage"),
                "steps": result.get("steps"),
                "seconds": result.get("seconds"),
            })
        else:
            row["status"] = "failed"
            failed.append(task)
        meta["cells"][task] = row
        print(f"[suite] {task} rc={rc} {row.get('status')}", flush=True)

    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["completed"] = len(meta["cells"]) - len(failed)
    meta["failed"] = failed
    (out_dir / "suite_summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[suite] completed {meta['completed']}/{len(meta['cells'])}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
