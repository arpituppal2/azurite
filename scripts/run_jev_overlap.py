#!/usr/bin/env python3
"""Run Azurite on the Jev-overlap tasks at their full official evaluation splits.

This is deliberately not the nine-cell sampled suite: the point is to compare
against the published Jev points on the same split sizes (7,600 AG News rows,
3,076 Banking77 rows, 5,500 CLINC150 rows, and so on). The fine-tuning recipe
stays the pre-registered one: 4,000 stratified train rows, 3,000 steps, batch
32, lr 1e-4, seed 0, native 1,024-byte context.

The runner bounds concurrency because six 113M cells at batch 32 do not fit on
the 97 GB card; three at a time did fit in the earlier suite waves.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# Published Jev 1.13.0 points from the Bonn study (arXiv 2609.37647, Table 2).
# The split size is the paper's n; our local split is stated alongside it.
JEV = {
    "ag_news": {"score": 0.885, "metric": "accuracy", "n": 7600, "local_n": 7600,
                "source": "Bonn Table 2"},
    "banking77": {"score": 0.797, "metric": "accuracy", "n": 3076, "local_n": 3076,
                  "source": "Bonn Table 2"},
    "clinc_oos": {"score": 0.895, "metric": "accuracy", "n": 5500, "local_n": 5500,
                  "source": "Bonn Table 2"},
    "emotion": {"score": 0.585, "metric": "accuracy", "n": 2000, "local_n": 2000,
                "source": "Bonn Table 2"},
    "financial_phrasebank": {"score": 0.730, "metric": "accuracy", "n": 970,
                             "local_n": 1000, "source": "Bonn Table 2",
                             "steps": 1500},
    "language_id": {"score": 0.996, "metric": "accuracy", "n": 10000, "local_n": 10000,
                    "source": "Bonn Table 2", "steps": 4000},
    "imdb": {"score": 0.965, "metric": "accuracy", "n": 25000, "local_n": 25000,
             "source": "Bonn Table 2", "max_seq": 1024, "steps": 2000},
    "rotten_tomatoes": {"score": 0.933, "metric": "accuracy", "n": 1066,
                        "local_n": 1066, "source": "Bonn Table 2"},
    "sst2": {"score": 0.964, "metric": "accuracy", "n": 872, "local_n": 872,
             "source": "Bonn Table 2"},
    "sms_spam": {"score": 0.938, "metric": "F1", "n": 5574, "local_n": 1115,
                 "source": "Bonn Table 2; local 80/20 split, not all rows",
                 "steps": 1000},
    "paws": {"score": 0.850, "metric": "accuracy", "n": 8000, "local_n": 8000,
             "source": "Bonn Table 2", "pack_fields": True, "steps": 2000},
    "anli": {"score": 0.739, "metric": "accuracy", "n": 3200, "local_n": 3200,
             "source": "Bonn Table 2", "pack_fields": True, "max_seq": 1024,
             "steps": 2000},
    "boolq": {"score": 0.913, "metric": "accuracy", "n": 3270, "local_n": 3270,
              "source": "Bonn Table 2", "pack_fields": True, "max_seq": 1024,
              "steps": 1500},
    "toxigen": {"score": 0.878, "metric": "accuracy", "n": 940, "local_n": 940,
                "source": "Bonn Table 2", "steps": 2000},
    "toxic_chat": {"score": 0.786, "metric": "F1", "n": 5083, "local_n": 5083,
                   "source": "Bonn Table 2", "steps": 1500},
    "prompt_injections": {"score": 0.741, "metric": "accuracy", "n": 116,
                          "local_n": 116, "source": "Bonn Table 2",
                          "steps": 600},
    "agb_de": {"score": 0.204, "metric": "F1", "n": 755, "local_n": 755,
               "source": "Bonn Table 2", "steps": 1000},
    "xnli": {"score": 0.860, "metric": "accuracy", "n": 5010,
             "source": "AY (no clean split; unreconciled)",
             "pack_fields": True, "max_seq": 1024, "examples": 20000,
             "steps": 2000},
}

DEFAULT_TASKS = [
    "ag_news", "banking77", "clinc_oos", "emotion",
    "financial_phrasebank", "language_id",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    # The 2026-10-30 sweep ran the 113M `large` encoder. The 403M `xlarge`
    # checkpoint needs its own rung or transfer_eval refuses the shape.
    ap.add_argument("--rung", default="large")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--test-examples", type=int, default=100_000)
    ap.add_argument("--examples", type=int, default=0,
                    help="train rows per task; 0 uses every row in the task's train split")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    # transfer_eval's 6 GiB default was sized for the 113M rung; a 403M encoder
    # plus Adam state is already past it before the first step.
    ap.add_argument("--mem-cap-gib", type=float, default=0.0)
    ap.add_argument("--amp", default="off", choices=["off", "auto", "bf16", "fp16"],
                    help="autocast for the fine-tune loop; recorded in the protocol")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "checkpoint": str(ckpt),
            "rung": args.rung,
            "checkpoint_sha256": __import__("hashlib").sha256(
                ckpt.read_bytes()
            ).hexdigest(),
            "train_rows": args.examples,
            "test_rows": "full official split (cap omitted)",
            "steps": args.steps,
            "batch": args.batch,
            "lr": args.lr,
            "seed": 0,
            "max_seq": args.max_seq,
            "path": "auto (dual)",
        },
        "cells": {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    pending = list(args.tasks)
    running: dict[str, subprocess.Popen] = {}
    logs: dict[str, object] = {}

    while pending or running:
        while pending and len(running) < args.max_parallel:
            task = pending.pop(0)
            cell = out / task
            cell.mkdir(parents=True, exist_ok=True)
            result = cell / "result.json"
            if result.exists():
                print(f"[jev] {task} already complete", flush=True)
                summary["cells"][task] = {"status": "existing", "result": str(result)}
                continue
            task_cfg = JEV.get(task, {})
            examples = args.examples or 10**9
            examples = int(task_cfg.get("examples", examples))
            max_seq = int(task_cfg.get("max_seq", args.max_seq))
            steps = int(task_cfg.get("steps", args.steps))
            cmd = [
                sys.executable, "-u", str(PROJECT / "scripts" / "transfer_eval.py"),
                "--task", task, "--arm", "byte", "--rung", args.rung,
                "--path", "auto", "--init", "pretrained", "--ckpt", str(ckpt),
                "--seed", "0", "--examples", str(examples),
                "--test-examples", str(args.test_examples),
                "--steps", str(steps), "--batch", str(args.batch),
                "--lr", str(args.lr), "--stratify", "--split-seed", "0",
                "--device", args.device, "--threads", "2",
                "--max-seq", str(max_seq),
                "--mem-cap-gib", str(args.mem_cap_gib),
                "--out", str(result), "--evidence-dir", str(cell / "evidence"),
                "--amp", args.amp,
            ]
            if task_cfg.get("pack_fields"):
                cmd.append("--pack-fields")
            if max_seq > 1024:
                cmd.append("--allow-context-extension")
            (cell / "launch.json").write_text(json.dumps({
                "cmd": cmd, "started_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "jev": JEV.get(task),
            }, indent=1) + "\n")
            log = (cell / "run.log").open("wb")
            logs[task] = log
            log.write((" ".join(cmd) + "\n\n").encode())
            log.flush()
            proc = subprocess.Popen(cmd, cwd=str(PROJECT), stdout=log,
                                    stderr=subprocess.STDOUT)
            running[task] = proc
            print(f"[jev] started {task} pid={proc.pid}", flush=True)

        time.sleep(5)
        for task, proc in list(running.items()):
            if proc.poll() is None:
                continue
            rc = proc.returncode
            logs[task].close()  # type: ignore[union-attr]
            cell = out / task
            result = cell / "result.json"
            row = {"status": "ok" if rc == 0 and result.exists() else "failed",
                   "returncode": rc, "jev": JEV.get(task)}
            if row["status"] == "ok":
                payload = json.loads(result.read_text())
                final = payload.get("final", {})
                row.update({
                    "n": final.get("n"),
                    "accuracy": final.get("accuracy"),
                    "nll": final.get("nll"),
                    "ece": final.get("ece"),
                    "accuracy_at_50pct_coverage": final.get(
                        "acc_at_50pct_coverage"),
                    "seconds": payload.get("seconds"),
                })
                if (row.get("accuracy") is not None and JEV.get(task)
                        and JEV[task]["metric"] == "accuracy"):
                    row["jev_gap_pp"] = round(
                        100 * (row["accuracy"] - JEV[task]["score"]), 3)
            summary["cells"][task] = row
            (out / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n")
            print(f"[jev] {task} {row['status']} {row.get('accuracy')}",
                  flush=True)
            del running[task]

    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
