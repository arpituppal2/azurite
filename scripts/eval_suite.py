#!/usr/bin/env python3
"""Run Azurite across the public comparator task suite.

The suite is the overlap between what Laya publishes and what Jev's public
third-party runs publish, restricted to datasets present in this checkout:

  task                    K     source row
  ag_news                 4     Laya 0.950/0.953/0.947; Jev 0.910
  emotion (DAIR)          6     Laya 0.595/0.600/0.573; Jev 0.480
  banking77              77     Laya 0.425; Jev 0.788 (AY, 77-way)
  clinc_oos             151     Jev 0.895
  massive_intent         60     Laya 0.783
  massive_scenario       18     Laya/MASSIVE secondary
  xnli                    3     Laya 0.860
  language_id            20     Jev 0.996
  financial_phrasebank    3     Jev 0.730

This is **not** a zero-shot comparison. Azurite has no instruction-tuned
zero-shot head; every run fine-tunes the decision head on the task's train
split, then evaluates the full test split. The random-initialisation arm uses
the identical architecture, rows, order, steps and seed, so the difference is
the pretraining contribution, not the fine-tuning budget.

The 37-dataset University of Bonn Jev study and Laya's multilingual rows are
not silently folded in: their datasets are not in this checkout. They are named
as missing in suite_summary.json.

Usage:
    .venv/bin/python scripts/eval_suite.py --threads 2
    .venv/bin/python scripts/eval_suite.py --tasks ag_news banking77 --arms pretrained
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "results" / "suite"
CKPT = PROJECT / "results" / "pretrain" / "l1_tiny_byte" / "ckpt.pt"

# task, train rows, epochs, why it is in the suite
TASKS = [
    ("ag_news", 4000, 4, "Laya 0.950/0.953/0.947; Jev 0.910"),
    ("emotion", 4000, 4, "Laya 0.595/0.600/0.573; Jev 0.480"),
    ("banking77", 4000, 4, "Laya 0.425; Jev 0.788 on AY 77-way"),
    ("clinc_oos", 4000, 4, "Jev 0.895"),
    ("massive_intent", 4000, 4, "Laya 0.783"),
    ("massive_scenario", 4000, 4, "MASSIVE secondary axis"),
    ("xnli", 4000, 4, "Laya 0.860"),
    ("language_id", 4000, 4, "Jev 0.996"),
    ("financial_phrasebank", 1200, 4, "Jev 0.730"),
]
RANDOM_CONTROLS = {"ag_news", "emotion", "banking77", "massive_intent", "xnli"}


def run_one(task: str, budget: int, epochs: float, arm: str, threads: int,
            test_examples: int, *, rung: str, ckpt: Path, device: str,
            out_dir: Path, tag_prefix: str, max_seq: int = 0,
            steps: int = 0, lr: float = 5e-4,
            train_examples: int = 0) -> dict:
    tag = f"{tag_prefix}{task}_{arm}_s0"
    out = out_dir / f"{tag}.json"
    evidence = out_dir / tag
    if out.exists():
        return {"tag": tag, "status": "skipped-existing", "path": str(out)}
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = train_examples or budget
    cmd = [
        sys.executable, "-u", str(PROJECT / "scripts" / "transfer_eval.py"),
        "--task", task, "--arm", "byte", "--rung", rung, "--path", "auto",
        "--init", arm, "--seed", "0",
        "--examples", str(n_rows), "--test-examples", str(test_examples),
        "--epochs", str(1 if steps else epochs), "--batch", "32", "--lr", str(lr),
        "--stratify", "--split-seed", "0", "--device", device,
        "--threads", str(threads), "--out", str(out),
        "--evidence-dir", str(evidence),
    ]
    if max_seq:
        cmd += ["--max-seq", str(max_seq)]
    if steps:
        cmd += ["--steps", str(steps)]
    if arm == "pretrained":
        cmd += ["--ckpt", str(ckpt)]
    started = time.time()
    log_path = out_dir / f"{tag}.log"
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    if proc.returncode != 0:
        return {"tag": tag, "status": "failed", "returncode": proc.returncode,
                "seconds": round(elapsed, 1), "log": str(log_path)}
    result = json.loads(out.read_text())
    final = result["final"]
    return {
        "tag": tag, "status": "ok", "task": task, "init": arm,
        "n": final.get("n"), "accuracy": final.get("accuracy"),
        "nll": final.get("nll"), "ece": final.get("ece"),
        "acc_at_50pct_coverage": final.get("acc_at_50pct_coverage"),
        "train_examples": result.get("train_examples"),
        "test_examples": result.get("test_examples"),
        "steps": result.get("steps"),
        "seconds": round(elapsed, 1),
        "result": str(out), "log": str(log_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=[t[0] for t in TASKS])
    ap.add_argument("--arms", nargs="+", choices=["pretrained", "random"],
                    default=["pretrained", "random"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--test-examples", type=int, default=2000,
                    help="rows per test split; 2000 keeps a full-suite CPU pass tractable")
    ap.add_argument("--rung", default="tiny", choices=["tiny", "small", "mid", "large"])
    ap.add_argument("--ckpt", default=str(CKPT),
                    help="pretrained checkpoint; ignored for the random arm")
    ap.add_argument("--device", default="cpu",
                    help="torch device, e.g. cpu, mps, cuda, auto")
    ap.add_argument("--out-dir", default=str(OUT),
                    help="directory for run JSON, logs and evidence")
    ap.add_argument("--tag-prefix", default="",
                    help="prefix for run tags, to keep a new rung from colliding")
    ap.add_argument("--max-seq", type=int, default=0,
                    help="checkpoint context length when it differs from the rung default")
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="fixed fine-tuning learning rate for every task")
    ap.add_argument("--steps", type=int, default=0,
                    help="exact optimizer steps for every task; overrides epochs")
    ap.add_argument("--train-examples", type=int, default=0,
                    help="fixed training rows for every task; overrides per-task budget")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint: {ckpt}")
    out_dir = Path(args.out_dir)
    wanted = [t for t in TASKS if t[0] in set(args.tasks)]
    summary = {
        "protocol": {
            "fine_tune": "task train split, stratified, fixed budget, seed 0",
            "eval": "full task test split, no threshold selection",
            "architecture": f"{args.rung} byte encoder",
            "checkpoint": str(ckpt),
            "device": args.device,
            "zero_shot": False,
            "threads": args.threads,
            "test_examples_cap": args.test_examples,
            "learning_rate": args.lr,
            "steps": args.steps,
            "train_examples_override": args.train_examples,
        },
        "runs": [],
        "missing": {
            "jev_37_dataset_study": "only 9 overlapping datasets are local; IMDB, SST-2, Rotten Tomatoes, SMS Spam, GoEmotions, ANLI, PAWS, Belebele, etc. are absent",
            "laya_multilingual": "XNLI de/zh/es/fr and MASSIVE de test splits are local, but the multilingual checkpoint and 13-language matrix are not run",
            "typed_decisions": "Laya's typed-decisions evaluation cases are not in this checkout",
            "jev_4_workflows": "TypeSafe workflow cases are not public",
        },
    }
    for task, budget, epochs, why in wanted:
        for arm in args.arms:
            if arm == "random" and task not in RANDOM_CONTROLS:
                continue
            print(f"[suite] {task} {arm}", flush=True)
            row = run_one(task, budget, epochs, arm, args.threads, args.test_examples,
                          rung=args.rung, ckpt=ckpt, device=args.device,
                          out_dir=out_dir, tag_prefix=args.tag_prefix,
                          max_seq=args.max_seq, steps=args.steps, lr=args.lr,
                          train_examples=args.train_examples)
            row["why"] = why
            summary["runs"].append(row)
            (out_dir / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    ok = [r for r in summary["runs"] if r["status"] == "ok"]
    summary["completed"] = len(ok)
    summary["failed"] = [r for r in summary["runs"] if r["status"] == "failed"]
    (out_dir / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[suite] completed {len(ok)} runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
