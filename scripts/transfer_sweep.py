#!/usr/bin/env python3
"""Run the pretrained-vs-random transfer matrix and aggregate it.

Every cell is the same fine-tuning budget with a different initialisation, so the
aggregate answers one question per task:

    at a fixed budget, what does from-scratch pretraining on this corpus buy?

Aggregation is paired. Both arms of a task are evaluated on the identical packed
test set, so row i is the same example in both, and the comparison is a McNemar
test on the discordant pairs rather than two overlapping confidence intervals.
With 1,000 test rows and accuracies in the 0.6–0.9 range, the unpaired standard
error is 1.3–1.5 points — larger than several of the effects worth reporting — so
the paired test is not decoration.

The exact binomial p-value is computed here rather than pulling in scipy: the
discordant count is small and `math.comb` is enough.

Usage:
    .venv/bin/python3 scripts/transfer_sweep.py \
        --ckpt results/pretrain/l1_tiny_byte/ckpt.pt --rungs tiny \
        --tasks ag_news emotion --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results" / "transfer"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b and c.

    Under H0 the b discordant pairs where only A is right and the c where only B
    is right are exchangeable, so b ~ Binomial(b + c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def run_cell(
    task: str, init: str, seed: int, ckpt: str, rung: str, arm: str,
    examples: int, epochs: float, batch: int, lr: float, test_examples: int,
) -> Path:
    out = RESULTS / f"{task}_{rung}_{arm}_{init}_s{seed}.json"
    cmd = [
        sys.executable, str(PROJECT / "scripts" / "transfer_eval.py"),
        "--task", task, "--init", init, "--seed", str(seed), "--rung", rung,
        "--arm", arm, "--examples", str(examples), "--epochs", str(epochs),
        "--batch", str(batch), "--lr", str(lr),
        "--test-examples", str(test_examples), "--out", str(out),
    ]
    if init == "pretrained":
        cmd += ["--ckpt", ckpt]
    print(f"\n=== {task} {init} seed {seed} ===", flush=True)
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], flush=True)
        print(proc.stderr[-2000:], flush=True)
        raise SystemExit(f"transfer_eval failed for {task}/{init}/s{seed}")
    print(proc.stdout.strip().splitlines()[-1], flush=True)
    print(f"  {time.perf_counter() - started:.0f}s", flush=True)
    return out


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover(arm: str) -> dict[tuple[str, str, str, str], list[dict]]:
    """Every finished cell on disk, grouped by (rung, task, arm, init).

    Grouping from the files rather than from this process's own matrix is what
    lets the report be regenerated after every cell: a sweep stopped halfway
    still produces a truthful document over the cells that finished.
    """
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for path in sorted(RESULTS.glob("*.json")):
        try:
            run = load(path)
        except json.JSONDecodeError:
            continue
        if "final" not in run or "predictions" not in run.get("final", {}):
            continue  # a cell older than the prediction-carrying version
        key = (run["rung"], run["task"], run["arm"], run["init"])
        groups.setdefault(key, []).append(run)
    return groups


def build_report(groups: dict[tuple[str, str, str, str], list[dict]]) -> str:
    """Markdown for whatever cells exist. Cells with one seed still report.

    Every headline number is read back out of the run files, never taken from the
    arguments of whichever invocation happens to be writing the report: the first
    version printed the caller's CLI defaults, so a report regenerated with a
    different command line described a budget no run had used.
    """
    keys = sorted(groups)
    rungs = sorted({k[0] for k in keys})
    tasks = sorted({k[1] for k in keys})
    arms = sorted({k[2] for k in keys})
    runs_all = [r for rs in groups.values() for r in rs]

    ckpts = sorted(
        {
            (r.get("loaded_from") or {}).get("path", "")
            for r in runs_all
            if r.get("loaded_from")
        }
    ) or ["(no cell loaded a checkpoint)"]
    budgets = sorted({(r["train_examples"], r["epochs"], r["batch"]) for r in runs_all})
    lrs = sorted({r["lr"] for r in runs_all})
    seeds_seen = sorted({r["seed"] for r in runs_all})

    lines = [
        "# Transfer: does pretraining buy decision accuracy at a fixed budget?",
        "",
        "Generated by `scripts/transfer_sweep.py`. Both initialisations, matched on",
        "rows, row order, steps, batch, optimiser, learning rate and seed.",
        "",
        f"- pretraining checkpoint: {', '.join(f'`{c}`' for c in ckpts)}",
        "- fine-tuning budgets actually used (train examples × epochs @ batch): "
        + ", ".join(f"{e:,} × {ep:g} @ {b}" for e, ep, b in budgets),
        f"- seeds present on disk: {seeds_seen}",
    ]
    priors: dict[str, dict] = {}
    priors_path = RESULTS / "PRIORS.json"
    if priors_path.exists():
        try:
            priors = json.loads(priors_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            priors = {}

    lines += ["", "| rung | arm | task | init | seeds | acc | majority prior | acc@50%cov | ECE | NLL |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]

    for rung in rungs:
        for arm in arms:
            for task in tasks:
                for init in ("random", "pretrained"):
                    runs = groups.get((rung, task, arm, init))
                    if not runs:
                        continue
                    def mean(key: str) -> float:
                        return sum(r["final"][key] for r in runs) / len(runs)

                    def spread(key: str) -> float:
                        vals = [r["final"][key] for r in runs]
                        return max(vals) - min(vals)

                    prior = priors.get(task, {}).get("majority_accuracy")
                    prior_cell = f"{prior:.4f}" if prior is not None else "—"
                    lines.append(
                        f"| {rung} | {arm} | {task} | {init} | {len(runs)} | "
                        f"{mean('accuracy'):.4f} ±{spread('accuracy'):.4f} | {prior_cell} | "
                        f"{mean('acc_at_50pct_coverage'):.4f} | "
                        f"{mean('ece'):.4f} | {mean('nll'):.4f} |"
                    )

    lines += ["", "## Paired comparison (identical test rows in both arms)", "",
              "| rung | task | Δacc (pretrained − random) | random-only right | pretrained-only right | p (exact McNemar) |",
              "|---|---|---:|---:|---:|---|"]
    for rung in rungs:
        for arm in arms:
            for task in tasks:
                rnd_runs = groups.get((rung, task, arm, "random"))
                pre_runs = groups.get((rung, task, arm, "pretrained"))
                if not rnd_runs or not pre_runs:
                    continue
                seeds_n = min(len(rnd_runs), len(pre_runs))
                rnd_runs, pre_runs = rnd_runs[:seeds_n], pre_runs[:seeds_n]
                deltas, ps, bs, cs = [], [], [], []
                for rnd, pre in zip(rnd_runs, pre_runs):
                    gr, gp = rnd["final"]["gold"], pre["final"]["gold"]
                    if gr != gp:
                        lines.append(
                            f"| {rung} | {task} | **not comparable** | — | — | "
                            "the two arms were evaluated on different test rows |"
                        )
                        break
                    pr, pp = rnd["final"]["predictions"], pre["final"]["predictions"]
                    b = sum(1 for x, y, g in zip(pr, pp, gr) if x == g and y != g)
                    c = sum(1 for x, y, g in zip(pr, pp, gr) if x != g and y == g)
                    bs.append(b)
                    cs.append(c)
                    ps.append(mcnemar_exact(b, c))
                    deltas.append(pre["final"]["accuracy"] - rnd["final"]["accuracy"])
                else:
                    if not deltas:
                        continue
                    mean_delta = sum(deltas) / len(deltas)
                    lines.append(
                        f"| {rung} | {task} | {mean_delta:+.4f} "
                        f"(per seed: {', '.join(f'{d:+.4f}' for d in deltas)}) | "
                        f"{sum(bs)} | {sum(cs)} | "
                        f"{', '.join(f'{p:.4f}' for p in ps)} |"
                    )

    lines += [
        "",
        "## How to read this",
        "",
        "A positive Δacc with *pretrained-only right* far exceeding *random-only right*",
        "means the encoder's pretrained features helped the decision. A Δacc near zero",
        "with large discordant counts in both directions means the pretext task taught",
        "the encoder things this decision does not use — a finding, not a failure, and",
        "the outcome that argues for changing the pretext objective rather than scaling",
        "the model.",
        "",
        "Limits, stated rather than buried:",
        "",
        "- the learning rate is fixed across arms rather than tuned per arm; rates",
        f"  present on disk: {lrs};",
        "- the spread shown is a max−min range over the seeds present, not a variance",
        "  estimate;",
        "- the majority prior is from `results/transfer/PRIORS.json`, measured on the",
        "  same frozen test split the model is scored on;",
        "- the per-seed McNemar p-value assumes test rows are exchangeable, which they",
        "  are not if the test set contains duplicates — the mean Δacc and its sign are",
        "  more trustworthy than any single p-value;",
        "- a matched budget is not each arm's best budget: with a small fine-tuning set",
        "  the random arm is the more likely of the two to be undertrained, which would",
        "  flatter the pretrained arm.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--rungs", nargs="+", default=["tiny"])
    ap.add_argument("--tasks", nargs="+", default=["ag_news", "emotion"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--examples", type=int, default=4000)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--test-examples", type=int, default=1000)
    ap.add_argument("--out", default=str(RESULTS / "TRANSFER.md"))
    ap.add_argument("--report-only", action="store_true",
                    help="aggregate the cells already on disk, run nothing")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)

    def refresh() -> None:
        out.write_text(build_report(discover(args.arm)) + "\n")

    if args.report_only:
        refresh()
        print(f"wrote {out.resolve().relative_to(PROJECT)}")
        return 0

    if not args.ckpt:
        raise SystemExit("--ckpt is required unless --report-only is given")

    for rung in args.rungs:
        for task in args.tasks:
            for init in ("random", "pretrained"):
                for seed in args.seeds:
                    run_cell(
                        task, init, seed, args.ckpt, rung, args.arm,
                        args.examples, args.epochs, args.batch, args.lr,
                        args.test_examples,
                    )
                    refresh()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
