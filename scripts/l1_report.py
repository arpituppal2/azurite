#!/usr/bin/env python3
"""L1 gate report: did pretraining actually beat the n-gram baseline?

The gate in PLAN.md §6 is not "loss went down" — loss always goes down. It is:

    held-out bits/byte must beat the bigram model fit on the same training
    region and scored on the same held-out split.

A byte-level transformer that cannot beat a bigram has learned nothing beyond
adjacent-byte statistics, and scaling it up would be spending compute to confirm
that. The baseline is read from a JSON rather than hard-coded, because it depends
on the token stream: the same architecture measured 4.117 bits/byte on a
math-only stream and 3.767 on the mixed stream, so a baseline carried across a
data change would make the gate meaningless.

The report also refuses to grade an incomplete run. A `summary.json` is written
only after the held-out evaluation finishes, so its absence is treated as "no
measurement", never as a pass.

Usage:
    .venv/bin/python3 scripts/l1_report.py
    .venv/bin/python3 scripts/l1_report.py --out results/pretrain/L1.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PRETRAIN = PROJECT / "results" / "pretrain"


def read_curve(run_dir: Path) -> list[dict]:
    path = run_dir / "train.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def find_baseline(run: dict) -> dict | None:
    """The baseline recorded for this exact token stream, if one exists."""
    stream = Path(run.get("token_stream", "")).name
    if not stream:
        return None
    candidates = sorted(PRETRAIN.glob("bigram_*.json"))
    matching = []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if Path(data.get("token_stream", "")).name == stream:
            matching.append((path, data))
    if not matching:
        return None
    # newest wins: a stream may have been measured more than once
    path, data = sorted(matching, key=lambda pd: pd[0].stat().st_mtime)[-1]
    data["_path"] = str(path.relative_to(PROJECT))
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=str(PRETRAIN / "L1.md"))
    args = ap.parse_args()

    def graded(path: Path) -> bool:
        """Exclude anything that is evidence of something other than a rung.

        A probe run writes a summary.json too, and grading a 25-step probe would
        report a rung as failing when the model had barely started training.
        """
        if path.name.startswith(("PROBE", "DISCARDED", "_")):
            print(f"skip {path.name}: not a graded run", flush=True)
            return False
        return True

    if args.runs:
        run_dirs = [
            Path(r) if Path(r).is_absolute() else PROJECT / r for r in args.runs
        ]
    else:
        run_dirs = sorted(
            p.parent for p in PRETRAIN.glob("*/summary.json") if graded(p.parent)
        )
    if not run_dirs:
        raise SystemExit(f"no completed runs (summary.json) under {PRETRAIN}")

    rows = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"skip {run_dir.name}: no summary.json (run incomplete)", flush=True)
            continue
        run = json.loads(summary_path.read_text(encoding="utf-8"))
        baseline = find_baseline(run)
        curve = read_curve(run_dir)
        heldout = run["heldout"]["bits_per_byte"]
        bigram = baseline["bigram_bits_per_byte"] if baseline else None
        unigram = baseline["unigram_bits_per_byte"] if baseline else None
        rows.append(
            {
                "run": run_dir.name,
                "rung": run["rung"],
                "arm": run["arm"],
                "params": run["params"],
                "tokens": run["tokens_seen"],
                "steps": run["steps"],
                "heldout": heldout,
                "bigram": bigram,
                "unigram": unigram,
                "margin": (bigram - heldout) if bigram is not None else None,
                "ratio": (bigram / heldout) if bigram is not None and heldout else None,
                "passed": bigram is not None and heldout < bigram,
                "seconds": run.get("seconds"),
                "tokens_per_s": round(run["tokens_seen"] / run["seconds"], 1)
                if run.get("seconds")
                else None,
                "first_loss": curve[0]["loss"] if curve else None,
                "last_loss": curve[-1]["loss"] if curve else None,
                "baseline_path": baseline["_path"] if baseline else None,
                "swap_mb": run.get("memory", {}).get("swap_delta_mb"),
            }
        )

    if not rows:
        raise SystemExit("no run had both a summary and a matching baseline")

    rows.sort(key=lambda r: (r["params"], r["run"]))
    any_failed = any(not r["passed"] for r in rows)

    lines = [
        "# L1 gate — does pretraining beat the n-gram baseline?",
        "",
        "Generated by `scripts/l1_report.py` from each run's `summary.json` and the",
        "bigram baseline fitted to *the same token stream*. The baseline is per-stream",
        "on purpose: the identical architecture scored 4.117 bits/byte on a math-only",
        "stream and 3.767 on the mixed one, so a baseline carried across a data change",
        "would silently change the threshold.",
        "",
        "| run | rung | arm | params | tokens | tokens/param | held-out bpb | bigram | unigram | margin | gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    def num(value, spec: str = ".4f") -> str:
        return format(value, spec) if value is not None else "—"

    for r in rows:
        lines.append(
            f"| `{r['run']}` | {r['rung']} | {r['arm']} | {r['params']:,} | "
            f"{r['tokens']:,} | {r['tokens'] / r['params']:.1f} | "
            f"**{r['heldout']:.4f}** | {num(r['bigram'])} | {num(r['unigram'])} | "
            f"{num(r['margin'], '+.4f')} | {'PASS' if r['passed'] else '**FAIL**'} |"
        )

    lines += ["", "## Detail", ""]
    for r in rows:
        lines.append(f"### `{r['run']}`")
        lines.append("")
        lines.append(f"- {r['params']:,} parameters, {r['steps']:,} steps, "
                     f"{r['tokens']:,} training tokens")
        if r["tokens_per_s"]:
            lines.append(f"- {r['tokens_per_s']:,} tokens/s; {r['seconds']:,} s wall clock")
        if r["first_loss"] is not None:
            lines.append(
                f"- train loss {r['first_loss']:.4f} → {r['last_loss']:.4f} "
                f"(the loss falling is not the gate; it is what any model does)"
            )
        if r["bigram"] is not None:
            lines.append(
                f"- held-out {r['heldout']:.4f} vs bigram {r['bigram']:.4f} and unigram "
                f"{r['unigram']:.4f} bits/byte → saves {r['margin']:.4f} bits/byte; "
                f"the model's cost per byte is 1/{r['ratio']:.2f} = "
                f"{1 / r['ratio']:.2f}× the bigram's"
            )
            lines.append(f"- baseline: `{r['baseline_path']}`")
        else:
            lines.append("- **no baseline for this token stream** — the gate cannot be graded")
        if r["swap_mb"] is not None:
            lines.append(f"- swap delta {r['swap_mb']:+.0f} MB during training")
        lines.append("")

    lines += [
        "## Protocol note, because the two numbers are not the same quantity",
        "",
        "The model's bits/byte is the mean cross-entropy over **masked positions at a",
        "30% masking rate with bidirectional context** (MosaicBERT's encoder recipe,",
        "80/10/10 corruption). The bigram's is the mean cross-entropy over **every",
        "position with one symbol of left-to-right context**.",
        "",
        "So the model sees 70% of the bytes around the position it is predicting, and",
        "the bigram sees one byte to its left. The model is expected to win, and a win",
        "does not by itself prove the model learned structure a better n-gram would",
        "miss. The gate is therefore a **falsification floor**: failing it would prove",
        "there is nothing here worth scaling, passing it rules out 'a bigram is just as",
        "good' but does not establish that the representation is good.",
        "",
        "For that reason these bits/byte figures are **not** comparable to a published",
        "autoregressive perplexity per byte, and are never presented as one. A matched",
        "comparison against an interpolated order-3 n-gram (the strongest n-gram whose",
        "table fits in memory: 260³ = 17.6M entries) is the next hardening step; raw",
        "higher orders are not used here because without interpolation their sparse",
        "contexts make them a weaker baseline than the bigram, which would flatter the",
        "model for the wrong reason.",
        "",
        "## What passing means",
        "",
        "*Reading the gate correctly depends on the protocol note above.*",
        "",
        "",
        "The model has learned structure beyond byte-adjacency: it must predict masked",
        "positions from context, so beating a bigram requires using information that a",
        "one-token window does not contain. What passing does **not** mean: that the",
        "representation is useful for decisions. That is a separate measurement",
        "(`scripts/transfer_eval.py`), and the two can come apart — a model can be a",
        "better language model and a worse encoder for a 4-way decision.",
        "",
        f"Gate: {'all runs pass' if not any_failed else '**at least one run fails**'}.",
        "",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: 8 + len(rows)]))
    print(f"\nwrote {out.resolve().relative_to(PROJECT)}")
    return 0 if not any_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
