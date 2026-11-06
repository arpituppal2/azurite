#!/usr/bin/env python3
"""Generate the thesis figures from the stored evaluation records.

Four figures are emitted, each built from the same records the tables use:

    fig_pooling.pdf          message-only / native / packed-5,120, both heads
    fig_budget.pdf           query-head accuracy against mean context length
    fig_train_test.pdf       the 5,120 arm's train/test separation
    fig_error_concentration.pdf  per-class Banking77 error concentration

The script verifies every per-example file against its recorded SHA-256,
recomputes the stored counts and Wilson intervals, checks every gold index
against `results/jev_matched/inputs.json`, and reads the comparator prompt budget
from `paper/generated/prompt_budget.json` after re-hashing its source files.
No plotted count is typed by hand.

    python thesis/gen_figures.py            # verify, write figures + macros
    python thesis/gen_figures.py --check    # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.artifacts import read_jsonl, sha256_file  # noqa: E402
from azurite.stats import wilson  # noqa: E402

Q = ROOT / "results" / "jev_query" / "eval"
L = ROOT / "results" / "jev_query_5120" / "eval"
M = ROOT / "results" / "jev_matched"
INPUTS = M / "inputs.json"
PROMPT_BUDGET = ROOT / "paper" / "generated" / "prompt_budget.json"
PROMPT_DIR = ROOT / "paper" / "data" / "jev_prompt"
OUT = ROOT / "thesis" / "figures"
MACROS = ROOT / "thesis" / "tables" / "macros_figures.tex"
N_ROWS = 3_080
N_LABELS = 77

# Print-safe, consistent across all four figures.
MEAN = "#B86B1E"
QUERY = "#245A86"
JEV = "#8C1D28"
GREY = "#6B6B6B"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def num(x: int) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def check_summary(summary_path: Path, directory: Path) -> tuple[dict, dict[str, list[dict]]]:
    summary = json.loads(summary_path.read_text())
    records: dict[str, list[dict]] = {}
    for name, arm in summary["arms"].items():
        path = directory / f"per_example_{name}.jsonl"
        if sha256_file(path) != arm["per_example_sha256"]:
            raise SystemExit(f"{path}: per-example hash mismatch")
        rows = read_jsonl(path)
        correct = sum(int(row["correct"]) for row in rows)
        if len(rows) != arm["n"] or correct != arm["correct"]:
            raise SystemExit(
                f"{path}: recomputed {correct}/{len(rows)} != stored "
                f"{arm['correct']}/{arm['n']}"
            )
        lo, hi = wilson(correct, len(rows))
        if abs(lo - arm["wilson95"][0]) > 1e-12 or abs(hi - arm["wilson95"][1]) > 1e-12:
            raise SystemExit(f"{path}: Wilson interval mismatch")
        seen = {int(row["i"]) for row in rows}
        if seen != set(range(arm["n"])):
            raise SystemExit(f"{path}: row indices are not 0..n-1")
        records[name] = rows
    return summary, records


def check_inputs() -> tuple[dict, dict[str, dict]]:
    data = json.loads(INPUTS.read_text())
    if len(data["rows"]) != N_ROWS or len(data["labels"]) != N_LABELS:
        raise SystemExit("inputs.json does not carry 3,080 rows and 77 labels")
    if data["jev_correct"] != sum(int(row["jev_correct"]) for row in data["rows"]):
        raise SystemExit("inputs.json Jev total does not match its rows")
    by_id = {}
    for row in data["rows"]:
        if row["gold"] != data["labels"][int(row["gold_index"])]:
            raise SystemExit(f"{row['id']}: gold label and index disagree")
        by_id[str(row["id"])] = row
    if len(by_id) != N_ROWS:
        raise SystemExit("inputs.json has duplicate row ids")
    return data, by_id


def prompt_budget() -> dict:
    budget = json.loads(PROMPT_BUDGET.read_text())
    for name in ("descriptions.json", "categories.json"):
        source = PROMPT_DIR / name
        if not source.is_file():
            raise SystemExit(f"missing comparator prompt source {source}")
        if sha256(source) != budget["source"][name]["sha256"]:
            raise SystemExit(f"{source}: hash does not match prompt_budget.json")
    if budget["n"] != N_ROWS or budget["labels"] != N_LABELS:
        raise SystemExit("prompt_budget.json does not cover the 3,080 rows")
    return budget


def merge_records(
    records: list[dict], by_id: dict[str, dict]
) -> list[tuple[dict, dict]]:
    merged = []
    for record in records:
        row = by_id.get(str(record["id"]))
        if row is None:
            raise SystemExit(f"{record['id']}: per-example row is absent from inputs")
        if int(record["gold_index"]) != int(row["gold_index"]):
            raise SystemExit(f"{record['id']}: gold index differs from inputs")
        merged.append((record, row))
    return merged


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, path: Path) -> None:
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Creator": "thesis/gen_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def fig_pooling(q: dict, m: dict, l: dict, path: Path) -> None:
    old = [q["arms"]["old_a0"]["accuracy"], q["arms"]["old_a2c"]["accuracy"],
           m["arms"]["examples24_5120"]["accuracy"]]
    new = [q["arms"]["query_a0"]["accuracy"], q["arms"]["query_a2c"]["accuracy"],
           l["arms"]["len_a2c"]["accuracy"]]
    x = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(x - width / 2, [100 * v for v in old], width, color=MEAN,
           edgecolor="black", linewidth=0.45, label="Mean-pool head")
    ax.bar(x + width / 2, [100 * v for v in new], width, color=QUERY,
           edgecolor="black", linewidth=0.45, label="Message-query head")
    for xi, value in zip(x - width / 2, old):
        ax.text(xi, 100 * value - 2.8, f"{100 * value:.2f}", ha="center",
                va="top", fontsize=7.4, color="white", fontweight="bold",
                zorder=5)
    for xi, value in zip(x + width / 2, new):
        ax.text(xi, 100 * value - 2.8, f"{100 * value:.2f}", ha="center",
                va="top", fontsize=7.4, color="white", fontweight="bold",
                zorder=5)
    ax.axhline(100 * q["state"]["jev_accuracy"], color=JEV, linestyle="--",
               linewidth=0.9, label="Jev reference")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "message only\n(no examples)",
        "native window\n(1,020 B)",
        "packed 5,120 B\n(all 24 examples)",
    ])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="0.92", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.16))
    save(fig, path)


def fig_budget(q: dict, l: dict, budget: dict, path: Path) -> None:
    arms = [q["arms"]["query_a0"], q["arms"]["query_a2c"], l["arms"]["len_a2c"]]
    xs = [arm["mean_context_bytes"] for arm in arms]
    ys = [100 * arm["accuracy"] for arm in arms]
    figs = [arm["mean_effective_examples"] for arm in arms]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    ax.plot(xs, ys, color=QUERY, marker="o", markersize=4.5,
            linewidth=1.2, label="Query head")
    offsets = [(10, -20), (-84, -26), (10, -20)]
    for x, y, ex, offset in zip(xs, ys, figs, offsets):
        ax.annotate(f"{y:.2f}%\n{ex:.1f} examples", (x, y),
                    textcoords="offset points", xytext=offset,
                    fontsize=7.2, color=QUERY)
    ax.axhline(100 * q["state"]["jev_accuracy"], color=JEV, linestyle="--",
               linewidth=0.9, label="Jev reference")
    ax.axvline(budget["prompt_bytes_median"], color=JEV, linestyle=":",
               linewidth=0.9)
    ax.scatter([budget["prompt_bytes_median"]], [100 * q["state"]["jev_accuracy"]],
               marker="D", s=24, color=JEV, zorder=3)
    ax.annotate(
        f"Jev prompt median\n{budget['prompt_bytes_median']:,} B",
        (budget["prompt_bytes_median"], 100 * q["state"]["jev_accuracy"]),
        textcoords="offset points", xytext=(-72, 12), fontsize=7.2, color=JEV,
    )
    ax.set_xscale("log")
    ax.set_xlim(40, 15_000)
    ax.set_ylim(50, 96)
    ax.set_xticks([50, 100, 500, 1_000, 5_000, 10_000])
    ax.set_xticklabels(["50", "100", "500", "1,000", "5,000", "10,000"])
    ax.set_xlabel("Mean context length (bytes, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(color="0.92", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center right")
    save(fig, path)


def fig_train_test(l: dict, path: Path) -> None:
    sep = l["train_test_separation"]
    train = 100 * sep["train_accuracy"]
    test = 100 * sep["test_accuracy"]
    tr_lo, tr_hi = wilson(sep["train_correct"], sep["train_n"])
    te_lo, te_hi = wilson(sep["test_correct"], sep["test_n"])
    values = [train, test]
    lo = [100 * tr_lo, 100 * te_lo]
    hi = [100 * tr_hi, 100 * te_hi]
    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    bars = ax.bar([0, 1], values, width=0.52, color=[GREY, QUERY],
                  edgecolor="black", linewidth=0.45)
    ax.errorbar([0, 1], values,
                yerr=[np.array(values) - np.array(lo), np.array(hi) - np.array(values)],
                fmt="none", ecolor="black", elinewidth=0.7, capsize=2.2)
    for bar, value, correct, n in zip(
        bars, values, [sep["train_correct"], sep["test_correct"]],
        [sep["train_n"], sep["test_n"]],
    ):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.55,
                f"{value:.2f}%\n{correct:,}/{n:,}", ha="center", va="bottom",
                fontsize=7.4)
    ax.axhline(100 * l["state"]["jev_accuracy"], color=JEV, linestyle="--",
               linewidth=0.9)
    ax.text(0.52, 100 * l["state"]["jev_accuracy"] + 0.35,
            f"Jev {100 * l['state']['jev_accuracy']:.2f}", color=JEV,
            fontsize=7.4, ha="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
    ax.annotate("", xy=(1.32, test), xytext=(1.32, train),
                arrowprops={"arrowstyle": "<->", "color": "black", "lw": 0.8})
    ax.text(1.38, 88.8, f"{train - test:.2f}\npoints", fontsize=7.2,
            va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["train, 4,000 rows", "test, 3,080 rows"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(80, 100)
    ax.set_xlim(-0.55, 1.95)
    ax.grid(axis="y", color="0.92", linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, path)


def class_errors(
    rows: list[tuple[dict, dict]], labels: list[str], *, jev: bool
) -> tuple[list[int], list[int], int]:
    errors = [0] * N_LABELS
    totals = [0] * N_LABELS
    for record, row in rows:
        index = int(row["gold_index"])
        totals[index] += 1
        correct = int(row["jev_correct"]) if jev else int(record["correct"])
        if not correct:
            errors[index] += 1
    if totals != [40] * N_LABELS:
        raise SystemExit("the Banking77 evaluation is not 40 rows per class")
    return errors, totals, sum(errors)


def error_summary(rows: list[tuple[dict, dict]], labels: list[str]) -> dict:
    az, _, az_total = class_errors(rows, labels, jev=False)
    jev, _, jev_total = class_errors(rows, labels, jev=True)
    order = sorted(range(N_LABELS), key=lambda i: (-az[i], -jev[i], labels[i]))
    share = np.cumsum([az[i] for i in order]) / az_total * 100
    share_jev = np.cumsum([jev[i] for i in order]) / jev_total * 100
    return {
        "az": az,
        "jev": jev,
        "order": order,
        "share": share,
        "share_jev": share_jev,
        "az_total": az_total,
        "jev_total": jev_total,
        "top10": float(share[9]),
        "top12": float(share[11]),
        "top_class": labels[order[0]],
        "top_class_count": az[order[0]],
        "top_class_jev": jev[order[0]],
        "classes_with_error": sum(1 for value in az if value),
    }


def fig_errors(
    labels: list[str], summary: dict, path: Path
) -> None:
    az, jev = summary["az"], summary["jev"]
    order = summary["order"]
    az_total, jev_total = summary["az_total"], summary["jev_total"]
    share, share_jev = summary["share"], summary["share_jev"]
    x = np.arange(1, N_LABELS + 1) / N_LABELS * 100

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.2, 3.9),
        gridspec_kw={"width_ratios": [1.0, 1.32], "wspace": 0.55},
    )
    ax1.plot([0, 100], [0, 100], color=GREY, linestyle=":", linewidth=0.9,
             label="equal share")
    ax1.plot(x, share, color=QUERY, linewidth=1.2,
             label=f"Azurite ({az_total} errors)")
    ax1.plot(x, share_jev, color=JEV, linewidth=1.0, linestyle="--",
             label=f"Jev ({jev_total} errors)")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Classes, most errors first (%)")
    ax1.set_ylabel("Cumulative errors (%)")
    ax1.grid(color="0.92", linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="lower right", fontsize=7)

    top = order[:12]
    y = np.arange(len(top))
    ax2.barh(y - 0.18, [az[i] for i in top], height=0.36, color=QUERY,
             edgecolor="black", linewidth=0.35, label="Azurite")
    ax2.barh(y + 0.18, [jev[i] for i in top], height=0.36, color=JEV,
             alpha=0.55, edgecolor="black", linewidth=0.35, label="Jev")
    ax2.set_yticks(y)
    ax2.set_yticklabels(
        [textwrap.fill(labels[i].replace("_", " "), 22) for i in top],
        fontsize=7.0,
    )
    ax2.yaxis.tick_right()
    ax2.tick_params(axis="y", length=0, pad=2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Errors (of 40 rows per class)")
    ax2.set_xlim(0, max(max(az[i] for i in top), max(jev[i] for i in top)) * 1.15)
    ax2.grid(axis="x", color="0.92", linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="upper center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, 1.14))
    save(fig, path)


def write_macros(stats: dict, budget: dict) -> None:
    def esc(label: str) -> str:
        return r"\texttt{" + label.replace("_", r"\_") + "}"

    lines = [
        "% generated by thesis/gen_figures.py — do not edit by hand",
        rf"\newcommand{{\FigErrorTotal}}{{{num(stats['az_total'])}}}",
        rf"\newcommand{{\FigErrorJevTotal}}{{{num(stats['jev_total'])}}}",
        rf"\newcommand{{\FigErrorTopTenShare}}{{{stats['top10']:.1f}}}",
        rf"\newcommand{{\FigErrorTopTwelveShare}}{{{stats['top12']:.1f}}}",
        rf"\newcommand{{\FigErrorTopClass}}{{{esc(stats['top_class'])}}}",
        rf"\newcommand{{\FigErrorTopClassCount}}{{{stats['top_class_count']}}}",
        rf"\newcommand{{\FigErrorTopClassJev}}{{{stats['top_class_jev']}}}",
        rf"\newcommand{{\FigErrorClassesWithError}}{{{stats['classes_with_error']}}}",
        rf"\newcommand{{\FigJevPromptMedian}}{{{num(budget['prompt_bytes_median'])}}}",
        rf"\newcommand{{\FigJevDefinitionsBytes}}{{{num(budget['definitions_bytes'])}}}",
        rf"\newcommand{{\FigJevExamplesBytes}}{{{num(budget['examples_bytes_median'])}}}",
        rf"\newcommand{{\FigTrainGapAbs}}{{{100 * abs(stats['train_minus_test']):.2f}}}",
    ]
    MACROS.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify without writing")
    args = ap.parse_args()

    q, q_records = check_summary(Q / "summary.json", Q)
    l, l_records = check_summary(L / "summary.json", L)
    m, m_records = check_summary(M / "summary.json", M)
    data, by_id = check_inputs()
    budget = prompt_budget()
    rows = merge_records(q_records["query_a2c"], by_id)
    if len(rows) != N_ROWS:
        raise SystemExit("query_a2c does not cover all 3,080 rows")

    stats = error_summary(rows, data["labels"])
    stats["train_minus_test"] = float(l["train_test_separation"]["test_minus_train"])
    if stats["az_total"] != sum(1 for record, _ in rows if not int(record["correct"])):
        raise SystemExit("query error total disagrees with the records")

    print(
        f"query errors {stats['az_total']}, Jev errors {stats['jev_total']}; "
        f"top-10 classes hold {stats['top10']:.1f}% of query errors; "
        f"largest class {stats['top_class']} ({stats['top_class_count']})"
    )
    if args.check:
        return 0

    style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pooling(q, m, l, OUT / "fig_pooling.pdf")
    fig_budget(q, l, budget, OUT / "fig_budget.pdf")
    fig_train_test(l, OUT / "fig_train_test.pdf")
    fig_errors(data["labels"], stats, OUT / "fig_error_concentration.pdf")
    write_macros(stats, budget)
    print(f"wrote {OUT} and {MACROS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
