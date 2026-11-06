#!/usr/bin/env python3
"""Generate the query-head table and macros from the stored evaluation.

Reads `results/jev_query/eval/summary.json` and its per-example files.
Recomputes every arm total and Wilson interval before writing a cell.

    python thesis/gen_query.py            # write fragments
    python thesis/gen_query.py --check    # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.stats import wilson  # noqa: E402

SRC = ROOT / "results" / "jev_query" / "eval"
OUT = ROOT / "thesis" / "tables"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def num(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def pp(x: float) -> str:
    return f"{100 * x:+.2f}".replace("-", "$-$")


def pp_raw(x: float) -> str:
    return f"{100 * x:+.2f}"


def science(p: float) -> str:
    if p == 0:
        return r"$<10^{-300}$"
    exp = int(math.floor(math.log10(p)))
    mant = p / 10 ** exp
    return rf"${mant:.2f}\times10^{{{exp}}}$"


def load() -> dict:
    summary = json.loads((SRC / "summary.json").read_text())
    for name, arm in summary["arms"].items():
        path = SRC / f"per_example_{name}.jsonl"
        if sha256(path) != arm["per_example_sha256"]:
            raise SystemExit(f"{name}: per-example hash mismatch")
        rows = [json.loads(line) for line in path.open()]
        k = sum(int(r["correct"]) for r in rows)
        if len(rows) != arm["n"] or k != arm["correct"]:
            raise SystemExit(f"{name}: recomputed {k}/{len(rows)} != stored")
        lo, hi = wilson(k, len(rows))
        if abs(lo - arm["wilson95"][0]) > 1e-12 or abs(hi - arm["wilson95"][1]) > 1e-12:
            raise SystemExit(f"{name}: Wilson interval mismatch")
    return summary


def _next_hours(scale: float, adaptation: float, evaluation: float) -> float:
    meta = json.loads((ROOT / "results/jev_query/train_meta.json").read_text())
    return meta["train_seconds"] / 3600 * scale + adaptation + evaluation


def macros(s: dict) -> list[str]:
    a, p, d = s["arms"], s["paired"], s["decision"]
    jev = s["state"]["jev_accuracy"]
    m = [
        rf"\newcommand{{\QueryJevPct}}{{{pct(jev)}}}",
        rf"\newcommand{{\QueryN}}{{{num(s['state']['n'])}}}",
        rf"\newcommand{{\QueryEff}}{{{a['query_a2c']['mean_effective_examples']:.1f}}}",
        rf"\newcommand{{\QueryEffMin}}{{{a['query_a2c']['min_effective_examples']}}}",
        rf"\newcommand{{\QueryEffMax}}{{{a['query_a2c']['max_effective_examples']}}}",
        rf"\newcommand{{\QueryVerdict}}{{{d['verdict']}}}",
        rf"\newcommand{{\QueryPrereg}}{{{s['prereg_sha256'][:12]}}}",
        rf"\newcommand{{\QueryHeadSha}}{{{s['heads']['query']['sha256'][:12]}}}",
        rf"\newcommand{{\QueryNonInferior}}{{{'yes' if d['noninferior_at_minus_2pp'] else 'no'}}}",
    ]
    for key, label in (("old_a0", "OldZero"), ("old_a2c", "OldTwo"),
                       ("query_a0", "QueryZero"), ("query_a2c", "QueryTwo")):
        arm, pair = a[key], p[key]
        m += [
            rf"\newcommand{{\QueryCorrect{label}}}{{{num(arm['correct'])}}}",
            rf"\newcommand{{\QueryPct{label}}}{{{pct(arm['accuracy'])}}}",
            rf"\newcommand{{\QueryLo{label}}}{{{pct(arm['wilson95'][0])}}}",
            rf"\newcommand{{\QueryHi{label}}}{{{pct(arm['wilson95'][1])}}}",
            rf"\newcommand{{\QueryGap{label}}}{{{pp_raw(pair['diff_vs_jev'])}}}",
            rf"\newcommand{{\QueryLower{label}}}{{{pp_raw(pair['bootstrap_lower_95'])}}}",
            rf"\newcommand{{\QueryP{label}}}{{{science(pair['mcnemar']['p'])}}}",
        ]
    m += [
        rf"\newcommand{{\QueryDiff}}{{{pp_raw(d['diff_vs_jev'])}}}",
        rf"\newcommand{{\QueryDiffAbs}}{{{abs(d['diff_vs_jev']) * 100:.2f}}}",
        rf"\newcommand{{\QueryLowerPrimary}}{{{pp_raw(d['bootstrap_lower_95'])}}}",
        rf"\newcommand{{\QueryPValue}}{{{science(d['mcnemar_p'])}}}",
        rf"\newcommand{{\QueryPValueText}}{{p = {science(d['mcnemar_p']).strip('$')}}}",
        rf"\newcommand{{\QueryMechPText}}{{p = {science(p['query_a2c_vs_old_a2c']['mcnemar']['p']).strip('$')}}}",
        rf"\newcommand{{\QueryMechGain}}{{{pp_raw(p['query_a2c_vs_old_a2c']['diff'])}}}",
        rf"\newcommand{{\QueryMechGainAbs}}{{{abs(p['query_a2c_vs_old_a2c']['diff']) * 100:.2f}}}",
        rf"\newcommand{{\QueryMechLower}}{{{pp_raw(p['query_a2c_vs_old_a2c']['bootstrap_lower_95'])}}}",
        rf"\newcommand{{\QueryMechP}}{{{science(p['query_a2c_vs_old_a2c']['mcnemar']['p'])}}}",
        rf"\newcommand{{\QueryVsZeroGain}}{{{pp_raw(p['query_a2c_vs_query_a0']['diff'])}}}",
        rf"\newcommand{{\QueryVsOldZeroGain}}{{{pp_raw(a['query_a2c']['accuracy'] - a['old_a0']['accuracy'])}}}",
        rf"\newcommand{{\QueryTrainMinutes}}{{{json.loads((ROOT / 'results/jev_query/train_meta.json').read_text())['train_seconds'] / 60:.0f}}}",
        rf"\newcommand{{\QuerySecStep}}{{{json.loads((ROOT / 'results/jev_query/train_meta.json').read_text())['ms_per_step'] / 1000:.2f}}}",
        rf"\newcommand{{\QueryLastLoss}}{{{json.loads((ROOT / 'results/jev_query/train_meta.json').read_text())['last_loss']:.4f}}}",
        # Projections, not measurements: the measured head time scaled by the
        # sequence length, plus a stated adaptation and evaluation allowance.
        rf"\newcommand{{\QueryNextShortHours}}{{{_next_hours(5.0, 0.5, 0.05):.1f}}}",
        rf"\newcommand{{\QueryNextLongLo}}{{{_next_hours(16.0, 1.5, 0.10):.1f}}}",
        rf"\newcommand{{\QueryNextLongHi}}{{{_next_hours(32.0, 3.0, 0.20):.1f}}}",
    ]
    return m


def table(s: dict) -> str:
    a, p = s["arms"], s["paired"]
    jev = s["state"]
    jlo, jhi = wilson(jev["jev_correct"], jev["n"])

    def row(label: str, arm: dict, examples: str, pair: dict | None) -> str:
        lo, hi = arm["wilson95"]
        diff = "---" if pair is None else pp(pair["diff_vs_jev"])
        pv = "---" if pair is None else science(pair["mcnemar"]["p"])
        body = (rf"{label} & {examples} & {num(arm['n'])} & {num(arm['correct'])} & "
                rf"{pct(arm['accuracy'])} & {pct(lo)}--{pct(hi)} & {diff} & {pv}")
        return body + " \\\\"

    rows = [
        row("Old head, message only", a["old_a0"], "0.0", p["old_a0"]),
        row("Old head, examples (control)", a["old_a2c"],
            f"{a['old_a2c']['mean_effective_examples']:.1f}", p["old_a2c"]),
        row("Query head, message only", a["query_a0"], "0.0", p["query_a0"]),
        row("Query head, examples (primary)", a["query_a2c"],
            f"{a['query_a2c']['mean_effective_examples']:.1f}", p["query_a2c"]),
        row("Jev 1.13, cited", {"n": jev["n"], "correct": jev["jev_correct"],
                                "accuracy": jev["jev_accuracy"],
                                "wilson95": (jlo, jhi)}, "24", None),
    ]
    return r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{The message-query head on all 3{,}080 official Banking77 rows, with
the exact 24 public Jev examples per row and the same gold/Jev vector as the
matched test. ``Examples'' is the mean number of complete lines that fit in the
1{,}020-byte context budget. The old head is the E4 mean-pool head; the query
head replaces mean pooling with a masked attention pool queried by the customer
message. \texttt{old\_a2c} is the head control under the same packing;
\texttt{query\_a0} is the example control. The P0 bar is Jev's \QueryJevPct\%.
Wilson intervals are 95\%;
McNemar is exact and two-sided; the bootstrap lower bound is the 5th percentile
of 10{,}000 paired resamples, seed 0. Read from
\texttt{results/jev\_query/eval/summary.json}.}
\label{tab:query-head}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Arm & Ex. & $n$ & Correct & Acc. & 95\% Wilson & $\Delta$ vs Jev & McNemar $p$ \\
 & & & & (\%) & & (pp) & \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    summary = load()
    if args.check:
        print(f"checked {len(summary['arms'])} arms; verdict {summary['decision']['verdict']}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "query.tex").write_text(table(summary), encoding="utf-8")
    (OUT / "macros_query.tex").write_text(
        "% generated by thesis/gen_query.py — do not edit by hand\n"
        + "\n".join(macros(summary)) + "\n", encoding="utf-8")
    print(f"wrote query.tex and macros_query.tex to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
