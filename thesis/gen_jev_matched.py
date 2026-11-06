#!/usr/bin/env python3
"""Generate the protocol-matched Banking77 table and macros.

Reads `results/jev_matched/summary.json` and every per-example record it names.
Recomputes each arm's total and Wilson interval from the records before writing
a cell, and refuses to run if a stored value disagrees. No number is typed.

    python thesis/gen_jev_matched.py            # write fragments
    python thesis/gen_jev_matched.py --check    # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.stats import wilson  # noqa: E402

SRC = ROOT / "results" / "jev_matched"
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


def science(p: float) -> str:
    if p == 0:
        return r"$<10^{-300}$"
    exp = int(math.floor(math.log10(p)))
    mant = p / 10 ** exp
    return rf"${mant:.2f}\times10^{{{exp}}}$"


def load() -> tuple[dict, dict[str, list[dict]]]:
    summary = json.loads((SRC / "summary.json").read_text())
    records: dict[str, list[dict]] = {}
    for name, arm in summary["arms"].items():
        path = SRC / f"per_example_{name}.jsonl"
        if sha256(path) != arm["per_example_sha256"]:
            raise SystemExit(f"{name}: per-example file hash mismatch")
        rows = [json.loads(line) for line in path.open()]
        if len(rows) != arm["n"]:
            raise SystemExit(f"{name}: {len(rows)} records != {arm['n']}")
        k = sum(int(r["correct"]) for r in rows)
        if k != arm["correct"]:
            raise SystemExit(f"{name}: recomputed {k} != stored {arm['correct']}")
        lo, hi = wilson(k, len(rows))
        if abs(lo - arm["wilson95"][0]) > 1e-12 or abs(hi - arm["wilson95"][1]) > 1e-12:
            raise SystemExit(f"{name}: Wilson interval disagrees with summary")
        records[name] = rows
    return summary, records


def macros(summary: dict, records: dict[str, list[dict]]) -> list[str]:
    a = summary["arms"]
    p = summary["paired"]
    primary = summary["decision"]["primary_arm"]
    assert primary == "examples24_5120"
    jev = summary["state"]
    jlo, jhi = wilson(jev["jev_correct"], jev["n"])

    def pct_macro(name: str, value: float) -> str:
        return rf"\newcommand{{\JevMatch{name}}}{{{pct(value)}}}"

    def ci_macro(name: str, lo: float, hi: float) -> list[str]:
        return [pct_macro(name + "Lo", lo), pct_macro(name + "Hi", hi)]

    m = [
        rf"\newcommand{{\JevMatchN}}{{{jev['n']:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\JevMatchCorrectJev}}{{{jev['jev_correct']:,}}}".replace(",", "{,}"),
        pct_macro("PctJev", jev["jev_accuracy"]),
    ]
    m += ci_macro("Jev", jlo, jhi)
    for key, label in (("no_examples_1024", "NoEx"),
                       ("examples24_1024", "ATwo"),
                       ("examples24_5120", "AOne")):
        arm = a[key]
        m.append(rf"\newcommand{{\JevMatchCorrect{label}}}{{{arm['correct']:,}}}".replace(",", "{,}"))
        m.append(pct_macro(f"Pct{label}", arm["accuracy"]))
        m += ci_macro(label, *arm["wilson95"])
        m.append(rf"\newcommand{{\JevMatchGap{label}}}{{{pp(p[key]['diff_vs_jev'])}}}".replace("$-$", "-"))
        m.append(rf"\newcommand{{\JevMatchLower{label}}}{{{pp(p[key]['bootstrap_lower_95'])}}}".replace("$-$", "-"))
        m.append(rf"\newcommand{{\JevMatchP{label}}}{{{science(p[key]['mcnemar']['p']).strip('$')}}}")
    m += [
        rf"\newcommand{{\JevMatchEffATwo}}{{{a['examples24_1024']['mean_effective_examples']:.1f}}}",
        rf"\newcommand{{\JevMatchEffATwoMin}}{{{a['examples24_1024']['min_effective_examples']}}}",
        rf"\newcommand{{\JevMatchEffATwoMax}}{{{a['examples24_1024']['max_effective_examples']}}}",
        rf"\newcommand{{\JevMatchEffAOne}}{{{a['examples24_5120']['mean_effective_examples']:.1f}}}",
        rf"\newcommand{{\JevMatchHeadSteps}}{{{num(summary['head']['steps'])}}}",
        rf"\newcommand{{\JevMatchArms}}{{{len(a)}}}",
        rf"\newcommand{{\JevMatchPreregSha}}{{{summary['prereg_sha256'][:12]}}}",
        rf"\newcommand{{\JevMatchAmendmentSha}}{{{sha256(SRC / 'PRE_REGISTRATION_AMENDMENT_1.md')[:12]}}}",
        rf"\newcommand{{\JevMatchVerdict}}{{the {summary['decision']['verdict']}}}",
        rf"\newcommand{{\JevMatchMargin}}{{{abs(summary['decision']['margin_points']):.1f}}}",
        rf"\newcommand{{\JevMatchGapPrimary}}{{{abs(a['no_examples_1024']['accuracy'] - jev['jev_accuracy']) * 100:.2f}}}",
        rf"\newcommand{{\JevMatchAbsGapAOne}}{{{abs(p[primary]['diff_vs_jev']) * 100:.2f}}}",
        rf"\newcommand{{\JevMatchAbsGapNoEx}}{{{abs(p['no_examples_1024']['diff_vs_jev']) * 100:.2f}}}",
    ]
    for key, label in (("no_examples_1024", "NoEx"), ("examples24_5120", "AOne")):
        counts = Counter(r["pred"] for r in records[key])
        n = len(records[key])
        m += [
            rf"\newcommand{{\JevMatchDistinct{label}}}{{{len(counts)}}}",
            rf"\newcommand{{\JevMatchTop{label}}}{{{100 * counts.most_common(1)[0][1] / n:.1f}}}",
        ]
        if label == "AOne":
            m.append(
                rf"\newcommand{{\JevMatchTopThree{label}}}{{{100 * sum(v for _, v in counts.most_common(3)) / n:.1f}}}"
            )
    return m


def table(summary: dict) -> str:
    a = summary["arms"]
    p = summary["paired"]
    jev = summary["state"]
    jlo, jhi = wilson(jev["jev_correct"], jev["n"])

    def row(label: str, arm: dict, examples: str, window: str, eff: str,
            pair: dict | None, cite: bool = False) -> str:
        lo, hi = arm["wilson95"]
        diff = "---" if pair is None else pp(pair["diff_vs_jev"])
        pval = "---" if pair is None else science(pair["mcnemar"]["p"])
        body = (rf"{label} & {examples} & {window} & {eff} & {num(arm['n'])} & "
                rf"{num(arm['correct'])} & {pct(arm['accuracy'])} & "
                rf"{pct(lo)}--{pct(hi)} & {diff} & {pval}")
        return body.rstrip() + " " + "\\\\"

    rows = [
        row("A0 message only", a["no_examples_1024"], "0", "1{,}024", "0.0", p["no_examples_1024"]),
        row("A2 24 examples, truncated", a["examples24_1024"], "24", "1{,}024",
            f"{a['examples24_1024']['mean_effective_examples']:.1f}", p["examples24_1024"]),
        row("A1 24 examples (primary)", a["examples24_5120"], "24", "5{,}120",
            f"{a['examples24_5120']['mean_effective_examples']:.1f}", p["examples24_5120"]),
        row("Jev 1.13, cited", {"n": jev["n"], "correct": jev["jev_correct"],
                                "accuracy": jev["jev_accuracy"],
                                "wilson95": (jlo, jhi)},
            "24", "vendor", "24", None),
    ]
    return r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{The protocol-matched Banking77 comparison. All 3{,}080 official test
rows, in the public run's order; the same 24 retrieved labelled examples per row,
in the same order; gold labels from the same frozen prediction file. Azurite is
not given the 77 category definitions, so its information budget is strictly
smaller than Jev's. The primary arm was fixed at A1 with a $-2.0$ point
non-inferiority margin before any forward pass. Wilson intervals are 95\%;
McNemar is exact and two-sided; the paired bootstrap lower bound is the 5th
percentile of 10{,}000 resamples, seed 0. Read from
\texttt{results/jev\_matched/summary.json}; the pre-registration hash is
\texttt{\JevMatchPreregSha} and the evaluator-only amendment is
\texttt{\JevMatchAmendmentSha}.}
\label{tab:jev-matched}
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
Arm & Ex. & Window & Eff. & $n$ & Correct & Acc. & 95\% Wilson & $\Delta$ vs Jev & McNemar $p$ \\
 & & (bytes) & ex. & & & (\%) & & (pp) & \\
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
    summary, records = load()
    fragment = table(summary)
    macro_text = ("% generated by thesis/gen_jev_matched.py — do not edit by hand\n"
                  + "\n".join(macros(summary, records)) + "\n")
    if args.check:
        print(f"checked {len(summary['arms'])} arms; verdict {summary['decision']['verdict']}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jev_matched.tex").write_text(fragment, encoding="utf-8")
    (OUT / "macros_jev.tex").write_text(macro_text, encoding="utf-8")
    print(f"wrote jev_matched.tex and macros_jev.tex to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
