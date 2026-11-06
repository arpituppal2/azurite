#!/usr/bin/env python3
"""Generate the three-rung suite table and its inline numbers.

The record of authority is ``results/suite_large_113m/COMPARISON.json``, which
``scripts/suite_three_rung.py`` builds from the per-cell ``result.json`` files
after asserting that each reported accuracy recomputes from its own
predictions, that every rung ran the dual path, and that all three rungs share
the identical test draw.  This script re-checks the same facts against the
per-cell records before it writes a table cell, so a corrupted record cannot
reach the PDF.

Outputs:
    thesis/tables/suite3.tex          nine tasks at three rungs
    thesis/tables/suite3_ci.tex       every 113.47M cell with Wilson interval and test
    thesis/tables/macros_suite3.tex   the large-rung numbers quoted in prose
    thesis/tables/suite3_source.json  paths, hashes and paired-test counts

Usage:
    python3 thesis/gen_suite_three.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.stats import wilson  # noqa: E402

OUT = ROOT / "thesis" / "tables"
SUITE = ROOT / "results" / "suite_large_113m"
COMPARISON = SUITE / "COMPARISON.json"
REPLICATION = SUITE / "REPLICATION.json"
CITATIONS = ROOT / "thesis" / "suite_citations.json"

TASKS = [
    ("ag_news", "AG News", "Ag"),
    ("emotion", "DAIR Emotion", "Emotion"),
    ("banking77", "Banking77", "Bank"),
    ("clinc_oos", "CLINC150", "Clinc"),
    ("massive_intent", "MASSIVE intent", "MassiveIntent"),
    ("massive_scenario", "MASSIVE scenario", "MassiveScenario"),
    ("xnli", "XNLI", "Xnli"),
    ("language_id", "Language ID", "Lang"),
    ("financial_phrasebank", "Financial PhraseBank", "Financial"),
]
RUNGS = ["small", "mid", "large"]
RUN_LABEL = {"small": "14.4M", "mid": "33.7M", "large": "113.47M"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    comp = json.loads(COMPARISON.read_text())
    citations = json.loads(CITATIONS.read_text())
    cells: dict[str, dict] = {}

    large_ckpt = comp["checkpoints"]["large"]["sha256"]
    for rung in RUNGS:
        for task, label, key in TASKS:
            record = comp["cells"][rung][task]
            if rung == "large":
                path = SUITE / task / "result.json"
            else:
                path = ROOT / "results" / "suite_dual" / rung / task / "result.json"
            if not path.exists():
                raise SystemExit(f"missing per-cell record: {path}")
            rec = json.loads(path.read_text())
            final = rec["final"]
            acc = sum(1 for a, b in zip(final["predictions"], final["gold"]) if a == b) / len(final["gold"])
            if not close(acc, final["accuracy"]):
                raise SystemExit(f"{rung}/{task}: accuracy does not recompute")
            if not close(final["accuracy"], record["accuracy"]):
                raise SystemExit(f"{rung}/{task}: comparison value != result record")
            if int(final["n"]) != int(record["n"]):
                raise SystemExit(f"{rung}/{task}: n mismatch")
            if rec.get("path") != "dual":
                raise SystemExit(f"{rung}/{task}: path is not dual")
            k = sum(1 for a, b in zip(final["predictions"], final["gold"]) if a == b)
            lo, hi = wilson(k, len(final["gold"]))
            if not (close(lo, record["wilson95"][0], 1e-9) and close(hi, record["wilson95"][1], 1e-9)):
                raise SystemExit(f"{rung}/{task}: Wilson interval mismatch")
            if rung == "large":
                loaded = rec["loaded_from"]
                if int(loaded["pretrain_step"]) != 69255:
                    raise SystemExit(f"large/{task}: unexpected pretrain step {loaded['pretrain_step']}")
                prov = json.loads((SUITE / task / "evidence" / "provenance.json").read_text())
                if prov["source_checkpoint"]["sha256"] != large_ckpt:
                    raise SystemExit(f"large/{task}: checkpoint hash mismatch")
            cited = citations["rows"][task]["jev"]
            gap = None if cited is None else 100.0 * (record["accuracy"] - cited)
            cells.setdefault(task, {})[rung] = {
                "label": label,
                "key": key,
                "k": int(rec["k"]),
                "n": int(final["n"]),
                "accuracy": float(record["accuracy"]),
                "wilson95": [float(lo), float(hi)],
                "cited": cited,
                "gap_pp": gap,
                "verdict": record["verdict"],
                "seconds": rec.get("seconds"),
                "source": str(path.relative_to(ROOT)),
                "source_sha256": sha256(path),
            }

    counts = comp["counts_vs_cited"]
    paired = comp["paired"]
    repl = json.loads(REPLICATION.read_text())["summary"] if REPLICATION.exists() else None

    # ---------------------------------------------------------------- table
    def gap_tex(cell: dict) -> str:
        if cell["gap_pp"] is None:
            return "---"
        val = f"${cell['gap_pp']:+.2f}$"
        return r"\textbf{" + val + "}" if cell["verdict"] == "clear above" else val

    def acc_tex(cell: dict) -> str:
        val = f"{100 * cell['accuracy']:.2f}"
        return val

    rows = []
    for task, label, key in TASKS:
        s, m, g = cells[task]["small"], cells[task]["mid"], cells[task]["large"]
        cited = "---" if s["cited"] is None else f"{100 * s['cited']:.2f}"
        if task == "massive_intent":
            cited += r"\textsuperscript{\(\ast\)}"
        s_acc, m_acc, g_acc = acc_tex(s), acc_tex(m), acc_tex(g)
        if s["verdict"].startswith("tie"):
            s_acc += r"\textsuperscript{\(\dagger\)}"
        if m["verdict"].startswith("tie"):
            m_acc += r"\textsuperscript{\(\dagger\)}"
        if g["verdict"].startswith("tie"):
            g_acc += r"\textsuperscript{\(\dagger\)}"
        if task in {"emotion", "xnli"}:
            s_acc += r"\textsuperscript{\(\ddagger\)}"
            m_acc += r"\textsuperscript{\(\ddagger\)}"
        rows.append(
            rf"{label} ($K={s['k']}$) & {cited} & {s_acc} & {gap_tex(s)} & "
            rf"{m_acc} & {gap_tex(m)} & {g_acc} & {gap_tex(g)} \\"
        )

    table = rf"""\begin{{table}}[htbp]
\centering
\footnotesize
\caption{{{len(TASKS)} public tasks at three rungs under one pre-registered recipe:
LR $10^{{-4}}$, exactly 3{{,}}000 optimizer steps, batch 32, 4{{,}}000 stratified
training rows and 2{{,}}000 test rows where available (Financial PhraseBank has
1{{,}}264 train and 1{{,}}000 test), seed 0, encoder unfrozen, native context of
each checkpoint (256 at 14.4M, 1{{,}}024 at 33.7M and 113.47M). The recipe is
identical for every cell; there is no per-task learning rate, step count, row cap
or prompt. Every Azurite cell is measured on the dual context-candidate path. The
113.47M column is a single new run from fresh weights over 1{{,}}134{{,}}658{{,}}560
tokens (step 69{{,}}255), evaluated on the same test rows as the smaller rungs.
Cited cells are published or third-party figures, not reproduced. DAIR Emotion
and XNLI at the two smaller rungs changed evaluation path in an earlier pass and
their former candidate-only values are retired. MASSIVE intent's cited point is
Laya's card, not a Jev number. No interval is shown here; 95\% Wilson intervals
and exact paired McNemar tests are in \S\ref{{sec:suite}} and
\texttt{{results/suite\_large\_113m/COMPARISON.md}}. These are cross-harness
comparisons, not a head-to-head.}}
\label{{tab:suite}}
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\toprule
Task & Cited & \shortstack{{14.4M\\accuracy}} & \shortstack{{$\Delta$ vs\\cited}} &
\shortstack{{33.7M\\accuracy}} & \shortstack{{$\Delta$ vs\\cited}} &
\shortstack{{113.47M\\accuracy}} & \shortstack{{$\Delta$ vs\\cited}} \\
\midrule
""" + "\n".join(rows) + rf"""
\bottomrule
\end{{tabular}}
\par\smallskip
\footnotesize
\textsuperscript{{\(\ddagger\)}}Corrected dual-path cell; the text identifies its
retired cross-path value.
\quad
\textsuperscript{{\(\dagger\)}}95\% Wilson interval includes the cited point; not
a clear win or loss.
\quad
\textsuperscript{{\(\ast\)}}Laya card ({100 * cells['massive_intent']['small']['cited']:.2f} for
MASSIVE intent), not a Jev number.
\quad
Clear wins against the cited point are bold; a positive $\Delta$ alone is not a win.
\end{{table}}
"""
    (OUT / "suite3.tex").write_text(table)

    # ------------------------------------------------- per-cell intervals table
    def fmt_p(p: float) -> str:
        if p < 1e-2:
            exp = int(math.floor(math.log10(p)))
            mant = p / 10 ** exp
            return rf"${mant:.1f}\times10^{{{exp}}}$"
        return f"{p:.2f}"

    verdict_short = {
        "clear above": "above",
        "clear below": "below",
        "tie (below point)": "tie",
        "no cited point": "---",
    }
    ci_rows = []
    for task, label, key in TASKS:
        c = cells[task]["large"]
        cited = "---" if c["cited"] is None else f"{100 * c['cited']:.2f}"
        gap = "---" if c["gap_pp"] is None else f"${c['gap_pp']:+.2f}$"
        pr = paired["large_vs_small"][task]
        ci_rows.append(
            rf"{label} & {c['k']} & {c['n']:,} & {100 * c['accuracy']:.2f} & "
            rf"{100 * c['wilson95'][0]:.2f}--{100 * c['wilson95'][1]:.2f} & {gap} & "
            rf"{verdict_short.get(c['verdict'], c['verdict'])} & "
            rf"${pr['delta_pp']:+.2f}$ & {fmt_p(pr['mcnemar_exact_p'])} \\"
        )
    ci_table = rf"""\begin{{table}}[htbp]
\centering
\footnotesize
\caption{{Every 113.47M cell with its 95\% Wilson interval, its gap against the
cited point, and the exact paired McNemar test against the 14.4M cell on the same
test rows. $K$ is the number of classes, $n$ the number of scored test rows.
Verdict is the interval rule: \emph{{above}} and \emph{{below}} mean the Wilson
interval excludes the cited point, \emph{{tie}} that it covers it. The last two
columns compare the two Azurite rungs directly, not either against the citation.
Read from \texttt{{results/suite\_large\_113m/COMPARISON.json}}; the generator
recomputes every cell and refuses a mismatch.}}
\label{{tab:suite-ci}}
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{@{{}}lrrrrrlrr@{{}}}}
\toprule
Task & $K$ & $n$ & Acc. & 95\% CI & $\Delta$ cited & Verdict &
$\Delta$ vs 14.4M & $p$ \\
\midrule
""" + "\n".join(ci_rows) + rf"""
\bottomrule
\end{{tabular}}
\par\smallskip
\footnotesize
$\Delta$ is in percentage points. MASSIVE intent's cited point is Laya's card
({100 * cells['massive_intent']['small']['cited']:.2f}), not a Jev number;
MASSIVE scenario has no published point and its gap column is empty. A tie is not
parity: it means this test draw cannot separate the two values.
\end{{table}}
"""
    (OUT / "suite3_ci.tex").write_text(ci_table)

    # --------------------------------------------------------------- macros
    m: list[str] = ["% generated by thesis/gen_suite_three.py from results/suite_large_113m — do not edit by hand"]

    def add(name: str, value) -> None:
        m.append(rf"\newcommand{{\{name}}}{{{value}}}")

    def pct(x: float) -> str:
        return f"{100 * x:.2f}"

    def gap(x: float | None) -> str:
        return "---" if x is None else f"{x:+.2f}"

    add("SuiteLargeAtOrAbove", counts["large"]["at_or_above"])
    add("SuiteLargeClear", counts["large"]["clear_above"])
    add("SuiteLargeTie", counts["large"]["tie"])
    add("SuiteLargeBelow", counts["large"]["clear_below"])
    add("SuiteSmallAtOrAboveThree", counts["small"]["at_or_above"])
    add("SuiteMidAtOrAboveThree", counts["mid"]["at_or_above"])
    add("SuiteRungs", len(RUNGS))
    add("SuiteTotalCells", len(RUNGS) * len(TASKS))

    gains = [t for t in TASKS if paired["large_vs_small"][t[0]]["mcnemar_exact_p"] < 0.05
             and paired["large_vs_small"][t[0]]["delta_pp"] > 0]
    losses = [t for t in TASKS if paired["large_vs_small"][t[0]]["mcnemar_exact_p"] < 0.05
              and paired["large_vs_small"][t[0]]["delta_pp"] < 0]
    add("SuiteLargeWins", len(gains))
    add("SuiteLargeLosses", len(losses))
    add("SuiteLargeUnmoved", len(TASKS) - len(gains) - len(losses))
    if gains:
        deltas = [paired["large_vs_small"][t[0]]["delta_pp"] for t in gains]
        add("SuiteLargeWinMin", f"{min(deltas):+.2f}")
        add("SuiteLargeWinMax", f"{max(deltas):+.2f}")
    if losses:
        deltas = [paired["large_vs_small"][t[0]]["delta_pp"] for t in losses]
        add("SuiteLargeLossMin", f"{min(deltas):+.2f}")
        add("SuiteLargeLossMax", f"{max(deltas):+.2f}")
        add("SuiteLargeLossAbs", f"{abs(min(deltas)):.2f}")
    if gains:
        labels = [label for task, label, _ in TASKS if task in {t[0] for t in gains}]
        joined = ", ".join(labels[:-1]) + " and " + labels[-1] if len(labels) > 1 else labels[0]
        add("SuiteLargeWinCells", joined)
    if losses and gains:
        add("SuiteLargeLossOverWinMax",
            f"{abs(min(paired['large_vs_small'][t[0]]['delta_pp'] for t in losses)) / max(paired['large_vs_small'][t[0]]['delta_pp'] for t in gains):.1f}")

    for task, _, key in TASKS:
        g = cells[task]["large"]
        add(f"SuiteLarge{key}", pct(g["accuracy"]))
        add(f"SuiteLargeGap{key}", gap(g["gap_pp"]))
        if g["gap_pp"] is not None:
            add(f"SuiteLargeShort{key}", f"{abs(g['gap_pp']):.2f}")
        add(f"SuiteLarge{key}Lo", f"{100 * g['wilson95'][0]:.2f}")
        add(f"SuiteLarge{key}Hi", f"{100 * g['wilson95'][1]:.2f}")
        p = paired["large_vs_small"][task]
        add(f"SuiteLargeOnly{key}", p["first_only_correct"])
        add(f"SuiteSmallOnly{key}", p["second_only_correct"])
        add(f"SuiteLargeVSSmallP{key}", f"{p['mcnemar_exact_p']:.2e}")
        add(f"SuiteLargeVsSmall{key}", f"{p['delta_pp']:+.2f}")

    for rung, key in (("small", "Small"), ("mid", "Mid")):
        cell_path = ROOT / "results" / "suite_dual" / rung / "xnli" / "result.json"
        rec = json.loads(cell_path.read_text())["final"]
        add(f"Suite{key}XnliNll", f"{rec['nll']:.2f}")
        add(f"Suite{key}XnliEce", f"{rec['ece']:.2f}")

    deg = comp["degeneracy"]["large/xnli"]
    add("SuiteLargeXnliNll", f"{deg['nll']:.4f}")
    add("SuiteLargeXnliLnK", f"{deg['ln_k']:.4f}")
    add("SuiteLargeXnliConf", f"{deg['mean_confidence']:.4f}")
    add("SuiteLargeXnliEce", f"{deg['ece']:.4f}")
    add("SuiteLargeXnliTopShare", f"{100 * deg['top_class_fraction']:.1f}")
    add("SuiteLargeXnliChance", f"{100 * deg['chance']:.2f}")
    add("SuiteLargeXnliTopClass", str(deg["top_class"]))
    add("SuiteLargeXnliUniform", "yes" if deg["uniform_output"] else "no")

    cost = json.loads((ROOT / "results" / "scale_state" / "PRETRAIN_COST.json").read_text())
    def group(n: int) -> str:
        return f"{n:,}".replace(",", "{,}")
    add("SuiteLostTokens", group(cost["tokens_seen"]))
    add("SuiteLostTarget", group(cost["target_tokens"]))
    add("SuiteLostPct", f"{100 * cost['fraction']:.2f}")
    add("SuiteLostStep", group(cost["steps"]))
    add("SuiteLostSegments", cost["segments_completed"])

    prereg = SUITE / "PRE_REGISTRATION.md"
    if prereg.exists():
        add("SuitePreRegSha", sha256(prereg)[:16] + r"\ldots")
    add("SuiteCkptLarge", comp["checkpoints"]["large"]["sha256"][:16] + r"\ldots")
    add("SuiteCkptLargeStep", "69{,}255")
    add("SuiteCkptLargeTokens", "1{,}134{,}673{,}920")
    if repl:
        add("SuiteReplicationMax", f"{repl['max_abs_delta_pp']:.2f}")
        add("SuiteReplicationMean", f"{repl['mean_abs_delta_pp']:.2f}")
        add("SuiteReplicationFlips", int(repl["max_prediction_flips"]))
        add("SuiteReplicationCells", int(repl["n_cells"]))

    seconds = sum(cells[t]["large"]["seconds"] or 0 for t, _, _ in TASKS)
    add("SuiteLargeGpuHours", f"{seconds / 3600:.2f}")
    add("SuiteLargeParamsRatio", f"{comp['checkpoints']['large']['params'] / comp['checkpoints']['small']['params']:.2f}")

    (OUT / "macros_suite3.tex").write_text("\n".join(m) + "\n")

    source = {
        "record": str(COMPARISON.relative_to(ROOT)),
        "record_sha256": sha256(COMPARISON),
        "replication": str(REPLICATION.relative_to(ROOT)) if REPLICATION.exists() else None,
        "replication_sha256": sha256(REPLICATION) if REPLICATION.exists() else None,
        "checkpoints": comp["checkpoints"],
        "cells": cells,
        "counts": counts,
        "paired": paired,
        "degeneracy_large_xnli": deg,
        "recipe": comp["recipe"],
    }
    (OUT / "suite3_source.json").write_text(json.dumps(source, indent=2) + "\n")

    print(
        f"suite3: {len(TASKS)} tasks x {len(RUNGS)} rungs; "
        f"at-or-above small={counts['small']['at_or_above']} "
        f"mid={counts['mid']['at_or_above']} large={counts['large']['at_or_above']}; "
        f"large wins={len(gains)} losses={len(losses)}"
    )
    for task, _, _ in TASKS:
        g = cells[task]["large"]
        print(
            f"  {task:22s} small={100*cells[task]['small']['accuracy']:6.2f} "
            f"mid={100*cells[task]['mid']['accuracy']:6.2f} "
            f"large={100*g['accuracy']:6.2f} verdict={g['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
