#!/usr/bin/env python3
"""Generate the measured nine-cell suite table from the dual-path record.

    thesis/tables/suite.tex          nine tasks, both rungs, cited gaps
    thesis/tables/macros_suite.tex   every suite number referenced in prose
    thesis/tables/suite_source.json  per-cell paths, hashes, n, gaps

The record of authority is ``results/suite_dual/SUITE.json``. Every Azurite
value below is read from that file (and its per-cell ``result.json`` is checked
against it); no number is typed from a paper, a log or a previous table. The
pre-fix suite is retained only as ``prior`` metadata so a retired value can be
identified as retired rather than silently replaced.

Counts are recomputed from the cells and asserted against the record's own
summary. A mismatch is a build failure, not a footnote.

Usage:
    python3 thesis/gen_suite.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.stats import wilson  # noqa: E402

OUT = ROOT / "thesis" / "tables"
SUITE_DIR = ROOT / "results" / "suite_dual"
SUITE = SUITE_DIR / "SUITE.json"
CHECKPOINTS = SUITE_DIR / "CHECKPOINTS.json"
CITATIONS = ROOT / "thesis" / "suite_citations.json"
DIAGNOSTICS = ROOT / "results" / "chance_diagnosis" / "model_diagnostics.json"
EMOTION = ROOT / "results" / "chance_diagnosis" / "emotion_canonical.json"
XNLI_FIELD = ROOT / "results" / "chance_diagnosis" / "xnli_pairfix" / "xnli_small_pairfix_s0.json"
PRIOR = ROOT / "results" / "suite_final"

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
RUNGS = {"small": "14.4M", "mid": "33.7M"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    record = json.loads(SUITE.read_text())
    citations = json.loads(CITATIONS.read_text())
    raw_cells = record["cells"]
    cells: dict[str, dict] = {rung: {} for rung in RUNGS}

    for rung in RUNGS:
        for task, label, key in TASKS:
            cell = raw_cells[f"{rung}/{task}"]
            result_path = SUITE_DIR / rung / task / "result.json"
            if not result_path.exists():
                raise SystemExit(f"missing per-cell result: {result_path}")
            result = json.loads(result_path.read_text())
            final = result["final"]
            if not close(final["accuracy"], cell["accuracy"]):
                raise SystemExit(
                    f"{rung}/{task}: SUITE accuracy {cell['accuracy']} does not "
                    f"match result {final['accuracy']}"
                )
            if int(final["n"]) != int(cell["n"]):
                raise SystemExit(f"{rung}/{task}: test n mismatch")
            if int(cell["correct"]) != int(round(cell["accuracy"] * cell["n"])):
                raise SystemExit(f"{rung}/{task}: correct count mismatch")
            lo, hi = wilson(int(cell["correct"]), int(cell["n"]))
            if not (close(lo, cell["wilson95"][0], 1e-4)
                    and close(hi, cell["wilson95"][1], 1e-4)):
                raise SystemExit(f"{rung}/{task}: Wilson interval mismatch")
            cited = citations["rows"][task]["jev"]
            if (cited is None) != (cell["jev"] is None):
                raise SystemExit(f"{rung}/{task}: cited-point presence mismatch")
            if cited is not None and not close(100 * cited, cell["jev"]):
                raise SystemExit(f"{rung}/{task}: cited-point value mismatch")
            gap = None if cited is None else round(100 * (cell["accuracy"] - cited), 2)
            if gap is not None and not close(gap, cell["gap_vs_jev_pp"], 0.011):
                raise SystemExit(f"{rung}/{task}: gap mismatch")
            prior = cell["prior"]["suite_final"]
            cells[rung][task] = {
                "label": label,
                "key": key,
                "k": int(result["k"]),
                "n": int(cell["n"]),
                "correct": int(cell["correct"]),
                "accuracy": float(cell["accuracy"]),
                "wilson95": [float(lo), float(hi)],
                "nll": float(cell["nll"]),
                "ece": float(cell["ece"]),
                "cited": cited,
                "gap_pp": gap,
                "at_or_above": bool(cell["at_or_above_jev"]),
                "clear_above": bool(cell["clear_above_jev"]),
                "tie": bool(cell["tie_with_jev"]),
                "path": cell["path"],
                "path_changed": bool(cell["path_changed"]),
                "score_changed": bool(cell["score_changed"]),
                "prior": {
                    "accuracy": float(prior["accuracy"]),
                    "path": prior["path"],
                    "source": prior["source"],
                },
                "delta_vs_prior_pp": float(cell["delta_vs_suite_pp"]),
                "result": str(result_path.relative_to(ROOT)),
                "result_sha256": sha256(result_path),
            }

    cited_tasks = [t for t, _, _ in TASKS if cells["small"][t]["cited"] is not None]
    summary = record["summary"]
    recomputed = {}
    for rung in RUNGS:
        recomputed[rung] = {
            "cells": len(TASKS),
            "with_cited_jev": len(cited_tasks),
            "at_or_above_jev": sum(cells[rung][t]["at_or_above"] for t in cited_tasks),
            "clear_above_jev": sum(cells[rung][t]["clear_above"] for t in cited_tasks),
            "tie_with_jev": sum(cells[rung][t]["tie"] for t in cited_tasks),
            "below_jev": sum(
                not cells[rung][t]["at_or_above"] for t in cited_tasks
            ),
            "no_cited_jev": len(TASKS) - len(cited_tasks),
        }
        if recomputed[rung] != summary[rung]:
            raise SystemExit(
                f"summary mismatch at {rung}: computed {recomputed[rung]} "
                f"vs record {summary[rung]}"
            )

    larger_worse = sum(
        cells["mid"][t]["accuracy"] < cells["small"][t]["accuracy"]
        for t, _, _ in TASKS
    )
    mid_minus_small = {
        t: round(100 * (cells["mid"][t]["accuracy"] - cells["small"][t]["accuracy"]), 2)
        for t, _, _ in TASKS
    }
    changed = sorted(record["changed_by_path_cells"])
    unchanged = sorted(record["unchanged_cells"])
    already_dual = {
        rung: [t for t, _, _ in TASKS if not cells[rung][t]["path_changed"]]
        for rung in RUNGS
    }
    exact_controls = [
        t for t, _, _ in TASKS
        if not cells["small"][t]["path_changed"] and not cells["small"][t]["score_changed"]
    ]

    if len(changed) != sum(
        cells[r][t]["path_changed"] for r in RUNGS for t, _, _ in TASKS
    ):
        raise SystemExit("changed-by-path cell list does not match cell flags")
    if len(unchanged) != 3:
        raise SystemExit(f"expected three exact controls, record has {len(unchanged)}")
    if set(exact_controls) != {x.split("/", 1)[1] for x in unchanged}:
        raise SystemExit("exact-control list does not match record")

    # ------------------------------------------------------------- table
    rows = []
    for task, label, key in TASKS:
        s = cells["small"][task]
        m = cells["mid"][task]
        cited = "---" if s["cited"] is None else f"{100 * s['cited']:.2f}"
        if task == "massive_intent":
            cited += r"\textsuperscript{\(\ast\)}"
        small = f"{100 * s['accuracy']:.2f}"
        mid = f"{100 * m['accuracy']:.2f}"
        if s["tie"]:
            small += r"\textsuperscript{\(\dagger\)}"
        if m["tie"]:
            mid += r"\textsuperscript{\(\dagger\)}"
        if task in {"emotion", "xnli"}:
            small += r"\textsuperscript{\(\ddagger\)}"
            mid += r"\textsuperscript{\(\ddagger\)}"

        def gap_tex(cell: dict) -> str:
            if cell["gap_pp"] is None:
                return "---"
            val = f"${cell['gap_pp']:+.2f}$"
            return r"\textbf{" + val + "}" if cell["clear_above"] else val

        rows.append(
            rf"{label} ($K={s['k']}$) & {cited} & {small} & {gap_tex(s)} & "
            rf"{mid} & {gap_tex(m)} \\"
        )

    table = rf"""\begin{{table}}[htbp]
\centering
\footnotesize
\caption{{{len(TASKS)} public tasks at the pre-registered final recipe: LR $10^{{-4}}$,
exactly 3{{,}}000 optimizer steps, batch 32, 4{{,}}000 stratified training rows and
2{{,}}000 test rows where available (Financial PhraseBank has 1{{,}}264 train and
1{{,}}000 test), seed 0, encoder unfrozen. The recipe is identical for every cell;
there is no per-task learning rate, step count, row cap or prompt. Every Azurite
cell is measured on the dual context-candidate path. Cited cells are published
or third-party figures, not reproduced. {len(changed)} of the {len(TASKS) * len(RUNGS)}
cells changed evaluation path in this pass: AG News, DAIR Emotion, XNLI and
Financial PhraseBank at both rungs; their former candidate-only cross-encoder
values are retired and are not capability measurements. The 14.4M controls
Banking77, CLINC150 and MASSIVE intent reproduce their prior predictions
exactly; the other already-dual cells move only by run-to-run
variation. MASSIVE intent's cited point is Laya's card, not a Jev number; the
other cited points are Jev. The larger rung is worse on {larger_worse} of
{len(TASKS)} cells. Single seed per cell;
95\% Wilson intervals are in the text. These are cross-harness comparisons, not
a head-to-head.}}
\label{{tab:suite}}
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Task & Cited & \shortstack{{14.4M\\accuracy}} & \shortstack{{$\Delta$ vs\\cited}} &
\shortstack{{33.7M\\accuracy}} & \shortstack{{$\Delta$ vs\\cited}} \\
\midrule
""" + "\n".join(rows) + rf"""
\bottomrule
\end{{tabular}}
\par\smallskip
\footnotesize
\textsuperscript{{\(\ddagger\)}}Corrected dual-path cell; the text identifies
its retired cross-path value.
\quad
\textsuperscript{{\(\dagger\)}}95\% Wilson interval includes the cited point;
not a clear win or loss.
\quad
\textsuperscript{{\(\ast\)}}Laya card ({100 * cells['small']['massive_intent']['cited']:.2f}),
not a Jev number.
\quad
Clear wins are bold; positive $\Delta$ alone is not a win. No interval is shown
here.
\end{{table}}
"""
    (OUT / "suite.tex").write_text(table)

    # ------------------------------------------------------------- macros
    def pct_(x: float) -> str:
        return f"{100 * x:.2f}"

    def gap_(x: float | None) -> str:
        return "---" if x is None else f"{x:+.2f}"

    diag_all = json.loads(DIAGNOSTICS.read_text())
    old_xnli = diag_all["xnli_small_old"]
    fixed_xnli = diag_all["xnli_small_fixed"]
    emotion = json.loads(EMOTION.read_text())["tasks"]["small"]
    xnli_field = json.loads(XNLI_FIELD.read_text())["final"]

    m = ["% generated by thesis/gen_suite.py from results/suite_dual/SUITE.json — do not edit by hand"]
    add = lambda name, value: m.append(rf"\newcommand{{\{name}}}{{{value}}}")

    add("SuiteTasks", len(TASKS))
    add("SuiteComparatorCells", len(cited_tasks))
    add("SuiteSmallAtOrAbove", recomputed["small"]["at_or_above_jev"])
    add("SuiteMidAtOrAbove", recomputed["mid"]["at_or_above_jev"])
    add("SuiteSmallClear", recomputed["small"]["clear_above_jev"])
    add("SuiteMidClear", recomputed["mid"]["clear_above_jev"])
    add("SuiteSmallTie", recomputed["small"]["tie_with_jev"])
    add("SuiteMidTie", recomputed["mid"]["tie_with_jev"])
    add("SuiteSmallBelow", recomputed["small"]["below_jev"])
    add("SuiteMidBelow", recomputed["mid"]["below_jev"])
    add("SuiteLargerWorse", larger_worse)
    add("SuiteChangedCells", len(changed))
    add("SuiteSmallCrossChanged", sum(cells["small"][t]["path_changed"] for t, _, _ in TASKS))
    add("SuiteMidCrossChanged", sum(cells["mid"][t]["path_changed"] for t, _, _ in TASKS))
    add("SuiteSmallAlreadyDual", len(already_dual["small"]))
    add("SuiteMidAlreadyDual", len(already_dual["mid"]))
    add("SuiteSmallExactControls", len(exact_controls))
    add("SuiteMidMinusSmallAg", f"{mid_minus_small['ag_news']:+.2f}")

    for task, _, key in TASKS:
        s = cells["small"][task]
        mid = cells["mid"][task]
        add(f"SuiteSmall{key}", pct_(s["accuracy"]))
        add(f"SuiteMid{key}", pct_(mid["accuracy"]))
        add(f"SuiteCited{key}", "---" if s["cited"] is None else f"{100 * s['cited']:.2f}")
        add(f"SuiteSmallGap{key}", gap_(s["gap_pp"]))
        add(f"SuiteMidGap{key}", gap_(mid["gap_pp"]))
        add(f"SuiteSmallPrior{key}", pct_(s["prior"]["accuracy"]))
        add(f"SuiteMidPrior{key}", pct_(mid["prior"]["accuracy"]))
        add(f"SuiteSmallDelta{key}", f"{s['delta_vs_prior_pp']:+.2f}")
        add(f"SuiteMidDelta{key}", f"{mid['delta_vs_prior_pp']:+.2f}")
        add(f"SuiteSmall{key}Lo", f"{100 * s['wilson95'][0]:.2f}")
        add(f"SuiteSmall{key}Hi", f"{100 * s['wilson95'][1]:.2f}")
        add(f"SuiteMid{key}Lo", f"{100 * mid['wilson95'][0]:.2f}")
        add(f"SuiteMid{key}Hi", f"{100 * mid['wilson95'][1]:.2f}")

    add("SuiteEmotionMajority", f"{100 * emotion['majority_baseline']:.2f}")
    add("SuiteEmotionUniform", f"{100 * emotion['uniform_chance']:.2f}")
    add("SuiteSmallEmotionNll", f"{cells['small']['emotion']['nll']:.4f}")
    add("SuiteMidEmotionNll", f"{cells['mid']['emotion']['nll']:.4f}")
    add("SuiteSmallEmotionEce", f"{cells['small']['emotion']['ece']:.4f}")
    add("SuiteMidEmotionEce", f"{cells['mid']['emotion']['ece']:.4f}")
    add("SuiteEmotionRungDrop", f"{100 * (cells['small']['emotion']['accuracy'] - cells['mid']['emotion']['accuracy']):.2f}")

    add("SuiteSmallXnliRaw", pct_(cells["small"]["xnli"]["prior"]["accuracy"]))
    add("SuiteMidXnliRaw", pct_(cells["mid"]["xnli"]["prior"]["accuracy"]))
    add("SuiteSmallXnliGain", f"{cells['small']['xnli']['delta_vs_prior_pp']:+.2f}")
    add("SuiteMidXnliGain", f"{cells['mid']['xnli']['delta_vs_prior_pp']:+.2f}")
    add("SuiteSmallXnliShortfall", f"{abs(cells['small']['xnli']['gap_pp']):.2f}")
    add("SuiteMidXnliShortfall", f"{abs(cells['mid']['xnli']['gap_pp']):.2f}")
    add("SuiteXnliChance", f"{100 / cells['small']['xnli']['k']:.2f}")
    add("SuiteSmallXnliPriorFix", pct_(xnli_field["accuracy"]))
    add("SuiteXnliAbsentTest", old_xnli["hypothesis_preservation_test"]["absent"])
    add("SuiteXnliAbsentTrain", old_xnli["hypothesis_preservation_train"]["absent"])
    add("SuiteXnliFullFixed", fixed_xnli["hypothesis_preservation_test"]["full"])
    add("SuiteXnliTestN", old_xnli["hypothesis_preservation_test"]["n"])
    add("SuiteXnliTrainN", old_xnli["hypothesis_preservation_train"]["n"])
    (OUT / "macros_suite.tex").write_text("\n".join(m) + "\n")

    # ------------------------------------------------------------- source
    source = {
        "record": str(SUITE.relative_to(ROOT)),
        "record_sha256": sha256(SUITE),
        "recipe": record["protocol"],
        "checkpoints": json.loads(CHECKPOINTS.read_text())["source"],
        "cells": cells,
        "counts": {
            "tasks": len(TASKS),
            "comparator_cells": len(cited_tasks),
            "small": recomputed["small"],
            "mid": recomputed["mid"],
            "larger_rung_worse": larger_worse,
            "mid_minus_small_pp": mid_minus_small,
            "changed_by_path_cells": changed,
            "unchanged_cells": unchanged,
            "already_dual": already_dual,
            "exact_14_4m_controls": exact_controls,
        },
        "retired": {
            "note": "candidate-only cross-encoder values superseded by the dual path; retained only for audit",
            "small": {
                t: cells["small"][t]["prior"]
                for t, _, _ in TASKS if cells["small"][t]["path_changed"]
            },
            "mid": {
                t: cells["mid"][t]["prior"]
                for t, _, _ in TASKS if cells["mid"][t]["path_changed"]
            },
            "xnli_field_aware_cross": {
                "accuracy": float(xnli_field["accuracy"]),
                "source": str(XNLI_FIELD.relative_to(ROOT)),
                "sha256": sha256(XNLI_FIELD),
            },
        },
        "citations": str(CITATIONS.relative_to(ROOT)),
    }
    (OUT / "suite_source.json").write_text(json.dumps(source, indent=2) + "\n")

    print(
        f"suite rows: {len(TASKS)} tasks x {len(RUNGS)} rungs; "
        f"comparators {len(cited_tasks)}; "
        f"at-or-above small={recomputed['small']['at_or_above_jev']}, "
        f"mid={recomputed['mid']['at_or_above_jev']}; larger-worse={larger_worse}"
    )
    for task, _, _ in TASKS:
        s = cells["small"][task]
        mid = cells["mid"][task]
        cited = "---" if s["cited"] is None else f"{100 * s['cited']:.2f}"
        print(
            f"  {task:22s} small={100 * s['accuracy']:6.2f} "
            f"mid={100 * mid['accuracy']:6.2f} cited={cited} "
            f"path_changed={s['path_changed']}/{mid['path_changed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
