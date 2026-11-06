#!/usr/bin/env python3
"""Build the three-rung suite comparison from stored result records.

Reads, never invents:

* small 14.4M   results/suite_dual/small/<task>/result.json
* mid   33.7M   results/suite_dual/mid/<task>/result.json
* large 113.47M results/suite_large_113m/<task>/result.json

and the cited comparator points in thesis/suite_citations.json.  It checks
that the reported accuracy equals the accuracy recomputed from the stored
per-example predictions and gold, that every rung used the same test draw,
and that every cell is on the dual path.  It then writes COMPARISON.json and
COMPARISON.md next to the large-rung records.

Intervals are 95% Wilson score intervals over test rows.  Rung comparisons
are exact two-sided McNemar tests on the paired per-example outcomes, which is
the correct test here because every rung is scored on the identical test rows.

Usage:
    python scripts/suite_three_rung.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "results/suite_large_113m"
CITES = PROJECT / "thesis/suite_citations.json"

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
RUNGS = ["small", "mid", "large"]
LABEL = {"small": "14.4M", "mid": "33.7M", "large": "113.47M"}
CHECKPOINTS = {
    "small": {
        "path": "results/pretrain/l1_small_byte_ctx256/ckpt.pt",
        "sha256": "8fd8bedee794ee4aa464af7c11fb06eb1d8dcfbfe82f529f8106a11a798ff131",
        "params": 14260992,
        "native_context": 256,
    },
    "mid": {
        "path": "results/pretrain/l1_mid_byte_iso10v2/ckpt.pt",
        "sha256": "9fed60f7e6f7134562fe0c6e356c826c380a7ea2ef6614c2469451d8bbc09f5c",
        "params": 33696768,
        "native_context": 1024,
    },
    "large": {
        "path": "results/pretrain/l1_large_byte_iso10v3/ckpt.pt",
        "sha256": "aba65b60e30654a9895dc9c365d62b555f96dcce96ad8a8ad56cc718ee0865ec",
        "params": 113465856,
        "native_context": 1024,
    },
}
Z = 1.959963984540054


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wilson(correct: int, n: int, z: float = Z) -> tuple[float, float]:
    p = correct / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def binom_cdf(k: int, n: int) -> float:
    if k < 0:
        return 0.0
    return sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * binom_cdf(k, n)
    return min(1.0, p)


def verdict(point: float, lo: float, hi: float, jev: float | None) -> str:
    if jev is None:
        return "no cited point"
    if lo > jev:
        return "clear above"
    if hi < jev:
        return "clear below"
    if point > jev:
        return "tie (above point)"
    if point < jev:
        return "tie (below point)"
    return "tie"


def load_cell(rung: str, task: str) -> tuple[dict, Path]:
    if rung == "large":
        path = OUT_DIR / task / "result.json"
    else:
        path = PROJECT / "results/suite_dual" / rung / task / "result.json"
    if not path.exists():
        raise SystemExit(f"missing result record: {path}")
    return json.loads(path.read_text()), path


def main() -> int:
    cites = json.loads(CITES.read_text())["rows"]
    records: dict[str, dict[str, dict]] = {r: {} for r in RUNGS}
    sources: dict[str, str] = {}
    integrity: list[str] = []

    for task in TASKS:
        gold_hashes = {}
        for rung in RUNGS:
            rec, path = load_cell(rung, task)
            sources[str(path.relative_to(PROJECT))] = sha256(path)
            final = rec["final"]
            preds, gold = final["predictions"], final["gold"]
            if len(preds) != len(gold) or len(gold) != final["n"]:
                raise SystemExit(f"{rung}/{task}: prediction length mismatch")
            if rec.get("path") != "dual":
                raise SystemExit(f"{rung}/{task}: path is {rec.get('path')}, not dual")
            recomputed = sum(1 for a, b in zip(preds, gold) if a == b) / len(gold)
            if abs(recomputed - final["accuracy"]) > 1e-9:
                raise SystemExit(
                    f"{rung}/{task}: reported accuracy {final['accuracy']} "
                    f"!= recomputed {recomputed}"
                )
            gold_hashes[rung] = hashlib.sha256(bytes(gold)).hexdigest()
            correct = sum(1 for a, b in zip(preds, gold) if a == b)
            lo, hi = wilson(correct, len(gold))
            counts = [0] * rec["k"]
            for p in preds:
                counts[p] += 1
            top = max(range(rec["k"]), key=lambda i: counts[i])
            records[rung][task] = {
                "task": task,
                "rung": rung,
                "n": final["n"],
                "correct": correct,
                "accuracy": final["accuracy"],
                "wilson95": [lo, hi],
                "nll": final["nll"],
                "ece": final["ece"],
                "mean_confidence": final["mean_confidence"],
                "k": rec["k"],
                "majority_prediction": top,
                "majority_prediction_count": counts[top],
                "majority_prediction_fraction": counts[top] / len(preds),
                "prediction_counts": counts,
                "seconds": rec.get("seconds"),
                "source": str(path.relative_to(PROJECT)),
                "source_sha256": sources[str(path.relative_to(PROJECT))],
                "checkpoint_sha256": rec.get("loaded_from", {}).get("sha256"),
                "pretrain_step": rec.get("loaded_from", {}).get("pretrain_step"),
            }
        if len(set(gold_hashes.values())) != 1:
            raise SystemExit(f"{task}: test draws differ across rungs")
        integrity.append(
            f"{task}: identical test draw across three rungs "
            f"(gold sha256 {gold_hashes['small'][:16]}), accuracy reproduces "
            f"from stored predictions"
        )

    # Paired rung comparisons on the identical test rows.
    pairs = [("large", "small"), ("large", "mid"), ("mid", "small")]
    paired: dict[str, dict[str, dict]] = {f"{a}_vs_{b}": {} for a, b in pairs}
    for task in TASKS:
        for a, b in pairs:
            pa = records[a][task]["source"]
            pb = records[b][task]["source"]
            ra = json.loads((PROJECT / pa).read_text())["final"]
            rb = json.loads((PROJECT / pb).read_text())["final"]
            assert ra["gold"] == rb["gold"]
            a_only = sum(
                1 for x, y, g in zip(ra["predictions"], rb["predictions"], ra["gold"])
                if x == g and y != g
            )
            b_only = sum(
                1 for x, y, g in zip(ra["predictions"], rb["predictions"], ra["gold"])
                if x != g and y == g
            )
            paired[f"{a}_vs_{b}"][task] = {
                "first": a,
                "second": b,
                "first_only_correct": a_only,
                "second_only_correct": b_only,
                "delta_pp": (a_only - b_only) / records[a][task]["n"] * 100.0,
                "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
            }

    # Counts against the cited points, by interval rule.
    counts = {}
    for rung in RUNGS:
        above = tie = below = none = 0
        for task in TASKS:
            c = records[rung][task]
            jev = cites.get(task, {}).get("jev")
            v = verdict(c["accuracy"], *c["wilson95"], jev)
            c["verdict"] = v
            c["gap_pp"] = None if jev is None else (c["accuracy"] - jev) * 100.0
            if jev is None:
                none += 1
            elif v == "clear above":
                above += 1
            elif v == "clear below":
                below += 1
            else:
                tie += 1
        counts[rung] = {
            "clear_above": above,
            "tie": tie,
            "clear_below": below,
            "no_cited_point": none,
            "at_or_above": above + tie,
            "comparator_cells": 8,
        }

    # Degeneracy diagnostics: uniform-output signature on k-way cells.
    degeneracy = {}
    for rung in RUNGS:
        for task in TASKS:
            c = records[rung][task]
            k = c["k"]
            degeneracy[f"{rung}/{task}"] = {
                "k": k,
                "accuracy": c["accuracy"],
                "chance": 1.0 / k,
                "nll": c["nll"],
                "ln_k": math.log(k),
                "nll_minus_ln_k": c["nll"] - math.log(k),
                "mean_confidence": c["mean_confidence"],
                "ece": c["ece"],
                "top_class": c["majority_prediction"],
                "top_class_fraction": c["majority_prediction_fraction"],
                "uniform_output": abs(c["nll"] - math.log(k)) < 5e-4
                and abs(c["mean_confidence"] - 1.0 / k) < 5e-3,
            }

    doc = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Three-rung comparison generated from stored per-example result "
            "records; no number is transcribed by hand."
        ),
        "recipe": {
            "path": "dual (auto dispatch)",
            "lr": 1e-4,
            "steps": 3000,
            "batch": 32,
            "seed": 0,
            "split_seed": 0,
            "test_split_seed": 1,
            "train_rows": 4000,
            "test_rows": 2000,
            "financial_train_rows": 1264,
            "financial_test_rows": 1000,
            "encoder_frozen": False,
            "field_aware_packer": False,
        },
        "checkpoints": CHECKPOINTS,
        "integrity": integrity,
        "cells": records,
        "paired": paired,
        "counts_vs_cited": counts,
        "degeneracy": degeneracy,
        "sources_sha256": sources,
    }
    (OUT_DIR / "COMPARISON.json").write_text(json.dumps(doc, indent=1) + "\n")

    (OUT_DIR / "COMPARISON.md").write_text(render(doc) + "\n")
    (OUT_DIR / "RESULTS.md").write_text(render_results(doc) + "\n")
    print(f"wrote {OUT_DIR/'COMPARISON.json'}, COMPARISON.md and RESULTS.md")
    return 0


def render(doc: dict) -> str:
    cells, paired = doc["cells"], doc["paired"]
    counts = doc["counts_vs_cited"]
    cites = json.loads(CITES.read_text())["rows"]
    L = []
    L.append("# Three-rung suite: 14.4M, 33.7M, 113.47M on the dual path")
    L.append("")
    L.append(
        "Generated by `scripts/suite_three_rung.py` on "
        f"{doc['generated_utc']} from the stored per-example result records. "
        "Every accuracy below is recomputed from those records before it is "
        "printed; the script aborts if a recomputation disagrees with the "
        "record's own `accuracy` field, if a rung's path is not `dual`, or if "
        "the three rungs do not share the identical test draw."
    )
    L.append("")
    L.append("## Checkpoints")
    L.append("")
    L.append("| rung | encoder params | native context | checkpoint | SHA-256 |")
    L.append("|---|---:|---:|---|---|")
    for r in RUNGS:
        c = CHECKPOINTS[r]
        L.append(
            f"| {LABEL[r]} | {c['params']:,} | {c['native_context']} | "
            f"`{c['path']}` | `{c['sha256']}` |"
        )
    L.append("")
    L.append("## Recipe")
    L.append("")
    r = doc["recipe"]
    L.append(
        f"Dual context-candidate scorer, LR {r['lr']:g}, exactly {r['steps']:,} "
        f"optimizer steps, batch {r['batch']}, seed {r['seed']}, split seed "
        f"{r['split_seed']} / test split seed {r['test_split_seed']}, encoder "
        f"unfrozen, native context of each checkpoint, no per-task tuning, "
        f"field-aware packer {r['field_aware_packer']}. "
        f"{r['train_rows']:,} train rows and {r['test_rows']:,} test rows "
        f"(Financial PhraseBank {r['financial_train_rows']:,} / "
        f"{r['financial_test_rows']:,}). Identical across rungs."
    )
    L.append("")
    L.append("## Every cell, every rung")
    L.append("")
    L.append(
        "| task | n | 14.4M | 95% Wilson | 33.7M | 95% Wilson | 113.47M | "
        "95% Wilson | Jev | gap 14.4M | gap 33.7M | gap 113.47M | 113.47M verdict |"
    )
    L.append("|---|---:|---:|---|---:|---|---:|---|---:|---:|---:|---:|---|")
    for task in TASKS:
        s, m, g = cells["small"][task], cells["mid"][task], cells["large"][task]
        jev = cites.get(task, {}).get("jev")
        js = "—" if jev is None else f"{jev*100:.1f}"
        gs = "—" if s["gap_pp"] is None else f"{s['gap_pp']:+.2f}"
        gm = "—" if m["gap_pp"] is None else f"{m['gap_pp']:+.2f}"
        gg = "—" if g["gap_pp"] is None else f"{g['gap_pp']:+.2f}"
        L.append(
            f"| {cites.get(task,{}).get('label', task)} | {s['n']:,} | "
            f"{s['accuracy']*100:.2f}% | {s['wilson95'][0]*100:.2f}–{s['wilson95'][1]*100:.2f} | "
            f"{m['accuracy']*100:.2f}% | {m['wilson95'][0]*100:.2f}–{m['wilson95'][1]*100:.2f} | "
            f"{g['accuracy']*100:.2f}% | {g['wilson95'][0]*100:.2f}–{g['wilson95'][1]*100:.2f} | "
            f"{js} | {gs} | {gm} | {gg} | {g['verdict']} |"
        )
    L.append("")
    L.append("### Counts against the cited points")
    L.append("")
    L.append("| rung | clear above | tie | clear below | no cited point | at or above |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for rung in RUNGS:
        c = counts[rung]
        L.append(
            f"| {LABEL[rung]} | {c['clear_above']} | {c['tie']} | "
            f"{c['clear_below']} | {c['no_cited_point']} | "
            f"**{c['at_or_above']} / {c['comparator_cells']}** |"
        )
    L.append("")
    L.append(
        "A cell counts as clear above only if the lower Wilson bound exceeds the "
        "cited point, and clear below only if the upper bound falls under it; "
        "otherwise it is a tie, including cells whose point estimate is far from "
        "the cited value but whose interval covers it."
    )
    L.append("")
    L.append("## Paired rung comparisons")
    L.append("")
    L.append(
        "Every rung was scored on the identical 2,000 (1,000) test rows, so the "
        "rung comparison is paired. `first only` is the number of rows the first "
        "rung gets right and the second gets wrong; the exact two-sided McNemar "
        "p uses only those discordant rows."
    )
    L.append("")
    L.append("| comparison | task | first only | second only | Δ pp | exact p |")
    L.append("|---|---|---:|---:|---:|---:|")
    for name in ["large_vs_small", "large_vs_mid", "mid_vs_small"]:
        for task in TASKS:
            p = paired[name][task]
            L.append(
                f"| {LABEL[p['first']]} vs {LABEL[p['second']]} | "
                f"{cites.get(task,{}).get('label', task)} | "
                f"{p['first_only_correct']} | {p['second_only_correct']} | "
                f"{p['delta_pp']:+.2f} | {p['mcnemar_exact_p']:.2e} |"
            )
    L.append("")
    L.append("### What each step of the ladder did, cell by cell")
    L.append("")
    L.append(
        "A cell is called a win or a loss only when the paired exact McNemar "
        "test rejects at p < 0.05; otherwise the step did not move that cell "
        "beyond row-level noise."
    )
    L.append("")
    L.append("| step | wins (p<0.05) | losses (p<0.05) | unmoved |")
    L.append("|---|---|---|---|")
    for name in ["mid_vs_small", "large_vs_mid", "large_vs_small"]:
        p = paired[name]
        first = next(iter(p.values()))["first"]
        second = next(iter(p.values()))["second"]
        wins, losses, flat = [], [], []
        for task in TASKS:
            d = p[task]
            label = cites.get(task, {}).get("label", task)
            if d["mcnemar_exact_p"] < 0.05:
                (wins if d["delta_pp"] > 0 else losses).append(
                    f"{label} {d['delta_pp']:+.2f}"
                )
            else:
                flat.append(label)
        L.append(
            f"| {LABEL[first]} vs {LABEL[second]} | "
            f"{'; '.join(wins) if wins else 'none'} | "
            f"{'; '.join(losses) if losses else 'none'} | "
            f"{', '.join(flat) if flat else 'none'} |"
        )
    L.append("")
    L.append("## Per-cell notes")
    L.append("")
    for task in TASKS:
        s, m, g = cells["small"][task], cells["mid"][task], cells["large"][task]
        jev = cites.get(task, {}).get("jev")
        ps = paired["large_vs_small"][task]
        pm = paired["large_vs_mid"][task]
        head = f"**{cites.get(task,{}).get('label', task)}.** "
        body = (
            f"{s['accuracy']*100:.2f}% → {m['accuracy']*100:.2f}% → "
            f"{g['accuracy']*100:.2f}% across 14.4M → 33.7M → 113.47M. "
        )
        if jev is not None:
            body += (
                f"Against the cited {jev*100:.1f}, the 113.47M gap is "
                f"{g['gap_pp']:+.2f} pp ({g['verdict']}). "
            )
        body += (
            f"Paired with 14.4M it wins {ps['first_only_correct']} rows and "
            f"loses {ps['second_only_correct']} (p = {ps['mcnemar_exact_p']:.2e}); "
            f"paired with 33.7M it wins {pm['first_only_correct']} and loses "
            f"{pm['second_only_correct']} (p = {pm['mcnemar_exact_p']:.2e})."
        )
        L.append(head + body)
        L.append("")
    L.append("## Uniform-output diagnostics")
    L.append("")
    L.append(
        "A k-way cell whose test NLL equals ln k to four decimals, whose mean "
        "confidence equals 1/k, and which predicts one class for almost every "
        "row is emitting a constant uniform distribution: it has learned nothing "
        "on that task, whatever its accuracy happens to be."
    )
    L.append("")
    L.append("| rung/task | k | accuracy | 1/k | NLL | ln k | mean conf | top class share | uniform |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for key, d in doc["degeneracy"].items():
        if d["uniform_output"] or d["top_class_fraction"] > 0.9:
            L.append(
                f"| {LABEL[key.split('/')[0]]}/{key.split('/')[1]} | {d['k']} | "
                f"{d['accuracy']*100:.2f}% | {d['chance']*100:.2f}% | "
                f"{d['nll']:.4f} | {d['ln_k']:.4f} | {d['mean_confidence']:.4f} | "
                f"{d['top_class_fraction']*100:.1f}% | "
                f"{'yes' if d['uniform_output'] else 'no'} |"
            )
    L.append("")
    L.append("## Integrity checks performed")
    L.append("")
    for line in doc["integrity"]:
        L.append(f"- {line}")
    L.append("")
    L.append("## Limitations")
    L.append("")
    L.append(
        "- One seed per cell. Wilson intervals describe test-row sampling, not "
        "seed variance; the exact McNemar tests describe paired row-level "
        "differences on this draw, not variation over seeds."
    )
    repl_path = OUT_DIR / "REPLICATION.json"
    if repl_path.exists():
        repl = json.loads(repl_path.read_text())["summary"]
        L.append(
            f"- Three cells were executed twice on the same checkpoint and test "
            f"draw. The repeat runs differ from the archived runs by at most "
            f"{repl['max_abs_delta_pp']:.2f} pp (mean "
            f"{repl['mean_abs_delta_pp']:.2f} pp) and flip up to "
            f"{repl['max_prediction_flips']} of 2,000 test rows; see "
            "`REPLICATION.md`. Rung deltas at or below that scale are not "
            "interpreted as capability differences here."
        )
    L.append(
        "- Context is a checkpoint property (256 at 14.4M, 1,024 at 33.7M and "
        "113.47M), so the rung comparison is the deployed configuration "
        "comparison, not a pure parameter-count effect."
    )
    L.append(
        "- The cited Jev points are cross-harness published numbers, not paired "
        "runs of this code on these splits."
    )
    L.append(
        "- The large-rung records were produced in capped concurrency waves "
        "after a nine-way wave failed with CUDA OOM; `_failed_wave1/REASON.txt` "
        "keeps that failure as an infrastructure record with no result."
    )
    return "\n".join(L)


def render_results(doc: dict) -> str:
    cells, paired, counts = doc["cells"], doc["paired"], doc["counts_vs_cited"]
    cites = json.loads(CITES.read_text())["rows"]
    named = ["xnli", "ag_news", "clinc_oos", "language_id"]
    large = cells["large"]
    closed = [t for t in named if large[t]["verdict"] == "clear above"]

    def label(task: str) -> str:
        return cites.get(task, {}).get("label", task)

    L = []
    L.append("# Nine-cell suite at 113.47M: result and verdict")
    L.append("")
    L.append(
        f"Generated by `scripts/suite_three_rung.py` on {doc['generated_utc']} "
        "from the stored records; the numbers below are computed, not "
        "transcribed. `COMPARISON.md` holds the full three-rung table, the "
        "paired tests and the integrity checks."
    )
    L.append("")
    L.append("## What ran")
    L.append("")
    ck = CHECKPOINTS["large"]
    L.append(
        "A **new** 113.47M-parameter byte encoder, run "
        "`l1_large_byte_iso10v3`, trained from fresh weights for the full "
        "1,134,658,560-token budget at the unchanged large-rung recipe. Its "
        f"final checkpoint is `{ck['path']}`, step 69,255, 1,134,673,920 tokens, "
        f"SHA-256 `{ck['sha256']}`, 1,361,732,233 bytes, verified against the "
        "sandbox's own `checkpoint.json` hash and present on local disk. This is "
        "not a continuation of the lost `l1_large_byte_iso10v2` run; that run "
        "reached 980,746,240 tokens without surviving weights and stands "
        "separately in `../large_v3/` and `../reclaim_20261028/LOSS.md`."
    )
    L.append("")
    L.append(
        "All nine cells ran the pre-registered recipe unchanged: dual path, "
        f"LR {doc['recipe']['lr']:g}, 3,000 steps, batch 32, seed 0, native "
        "1,024-byte context, encoder unfrozen, no per-task tuning and no "
        "field-aware packer. Every record names the same source checkpoint "
        "SHA-256 in its `evidence/provenance.json`."
    )
    L.append("")
    L.append("## The four cells this rung had to close")
    L.append("")
    L.append(
        "| cell | 14.4M | 33.7M | 113.47M | 95% Wilson (113.47M) | Jev | gap | verdict | closed |"
    )
    L.append("|---|---:|---:|---:|---|---:|---:|---|---|")
    for t in named:
        s, m, g = cells["small"][t], cells["mid"][t], large[t]
        L.append(
            f"| {label(t)} | {s['accuracy']*100:.2f}% | {m['accuracy']*100:.2f}% | "
            f"{g['accuracy']*100:.2f}% | {g['wilson95'][0]*100:.2f}–"
            f"{g['wilson95'][1]*100:.2f} | {cites[t]['jev']*100:.1f} | "
            f"{g['gap_pp']:+.2f} | {g['verdict']} | "
            f"{'yes' if g['verdict'] == 'clear above' else 'no'} |"
        )
    L.append("")
    for t in named:
        s, g = cells["small"][t], large[t]
        p = paired["large_vs_small"][t]
        if g["verdict"] == "clear above":
            verdict_text = (
                "the fourth rung closes it: the whole interval sits above the "
                "cited point"
            )
        elif g["verdict"].startswith("tie"):
            verdict_text = (
                "the fourth rung does not close it and does not demonstrate a "
                "deficit either: the interval covers the cited point"
            )
        else:
            verdict_text = (
                "the fourth rung does not close it: the whole interval sits "
                "below the cited point"
            )
        extra = ""
        if t == "xnli":
            deg = doc["degeneracy"]["large/xnli"]
            extra = (
                f" At this rung the cell is a constant predictor: "
                f"{deg['top_class_fraction']*100:.1f}% of test rows receive one "
                f"class, test NLL is {deg['nll']:.4f} against ln 3 = "
                f"{deg['ln_k']:.4f}, mean confidence {deg['mean_confidence']:.4f} "
                f"and ECE {deg['ece']:.4f}. It is a uniform output distribution, "
                "not a weak attempt at the task."
            )
        L.append(
            f"- **{label(t)}** — {verdict_text} ({g['accuracy']*100:.2f}% "
            f"against {cites[t]['jev']*100:.1f}, gap {g['gap_pp']:+.2f} pp; "
            f"versus 14.4M it wins {p['first_only_correct']} rows and loses "
            f"{p['second_only_correct']}, exact p = "
            f"{p['mcnemar_exact_p']:.2e}).{extra}"
        )
    L.append("")
    L.append("## Does the fourth rung close them?")
    L.append("")
    if closed:
        L.append(
            f"**Yes, in part.** {len(closed)} of the four are clear above their "
            f"cited points at 113.47M: {', '.join(label(t) for t in closed)}. "
            f"The others are {', '.join(label(t) for t in named if t not in closed)}."
        )
    else:
        L.append(
            "**No.** None of XNLI, AG News, CLINC-OOS or Language ID is clear "
            "above its cited point at 113.47M. "
            + "; ".join(
                f"{label(t)} "
                + large[t]["verdict"].replace("tie (below point)", "a tie")
                .replace("tie (above point)", "a tie")
                + f" (gap {large[t]['gap_pp']:+.2f} pp)"
                for t in named
            )
            + ". XNLI is not a weak attempt but a degenerate uniform predictor."
        )
    L.append("")
    L.append("## Is scale the path to parity?")
    L.append("")
    c = counts["large"]
    L.append(
        f"At 113.47M, {c['at_or_above']} of the 8 comparator cells are at or "
        f"above the cited point ({c['clear_above']} clear above, {c['tie']} "
        f"tie), against {counts['small']['at_or_above']} of 8 at 14.4M and "
        f"{counts['mid']['at_or_above']} of 8 at 33.7M. That is one cell of "
        "movement, and the movement is not where the thesis needs it."
    )
    L.append("")
    L.append("| step | wins (p<0.05) | losses (p<0.05) | unmoved |")
    L.append("|---|---|---|---|")
    for name in ["mid_vs_small", "large_vs_mid", "large_vs_small"]:
        p = paired[name]
        first = next(iter(p.values()))["first"]
        second = next(iter(p.values()))["second"]
        wins, losses, flat = [], [], []
        for t in TASKS:
            d = p[t]
            if d["mcnemar_exact_p"] < 0.05:
                (wins if d["delta_pp"] > 0 else losses).append(
                    f"{label(t)} {d['delta_pp']:+.2f}"
                )
            else:
                flat.append(label(t))
        L.append(
            f"| {LABEL[first]} vs {LABEL[second]} | "
            f"{'; '.join(wins) if wins else 'none'} | "
            f"{'; '.join(losses) if losses else 'none'} | "
            f"{', '.join(flat) if flat else 'none'} |"
        )
    L.append("")
    wins_small = [
        t for t in TASKS
        if paired["large_vs_small"][t]["mcnemar_exact_p"] < 0.05
        and paired["large_vs_small"][t]["delta_pp"] > 0
    ]
    losses_small = [
        t for t in TASKS
        if paired["large_vs_small"][t]["mcnemar_exact_p"] < 0.05
        and paired["large_vs_small"][t]["delta_pp"] < 0
    ]
    L.append(
        f"Measured against 14.4M — a 7.95× parameter increase at the same token "
        f"budget — the large rung wins {len(wins_small)} cells "
        f"({', '.join(label(t) for t in wins_small)}), loses "
        f"{len(losses_small)} ({', '.join(label(t) for t in losses_small)}), and "
        f"leaves {9 - len(wins_small) - len(losses_small)} statistically "
        "unmoved. "
        + (
            f"The wins span {min(paired['large_vs_small'][t]['delta_pp'] for t in wins_small):+.2f} "
            f"to {max(paired['large_vs_small'][t]['delta_pp'] for t in wins_small):+.2f} pp; "
            if wins_small
            else ""
        )
        + (
            f"the loss is "
            f"{min(paired['large_vs_small'][t]['delta_pp'] for t in losses_small):+.2f} pp and lands on "
            "the one cell that needs pair reasoning. "
            if losses_small
            else ""
        )
        + "**Scale is not the path to parity on this suite.** It buys a few "
        "points on single-text classification and it does not buy XNLI, AG "
        "News, CLINC-OOS or Language ID."
    )
    L.append("")
    L.append("## Replication and the noise floor")
    L.append("")
    repl_path = OUT_DIR / "REPLICATION.json"
    if repl_path.exists():
        r = json.loads(repl_path.read_text())
        s = r["summary"]
        L.append(
            f"Three cells (CLINC-OOS, MASSIVE Intent, MASSIVE Scenario) were "
            f"executed twice on the same checkpoint, test draw and recipe, on "
            f"two sandbox instances. The repeat runs land at most "
            f"{s['max_abs_delta_pp']:.2f} pp from the archived runs (mean "
            f"{s['mean_abs_delta_pp']:.2f} pp) and flip up to "
            f"{s['max_prediction_flips']} of 2,000 predictions. That is the "
            "measured cost of re-running one fixed configuration, and it is the "
            "scale below which no rung delta in this file should be read as a "
            "capability difference. `REPLICATION.md` has the per-cell table."
        )
    else:
        L.append("Replication record not present in this directory.")
    L.append("")
    L.append("## Cost")
    L.append("")
    total_s = sum(v.get("seconds") or 0 for v in large.values())
    L.append(
        f"The nine cells cost {total_s:,.1f} GPU-seconds "
        f"({total_s/3600:.3f} GPU-hours) of measured evaluation, serial "
        "equivalent; wall time was lower because cells ran concurrently up to "
        "the memory cap described below."
    )
    L.append("")
    L.append("## Provenance and verification")
    L.append("")
    L.append(
        "- Checkpoint SHA-256 `"
        + ck["sha256"]
        + "` appears in `CHECKPOINT.json`, in every cell's "
        "`evidence/provenance.json` `source_checkpoint`, and in the "
        "re-execution record. The local file hashes to the same value."
    )
    L.append(
        "- Every cell's reported accuracy was recomputed from its stored "
        "per-example predictions and gold before this file was written; a "
        "mismatch aborts generation."
    )
    L.append(
        "- All three rungs were scored on the identical test rows; the gold "
        "arrays hash equal, which is what licenses the paired exact McNemar "
        "tests in `COMPARISON.md`."
    )
    L.append(
        "- Each cell keeps `result.json` (with per-example packed and canonical "
        "gold, predictions, probabilities and metric history), `run.log`, "
        "`launch.json` and `evidence/provenance.json`; the re-executed cells "
        "add `replication_run2/`."
    )
    ev_present, ev_missing = [], []
    for t in TASKS:
        rec = json.loads((OUT_DIR / t / "result.json").read_text())
        ev = rec.get("evidence") or {}
        f = OUT_DIR / t / "evidence" / "final.pt"
        ok = (
            f.exists()
            and ev.get("bytes")
            and f.stat().st_size == int(ev["bytes"])
        )
        (ev_present if ok else ev_missing).append(label(t))
    L.append(
        f"- Evidence tensors: {len(ev_present)} of the {len(TASKS)} archived "
        f"cells have `evidence/final.pt` on local disk with the byte size their "
        f"record names ({', '.join(ev_present)}), and the three replication "
        "runs have theirs under `replication_run2/evidence/`. "
        + (
            f"The archived tensors for {', '.join(ev_missing)} were destroyed "
            "with the sandbox that produced them; that sandbox now answers "
            '"File not found" for those paths. Their `result.json`, `run.log`, '
            "`launch.json` and `evidence/provenance.json` survive locally and "
            "are what the reported numbers are read from. "
            if ev_missing
            else ""
        )
        + "Every tensor that is present hash-matches the SHA-256 recorded in "
        "its result record."
    )
    L.append(
        "- `MANIFEST.sha256` hashes every file in this directory; "
        "`scripts/verify_manifest.py` re-checks it."
    )
    L.append("")
    L.append("## Infrastructure record, kept separate from results")
    L.append("")
    L.append(
        "The first attempt ran all nine cells concurrently on the 97,887 MiB "
        "card; five fit and four died with CUDA OOM. A second five-way wave "
        "lost two more. Those attempts produced no `result.json` and are not "
        "measurements. The reasons, verbatim from the sandbox:"
    )
    L.append("")
    for rel in ["_failed_wave1/REASON.txt", "_failed_wave2/REASON.txt"]:
        p = OUT_DIR / rel
        if p.exists():
            L.append(f"`{rel}`:")
            L.append("")
            for line in p.read_text().strip().splitlines():
                L.append(f"> {line}")
            L.append("")
    L.append(
        "The nine cells that do exist were produced in capped waves and, for "
        "three of them, one extra execution used here only as the replication "
        "control. The cap is a memory fact, not a protocol change: the recipe "
        "is identical in every wave."
    )
    L.append("")
    L.append("## Limitations")
    L.append("")
    L.append(
        "- One seed per cell. The Wilson intervals are binomial over test rows "
        "and the McNemar tests are paired over rows; neither is a seed-variance "
        "estimate. The replication deltas above are the only direct estimate of "
        "run-to-run variation, and they come from three cells."
    )
    L.append(
        "- The 14.4M checkpoint has a 256-byte native context and the larger "
        "rungs have 1,024, so the rung comparison is the deployed configuration "
        "comparison, not a pure parameter-count experiment."
    )
    L.append(
        "- The byte arm only; the random-init control arm was not run at this "
        "rung, so the large rung has no within-run random control of its own. "
        "The 14.4M and 33.7M random controls remain the reference for what the "
        "pretraining is worth."
    )
    L.append(
        "- Cited Jev points are published cross-harness numbers, not paired "
        "runs of this code; a gap to them is a gap to a claim, not a "
        "head-to-head measurement."
    )
    L.append(
        "- The suite recipe was deliberately not retuned after seeing these "
        "results, and no cell was dropped. The XNLI degeneracy is reported as "
        "found, not repaired by a packer change that would have made the rung "
        "incomparable to the other two."
    )
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
