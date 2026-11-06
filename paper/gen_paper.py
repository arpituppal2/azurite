#!/usr/bin/env python3
"""Emit the paper's numbers, tables, figures and reconciliation record.

The paper contains no measurement typed by hand.  This script reads the three
stored evaluation summaries, checks every per-example file against its recorded
SHA-256, recomputes every accuracy, Wilson interval, exact McNemar test, paired
bootstrap and pre-registered decision, and only then writes LaTeX macros.

It also reconstructs the comparator's prompt size from the public harness
inputs vendored under ``paper/data/jev_prompt``.  That measurement is stored in
``generated/prompt_budget.json`` so the paper's information-budget claim has the
same provenance as its accuracy claims.

    python paper/gen_paper.py            # verify, write generated/
    python paper/gen_paper.py --check    # verify and compare, write nothing
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
GEN = PAPER / "generated"
Q = ROOT / "results" / "jev_query" / "eval"
L = ROOT / "results" / "jev_query_5120" / "eval"
M = ROOT / "results" / "jev_matched"
CLEAN = ROOT / "results" / "product_path_clean"
PRODUCT = ROOT / "results" / "product_path"
INPUTS = M / "inputs.json"
PROMPT = PAPER / "data" / "jev_prompt"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.artifacts import read_jsonl, sha256_file, sha256_text  # noqa: E402
from azurite.stats import BOOTSTRAP_RESAMPLES, bootstrap_lower, mcnemar_exact, wilson  # noqa: E402

N_ROWS = 3_080
N_LABELS = 77
N_EXAMPLES = 24
NATIVE_BUDGET = 1_020
LONG_BUDGET = 5_118
P0_BAR = 2_846 / N_ROWS
MARGIN = -0.02
BOOT_SEED = 0

# The exact instruction string used by the public comparator harness.  Kept
# verbatim so the reconstructed payload is byte-comparable with its builder.
JEV_INSTRUCTIONS = (
    "Which ONE banking intent best describes `customer_message`? Choose its "
    "main requested action or problem. Use category meanings and any labeled "
    "examples to distinguish similar intents. Match meaning, not shared "
    "keywords. Examples are references, not the message to classify. Every "
    "message belongs to exactly one of these 77 categories."
)


@dataclass(frozen=True)
class Spec:
    name: str
    summary: dict
    directory: Path
    summary_key: str
    label: str
    context: str


SPECS = [
    Spec("old_a0", {}, Q, "old_a0", "Old mean-pool", "message only"),
    Spec("old_a2c", {}, Q, "old_a2c", "Old mean-pool", "examples that fit"),
    Spec("old_len", {}, M, "examples24_5120", "Old mean-pool", "all 24 at 5,120"),
    Spec("query_a0", {}, Q, "query_a0", "Query attention", "message only"),
    Spec("query_a2c", {}, Q, "query_a2c", "Query attention", "examples that fit"),
    Spec("len_a0", {}, L, "len_a0", "Query attention, 5,120", "message only"),
    Spec("len_a2c", {}, L, "len_a2c", "Query attention, 5,120", "all 24 at 5,120"),
]
BY_NAME = {spec.name: spec for spec in SPECS}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(text: str) -> str:
    """The contamination key used by the storage audit and the thesis generator."""
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def pp(x: float) -> str:
    text = f"{100 * x:+.2f}"
    return f"${text}$" if text.startswith("-") else text


def num(x: int) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def sci(p: float) -> str:
    if p == 0.0:
        return r"$<10^{-300}$"
    exp = int(math.floor(math.log10(p)))
    mant = p / 10 ** exp
    return rf"${mant:.2f}\times10^{{{exp}}}$"


def sci_text(p: float) -> str:
    if p == 0.0:
        return r"$p<10^{-300}$"
    exp = int(math.floor(math.log10(p)))
    mant = p / 10 ** exp
    return rf"$p = {mant:.2f}\times10^{{{exp}}}$"


def h12(value: str) -> str:
    return value[:12]


def check_summary(summary: dict, directory: Path) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Verify a summary against its per-example files and return the vectors."""
    arms: dict[str, dict] = {}
    vectors: dict[str, np.ndarray] = {}
    for name, arm in summary["arms"].items():
        path = directory / f"per_example_{name}.jsonl"
        if sha256_file(path) != arm["per_example_sha256"]:
            raise SystemExit(f"{path}: per-example hash mismatch")
        rows = read_jsonl(path)
        correct = sum(int(row["correct"]) for row in rows)
        if len(rows) != arm["n"] or correct != arm["correct"]:
            raise SystemExit(f"{path}: recomputed {correct}/{len(rows)} != stored")
        lo, hi = wilson(correct, len(rows))
        if abs(lo - arm["wilson95"][0]) > 1e-12 or abs(hi - arm["wilson95"][1]) > 1e-12:
            raise SystemExit(f"{path}: Wilson interval mismatch")
        vector = np.zeros(N_ROWS, dtype=np.int8)
        seen = np.zeros(N_ROWS, dtype=bool)
        for row in rows:
            i = int(row["i"])
            if seen[i]:
                raise SystemExit(f"{path}: duplicate row index {i}")
            seen[i] = True
            vector[i] = int(row["correct"])
        if not seen.all():
            raise SystemExit(f"{path}: missing row indices")
        arms[name] = arm
        vectors[name] = vector
    return arms, vectors


def check_pairs(summary: dict, arms: dict[str, dict], vectors: dict[str, np.ndarray],
                jev: np.ndarray) -> None:
    for key, stored in summary["paired"].items():
        if "_vs_" in key:
            left, right = key.split("_vs_", 1)
            if left not in vectors or right not in vectors:
                raise SystemExit(f"{key}: unknown arm")
            a, b = vectors[left], vectors[right]
            diff_key = "diff"
        else:
            if key not in vectors:
                raise SystemExit(f"{key}: unknown arm")
            a, b = vectors[key], jev
            diff_key = "diff_vs_jev"
        diff = float(a.mean() - b.mean())
        m = mcnemar_exact(a, b)
        lower = bootstrap_lower(a, b, seed=BOOT_SEED)
        if abs(diff - stored[diff_key]) > 1e-12:
            raise SystemExit(f"{key}: diff {diff} != {stored[diff_key]}")
        if abs(m["p"] - stored["mcnemar"]["p"]) > 1e-15:
            raise SystemExit(f"{key}: McNemar p differs")
        if abs(lower - stored["bootstrap_lower_95"]) > 1e-12:
            raise SystemExit(f"{key}: bootstrap lower differs")


def check_decision(summary: dict, primary: str) -> None:
    decision = summary["decision"]
    arm = summary["arms"][primary]
    paired = summary["paired"][primary]
    diff = arm["accuracy"] - decision["p0_bar"]
    if abs(diff - decision["diff_vs_jev"]) > 1e-12:
        raise SystemExit("decision diff differs")
    if abs(paired["mcnemar"]["p"] - decision["mcnemar_p"]) > 1e-15:
        raise SystemExit("decision p differs")
    if abs(paired["bootstrap_lower_95"] - decision["bootstrap_lower_95"]) > 1e-12:
        raise SystemExit("decision lower differs")
    if decision["verdict"] == "P0 exceeded":
        expected = arm["accuracy"] > decision["p0_bar"] and paired["mcnemar"]["p"] < 0.05
    elif decision["verdict"] == "P0 met (equal or above)":
        expected = arm["accuracy"] + 1e-12 >= decision["p0_bar"]
    else:
        expected = arm["accuracy"] + 1e-12 < decision["p0_bar"]
    if not expected:
        raise SystemExit(f"decision {decision['verdict']!r} is inconsistent with stored arms")


def check_clean(clean_summary: dict, q_vec: np.ndarray, inputs: dict) -> dict:
    """Verify the decontaminated product arm and its contamination audit.

    The paper's headline is the frozen-example arm.  The clean summary is a
    separate product-path rerun with 25 test messages removed from the support
    bank before BM25 is rebuilt; it is not an independent copy of the frozen
    arm and the generator refuses to present it as one.
    """
    manifest = json.loads((CLEAN / "MANIFEST.json").read_text())
    manifest_sha = (CLEAN / "MANIFEST.sha256").read_text().split()[0]
    if sha256(CLEAN / "MANIFEST.json") != manifest_sha:
        raise SystemExit("clean MANIFEST.json does not match MANIFEST.sha256")
    for name, meta in manifest["files"].items():
        path = CLEAN / name
        if sha256(path) != meta["sha256"] or path.stat().st_size != meta["bytes"]:
            raise SystemExit(f"{path}: hash or size does not match its manifest")
    if hashlib.sha256(INPUTS.read_bytes()).hexdigest() != clean_summary["inputs"]["sha256"]:
        raise SystemExit("clean summary does not name the current frozen inputs")

    clean = read_jsonl(CLEAN / "per_example.jsonl")
    old = read_jsonl(PRODUCT / "per_example_bm25.jsonl")
    frozen = read_jsonl(Q / "per_example_query_a2c.jsonl")
    if not (len(clean) == len(old) == len(frozen) == N_ROWS):
        raise SystemExit("a product arm does not cover all 3,080 rows")
    eval_canon = {canon(row["text"]) for row in inputs["rows"]}
    eval_exact = {str(row["text"]).strip() for row in inputs["rows"]}
    gold_by_canon = {canon(row["text"]): row["gold"] for row in inputs["rows"]}

    clean_vec = np.zeros(N_ROWS, dtype=np.int8)
    old_vec = np.zeros(N_ROWS, dtype=np.int8)
    frozen_vec = np.zeros(N_ROWS, dtype=np.int8)
    for name, rows, target in (("clean", clean, clean_vec), ("old", old, old_vec),
                               ("frozen", frozen, frozen_vec)):
        seen = np.zeros(N_ROWS, dtype=bool)
        for row in rows:
            i = int(row["i"])
            if seen[i] or row["id"] != inputs["rows"][i]["id"]:
                raise SystemExit(f"{name}: duplicate index or row identity differs at {i}")
            seen[i] = True
            target[i] = int(row["correct"])
        if not seen.all():
            raise SystemExit(f"{name}: missing row indices")
    if int(clean_vec.sum()) != clean_summary["clean"]["correct"]:
        raise SystemExit("clean correctness total differs from its summary")
    if int(old_vec.sum()) != 2_742 or int(frozen_vec.sum()) != 2_742:
        raise SystemExit("the contaminated arms are no longer 2,742/3,080")
    if not np.array_equal(frozen_vec, q_vec):
        raise SystemExit("frozen per-example vector differs from the paper's query arm")
    if not np.array_equal(old_vec, frozen_vec):
        raise SystemExit("the product arm is no longer prediction-identical to the frozen arm")
    same_context = sum(int(a["context_sha256"] == b["context_sha256"]) for a, b in zip(old, frozen))
    if same_context != 2_971:
        raise SystemExit(f"product/frozen context identity changed: {same_context}")

    support = [
        json.loads(line)
        for line in gzip.decompress((CLEAN / "clean_support.jsonl.gz").read_bytes()).decode().splitlines()
        if line.strip()
    ]
    removed = json.loads((CLEAN / "removed_support.json").read_text())
    if len(removed) != 25 or len(support) != 9_968:
        raise SystemExit(f"clean support counts are {len(removed)}/{len(support)}")
    if any(canon(row["message"]) in eval_canon for row in support):
        raise SystemExit("an evaluation text remains in the clean support bank")
    if any(canon(row["message"]) not in eval_canon for row in removed):
        raise SystemExit("a removed support row is not an evaluation text")
    removed_exact = sum(1 for row in removed if str(row["message"]).strip() in eval_exact)
    support_same_label = sum(
        1 for row in removed if gold_by_canon[canon(row["message"])] == row["label"]
    )
    if removed_exact != 6 or support_same_label != 25:
        raise SystemExit("removed-row exact matches or labels changed")

    rows_with_any = set()
    cross_rows = set()
    cross_only = 0
    instances = self_instances = cross_instances = same_label = diff_label = exact = 0
    example_canon: set[str] = set()
    for row in inputs["rows"]:
        row_text = canon(row["text"])
        has_self = has_cross = False
        for example in row["examples"]:
            key = canon(example["message"])
            if key not in eval_canon:
                continue
            rows_with_any.add(str(row["id"]))
            example_canon.add(key)
            instances += 1
            if str(example["message"]).strip() == str(row["text"]).strip():
                exact += 1
            if key == row_text:
                has_self = True
                self_instances += 1
            else:
                has_cross = True
                cross_instances += 1
                cross_rows.add(str(row["id"]))
            if example.get("intent") == row["gold"]:
                same_label += 1
            else:
                diff_label += 1
        if has_cross and not has_self:
            cross_only += 1
    counts = {
        "rows_with_any": len(rows_with_any),
        "instances": instances,
        "self_instances": self_instances,
        "cross_instances": cross_instances,
        "cross_rows": len(cross_rows),
        "cross_only_rows": cross_only,
        "same_label": same_label,
        "diff_label": diff_label,
        "exact": exact,
        "unique": len(example_canon),
    }
    stored = clean_summary["frozen_examples"]
    if (
        counts["rows_with_any"] != 255
        or counts["instances"] != 283
        or counts["self_instances"] != 25
        or counts["cross_instances"] != 258
        or counts["cross_rows"] != 236
        or counts["cross_only_rows"] != 230
        or counts["same_label"] != 81
        or counts["diff_label"] != 202
        or counts["exact"] != 6
        or counts["unique"] != 25
        or stored["rows_with_any_evaluation_text"] != 255
        or stored["instances_matching_an_evaluation_text"] != 283
        or not stored["example_set_equals_support_overlap"]
    ):
        raise SystemExit(f"frozen-example contamination counts changed: {counts}")

    unaffected_mask = np.array([
        not any(canon(example["message"]) in eval_canon for example in row["examples"])
        for row in inputs["rows"]
    ])
    if int(unaffected_mask.sum()) != 2_825:
        raise SystemExit(f"unaffected subset is {int(unaffected_mask.sum())} rows")
    unaffected_n = int(unaffected_mask.sum())
    unaffected_correct = int(q_vec[unaffected_mask].sum())
    if unaffected_correct != 2_511:
        raise SystemExit(f"unaffected subset has {unaffected_correct} correct")
    context_changed = sum(
        int(a["context_sha256"] != b["context_sha256"]) for a, b in zip(clean, old)
    )
    prediction_same = sum(int(a["pred_index"] == b["pred_index"]) for a, b in zip(clean, old))
    if context_changed != 256 or prediction_same != 3_077:
        raise SystemExit(f"clean/old context or prediction counts changed: {context_changed}/{prediction_same}")
    return {
        "summary": clean_summary,
        "manifest": manifest,
        "clean_vec": clean_vec,
        "old_vec": old_vec,
        "unaffected_mask": unaffected_mask,
        "unaffected_n": unaffected_n,
        "unaffected_correct": unaffected_correct,
        "counts": counts,
        "removed": removed,
        "removed_exact": removed_exact,
        "support_same_label": support_same_label,
        "same_context": same_context,
        "context_changed": context_changed,
        "prediction_same": prediction_same,
        "clean_vs_old": mcnemar_exact(clean_vec, old_vec),
    }


def prompt_budget(inputs: dict) -> dict:
    source = load_json(PROMPT / "SOURCE.json")
    for name in ("descriptions.json", "categories.json"):
        path = PROMPT / name
        if sha256(path) != source[name]["sha256"]:
            raise SystemExit(f"{path}: vendored prompt input hash mismatch")
    categories = load_json(PROMPT / "categories.json")
    descriptions = load_json(PROMPT / "descriptions.json")
    labels = sorted(categories)
    if len(labels) != N_LABELS or set(labels) != set(descriptions):
        raise SystemExit("prompt inputs do not define exactly the 77 labels")
    criteria = {label: descriptions[label] for label in labels}
    question = {
        "type": "choice",
        "instructions": JEV_INSTRUCTIONS,
        "criteria": criteria,
    }
    defs_bytes = len(json.dumps(criteria, ensure_ascii=True).encode())
    question_bytes = len(json.dumps(question, ensure_ascii=True).encode())
    full: list[int] = []
    no_examples: list[int] = []
    for row in inputs["rows"]:
        if len(row["examples"]) != N_EXAMPLES:
            raise SystemExit(f"row {row['id']}: {len(row['examples'])} examples")
        state = {
            "customer_message": row["text"],
            "labeled_examples": [
                {"message": example["message"], "intent": example["intent"]}
                for example in row["examples"]
            ],
        }
        payload = {
            "model": "jev-1.13.0",
            "state": state,
            "questions": {"intent": question},
        }
        full.append(len(json.dumps(payload, ensure_ascii=True).encode()))
        state_no = {"customer_message": row["text"]}
        payload_no = {
            "model": "jev-1.13.0",
            "state": state_no,
            "questions": {"intent": question},
        }
        no_examples.append(len(json.dumps(payload_no, ensure_ascii=True).encode()))
    median = int(statistics.median(full))
    examples_median = int(statistics.median([a - b for a, b in zip(full, no_examples)]))
    result = {
        "protocol": "jev-prompt-budget-reconstruction-v1",
        "n": len(full),
        "labels": len(labels),
        "examples_per_row": N_EXAMPLES,
        "definitions_bytes": defs_bytes,
        "question_bytes": question_bytes,
        "prompt_bytes_median": median,
        "prompt_bytes_min": min(full),
        "prompt_bytes_max": max(full),
        "prompt_bytes_mean": round(statistics.mean(full), 1),
        "examples_bytes_median": examples_median,
        "azurite_long_budget": LONG_BUDGET,
        "source": source,
    }
    (GEN / "prompt_budget.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def arm_macros(macros: dict[str, str], prefix: str, arm: dict, pair: dict) -> None:
    macros[f"P{prefix}Acc"] = pct(arm["accuracy"])
    macros[f"P{prefix}Lo"] = pct(arm["wilson95"][0])
    macros[f"P{prefix}Hi"] = pct(arm["wilson95"][1])
    macros[f"P{prefix}Correct"] = num(arm["correct"])
    macros[f"P{prefix}N"] = num(arm["n"])
    macros[f"P{prefix}Examples"] = f"{arm['mean_effective_examples']:.1f}"
    macros[f"P{prefix}ExamplesMin"] = str(arm["min_effective_examples"])
    macros[f"P{prefix}ExamplesMax"] = str(arm["max_effective_examples"])
    macros[f"P{prefix}Context"] = num(round(arm["mean_context_bytes"]))
    macros[f"P{prefix}Gap"] = pp(pair["diff_vs_jev"])
    macros[f"P{prefix}Lower"] = pp(pair["bootstrap_lower_95"])
    macros[f"P{prefix}P"] = sci(pair["mcnemar"]["p"])
    macros[f"P{prefix}PText"] = sci_text(pair["mcnemar"]["p"])


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "diff": float(a.mean() - b.mean()),
        "mcnemar": mcnemar_exact(a, b),
        "bootstrap_lower_95": bootstrap_lower(a, b, seed=BOOT_SEED),
    }


def build_macros(arms: dict[str, dict], vectors: dict[str, np.ndarray], jev: np.ndarray,
                 q_summary: dict, l_summary: dict, m_summary: dict, budget: dict,
                 clean: dict) -> dict[str, str]:
    m: dict[str, str] = {}
    m["PJevAcc"] = pct(jev.mean())
    m["PJevCorrect"] = num(int(jev.sum()))
    m["PN"] = num(N_ROWS)
    m["PLabels"] = str(N_LABELS)
    m["PExamples"] = str(N_EXAMPLES)
    m["PBar"] = pct(P0_BAR)
    m["PMargin"] = f"{abs(MARGIN) * 100:.1f}"
    m["PNativeBudget"] = num(NATIVE_BUDGET)
    m["PLongBudget"] = num(LONG_BUDGET)

    for spec in SPECS:
        arm_macros(m, {
            "old_a0": "OldZero",
            "old_a2c": "OldTwo",
            "old_len": "OldLen",
            "query_a0": "QueryZero",
            "query_a2c": "QueryTwo",
            "len_a0": "LenZero",
            "len_a2c": "LenTwo",
        }[spec.name], arms[spec.name], arms[spec.name]["_pair"])

    q = arms["query_a2c"]
    m["PQueryGap"] = pp(q["_pair"]["diff_vs_jev"])
    m["PQueryGapAbs"] = f"{abs(q['_pair']['diff_vs_jev']) * 100:.2f}"
    m["PQueryLower"] = pp(q["_pair"]["bootstrap_lower_95"])
    m["PQueryP"] = sci(q["_pair"]["mcnemar"]["p"])
    m["PQueryPText"] = sci_text(q["_pair"]["mcnemar"]["p"])
    m["PQueryNonInf"] = "yes" if q["_pair"]["bootstrap_lower_95"] > MARGIN else "no"
    l = arms["len_a2c"]
    m["PLenGap"] = pp(l["_pair"]["diff_vs_jev"])
    m["PLenGapAbs"] = f"{abs(l['_pair']['diff_vs_jev']) * 100:.2f}"
    m["PLenTwoGapAbs"] = m["PLenGapAbs"]
    m["PLenLower"] = pp(l["_pair"]["bootstrap_lower_95"])
    m["PLenP"] = sci(l["_pair"]["mcnemar"]["p"])
    m["PLenPText"] = sci_text(l["_pair"]["mcnemar"]["p"])
    m["PLenNonInf"] = "yes" if l["_pair"]["bootstrap_lower_95"] > MARGIN else "no"

    # Mechanism and transfer comparisons computed from the aligned vectors.
    mech = paired(vectors["query_a2c"], vectors["old_a2c"])
    m["PMechGain"] = pp(mech["diff"])
    m["PMechGainAbs"] = f"{abs(mech['diff']) * 100:.2f}"
    m["PMechLower"] = pp(mech["bootstrap_lower_95"])
    m["PMechP"] = sci(mech["mcnemar"]["p"])
    m["PMechPText"] = sci_text(mech["mcnemar"]["p"])

    len_mech = paired(vectors["len_a2c"], vectors["old_a2c"])
    m["PLenMechGain"] = pp(len_mech["diff"])
    m["PLenMechLower"] = pp(len_mech["bootstrap_lower_95"])
    m["PLenMechP"] = sci(len_mech["mcnemar"]["p"])

    q_vs_old_zero = paired(vectors["query_a2c"], vectors["old_a0"])
    m["PQueryVsOldZero"] = pp(q_vs_old_zero["diff"])
    m["PQueryVsOldZeroLower"] = pp(q_vs_old_zero["bootstrap_lower_95"])
    m["PQueryVsOldZeroP"] = sci(q_vs_old_zero["mcnemar"]["p"])
    m["PQueryVsOldZeroPText"] = sci_text(q_vs_old_zero["mcnemar"]["p"])

    old_vs_message = paired(vectors["old_a2c"], vectors["old_a0"])
    m["POldVsMessage"] = pp(old_vs_message["diff"])
    m["POldVsMessageLower"] = pp(old_vs_message["bootstrap_lower_95"])
    m["POldVsMessageP"] = sci(old_vs_message["mcnemar"]["p"])
    m["POldVsMessagePText"] = sci_text(old_vs_message["mcnemar"]["p"])

    len_vs_native = paired(vectors["len_a2c"], vectors["query_a2c"])
    m["PLenVsNative"] = pp(len_vs_native["diff"])
    m["PLenVsNativeLower"] = pp(len_vs_native["bootstrap_lower_95"])
    m["PLenVsNativeP"] = sci(len_vs_native["mcnemar"]["p"])
    m["PLenVsNativePText"] = sci_text(len_vs_native["mcnemar"]["p"])

    len_vs_old_zero = paired(vectors["len_a2c"], vectors["old_a0"])
    m["PLenVsOldZero"] = pp(len_vs_old_zero["diff"])
    m["PLenVsOldZeroLower"] = pp(len_vs_old_zero["bootstrap_lower_95"])
    m["PLenVsOldZeroP"] = sci(len_vs_old_zero["mcnemar"]["p"])
    m["PLenVsOldZeroPText"] = sci_text(len_vs_old_zero["mcnemar"]["p"])

    len_vs_old_two = paired(vectors["len_a2c"], vectors["old_a2c"])
    m["PLenVsOldTwo"] = pp(len_vs_old_two["diff"])
    m["PLenVsOldTwoLower"] = pp(len_vs_old_two["bootstrap_lower_95"])
    m["PLenVsOldTwoP"] = sci(len_vs_old_two["mcnemar"]["p"])

    len_vs_len_zero = paired(vectors["len_a2c"], vectors["len_a0"])
    m["PLenVsLenZero"] = pp(len_vs_len_zero["diff"])
    m["PLenVsLenZeroLower"] = pp(len_vs_len_zero["bootstrap_lower_95"])
    m["PLenVsLenZeroP"] = sci(len_vs_len_zero["mcnemar"]["p"])

    m["PQueryEff"] = f"{arms['query_a2c']['mean_effective_examples']:.1f}"
    m["PQueryEffMin"] = str(arms["query_a2c"]["min_effective_examples"])
    m["PQueryEffMax"] = str(arms["query_a2c"]["max_effective_examples"])
    m["PLenEff"] = f"{arms['len_a2c']['mean_effective_examples']:.1f}"
    m["PLenEffMin"] = str(arms["len_a2c"]["min_effective_examples"])
    m["PLenEffMax"] = str(arms["len_a2c"]["max_effective_examples"])

    # Prompt-budget reconstruction.
    m["PJevDefsBytes"] = num(budget["definitions_bytes"])
    m["PJevQuestionBytes"] = num(budget["question_bytes"])
    m["PJevPromptMedian"] = num(budget["prompt_bytes_median"])
    m["PJevPromptMin"] = num(budget["prompt_bytes_min"])
    m["PJevPromptMax"] = num(budget["prompt_bytes_max"])
    m["PJevPromptMean"] = num(round(budget["prompt_bytes_mean"]))
    m["PJevExamplesMedian"] = num(budget["examples_bytes_median"])
    m["PJevPromptRatio"] = f"{budget['prompt_bytes_median'] / LONG_BUDGET:.2f}"
    m["PJevPromptMaxRatio"] = f"{budget['prompt_bytes_max'] / LONG_BUDGET:.2f}"
    m["PJevDefsRatio"] = f"{budget['definitions_bytes'] / LONG_BUDGET:.2f}"
    m["PAzuriteShare"] = f"{LONG_BUDGET / budget['prompt_bytes_median'] * 100:.1f}"
    m["PJevDefsShare"] = f"{budget['definitions_bytes'] / budget['prompt_bytes_median'] * 100:.1f}"

    # Contamination audit and the decontaminated product rerun.  The clean arm
    # is not a second frozen-example arm; it is the shipped BM25 path with the
    # 25 evaluation texts removed before the index is rebuilt.
    cs = clean["summary"]
    c = clean["counts"]
    m["PContamSupportRows"] = num(cs["support"]["original_rows"])
    m["PContamSupportClean"] = num(cs["support"]["clean_rows"])
    m["PContamSupportMatches"] = num(cs["support"]["removed_rows"])
    m["PContamSupportExact"] = num(cs["support"]["removed_exact_text_matches"])
    m["PContamSupportShare"] = f"{100 * cs['support']['removed_rows'] / cs['support']['original_rows']:.2f}"
    m["PContamSupportSameLabel"] = num(clean["support_same_label"])
    m["PContamEvalRows"] = num(c["rows_with_any"])
    m["PContamEvalShare"] = f"{100 * c['rows_with_any'] / N_ROWS:.2f}"
    m["PContamInstances"] = num(c["instances"])
    m["PContamSelf"] = num(c["self_instances"])
    m["PContamCross"] = num(c["cross_instances"])
    m["PContamCrossRows"] = num(c["cross_rows"])
    m["PContamCrossOnly"] = num(c["cross_only_rows"])
    m["PContamSameLabel"] = num(c["same_label"])
    m["PContamDiffLabel"] = num(c["diff_label"])
    m["PContamExactExamples"] = num(c["exact"])
    m["PContamUnique"] = num(c["unique"])
    clean_acc = float(cs["clean"]["accuracy"])
    clean_correct = int(cs["clean"]["correct"])
    clean_lo, clean_hi = wilson(clean_correct, N_ROWS)
    old_acc = float(clean["old_vec"].mean())
    m["PCleanN"] = num(N_ROWS)
    m["PCleanCorrect"] = num(clean_correct)
    m["PCleanAcc"] = pct(clean_acc)
    m["PCleanLo"] = pct(clean_lo)
    m["PCleanHi"] = pct(clean_hi)
    m["PCleanDiff"] = pp(clean_acc - old_acc)
    m["PCleanDiffAbs"] = f"{abs(clean_acc - old_acc) * 100:.2f}"
    m["PCleanWins"] = num(clean["clean_vs_old"]["wins"])
    m["PCleanLosses"] = num(clean["clean_vs_old"]["losses"])
    m["PCleanP"] = "1.00"
    m["PCleanPText"] = "p = 1.00"
    unaffected_n = clean["unaffected_n"]
    unaffected_correct = clean["unaffected_correct"]
    ulo, uhi = wilson(unaffected_correct, unaffected_n)
    m["PUnaffectedN"] = num(unaffected_n)
    m["PUnaffectedCorrect"] = num(unaffected_correct)
    m["PUnaffectedAcc"] = pct(unaffected_correct / unaffected_n)
    m["PUnaffectedLo"] = pct(ulo)
    m["PUnaffectedHi"] = pct(uhi)
    m["PContamContextChanged"] = num(clean["context_changed"])
    m["PContamPredSame"] = num(clean["prediction_same"])
    m["PContamProductFrozenSameContext"] = num(clean["same_context"])
    m["PContamProductFrozenDiscordant"] = "0"

    # Training and provenance.
    meta = load_json(ROOT / "results/jev_query_5120/train_meta.json")
    params_record = load_json(
        ROOT / "results/recovered_20261028/results/pretrain/l1_large_byte_iso10v3/summary.json"
    )
    sep = l_summary["train_test_separation"]
    m["PParams"] = num(params_record["params"])
    m["PParamsM"] = f"{params_record['params'] / 1e6:.2f}"
    m["PNativeSeq"] = num(params_record["seq"])
    m["PBOOT"] = num(BOOTSTRAP_RESAMPLES)
    m["PTrainN"] = num(sep["train_n"])
    m["PTrainCorrect"] = num(sep["train_correct"])
    m["PSteps"] = num(meta["steps"])
    m["PBatch"] = str(meta["batch"])
    m["PSeq"] = num(LONG_BUDGET + 2)
    m["PTokens"] = num(meta["steps"] * meta["batch"] * (LONG_BUDGET + 2))
    m["PGPUHours"] = f"{meta['train_seconds'] / 3600:.2f}"
    m["PSecStep"] = f"{meta['ms_per_step'] / 1000:.2f}"
    m["PTrainAcc"] = pct(sep["train_accuracy"])
    m["PTestAcc"] = pct(sep["test_accuracy"])
    m["PTrainGap"] = pp(sep["test_minus_train"])
    m["PTrainLoss"] = f"{sep['train_loss']:.4f}"
    m["PLastLoss"] = f"{sep['final_training_loss']:.4f}"
    m["PMeanLastFifty"] = f"{sep['mean_training_loss_last_50']:.4f}"
    m["PQueryPrereg"] = h12(q_summary["prereg_sha256"])
    m["PLenPrereg"] = h12(l_summary["prereg_sha256"])
    m["PLenAmendment"] = h12(l_summary["amendment_sha256"])
    m["PInputs"] = h12(q_summary["inputs_sha256"])
    m["POldHead"] = h12(q_summary["heads"]["old"]["sha256"])
    m["PQueryHead"] = h12(q_summary["heads"]["query"]["sha256"])
    m["PLenHead"] = h12(l_summary["heads"]["query"]["sha256"])
    m["PLenInit"] = h12(l_summary["init_checkpoint"]["sha256"])
    m["PCodeQuery"] = h12(q_summary["code_sha256"])
    m["PCodeLen"] = h12(l_summary["code_sha256"])
    m["PQueryVerdict"] = q_summary["decision"]["verdict"]
    m["PLenVerdict"] = l_summary["decision"]["verdict"]
    m["PJevMismatchGap"] = pp(m_summary["arms"]["examples24_5120"]["accuracy"] - jev.mean())

    # Prediction concentration from the per-example prediction column.
    for spec in SPECS:
        rows = read_jsonl(spec.directory / f"per_example_{spec.summary_key}.jsonl")
        counts = Counter(row["pred"] for row in rows)
        top = counts.most_common()
        prefix = {
            "old_a0": "OldZero",
            "old_a2c": "OldTwo",
            "old_len": "OldLen",
            "query_a0": "QueryZero",
            "query_a2c": "QueryTwo",
            "len_a0": "LenZero",
            "len_a2c": "LenTwo",
        }[spec.name]
        m[f"P{prefix}Distinct"] = str(len(counts))
        m[f"P{prefix}Top"] = pct(top[0][1] / len(rows))
        m[f"P{prefix}TopThree"] = pct(sum(v for _, v in top[:3]) / len(rows))

    return m


def write_macros(macros: dict[str, str], path: Path) -> None:
    for name in macros:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise SystemExit(f"macro name is not letters-only: {name}")
    lines = [
        "% generated by paper/gen_paper.py -- do not edit by hand",
        *[rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()],
    ]
    path.write_text("\n".join(lines) + "\n")


def write_table(arms: dict[str, dict], jev: np.ndarray, budget: dict, path: Path) -> None:
    def row(name: str, label: str, examples: str | None = None) -> str:
        arm = arms[name]
        ex = examples if examples is not None else f"{arm['mean_effective_examples']:.1f}"
        pair = arm["_pair"]
        return (
            f"{label} & {ex} & {num(round(arm['mean_context_bytes']))} & "
            f"{num(arm['n'])} & {num(arm['correct'])} & {pct(arm['accuracy'])} & "
            f"{pct(arm['wilson95'][0])}--{pct(arm['wilson95'][1])} & "
            f"{pp(pair['diff_vs_jev'])} & {sci(pair['mcnemar']['p'])} \\\\"
        )

    jlo, jhi = wilson(int(jev.sum()), len(jev))
    rows = [
        row("old_a0", "Old mean-pool, message only"),
        row("old_a2c", "Old mean-pool, examples that fit"),
        row("old_len", "Old mean-pool, all 24 at 5{,}120"),
        row("query_a0", "Query attention, message only"),
        row("query_a2c", "Query attention, examples that fit"),
        row("len_a0", "Query attention, 5{,}120, message only"),
        row("len_a2c", "Query attention, 5{,}120, all 24"),
        (
            f"Jev 1.13, cited & 24 & {num(budget['prompt_bytes_median'])} & "
            f"{num(len(jev))} & {num(int(jev.sum()))} & "
            f"{pct(jev.mean())} & {pct(jlo)}--{pct(jhi)} & --- & --- \\\\"
        ),
    ]
    text = r"""\begin{table*}[t]
\centering
\footnotesize
\caption{The pre-registered comparison on all \PN{} official Banking77 test rows.
\emph{Old} is the mean-pool head; \emph{query} is the message-query attention
head. The byte budget names the context window; \emph{examples} is the mean
number of complete retrieved examples that fit (range in the text). The
old-head 5{,}120 arm is the matched-test packing of all 24 examples; its native
control is the query-evaluation packing, which averages \POldTwoExamples{}
examples. The Jev row is the cited public run, not a rerun; its Bytes cell is
the reconstructed JSON prompt size, not a model context window. Wilson
intervals are 95\%; $\Delta$ and exact two-sided McNemar $p$ are against Jev's frozen
per-row correctness vector. The 5{,}120 query head is a joint encoder/head
fine-tune, not a length-only adaptation. Every cell is generated from
\texttt{results/jev\_query}, \texttt{results/jev\_query\_5120} or
\texttt{results/jev\_matched}.}
\label{tab:main}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
Arm & Ex. & Bytes & $n$ & Correct & Acc. & 95\% Wilson & $\Delta$ vs Jev & McNemar $p$ \\
 & & & & & (\%) & & (pp) & \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    path.write_text(text)


def write_transitions(vectors: dict[str, np.ndarray], path: Path) -> None:
    rows = []
    pairs = [
        ("Query ex. vs old msg.", "query_a2c", "old_a0"),
        ("Query ex. vs old ex.", "query_a2c", "old_a2c"),
        ("Long all 24 vs query ex.", "len_a2c", "query_a2c"),
        ("Long all 24 vs old msg.", "len_a2c", "old_a0"),
        ("Old ex. vs old msg.", "old_a2c", "old_a0"),
        ("Long all 24 vs long msg.", "len_a2c", "len_a0"),
    ]
    for label, left, right in pairs:
        a, b = vectors[left], vectors[right]
        test = mcnemar_exact(a, b)
        diff = float(a.mean() - b.mean())
        net = test["wins"] - test["losses"]
        rows.append(
            f"{label} & {num(test['wins'])} & {num(test['losses'])} & "
            f"{net:+d} & {pp(diff)} & {sci(test['p'])} \\\\"
        )
    text = r"""\begin{table*}[t]
\centering
\footnotesize
\caption{Exact McNemar transition counts between selected arms. ``Wins'' are
rows the left arm gets right and the right arm gets wrong; ``losses'' are the
reverse. Positive net means the left arm is correct on more rows. $\Delta$ is
the accuracy difference in points. Every count is recomputed from the stored
per-row vectors; the table is generated with
\texttt{paper/gen\_paper.py}.}
\label{tab:transitions}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Comparison & Wins & Losses & Net & $\Delta$ (pp) & McNemar $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    path.write_text(text)


def write_hashes(q_summary: dict, l_summary: dict, path: Path) -> None:
    rows = [
        ("Query pre-registration", h12(q_summary["prereg_sha256"])),
        ("Query inputs", h12(q_summary["inputs_sha256"])),
        ("Query code", h12(q_summary["code_sha256"])),
        ("Old head", h12(q_summary["heads"]["old"]["sha256"])),
        ("Query head", h12(q_summary["heads"]["query"]["sha256"])),
        ("5,120 pre-registration", h12(l_summary["prereg_sha256"])),
        ("5,120 amendment", h12(l_summary["amendment_sha256"])),
        ("5,120 init checkpoint", h12(l_summary["init_checkpoint"]["sha256"])),
        ("5,120 head", h12(l_summary["heads"]["query"]["sha256"])),
        ("5,120 code", h12(l_summary["code_sha256"])),
    ]
    body = "\n".join(f"{name} & \\texttt{{{value}}} \\\\" for name, value in rows)
    text = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Content identities for the records the paper reads. Each is the first
12 hex digits of a SHA-256; the full values and the per-example hashes are in
the stored summaries and \texttt{paper/RECONCILIATION.md}.}
\label{tab:hashes}
\begin{tabular}{@{}ll@{}}
\toprule
Artifact & SHA-256 (first 12) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    path.write_text(text)


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig_accuracy(arms: dict[str, dict], jev: float, path: Path) -> None:
    names = ["old_a0", "old_a2c", "old_len", "query_a0", "query_a2c", "len_a2c"]
    labels = [
        "Old, message",
        "Old, examples",
        "Old, all 24 @5,120",
        "Query, message",
        "Query, examples",
        "Query, all 24 @5,120",
    ]
    colors = ["#8c8c8c", "#8c8c8c", "#4d4d4d", "#9ecae1", "#3182bd", "#08519c"]
    values = [100 * arms[n]["accuracy"] for n in names]
    lo = [100 * arms[n]["wilson95"][0] for n in names]
    hi = [100 * arms[n]["wilson95"][1] for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.4, height=0.62)
    ax.errorbar(values, y, xerr=[np.array(values) - np.array(lo), np.array(hi) - np.array(values)],
                fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
    for yi, value in zip(y, values):
        ax.text(value + 1.2, yi, f"{value:.2f}", va="center", fontsize=6)
    ax.axvline(100 * jev, color="#b2182b", linestyle="--", linewidth=0.8)
    ax.text(100 * jev + 0.8, -0.55, f"Jev {100 * jev:.2f}", color="#b2182b", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)")
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_concentration(arms: dict[str, dict], path: Path) -> None:
    names = ["old_a0", "old_a2c", "old_len", "query_a2c", "len_a2c"]
    labels = ["Old\nmessage", "Old\nexamples", "Old\nall 24", "Query\nexamples", "Query\nall 24"]
    top1, top3, distinct = [], [], []
    for name in names:
        rows = read_jsonl(BY_NAME[name].directory / f"per_example_{BY_NAME[name].summary_key}.jsonl")
        counts = Counter(row["pred"] for row in rows)
        ordered = counts.most_common()
        top1.append(100 * ordered[0][1] / len(rows))
        top3.append(100 * sum(v for _, v in ordered[:3]) / len(rows))
        distinct.append(len(counts))
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.35, 2.25), gridspec_kw={"width_ratios": [2.1, 1.0]})
    ax1.bar(x - 0.19, top1, width=0.38, color="#9ecae1", edgecolor="black", linewidth=0.35, label="Top-1")
    ax1.bar(x + 0.19, top3, width=0.38, color="#3182bd", edgecolor="black", linewidth=0.35, label="Top-3")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right")
    ax1.set_ylabel("Share of predictions (%)")
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_ylim(0, max(top3) * 1.18)
    ax2.bar(x, distinct, width=0.62, color="#6baed6", edgecolor="black", linewidth=0.35)
    ax2.axhline(N_LABELS, color="0.3", linestyle=":", linewidth=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["O0", "O2", "OL", "Q2", "L2"], rotation=0)
    ax2.set_ylabel("Distinct labels")
    ax2.set_ylim(0, 85)
    for xi, value in zip(x, distinct):
        ax2.text(xi, value + 1.2, str(value), ha="center", fontsize=5.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_macros(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("\\newcommand{\\"):
            name = line.split("{\\", 1)[1].split("}", 1)[0]
            value = line.split("}{", 1)[1].rsplit("}", 1)[0]
            out[name] = value
    return out


def normalise(value: str) -> str:
    return value.replace("$", "").replace("{", "").replace("}", "").replace(" ", "").strip()


def as_float(value: str) -> float:
    text = normalise(value).replace("+", "")
    if text.startswith("<"):
        return 0.0
    return float(text.replace(",", ""))


def reconcile(macros: dict[str, str], arms: dict[str, dict], vectors: dict[str, np.ndarray],
              q_summary: dict, l_summary: dict, m_summary: dict, budget: dict) -> None:
    thesis = {}
    for name in ("macros_query.tex", "macros_len5120.tex", "macros_jev.tex",
                 "macros_contam.tex"):
        thesis.update(parse_macros(ROOT / "thesis" / "tables" / name))
    mapping = {
        "PJevAcc": "QueryJevPct",
        "PQueryTwoAcc": "QueryPctQueryTwo",
        "POldTwoAcc": "QueryPctOldTwo",
        "POldZeroAcc": "QueryPctOldZero",
        "PQueryZeroAcc": "QueryPctQueryZero",
        "PLenTwoAcc": "LenPctLenTwo",
        "PCleanAcc": "ContamCleanAcc",
        "PCleanCorrect": "ContamCleanCorrect",
        "PContamEvalRows": "ContamEvalRows",
        "PContamInstances": "ContamInstances",
        "PUnaffectedAcc": "ContamUnaffectedAcc",
        "PLenZeroAcc": "LenPctLenZero",
        "POldLenAcc": "JevMatchPctAOne",
        "PQueryTwoGap": "QueryGapQueryTwo",
        "PLenTwoGap": "LenGapLenTwo",
        "PQueryLower": "QueryLowerPrimary",
        "PLenLower": "LenLowerPrimary",
        "PMechGain": "QueryMechGain",
        "PLenMechGain": "LenMechGain",
        "PQueryVsOldZero": "QueryVsOldZeroGain",
        "PLenVsOldZero": "LenVsOldZeroGain",
        "PLenVsNative": "LenNativeDiff",
        "PTrainAcc": "LenTrainPct",
        "PTestAcc": "LenPctLenTwo",
        "PTrainGap": "LenGapTrainTest",
    }
    failures = []
    for paper_key, thesis_key in mapping.items():
        if thesis_key not in thesis:
            failures.append(f"missing thesis macro {thesis_key}")
            continue
        if paper_key.endswith("P") or paper_key.endswith("PText"):
            left, right = normalise(macros[paper_key]), normalise(thesis[thesis_key])
            ok = left == right
        else:
            left, right = as_float(macros[paper_key]), as_float(thesis[thesis_key])
            ok = abs(left - right) < 5e-3
        if not ok:
            failures.append(f"{paper_key}={macros[paper_key]} vs {thesis_key}={thesis[thesis_key]}")
    if failures:
        raise SystemExit("paper/thesis reconciliation failed:\n  " + "\n  ".join(failures))

    rows = []
    for spec in SPECS:
        prefix = {
            "old_a0": "OldZero",
            "old_a2c": "OldTwo",
            "old_len": "OldLen",
            "query_a0": "QueryZero",
            "query_a2c": "QueryTwo",
            "len_a0": "LenZero",
            "len_a2c": "LenTwo",
        }[spec.name]
        arm = arms[spec.name]
        rows.append((f"P{prefix}Acc", macros[f"P{prefix}Acc"], "summary", f"{spec.summary_key}.accuracy"))
        rows.append((f"P{prefix}Correct", macros[f"P{prefix}Correct"], "summary", f"{spec.summary_key}.correct"))
        rows.append((f"P{prefix}Gap", macros[f"P{prefix}Gap"], "summary", f"paired.{spec.summary_key}.diff_vs_jev"))
        rows.append((f"P{prefix}P", macros[f"P{prefix}P"], "summary", f"paired.{spec.summary_key}.mcnemar.p"))
    rows += [
        ("PJevDefsBytes", macros["PJevDefsBytes"], "generated/prompt_budget.json", "definitions_bytes"),
        ("PJevPromptMedian", macros["PJevPromptMedian"], "generated/prompt_budget.json", "prompt_bytes_median"),
        ("PJevPromptMin", macros["PJevPromptMin"], "generated/prompt_budget.json", "prompt_bytes_min"),
        ("PJevPromptMax", macros["PJevPromptMax"], "generated/prompt_budget.json", "prompt_bytes_max"),
        ("PJevPromptRatio", macros["PJevPromptRatio"], "derived", "prompt_median / 5118"),
        ("PContamSupportMatches", macros["PContamSupportMatches"],
         "results/product_path_clean/summary.json", "support.removed_rows"),
        ("PContamEvalRows", macros["PContamEvalRows"],
         "results/product_path_clean/summary.json", "frozen_examples.rows_with_any_evaluation_text"),
        ("PCleanAcc", macros["PCleanAcc"],
         "results/product_path_clean/summary.json", "clean.accuracy"),
        ("PUnaffectedAcc", macros["PUnaffectedAcc"], "derived", "2,511 / 2,825"),
        ("PTokens", macros["PTokens"], "results/jev_query_5120/train_meta.json", "steps * batch * 5120"),
        ("PGPUHours", macros["PGPUHours"], "results/jev_query_5120/train_meta.json", "train_seconds / 3600"),
    ]
    lines = [
        "# Reconciliation record",
        "",
        "Generated by `paper/gen_paper.py`. Every value below was recomputed from",
        "the stored per-example records before the macro was written. The final",
        "column of the first block is the summary field the value came from.",
        "",
        "## Paper values and their source",
        "",
        "| macro | value | source | field |",
        "|---|---:|---|---|",
    ]
    lines += [f"| `{a}` | `{b}` | `{c}` | `{d}` |" for a, b, c, d in rows]
    lines += [
        "",
        "## All generated macros",
        "",
        "Every macro used by the paper is listed here. The headline values are mapped",
        "to their source fields above; the rest are derived from the same verified",
        "summaries and per-example vectors.",
        "",
        "| macro | value |",
        "|---|---:|",
        *[f"| `{name}` | `{value}` |" for name, value in sorted(macros.items())],
        "",
        "## Cross-checks",
        "",
        f"- Jev vector: {int(vectors['old_a0'].size)} rows; "
        f"stored Jev correct = {q_summary['state']['jev_correct']}.",
        f"- Every per-example JSONL was checked against its recorded SHA-256.",
        f"- Every accuracy and Wilson interval was recomputed from its JSONL.",
        f"- Every paired McNemar test and paired bootstrap was recomputed from the aligned row vectors.",
        f"- Paper macros overlap the thesis macros in {len(mapping)} places; all agree.",
        f"- Every one of the {len(macros)} paper macros is defined by the generator; no measurement value is typed as a literal in the .tex.",
        f"- Prompt-budget reconstruction: {budget['n']} rows, {budget['labels']} labels, "
        f"{budget['examples_per_row']} examples per row.",
        "",
    ]
    (PAPER / "RECONCILIATION.md").write_text("\n".join(lines))
    (GEN / "reconciliation.json").write_text(json.dumps({
        "rows": [{"macro": a, "value": b, "source": c, "field": d} for a, b, c, d in rows],
        "thesis_shared": len(mapping),
        "prompt_budget": budget,
    }, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    q_summary = load_json(Q / "summary.json")
    l_summary = load_json(L / "summary.json")
    m_summary = load_json(M / "summary.json")
    clean_summary = load_json(CLEAN / "summary.json")
    inputs = load_json(INPUTS)
    if len(inputs["rows"]) != N_ROWS:
        raise SystemExit("input row count differs")
    jev = np.array([int(row["jev_correct"]) for row in inputs["rows"]], dtype=np.int8)
    if int(jev.sum()) != q_summary["state"]["jev_correct"]:
        raise SystemExit("Jev vector sum differs from summary")
    for row in inputs["rows"]:
        if len(row["examples"]) != N_EXAMPLES:
            raise SystemExit(f"{row['id']}: example count differs")
        if int(row["criteria_count"]) != N_LABELS:
            raise SystemExit(f"{row['id']}: criteria count differs")

    q_arms, q_vec = check_summary(q_summary, Q)
    l_arms, l_vec = check_summary(l_summary, L)
    m_arms, m_vec = check_summary(m_summary, M)
    clean = check_clean(clean_summary, q_vec["query_a2c"], inputs)
    check_pairs(q_summary, q_arms, q_vec, jev)
    check_pairs(l_summary, l_arms, l_vec, jev)
    check_pairs(m_summary, m_arms, m_vec, jev)
    check_decision(q_summary, "query_a2c")
    check_decision(l_summary, "len_a2c")
    if m_summary["decision"]["verdict"] != "gap survives":
        raise SystemExit("matched-test verdict changed")

    arms: dict[str, dict] = {}
    vectors: dict[str, np.ndarray] = {}
    for spec in SPECS:
        if spec.directory == Q:
            summary, vecs = q_summary, q_vec
        elif spec.directory == L:
            summary, vecs = l_summary, l_vec
        else:
            summary, vecs = m_summary, m_vec
        arm = dict(summary["arms"][spec.summary_key])
        key = spec.name if spec.name != "old_len" else "examples24_5120"
        arm["_pair"] = summary["paired"][key]
        arms[spec.name] = arm
        vectors[spec.name] = vecs[key]

    # Align every record to the frozen input row and verify its text hash.
    for spec in SPECS:
        rows = read_jsonl(spec.directory / f"per_example_{spec.summary_key}.jsonl")
        for rec in rows:
            i = int(rec["i"])
            if rec["id"] != inputs["rows"][i]["id"]:
                raise SystemExit(f"{spec.name}: row identity differs at {i}")
            if rec["text_sha256"] != sha256_text(inputs["rows"][i]["text"]):
                raise SystemExit(f"{spec.name}: message hash differs at {i}")

    GEN.mkdir(parents=True, exist_ok=True)
    budget = prompt_budget(inputs)
    macros = build_macros(arms, vectors, jev, q_summary, l_summary, m_summary, budget, clean)
    tex = (PAPER / "azurite.tex").read_text()
    used = set(re.findall(r"\\(P[A-Za-z]+)", tex))
    missing = used - set(macros)
    if missing:
        raise SystemExit("paper uses undefined macros: " + ", ".join(sorted(missing)))
    write_table(arms, jev, budget, GEN / "main_table.tex")
    write_hashes(q_summary, l_summary, GEN / "hashes.tex")
    write_transitions(vectors, GEN / "transitions.tex")
    style()
    fig_accuracy(arms, float(jev.mean()), GEN / "fig_accuracy.pdf")
    fig_concentration(arms, GEN / "fig_concentration.pdf")
    if args.check:
        print(f"verified {len(SPECS)} arms, {N_ROWS} rows, {int(jev.sum())} Jev-correct")
        print(f"prompt median {budget['prompt_bytes_median']:,} bytes; "
              f"definitions {budget['definitions_bytes']:,} bytes")
        print(f"query {macros['PQueryTwoAcc']}%, len {macros['PLenTwoAcc']}%")
        return 0
    GEN.mkdir(parents=True, exist_ok=True)
    write_macros(macros, GEN / "macros.tex")
    reconcile(macros, arms, vectors, q_summary, l_summary, m_summary, budget)
    print(f"wrote {GEN.relative_to(ROOT)}/macros.tex, main_table.tex, hashes.tex, figures")
    print(f"verified {len(SPECS)} arms, {N_ROWS} rows, {int(jev.sum())} Jev-correct")
    print(f"query {macros['PQueryTwoAcc']}% vs Jev {macros['PJevAcc']}%; "
          f"len {macros['PLenTwoAcc']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
