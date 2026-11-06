#!/usr/bin/env python3
"""Emit the contamination audit and decontaminated-product macros.

The thesis already reports the frozen-example arm, which is the protocol the
paper uses.  This generator adds the two facts that arm cannot supply by
itself: exactly which evaluation texts occur in the support bank, and what the
shipped product path scores after every one of those texts is removed and the
retrieval is run again.

Everything is read from the stored records and checked before a macro is
written:

    results/product_path_clean/summary.json
    results/product_path_clean/MANIFEST.json
    results/product_path_clean/per_example.jsonl
    results/product_path_clean/clean_support.jsonl.gz
    results/product_path_clean/removed_support.json
    results/product_path/per_example_bm25.jsonl
    results/jev_query/eval/per_example_query_a2c.jsonl
    results/jev_matched/inputs.json

    python thesis/gen_contamination.py            # write macros
    python thesis/gen_contamination.py --check    # verify only
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.stats import mcnemar_exact, wilson  # noqa: E402

CLEAN = ROOT / "results" / "product_path_clean"
OLD = ROOT / "results" / "product_path"
FROZEN = ROOT / "results" / "jev_query" / "eval"
INPUTS = ROOT / "results" / "jev_matched" / "inputs.json"
OUT = ROOT / "thesis" / "tables"
N_ROWS = 3_080
N_LABELS = 77


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canon(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def num(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def pp(x: float) -> str:
    text = f"{100 * x:+.2f}"
    return f"${text}$" if text.startswith("-") else text


def load() -> dict:
    summary = json.loads((CLEAN / "summary.json").read_text())
    manifest = json.loads((CLEAN / "MANIFEST.json").read_text())
    manifest_sha = (CLEAN / "MANIFEST.sha256").read_text().split()[0]
    if sha256_file(CLEAN / "MANIFEST.json") != manifest_sha:
        raise SystemExit("clean MANIFEST.json does not match MANIFEST.sha256")
    for name, meta in manifest["files"].items():
        path = CLEAN / name
        if sha256_file(path) != meta["sha256"] or path.stat().st_size != meta["bytes"]:
            raise SystemExit(f"{path}: hash or size does not match the manifest")
    if sha256_file(CLEAN / "per_example.jsonl") != manifest["files"]["per_example.jsonl"]["sha256"]:
        raise SystemExit("clean per-example hash mismatch")

    inputs_bytes = INPUTS.read_bytes()
    inputs = json.loads(inputs_bytes)
    if len(inputs["rows"]) != N_ROWS or len(inputs["labels"]) != N_LABELS:
        raise SystemExit("inputs do not carry 3,080 rows and 77 labels")
    if hashlib.sha256(inputs_bytes).hexdigest() != summary["inputs"]["sha256"]:
        raise SystemExit("clean summary does not name the current inputs")
    eval_canon = {canon(row["text"]) for row in inputs["rows"]}
    eval_exact = {str(row["text"]).strip() for row in inputs["rows"]}
    gold_by_canon = {canon(row["text"]): row["gold"] for row in inputs["rows"]}

    clean = read_jsonl(CLEAN / "per_example.jsonl")
    old = read_jsonl(OLD / "per_example_bm25.jsonl")
    frozen = read_jsonl(FROZEN / "per_example_query_a2c.jsonl")
    if not (len(clean) == len(old) == len(frozen) == N_ROWS):
        raise SystemExit("one of the per-example files is not the 3,080-row set")
    for name, rows in (("clean", clean), ("old", old), ("frozen", frozen)):
        if {str(row["id"]) for row in rows} != {str(row["id"]) for row in inputs["rows"]}:
            raise SystemExit(f"{name}: row ids differ from the frozen inputs")
        if len({int(row["i"]) for row in rows}) != N_ROWS:
            raise SystemExit(f"{name}: row indices are not unique")

    clean_c = sum(int(row["correct"]) for row in clean)
    old_c = sum(int(row["correct"]) for row in old)
    frozen_c = sum(int(row["correct"]) for row in frozen)
    if clean_c != summary["clean"]["correct"]:
        raise SystemExit("clean per-example total differs from its summary")
    if old_c != 2_742 or frozen_c != 2_742:
        raise SystemExit("the contaminated arms are no longer 2,742/3,080")

    clean_by = {int(row["i"]): int(row["correct"]) for row in clean}
    old_by = {int(row["i"]): int(row["correct"]) for row in old}
    frozen_by = {int(row["i"]): int(row["correct"]) for row in frozen}
    indices = sorted(clean_by)
    if indices != list(range(N_ROWS)):
        raise SystemExit("row indices are not 0..3,079")
    clean_vec = [clean_by[i] for i in indices]
    old_vec = [old_by[i] for i in indices]
    frozen_vec = [frozen_by[i] for i in indices]
    if mcnemar_exact(old_vec, frozen_vec)["wins"] or mcnemar_exact(old_vec, frozen_vec)["losses"]:
        raise SystemExit("the product and frozen arms are no longer prediction-identical")
    if sum(int(a["pred_index"]) == int(b["pred_index"]) for a, b in zip(old, frozen)) != N_ROWS:
        raise SystemExit("the product and frozen predictions differ")
    same_context = sum(int(a["context_sha256"] == b["context_sha256"]) for a, b in zip(old, frozen))
    if same_context != 2_971:
        raise SystemExit(f"product/frozen context identity changed: {same_context}")

    # The clean support bank must contain no evaluation text under the same
    # normalisation that selected the 25 removals.
    support = [
        json.loads(line)
        for line in gzip.decompress((CLEAN / "clean_support.jsonl.gz").read_bytes()).decode().splitlines()
        if line.strip()
    ]
    removed = json.loads((CLEAN / "removed_support.json").read_text())
    if len(removed) != 25 or len(support) != 9_968:
        raise SystemExit(f"removed/support counts are {len(removed)}/{len(support)}")
    if any(canon(row["message"]) in eval_canon for row in support):
        raise SystemExit("an evaluation text remains in the clean support bank")
    if any(canon(row["message"]) not in eval_canon for row in removed):
        raise SystemExit("a removed support row is not an evaluation text")
    if len({canon(row["message"]) for row in removed}) != 25:
        raise SystemExit("the removed support rows are not 25 unique messages")
    exact_removed = sum(1 for row in removed if str(row["message"]).strip() in eval_exact)
    support_same_label = sum(
        1 for row in removed if gold_by_canon[canon(row["message"])] == row["label"]
    )
    if exact_removed != 6 or support_same_label != 25:
        raise SystemExit("removed-row exact matches or labels changed")

    # The frozen example set is a separate record of the same overlap.  Count
    # both instances and rows; the latter is what the contamination share is
    # about.
    example_rows = set()
    example_cross_rows = set()
    example_instances = 0
    example_self = 0
    example_cross = 0
    example_same = 0
    example_diff = 0
    example_exact = 0
    example_eval_canon: set[str] = set()
    for row in inputs["rows"]:
        row_text = canon(row["text"])
        for example in row["examples"]:
            key = canon(example["message"])
            if key not in eval_canon:
                continue
            example_rows.add(str(row["id"]))
            example_eval_canon.add(key)
            example_instances += 1
            if str(example["message"]).strip() == str(row["text"]).strip():
                example_exact += 1
            if key == row_text:
                example_self += 1
            else:
                example_cross += 1
                example_cross_rows.add(str(row["id"]))
            if example.get("intent") == row["gold"]:
                example_same += 1
            else:
                example_diff += 1
    cross_only = 0
    for row in inputs["rows"]:
        keys = [canon(example["message"]) for example in row["examples"]]
        if any(key in eval_canon and key != canon(row["text"]) for key in keys) and not any(
            key == canon(row["text"]) for key in keys
        ):
            cross_only += 1
    counts = {
        "rows_with_any": len(example_rows),
        "instances": example_instances,
        "self_instances": example_self,
        "cross_instances": example_cross,
        "cross_rows": len(example_cross_rows),
        "cross_only_rows": cross_only,
        "same_label": example_same,
        "diff_label": example_diff,
        "exact": example_exact,
        "unique": len(example_eval_canon),
    }
    stored = summary["frozen_examples"]
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
        or stored["rows_with_cross_row_retrieval"] != 236
        or stored["rows_with_cross_only"] != 230
        or stored["instances_matching_an_evaluation_text"] != 283
        or not stored["example_set_equals_support_overlap"]
    ):
        raise SystemExit(f"frozen-example contamination counts changed: {counts}")

    # Rows with no evaluation text in any retrieved example are the only
    # genuinely unaffected rows.  The old self-excluded subset is not that set.
    unaffected = [
        int(row["i"]) for row in frozen
        if not any(canon(example["message"]) in eval_canon for example in inputs["rows"][int(row["i"])]["examples"])
    ]
    if len(unaffected) != 2_825:
        raise SystemExit(f"unaffected subset is {len(unaffected)} rows")
    unaffected_correct = sum(frozen_by[i] for i in unaffected)
    if unaffected_correct != 2_511:
        raise SystemExit(f"unaffected subset has {unaffected_correct} correct")

    clean_minus_old = mcnemar_exact(clean_vec, old_vec)
    clean_minus_frozen = mcnemar_exact(clean_vec, frozen_vec)
    if clean_minus_old != {"wins": 2, "losses": 1, "p": 1.0}:
        raise SystemExit(f"clean-vs-old McNemar changed: {clean_minus_old}")
    if clean_minus_frozen != clean_minus_old:
        raise SystemExit("clean-vs-frozen McNemar differs from clean-vs-old")
    context_changed = sum(
        int(a["context_sha256"] != b["context_sha256"]) for a, b in zip(clean, old)
    )
    prediction_same = sum(
        int(a["pred_index"] == b["pred_index"]) for a, b in zip(clean, old)
    )
    if context_changed != 256 or prediction_same != 3_077:
        raise SystemExit(f"clean/old context or prediction counts changed: {context_changed}/{prediction_same}")

    return {
        "summary": summary,
        "clean": clean,
        "old": old,
        "frozen": frozen,
        "support": support,
        "removed": removed,
        "counts": counts,
        "unaffected_n": len(unaffected),
        "unaffected_correct": unaffected_correct,
        "clean_correct": clean_c,
        "old_correct": old_c,
        "frozen_correct": frozen_c,
        "clean_minus_old": clean_minus_old,
        "context_changed": context_changed,
        "prediction_same": prediction_same,
        "same_context": same_context,
        "exact_removed": exact_removed,
        "support_same_label": support_same_label,
        "eval_canon": eval_canon,
    }


def macros(d: dict) -> list[str]:
    s = d["summary"]
    clean = s["clean"]
    c = d["counts"]
    lo, hi = wilson(d["clean_correct"], N_ROWS)
    ulo, uhi = wilson(d["unaffected_correct"], d["unaffected_n"])
    m = [
        r"\newcommand{\ContamSupportOriginal}{" + num(9_993) + "}",
        r"\newcommand{\ContamSupportClean}{" + num(9_968) + "}",
        r"\newcommand{\ContamSupportMatches}{" + num(len(d["removed"])) + "}",
        r"\newcommand{\ContamSupportExact}{" + num(d["exact_removed"]) + "}",
        r"\newcommand{\ContamSupportShare}{" + f"{100 * len(d['removed']) / 9_993:.2f}" + "}",
        r"\newcommand{\ContamSupportSameLabel}{" + num(d["support_same_label"]) + "}",
        r"\newcommand{\ContamEvalRows}{" + num(c["rows_with_any"]) + "}",
        r"\newcommand{\ContamEvalShare}{" + f"{100 * c['rows_with_any'] / N_ROWS:.2f}" + "}",
        r"\newcommand{\ContamInstances}{" + num(c["instances"]) + "}",
        r"\newcommand{\ContamSelf}{" + num(c["self_instances"]) + "}",
        r"\newcommand{\ContamCross}{" + num(c["cross_instances"]) + "}",
        r"\newcommand{\ContamCrossRows}{" + num(c["cross_rows"]) + "}",
        r"\newcommand{\ContamCrossOnlyRows}{" + num(c["cross_only_rows"]) + "}",
        r"\newcommand{\ContamSameLabel}{" + num(c["same_label"]) + "}",
        r"\newcommand{\ContamDiffLabel}{" + num(c["diff_label"]) + "}",
        r"\newcommand{\ContamExactExamples}{" + num(c["exact"]) + "}",
        r"\newcommand{\ContamUniqueExamples}{" + num(c["unique"]) + "}",
        r"\newcommand{\ContamCleanN}{" + num(N_ROWS) + "}",
        r"\newcommand{\ContamCleanCorrect}{" + num(d["clean_correct"]) + "}",
        r"\newcommand{\ContamCleanAcc}{" + pct(clean["accuracy"]) + "}",
        r"\newcommand{\ContamCleanLo}{" + pct(lo) + "}",
        r"\newcommand{\ContamCleanHi}{" + pct(hi) + "}",
        r"\newcommand{\ContamCleanDiff}{" + pp(clean["accuracy"] - d["old_correct"] / N_ROWS) + "}",
        r"\newcommand{\ContamCleanDiffAbs}{" + f"{abs(clean['accuracy'] - d['old_correct'] / N_ROWS) * 100:.2f}" + "}",
        r"\newcommand{\ContamCleanWins}{" + num(d["clean_minus_old"]["wins"]) + "}",
        r"\newcommand{\ContamCleanLosses}{" + num(d["clean_minus_old"]["losses"]) + "}",
        r"\newcommand{\ContamCleanPText}{p = 1.00}",
        r"\newcommand{\ContamUnaffectedN}{" + num(d["unaffected_n"]) + "}",
        r"\newcommand{\ContamUnaffectedCorrect}{" + num(d["unaffected_correct"]) + "}",
        r"\newcommand{\ContamUnaffectedAcc}{" + pct(d["unaffected_correct"] / d["unaffected_n"]) + "}",
        r"\newcommand{\ContamUnaffectedLo}{" + pct(ulo) + "}",
        r"\newcommand{\ContamUnaffectedHi}{" + pct(uhi) + "}",
        r"\newcommand{\ContamCleanContextChanged}{" + num(d["context_changed"]) + "}",
        r"\newcommand{\ContamCleanPredSame}{" + num(d["prediction_same"]) + "}",
        r"\newcommand{\ContamProductFrozenSameContext}{" + num(d["same_context"]) + "}",
        r"\newcommand{\ContamProductFrozenDiscordant}{0}",
    ]
    for name in m:
        if not re.match(r"\\newcommand\{\\[A-Za-z]+\}", name):
            raise SystemExit(f"bad macro: {name}")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    data = load()
    text = (
        "% generated by thesis/gen_contamination.py — do not edit by hand\n"
        + "\n".join(macros(data))
        + "\n"
    )
    if args.check:
        print(
            f"clean {data['clean_correct']}/{N_ROWS}; unaffected "
            f"{data['unaffected_correct']}/{data['unaffected_n']}; "
            f"removed {len(data['removed'])}"
        )
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "macros_contam.tex").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / 'macros_contam.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
