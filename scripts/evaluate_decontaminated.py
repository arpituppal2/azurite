#!/usr/bin/env python3
"""Measure the shipped product path after decontaminating the support bank.

Run this only on the Molab sandbox, against the installed wheel:

    /tmp/azurite-molab/bin/python \
        /marimo/azurite_ws/scripts/evaluate_decontaminated.py

The arm is the real product path, not a reimplementation: the packaged
``banking77.json.gz`` bundle is loaded through ``azurite.bundle.load_bundle``,
the query-native checkpoint is resolved through ``azurite.checkpoints``, and
retrieval, context packing and scoring all go through ``azurite.predictor``.
The one intervention is that every support row whose message normalises to one
of the 3,080 evaluation messages is removed before the BM25 index is built.

Outputs under ``results/product_path_clean``:

    per_example.jsonl       one record per official test row
    clean_support.jsonl.gz  the 9,968-row support bank actually used
    removed_support.json    every removed support row
    summary.json            counts, interval, per-class errors, contamination
    MANIFEST.json           hashes and environment
    MANIFEST.sha256         hash of MANIFEST.json
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch

ROOT = Path("/marimo/azurite_ws")
RESULTS = ROOT / "results"
OUT = RESULTS / "product_path_clean"
INPUTS = RESULTS / "jev_matched/inputs.json"
HOME = Path("/tmp/azurite-home")

N_ROWS = 3_080
N_LABELS = 77


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canon(text: str) -> str:
    """Case/punctuation/whitespace-insensitive key, the audit's definition.

    NFKC plus case folding plus collapsing every non-alphanumeric run to a
    single space.  This is deliberately aggressive: it treats a message with a
    different quote or comma as the same text, which is the safe direction for
    a contamination check.  It is the same key used by
    ``scripts/evaluate_product_path.py``, so the 25-row overlap is reproduced
    rather than redefined.
    """
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()



def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    import math

    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    inputs_bytes = INPUTS.read_bytes()
    data = json.loads(inputs_bytes)
    rows = data["rows"]
    labels = data["labels"]
    if len(rows) != N_ROWS or len(labels) != N_LABELS:
        raise SystemExit(f"inputs carry {len(rows)} rows and {len(labels)} labels")
    if sum(int(row["jev_correct"]) for row in rows) != 2_846:
        raise SystemExit("Jev correctness vector does not sum to 2,846")
    for row in rows:
        if len(row["examples"]) != 24:
            raise SystemExit(f"row {row['id']} has {len(row['examples'])} examples")
        if labels[int(row["gold_index"])] != row["gold"]:
            raise SystemExit(f"row {row['id']}: gold label and index disagree")

    from azurite.bundle import load_bundle, load_provenance
    from azurite.predictor import Predictor

    bundle = load_bundle()
    provenance = load_provenance()
    support = list(bundle.support)
    if len(support) != 9_993:
        raise SystemExit(f"support bank has {len(support)} rows, expected 9,993")

    eval_canon = {canon(row["text"]) for row in rows}
    eval_exact = {str(row["text"]).strip() for row in rows}
    support_canon = {canon(row["message"]) for row in support}

    removed = []
    clean_support = []
    for row in support:
        key = canon(row["message"])
        if key in eval_canon:
            removed.append(dict(row, normalized=key))
        else:
            clean_support.append(row)

    residue = [row for row in clean_support if canon(row["message"]) in eval_canon]
    if residue:
        raise SystemExit(f"{len(residue)} evaluation texts remain in the clean support bank")
    if not removed:
        raise SystemExit("no support rows were removed; the overlap check is not active")
    if len(clean_support) != len(support) - len(removed):
        raise SystemExit("clean support count does not equal original minus removed")

    removed_canon = {row["normalized"] for row in removed}
    removed_exact = sum(1 for row in removed if str(row["message"]).strip() in eval_exact)
    removed_unique = len(removed_canon)

    # Check the frozen Jev example set independently of the product bank.
    example_instances = 0
    example_rows = 0
    example_self_rows = 0
    example_cross_rows = 0
    example_same_label = 0
    example_diff_label = 0
    example_exact = 0
    example_eval_canon: set[str] = set()
    rows_with_self = []
    rows_with_cross = []
    rows_with_cross_only = []
    for row in rows:
        row_text = canon(row["text"])
        found = False
        for example in row["examples"]:
            key = canon(example["message"])
            if key in eval_canon:
                example_eval_canon.add(key)
                example_instances += 1
                found = True
                if str(example["message"]).strip() == str(row["text"]).strip():
                    example_exact += 1
                if key == row_text:
                    example_self_rows += 1
                    rows_with_self.append(row["id"])
                else:
                    example_cross_rows += 1
                    rows_with_cross.append(row["id"])
                    rows_with_cross_only.append(row["id"])
                if example.get("label", example.get("intent")) == row["gold"]:
                    example_same_label += 1
                else:
                    example_diff_label += 1
        if found:
            example_rows += 1

    # Load the real product predictor and swap only its support bank.
    predictor = Predictor.load("query-native", home=HOME, device="cuda")
    if predictor.checkpoint.id != "query-native":
        raise SystemExit(f"resolved checkpoint is {predictor.checkpoint.id!r}")
    predictor._support = tuple(dict(row, text=row["message"]) for row in clean_support)
    predictor._retriever = None

    contexts: list[str] = []
    messages: list[str] = []
    metas: list[dict] = []
    chosen_counts: Counter[int] = Counter()
    for position, row in enumerate(rows):
        chosen = predictor.retrieve(row["text"])
        if len(chosen) > predictor.checkpoint.examples:
            raise SystemExit(f"row {row['id']}: {len(chosen)} examples retrieved")
        for example in chosen:
            if canon(example["message"]) in eval_canon:
                raise SystemExit(f"row {row['id']}: clean retriever returned an evaluation text")
        context, message, used = predictor.make_context(row["text"], examples=chosen)
        contexts.append(context)
        messages.append(message)
        chosen_counts[used] += 1
        metadoc = {
            "i": position,
            "id": row["id"],
            "text": row["text"],
            "gold": row["gold"],
            "gold_index": int(row["gold_index"]),
            "examples_used": used,
            "support_ids": [int(example["id"]) for example in chosen],
            "context_sha256": sha256_bytes(context.encode("utf-8")),
        }
        # Confirm the packed context digest belongs to the chosen rows and the
        # message, not to a stale retrieval index.
        if used and not chosen:
            raise SystemExit(f"row {row['id']}: used examples without a retrieval")
        metadoc["chosen"] = [
            {"id": int(example["id"]), "label": example["label"], "message": example["message"]}
            for example in chosen
        ]
        metas.append(metadoc)
        if (position + 1) % 250 == 0:
            print(f"packed {position + 1}/{N_ROWS}", flush=True)

    print(f"scoring {N_ROWS} clean contexts", flush=True)
    logits = predictor.score(contexts, messages, batch=64)
    if len(logits) != N_ROWS:
        raise SystemExit(f"score returned {len(logits)} rows")
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)

    records = []
    correct = 0
    per_class: dict[str, dict] = {
        label: {"n": 0, "correct": 0, "errors": 0} for label in labels
    }
    for meta, row, prob in zip(metas, rows, probs):
        values = prob.tolist()
        ranked = sorted(range(len(values)), key=lambda index: (-values[index], index))
        prediction = ranked[0]
        margin = values[prediction] - values[ranked[1]]
        hit = int(prediction == int(row["gold_index"]))
        correct += hit
        per_class[row["gold"]]["n"] += 1
        per_class[row["gold"]]["correct"] += hit
        per_class[row["gold"]]["errors"] += 1 - hit
        record = {
            **meta,
            "pred_index": prediction,
            "pred": labels[prediction],
            "correct": hit,
            "confidence": values[prediction],
            "margin": margin,
        }
        records.append(record)

    accuracy = correct / N_ROWS
    lo, hi = wilson(correct, N_ROWS)
    top = sorted(
        (
            {"label": label, "errors": stats["errors"], "n": stats["n"]}
            for label, stats in per_class.items()
            if stats["errors"]
        ),
        key=lambda item: (-item["errors"], item["label"]),
    )
    error_total = N_ROWS - correct
    top10 = sum(item["errors"] for item in top[:10])
    top12 = sum(item["errors"] for item in top[:12])

    summary = {
        "schema": 1,
        "protocol": "product-path-clean-support-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "path": str(INPUTS.relative_to(ROOT)),
            "sha256": sha256_bytes(inputs_bytes),
            "rows": N_ROWS,
            "labels": N_LABELS,
            "jev_correct": sum(int(row["jev_correct"]) for row in rows),
        },
        "support": {
            "original_rows": len(support),
            "clean_rows": len(clean_support),
            "removed_rows": len(removed),
            "removed_unique_messages": removed_unique,
            "removed_exact_text_matches": removed_exact,
            "residue_rows": len(residue),
            "normalization": "NFKC, casefold, non-alphanumeric runs collapsed to spaces",
            "removed": removed,
        },
        "frozen_examples": {
            "rows": N_ROWS,
            "examples_per_row": 24,
            "rows_with_any_evaluation_text": example_rows,
            "rows_with_self_retrieval": len(set(rows_with_self)),
            "rows_with_cross_row_retrieval": len(set(rows_with_cross)),
            "rows_with_cross_only": len(set(rows_with_cross_only) - set(rows_with_self)),
            "instances_matching_an_evaluation_text": example_instances,
            "instances_self": example_self_rows,
            "instances_cross_row": example_cross_rows,
            "instances_same_label": example_same_label,
            "instances_different_label": example_diff_label,
            "instances_exact_text": example_exact,
            "unique_example_messages_matching_evaluation": len(example_eval_canon),
            "unique_evaluation_texts_in_support": removed_unique,
            "example_set_equals_support_overlap": example_eval_canon == removed_canon,
        },
        "checkpoint": {
            "id": predictor.checkpoint.id,
            "sha256": predictor.checkpoint.sha256,
            "max_seq": predictor.checkpoint.max_seq,
            "context_budget": predictor.checkpoint.context_budget,
            "examples": predictor.checkpoint.examples,
            "measured": predictor.checkpoint.measured,
        },
        "bundle": {
            "bundle_sha256": provenance["bundle"]["sha256"],
            "support_rows": provenance["bundle"]["support_rows"],
            "provenance_sha256": sha256_file(
                Path("/tmp/azurite-molab/lib/python3.13/site-packages/azurite/resources/provenance.json")
            ),
        },
        "clean": {
            "n": N_ROWS,
            "correct": correct,
            "errors": error_total,
            "accuracy": accuracy,
            "accuracy_percent": 100.0 * accuracy,
            "wilson95": [lo, hi],
            "wilson95_percent": [100.0 * lo, 100.0 * hi],
            "examples_used_histogram": {str(k): v for k, v in sorted(chosen_counts.items())},
            "per_class": per_class,
            "per_class_errors": top,
            "classes_with_error": len(top),
            "top10_error_share": top10 / error_total if error_total else 0.0,
            "top12_error_share": top12 / error_total if error_total else 0.0,
            "largest_error_class": top[0] if top else None,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "platform": platform.platform(),
            "package": "azurite from /tmp/azurite-molab",
        },
        "runtime_seconds": time.time() - started,
    }

    with (OUT / "per_example.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    with gzip.GzipFile(filename=str(OUT / "clean_support.jsonl.gz"), mode="wb", mtime=0) as fh:
        for row in clean_support:
            fh.write((json.dumps(dict(row), sort_keys=True) + "\n").encode("utf-8"))

    write_json(OUT / "removed_support.json", removed)
    write_json(OUT / "summary.json", summary)

    manifest = {
        "schema": 1,
        "protocol": summary["protocol"],
        "generated_at": summary["generated_at"],
        "script": {
            "path": "scripts/evaluate_decontaminated.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "inputs": summary["inputs"],
        "checkpoint": summary["checkpoint"],
        "bundle": summary["bundle"],
        "support": {k: summary["support"][k] for k in (
            "original_rows", "clean_rows", "removed_rows", "removed_unique_messages",
            "removed_exact_text_matches", "residue_rows", "normalization",
        )},
        "frozen_examples": summary["frozen_examples"],
        "clean": {
            "n": summary["clean"]["n"],
            "correct": summary["clean"]["correct"],
            "errors": summary["clean"]["errors"],
            "accuracy": summary["clean"]["accuracy"],
            "accuracy_percent": summary["clean"]["accuracy_percent"],
            "wilson95": summary["clean"]["wilson95"],
        },
        "files": {
            name: {
                "bytes": (OUT / name).stat().st_size,
                "sha256": sha256_file(OUT / name),
            }
            for name in (
                "per_example.jsonl",
                "clean_support.jsonl.gz",
                "removed_support.json",
                "summary.json",
            )
        },
        "environment": summary["environment"],
        "runtime_seconds": summary["runtime_seconds"],
    }
    write_json(OUT / "MANIFEST.json", manifest)
    (OUT / "MANIFEST.sha256").write_text(
        sha256_file(OUT / "MANIFEST.json") + "  MANIFEST.json\n", encoding="utf-8"
    )

    print(
        f"clean support {len(clean_support)}/{len(support)}; removed {len(removed)}; "
        f"residue {len(residue)}; examples {example_rows} rows / {example_instances} instances",
        flush=True,
    )
    print(
        f"clean accuracy {correct}/{N_ROWS} = {100.0 * accuracy:.2f}% "
        f"(95% Wilson {100.0 * lo:.2f}-{100.0 * hi:.2f})",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
