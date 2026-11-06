#!/usr/bin/env python3
"""Measure the installed product path on all 3,080 official Banking77 rows.

This is deliberately *not* a reimplementation of the measured arm.  It imports
the installed `azurite` package, loads its content-addressed checkpoint through
`Predictor.load`, lets the packaged BM25 index retrieve examples from the
9,993-message support split, packs the context with the shipped packer, and
takes the argmax of the shipped 77-way head.  A sample is replayed through the
actual `azurite predict --json` command, and the batch records are refused if
that command disagrees.

The script also checks whether the support bank contains any evaluation
messages under exact or punctuation/case/whitespace-normalised text.  Banking77
itself contains a small number of cross-split duplicates; if a match exists it
is reported here rather than hidden.  The ``self_retrieval_free`` field excludes only the
rows whose own text is in the support bank; it is **not** a fully clean subset,
because another 230 rows retrieve a different test row.  Use
``scripts/evaluate_decontaminated.py`` for the clean support-bank rerun.

Run it against the installed wheel, from outside the repository:

    cd /tmp
    env -u PYTHONPATH AZURITE_HOME=/tmp/azurite-home2 \\
      /tmp/azurite-fresh/bin/python \\
      /path/to/repo/scripts/evaluate_product_path.py --device mps

Outputs under `results/product_path/`:

    per_example_bm25.jsonl   one record per official test row
    summary.json             accuracy, Wilson interval, per-class errors,
                             frozen-arm comparison, overlap audit, CLI parity
    MANIFEST.json            source/output hashes and runtime identity
    MANIFEST.sha256          one line per file, for direct verification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "product_path"
INPUTS = ROOT / "results" / "jev_matched" / "inputs.json"
SUPPORT = ROOT / "results" / "jev_query" / "support.jsonl"
FROZEN = ROOT / "results" / "jev_query" / "eval" / "per_example_query_a2c.jsonl"
FROZEN_SUMMARY = ROOT / "results" / "jev_query" / "eval" / "summary.json"
PROTOCOL = "product-bm25-banking77-v1"

import azurite  # noqa: E402
from azurite import Predictor  # noqa: E402
from azurite.artifacts import sha256_text  # noqa: E402
from azurite.stats import mcnemar_exact, wilson  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(text: str) -> str:
    """Case/punctuation/whitespace-insensitive form for the overlap audit."""
    value = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def overlap_audit(rows: list[dict], support: list[dict]) -> dict:
    by_canonical: dict[str, list[dict]] = defaultdict(list)
    by_exact: dict[str, list[dict]] = defaultdict(list)
    for row in support:
        by_canonical[canonical(str(row["message"]))].append(row)
        by_exact[str(row["message"]).strip()].append(row)

    matches = []
    exact = 0
    conflicting = []
    for row in rows:
        text = str(row["text"])
        exact_hits = by_exact.get(text.strip(), [])
        canonical_hits = by_canonical.get(canonical(text), [])
        if exact_hits:
            exact += 1
        if not canonical_hits:
            continue
        same = [hit for hit in canonical_hits if hit["label"] == row["gold"]]
        if len(same) != len(canonical_hits):
            conflicting.append(str(row["id"]))
        matches.append(
            {
                "id": str(row["id"]),
                "gold": row["gold"],
                "support_ids": sorted(int(hit["id"]) for hit in canonical_hits),
                "support_labels": sorted({str(hit["label"]) for hit in canonical_hits}),
                "exact_text_match": bool(exact_hits),
                "same_gold_label": bool(same),
            }
        )

    ids_overlap = sorted(
        set(str(row["id"]) for row in rows) & set(str(row["id"]) for row in support)
    )
    return {
        "support_rows": len(support),
        "eval_rows": len(rows),
        "exact_text_matches": exact,
        "canonical_matches": len(matches),
        "canonical_match_share": len(matches) / max(1, len(rows)),
        "id_overlaps": len(ids_overlap),
        "label_conflicts": conflicting,
        "matches": matches,
        "matched_ids": [item["id"] for item in matches],
    }


def class_table(rows: list[dict]) -> list[dict]:
    table = []
    for label in sorted({str(row["gold"]) for row in rows}):
        subset = [row for row in rows if row["gold"] == label]
        correct = sum(int(row["correct"]) for row in subset)
        table.append(
            {
                "label": label,
                "n": len(subset),
                "correct": correct,
                "errors": len(subset) - correct,
                "accuracy": correct / len(subset),
            }
        )
    return sorted(table, key=lambda item: (-item["errors"], item["label"]))


def subset_stats(records: list[dict], keep: set[str]) -> dict:
    part = [row for row in records if str(row["id"]) in keep]
    k = sum(int(row["correct"]) for row in part)
    lo, hi = wilson(k, len(part))
    return {
        "n": len(part),
        "correct": k,
        "accuracy": k / len(part),
        "wilson95": [lo, hi],
    }


def paired(product: list[dict], frozen: list[dict], label: str) -> dict:
    left = np.asarray([row["correct"] for row in product], dtype=np.int8)
    right = np.asarray([row["correct"] for row in frozen], dtype=np.int8)
    test = mcnemar_exact(left, right)
    return {
        "against": label,
        "product_accuracy": float(left.mean()),
        "comparison_accuracy": float(right.mean()),
        "diff": float(left.mean() - right.mean()),
        "both_correct": int(((left == 1) & (right == 1)).sum()),
        "product_only": int(((left == 1) & (right == 0)).sum()),
        "comparison_only": int(((left == 0) & (right == 1)).sum()),
        "both_wrong": int(((left == 0) & (right == 0)).sum()),
        "mcnemar_exact": test,
    }


def cli_parity(
    rows: list[dict], records: list[dict], *, device: str, n: int
) -> dict:
    if n <= 0:
        return {"n": 0, "checked": [], "matches": 0}
    indexes = sorted({round(i * (len(rows) - 1) / (n - 1)) for i in range(n)})
    by_id = {str(row["id"]): row for row in records}
    checked = []
    for index in indexes:
        row = rows[index]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "azurite",
                "predict",
                str(row["text"]),
                "--json",
                "--device",
                device,
                "--checkpoint",
                "query-native",
            ],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"CLI parity row {row['id']} failed ({proc.returncode}): "
                f"{proc.stderr[-1000:]}"
            )
        payload = json.loads(proc.stdout)
        record = by_id[str(row["id"])]
        matches = {
            "intent": payload["intent"] == record["pred"],
            "intent_index": int(payload["intent_index"]) == int(record["pred_index"]),
            "context_sha256": payload["context_sha256"] == record["context_sha256"],
            "examples_used": int(payload["examples_used"]) == int(record["examples_used"]),
            "checkpoint_sha256": payload["checkpoint_sha256"] == record["checkpoint_sha256"],
        }
        checked.append(
            {
                "id": str(row["id"]),
                "intent": payload["intent"],
                "stored_intent": record["pred"],
                "context_sha256": payload["context_sha256"],
                "stored_context_sha256": record["context_sha256"],
                "examples_used": int(payload["examples_used"]),
                "stored_examples_used": int(record["examples_used"]),
                "matches": matches,
                "all_match": all(matches.values()),
            }
        )
        print(f"  CLI parity {row['id']}: {'ok' if checked[-1]['all_match'] else 'MISMATCH'}", flush=True)
    return {
        "n": len(checked),
        "matches": sum(int(item["all_match"]) for item in checked),
        "checked": checked,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=INPUTS, type=Path)
    ap.add_argument("--support", default=SUPPORT, type=Path)
    ap.add_argument("--frozen", default=FROZEN, type=Path)
    ap.add_argument("--out-dir", default=OUT, type=Path)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cli-parity", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="smoke only; never report")
    args = ap.parse_args()

    package_file = Path(azurite.__file__).resolve()
    if package_file.is_relative_to(ROOT):
        raise SystemExit(
            f"imported azurite from the repository ({package_file}); "
            "run with the installed wheel's Python from outside the repo"
        )

    data = json.loads(args.inputs.read_text())
    rows = data["rows"]
    if args.limit:
        rows = rows[: args.limit]
    if data.get("task") != "banking77" or (not args.limit and len(rows) != 3_080):
        raise SystemExit(f"expected 3,080 Banking77 rows, got {len(rows)}")
    support = load_jsonl(args.support)
    frozen_records = load_jsonl(args.frozen)
    frozen_summary = json.loads(FROZEN_SUMMARY.read_text())
    wanted = {str(row["id"]) for row in rows}
    frozen_records = [row for row in frozen_records if str(row["id"]) in wanted]
    if len(frozen_records) != len(rows):
        raise SystemExit("frozen records do not cover the evaluation rows")
    frozen = {str(row["id"]): row for row in frozen_records}
    if set(frozen) != {str(row["id"]) for row in rows}:
        raise SystemExit("frozen record ids differ from the official row ids")

    overlap = overlap_audit(rows, support)
    print(
        f"overlap: {overlap['canonical_matches']}/{len(rows)} eval rows match "
        f"support under normalised text ({overlap['exact_text_matches']} exact)",
        flush=True,
    )

    predictor = Predictor.load("query-native", device=args.device)
    bundle = predictor.bundle
    checkpoint = predictor.checkpoint
    if len(bundle.support) != len(support):
        raise SystemExit("installed bundle support count differs from the source split")
    provenance = json.loads(
        (package_file.parent / "resources" / "provenance.json").read_text()
    )
    if provenance["sources"]["labels_and_eval_rows"]["sha256"] != sha256_file(args.inputs):
        raise SystemExit("installed bundle provenance does not match the evaluation inputs")
    if provenance["sources"]["support"]["sha256"] != sha256_file(args.support):
        raise SystemExit("installed bundle provenance does not match the source support")

    started = time.perf_counter()
    contexts: list[str] = []
    messages: list[str] = []
    chosen_ids: list[list[int]] = []
    used_counts: list[int] = []
    for n, row in enumerate(rows, 1):
        text = str(row["text"])
        chosen = predictor.retrieve(text)
        context, message, used = predictor.make_context(text, examples=chosen)
        contexts.append(context)
        messages.append(message)
        chosen_ids.append([int(item["id"]) for item in chosen])
        used_counts.append(used)
        if n == 1 or n % 500 == 0:
            print(f"  contexts {n}/{len(rows)} ({time.perf_counter() - started:.1f}s)", flush=True)

    scores: list[list[float]] = []
    batches = 0
    for start in range(0, len(rows), args.batch):
        stop = start + args.batch
        scores.extend(
            predictor.score(contexts[start:stop], messages[start:stop], batch=args.batch)
        )
        batches += 1
        if predictor.device.type == "mps" and batches % 32 == 0:
            # Batch 32 on unified memory was killed late in the run; release
            # the cached MPS blocks between chunks instead of raising peak use.
            torch.mps.empty_cache()
        if start == 0 or batches % 40 == 0:
            print(
                f"  scored {min(stop, len(rows))}/{len(rows)} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )

    records = []
    for row, context, used, support_ids, logits in zip(
        rows, contexts, used_counts, chosen_ids, scores
    ):
        pred = int(max(range(len(logits)), key=lambda i: logits[i]))
        gold = int(row["gold_index"])
        records.append(
            {
                "protocol": PROTOCOL,
                "id": str(row["id"]),
                "i": int(row["i"]),
                "text_sha256": sha256_text(str(row["text"])),
                "gold": str(row["gold"]),
                "gold_index": gold,
                "pred": bundle.labels[pred],
                "pred_index": pred,
                "correct": int(pred == gold),
                "context_sha256": sha256_text(context),
                "context_bytes": len(context.encode("utf-8")),
                "examples_used": int(used),
                "support_ids": support_ids,
                "checkpoint": checkpoint.id,
                "checkpoint_sha256": checkpoint.sha256,
                "device": str(predictor.device),
            }
        )

    # The single-call product API must agree with the batched run on a sample.
    single = {}
    for index in sorted({round(i * (len(rows) - 1) / 15) for i in range(16)}):
        prediction = predictor.predict(str(rows[index]["text"]))
        record = records[index]
        single[str(rows[index]["id"])] = {
            "intent": prediction.intent,
            "intent_index": prediction.intent_index,
            "context_sha256": prediction.to_dict()["context_sha256"],
            "examples_used": prediction.examples_used,
            "matches_batch": (
                prediction.intent == record["pred"]
                and prediction.intent_index == record["pred_index"]
                and prediction.to_dict()["context_sha256"] == record["context_sha256"]
                and prediction.examples_used == record["examples_used"]
            ),
        }
    if not all(item["matches_batch"] for item in single.values()):
        raise SystemExit("batched records disagree with Predictor.predict on the sample")

    parity = cli_parity(rows, records, device=str(predictor.device), n=args.cli_parity)
    if parity["matches"] != parity["n"]:
        raise SystemExit(
            f"CLI parity {parity['matches']}/{parity['n']}; refusing to write records"
        )

    product_correct = np.asarray([row["correct"] for row in records], dtype=np.int8)
    frozen_correct = np.asarray(
        [int(frozen[str(row["id"])]["correct"]) for row in rows], dtype=np.int8
    )
    if not args.limit and int(frozen_correct.sum()) != int(
        frozen_summary["arms"]["query_a2c"]["correct"]
    ):
        raise SystemExit("frozen correctness does not match its summary")
    if int(product_correct.sum()) != sum(int(row["correct"]) for row in records):
        raise SystemExit("record correctness is inconsistent")

    k = int(product_correct.sum())
    lo, hi = wilson(k, len(records))
    matched = set(overlap["matched_ids"])
    self_retrieval_free_ids = {str(row["id"]) for row in rows} - matched
    self_retrieval_free = subset_stats(records, self_retrieval_free_ids)
    self_retrieval_free_frozen = [
        {"id": row["id"], "correct": int(frozen[str(row["id"])]["correct"])}
        for row in rows
        if str(row["id"]) in self_retrieval_free_ids
    ]
    self_retrieval_free_product = [
        row for row in records if str(row["id"]) in self_retrieval_free_ids
    ]
    matched_rows = [row for row in records if str(row["id"]) in matched]
    matched_stats = {
        "n": len(matched_rows),
        "correct": sum(int(row["correct"]) for row in matched_rows),
        "accuracy": (
            sum(int(row["correct"]) for row in matched_rows) / len(matched_rows)
            if matched_rows
            else None
        ),
    }
    jev_correct = np.asarray([int(row["jev_correct"]) for row in rows], dtype=np.int8)

    frozen_stats = {
        "arm": "query_a2c",
        "n": int(len(frozen_correct)),
        "correct": int(frozen_correct.sum()),
        "accuracy": float(frozen_correct.mean()),
        "wilson95": list(wilson(int(frozen_correct.sum()), len(frozen_correct))),
    }
    same_context = sum(
        int(record["context_sha256"] == frozen[str(record["id"])]["context_sha256"])
        for record in records
    )
    same_prediction = sum(
        int(record["pred_index"] == int(frozen[str(record["id"])]["pred_index"]))
        for record in records
    )
    summary = {
        "schema": 1,
        "protocol": PROTOCOL,
        "task": "banking77",
        "n": len(records),
        "labels": len(bundle.labels),
        "checkpoint": {
            "id": checkpoint.id,
            "sha256": checkpoint.sha256,
            "context_budget": checkpoint.context_budget,
            "max_seq": checkpoint.max_seq,
        },
        "bundle": {
            "sha256": provenance["bundle"]["sha256"],
            "support_rows": len(bundle.support),
            "source_support_sha256": provenance["sources"]["support"]["sha256"],
        },
        "accuracy": k / len(records),
        "correct": k,
        "wilson95": [lo, hi],
        "per_class": class_table(records),
        "overlap": overlap,
        "matched_rows": matched_stats,
        "self_retrieval_free": {
            **self_retrieval_free,
            "note": (
                "self-retrieval-free only: excludes rows whose own normalised text "
                "occurs in the support bank; 230 rows still retrieve another test "
                "row. Not a fully clean subset; see evaluate_decontaminated.py."
            ),
        },
        "frozen_arm": frozen_stats,
        "comparison_frozen": paired(records, [
            {"correct": int(frozen[str(row["id"])]["correct"])} for row in rows
        ], "query_a2c frozen-example arm"),
        "frozen_context_identity": {
            "same_context_sha256": same_context,
            "different_context_sha256": len(records) - same_context,
            "same_prediction_index": same_prediction,
            "note": (
                "the packaged BM25 retriever reproduces the frozen example "
                "contexts for most rows; this is not an independent retrieval arm"
            ),
        },
        "comparison_frozen_self_retrieval_free": paired(
            self_retrieval_free_product,
            self_retrieval_free_frozen,
            "query_a2c frozen-example arm on self-retrieval-free rows",
        ),
        "comparison_jev": paired(records, [
            {"correct": int(row["jev_correct"])} for row in rows
        ], "Jev stored correctness"),
        "cli_parity": {"n": parity["n"], "matches": parity["matches"]},
        "single_call_check": {
            "n": len(single),
            "matches": sum(int(item["matches_batch"]) for item in single.values()),
        },
        "retrieval": {
            "mean_examples_used": float(np.mean(used_counts)),
            "min_examples_used": int(min(used_counts)),
            "max_examples_used": int(max(used_counts)),
            "mean_context_bytes": float(
                np.mean([row["context_bytes"] for row in records])
            ),
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "device": str(predictor.device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "per_example_bm25.jsonl"
    with records_path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")

    try:
        dist = distribution("azurite-encoder")
        package_version = dist.version
        package_location = str(dist.locate_file(""))
    except PackageNotFoundError:
        package_version = version("azurite-encoder")
        package_location = str(package_file.parent)
    try:
        wheel = next((ROOT / "dist" / "pkg").glob("azurite_encoder-*.whl"))
        wheel_sha = sha256_file(wheel)
        wheel_name = wheel.name
    except StopIteration:
        wheel_sha = None
        wheel_name = None

    manifest = {
        "schema": 1,
        "protocol": PROTOCOL,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package": {
            "name": "azurite-encoder",
            "version": package_version,
            "file": str(package_file),
            "location": package_location,
            "wheel": wheel_name,
            "wheel_sha256": wheel_sha,
        },
        "checkpoint": {
            "id": checkpoint.id,
            "sha256": checkpoint.sha256,
            "path": str(azurite.checkpoints.resolve("query-native").path),
            "file_sha256": sha256_file(azurite.checkpoints.resolve("query-native").path),
        },
        "sources": {
            "inputs": {"path": str(args.inputs), "sha256": sha256_file(args.inputs)},
            "support": {"path": str(args.support), "sha256": sha256_file(args.support)},
            "frozen_records": {"path": str(args.frozen), "sha256": sha256_file(args.frozen)},
            "frozen_summary": {
                "path": str(FROZEN_SUMMARY),
                "sha256": sha256_file(FROZEN_SUMMARY),
            },
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        },
        "outputs": {
            "per_example": {
                "path": str(records_path),
                "sha256": sha256_file(records_path),
                "rows": len(records),
            },
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
        "summary": {
            "accuracy": summary["accuracy"],
            "correct": summary["correct"],
            "wilson95": summary["wilson95"],
            "overlap_matches": overlap["canonical_matches"],
            "self_retrieval_free_accuracy": summary["self_retrieval_free"]["accuracy"],
            "frozen_accuracy": summary["frozen_arm"]["accuracy"],
            "diff_vs_frozen": summary["comparison_frozen"]["diff"],
            "mcnemar_p": summary["comparison_frozen"]["mcnemar_exact"]["p"],
            "same_context_sha256": same_context,
            "same_prediction_index": same_prediction,
        },
        "runtime": summary["runtime"],
    }
    manifest_path = args.out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    lines = []
    for path in (records_path, summary_path, manifest_path):
        lines.append(f"{sha256_file(path)}  ./{path.name}")
    (args.out_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

    print(
        f"product accuracy {k}/{len(records)} = {100 * k / len(records):.2f}% "
        f"(95% Wilson {100 * lo:.2f}-{100 * hi:.2f})",
        flush=True,
    )
    print(
        f"frozen arm {frozen_stats['correct']}/{frozen_stats['n']} = "
        f"{100 * frozen_stats['accuracy']:.2f}%; "
        f"paired diff {100 * summary['comparison_frozen']['diff']:+.2f} pp; "
        f"McNemar p={summary['comparison_frozen']['mcnemar_exact']['p']:.3g}",
        flush=True,
    )
    print(
        f"self-retrieval-excluded (not clean) {self_retrieval_free['correct']}/{self_retrieval_free['n']} = "
        f"{100 * self_retrieval_free['accuracy']:.2f}% "
        f"({overlap['canonical_matches']} support/eval duplicates excluded)",
        flush=True,
    )
    print(f"wrote {records_path}, {summary_path}, {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
