#!/usr/bin/env python3
"""Build the protocol-matched Banking77 input record from the public Jev run.

This script does not load a model. It fixes the row set, the gold labels and the
exact 24 labelled examples Jev received for each test row, then writes one
deterministic JSON record that the Molab evaluator consumes. The row set is the
public run's 3,080 official BANKING77 test rows; no row is filtered or
deduplicated.

Usage:
    python3 scripts/jev_matched_inputs.py \
        --public-root /tmp/jev-banking77-experiment \
        --out results/jev_matched/inputs.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.decisions import TASKS, candidate_bank  # noqa: E402

PROTOCOL = "jev-matched-banking77-inputs-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-root", required=True, type=Path)
    ap.add_argument("--out", default=ROOT / "results/jev_matched/inputs.json", type=Path)
    args = ap.parse_args()

    public = args.public_root
    tests_path = public / "data/test_inputs.json"
    preds_path = public / "results/predictions.csv"
    requests_dir = public / "runs/test-v1"

    tests = json.loads(tests_path.read_text())
    preds = list(csv.DictReader(preds_path.open()))
    if len(tests) != 3080 or len(preds) != 3080:
        raise SystemExit(f"expected 3,080 rows, got tests={len(tests)} preds={len(preds)}")
    if [t["id"] for t in tests] != [p["id"] for p in preds]:
        raise SystemExit("test_inputs.json and predictions.csv are not in the same id order")

    spec = TASKS["banking77"]
    labels = candidate_bank(spec).labels
    label_index = {label: i for i, label in enumerate(labels)}
    if len(labels) != 77 or len(label_index) != 77:
        raise SystemExit(f"expected 77 unique labels, got {len(labels)}")

    rows = []
    truth_counts: dict[str, int] = {}
    jev_correct = 0
    for i, (test, pred) in enumerate(zip(tests, preds)):
        request_path = requests_dir / test["id"] / "request.json"
        if not request_path.exists():
            raise SystemExit(f"missing request: {request_path}")
        request = json.loads(request_path.read_text())
        message = request["state"].get("customer_message")
        if message != test["text"]:
            raise SystemExit(f"{test['id']}: request message does not match test_inputs")
        examples = request["state"].get("labeled_examples", [])
        if len(examples) != 24:
            raise SystemExit(f"{test['id']}: expected 24 examples, got {len(examples)}")
        if request["questions"]["intent"].get("criteria") is None:
            raise SystemExit(f"{test['id']}: missing intent criteria")
        truth = pred["truth"]
        if truth not in label_index:
            raise SystemExit(f"{test['id']}: gold label {truth!r} not in the 77-label bank")
        correct = int(pred["correct"] == "True")
        if correct != int(pred["prediction"] == truth):
            raise SystemExit(f"{test['id']}: correctness column disagrees with prediction")
        jev_correct += correct
        truth_counts[truth] = truth_counts.get(truth, 0) + 1
        rows.append({
            "id": test["id"],
            "i": i,
            "text": test["text"],
            "gold": truth,
            "gold_index": label_index[truth],
            "examples": [{"message": e["message"], "intent": e["intent"]} for e in examples],
            "jev_prediction": pred["prediction"],
            "jev_correct": correct,
            "criteria_count": len(request["questions"]["intent"]["criteria"]),
            "criteria_sha256": sha256_json(request["questions"]["intent"]["criteria"]),
        })

    if jev_correct != 2846:
        raise SystemExit(f"public Jev total changed: {jev_correct} correct, expected 2846")
    if len(truth_counts) != 77:
        raise SystemExit(f"expected all 77 labels in test, got {len(truth_counts)}")

    record = {
        "protocol": PROTOCOL,
        "public": {
            "repository": "https://github.com/simonmesmith/jev-banking77-experiment",
            "commit": "5cac4ff7783a4cfc0badba4124a309dd2a2b9862",
            "files": {
                "data/test_inputs.json": sha256_file(tests_path),
                "results/predictions.csv": sha256_file(preds_path),
                "configs/frozen.json": sha256_file(public / "configs/frozen.json"),
                "runs/test-v1/requests": requests_concat_sha256(requests_dir),
            },
        },
        "task": "banking77",
        "split": "test",
        "n": len(rows),
        "labels": labels,
        "jev_correct": jev_correct,
        "jev_accuracy": jev_correct / len(rows),
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False)
    args.out.write_text(raw + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    print(f"sha256 {sha256_file(args.out)}")
    print(f"rows {len(rows)}  jev {jev_correct}/{len(rows)} = {jev_correct/len(rows):.6f}")
    return 0


def requests_concat_sha256(requests_dir: Path) -> str:
    h = hashlib.sha256()
    paths = sorted(requests_dir.glob("*/request.json"))
    if len(paths) != 3080:
        raise SystemExit(f"expected 3,080 request files, got {len(paths)}")
    for path in paths:
        h.update(path.relative_to(requests_dir.parents[1]).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
