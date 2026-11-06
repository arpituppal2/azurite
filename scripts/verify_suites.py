#!/usr/bin/env python3
"""Verify the dev and locked suite manifests.

Checks, not vibes:
  * weights sum to 1 and no dataset exceeds 25%;
  * row counts match the stored datasets;
  * row digests and item-file hashes match;
  * dev train/test document overlap is zero;
  * dev and locked contexts do not overlap;
  * every stored file hash still matches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "research" / "suites"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(rows, task: str) -> str:
    items = [{
        "task": task,
        "id": row["id"],
        "label": int(row["label"]),
        "context_sha256": sha256_text(row["context"]),
        "question_sha256": sha256_text(row["question"]),
    } for row in rows]
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def contexts(rows) -> set[str]:
    return {sha256_text(x) for x in rows["context"]}


def main() -> int:
    dev = json.loads((SUITES / "dev_suite.json").read_text())
    locked = json.loads((SUITES / "locked_suite.json").read_text())
    errors: list[str] = []

    for name, manifest, weight_key in [
        ("dev", dev, "weight"),
        ("locked", locked, "weight"),
    ]:
        total = sum(t[weight_key] for t in manifest["tasks"])
        if abs(total - 1.0) > 0.002:
            errors.append(f"{name}: weights sum to {total:.4f}")
        for task in manifest["tasks"]:
            if task[weight_key] > 0.25 + 1e-9:
                errors.append(f"{name}/{task['task']}: weight {task[weight_key]} > 0.25")

    dev_contexts = {}
    for task in dev["tasks"]:
        path = ROOT / task["path"]
        train = load_from_disk(str(path / "train"))
        test = load_from_disk(str(path / "test"))
        if len(train) != task["train_rows"]:
            errors.append(f"dev/{task['task']}: train rows {len(train)} != {task['train_rows']}")
        if len(test) != task["test_rows"]:
            errors.append(f"dev/{task['task']}: test rows {len(test)} != {task['test_rows']}")
        if digest(train, task["task"]) != task["train_digest"]:
            errors.append(f"dev/{task['task']}: train digest mismatch")
        if digest(test, task["task"]) != task["test_digest"]:
            errors.append(f"dev/{task['task']}: test digest mismatch")
        overlap = contexts(train) & contexts(test)
        if overlap:
            errors.append(f"dev/{task['task']}: train/test context overlap {len(overlap)}")
        dev_contexts[task["task"]] = contexts(train) | contexts(test)
        for rel, expected in task["files_sha256"].items():
            if sha256_file(ROOT / rel) != expected:
                errors.append(f"dev/{task['task']}: file hash mismatch {rel}")

    locked_contexts = {}
    for task in locked["tasks"]:
        path = ROOT / task["path"] / "test"
        rows = load_from_disk(str(path))
        if len(rows) != task["rows"]:
            errors.append(f"locked/{task['task']}: rows {len(rows)} != {task['rows']}")
        if digest(rows, task["task"]) != task["digest"]:
            errors.append(f"locked/{task['task']}: digest mismatch")
        locked_contexts[task["task"]] = contexts(rows)

    # Dev task -> locked task name mapping for the overlapping datasets.
    pairs = {
        "banking77_ctx": "banking77",
        "clinc_oos_ctx": "clinc150_plus",
        "boolq_ctx": "boolq",
        "squad2_answerability_ctx": "squad2_answerability",
        "contract_nli_ctx": "contract_nli",
    }
    for dev_task, locked_task in pairs.items():
        overlap = dev_contexts[dev_task] & locked_contexts[locked_task]
        if overlap:
            errors.append(f"{dev_task}/{locked_task}: dev/locked context overlap {len(overlap)}")

    for name, item_file, expected in [
        ("dev", "dev_items.jsonl", dev["items_sha256"]),
        ("locked", "locked_items.jsonl", locked["items_sha256"]),
    ]:
        data = (SUITES / item_file).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            errors.append(f"{name}: {item_file} hash mismatch")

    print(f"dev tasks: {len(dev['tasks'])}, locked tasks: {len(locked['tasks'])}")
    print(f"dev weight sum: {sum(t['weight'] for t in dev['tasks']):.4f}")
    print(f"locked weight sum: {sum(t['weight'] for t in locked['tasks']):.4f}")
    if errors:
        print("\nFAIL")
        for error in errors:
            print(" -", error)
        return 1
    print("PASS: weights, counts, digests, file hashes, and disjointness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
