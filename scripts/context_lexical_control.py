#!/usr/bin/env python3
"""A lexical control for the context-scaling dev tasks.

The 14.4M heads collapsed to the majority class on BoolQ and SQuAD 2.0. That
makes their context curves uninformative: a model that cannot learn the task
cannot tell us whether a longer window would help. This control asks whether the
task itself carries a learnable signal and whether that signal grows with the
amount of context a simple model can see.

It is not a comparator for Azurite and not a capability result. It is a
task-validity check, with the same rows and the same byte-budget rules.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "research" / "suites" / "dev_suite.json"
OUT = ROOT / "results" / "context_scaling"
LENGTHS = [256, 512, 1024, 2048, 4096]
TASKS = ["boolq_ctx", "squad2_answerability_ctx", "contract_nli_ctx"]


def truncate(text: str, n_bytes: int) -> str:
    return text.encode("utf-8")[:n_bytes].decode("utf-8", errors="ignore")


def field_text(context: str, question: str, length: int) -> str:
    if not question:
        return truncate(context, length)
    per = max(1, (length - 1) // 2)
    return truncate(context, per) + "\n" + truncate(question, per)


def main() -> int:
    manifest = json.loads(DEV.read_text())
    by_task = {t["task"]: t for t in manifest["tasks"]}
    records = {}
    lines = [
        "# Lexical control for the context-scaling dev tasks",
        "",
        "TF-IDF (1–2 grams) + logistic regression, class-balanced, on the same",
        "2,499–2,500 train / 748–750 test rows and the same byte budgets as the",
        "neural curve. This is a task-validity control, **not** an Azurite result.",
        "",
        "| task | length | accuracy | majority | vs majority | Δ vs 256 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        meta = by_task[task]
        base = ROOT / meta["path"]
        train = load_from_disk(str(base / "train"))
        test = load_from_disk(str(base / "test"))
        y_train = np.array(train["label"], dtype=int)
        y_test = np.array(test["label"], dtype=int)
        majority = 100.0 * Counter(y_test.tolist()).most_common(1)[0][1] / len(y_test)
        cells = []
        for length in LENGTHS:
            x_train = [field_text(c, q, length) for c, q in zip(train["context"], train["question"])]
            x_test = [field_text(c, q, length) for c, q in zip(test["context"], test["question"])]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000,
                                  sublinear_tf=True)
            xtr = vec.fit_transform(x_train)
            xte = vec.transform(x_test)
            clf = LogisticRegression(max_iter=2_000, class_weight="balanced", n_jobs=1)
            clf.fit(xtr, y_train)
            pred = clf.predict(xte)
            acc = 100.0 * float((pred == y_test).mean())
            cells.append({
                "max_seq": length,
                "accuracy": round(acc, 2),
                "majority_accuracy": round(majority, 2),
                "delta_vs_majority_pp": round(acc - majority, 2),
                "n_features": int(xtr.shape[1]),
            })
            lines.append(
                f"| `{task}` | {length} | {acc:.2f} | {majority:.2f} | "
                f"{acc-majority:+.2f} | {acc-cells[0]['accuracy']:+.2f} |"
            )
        records[task] = {"cells": cells, "rows": {"train": len(train), "test": len(test)}}

    payload = {
        "protocol": "context-lexical-control-v1",
        "note": (
            "TF-IDF + class-balanced logistic regression; same dev rows and byte "
            "budgets. A task-validity control, not a model capability result."
        ),
        "lengths": LENGTHS,
        "tasks": records,
    }
    text = json.dumps(payload, indent=2) + "\n"
    (OUT / "LEXICAL_CONTROL.json").write_text(text)
    (OUT / "LEXICAL_CONTROL.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "LEXICAL_CONTROL.json")
    print("sha256", hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
