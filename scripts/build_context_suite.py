#!/usr/bin/env python3
"""Materialise the compact dev suite and the broader locked suite.

The dev suite is built **only from training splits**. The locked suite is built
from official test/validation splits and is hashed but not evaluated here. This
separation is the point: the context-scaling curve must not be produced on rows
that later serve as the locked test.

Both suites use one row schema:

    id, context, question, label, label_text, source, source_split, source_index

`context` is the document or message, `question` is the query (empty for a
single-field classification task), and `label` is a ClassLabel whose names are
the candidate set. Rows are grouped by document before the train/test split so
two questions over the same passage cannot land on opposite sides of the split.

Run:
    .venv/bin/python scripts/build_context_suite.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RAW = DATA / "raw"
SUITES = PROJECT / "research" / "suites"
os.environ.setdefault("HF_HOME", str(DATA / ".hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DATA / ".hf" / "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import datasets as hf  # noqa: E402
from datasets import ClassLabel, Dataset, Features, Value, load_dataset, load_from_disk  # noqa: E402

DEV_TRAIN = 2_500
DEV_TEST = 750
LOCKED_ROWS = 500
SEED = 0

DEV_TASKS = {
    "banking77_ctx": {
        "family": "classification",
        "source": "data/raw/banking77/default/train",
        "licence": "CC-BY-4.0",
        "product_admissible": True,
        "fields": ("text",),
        "label_mode": "banking77",
    },
    "clinc_oos_ctx": {
        "family": "classification",
        "source": "data/raw/clinc_oos/plus/train",
        "licence": "CC-BY-3.0",
        "product_admissible": True,
        "fields": ("text",),
        "label_mode": "classlabel_intent",
    },
    "boolq_ctx": {
        "family": "document_grounded",
        "source": "google/boolq:train",
        "licence": "CC-BY-SA-3.0",
        "product_admissible": True,
        "fields": ("passage", "question"),
        "label_mode": "boolq",
    },
    "squad2_answerability_ctx": {
        "family": "document_grounded_abstention",
        "source": "rajpurkar/squad_v2:train",
        "licence": "CC-BY-SA-4.0",
        "product_admissible": True,
        "fields": ("context", "question"),
        "label_mode": "squad2",
    },
    "contract_nli_ctx": {
        "family": "policy_rule",
        "source": "presencesw/contract-nli:train",
        "licence": "CC-BY-4.0 (mirror; canonical ContractNLI)",
        "product_admissible": True,
        "fields": ("sentence1", "sentence2"),
        "label_mode": "contract_nli",
    },
}

LOCKED_TASKS = {
    "banking77": ("classification", "data/raw/banking77/default/test", "CC-BY-4.0", True),
    "clinc150_plus": ("classification", "data/raw/clinc_oos/plus/test", "CC-BY-3.0", True),
    "ag_news": ("classification", "data/raw/ag_news/default/test", "UNRESOLVED", False),
    "emotion": ("classification", "data/raw/emotion/split/test", "UNRESOLVED", False),
    "massive_intent_en": ("classification", "data/raw/massive_intent/en/test", "CC-BY-4.0", True),
    "massive_scenario_en": ("classification", "data/raw/massive_scenario/en/test", "CC-BY-4.0", True),
    "xnli_en": ("document_grounded", "data/raw/xnli/en/test", "UNRESOLVED", False),
    "language_id": ("classification", "data/raw/language_id/default/test", "UNRESOLVED", False),
    "financial_phrasebank": ("classification", "data/raw/financial_phrasebank/default/test", "UNRESOLVED", False),
    "boolq": ("document_grounded", "google/boolq:validation", "CC-BY-SA-3.0", True),
    "squad2_answerability": ("document_grounded_abstention", "rajpurkar/squad_v2:validation", "CC-BY-SA-4.0", True),
    "contract_nli": ("policy_rule", "presencesw/contract-nli:test", "CC-BY-4.0 (mirror; canonical ContractNLI)", True),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_source(source: str):
    if source.startswith("data/"):
        return load_from_disk(str(PROJECT / source))
    name, _, split = source.partition(":")
    if not split:
        raise ValueError(f"HF source needs path:split, got {source}")
    return load_dataset(name, split=split)


def _banking_labels(ds) -> list[str]:
    pairs = {}
    for row in ds:
        pairs[int(row["label"])] = str(row["label_text"])
    return [pairs[i] for i in sorted(pairs)]


def source_split(source: str) -> str:
    if source.startswith("data/"):
        parts = source.split("/")
        return "/".join(parts[-2:])
    return source.split(":", 1)[1] if ":" in source else "train"


def normalise(source: str, mode: str) -> tuple[list[dict], list[str]]:
    """Return rows in the uniform schema and the candidate-label names."""
    ds = load_source(source)
    rows: list[dict] = []
    if mode == "banking77":
        names = _banking_labels(ds)
        for i, row in enumerate(ds):
            rows.append({
                "id": f"{source}:{i}",
                "context": str(row["text"]),
                "question": "",
                "label": int(row["label"]),
                "label_text": str(row["label_text"]),
                "source": source.split(":")[0],
                "source_split": "train",
                "source_index": i,
            })
        return rows, names
    if mode in {"classlabel_intent", "classlabel_label"}:
        field = "intent" if mode == "classlabel_intent" else "label"
        feature = ds.features[field]
        if not isinstance(feature, ClassLabel):
            raise TypeError(f"{source}: {field} is not a ClassLabel")
        names = list(feature.names)
        for i, row in enumerate(ds):
            label = int(row[field])
            rows.append({
                "id": f"{source}:{i}",
                "context": str(row["text"]),
                "question": "",
                "label": label,
                "label_text": names[label],
                "source": source.split(":")[0],
                "source_split": source_split(source),
                "source_index": i,
            })
        return rows, names
    if mode == "classlabel_label_text":
        names = _banking_labels(ds)
        for i, row in enumerate(ds):
            label = int(row["label"])
            rows.append({
                "id": f"{source}:{i}",
                "context": str(row["text"]),
                "question": "",
                "label": label,
                "label_text": str(row["label_text"]),
                "source": source.split(":")[0],
                "source_split": source_split(source),
                "source_index": i,
            })
        return rows, names
    if mode == "massive_label":
        names = sorted({str(row["label_text"]) for row in ds})
        lookup = {name: i for i, name in enumerate(names)}
        for i, row in enumerate(ds):
            label = lookup[str(row["label_text"])]
            rows.append({
                "id": f"{source}:{i}",
                "context": str(row["text"]),
                "question": "",
                "label": label,
                "label_text": names[label],
                "source": source.split(":")[0],
                "source_split": source_split(source),
                "source_index": i,
            })
        return rows, names
    if mode == "boolq":
        names = ["false", "true"]
        for i, row in enumerate(ds):
            rows.append({
                "id": f"{source}:{i}",
                "context": str(row["passage"]),
                "question": str(row["question"]),
                "label": int(bool(row["answer"])),
                "label_text": names[int(bool(row["answer"]))],
                "source": source.split(":")[0],
                "source_split": source_split(source),
                "source_index": i,
            })
        return rows, names
    if mode == "squad2":
        names = ["unanswerable", "answerable"]
        for i, row in enumerate(ds):
            answers = row["answers"]["text"] if row["answers"] else []
            label = int(bool(answers))
            rows.append({
                "id": f"{source}:{i}",
                "context": str(row["context"]),
                "question": str(row["question"]),
                "label": label,
                "label_text": names[label],
                "source": source.split(":")[0],
                "source_split": source_split(source),
                "source_index": i,
            })
        return rows, names
    if mode == "contract_nli":
        raw_names = ["Entailment", "Contradiction", "NotMentioned"]
        names = ["entailment", "contradiction", "not_mentioned"]
        lookup = {raw: i for i, raw in enumerate(raw_names)}
        for i, row in enumerate(ds):
            raw = str(row["gold_label"])
            if raw not in lookup:
                raise ValueError(f"unknown ContractNLI label {raw!r}")
            label = lookup[raw]
            rows.append({
                "id": f"{source}:{i}",
                "context": str(row["sentence1"]),
                "question": str(row["sentence2"]),
                "label": label,
                "label_text": names[label],
                "source": source.split(":")[0],
                "source_split": source_split(source),
                "source_index": i,
            })
        return rows, names
    if mode == "xnli":
        feature = ds.features["label"]
        names = list(feature.names) if isinstance(feature, ClassLabel) else ["entailment", "neutral", "contradiction"]
        for i, row in enumerate(ds):
            rows.append({
                "id": f"{source}:{i}",
                "context": str(row["premise"]),
                "question": str(row["hypothesis"]),
                "label": int(row["label"]),
                "label_text": names[int(row["label"])],
                "source": source.split(":")[0],
                "source_split": source_split(source),
                "source_index": i,
            })
        return rows, names
    if mode == "language_id":
        names = sorted({str(x) for x in ds["labels"]})
        lookup = {name: i for i, name in enumerate(names)}
        for i, row in enumerate(ds):
            label = lookup[str(row["labels"])]
            rows.append({
                "id": f"{source}:{i}",
                "context": str(row["text"]),
                "question": "",
                "label": label,
                "label_text": names[label],
                "source": source.split(":")[0],
                "source_split": source_split(source),
                "source_index": i,
            })
        return rows, names
    raise ValueError(mode)


def mode_for_source(name: str, source: str) -> str:
    if name == "banking77":
        return "classlabel_label_text"
    if name == "clinc150_plus":
        return "classlabel_intent"
    if name in {"ag_news", "emotion"}:
        return "classlabel_label"
    if name in {"massive_intent_en", "massive_scenario_en"}:
        return "massive_label"
    if name == "financial_phrasebank":
        return "classlabel_label_text"
    if name == "xnli_en":
        return "xnli"
    if name == "language_id":
        return "language_id"
    if name == "boolq":
        return "boolq"
    if name == "squad2_answerability":
        return "squad2"
    if name == "contract_nli":
        return "contract_nli"
    raise ValueError(f"no normaliser for {name} ({source})")


def group_key(row: dict) -> str:
    # Document-level grouping: two questions over one passage never cross the
    # train/test boundary.
    return sha256_text(row["context"])


def group_split(rows: list[dict], n_train: int, n_test: int, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    train: list[dict] = []
    test: list[dict] = []
    for key in keys:
        group = groups[key]
        if len(train) + len(group) <= n_train:
            train.extend(group)
        elif len(test) + len(group) <= n_test:
            test.extend(group)
    if not train:
        raise ValueError("group split produced an empty train side")
    if n_test and not test:
        raise ValueError("group split produced an empty test side")
    train.sort(key=lambda r: (r["source"], int(r["source_index"])))
    test.sort(key=lambda r: (r["source"], int(r["source_index"])))
    return train, test


def make_dataset(rows: list[dict], names: list[str], source: str) -> Dataset:
    features = Features({
        "id": Value("string"),
        "context": Value("string"),
        "question": Value("string"),
        "label": ClassLabel(names=names),
        "label_text": Value("string"),
        "source": Value("string"),
        "source_split": Value("string"),
        "source_index": Value("int32"),
    })
    columns = {key: [row[key] for row in rows] for key in features}
    return Dataset.from_dict(columns, features=features)


def row_digest(rows: list[dict], task: str) -> str:
    items = [{
        "task": task,
        "id": row["id"],
        "label": int(row["label"]),
        "context_sha256": sha256_text(row["context"]),
        "question_sha256": sha256_text(row["question"]),
    } for row in rows]
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def write_items(path: Path, rows_by_task: dict[str, list[dict]]) -> str:
    lines = []
    for task, rows in sorted(rows_by_task.items()):
        for row in rows:
            lines.append(json.dumps({
                "task": task,
                "id": row["id"],
                "label": int(row["label"]),
                "context_sha256": sha256_text(row["context"]),
                "question_sha256": sha256_text(row["question"]),
            }, sort_keys=True))
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def context_stats(rows: list[dict]) -> dict:
    lens = sorted(len(r["context"].encode("utf-8")) + len(r["question"].encode("utf-8")) for r in rows)
    n = len(lens)
    return {
        "n": n,
        "bytes_p50": lens[n // 2],
        "bytes_p90": lens[min(n - 1, int(0.9 * n))],
        "bytes_max": lens[-1],
    }


DEV_TO_LOCKED = {
    "banking77_ctx": "banking77",
    "clinc_oos_ctx": "clinc150_plus",
    "boolq_ctx": "boolq",
    "squad2_answerability_ctx": "squad2_answerability",
    "contract_nli_ctx": "contract_nli",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-only", action="store_true",
                    help="rebuild only the locked suite, keeping the frozen dev files")
    args = ap.parse_args()
    SUITES.mkdir(parents=True, exist_ok=True)

    dev_tasks = []
    dev_items: dict[str, list[dict]] = {}
    dev_hashes: dict[str, set[str]] = {}
    if args.locked_only:
        for task in DEV_TASKS:
            path = RAW / task
            if not path.exists():
                raise SystemExit(f"missing dev task {path}; run without --locked-only first")
            train = load_from_disk(str(path / "train"))
            test = load_from_disk(str(path / "test"))
            dev_hashes[task] = {
                sha256_text(x) for x in list(train["context"]) + list(test["context"])
            }
    else:
        for task, cfg in DEV_TASKS.items():
            print(f"[dev] {task}", flush=True)
            rows, names = normalise(cfg["source"], cfg["label_mode"])
            train, test = group_split(rows, DEV_TRAIN, DEV_TEST, SEED)
            out = RAW / task
            if out.exists():
                raise SystemExit(f"refusing to overwrite {out}")
            make_dataset(train, names, cfg["source"]).save_to_disk(str(out / "train"))
            make_dataset(test, names, cfg["source"]).save_to_disk(str(out / "test"))
            files = sorted((out).rglob("*"))
            dev_items[task] = train + test
            dev_hashes[task] = {sha256_text(r["context"]) for r in train + test}
            dev_tasks.append({
                "task": task,
                "family": cfg["family"],
                "weight": round(1 / len(DEV_TASKS), 4),
                "source": cfg["source"],
                "licence": cfg["licence"],
                "product_admissible": cfg["product_admissible"],
                "train_rows": len(train),
                "test_rows": len(test),
                "train_label_counts": dict(sorted(Counter(int(r["label"]) for r in train).items())),
                "test_label_counts": dict(sorted(Counter(int(r["label"]) for r in test).items())),
                "train_context": context_stats(train),
                "test_context": context_stats(test),
                "train_digest": row_digest(train, task),
                "test_digest": row_digest(test, task),
                "path": str(out.relative_to(PROJECT)),
                "files_sha256": {
                    str(p.relative_to(PROJECT)): sha256_file(p) for p in files if p.is_file()
                },
            })

        dev_manifest = {
            "schema_version": 1,
            "suite": "azurite-context-dev-v1",
            "status": "frozen",
            "frozen": "2026-10-30",
            "seed": SEED,
            "rule": {
                "constructed_from": "training splits only",
                "max_dataset_weight": 0.25,
                "weights": "equal task weights; family macro-average is primary",
                "locked_rows_used": 0,
            },
            "tasks": dev_tasks,
            "items_sha256": None,
            "note": (
                "Compact context-scaling suite. Banking77/CLINC are short-context "
                "controls; BoolQ/SQuAD2 are document-grounded; ContractNLI is "
                "policy/rule-conditioned. No visual, tool-action, or ordinal row is "
                "included because no licensed local task exists."
            ),
        }
        dev_items_path = SUITES / "dev_items.jsonl"
        dev_manifest["items_sha256"] = write_items(dev_items_path, dev_items)
        (SUITES / "dev_suite.json").write_text(json.dumps(dev_manifest, indent=2) + "\n")

    locked_tasks = []
    locked_items: dict[str, list[dict]] = {}
    for task, (family, source, licence, admissible) in LOCKED_TASKS.items():
        print(f"[locked] {task}", flush=True)
        mode = mode_for_source(task, source)
        rows, names = normalise(source, mode)
        # Never let a dev document enter the locked suite. BoolQ's official
        # validation split shares 45 passages with its train split; that is
        # exactly the leak a nominal split misses.
        dev_task = next((d for d, l in DEV_TO_LOCKED.items() if l == task), None)
        if dev_task:
            blocked = dev_hashes[dev_task]
            rows = [row for row in rows if sha256_text(row["context"]) not in blocked]
        # Locked rows are a deterministic, document-disjoint 500-row sample of
        # the official test/validation split. No model output is produced here.
        sampled, _ = group_split(rows, LOCKED_ROWS, 0, SEED + 1000)
        out = RAW / "locked" / task
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        make_dataset(sampled, names, source).save_to_disk(str(out / "test"))
        locked_items[task] = sampled
        locked_tasks.append({
            "task": task,
            "family": family,
            "weight": round(1 / len(LOCKED_TASKS), 4),
            "source": source,
            "licence": licence,
            "product_admissible": admissible,
            "rows": len(sampled),
            "label_counts": dict(sorted(Counter(int(r["label"]) for r in sampled).items())),
            "context": context_stats(sampled),
            "digest": row_digest(sampled, task),
            "path": str(out.relative_to(PROJECT)),
            "previously_inspected": task in {
                "banking77", "clinc150_plus", "ag_news", "emotion",
                "massive_intent_en", "massive_scenario_en", "xnli_en",
                "language_id", "financial_phrasebank",
            },
        })

    locked_manifest = {
        "schema_version": 1,
        "suite": "azurite-locked-v1",
        "status": "sealed",
        "sealed": "2026-10-30",
        "seed": SEED + 1000,
        "rule": {
            "max_dataset_weight": 0.25,
            "weights": "equal task weights; family macro-average is primary",
            "opened": False,
            "no_tuning": "No threshold, prompt, architecture, or hyper-parameter may be chosen on these rows.",
        },
        "tasks": locked_tasks,
        "items_sha256": None,
        "coverage": {
            "families_present": sorted({t["family"] for t in locked_tasks}),
            "families_missing": ["tool_action_abstention", "relevance_preference_ordinal", "multimodal_visual"],
            "missing_reason": (
                "No licensed local task has been materialised for these families. "
                "They remain blocked, not replaced by a toy task."
            ),
        },
        "note": (
            "Broader than the dev suite but not complete across six families. "
            "Several rows were inspected by earlier work; this is a procedural "
            "lock from 2026-10-30, not a claim that no row was ever seen."
        ),
    }
    locked_items_path = SUITES / "locked_items.jsonl"
    locked_manifest["items_sha256"] = write_items(locked_items_path, locked_items)
    (SUITES / "locked_suite.json").write_text(json.dumps(locked_manifest, indent=2) + "\n")

    print("wrote", SUITES / "dev_suite.json")
    print("wrote", SUITES / "locked_suite.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
