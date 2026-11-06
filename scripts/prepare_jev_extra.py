#!/usr/bin/env python3
"""Normalize the extra fixed-label Jev benchmarks into `data/raw`.

Only datasets whose Jev primitive is a single Choice or Noul are here. The
multiple-choice, multi-label and ordinal families need a different harness and
are deliberately not forced into this one.

Each output has a ClassLabel `label` and one or more text columns. The loader in
`azurite/decisions.py` already knows how to build a candidate bank from a
ClassLabel, so no model change is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def save(name: str, splits: dict[str, Dataset], label_names: list[str]) -> None:
    out = RAW / name
    payload = DatasetDict({split: ds for split, ds in splits.items()})
    for split in payload:
        payload[split] = payload[split].cast_column(
            "label", ClassLabel(names=label_names)
        )
    out.mkdir(parents=True, exist_ok=True)
    payload.save_to_disk(str(out))
    print(name, {split: len(ds) for split, ds in payload.items()})


def keep(ds: Dataset, columns: list[str]) -> Dataset:
    return ds.remove_columns([c for c in ds.column_names if c not in columns])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=[
        "imdb", "rotten_tomatoes", "sst2", "sms_spam", "paws", "anli",
        "boolq", "toxigen", "toxic_chat", "prompt_injections", "agb_de",
    ])
    args = ap.parse_args()

    if "imdb" in args.tasks:
        ds = load_dataset("stanfordnlp/imdb")
        save("imdb", {"train": keep(ds["train"], ["text", "label"]),
                      "test": keep(ds["test"], ["text", "label"])},
             ["negative", "positive"])

    if "rotten_tomatoes" in args.tasks:
        ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        save("rotten_tomatoes",
             {"train": keep(ds["train"], ["text", "label"]),
              "test": keep(ds["test"], ["text", "label"])},
             ["negative", "positive"])

    if "sst2" in args.tasks:
        ds = load_dataset("stanfordnlp/sst2")
        save("sst2", {"train": keep(ds["train"], ["sentence", "label"]),
                      "validation": keep(ds["validation"], ["sentence", "label"])},
             ["negative", "positive"])

    if "sms_spam" in args.tasks:
        ds = load_dataset("ucirvine/sms_spam")["train"]
        # The HF release has one split. Hold out a fixed 20% test set so the
        # fine-tune has train labels; this is not Jev's all-rows protocol.
        split = ds.train_test_split(test_size=0.2, seed=0, stratify_by_column="label")
        save("sms_spam",
             {"train": keep(split["train"], ["sms", "label"]),
              "test": keep(split["test"], ["sms", "label"])},
             ["ham", "spam"])

    if "paws" in args.tasks:
        ds = load_dataset("google-research-datasets/paws", "labeled_final")
        save("paws", {"train": keep(ds["train"], ["sentence1", "sentence2", "label"]),
                      "test": keep(ds["test"], ["sentence1", "sentence2", "label"])},
             ["different", "paraphrase"])

    if "anli" in args.tasks:
        ds = load_dataset("facebook/anli", "plain_text")
        train = concatenate_datasets([ds[f"train_r{i}"] for i in (1, 2, 3)])
        test = concatenate_datasets([ds[f"test_r{i}"] for i in (1, 2, 3)])
        save("anli", {"train": keep(train, ["premise", "hypothesis", "label"]),
                      "test": keep(test, ["premise", "hypothesis", "label"])},
             ["entailment", "neutral", "contradiction"])

    if "boolq" in args.tasks:
        ds = load_dataset("google/boolq")
        def norm(rows):
            return Dataset.from_dict({
                "passage": rows["passage"],
                "question": rows["question"],
                "label": [int(x) for x in rows["answer"]],
            })
        save("boolq", {"train": norm(ds["train"]), "validation": norm(ds["validation"])},
             ["no", "yes"])

    if "toxigen" in args.tasks:
        ds = load_dataset("toxigen/toxigen-data", "annotated")
        for split in ("train", "test"):
            ds[split] = Dataset.from_dict({
                "text": ds[split]["text"],
                "label": [int(float(x) >= 3.0) for x in ds[split]["toxicity_human"]],
            })
        save("toxigen", {"train": ds["train"], "test": ds["test"]},
             ["not_toxic", "toxic"])

    if "toxic_chat" in args.tasks:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
        save("toxic_chat",
             {"train": keep(ds["train"], ["user_input", "toxicity"]).rename_column(
                  "toxicity", "label"),
              "test": keep(ds["test"], ["user_input", "toxicity"]).rename_column(
                  "toxicity", "label")},
             ["not_toxic", "toxic"])

    if "prompt_injections" in args.tasks:
        ds = load_dataset("deepset/prompt-injections")
        save("prompt_injections",
             {"train": keep(ds["train"], ["text", "label"]),
              "test": keep(ds["test"], ["text", "label"])},
             ["benign", "injection"])

    if "agb_de" in args.tasks:
        ds = load_dataset("d4br4/agb-de")
        for split in ("train", "test"):
            print(split, ds[split].features["label"])
        save("agb_de", {"train": keep(ds["train"], ["text", "label"]),
                        "test": keep(ds["test"], ["text", "label"])},
             ["valid", "void"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
