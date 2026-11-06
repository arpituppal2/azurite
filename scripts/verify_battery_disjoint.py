#!/usr/bin/env python3
"""Verify the locked 32k pack sources are disjoint from train/dev sources.

The long-context builder draws target passages and filler from the same
official split.  This records document-hash inventories for the train and test
splits, plus the intersection, before the locked pack battery is frozen.  It
does not read any evaluation battery rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from azurite.decisions import TASKS, candidate_bank, load_examples  # noqa: E402
from build_long_context import document_of  # noqa: E402

SOURCES = ("squad2_answerability_ctx", "contract_nli_ctx", "boolq_ctx")


def inventory(split: str) -> tuple[set[str], dict[str, int]]:
    hashes: set[str] = set()
    counts: dict[str, int] = {}
    for name in SOURCES:
        spec = TASKS[name]
        examples = load_examples(spec, candidate_bank(spec), split=split)
        for ex in examples:
            text = document_of(ex)
            if text.strip():
                hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
        counts[name] = len(examples)
    return hashes, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train, train_counts = inventory("train")
    test, test_counts = inventory("test")
    dev = train  # the dev packs were built with --split train
    overlap_train = sorted(train & test)
    overlap_dev = sorted(dev & test)
    payload = {
        "sources": list(SOURCES),
        "split_counts": {"train": train_counts, "test": test_counts},
        "unique_document_hashes": {"train": len(train), "test": len(test),
                                   "dev_subset_of_train": len(dev)},
        "locked": args.locked,
        "overlap_train": len(overlap_train),
        "overlap_train_sha256": hashlib.sha256("\n".join(overlap_train).encode()).hexdigest(),
        "overlap_dev": len(overlap_dev),
        "overlap_dev_sha256": hashlib.sha256("\n".join(overlap_dev).encode()).hexdigest(),
        "zero_overlap": not overlap_train and not overlap_dev,
        "method": "sha256 of document_of(example) for every official train/test row in the three sources; dev is the train-split subset",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload, indent=1))
    return 0 if payload["zero_overlap"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
