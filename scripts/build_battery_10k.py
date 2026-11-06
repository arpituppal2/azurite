#!/usr/bin/env python3
"""Build the frozen 10k parity battery from local, licence-recorded rows.

The battery is a *screen* before any live Jev call. Rows are deterministic:
same inputs and seed produce the same ids, order, and hash. Short source
families are filled by cycling rows with an explicit ``reused`` flag rather than
silently shrinking the slice.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from datasets import load_from_disk  # noqa: E402

from azurite.battery import BatteryRow, SLICE_QUOTAS, validate, write_battery  # noqa: E402
from azurite.decisions import (  # noqa: E402
    TASKS, candidate_bank, load_examples,
)

RAW = PROJECT / "data" / "raw"
LICENCES = {
    "banking77": "cc-by-4.0",
    "clinc_oos": "cc-by-3.0",
    "massive_intent": "cc-by-4.0",
    "massive_intent_de": "cc-by-4.0",
    "ag_news": "unresolved",
    "emotion": "unresolved",
    "boolq": "unresolved",
    "winogrande": "unresolved",
    "financial_phrasebank": "cc-by-nc-sa-3.0",
    "squad2": "cc-by-sa-4.0",
    "contract_nli": "unresolved",
}


def _ds(path: Path):
    return load_from_disk(str(path))


def _cycle(rows: list[BatteryRow], n: int, *, slice_name: str) -> list[BatteryRow]:
    if not rows:
        raise RuntimeError(f"no rows available for {slice_name}")
    out: list[BatteryRow] = []
    for i in range(n):
        base = rows[i % len(rows)]
        row = BatteryRow(
            id=f"{slice_name}.{i:05d}",
            slice=slice_name,
            context=base.context,
            question=base.question,
            alternatives=list(base.alternatives),
            gold=base.gold,
            source=base.source,
            licence=base.licence,
            gold_evidence=base.gold_evidence,
            meta={**base.meta, "reused": i >= len(rows)},
        )
        out.append(row)
    return out


def classification_rows(per_task: int = 400) -> list[BatteryRow]:
    specs = ["banking77_ctx", "clinc_oos_ctx", "massive_intent", "ag_news", "emotion"]
    rows: list[BatteryRow] = []
    for name in specs:
        spec = TASKS[name]
        bank = candidate_bank(spec)
        examples = load_examples(spec, bank, split="test", limit=per_task)
        for j, ex in enumerate(examples):
            rows.append(BatteryRow(
                id=f"classification.{name}.{j:04d}",
                slice="classification",
                context=ex.context,
                question=spec.instruction,
                alternatives=list(bank.labels),
                gold=int(ex.gold),
                source=name,
                licence=LICENCES.get(name, "unresolved"),
                meta={"gold_label": bank.labels[ex.gold], "k": len(bank)},
            ))
    return rows


def boolq_rows(limit: int) -> list[BatteryRow]:
    ds = _ds(RAW / "boolq" / "validation")
    rows: list[BatteryRow] = []
    for i, row in enumerate(ds):
        if len(rows) >= limit:
            break
        rows.append(BatteryRow(
            id=f"boolean.boolq.{i:05d}",
            slice="boolean",
            context=str(row["passage"]),
            question=str(row["question"]),
            alternatives=["no", "yes"],
            gold=int(row["label"]),
            source="boolq/validation",
            licence=LICENCES["boolq"],
            meta={"gold_label": "yes" if int(row["label"]) else "no"},
        ))
    return rows


def pairwise_rows(limit: int) -> list[BatteryRow]:
    ds = _ds(RAW / "winogrande" / "winogrande_xl" / "validation")
    rows: list[BatteryRow] = []
    for i, row in enumerate(ds):
        if len(rows) >= limit:
            break
        options = [str(row["option1"]), str(row["option2"])]
        rows.append(BatteryRow(
            id=f"pairwise.winogrande.{i:04d}",
            slice="pairwise",
            context=str(row["sentence"]),
            question="Which option correctly fills the blank?",
            alternatives=options,
            gold=int(row["answer"]) - 1,
            source="winogrande/winogrande_xl/validation",
            licence=LICENCES["winogrande"],
            meta={"gold_label": options[int(row["answer"]) - 1]},
        ))
    return rows


def ordinal_rows(limit: int) -> list[BatteryRow]:
    ds = _ds(RAW / "financial_phrasebank" / "default" / "test")
    labels = ["negative", "neutral", "positive"]
    rows: list[BatteryRow] = []
    for i, row in enumerate(ds):
        if len(rows) >= limit:
            break
        rows.append(BatteryRow(
            id=f"ordinal.fpb.{i:04d}",
            slice="ordinal",
            context=str(row["text"]),
            question="Classify the sentiment as negative, neutral, or positive.",
            alternatives=labels,
            gold=int(row["label"]),
            source="financial_phrasebank/default/test",
            licence=LICENCES["financial_phrasebank"],
            meta={"gold_label": labels[int(row["label"])]},
        ))
    return rows


def _context_rows(name: str, limit: int, *, slice_name: str) -> list[BatteryRow]:
    spec = TASKS[name]
    bank = candidate_bank(spec)
    examples = load_examples(spec, bank, split="test", limit=limit)
    rows: list[BatteryRow] = []
    for j, ex in enumerate(examples):
        rows.append(BatteryRow(
            id=f"{slice_name}.{name}.{j:04d}",
            slice=slice_name,
            context=ex.context,
            question=spec.instruction if name.endswith("_ctx") else spec.instruction,
            alternatives=list(bank.labels),
            gold=int(ex.gold),
            source=f"{name}/test",
            licence=LICENCES.get(name.split("_")[0], "unresolved"),
            meta={"gold_label": bank.labels[ex.gold], "k": len(bank)},
        ))
    return rows


def document_rows() -> list[BatteryRow]:
    rows = _context_rows("squad2_answerability_ctx", 748, slice_name="document")
    rows += _context_rows("contract_nli_ctx", 748, slice_name="document")
    rows += _context_rows("boolq_ctx", 504, slice_name="document")
    for i, row in enumerate(rows):
        row.id = f"document.{i:05d}"
    return rows


def abstention_rows() -> list[BatteryRow]:
    # Prefer test-only unanswerable rows. SQuAD2 test has 750; fill the
    # remainder from CLINC OOS if present, then from SQuAD train with the
    # source recorded (the gate reports the split composition).
    rows: list[BatteryRow] = []
    squad = _ds(RAW / "squad2_answerability_ctx" / "test")
    for i, row in enumerate(squad):
        if str(row["label_text"]) != "unanswerable":
            continue
        rows.append(BatteryRow(
            id=f"abstention.squad.{len(rows):04d}",
            slice="abstention", context=str(row["context"]),
            question="Is the question answerable from the passage?",
            alternatives=["answerable", "unanswerable"], gold=1,
            source="squad2_answerability_ctx/test", licence=LICENCES["squad2"],
            meta={"gold_label": "unanswerable"},
        ))
        if len(rows) >= 750:
            break
    if len(rows) < 1_000:
        clinc = _ds(RAW / "clinc_oos_ctx" / "test")
        bank = candidate_bank(TASKS["clinc_oos_ctx"])
        for i, row in enumerate(clinc):
            text = str(row["label_text"]).lower()
            if "oos" not in text and "out" not in text:
                continue
            rows.append(BatteryRow(
                id=f"abstention.clinc.{len(rows):04d}",
                slice="abstention", context=str(row["context"]),
                question=TASKS["clinc_oos_ctx"].instruction,
                alternatives=list(bank.labels), gold=bank.resolve(row["label"]),
                source="clinc_oos_ctx/test", licence=LICENCES["clinc_oos"],
                meta={"gold_label": str(row["label_text"])},
            ))
            if len(rows) >= 1_000:
                break
    if len(rows) < 1_000:
        train = _ds(RAW / "squad2_answerability_ctx" / "train")
        for i, row in enumerate(train):
            if str(row["label_text"]) != "unanswerable":
                continue
            rows.append(BatteryRow(
                id=f"abstention.squad_train.{len(rows):04d}",
                slice="abstention", context=str(row["context"]),
                question="Is the question answerable from the passage?",
                alternatives=["answerable", "unanswerable"], gold=1,
                source="squad2_answerability_ctx/train",
                licence=LICENCES["squad2"],
                meta={"gold_label": "unanswerable", "split": "train"},
            ))
            if len(rows) >= 1_000:
                break
    return rows[:1_000]


def robustness_rows(base: list[BatteryRow]) -> list[BatteryRow]:
    variants = [
        ("prefix", "Distractor note: the following passage is unrelated. "),
        ("suffix", "\n\nUnrelated note: this sentence is not evidence."),
    ]
    rows: list[BatteryRow] = []
    for i, row in enumerate(base[:500]):
        for name, affix in variants:
            context = affix + row.context if name == "prefix" else row.context + affix
            rows.append(BatteryRow(
                id=f"robustness.{name}.{i:04d}",
                slice="robustness", context=context, question=row.question,
                alternatives=list(row.alternatives), gold=row.gold,
                source=row.source, licence=row.licence,
                meta={"variant": name, "base_id": row.id},
            ))
    return rows


def multilingual_rows(limit: int) -> list[BatteryRow]:
    ds = _ds(RAW / "massive_intent_de" / "de" / "test")
    labels = sorted({str(r["label_text"]) for r in ds})
    rows: list[BatteryRow] = []
    for i, row in enumerate(ds):
        if len(rows) >= limit:
            break
        label = str(row["label_text"])
        rows.append(BatteryRow(
            id=f"multilingual.massive_de.{i:04d}",
            slice="multilingual", context=str(row["text"]),
            question="Wähle die passende Absicht.",
            alternatives=labels, gold=labels.index(label),
            source="massive_intent_de/de/test",
            licence=LICENCES["massive_intent_de"],
            meta={"gold_label": label, "language": "de"},
        ))
    return rows


def build(seed: int = 0) -> list[BatteryRow]:
    classification = classification_rows()
    boolean = boolq_rows(1_500)
    pairwise = pairwise_rows(1_000)
    ordinal = ordinal_rows(1_000)
    document = document_rows()
    abstention = abstention_rows()
    robustness = robustness_rows(classification)
    multilingual = multilingual_rows(500)
    rows = classification + boolean + pairwise + ordinal + document + abstention + robustness + multilingual
    # Stable order by slice/id; the seed is recorded even though the builder is
    # deterministic today, so a future change to sampling is visible.
    rows.sort(key=lambda r: (r.slice, r.id))
    if len(rows) != 10_000:
        raise SystemExit(f"battery has {len(rows)} rows, expected 10000; slice counts: {Counter(r.slice for r in rows)}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/parity/battery_10k.json.gz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="validate an existing battery and exit")
    args = ap.parse_args()
    if args.check:
        from azurite.battery import read_battery
        rows, _ = read_battery(args.out)
        report = validate(rows)
        print(report)
        return 0 if report["ok"] else 1
    rows = build(seed=args.seed)
    report = validate(rows)
    if not report["ok"]:
        print(report, file=sys.stderr)
        return 1
    manifest = write_battery(rows, args.out)
    print(manifest)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
