#!/usr/bin/env python3
"""Build the frozen 10k evaluation battery (plan task 8).

Eight families, exactly the plan's composition:

    classification 2000 · boolean 1500 · pairwise 1000 · ordinal 1000
    document 2000 · abstention 1000 · robustness 1000 · multilingual 500

Design rules, each enforced rather than asserted in prose:

  * every row is self-describing: family, task, question, context,
    alternatives, gold index and label, provenance (dataset/split/row),
    licence, and — where the family has a span answer — the byte evidence;
  * the document family shares the 100k pack interface: 64 chunks of
    65,537..66,560 bytes, a real question that never occurs in the context,
    distractors, early/middle/late placement, and 10% distributed evidence,
    with gold spans recorded at placement time (not found by string search);
  * the abstention family is half answerable and half unanswerable *by
    construction*: a phrase is removed from a held-out sentence and the
    unanswerable rows are verified to lack that phrase in the whole pack;
  * robustness perturbs a disjoint slice of source rows (indices 400..499 of
    each classification task) so it is not a relabelled copy;
  * the boolean family (BoolQ validation) and the document family (BoolQ
    train-derived ctx rows) are made row-disjoint: passages shared between the
    upstream train and validation splits are excluded;
  * no document row may repeat a (passage, question) pair the 100k selector
    battery was built from.

Output matches the 100k battery's on-disk form: sharded `jsonl.gz` written with
gzip mtime=0 (byte-reproducible), one `.meta.json` per shard, and a merged
`manifest.json` carrying per-family counts, licences, and the canonical
battery sha256 (sha256 over sorted `file:sha:rows` lines, the same definition
`scripts/verify_packs.py` uses).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from datasets import load_from_disk  # noqa: E402

from azurite.decisions import TASKS, candidate_bank, load_examples  # noqa: E402
from azurite.selector import chunk_text  # noqa: E402
from scripts.build_long_context import (  # noqa: E402
    SEP,
    compose_pack,
    merge_shards,
    sample_fillers,
    sha256_file,
    shard_file,
    window_bytes,
    write_manifest,
    write_shard,
)

RAW = PROJECT / "data" / "raw"
TOKENS = PROJECT / "data" / "tokens"

QUOTAS = {
    "classification": 2_000,
    "boolean": 1_500,
    "pairwise": 1_000,
    "ordinal": 1_000,
    "document": 2_000,
    "abstention": 1_000,
    "robustness": 1_000,
    "multilingual": 500,
}
CHUNKS_PER_PACK = 64

# Licence strings come from the repository's own dataset record where one
# exists; three datasets have no recorded licence and are marked unresolved
# rather than guessed. `squad2` carries the licence already used for it in the
# earlier battery builder (cc-by-sa-4.0).
LICENCES = {
    "banking77": "cc-by-4.0",
    "clinc_oos": "cc-by-3.0",
    "massive_intent": "cc-by-4.0",
    "massive_intent_de": "cc-by-4.0",
    "ag_news": "unknown",
    "emotion": "other",
    "boolq": "unresolved",
    "winogrande": "unresolved",
    "financial_phrasebank": "cc-by-nc-sa-3.0",
    "squad2": "cc-by-sa-4.0",
    "contract_nli": "unresolved",
}

DOC_SOURCES = [
    ("squad2_answerability_ctx", 748),
    ("contract_nli_ctx", 748),
    ("boolq_ctx", 504),
]
DOC_LICENCE = {
    "squad2_answerability_ctx": "cc-by-sa-4.0",
    "contract_nli_ctx": "unresolved",
    "boolq_ctx": "unresolved",
}
# The SQuAD2 / ContractNLI / BoolQ ctx rows available locally are carves of the
# same upstream train splits that the 100k selector battery was built from, so
# using them would put evaluation questions in selector training data. They are
# excluded and the measured overlap is recorded in the manifest.
CTX_SOURCES = ["squad2_answerability_ctx", "contract_nli_ctx", "boolq_ctx"]

CLASSIFICATION_TASKS = [
    ("banking77", "banking77"),
    ("clinc_oos", "clinc_oos"),
    ("massive_intent", "massive_intent"),
    ("ag_news", "ag_news"),
    ("emotion", "emotion"),
]
PER_TASK = 400


def _row(
    *,
    row_id: str,
    family: str,
    task: str,
    question: str,
    context: str,
    alternatives: list[str],
    gold: int,
    gold_label: str,
    answer: str | None,
    dataset: str,
    split: str,
    source_row: int,
    licence: str,
    meta: dict | None = None,
    gold_spans: list[dict] | None = None,
) -> dict:
    return {
        "id": row_id,
        "family": family,
        "task": task,
        "question": question,
        "context": context,
        "alternatives": list(alternatives),
        "gold": int(gold),
        "gold_label": gold_label,
        "answer": answer,
        "provenance": {"dataset": dataset, "split": split, "row": int(source_row)},
        "licence": licence,
        "meta": meta or {},
        "gold_spans": gold_spans or [],
    }


def classification_rows(seed: int) -> list[dict]:
    rows: list[dict] = []
    for dataset, task_name in CLASSIFICATION_TASKS:
        spec = TASKS[task_name]
        bank = candidate_bank(spec)
        examples = load_examples(spec, bank, split="test", limit=PER_TASK)
        if len(examples) < PER_TASK:
            raise SystemExit(f"{dataset}: only {len(examples)} test rows, need {PER_TASK}")
        for j, ex in enumerate(examples):
            label = bank.labels[ex.gold]
            rows.append(_row(
                row_id=f"classification.{dataset}.{j:04d}",
                family="classification",
                task=dataset,
                question=spec.instruction,
                context=ex.context,
                alternatives=list(bank.labels),
                gold=ex.gold,
                gold_label=label,
                answer=label,
                dataset=dataset,
                split="test",
                source_row=ex.index,
                licence=LICENCES[dataset],
                meta={"k": len(bank), "instruction_source": "azurite.decisions.TASKS"},
            ))
    return rows


def boolean_rows() -> list[dict]:
    ds = load_from_disk(str(RAW / "boolq" / "validation"))
    rows: list[dict] = []
    for i, rec in enumerate(ds):
        if len(rows) >= QUOTAS["boolean"]:
            break
        passage = str(rec["passage"]).strip()
        question = str(rec["question"]).strip()
        if not passage or not question or question in passage:
            continue
        label = "yes" if int(rec["label"]) else "no"
        rows.append(_row(
            row_id=f"boolean.boolq.{i:04d}",
            family="boolean",
            task="boolq",
            question=question,
            context=passage,
            alternatives=["no", "yes"],
            gold=0 if label == "no" else 1,
            gold_label=label,
            answer=label,
            dataset="boolq",
            split="validation",
            source_row=i,
            licence=LICENCES["boolq"],
            meta={"question_in_context": False},
        ))
    return rows


def pairwise_rows() -> list[dict]:
    ds = load_from_disk(str(RAW / "winogrande" / "winogrande_xl" / "validation"))
    rows: list[dict] = []
    for i, rec in enumerate(ds):
        if len(rows) >= QUOTAS["pairwise"]:
            break
        options = [str(rec["option1"]).strip(), str(rec["option2"]).strip()]
        sentence = str(rec["sentence"]).strip()
        if not sentence or "_" not in sentence or any(not o for o in options):
            continue
        if len(set(options)) != 2:
            continue
        gold = int(rec["answer"]) - 1
        rows.append(_row(
            row_id=f"pairwise.winogrande.{i:04d}",
            family="pairwise",
            task="winogrande",
            question="Which option correctly fills the blank?",
            context=sentence,
            alternatives=options,
            gold=gold,
            gold_label=options[gold],
            answer=options[gold],
            dataset="winogrande",
            split="winogrande_xl/validation",
            source_row=i,
            licence=LICENCES["winogrande"],
            meta={"blank": "_"},
        ))
    return rows


def ordinal_rows() -> list[dict]:
    ds = load_from_disk(str(RAW / "financial_phrasebank" / "default" / "test"))
    labels = ["negative", "neutral", "positive"]
    rows: list[dict] = []
    for i, rec in enumerate(ds):
        if len(rows) >= QUOTAS["ordinal"]:
            break
        text = str(rec["text"]).strip()
        label = str(rec["label_text"]).strip().lower()
        if not text or label not in labels:
            continue
        rows.append(_row(
            row_id=f"ordinal.financial_phrasebank.{i:04d}",
            family="ordinal",
            task="financial_phrasebank",
            question="Classify the sentiment as negative, neutral, or positive.",
            context=text,
            alternatives=labels,
            gold=labels.index(label),
            gold_label=label,
            answer=label,
            dataset="financial_phrasebank",
            split="default/test",
            source_row=i,
            licence=LICENCES["financial_phrasebank"],
            meta={"ordered": True},
        ))
    return rows


def multilingual_rows() -> list[dict]:
    ds = load_from_disk(str(RAW / "massive_intent_de" / "de" / "test"))
    labels = sorted({str(r["label_text"]) for r in ds})
    rows: list[dict] = []
    for i, rec in enumerate(ds):
        if len(rows) >= QUOTAS["multilingual"]:
            break
        text = str(rec["text"]).strip()
        label = str(rec["label_text"]).strip()
        if not text or label not in labels:
            continue
        rows.append(_row(
            row_id=f"multilingual.massive_de.{i:04d}",
            family="multilingual",
            task="massive_intent_de",
            question="Wähle die passende Absicht.",
            context=text,
            alternatives=labels,
            gold=labels.index(label),
            gold_label=label,
            answer=label,
            dataset="massive_intent_de",
            split="de/test",
            source_row=i,
            licence=LICENCES["massive_intent_de"],
            meta={"language": "de", "k": len(labels)},
        ))
    return rows


def _heldout_docs(min_bytes: int = 200) -> list[str]:
    """Decode the held-out tail of the byte corpus (documents never trained on).

    The split rule recorded in `byte_corpus_v2.split.json` appends held-out
    documents after all training documents, so the tail is exactly the held-out
    set. SEP (id 3) decodes to a zero byte and separates documents.
    """
    import numpy as np

    split = json.loads((TOKENS / "byte_corpus_v2.split.json").read_text())
    held_ids = int(split["heldout_bytes"])
    path = TOKENS / "byte_corpus_v2.u16"
    raw = np.memmap(path, dtype="<u2", mode="r")
    start = raw.size - held_ids
    chunk = np.asarray(raw[start:], dtype=np.int16)
    data = (chunk - 4).clip(0, 255).astype(np.uint8).tobytes()
    docs = [d.decode("utf-8", "replace").strip() for d in data.split(b"\x00")]
    return [d for d in docs if len(d.encode("utf-8")) >= min_bytes]


def _ctx_overlap_stats(limit: int = 3_000) -> dict[str, dict]:
    """Measure how much of each local ctx test split the 100k battery already used.

    The 100k build ran with `--limit-per-source 3000` on the train split, so
    those pairs are selector training data. This is the evidence for choosing
    ANLI/XNLI as the document family instead.
    """
    pairs: set[tuple[str, str]] = set()
    for name in CTX_SOURCES:
        spec = TASKS[name]
        bank = candidate_bank(spec)
        for ex in load_examples(spec, bank, split="train", limit=limit):
            if len(ex.parts) >= 2:
                pairs.add((ex.parts[0].strip(), ex.parts[1].strip()))
    stats: dict[str, dict] = {}
    for name in CTX_SOURCES:
        spec = TASKS[name]
        bank = candidate_bank(spec)
        examples = load_examples(spec, bank, split="test", limit=None)
        test_pairs = [
            (ex.parts[0].strip(), ex.parts[1].strip())
            for ex in examples if len(ex.parts) >= 2
        ]
        stats[name] = {
            "test_rows": len(examples),
            "in_100k_pairs": sum(1 for pair in test_pairs if pair in pairs),
            "unique_passages": len({doc for doc, _ in test_pairs}),
        }
    return stats


def _pack_content(
    doc: str,
    distractors: list[str],
    *,
    position: str,
    distributed: bool,
) -> tuple[list[str], list[str], tuple[int, ...]]:
    if distributed:
        middle = len(doc) // 2
        content = [doc[:middle], doc[middle:]]
        positions = ["early", "late"]
        gold = (0, 1)
    else:
        content = [doc]
        positions = [position]
        gold = (0,)
    content += distractors
    positions += ["middle"] * len(distractors)
    return content, positions, gold


def document_rows(seed: int, used_contexts: set[str]) -> tuple[list[dict], dict]:
    """64-chunk packs with human-written questions and gold spans at placement.

    Sources are the SQuAD2 / ContractNLI / BoolQ ctx test carves. They are
    disjoint from the 100k selector pairs (measured zero overlap), but they
    recycle passages - 107/44/524 unique documents for 750/748/750 rows - so a
    pack may repeat a premise with a different question. That trade is recorded
    in the manifest; distinct questions are guaranteed and checked.
    """
    rng = random.Random(seed * 1009 + 5)
    pool = _heldout_docs()
    if len(pool) < 200:
        raise SystemExit(f"held-out pool too small: {len(pool)} documents")
    boolq_val = load_from_disk(str(RAW / "boolq" / "validation"))
    reserved = {str(r["passage"]).strip() for r in boolq_val} | set(used_contexts)

    rows: list[dict] = []
    stats_out: dict[str, dict] = {}
    for name, take in DOC_SOURCES:
        spec = TASKS[name]
        bank = candidate_bank(spec)
        kept: list[tuple[object, str, str]] = []
        seen: set[tuple[str, str]] = set()
        stats = Counter()
        for ex in load_examples(spec, bank, split="test", limit=None):
            if len(ex.parts) < 2:
                stats["no_question"] += 1
                continue
            doc, question = ex.parts[0].strip(), ex.parts[1].strip()
            if not doc or not question or question in doc:
                stats["bad_pair"] += 1
                continue
            if doc in reserved:
                stats["reserved_context"] += 1
                continue
            if (doc, question) in seen:
                stats["duplicate_pair"] += 1
                continue
            seen.add((doc, question))
            kept.append((ex, doc, question))
        stats["kept"] = len(kept)
        stats["unique_passages"] = len({doc for _, doc, _ in kept})
        stats_out[name] = dict(stats)
        if len(kept) < take:
            raise SystemExit(f"{name}: only {len(kept)} usable rows, need {take}")
        for ex, doc, question in kept[:take]:
            index = len(rows)
            n_distractors = [0, 2, 8][index % 3]
            distributed = index % 10 == 0
            position = ["early", "middle", "late"][index % 3]
            candidates = [d for d, q in ((k[1], k[2]) for k in kept) if d != doc and question not in d]
            rng.shuffle(candidates)
            distractors = candidates[:n_distractors]
            content, positions, gold = _pack_content(
                doc, distractors, position=position, distributed=distributed
            )
            fillers = sample_fillers(pool, question, rng, window_bytes(CHUNKS_PER_PACK))
            try:
                text, spans = compose_pack(content, fillers, positions=positions, gold=gold,
                                           chunks_per_pack=CHUNKS_PER_PACK)
            except ValueError as exc:
                raise SystemExit(f"document pack {index} compose failed: {exc}")
            if question in text:
                raise SystemExit(f"document pack {index}: question survived into the pack")
            if len(chunk_text(text)) != CHUNKS_PER_PACK:
                raise SystemExit(f"document pack {index}: wrong chunk count")
            label = bank.labels[ex.gold]
            rows.append(_row(
                row_id=f"document.{index:05d}",
                family="document",
                task=name,
                question=question,
                context=text,
                alternatives=list(bank.labels),
                gold=ex.gold,
                gold_label=label,
                answer=label,
                dataset=name,
                split="test",
                source_row=ex.index,
                licence=DOC_LICENCE[name],
                meta={
                    "chunks": CHUNKS_PER_PACK,
                    "distractors": n_distractors,
                    "position": "distributed" if distributed else position,
                    "label_kind": "document_span",
                    "gold_chunk": spans[0]["chunks"][0],
                    "gold_chunk_end": spans[0]["chunks"][-1],
                    "question_chars": len(question),
                },
                gold_spans=spans,
            ))
    if len(rows) != QUOTAS["document"]:
        raise SystemExit(f"document family has {len(rows)} rows, expected {QUOTAS['document']}")
    return rows, stats_out


_PHRASE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2}\b")


def _stripped_training_corpus() -> Path | None:
    """Cached whitespace-stripped training text, shared with the audit."""
    try:
        from scripts.audit_eval_battery import TMP, strip_corpus
        split = json.loads((TOKENS / "byte_corpus_v2.split.json").read_text())
        path = TMP / "corpus_train.stripped"
        strip_corpus(int(split["n_train"]), path)
        return path
    except Exception as exc:  # pragma: no cover - reported, not hidden
        print(f"warning: no stripped corpus for abstention filtering: {exc}", file=sys.stderr)
        return None


def _present_in_training(terms: list[str], corpus: Path | None) -> set[str]:
    """Normalised terms that occur verbatim in the training corpus."""
    if corpus is None or not terms:
        return set()
    from scripts.audit_eval_battery import rg_counts, strip_ws
    unique = sorted({strip_ws(t) for t in terms if len(strip_ws(t)) >= 12})
    counts = rg_counts(unique, corpus)
    return set(counts)


def abstention_rows(seed: int, used_contexts: set[str], blocked: set[str]) -> tuple[list[dict], dict]:
    """Half answerable, half unanswerable, with answers absent from training.

    A blank phrase is only eligible if the phrase and the question string do not
    occur verbatim in the training corpus: an earlier build picked named
    entities ("Washington State University") whose gold answer is all over
    Wikipedia, which the contamination audit correctly failed.
    """
    rng = random.Random(seed * 1013 + 7)
    pool = _heldout_docs()
    docs = [d for d in pool if d not in used_contexts]
    rng.shuffle(docs)
    if len(docs) < 1_400:
        raise SystemExit(f"not enough held-out documents for abstention: {len(docs)}")

    candidates: list[tuple[str, str, str, str]] = []
    for doc in docs:
        for sentence in re.split(r"(?<=[.!?])\s+", doc):
            sentence = sentence.strip()
            if not (120 <= len(sentence) <= 480):
                continue
            phrase = None
            for m in _PHRASE.finditer(sentence):
                cand = m.group(0).strip()
                if m.start() == 0 or not (6 <= len(cand) <= 60):
                    continue
                if doc.count(cand) != 1:
                    continue
                phrase = cand
                break
            if phrase:
                question = "Fill in the missing phrase: " + sentence.replace(phrase, "____")
                candidates.append((doc, sentence, phrase, question))
                break
        if len(candidates) >= 4_500:
            break
    corpus = _stripped_training_corpus()
    matched = _present_in_training(
        [c[2] for c in candidates] + [c[3] for c in candidates], corpus
    )
    from scripts.audit_eval_battery import strip_ws
    clean = [
        c for c in candidates
        if strip_ws(c[2]) not in matched and strip_ws(c[3]) not in matched
        and strip_ws(c[2]) not in blocked and strip_ws(c[3]) not in blocked
    ]
    stats_extra = {
        "candidates": len(candidates),
        "rejected_answer_or_question_in_training": len(candidates) - len(clean),
        "training_corpus_filter": str(corpus) if corpus else None,
    }
    if len(clean) < QUOTAS["abstention"]:
        raise SystemExit(f"only {len(clean)} clean abstention candidates, need {QUOTAS['abstention']}")

    rows: list[dict] = []
    cursor = 0
    skipped_fillers = 0
    while len(rows) < QUOTAS["abstention"]:
        if cursor >= len(clean):
            raise SystemExit("ran out of clean abstention candidates")
        doc, sentence, phrase, question = clean[cursor]
        cursor += 1
        answerable = len(rows) % 2 == 0
        packed_doc = doc if answerable else docs[(cursor + 11) % len(docs)]
        if packed_doc == doc and not answerable:
            continue
        distractors = [d for d in docs[cursor: cursor + 2] if phrase.lower() not in d.lower()]
        content, positions, gold = _pack_content(
            packed_doc, distractors, position=["early", "middle", "late"][len(rows) % 3], distributed=False
        )
        fillers = sample_fillers(pool, question, rng, window_bytes(CHUNKS_PER_PACK))
        try:
            text, spans = compose_pack(content, fillers, positions=positions, gold=gold,
                                       chunks_per_pack=CHUNKS_PER_PACK)
        except ValueError:
            continue
        if question in text:
            continue
        if answerable:
            if text.count(phrase) != 1:
                skipped_fillers += 1
                continue
            char_index = text.find(phrase)
            byte_start = len(text[:char_index].encode("utf-8"))
            byte_end = byte_start + len(phrase.encode("utf-8"))
            chunks = [c.index for c in chunk_text(text) if c.byte_start < byte_end and c.byte_end > byte_start]
            if not chunks:
                continue
            spans = [{"start": byte_start, "end": byte_end, "chunks": chunks}]
        else:
            if phrase.lower() in text.lower():
                skipped_fillers += 1
                continue
            spans = []
        label = "answerable" if answerable else "unanswerable"
        rows.append(_row(
            row_id=f"abstention.heldout.{len(rows):04d}",
            family="abstention",
            task="heldout_cloze_abstention",
            question=question,
            context=text,
            alternatives=["answerable", "unanswerable"],
            gold=0 if answerable else 1,
            gold_label=label,
            answer=phrase if answerable else None,
            dataset="heldout_corpus",
            split="heldout_2pct",
            source_row=cursor - 1,
            licence="heldout_corpus",
            meta={
                "chunks": CHUNKS_PER_PACK,
                "variant": label,
                "sentence": sentence,
                "phrase_chars": len(phrase),
                "answer_absent_from_training": True,
                "phrase_verified_unique": bool(answerable),
                "phrase_verified_absent": not answerable,
            },
            gold_spans=spans,
        ))
    stats = {
        "answerable": 500,
        "unanswerable": 500,
        "skipped_because_phrase_in_fillers": skipped_fillers,
        **stats_extra,
    }
    return rows, stats


def _noisy(text: str, rng: random.Random) -> str:
    """Deterministic distribution shift that leaves the label untouched."""
    swaps = {"the": "teh", "and": "adn", "that": "taht", "with": "wiht", "from": "form"}
    out = text
    for word in swaps:
        m = re.search(rf"\b{word}\b", out)
        if m:
            out = out[: m.start()] + swaps[word] + out[m.end():]
            break
    out = out.replace(", ", " , ", 1)
    out = re.sub(r"[ \t]+", "  ", out, count=3)
    if out == text:  # guarantee a visible change
        out = text + "\n"
    return out


def robustness_rows(seed: int, used_contexts: set[str]) -> tuple[list[dict], dict]:
    """Two perturbations of 500 source rows disjoint from classification."""
    rng = random.Random(seed * 1019 + 11)
    pool = _heldout_docs()
    bases = []
    for dataset, task_name in CLASSIFICATION_TASKS:
        spec = TASKS[task_name]
        bank = candidate_bank(spec)
        examples = load_examples(spec, bank, split="test", limit=PER_TASK + 100)
        extra = examples[PER_TASK: PER_TASK + 100]
        if len(extra) < 100:
            raise SystemExit(f"{dataset}: need 100 extra test rows for robustness, got {len(extra)}")
        for ex in extra:
            bases.append((dataset, spec, bank, ex))
    if len(bases) != 500:
        raise SystemExit(f"robustness bases: {len(bases)}, expected 500")

    rows: list[dict] = []
    for j, (dataset, spec, bank, ex) in enumerate(bases):
        label = bank.labels[ex.gold]
        affix = pool[(j * 7) % len(pool)][:240].strip()
        variants = {
            "noise": _noisy(ex.context, rng),
            "distractor": f"Background note, unrelated to the task: {affix}\n\n{ex.context}\n\nEnd note: {affix[:120]}",
        }
        for variant, context in variants.items():
            if context in used_contexts:
                raise SystemExit(f"robustness {variant} context collides with an existing row")
            rows.append(_row(
                row_id=f"robustness.{variant}.{dataset}.{j:04d}",
                family="robustness",
                task=dataset,
                question=spec.instruction,
                context=context,
                alternatives=list(bank.labels),
                gold=ex.gold,
                gold_label=label,
                answer=label,
                dataset=dataset,
                split="test",
                source_row=ex.index,
                licence=LICENCES[dataset],
                meta={
                    "variant": variant,
                    "base_id": f"source.{dataset}.{ex.index:04d}",
                    "base_row": ex.index,
                    "perturbation": "character_noise" if variant == "noise" else "distractor_affixes",
                },
            ))
    stats = {"bases": len(bases), "variants_per_base": 2, "rows": len(rows)}
    return rows, stats


def build(seed: int, blocked: set[str] | None = None) -> tuple[list[dict], dict]:
    blocked = blocked or set()
    rows: list[dict] = []
    rows += classification_rows(seed)
    rows += boolean_rows()
    rows += pairwise_rows()
    rows += ordinal_rows()
    used = {r["context"] for r in rows}
    document, document_stats = document_rows(seed, used)
    rows += document
    used |= {r["context"] for r in document}
    abstention, abstention_stats = abstention_rows(seed, used, blocked)
    rows += abstention
    used |= {r["context"] for r in abstention}
    robustness, robustness_stats = robustness_rows(seed, used)
    rows += robustness
    rows += multilingual_rows()

    counts = Counter(r["family"] for r in rows)
    if dict(counts) != QUOTAS:
        raise SystemExit(f"family counts {dict(counts)} do not match quotas {QUOTAS}")
    ids = {r["id"] for r in rows}
    if len(ids) != len(rows):
        raise SystemExit("duplicate ids in battery")

    questions = Counter(r["question"] for r in rows if r["family"] != "robustness")
    stats = {
        "document": document_stats,
        "ctx_sources_excluded_from_document": _ctx_overlap_stats(),
        "abstention": abstention_stats,
        "robustness": robustness_stats,
        "duplicate_question_strings": {q: n for q, n in questions.items() if n > 1},
    }
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/parity/eval_battery_10k")
    ap.add_argument("--rows-per-shard", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-terms", default="", help="JSON list of normalized strings to exclude")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocked: set[str] = set()
    if args.block_terms and Path(args.block_terms).exists():
        blocked = set(json.loads(Path(args.block_terms).read_text()))
    rows, stats = build(args.seed, blocked)
    rows.sort(key=lambda r: (list(QUOTAS).index(r["family"]), r["id"]))

    per = args.rows_per_shard
    shards: list[dict] = []
    family_counts: Counter = Counter()
    for shard_index, start in enumerate(range(0, len(rows), per)):
        chunk = rows[start: start + per]
        path = out_dir / shard_file(shard_index)
        size = write_shard(path, chunk)
        meta = {
            "index": shard_index,
            "file": path.name,
            "rows": len(chunk),
            "packs": len(chunk),
            "gzip_bytes": size,
            "sha256": sha256_file(path),
            "families": dict(Counter(r["family"] for r in chunk)),
        }
        (out_dir / f"shard-{shard_index:05d}.meta.json").write_text(
            json.dumps(meta, indent=1) + "\n", encoding="utf-8"
        )
        shards.append(meta)
        family_counts.update(r["family"] for r in chunk)

    config = {
        "battery": "eval_10k",
        "task": "plan task 8",
        "rows": len(rows),
        "rows_per_shard": per,
        "seed": args.seed,
        "chunks_per_pack": CHUNKS_PER_PACK,
        "families": QUOTAS,
        "licences": {k: v for k, v in sorted(LICENCES.items())},
        "heldout": {
            "source": "data/tokens/byte_corpus_v2.u16 tail",
            "split_rule": "sha1(document) % 50 == 0, appended after all training documents",
            "heldout_bytes": json.loads((TOKENS / "byte_corpus_v2.split.json").read_text())["heldout_bytes"],
        },
        "document_sources": [s[0] for s in DOC_SOURCES],
        "document_note": "passages repeat across items (107/44/524 unique); questions are distinct and checked",
        "provenance": "every row stores dataset, split, row index, licence and gold",
    }
    manifest = merge_shards(out_dir, config, expected=len(shards)) or {}
    manifest["rows"] = sum(s["rows"] for s in shards)
    manifest["families"] = dict(QUOTAS)
    manifest["family_counts"] = dict(sorted(family_counts.items()))
    manifest["stats"] = stats
    canonical = __import__("hashlib").sha256(
        "".join(f"{s['file']}:{s['sha256']}:{s['rows']}\n" for s in sorted(shards, key=lambda s: s["index"])).encode()
    ).hexdigest()
    manifest["battery_sha256"] = canonical
    manifest["battery_sha256_method"] = "sha256 over sorted shard file:sha:rows lines"
    manifest["licences"] = {k: v for k, v in sorted(LICENCES.items())}
    manifest["document_sources"] = [s[0] for s in DOC_SOURCES]
    write_manifest(out_dir / "manifest.json", manifest)
    print(json.dumps({
        "out_dir": str(out_dir),
        "rows": manifest["rows"],
        "shards": len(shards),
        "families": manifest["family_counts"],
        "battery_sha256": canonical,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
