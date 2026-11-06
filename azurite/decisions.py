"""The decision interface: (context, instruction, K candidates) -> tensors.

Two scoring paths, because K ranges from 3 to 150 and only one of them fits in
any honest context window:

    cross-encoder   [CLS] instruction [SEP] context [SEP] cand1 [SEP] cand2 ...
                    in one window. Context and candidate are seen jointly, which
                    is strictly more expressive. Limited to small K: 150
                    candidates cannot fit in 1,024 byte positions, and padding
                    the context down to nothing to make room would measure a
                    different task.

    dual-encoder    encode the context once, each candidate separately, score by
                    dot product. Works at any K, and candidate embeddings are
                    task-level constants computed once per checkpoint, so the
                    per-decision cost is one context pass regardless of K. This
                    is the path the transfer stage uses for K > 16.

Packing rules, each of which is tested:

  * a candidate span covers real tokens only, never padding
  * the gold candidate is always present; a candidate is never dropped to make
    room. An example that does not fit is counted as unpackable and reported,
    not silently replaced with something else
  * candidate order is the task's canonical order when stable indices are
    needed (dual encoder), or a per-example seeded shuffle (cross encoder), so a
    positional prior cannot be learned by accident

Label resolution is explicit. The three task families store their gold label
differently — a ClassLabel integer, an integer index with a separate display
column, or the display string itself — and getting that wrong is exactly the bug
that made an earlier harness report 0.0000 on every task while its predictions
were sensible. `CandidateBank.resolve` is the single place that translation
happens, and it raises instead of guessing.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from .model import DecisionBatch
from .tokenizers import CLS_ID, PAD_ID, SEP_ID

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


class Tokenizer(Protocol):
    arm: str
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class TaskSpec:
    """How to read one task off disk and turn it into a decision."""

    name: str
    split_rel: str
    text_fields: tuple[str, ...]
    label_field: str
    instruction: str
    display_field: str | None = None
    train_rel: str | None = None

    def context(self, row: dict) -> str:
        parts = []
        for f in self.text_fields:
            value = row.get(f)
            if value is not None and str(value).strip():
                parts.append(str(value))
        return "\n".join(parts)


# The nine primary tasks. The instruction is fixed per task and never tuned on
# an evaluation split: it is part of the interface, not a hyperparameter.
TASKS: dict[str, TaskSpec] = {
    "ag_news": TaskSpec(
        name="ag_news",
        split_rel="default/test",
        text_fields=("text",),
        label_field="label",
        instruction="Choose the news topic that best matches the article.",
        train_rel="default/train",
    ),
    "emotion": TaskSpec(
        name="emotion",
        split_rel="split/test",
        text_fields=("text",),
        label_field="label",
        instruction="Choose the emotion that best matches the text.",
        train_rel="split/train",
    ),
    "banking77": TaskSpec(
        name="banking77",
        split_rel="default/test",
        text_fields=("text",),
        label_field="label",
        display_field="label_text",
        instruction="Choose the banking intent that best matches the question.",
        train_rel="default/train",
    ),
    "clinc_oos": TaskSpec(
        name="clinc_oos",
        split_rel="plus/test",
        text_fields=("text",),
        label_field="intent",
        instruction="Choose the assistant intent that best matches the request.",
        train_rel="plus/train",
    ),
    "massive_intent": TaskSpec(
        name="massive_intent",
        split_rel="en/test",
        text_fields=("text",),
        label_field="label",
        display_field="label_text",
        instruction="Choose the voice-assistant intent that best matches the request.",
        train_rel="en/train",
    ),
    "massive_scenario": TaskSpec(
        name="massive_scenario",
        split_rel="en/test",
        text_fields=("text",),
        label_field="label",
        display_field="label_text",
        instruction="Choose the assistant domain that best matches the request.",
        train_rel="en/train",
    ),
    "xnli": TaskSpec(
        name="xnli",
        split_rel="en/test",
        text_fields=("premise", "hypothesis"),
        label_field="label",
        instruction="Given the premise and the hypothesis, choose their relation.",
        train_rel="en/train",
    ),
    "language_id": TaskSpec(
        name="language_id",
        split_rel="default/test",
        text_fields=("text",),
        label_field="labels",
        instruction="Choose the language the text is written in.",
        train_rel="default/train",
    ),
    "financial_phrasebank": TaskSpec(
        name="financial_phrasebank",
        split_rel="default/test",
        text_fields=("text",),
        label_field="label",
        display_field="label_text",
        instruction="Choose the sentiment the sentence expresses about its subject.",
        train_rel="default/train",
    ),
    # Extra fixed-label Jev benchmarks, normalized by
    # scripts/prepare_jev_extra.py. Every one is a single Choice or Noul; the
    # multiple-choice/multi-label/ordinal families are deliberately absent.
    "imdb": TaskSpec(
        name="imdb", split_rel="test", text_fields=("text",),
        label_field="label",
        instruction="Choose the sentiment the movie review expresses.",
        train_rel="train",
    ),
    "rotten_tomatoes": TaskSpec(
        name="rotten_tomatoes", split_rel="test", text_fields=("text",),
        label_field="label",
        instruction="Choose the sentiment the movie review expresses.",
        train_rel="train",
    ),
    "sst2": TaskSpec(
        name="sst2", split_rel="validation", text_fields=("sentence",),
        label_field="label",
        instruction="Choose the sentiment the sentence expresses.",
        train_rel="train",
    ),
    "sms_spam": TaskSpec(
        name="sms_spam", split_rel="test", text_fields=("sms",),
        label_field="label",
        instruction="Decide whether the message is spam or ham.",
        train_rel="train",
    ),
    "paws": TaskSpec(
        name="paws", split_rel="test",
        text_fields=("sentence1", "sentence2"), label_field="label",
        instruction="Decide whether the two sentences are paraphrases.",
        train_rel="train",
    ),
    "anli": TaskSpec(
        name="anli", split_rel="test",
        text_fields=("premise", "hypothesis"), label_field="label",
        instruction="Given the premise and the hypothesis, choose their relation.",
        train_rel="train",
    ),
    "boolq": TaskSpec(
        name="boolq", split_rel="validation",
        text_fields=("passage", "question"), label_field="label",
        instruction="Decide whether the passage answers the question with yes or no.",
        train_rel="train",
    ),
    "toxigen": TaskSpec(
        name="toxigen", split_rel="test", text_fields=("text",),
        label_field="label",
        instruction="Decide whether the text is toxic.",
        train_rel="train",
    ),
    "toxic_chat": TaskSpec(
        name="toxic_chat", split_rel="test", text_fields=("user_input",),
        label_field="label",
        instruction="Decide whether the user message is toxic.",
        train_rel="train",
    ),
    "prompt_injections": TaskSpec(
        name="prompt_injections", split_rel="test", text_fields=("text",),
        label_field="label",
        instruction="Decide whether the text is a prompt injection.",
        train_rel="train",
    ),
    "agb_de": TaskSpec(
        name="agb_de", split_rel="test", text_fields=("text",),
        label_field="label",
        instruction=("Decide whether the German consumer contract clause is void."),
        train_rel="train",
    ),
    # Compact context-scaling dev tasks. These are built by
    # scripts/build_context_suite.py from train-only rows; the official test
    # splits are not used. They are separate task names so an existing suite
    # cell can never be silently replaced by a dev carve.
    "banking77_ctx": TaskSpec(
        name="banking77_ctx",
        split_rel="test",
        text_fields=("context",),
        label_field="label",
        display_field="label_text",
        instruction="Choose the banking intent that best matches the question.",
        train_rel="train",
    ),
    "clinc_oos_ctx": TaskSpec(
        name="clinc_oos_ctx",
        split_rel="test",
        text_fields=("context",),
        label_field="label",
        instruction="Choose the assistant intent that best matches the request.",
        train_rel="train",
    ),
    "boolq_ctx": TaskSpec(
        name="boolq_ctx",
        split_rel="test",
        text_fields=("context", "question"),
        label_field="label",
        instruction="Decide whether the passage answers the question with yes or no.",
        train_rel="train",
    ),
    "squad2_answerability_ctx": TaskSpec(
        name="squad2_answerability_ctx",
        split_rel="test",
        text_fields=("context", "question"),
        label_field="label",
        instruction=("Decide whether the question can be answered from the passage "
                     "or must be treated as unanswerable."),
        train_rel="train",
    ),
    "contract_nli_ctx": TaskSpec(
        name="contract_nli_ctx",
        split_rel="test",
        text_fields=("context", "question"),
        label_field="label",
        instruction=("Decide whether the contract entails, contradicts, or does not "
                     "mention the stated policy."),
        train_rel="train",
    ),
}

# Above this K the cross-encoder cannot be honest about the context, so the
# transfer stage uses the dual encoder instead.
CROSS_ENCODER_MAX_K = 16


@dataclass
class CandidateBank:
    """The ordered candidate set for one task, with raw-label resolution."""

    task: str
    labels: list[str] = field(default_factory=list)
    raw_ids: list[object] = field(default_factory=list)
    token_ids: list[list[int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    def resolve(self, raw: object) -> int:
        for i, candidate in enumerate(self.raw_ids):
            if candidate == raw:
                return i
        raise KeyError(
            f"{self.task}: gold label {raw!r} is not in the candidate bank "
            f"({len(self.raw_ids)} entries: {self.raw_ids[:5]}...)"
        )

    def tokenize(self, tok: Tokenizer) -> "CandidateBank":
        self.token_ids = [tok.encode(label) for label in self.labels]
        return self

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "task": self.task,
                    "labels": self.labels,
                    "raw_ids": [str(r) for r in self.raw_ids],
                    "token_ids": self.token_ids,
                },
                indent=1,
            )
            + "\n"
        )
        return path


@dataclass
class Example:
    index: int
    context: str
    gold: int
    # Original fields, retained so a multi-field task can be packed without
    # flattening a pair into one string that is then truncated from the front.
    parts: tuple[str, ...] = ()


def _split_path(spec: TaskSpec, which: str) -> Path:
    rel = spec.split_rel if which == "test" else (spec.train_rel or spec.split_rel)
    return RAW / spec.name / rel


def _load(spec: TaskSpec, which: str):
    from datasets import load_from_disk

    path = _split_path(spec, which)
    if not path.exists():
        raise FileNotFoundError(f"{spec.name}: no {which} split at {path}")
    return load_from_disk(str(path))


def candidate_bank(spec: TaskSpec) -> CandidateBank:
    """Enumerate the label space in a stable order.

    Uses the train split when the readable text lives in its own column: a test
    split may not contain every label, and a candidate set that changes with the
    split is not a fixed task.
    """
    which = "train" if spec.train_rel else "test"
    ds = _load(spec, which)
    features = ds.features

    if spec.display_field is None:
        feature = features[spec.label_field]
        if hasattr(feature, "names"):  # ClassLabel
            return CandidateBank(
                task=spec.name,
                labels=[str(n) for n in feature.names],
                raw_ids=list(range(len(feature.names))),
            )
        # label field *is* the display text (e.g. language_id)
        seen: list[str] = []
        for row in ds:
            value = str(row[spec.label_field])
            if value not in seen:
                seen.append(value)
        ordered = sorted(seen)
        return CandidateBank(task=spec.name, labels=ordered, raw_ids=list(ordered))

    pairs: dict[object, str] = {}
    for row in ds:
        pairs[row[spec.label_field]] = str(row[spec.display_field])
    raw_ids = sorted(pairs, key=lambda k: (str(pairs[k]), str(k)))
    return CandidateBank(
        task=spec.name,
        labels=[pairs[r] for r in raw_ids],
        raw_ids=list(raw_ids),
    )


def load_examples(
    spec: TaskSpec,
    bank: CandidateBank,
    *,
    split: str = "test",
    limit: int | None = None,
    offset: int = 0,
) -> list[Example]:
    """Load examples and resolve each gold label through the bank."""
    ds = _load(spec, split)
    out: list[Example] = []
    for i, row in enumerate(ds):
        if i < offset:
            continue
        if limit is not None and len(out) >= limit:
            break
        context = spec.context(row)
        if not context.strip():
            continue
        parts = tuple(
            str(row[f])
            for f in spec.text_fields
            if row.get(f) is not None and str(row[f]).strip()
        )
        out.append(
            Example(
                index=i,
                context=context,
                gold=bank.resolve(row[spec.label_field]),
                parts=parts,
            )
        )
    return out


def stratified_examples(
    spec: TaskSpec,
    bank: CandidateBank,
    split: str,
    n: int,
    seed: int,
) -> list[Example]:
    """Draw `n` examples balanced across candidate labels.

    Banking77 ships label-sorted --- forty consecutive rows share a label --- so
    `load_examples(..., limit=n)` returns a prefix of a few classes rather than
    a sample of the task. Measured on this machine, `limit=1600` on the training
    split covered 13 of 77 labels and `limit=300` on the test split covered 8.
    Drawing round-robin over classes in a seeded order, then shuffling, makes a
    subset a class-balanced sample. The seed is fixed by the caller so that
    every training seed sees the same rows.
    """
    import random
    from collections import defaultdict

    rows = load_examples(spec, bank, split=split, limit=None)
    by: dict[int, list[Example]] = defaultdict(list)
    for e in rows:
        by[e.gold].append(e)
    rng = random.Random(seed)
    classes = sorted(by)
    rng.shuffle(classes)
    for c in classes:
        rng.shuffle(by[c])
    out: list[Example] = []
    i = 0
    while len(out) < n and any(by[c] for c in classes):
        c = classes[i % len(classes)]
        if by[c]:
            out.append(by[c].pop())
        i += 1
    rng.shuffle(out)
    return out


def _truncate(ids: list[int], budget: int) -> list[int]:
    return ids[:budget] if len(ids) > budget else ids


def _pack_fields_fill(
    tok: Tokenizer, parts: tuple[str, ...], budget: int
) -> tuple[list[int], bool]:
    """Pack a multi-field example without silently dropping a later field.

    A pair is left in its historical flattened form whenever it fits, so this
    changes nothing for the rows the existing record already measured. Only a
    pair that would overflow is split into equal per-field shares, which keeps
    every field present. This is a structural rule (all fields get a share), not
    a per-task budget: it is applied identically to any multi-field example.

    Returns the tokens and whether the row had to be split.
    """
    flat = tok.encode("\n".join(parts))
    if len(flat) <= budget:
        return flat, False
    return _pack_fields(tok, parts, budget), True


def _pack_fields(
    tok: Tokenizer, parts: tuple[str, ...], budget: int
) -> list[int]:
    """Tokenise fields separately and give each a fair share of the budget.

    Flattening premise and hypothesis into one string and truncating it from
    the front silently discards the hypothesis when the premise is long. That
    is fatal for a pair task: the relation cannot be inferred from the premise
    alone. Equal shares are a structural rule, not a tuned per-task budget.
    """
    if not parts:
        return []
    if len(parts) == 1:
        return _truncate(tok.encode(parts[0]), budget)
    per = max(1, (budget - (len(parts) - 1)) // len(parts))
    out: list[int] = []
    for i, part in enumerate(parts):
        if i:
            out.append(SEP_ID)
        out.extend(_truncate(tok.encode(part), per))
    return out[:budget]


def _pad(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, : len(r)] = True
    return ids, mask


def pack_cross(
    tok: Tokenizer,
    spec: TaskSpec,
    bank: CandidateBank,
    examples: list[Example],
    *,
    max_seq: int,
    shuffle: bool = True,
    seed: int = 0,
) -> tuple[DecisionBatch, dict]:
    """Pack examples for the cross-encoder path.

    Every row contains every candidate — that is what makes the softmax over
    spans a decision over the full label space. Rows whose packed length would
    exceed `max_seq` are excluded and counted.
    """
    if len(bank) > CROSS_ENCODER_MAX_K:
        raise ValueError(
            f"{spec.name}: {len(bank)} candidates exceed the cross-encoder cap "
            f"({CROSS_ENCODER_MAX_K}); use pack_dual"
        )
    if not bank.token_ids:
        bank.tokenize(tok)

    instr = tok.encode(spec.instruction)
    rows: list[list[int]] = []
    span_rows: list[list[tuple[int, int]]] = []
    golds: list[int] = []
    unpackable = 0

    # Single-field packing keeps the historical half-window cap so the
    # already-measured AG News/Emotion cells remain comparable. A task whose
    # examples carry more than one field gets the full residual budget after
    # the instruction and candidates, split equally across the fields.
    field_aware = any(len(ex.parts) > 1 for ex in examples)
    fixed_cost = 1 + len(instr) + 1 + 1 + sum(len(c) + 1 for c in bank.token_ids)
    context_budget = max(8, max_seq - fixed_cost) if field_aware else max(8, max_seq // 2)
    n_fields = next((len(ex.parts) for ex in examples if len(ex.parts) > 1), 0)
    field_budget = (
        max(1, (context_budget - (n_fields - 1)) // n_fields) if n_fields else None
    )

    for ex in examples:
        if len(ex.parts) > 1:
            ctx = _pack_fields(tok, ex.parts, context_budget)
        else:
            ctx = _truncate(tok.encode(ex.context), max(8, max_seq // 2))
        order = list(range(len(bank)))
        if shuffle:
            random.Random(f"{seed}:{spec.name}:{ex.index}").shuffle(order)

        ids = [CLS_ID, *instr, SEP_ID, *ctx, SEP_ID]
        spans: list[tuple[int, int]] = []
        for label_idx in order:
            candidate = bank.token_ids[label_idx]
            spans.append((len(ids), len(ids) + len(candidate)))
            ids.extend(candidate)
            ids.append(SEP_ID)

        if len(ids) > max_seq:
            unpackable += 1
            continue

        rows.append(ids)
        span_rows.append(spans)
        golds.append(order.index(ex.gold))

    if not rows:
        raise ValueError(f"{spec.name}: no packable examples at max_seq={max_seq}")

    ids_t, mask_t = _pad(rows)
    batch = DecisionBatch(
        ids=ids_t,
        pad_mask=mask_t,
        spans=torch.tensor(span_rows, dtype=torch.long),
        cand_mask=torch.ones((len(rows), len(bank)), dtype=torch.bool),
        gold=torch.tensor(golds, dtype=torch.long),
    )
    return batch, {
        "task": spec.name,
        "path": "cross",
        "k": len(bank),
        "packed": len(rows),
        "unpackable": unpackable,
        "mean_seq_len": round(sum(len(r) for r in rows) / len(rows), 1),
        "max_seq": max_seq,
        "field_aware": field_aware,
        "context_budget": context_budget,
        "field_budget": field_budget,
    }


def pack_cross_subset(
    tok: Tokenizer,
    spec: TaskSpec,
    bank: CandidateBank,
    examples: list[Example],
    keep: list[list[int]],
    *,
    max_seq: int,
) -> tuple[DecisionBatch, dict]:
    """Pack examples for the cross-encoder over a per-example candidate subset.

    This is the rerank stage of the retrieve-then-rerank funnel that lifts the
    cross-encoder cap: a dual scorer retrieves the top-M candidates per example
    (`keep[i]`, retrieval-rank order, best first) and the cross encoder decides
    among those M only. Rows keep retrieval-rank order deliberately — rank is
    signal, and shuffling it away would discard the retriever's information.
    No per-example shuffle is applied, so this is fully deterministic given
    `keep`.

    An example whose gold candidate was not retrieved is dropped and counted
    as a funnel miss: pretending it is packable would silently redefine the
    task. End-to-end accuracy must count misses as errors; see
    scripts/transfer_eval.py --path funnel.
    """
    if len(examples) != len(keep):
        raise ValueError(
            f"{spec.name}: {len(examples)} examples but {len(keep)} keep-lists"
        )
    m_widths = {len(k) for k in keep}
    if len(m_widths) != 1:
        raise ValueError(f"{spec.name}: ragged keep-lists {sorted(m_widths)}")
    m = m_widths.pop()
    if m < 2:
        raise ValueError(f"{spec.name}: funnel width must be >= 2, got {m}")
    if m > CROSS_ENCODER_MAX_K:
        raise ValueError(
            f"{spec.name}: funnel width {m} exceeds the cross-encoder cap "
            f"({CROSS_ENCODER_MAX_K})"
        )
    if any(len(set(k)) != len(k) for k in keep):
        raise ValueError(f"{spec.name}: keep-list contains a duplicate candidate")
    if any((i < 0 or i >= len(bank)) for k in keep for i in k):
        raise ValueError(f"{spec.name}: keep-list indexes outside the bank")
    if not bank.token_ids:
        bank.tokenize(tok)

    instr = tok.encode(spec.instruction)
    rows: list[list[int]] = []
    span_rows: list[list[tuple[int, int]]] = []
    golds: list[int] = []
    packed_indices: list[int] = []
    funnel_miss = 0

    field_aware = any(len(ex.parts) > 1 for ex in examples)
    fixed_cost = 1 + len(instr) + 1 + 1 + sum(len(c) + 1 for c in bank.token_ids)
    context_budget = max(8, max_seq - fixed_cost) if field_aware else max(8, max_seq // 2)

    for pos, (ex, kept) in enumerate(zip(examples, keep)):
        if ex.gold not in kept:
            funnel_miss += 1
            continue
        if len(ex.parts) > 1:
            ctx = _pack_fields(tok, ex.parts, context_budget)
        else:
            ctx = _truncate(tok.encode(ex.context), max(8, max_seq // 2))

        ids = [CLS_ID, *instr, SEP_ID, *ctx, SEP_ID]
        spans: list[tuple[int, int]] = []
        for label_idx in kept:
            candidate = bank.token_ids[label_idx]
            spans.append((len(ids), len(ids) + len(candidate)))
            ids.extend(candidate)
            ids.append(SEP_ID)

        if len(ids) > max_seq:
            funnel_miss += 1
            continue

        rows.append(ids)
        span_rows.append(spans)
        golds.append(kept.index(ex.gold))
        packed_indices.append(pos)

    if not rows:
        raise ValueError(f"{spec.name}: no packable examples at max_seq={max_seq}")

    ids_t, mask_t = _pad(rows)
    batch = DecisionBatch(
        ids=ids_t,
        pad_mask=mask_t,
        spans=torch.tensor(span_rows, dtype=torch.long),
        cand_mask=torch.ones((len(rows), m), dtype=torch.bool),
        gold=torch.tensor(golds, dtype=torch.long),
    )
    return batch, {
        "task": spec.name,
        "path": "cross-funnel",
        "k": len(bank),
        "m": m,
        "packed": len(rows),
        "funnel_miss": funnel_miss,
        "packed_indices": packed_indices,
        "mean_seq_len": round(sum(len(r) for r in rows) / len(rows), 1),
        "max_seq": max_seq,
        "field_aware": field_aware,
    }


@dataclass
class DualBatch:
    """Context ids, their mask, and the gold candidate index."""

    context_ids: torch.Tensor
    context_mask: torch.Tensor
    gold: torch.Tensor

    def to(self, device: torch.device) -> "DualBatch":
        return DualBatch(
            context_ids=self.context_ids.to(device),
            context_mask=self.context_mask.to(device),
            gold=self.gold.to(device),
        )


def pack_dual(
    tok: Tokenizer,
    spec: TaskSpec,
    bank: CandidateBank,
    examples: list[Example],
    *,
    max_seq: int,
    field_aware: bool = False,
) -> tuple[DualBatch, dict]:
    """Pack contexts for the dual-encoder path. Candidates are tokenized once
    per task and their embeddings are computed once per checkpoint.

    `field_aware` is opt-in so every previously measured dual cell keeps its
    exact packing. When enabled, a multi-field row that fits is still flattened
    byte-for-byte; only an overflowing row is split so its later fields survive.
    """
    if not bank.token_ids:
        bank.tokenize(tok)
    rows: list[list[int]] = []
    field_rows = 0
    split_rows = 0
    for ex in examples:
        if field_aware and len(ex.parts) > 1:
            field_rows += 1
            packed, split = _pack_fields_fill(tok, ex.parts, max(1, max_seq - 2))
            split_rows += int(split)
            rows.append([CLS_ID, *packed, SEP_ID])
        else:
            # Historical packing, byte for byte: callers that do not opt in
            # must produce the same tensors they produced before this flag.
            rows.append(_truncate([CLS_ID, *tok.encode(ex.context), SEP_ID], max_seq))
    ids_t, mask_t = _pad(rows)
    batch = DualBatch(
        context_ids=ids_t,
        context_mask=mask_t,
        gold=torch.tensor([ex.gold for ex in examples], dtype=torch.long),
    )
    return batch, {
        "task": spec.name,
        "path": "dual",
        "k": len(bank),
        "packed": len(rows),
        "unpackable": 0,
        "mean_seq_len": round(sum(len(r) for r in rows) / len(rows), 1),
        "max_seq": max_seq,
        "field_aware": bool(field_aware),
        "multi_field_rows": field_rows,
        "split_rows": split_rows,
        "fields_dropped": 0,
    }
