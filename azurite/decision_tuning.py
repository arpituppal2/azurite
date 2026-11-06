"""Candidate-conditioned decision tuning for the dual head.

This module is the testable implementation behind ``scripts/train_decision.py``.
It keeps the existing ``AzuriteDual`` head and candidate bank, and adds the
task-7 training controls:

* candidate dropout (the gold candidate is never dropped);
* label paraphrases, with deterministic per-batch selection;
* hard negatives ranked by BM25 over the candidate labels;
* a train-split-only carve, with a disjoint validation carve;
* reader freeze, warmup-then-unfreeze, or full unfreeze;
* resumable model/optimizer/RNG state; and
* a held-out validation gate whose rows are written one per example.

The validation gate is deliberately strict: it does not infer evidence
precision. If a caller has no per-row evidence measurement, it passes ``None``
and the checkpoint is not accepted. That is better than manufacturing a pass
from decision accuracy alone.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .decisions import (
    CandidateBank,
    Example,
    TaskSpec,
    load_examples,
    pack_dual,
)
from .model import AzuriteDual
from .retrieval import BM25
from .tokenizers import PAD_ID

DEFAULT_TASKS = (
    "banking77",
    "clinc_oos",
    "massive_intent",
    "ag_news",
    "emotion",
    "boolq",
    "xnli",
    "financial_phrasebank",
)

FAMILY_BY_TASK = {
    "banking77": "classification",
    "clinc_oos": "classification",
    "massive_intent": "classification",
    "massive_scenario": "classification",
    "ag_news": "classification",
    "emotion": "classification",
    "boolq": "boolean",
    "xnli": "pairwise",
    "anli": "pairwise",
    "paws": "pairwise",
    "financial_phrasebank": "ordinal",
    "language_id": "multilingual",
    "massive_intent_de": "multilingual",
}


def label_variants(label: str) -> list[str]:
    """Small, deterministic paraphrase set for a candidate label.

    These are interface variants, not a claim that they are semantic
    paraphrases of every task. The point is to stop the head from treating one
    exact display string as the only surface form of a label.
    """
    base = str(label).strip()
    if not base:
        return [base]
    out = [base]

    def add(value: str) -> None:
        if value and value not in out:
            out.append(value)

    add(base.replace("_", " "))
    add(base.replace("-", " "))
    add(base[:1].upper() + base[1:] if base[:1].islower() else base[:1].lower() + base[1:])
    add(base + ".")
    add(f"the {base}")
    add(f"a {base}")
    return out


def paraphrase_labels(
    labels: Sequence[str], rng: random.Random, *, probability: float = 0.25
) -> list[str]:
    """Choose a label variant per candidate without changing candidate order."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("paraphrase probability must be in [0, 1]")
    out: list[str] = []
    for label in labels:
        variants = label_variants(label)
        if probability and len(variants) > 1 and rng.random() < probability:
            out.append(rng.choice(variants))
        else:
            out.append(str(label))
    return out


def hard_negative_indices(
    context: str,
    labels: Sequence[str],
    gold: int,
    *,
    k: int = 8,
) -> list[int]:
    """Rank non-gold candidates by BM25 similarity to the context.

    BM25 is used only to choose hard negatives; it is not a candidate scorer in
    the model's loss. The ranking is deterministic.
    """
    if not labels:
        return []
    if not 0 <= gold < len(labels):
        raise ValueError(f"gold {gold} outside {len(labels)} candidates")
    if k <= 0:
        return []
    index = BM25([{"id": i, "text": str(label)} for i, label in enumerate(labels)])
    return [i for i in index.rank(context) if i != gold][:k]


def select_candidate_indices(
    context: str,
    labels: Sequence[str],
    gold: int,
    rng: random.Random,
    *,
    dropout: float = 0.25,
    hard_negatives: int = 8,
    max_candidates: int = 64,
) -> list[int]:
    """Return the candidate indices for one example.

    The gold index is always present. Hard negatives are retained first, then
    the remaining candidates are kept with probability ``1 - dropout`` until
    the cap. At least two candidates are kept when the bank has two or more, so
    the softmax never degenerates to a one-class problem.
    """
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("candidate dropout must be in [0, 1]")
    n = len(labels)
    if n == 0:
        raise ValueError("candidate bank is empty")
    if not 0 <= gold < n:
        raise ValueError(f"gold {gold} outside {n} candidates")
    max_keep = max(2, min(int(max_candidates), n))
    if n == 1:
        return [0]

    keep: set[int] = {gold}
    for index in hard_negative_indices(context, labels, gold, k=hard_negatives):
        if len(keep) >= max_keep:
            break
        keep.add(index)

    optional = [i for i in range(n) if i not in keep]
    rng.shuffle(optional)
    for index in optional:
        if len(keep) >= max_keep:
            break
        if rng.random() >= dropout:
            keep.add(index)
    if len(keep) < 2:
        for index in optional:
            if index not in keep:
                keep.add(index)
                break
    return sorted(keep)


def split_train_validation(
    examples: Sequence[Example], *, fraction: float = 0.1, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Shuffle a train-split carve into disjoint training and validation rows."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must be in (0, 1)")
    rows = list(examples)
    if len(rows) < 2:
        return rows, []
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n_validation = max(1, min(len(rows) - 1, int(round(len(rows) * fraction))))
    validation_ids = set(order[:n_validation])
    train = [row for i, row in enumerate(rows) if i not in validation_ids]
    validation = [row for i, row in enumerate(rows) if i in validation_ids]
    return train, validation


def load_train_validation(
    spec: TaskSpec,
    bank: CandidateBank,
    *,
    limit: int | None = None,
    fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[Example], list[Example]]:
    """Load only the source train split, then carve validation from it."""
    rows = load_examples(spec, bank, split="train", limit=limit)
    return split_train_validation(rows, fraction=fraction, seed=seed)


def set_reader_trainable(dual: AzuriteDual, trainable: bool) -> int:
    """Set every reader parameter's requires_grad and return the count changed."""
    encoder = getattr(dual, "encoder", dual)
    changed = 0
    for param in encoder.parameters():
        if bool(param.requires_grad) != bool(trainable):
            changed += 1
        param.requires_grad = bool(trainable)
    return changed


def build_optimizer(
    dual: AzuriteDual,
    *,
    head_lr: float = 1e-3,
    reader_lr: float = 3e-5,
    reader_mode: str = "warmup",
) -> torch.optim.Optimizer:
    """Build the two-rate optimizer matching the reader freeze mode.

    ``warmup`` includes the reader group at lr 0 and leaves it frozen. The
    training loop calls :func:`activate_reader` at the configured step, which
    flips requires_grad and enables the reader learning rate without rebuilding
    the optimizer.
    """
    if reader_mode not in ("freeze", "warmup", "unfreeze"):
        raise ValueError(f"unknown reader mode {reader_mode!r}")
    head_params = [*dual.proj.parameters(), dual.log_scale]
    groups: list[dict] = [{"params": head_params, "lr": head_lr, "name": "head"}]
    if reader_mode == "freeze":
        set_reader_trainable(dual, False)
    elif reader_mode == "warmup":
        set_reader_trainable(dual, False)
        groups.append({
            "params": list(dual.encoder.parameters()),
            "lr": 0.0,
            "name": "reader",
        })
    else:
        set_reader_trainable(dual, True)
        groups.append({
            "params": list(dual.encoder.parameters()),
            "lr": reader_lr,
            "name": "reader",
        })
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.1)


def activate_reader(
    dual: AzuriteDual,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    freeze_steps: int,
    reader_mode: str,
    reader_lr: float,
) -> bool:
    """Unfreeze the reader once when a warmup run reaches its boundary."""
    if reader_mode != "warmup" or step < freeze_steps:
        return False
    changed = set_reader_trainable(dual, True)
    for group in optimizer.param_groups:
        if group.get("name") == "reader":
            group["lr"] = reader_lr
    return changed > 0


def _pad_rows(rows: Sequence[Sequence[int]], *, width: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    w = width or max((len(r) for r in rows), default=1)
    ids = torch.full((len(rows), w), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(rows), w), dtype=torch.bool)
    for i, row in enumerate(rows):
        take = list(row)[:w]
        if take:
            ids[i, : len(take)] = torch.tensor(take, dtype=torch.long)
            mask[i, : len(take)] = True
    return ids, mask


def train_batch(
    dual: AzuriteDual,
    tokenizer,
    spec: TaskSpec,
    bank: CandidateBank,
    examples: Sequence[Example],
    *,
    device: str | torch.device = "cpu",
    max_seq: int = 4096,
    rng: random.Random,
    candidate_dropout: float = 0.25,
    hard_negatives: int = 8,
    max_candidates: int = 64,
    paraphrase_probability: float = 0.25,
) -> tuple[torch.Tensor, dict]:
    """One forward/backward-compatible loss with dropout, paraphrases, and hard negatives."""
    context_batch, pack_stats = pack_dual(
        tokenizer, spec, bank, list(examples), max_seq=max_seq, field_aware=True
    )
    labels = paraphrase_labels(
        bank.labels, rng, probability=paraphrase_probability
    )
    cand_rows = [tokenizer.encode(label)[:256] for label in labels]
    cand_ids, cand_mask = _pad_rows(cand_rows)
    valid = torch.zeros((len(examples), len(labels)), dtype=torch.bool)
    kept: list[int] = []
    for row, example in enumerate(examples):
        indices = select_candidate_indices(
            example.context,
            labels,
            example.gold,
            rng,
            dropout=candidate_dropout,
            hard_negatives=hard_negatives,
            max_candidates=max_candidates,
        )
        if example.gold not in indices:
            raise AssertionError("candidate dropout removed the gold candidate")
        valid[row, indices] = True
        kept.append(len(indices))
    gold = torch.tensor([example.gold for example in examples], dtype=torch.long)
    context_ids = context_batch.context_ids.to(device)
    context_mask = context_batch.context_mask.to(device)
    cand_ids = cand_ids.to(device)
    cand_mask = cand_mask.to(device)
    valid = valid.to(device)
    gold = gold.to(device)
    logits = dual(context_ids, context_mask, cand_ids, cand_mask, valid)
    loss = F.cross_entropy(logits, gold)
    stats = {
        "loss": float(loss.detach()),
        "batch": len(examples),
        "candidates": len(labels),
        "kept_mean": sum(kept) / len(kept),
        "tokens": int(context_ids.numel() + cand_ids.numel()),
        "packed": int(pack_stats["packed"]),
        "unpackable": int(pack_stats["unpackable"]),
    }
    return loss, stats


@torch.no_grad()
def evaluate_examples(
    dual: AzuriteDual,
    tokenizer,
    spec: TaskSpec,
    bank: CandidateBank,
    examples: Sequence[Example],
    *,
    device: str | torch.device = "cpu",
    max_seq: int = 4096,
) -> list[dict]:
    """Score validation rows with all candidates and return per-row records."""
    context_batch, _ = pack_dual(
        tokenizer, spec, bank, list(examples), max_seq=max_seq, field_aware=True
    )
    cand_rows = [tokenizer.encode(label)[:256] for label in bank.labels]
    cand_ids, cand_mask = _pad_rows(cand_rows)
    valid = torch.ones((len(examples), len(bank.labels)), dtype=torch.bool)
    logits = dual(
        context_batch.context_ids.to(device),
        context_batch.context_mask.to(device),
        cand_ids.to(device),
        cand_mask.to(device),
        valid.to(device),
    )
    probs = F.softmax(logits.float(), dim=-1).cpu()
    bm25 = BM25([{"id": i, "text": label} for i, label in enumerate(bank.labels)])
    rows: list[dict] = []
    for i, example in enumerate(examples):
        choice = int(torch.argmax(logits[i]).item())
        bm25_choice = int(bm25.rank(example.context)[0])
        rows.append({
            "id": f"{spec.name}:{example.index}",
            "task": spec.name,
            "family": FAMILY_BY_TASK.get(spec.name, spec.name),
            "gold": int(example.gold),
            "choice": choice,
            "bm25_choice": bm25_choice,
            "correct": bool(choice == example.gold),
            "gold_label": bank.labels[example.gold],
            "choice_label": bank.labels[choice],
            "scores": [float(x) for x in probs[i].tolist()],
        })
    return rows


def validation_gate(
    records: Sequence[dict],
    *,
    evidence_precision: float | None = None,
    min_family_n: int = 1,
) -> dict:
    """Decide whether a decision checkpoint is acceptable.

    Every observed family must strictly beat both the majority-class control and
    the BM25-over-labels control. Evidence precision is required and must be at
    least 0.80; ``None`` means "not measured", which fails closed.
    """
    failures: list[str] = []
    by_family: dict[str, list[dict]] = {}
    for record in records:
        by_family.setdefault(str(record["family"]), []).append(record)
    families: dict[str, dict] = {}
    for family, rows in sorted(by_family.items()):
        n = len(rows)
        correct = sum(1 for row in rows if int(row["choice"]) == int(row["gold"]))
        bm25_correct = sum(
            1 for row in rows if int(row["bm25_choice"]) == int(row["gold"])
        )
        majority_class, majority_count = Counter(
            int(row["gold"]) for row in rows
        ).most_common(1)[0]
        accuracy = correct / n if n else 0.0
        majority_accuracy = majority_count / n if n else 0.0
        bm25_accuracy = bm25_correct / n if n else 0.0
        families[family] = {
            "n": n,
            "correct": correct,
            "accuracy": accuracy,
            "majority_class": majority_class,
            "majority_accuracy": majority_accuracy,
            "bm25_correct": bm25_correct,
            "bm25_accuracy": bm25_accuracy,
        }
        if n < min_family_n:
            failures.append(f"family {family}: n={n} below {min_family_n}")
        if accuracy <= majority_accuracy:
            failures.append(
                f"family {family}: {accuracy:.4f} <= majority {majority_accuracy:.4f}"
            )
        if accuracy <= bm25_accuracy:
            failures.append(
                f"family {family}: {accuracy:.4f} <= BM25 {bm25_accuracy:.4f}"
            )
    if not records:
        failures.append("no validation records")
    if evidence_precision is None:
        failures.append("evidence precision not measured")
    elif evidence_precision < 0.80:
        failures.append(f"evidence precision {evidence_precision:.4f} < 0.80")
    n = len(records)
    correct = sum(1 for row in records if int(row["choice"]) == int(row["gold"]))
    return {
        "passed": not failures,
        "failures": failures,
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "evidence_precision": evidence_precision,
        "families": families,
    }


def save_head_npz(path: str | Path, dual: AzuriteDual) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        proj=dual.proj.weight.detach().cpu().numpy(),
        log_scale=np.asarray([float(dual.log_scale.detach().cpu())]),
    )
    return out


def save_training_state(
    path: str | Path,
    dual: AzuriteDual,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    tokens_seen: int,
    args: dict,
    rng: random.Random,
) -> Path:
    """Atomically save weights, optimizer, progress, and RNG state."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    payload = {
        "schema": 1,
        "model": dual.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "args": args,
        "rng": rng.getstate(),
    }
    torch.save(payload, tmp)
    os.replace(tmp, out)
    return out


def load_training_state(
    path: str | Path,
    dual: AzuriteDual,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: str | torch.device = "cpu",
) -> tuple[dict, random.Random]:
    payload = torch.load(path, map_location=device, weights_only=False)
    dual.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    rng = random.Random()
    rng.setstate(payload["rng"])
    return payload, rng


def write_records(path: str | Path, records: Iterable[dict]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    return out
