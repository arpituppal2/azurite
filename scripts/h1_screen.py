#!/usr/bin/env python3
"""H1 development screen: candidate-conditioned decisions on the dev suite.

This is the screen pre-registered in `research/PIVOT_REVIEW.md`, not a
confirmatory run. Every arm sees the same rows, the same K candidates, the same
candidate order, and the same byte cap. The v1 dev test has already been
inspected; a result here is development evidence, not a capability claim.

Arms:
  prior       choose the candidate with the highest train label frequency
  tfidf       TF-IDF + logistic regression, scored over the candidate set
  byte        the 14.4M Azurite encoder with a mean-pooled pair head
  minilm      sentence-transformers/all-MiniLM-L6-v2, sequence classifier
  modernbert  answerdotai/ModernBERT-base, sequence classifier

The script writes per-example records and a summary. It does not touch the
locked suite. A failed arm is recorded as failed, not dropped silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"
SUITES = ROOT / "research" / "suites"
DEFAULT_OUT = ROOT / "results" / "h1_screen"
CHECKPOINT = ROOT / "results" / "pretrain" / "l1_small_byte_ctx256" / "ckpt.pt"

TASKS = [
    "banking77_ctx",
    "clinc_oos_ctx",
    "boolq_ctx",
    "squad2_answerability_ctx",
    "contract_nli_ctx",
]
INSTRUCTION = {
    "banking77_ctx": "Choose the banking intent that best matches the question.",
    "clinc_oos_ctx": "Choose the assistant intent that best matches the request.",
    "boolq_ctx": "Decide whether the passage answers the question with yes or no.",
    "squad2_answerability_ctx": (
        "Decide whether the question can be answered from the passage or must be "
        "treated as unanswerable."
    ),
    "contract_nli_ctx": (
        "Decide whether the contract entails, contradicts, or does not mention the "
        "stated policy."
    ),
}
MODEL_IDS = {
    "modernbert": ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8"),
}
MINILM_ID = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
)


@dataclass
class Example:
    task: str
    id: str
    prefix: str
    candidates: list[str]
    gold: int
    label: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_for(*parts: object) -> int:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def truncate_bytes(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def make_prefix(task: str, row: dict, budget: int) -> str:
    # Question first: at a 256-byte cap, ContractNLI's 5-9 KB contract would
    # otherwise consume the whole prefix and truncate away the policy being
    # decided. The order is identical for every arm.
    text = "\n".join(
        part for part in [
            INSTRUCTION[task],
            str(row.get("question", "")),
            str(row.get("context", "")),
        ] if part
    )
    return truncate_bytes(text, budget)


def candidates_for(
    task: str, row: dict, labels: list[str], k: int, seed: int
) -> tuple[list[str], int]:
    gold = str(row["label_text"])
    if gold not in labels:
        raise ValueError(f"{task}/{row['id']}: gold {gold!r} not in train labels")
    rng = random.Random(seed_for(task, row["id"], seed))
    if k >= len(labels):
        candidates = list(labels)
    else:
        others = [label for label in labels if label != gold]
        rng.shuffle(others)
        candidates = [gold, *others[: k - 1]]
    rng.shuffle(candidates)
    return candidates, candidates.index(gold)


def load_examples(task: str, split: str, k: int, seed: int, max_bytes: int) -> list[Example]:
    path = RAW / task / split
    ds = load_from_disk(str(path))
    rows = [dict(row) for row in ds]
    if not rows:
        return []
    if "label_text" not in rows[0]:
        raise ValueError(f"{task}: label_text is required for candidate construction")
    train = load_from_disk(str(RAW / task / "train"))
    labels = sorted({str(row["label_text"]) for row in train})
    examples = []
    for row in rows:
        gold = str(row["label_text"])
        # Reserve at most 64 bytes for the candidate span; the rest is context.
        prefix_budget = max(32, max_bytes - 80)
        prefix = make_prefix(task, row, prefix_budget)
        candidates, gold_idx = candidates_for(task, row, labels, k, seed)
        examples.append(Example(
            task=task,
            id=str(row["id"]),
            prefix=prefix,
            candidates=candidates,
            gold=gold_idx,
            label=gold,
        ))
    return examples


def format_candidate(candidate: str, max_bytes: int = 64) -> str:
    return truncate_bytes(f"Candidate: {candidate}", max_bytes)


def byte_ids(prefix: str, candidate: str, max_seq: int) -> list[int]:
    # Special ids follow azurite/tokenizers.py: pad=0, mask=1, cls=2, sep=3.
    prefix_ids = [2, *[b + 4 for b in prefix.encode("utf-8")], 3]
    cand_ids = [*[b + 4 for b in candidate.encode("utf-8")], 3]
    ids = prefix_ids + cand_ids
    if len(ids) > max_seq:
        # Keep the candidate and the start of the context; truncate the middle.
        keep = max_seq - len(cand_ids)
        ids = prefix_ids[:keep] + cand_ids
    return ids


def pad_batch(sequences: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(seq) for seq in sequences)
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), length), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
    return ids, mask


class ByteScorer(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(encoder.cfg.d_model, 1)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(ids, mask)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def load_byte_model(device: torch.device) -> ByteScorer:
    from azurite.config import EncoderConfig
    from azurite.model import Azurite

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg_dict = payload["config"]
    cfg = EncoderConfig(**cfg_dict)
    encoder = Azurite(cfg)
    missing, unexpected = encoder.load_state_dict(payload["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:5]}")
    if missing:
        print(f"warning: missing encoder keys: {missing[:5]}", file=sys.stderr)
    return ByteScorer(encoder).to(device)


def hf_batch(
    tokenizer,
    batch: list[Example],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefixes = []
    candidates = []
    gold = []
    for ex in batch:
        for cand in ex.candidates:
            prefixes.append(ex.prefix)
            candidates.append(format_candidate(cand))
        gold.append(ex.gold)
    enc = tokenizer(
        prefixes,
        candidates,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device)
    return input_ids, attention, torch.tensor(gold, device=device)


def byte_batch(
    batch: list[Example],
    max_seq: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs = []
    gold = []
    for ex in batch:
        for cand in ex.candidates:
            seqs.append(byte_ids(ex.prefix, format_candidate(cand), max_seq))
        gold.append(ex.gold)
    ids, mask = pad_batch(seqs)
    return ids.to(device), mask.to(device), torch.tensor(gold, device=device)


def batches(examples: list[Example], size: int) -> Iterable[list[Example]]:
    for i in range(0, len(examples), size):
        yield examples[i : i + size]


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=-1, keepdims=True)


@torch.no_grad()
def evaluate_hf(model, tokenizer, examples, batch_size, max_length, device):
    model.eval()
    predictions = []
    probabilities = []
    for batch in batches(examples, batch_size):
        input_ids, attention, gold = hf_batch(tokenizer, batch, max_length, device)
        k = len(batch[0].candidates)
        logits = model(input_ids=input_ids, attention_mask=attention).logits.view(-1, k)
        probabilities.extend(F.softmax(logits, -1).cpu().tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
    return predictions, probabilities


@torch.no_grad()
def evaluate_byte(model, examples, batch_size, max_seq, device):
    model.eval()
    predictions = []
    probabilities = []
    for batch in batches(examples, batch_size):
        ids, mask, gold = byte_batch(batch, max_seq, device)
        k = len(batch[0].candidates)
        logits = model(ids, mask).view(-1, k)
        probabilities.extend(F.softmax(logits, -1).cpu().tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
    return predictions, probabilities


def prior_predictions(
    train: list[Example], test: list[Example]
) -> tuple[list[int], list[list[float]]]:
    counts = Counter(ex.label for ex in train)
    total = sum(counts.values()) or 1
    predictions = []
    probabilities = []
    for ex in test:
        scores = np.array([counts.get(c, 0) / total for c in ex.candidates])
        if scores.sum() <= 0:
            scores = np.ones(len(ex.candidates))
        probs = scores / scores.sum()
        probabilities.append(probs.tolist())
        predictions.append(int(np.argmax(probs)))
    return predictions, probabilities


def minilm_predictions(
    train: list[Example], test: list[Example], device: torch.device
) -> tuple[list[int], list[list[float]]]:
    """Frozen MiniLM pair embeddings + logistic regression.

    The encoder is not fine-tuned; this is the cheap pretrained alternative in
    the pre-registration, not a second cross-encoder.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MINILM_ID[0], revision=MINILM_ID[1], device=str(device))
    model.eval()

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    train_prefix = embed([ex.prefix for ex in train])
    test_prefix = embed([ex.prefix for ex in test])
    candidate_text = {
        format_candidate(c) for ex in [*train, *test] for c in ex.candidates
    }
    unique = sorted(candidate_text)
    embedded = embed(unique)
    candidate_emb = {text: embedded[i] for i, text in enumerate(unique)}

    def pair_features(prefix: np.ndarray, candidate: str) -> np.ndarray:
        v = candidate_emb[format_candidate(candidate)]
        return np.concatenate([prefix, v, np.abs(prefix - v), prefix * v])

    rows = []
    labels = []
    for i, ex in enumerate(train):
        for candidate in ex.candidates:
            rows.append(pair_features(train_prefix[i], candidate))
            labels.append(1 if candidate == ex.label else 0)
    classifier = LogisticRegression(
        max_iter=1_000, C=1.0, class_weight="balanced", random_state=0
    )
    classifier.fit(np.stack(rows), labels)

    predictions = []
    probabilities = []
    for i, ex in enumerate(test):
        features = np.stack([pair_features(test_prefix[i], c) for c in ex.candidates])
        scores = classifier.decision_function(features)
        probs = softmax_rows(scores)
        probabilities.append(probs.tolist())
        predictions.append(int(np.argmax(probs)))
    return predictions, probabilities


def tfidf_predictions(
    train: list[Example], test: list[Example]
) -> tuple[list[int], list[list[float]]]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, min_df=2, sublinear_tf=True)
    x_train = vectorizer.fit_transform([ex.prefix for ex in train])
    classifier = LogisticRegression(max_iter=2_000, C=4.0, class_weight="balanced")
    classifier.fit(x_train, [ex.label for ex in train])
    classes = list(classifier.classes_)
    class_index = {label: i for i, label in enumerate(classes)}
    x_test = vectorizer.transform([ex.prefix for ex in test])
    proba = classifier.predict_proba(x_test)
    predictions = []
    probabilities = []
    for row, ex in enumerate(test):
        scores = np.array([
            proba[row, class_index[c]] if c in class_index else 0.0
            for c in ex.candidates
        ])
        probs = scores / scores.sum() if scores.sum() > 0 else np.ones(len(scores)) / len(scores)
        probabilities.append(probs.tolist())
        predictions.append(int(np.argmax(probs)))
    return predictions, probabilities


def score(test: list[Example], predictions: list[int]) -> dict:
    correct = sum(int(pred == ex.gold) for ex, pred in zip(test, predictions))
    n = len(test)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0}


def wilson(correct: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibration_metrics(probabilities: list[list[float]], correct: list[int]) -> dict:
    """Descriptive ECE and AURC; no threshold is fitted."""
    confidence = np.array([max(row) for row in probabilities], dtype=float)
    hit = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(hit[mask].mean() - confidence[mask].mean())
    order = np.argsort(-confidence, kind="stable")
    risk = np.cumsum(1.0 - hit[order]) / np.arange(1, len(order) + 1)
    aurc = float(risk.mean()) if len(risk) else 0.0
    return {"ece": round(float(ece), 6), "aurc": round(aurc, 6)}


def mcnemar(reference: list[int], challenger: list[int]) -> dict:
    """Exact two-sided McNemar on paired correctness."""
    b = sum(int(c and not r) for r, c in zip(reference, challenger))
    c = sum(int(r and not c) for r, c in zip(reference, challenger))
    if b + c == 0:
        return {"b": b, "c": c, "p": 1.0}
    from scipy.stats import binomtest

    p = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "p": round(float(p), 6)}


def paired_bootstrap_macro(
    correctness: dict[str, list[int]],
    challenger: str,
    reference: str,
    reps: int = 2_000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap of the macro-accuracy delta over task examples."""
    rng = np.random.default_rng(seed)
    tasks = sorted(correctness)
    arrays = {
        arm: {task: np.asarray(correctness[task][arm], dtype=float) for task in tasks}
        for arm in {challenger, reference}
    }
    deltas = np.empty(reps, dtype=float)
    for r in range(reps):
        task_deltas = []
        for task in tasks:
            n = len(arrays[reference][task])
            idx = rng.integers(0, n, n)
            task_deltas.append(
                arrays[challenger][task][idx].mean()
                - arrays[reference][task][idx].mean()
            )
        deltas[r] = float(np.mean(task_deltas))
    observed = float(
        np.mean([
            arrays[challenger][task].mean() - arrays[reference][task].mean()
            for task in tasks
        ])
    )
    return {
        "delta_pp": round(100 * observed, 3),
        "lo_pp": round(100 * float(np.quantile(deltas, 0.025)), 3),
        "hi_pp": round(100 * float(np.quantile(deltas, 0.975)), 3),
        "reps": reps,
    }


def save_records(
    path: Path,
    test: list[Example],
    predictions: list[int],
    probabilities: list[list[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex, pred, probs in zip(test, predictions, probabilities):
            fh.write(json.dumps({
                "task": ex.task,
                "id": ex.id,
                "label": ex.label,
                "gold_index": ex.gold,
                "pred_index": int(pred),
                "candidates": ex.candidates,
                "prob_gold": round(float(probs[ex.gold]), 6),
                "prob_pred": round(float(probs[pred]), 6),
                "correct": bool(pred == ex.gold),
                "prefix_sha256": sha256_text(ex.prefix),
            }, sort_keys=True) + "\n")


def train_hf(arm: str, task: str, train: list[Example], args, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_id, revision = MODEL_IDS[arm]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, num_labels=1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    started = time.time()
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(seed_for("order", arm, task, args.seed, epoch)).shuffle(order)
        for batch_indices in batches(order, args.batch_size):
            batch = [train[i] for i in batch_indices]
            input_ids, attention, gold = hf_batch(tokenizer, batch, args.max_tokens, device)
            k = len(batch[0].candidates)
            logits = model(input_ids=input_ids, attention_mask=attention).logits.view(-1, k)
            loss = F.cross_entropy(logits, gold)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print(f"  {arm}/{task} epoch {epoch+1} step {step} loss {loss.item():.4f}", flush=True)
    return model, tokenizer, {"steps": step, "seconds": round(time.time() - started, 1)}


def train_byte(task: str, train: list[Example], args, device):
    model = load_byte_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    started = time.time()
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(seed_for("order", "byte", task, args.seed, epoch)).shuffle(order)
        for batch_indices in batches(order, args.batch_size):
            batch = [train[i] for i in batch_indices]
            ids, mask, gold = byte_batch(batch, args.max_seq, device)
            k = len(batch[0].candidates)
            logits = model(ids, mask).view(-1, k)
            loss = F.cross_entropy(logits, gold)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print(f"  byte/{task} epoch {epoch+1} step {step} loss {loss.item():.4f}", flush=True)
    return model, {"steps": step, "seconds": round(time.time() - started, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="prior,tfidf,byte,minilm,modernbert")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-bytes", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(args.out) if args.out else DEFAULT_OUT / run_id
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"device {device}; out {out}; arms {arms}; tasks {tasks}", flush=True)

    summary = {
        "protocol": "h1-candidate-screen-v1",
        "date": run_id,
        "args": vars(args),
        "device": str(device),
        "note": "Development screen on an inspected dev test; not a capability claim.",
        "packing": "instruction, question, context; 256-byte cap for every arm",
        "tasks": {},
    }
    correctness_all: dict[str, dict[str, list[int]]] = {}
    for task in tasks:
        print(f"\n=== {task} ===", flush=True)
        train = load_examples(task, "train", args.k, args.seed, args.max_bytes)
        test = load_examples(task, "test", args.k, args.seed, args.max_bytes)
        if args.limit_train:
            train = train[: args.limit_train]
        if args.limit_test:
            test = test[: args.limit_test]
        print(f"rows train={len(train)} test={len(test)} k={args.k}", flush=True)
        task_dir = out / task
        task_dir.mkdir(parents=True, exist_ok=True)
        result = {"train_rows": len(train), "test_rows": len(test), "arms": {}}
        for arm in arms:
            print(f"-- {arm}", flush=True)
            try:
                if arm == "prior":
                    preds, probs = prior_predictions(train, test)
                    meta = {"seconds": 0.0}
                elif arm == "tfidf":
                    started = time.time()
                    preds, probs = tfidf_predictions(train, test)
                    meta = {"seconds": round(time.time() - started, 1)}
                elif arm == "minilm":
                    started = time.time()
                    preds, probs = minilm_predictions(train, test, device)
                    meta = {"seconds": round(time.time() - started, 1)}
                elif arm == "byte":
                    model, meta = train_byte(task, train, args, device)
                    preds, probs = evaluate_byte(
                        model, test, args.batch_size, args.max_seq, device
                    )
                    torch.save(model.state_dict(), task_dir / f"{arm}.pt")
                elif arm in MODEL_IDS:
                    model, tokenizer, meta = train_hf(arm, task, train, args, device)
                    preds, probs = evaluate_hf(
                        model, tokenizer, test, args.batch_size, args.max_tokens, device
                    )
                    model.save_pretrained(task_dir / f"{arm}_model")
                    tokenizer.save_pretrained(task_dir / f"{arm}_model")
                else:
                    raise ValueError(f"unknown arm {arm!r}")
                metrics = score(test, preds)
                correct = [int(pred == ex.gold) for ex, pred in zip(test, preds)]
                metrics["wilson95"] = [round(v, 4) for v in wilson(metrics["correct"], metrics["n"])]
                metrics.update(calibration_metrics(probs, correct))
                save_records(task_dir / f"{arm}.jsonl", test, preds, probs)
                result["arms"][arm] = {**metrics, **meta, "status": "ok"}
                correctness_all.setdefault(task, {})[arm] = correct
                print(
                    f"  {arm}: {metrics['correct']}/{metrics['n']} = "
                    f"{metrics['accuracy']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                result["arms"][arm] = {"status": "failed", "error": repr(exc)}
                print(f"  {arm} FAILED: {exc!r}", flush=True)
        summary["tasks"][task] = result
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Cross-task summaries, computed only over tasks where every needed arm ran.
    macro = {}
    for arm in arms:
        values = [
            result["arms"][arm]["accuracy"]
            for result in summary["tasks"].values()
            if result["arms"].get(arm, {}).get("status") == "ok"
        ]
        if values:
            macro[arm] = round(float(np.mean(values)), 6)
    summary["macro_accuracy"] = macro
    summary["bootstrap_vs_byte"] = {}
    for challenger in ("tfidf", "minilm", "modernbert", "prior"):
        if challenger == "byte" or challenger not in macro:
            continue
        filtered = {
            task: {"byte": correctness["byte"], challenger: correctness[challenger]}
            for task, correctness in correctness_all.items()
            if "byte" in correctness and challenger in correctness
        }
        if filtered and "byte" in macro:
            summary["bootstrap_vs_byte"][challenger] = paired_bootstrap_macro(
                filtered, challenger, "byte"
            )
    for task, result in summary["tasks"].items():
        reference = correctness_all.get(task, {}).get("byte")
        if not reference:
            continue
        for arm, metrics in result["arms"].items():
            challenger = correctness_all.get(task, {}).get(arm)
            if arm != "byte" and challenger:
                metrics["mcnemar_vs_byte"] = mcnemar(reference, challenger)

    pretrained = [arm for arm in ("minilm", "modernbert") if arm in macro]
    best = max(pretrained, key=lambda arm: macro[arm]) if pretrained else None
    gates = {}
    if best and "byte" in macro and "tfidf" in macro:
        contract = summary["tasks"].get("contract_nli_ctx", {}).get("arms", {})
        squad = summary["tasks"].get("squad2_answerability_ctx", {}).get("arms", {})
        contract_margin = (
            contract[best]["accuracy"] - contract["prior"]["accuracy"]
            if best in contract and contract.get("prior", {}).get("status") == "ok"
            else None
        )
        squad_margin = (
            squad[best]["accuracy"] - squad["prior"]["accuracy"]
            if best in squad and squad.get("prior", {}).get("status") == "ok"
            else None
        )
        per_task_delta = {
            task: round(100 * (result["arms"][best]["accuracy"] - result["arms"]["byte"]["accuracy"]), 3)
            for task, result in summary["tasks"].items()
            if best in result["arms"] and result["arms"].get("byte", {}).get("status") == "ok"
        }
        gates = {
            "best_pretrained": best,
            "beats_tfidf_by_3pp": round(100 * (macro[best] - macro["tfidf"]), 3),
            "beats_byte_by_3pp": round(100 * (macro[best] - macro["byte"]), 3),
            "bootstrap_lo_above_zero": summary["bootstrap_vs_byte"].get(best, {}).get("lo_pp"),
            "contract_nli_margin_pp": None if contract_margin is None else round(100 * contract_margin, 3),
            "squad_margin_pp": None if squad_margin is None else round(100 * squad_margin, 3),
            "per_task_delta_vs_byte_pp": per_task_delta,
            "worst_task_delta_pp": min(per_task_delta.values()) if per_task_delta else None,
        }
        gates["stage1_supported"] = bool(
            gates["beats_tfidf_by_3pp"] >= 3.0
            and gates["beats_byte_by_3pp"] >= 3.0
            and (gates["bootstrap_lo_above_zero"] or -1) > 0
            and (gates["contract_nli_margin_pp"] or -999) >= 10.0
            and (gates["squad_margin_pp"] or -999) >= 5.0
            and (gates["worst_task_delta_pp"] or -999) >= -2.0
        )
        gates["stage2_required"] = True
        gates["h1_supported"] = False  # Stage 2 has not run; never claim it here.
    summary["gates"] = gates
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nmacro", macro, flush=True)
    print("gates", json.dumps(gates, indent=2), flush=True)
    print("wrote", out / "summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
