#!/usr/bin/env python3
"""Decode vs score: one encoder, two ways to answer a typed decision.

The comparison this study needs is not "Azurite vs a big vendor" — it cannot run
a vendor's model — it is the *mechanism* question underneath the product claim:

    given the same bytes, the same encoder and the same training budget, is it
    faster and more accurate to (a) score every candidate in parallel, or
    (b) generate the label and check it against the candidate set?

Arms, matched on everything except the mechanism:

    score    AzuriteDecision (K<=16) or AzuriteDual (K>16). Every candidate is
             scored in one pass; the label is never emitted, so an out-of-bank
             answer is impossible by construction.
    decode   the same encoder, fine-tuned as a conditional byte model: given
             [CLS] instruction [SEP] context [SEP], fill L masked byte slots
             (iterative mask-fill, the generative mode a bidirectional byte
             encoder admits) and match the emitted string to the candidate bank.
             Out-of-bank output is possible and is counted as invalid.

Why mask-fill and not causal autoregression: a causal LM is a different set of
weights, so a causal decode arm would confound "generation" with "a differently
trained model". Mask-fill keeps the encoder, the data and the init identical and
changes only the interface and the loss — which is the comparison worth making.
The cost of that choice is stated in the report: mask-fill needs L sequential
passes, so its latency is a *lower bound* on what an autoregressive decoder would
pay at the same label length.

Both arms see the same rows in the same order for the same number of steps at the
same batch size, LR and seed. Reported per arm: test accuracy, invalid-output
rate, batch-1 latency p50/p95 over the same test rows, parameter count, and the
bytes the checkpoint occupies on disk.

Usage:
    python scripts/decode_vs_score.py --task ag_news --seeds 0 1 2 --threads 6
    python scripts/decode_vs_score.py --task banking77 --seeds 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import rung  # noqa: E402
from azurite.decisions import (  # noqa: E402
    CROSS_ENCODER_MAX_K,
    TASKS,
    candidate_bank,
    load_examples,
    pack_cross,
    pack_dual,
    stratified_examples,
)
from azurite.model import Azurite, AzuriteDecision, AzuriteDual  # noqa: E402
from azurite.tokenizers import CLS_ID, MASK_ID, PAD_ID, SEP_ID, load as load_tokenizer  # noqa: E402

RESULTS = PROJECT / "results" / "decode_vs_score"

sys.path.insert(0, str(PROJECT / "scripts"))
from transfer_eval import batch_schedule, candidate_tensor, dual_part, slice_batch  # noqa: E402


def init_encoder(cfg, ckpt: str | None, seed: int) -> Azurite:
    """Identical initialisation for both arms: same seed, same load."""
    torch.manual_seed(seed)
    enc = Azurite(cfg)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        report = enc.load_state_dict(state["model"], strict=True)
        if report.missing_keys or report.unexpected_keys:
            raise SystemExit(f"checkpoint mismatch: {report}")
    return enc


# ---------------------------------------------------------------------------
# decode arm: data, training, inference
# ---------------------------------------------------------------------------


def decode_rows(tok, spec, bank, examples, max_seq: int) -> tuple[list, dict]:
    """Prefix + label for teacher-forced training; also the inference prefix."""
    instr = tok.encode(spec.instruction)
    budget = max(8, max_seq - len(instr) - 8 - max(len(c) for c in bank.token_ids))
    rows, skipped = [], 0
    for ex in examples:
        ctx = tok.encode(ex.context)[:budget]
        prefix = [CLS_ID, *instr, SEP_ID, *ctx, SEP_ID]
        label = bank.token_ids[ex.gold]
        if len(prefix) + len(label) + 1 > max_seq:
            skipped += 1
            continue
        rows.append((prefix, label, ex.gold))
    return rows, {"packed": len(rows), "unpackable": skipped, "path": "decode",
                  "k": len(bank), "max_seq": max_seq,
                  "mean_seq_len": round(sum(len(p) + len(l) + 1 for p, l, _ in rows) / max(1, len(rows)), 1)}


def pad_decode_batch(rows, label_len: int, generator=None, mask_prob: float = 0.6
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt a random subset of the label slots; score only those.

    The first version masked every label slot on every step, which made the
    training distribution and the inference loop disjoint: at inference the model
    fills one slot at a time and must condition on the slots already committed,
    and it had never seen a partially revealed label. The measured symptom was
    blunt — 0% accuracy with a 100% invalid-output rate — because all-masked
    inputs carry no intra-label information, so each byte was predicted
    marginally and the emitted string was never a candidate.

    Now a quarter of examples are fully masked (so the first inference round is
    in-distribution) and the rest reveal a random subset.
    """
    width = max(len(p) + label_len + 1 for p, _, _ in rows)
    ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, (prefix, label, _) in enumerate(rows):
        seq = prefix + list(label) + [SEP_ID]
        if generator is None:
            chosen = set(range(len(label)))
        else:
            if torch.rand((), generator=generator).item() < 0.25:
                chosen = set(range(len(label)))
            else:
                chosen = {j for j in range(len(label))
                          if torch.rand((), generator=generator).item() < mask_prob}
                if not chosen:
                    chosen = {int(torch.randint(len(label), (1,), generator=generator))}
        for j in chosen:
            seq[len(prefix) + j] = MASK_ID
            tgt[i, len(prefix) + j] = label[j]
        ids[i, : len(seq)] = torch.tensor(seq)
        mask[i, : len(seq)] = True
    return ids, mask, tgt


def train_decode(model, rows, *, steps, batch, lr, seed, label_len, device, guard=None):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    model.train()
    gen = torch.Generator().manual_seed(seed * 7919 + 13)
    for idx in batch_schedule(len(rows), batch, steps, seed):
        sub = [rows[i] for i in idx.tolist()]
        ids, mask, tgt = pad_decode_batch(sub, label_len, generator=gen)
        opt.zero_grad(set_to_none=True)
        hidden = model(ids.to(device), mask.to(device))
        logits = model.mlm(hidden)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.to(device).reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise SystemExit("non-finite decode loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if guard is not None:
            guard.check()
    return model


@torch.no_grad()
def decode_one(model, prefix: list[int], label_len: int, device, steps: int | None = None) -> list[int]:
    """Iterative mask-fill: commit the most confident slot each round."""
    ids = prefix + [MASK_ID] * label_len + [SEP_ID]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    filled = [None] * label_len
    rounds = label_len if steps is None else min(steps, label_len)
    slots_per_round = max(1, math.ceil(label_len / rounds))
    for _ in range(rounds):
        logits = model.mlm(model(x))[0, len(prefix): len(prefix) + label_len]
        probs = F.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(-1)
        conf = conf.clone()
        for j, v in enumerate(filled):
            if v is not None:
                conf[j] = -1.0
        order = torch.argsort(conf, descending=True)
        committed = 0
        for j in order.tolist():
            if filled[j] is not None:
                continue
            filled[j] = int(pred[j])
            x[0, len(prefix) + j] = int(pred[j])
            committed += 1
            if committed >= slots_per_round:
                break
    return filled


def normalise(s: str) -> str:
    return " ".join(s.strip().lower().split())


@torch.no_grad()
def evaluate_decode(model, rows, bank, tok, device, *, label_len, steps, latency_n) -> dict:
    model.eval()
    names = [normalise(l) for l in bank.labels]
    correct = invalid = 0
    lat = []
    preds = []
    for prefix, label, gold in rows:
        t0 = time.perf_counter()
        out = decode_one(model, prefix, label_len, device, steps=steps)
        lat.append(time.perf_counter() - t0)
        text = normalise(tok.decode([b for b in out if b >= 4]))
        if text in names:
            pred = names.index(text)
        else:
            pred = -1
            invalid += 1
        preds.append(pred)
        correct += int(pred == gold)
    lat.sort()
    return {
        "n": len(rows),
        "accuracy": round(correct / max(1, len(rows)), 4),
        "invalid_rate": round(invalid / max(1, len(rows)), 4),
        "latency_p50_ms": round(1000 * lat[len(lat) // 2], 2),
        "latency_p95_ms": round(1000 * lat[int(0.95 * (len(lat) - 1))], 2),
        "latency_n": len(lat),
    }


# ---------------------------------------------------------------------------
# score arm
# ---------------------------------------------------------------------------


def train_score(model, batch, *, steps, batch_size, lr, seed, device, path, cand_args=None, guard=None):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    n = batch.gold.shape[0]
    model.train()
    for idx in batch_schedule(n, batch_size, steps, seed):
        opt.zero_grad(set_to_none=True)
        if path == "cross":
            part = slice_batch(batch, idx).to(device)
            logits = model(part)
            loss = AzuriteDecision.loss(logits, part.gold)
        else:
            part = dual_part(batch, idx, device)
            logits = model(part.context_ids, part.context_mask, *cand_args)
            loss = F.cross_entropy(logits, part.gold)
        if not torch.isfinite(loss):
            raise SystemExit("non-finite score loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if guard is not None:
            guard.check()
    return model


@torch.no_grad()
def evaluate_score(model, batch, device, path, cand_args, latency_n: int) -> dict:
    model.eval()
    n = batch.gold.shape[0]
    if path == "cross":
        logits = model(slice_batch(batch, torch.arange(n)).to(device))
    else:
        logits = model(batch.context_ids.to(device), batch.context_mask.to(device), *cand_args)
    pred = logits.argmax(-1).cpu()
    gold = batch.gold.cpu()
    acc = (pred == gold).float().mean().item()
    # risk-coverage: the scorer's own max-softmax is the abstention signal, so
    # the curve below is free evidence about how accuracy behaves when the
    # model is allowed to say "not sure" instead of guessing.
    probs = torch.softmax(logits.float(), -1).cpu()
    conf = probs.max(-1).values
    hit = (pred == gold).float()
    order = torch.argsort(conf, descending=True)
    coverage = {}
    for c in (0.5, 0.7, 0.9, 1.0):
        k = max(1, int(round(c * n)))
        coverage[str(c)] = round(float(hit[order[:k]].mean()), 4)
    lat = []
    single = min(latency_n, n)
    for i in range(single):
        idx = torch.tensor([i])
        t0 = time.perf_counter()
        if path == "cross":
            model(slice_batch(batch, idx).to(device))
        else:
            part = dual_part(batch, idx, device)
            model(part.context_ids, part.context_mask, *cand_args)
        lat.append(time.perf_counter() - t0)
    lat.sort()
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "invalid_rate": 0.0,
        "latency_p50_ms": round(1000 * lat[len(lat) // 2], 2),
        "latency_p95_ms": round(1000 * lat[int(0.95 * (len(lat) - 1))], 2),
        "latency_n": len(lat),
        "coverage": coverage,
        "correct": [int(x) for x in hit.tolist()],
        "confidence": [round(float(x), 4) for x in conf.tolist()],
    }


def checkpoint_bytes(model) -> int:
    with tempfile.NamedTemporaryFile(suffix=".pt") as fh:
        torch.save({"model": model.state_dict()}, fh.name)
        return Path(fh.name).stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--rung", default="tiny")
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--ckpt", default="", help="pretrained encoder to initialise both arms")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--examples", type=int, default=3000)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--test-examples", type=int, default=300)
    ap.add_argument("--latency-n", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--arms", nargs="+", default=["score", "decode"],
                    choices=["score", "decode"],
                    help="which interfaces to train; score-only halves a sweep's CPU cost")
    ap.add_argument("--tag", default="", help="suffix on the result filename, e.g. rows8k")
    ap.add_argument("--stratify", action="store_true",
                    help=("class-balanced draw of the train/test subsets; required "
                          "for a real 77-way number, because the splits are "
                          "label-sorted"))
    ap.add_argument("--split-seed", type=int, default=0,
                    help="fixed across training seeds so every seed sees the same rows")
    ap.add_argument("--permute-train-labels", action="store_true",
                    help="shuffle training labels only; test labels stay true, so a score "
                         "above chance means the encoder is using a leakage cue rather than "
                         "learning the train-label relation")
    args = ap.parse_args()
    if "decode" in args.arms and "score" not in args.arms:
        raise SystemExit("--arms decode requires score: the comparison is decode-vs-score")

    torch.set_num_threads(args.threads)
    device = args.device
    tok = load_tokenizer(args.arm)
    cfg = rung(args.rung, args.arm)
    spec = TASKS[args.task]
    bank = candidate_bank(spec)
    if not bank.token_ids:
        bank.tokenize(tok)
    label_len = max(len(c) for c in bank.token_ids)
    path = "cross" if len(bank) <= CROSS_ENCODER_MAX_K else "dual"

    if args.stratify:
        train_rows = stratified_examples(spec, bank, "train", args.examples, args.split_seed)
        test_rows = stratified_examples(spec, bank, "test", args.test_examples, args.split_seed + 1)
    else:
        train_rows = load_examples(spec, bank, split="train", limit=args.examples)
        test_rows = load_examples(spec, bank, split="test", limit=args.test_examples)
    if path == "cross":
        train_batch, pack_stats = pack_cross(tok, spec, bank, train_rows, max_seq=cfg.max_seq, seed=0)
        test_batch, test_stats = pack_cross(tok, spec, bank, test_rows, max_seq=cfg.max_seq, seed=0)
        cand_args = None
    else:
        train_batch, pack_stats = pack_dual(tok, spec, bank, train_rows, max_seq=cfg.max_seq)
        test_batch, test_stats = pack_dual(tok, spec, bank, test_rows, max_seq=cfg.max_seq)
        ci, cm = candidate_tensor(bank, tok)
        cand_args = (ci.to(device), cm.to(device), torch.ones(len(bank), dtype=torch.bool, device=device))

    # The permutation must run after either packing path. It previously sat
    # inside the cross-encoder branch, so `--permute-train-labels` was silently
    # a no-op on banking77's 77-way dual encoder: the control scored 0.848
    # against a 0.861 baseline and could not have detected leakage.
    if args.permute_train_labels:
        from dataclasses import replace
        gen = torch.Generator().manual_seed(args.split_seed + 20261028)
        perm = torch.randperm(train_batch.gold.numel(), generator=gen)
        train_batch = replace(train_batch, gold=train_batch.gold[perm])

    dec_train = dec_test = dec_stats = None
    if "decode" in args.arms:
        dec_train, dec_stats = decode_rows(tok, spec, bank, train_rows, cfg.max_seq)
        dec_test, _ = decode_rows(tok, spec, bank, test_rows, cfg.max_seq)

    steps = max(1, math.ceil(len(train_batch.gold) / args.batch) * int(args.epochs))
    print(f"task={args.task} K={len(bank)} path={path} device={device} "
          f"train={len(train_batch.gold)} test={len(test_batch.gold)} steps={steps} "
          f"label_len={label_len}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        rec: dict = {"task": args.task, "rung": args.rung, "arm": args.arm, "seed": seed,
                     "k": len(bank), "path": path, "device": device,
                     "train_examples": int(len(train_batch.gold)), "steps": steps,
                     "batch": args.batch, "lr": args.lr, "epochs": args.epochs,
                     "test_examples": int(len(test_batch.gold)), "label_len": label_len,
                     "torch_threads": args.threads, "ckpt": args.ckpt or None,
                     "stratified": bool(args.stratify), "split_seed": args.split_seed,
                     "permute_train_labels": bool(args.permute_train_labels),
                     "pack_score": pack_stats, "pack_decode": dec_stats}

        enc = init_encoder(cfg, args.ckpt or None, seed)
        score_model = (AzuriteDecision(enc) if path == "cross" else AzuriteDual(enc)).to(device)
        t0 = time.perf_counter()
        train_score(score_model, train_batch, steps=steps, batch_size=args.batch, lr=args.lr,
                    seed=seed, device=device, path=path, cand_args=cand_args)
        rec["score_train_s"] = round(time.perf_counter() - t0, 1)
        rec["score"] = evaluate_score(score_model, test_batch, device, path, cand_args, args.latency_n)
        rec["score"]["params"] = sum(p.numel() for p in score_model.parameters())
        rec["score"]["checkpoint_bytes"] = checkpoint_bytes(score_model)

        if "decode" in args.arms:
            enc2 = init_encoder(cfg, args.ckpt or None, seed)
            t0 = time.perf_counter()
            train_decode(enc2.to(device), dec_train, steps=steps, batch=args.batch, lr=args.lr,
                         seed=seed, label_len=label_len, device=device)
            rec["decode_train_s"] = round(time.perf_counter() - t0, 1)
            rec["decode_1"] = evaluate_decode(enc2, dec_test, bank, tok, device,
                                              label_len=label_len, steps=1, latency_n=args.latency_n)
            rec["decode_full"] = evaluate_decode(enc2, dec_test, bank, tok, device,
                                                 label_len=label_len, steps=None, latency_n=args.latency_n)
            rec["decode_full"]["params"] = sum(p.numel() for p in enc2.parameters())
            rec["decode_full"]["checkpoint_bytes"] = checkpoint_bytes(enc2)
        else:
            rec["decode_1"] = None
            rec["decode_full"] = None
        rec["arms"] = list(args.arms)

        suffix = f"_{args.tag}" if args.tag else ""
        out = out_dir / f"{args.task}_{args.rung}_{args.arm}{suffix}_s{seed}.json"
        out.write_text(json.dumps(rec, indent=1) + "\n")
        summary = {
            "seed": seed, "arms": rec["arms"],
            "score_acc": rec["score"]["accuracy"], "score_p50": rec["score"]["latency_p50_ms"],
            "coverage": rec["score"].get("coverage"),
        }
        if rec["decode_full"] is not None:
            summary.update({
                "decode1_acc": rec["decode_1"]["accuracy"],
                "decode1_p50": rec["decode_1"]["latency_p50_ms"],
                "decode_acc": rec["decode_full"]["accuracy"],
                "decode_p50": rec["decode_full"]["latency_p50_ms"],
                "decode_invalid": rec["decode_full"]["invalid_rate"],
            })
        print(json.dumps(summary), flush=True)
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
