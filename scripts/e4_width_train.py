#!/usr/bin/env python3
"""E4: fine-tune at one candidate width |C| and evaluate at that width.

The candidate-set sweep is only an experiment if the model is trained at the
width it is tested at.  With one fixed model and one fixed bank, restricting the
candidates to a subset that still contains the gold can only delete wrong
argmaxes: accuracy at |C| < K is a deterministic function of the full-K ranking
and is guaranteed to be at least the full-K accuracy.  That version of the sweep
would "hold" by construction and would test nothing.

This script therefore trains a fresh decision head from the same pretrained
checkpoint, on the same 4,000 stratified rows, with the same 3,000-step AdamW
recipe, but with every training row packed against a fresh sample of |C|-1
distractors plus its gold label.  Evaluation packs each test row against an
independently sampled |C|-way set.  Width is the only thing that changes.

Usage:
    .venv/bin/python scripts/e4_width_train.py --task banking77 --rung large \
        --width 32 --steps 3000 --out-dir results/e4_candidate_sweep/large/banking77
    .venv/bin/python scripts/e4_width_train.py ... --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import rung  # noqa: E402
from azurite.decisions import TASKS, candidate_bank, _load, stratified_examples  # noqa: E402
from azurite.model import Azurite, AzuriteDual  # noqa: E402
from azurite.tokenizers import PAD_ID, load as load_tokenizer  # noqa: E402

sys.path.insert(0, str(PROJECT / "scripts"))
from transfer_eval import batch_schedule  # noqa: E402

CLIP = 1.0
WEIGHT_DECAY = 0.01


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    raw = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def sample_subset(k: int, width: int, gold: int, rng: random.Random) -> list[int]:
    """Deterministically pack one row against `width` candidates, gold included.

    The gold is inserted at a random slot so a model cannot learn "gold is always
    last"; the remaining slots are uniform draws without replacement from the
    other k-1 classes.
    """
    if not 2 <= width <= k:
        raise ValueError(f"width {width} outside [2, {k}]")
    pool = [c for c in range(k) if c != gold]
    chosen = rng.sample(pool, width - 1)
    chosen.append(gold)
    rng.shuffle(chosen)
    return chosen


def gold_slot(subset: list[int], gold: int) -> int:
    return subset.index(gold)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


@torch.no_grad()
def embed_contexts(model: AzuriteDual, batch, device: str, chunk: int = 64) -> torch.Tensor:
    model.eval()
    out = []
    for start in range(0, batch.context_ids.shape[0], chunk):
        ids = batch.context_ids[start:start + chunk].to(device)
        mask = batch.context_mask[start:start + chunk].to(device)
        out.append(model.embed(ids, mask).cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_candidates(model: AzuriteDual, cand_ids: torch.Tensor, cand_mask: torch.Tensor,
                     device: str) -> torch.Tensor:
    return model.embed(cand_ids.to(device), cand_mask.to(device)).cpu()


def evaluate_width(ctx: torch.Tensor, cand: torch.Tensor, gold: list[int], width: int,
                   *, task: str, seed: int, scale: float, chunk: int = 256) -> dict:
    """Accuracy of the frozen model on test rows packed with independent draws."""
    scale = float(scale)
    n = len(gold)
    preds: list[int] = []
    correct: list[int] = []
    nll_total = 0.0
    for start in range(0, n, chunk):
        idx = list(range(start, min(n, start + chunk)))
        logits_rows = []
        gold_slots = []
        for i in idx:
            rng = random.Random(stable_seed("e4-test", task, width, seed, i))
            subset = sample_subset(cand.shape[0], width, gold[i], rng)
            logits_rows.append(scale * (ctx[i] @ cand[subset].T))
            gold_slots.append(gold_slot(subset, gold[i]))
        logits = torch.stack(logits_rows)
        nll_total += torch.nn.functional.cross_entropy(
            logits, torch.tensor(gold_slots), reduction="sum"
        ).item()
        top = logits.argmax(-1).tolist()
        for row, j in enumerate(top):
            # map slot back to class id by regenerating the same subset
            i = idx[row]
            rng = random.Random(stable_seed("e4-test", task, width, seed, i))
            subset = sample_subset(cand.shape[0], width, gold[i], rng)
            preds.append(subset[j])
            correct.append(int(subset[j] == gold[i]))
    k = sum(correct)
    lo, hi = wilson(k, n)
    return {
        "width": width,
        "n": n,
        "correct": k,
        "accuracy": k / n,
        "wilson95": [lo, hi],
        "chance": 1.0 / width,
        "nll": nll_total / n,
        "predictions": preds,
        "correct_bits_hex": bits_hex(correct),
    }


def bits_hex(bits: list[int]) -> str:
    out = bytearray()
    acc = 0
    n = 0
    for b in bits:
        acc |= (b & 1) << n
        n += 1
        if n == 8:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc)
    return out.hex()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--rung", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=4000)
    ap.add_argument("--test-examples", type=int, default=2000)
    ap.add_argument("--max-seq", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["cpu", "mps", "cuda", "auto"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="50 steps and 100 test rows; never a reported result")
    args = ap.parse_args()

    if args.smoke:
        args.steps = 50
        args.test_examples = 100

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "result.json").exists():
        raise SystemExit(f"refusing to overwrite {out_dir/'result.json'}")

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    tok = load_tokenizer("byte")
    cfg = rung(args.rung, "byte")
    if args.max_seq:
        import dataclasses
        cfg = dataclasses.replace(cfg, max_seq=args.max_seq)

    spec = TASKS[args.task]
    bank = candidate_bank(spec)
    k = len(bank)
    if args.width > k:
        raise SystemExit(f"--width {args.width} exceeds K={k} for {args.task}")

    train_rows = stratified_examples(spec, bank, "train", args.examples, args.seed)
    test_rows = stratified_examples(spec, bank, "test", args.test_examples, args.seed + 1)
    train_ctx = [row.context for row in train_rows]

    # Pack contexts exactly as the suite does (dual path, no field splitting).
    from azurite.decisions import pack_dual
    train_batch, train_stats = pack_dual(tok, spec, bank, train_rows, max_seq=cfg.max_seq)
    test_batch, test_stats = pack_dual(tok, spec, bank, test_rows, max_seq=cfg.max_seq)
    if train_stats["unpackable"] or test_stats["unpackable"]:
        raise SystemExit("packing discarded rows; refusing to proceed")

    # Candidate tensors: tokenized once, width is applied by gathering rows.
    if not bank.token_ids:
        bank.tokenize(tok)
    width_max = max(len(c) for c in bank.token_ids)
    cand_ids = torch.full((k, width_max), PAD_ID, dtype=torch.long)
    cand_mask = torch.zeros((k, width_max), dtype=torch.bool)
    for i, c in enumerate(bank.token_ids):
        cand_ids[i, : len(c)] = torch.tensor(c, dtype=torch.long)
        cand_mask[i, : len(c)] = True

    # The decision head is randomly initialised on top of the pretrained encoder.
    # Without this the head differs on every launch and two widths are two
    # different models for a reason that has nothing to do with width. The suite
    # seeds here; the first E4 launch did not, and the width-K harness check
    # caught it (87.30% against the published 88.45%).
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    encoder = Azurite(cfg)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = state.get("config", {})
    if (saved.get("d_model"), saved.get("n_layer"), saved.get("max_seq")) != (
        cfg.d_model, cfg.n_layer, cfg.max_seq
    ):
        raise SystemExit(f"checkpoint config {saved} does not match rung config {cfg.to_dict()}")
    encoder.load_state_dict(state["model"], strict=True)
    model = AzuriteDual(encoder).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    gold_train = train_batch.gold.tolist()
    gold_test = test_batch.gold.tolist()
    n_train = len(gold_train)

    schedule = list(batch_schedule(n_train, args.batch, args.steps, args.seed))
    print(f"task={args.task} K={k} width={args.width} steps={args.steps} "
          f"device={device} train={n_train} test={len(gold_test)}", flush=True)

    t0 = time.perf_counter()
    losses: list[float] = []
    for step, idx_t in enumerate(schedule):
        model.train()
        opt.zero_grad(set_to_none=True)
        idx = idx_t.tolist()
        ids = train_batch.context_ids[idx].to(device)
        mask = train_batch.context_mask[idx].to(device)
        ctx = model.embed(ids, mask)
        cand = model.embed(cand_ids.to(device), cand_mask.to(device))
        logits = model.log_scale.exp() * (ctx @ cand.T)

        rows = []
        slots = []
        for row, i in enumerate(idx):
            rng = random.Random(stable_seed("e4-train", args.task, args.width,
                                            args.seed, step, i))
            subset = sample_subset(k, args.width, gold_train[i], rng)
            rows.append(subset)
            slots.append(gold_slot(subset, gold_train[i]))
        gathered = torch.stack([
            logits[r][torch.tensor(rows[r], device=device)] for r in range(len(rows))
        ])
        loss = torch.nn.functional.cross_entropy(
            gathered, torch.tensor(slots, device=device)
        )
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 500 == 0 or step == len(schedule) - 1:
            rate = (time.perf_counter() - t0) / (step + 1)
            print(f"  step {step:>4}/{len(schedule)} loss {float(loss):.4f} "
                  f"{rate*1000:.0f} ms/step", flush=True)

    train_seconds = time.perf_counter() - t0
    ctx_test = embed_contexts(model, test_batch, device)
    cand_emb = embed_candidates(model, cand_ids, cand_mask, device)
    scale = float(model.log_scale.exp().item())

    # Final evaluation at the trained width. No model selection: one number.
    at_width = evaluate_width(ctx_test, cand_emb, gold_test, args.width,
                              task=args.task, seed=args.seed, scale=scale)
    # Full-bank reference from the same weights: shows what width training cost
    # on the original bank rather than on a subset of it.
    full_logits = scale * (ctx_test.to(device) @ cand_emb.to(device).T)
    full_pred = full_logits.argmax(-1).tolist()
    full_correct = [int(p == g) for p, g in zip(full_pred, gold_test)]
    full_acc = sum(full_correct) / len(full_correct)
    full_bits = bits_hex(full_correct)

    head_path = out_dir / "final.pt"
    torch.save({
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "path": "dual",
        "task": args.task,
        "seed": args.seed,
        "init": "pretrained",
        "steps": args.steps,
        "width": args.width,
    }, head_path)

    result = {
        "protocol": "E4-width-training",
        "task": args.task,
        "rung": args.rung,
        "k": k,
        "width": args.width,
        "seed": args.seed,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "examples": args.examples,
        "n": at_width["n"],
        "accuracy": at_width["accuracy"],
        "correct": at_width["correct"],
        "wilson95": at_width["wilson95"],
        "chance": at_width["chance"],
        "nll": at_width["nll"],
        "predictions": at_width["predictions"],
        "correct_bits_hex": at_width["correct_bits_hex"],
        "gold_bits_hex": bits_hex(gold_test),
        "full_k": {
            "accuracy": full_acc,
            "correct": sum(full_correct),
            "correct_bits_hex": full_bits,
            "wilson95": list(wilson(sum(full_correct), len(full_correct))),
        },
        "train_seconds": round(train_seconds, 1),
        "final_loss": round(sum(losses[-50:]) / min(50, len(losses)), 4),
        "device": device,
        "config": cfg.to_dict(),
        "train_packing": train_stats,
        "test_packing": test_stats,
        "head": {"path": (str(head_path.relative_to(PROJECT))
                          if head_path.is_relative_to(PROJECT) else str(head_path)),
                 "sha256": sha256_file(head_path)},
        "source_checkpoint": {
            "path": args.ckpt,
            "sha256": sha256_file(Path(args.ckpt)),
            "bytes": Path(args.ckpt).stat().st_size,
        },
        "sampling_rule": (
            "per row: gold plus width-1 uniform draws without replacement from the "
            "other K-1 classes, order shuffled; train seed sha256('e4-train', task, "
            "width, seed, step, row), test seed sha256('e4-test', task, width, seed, row)"
        ),
        "code_sha256": sha256_file(Path(__file__)),
        "runtime": {
            "device": device,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "datasets": {s: _load(spec, s)._fingerprint for s in ("train", "test")},
        "test_gold_sha256": hashlib.sha256(json.dumps(gold_test).encode()).hexdigest(),
        "train_gold_sha256": hashlib.sha256(json.dumps(gold_train).encode()).hexdigest(),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_dir/'result.json'} accuracy={result['accuracy']:.4f} "
          f"full-K={full_acc:.4f} train={train_seconds:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
