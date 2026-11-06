#!/usr/bin/env python3
"""Fine-tune a decision head on top of either a pretrained or a random encoder.

This is the study's central comparison in miniature. The question it answers is
not "does Azurite beat Laya" — at 2.4M parameters it cannot — but the question
that has to be answered *before* spending on a larger rung:

    at a fixed fine-tuning budget, does from-scratch pretraining on the corpus
    buy anything on a decision task, and how much?

The comparison is matched on everything except initialisation: same task rows in
the same order, same number of steps, same batch size, same optimiser, same LR,
same seed. Any accuracy difference is then attributable to the initial weights.

Both conditions train the **same architecture**. The pretrained arm loads the
encoder weights from a `scripts/pretrain.py` checkpoint and discards nothing;
the random arm is the identical model with its default initialisation. The head
(span projection + score) is freshly initialised in both arms, so its parameters
are not a confound.

Reported per run: test accuracy, test NLL, ECE (10 equal-width bins), and
accuracy at 50% and 100% coverage — the three quantities the claim in PLAN.md
§1.2 is stated in terms of.

A limitation stated up front, because it is the one a reader will ask about:
**the learning rate is not tuned per arm.** A fixed 5e-4 is used for both, which
is a matched comparison but not each arm's best. Reporting a per-arm LR sweep is
the obvious next step; reporting one LR for both is the honest minimum.

Usage:
    .venv/bin/python3 scripts/transfer_eval.py --task ag_news --init pretrained \
        --ckpt results/pretrain/l1_tiny_byte/ckpt.pt --seed 0
    .venv/bin/python3 scripts/transfer_eval.py --task ag_news --init random --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import math
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import EncoderConfig, rung  # noqa: E402
from azurite.decisions import (  # noqa: E402
    CROSS_ENCODER_MAX_K,
    TASKS,
    candidate_bank,
    load_examples,
    pack_cross,
    pack_dual,
    stratified_examples,
    _load,
)
from azurite.env import MemoryGuard, configure_mps  # noqa: E402
from azurite.model import Azurite, AzuriteDecision, AzuriteDual, DecisionBatch  # noqa: E402
from azurite.paths import rel as _rel  # noqa: E402
from azurite.tokenizers import PAD_ID, load as load_tokenizer  # noqa: E402

RESULTS = PROJECT / "results" / "transfer"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict) -> str:
    """Content identity independent of torch.save container metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def draw_provenance(examples, bank, task: str, seed: int, path: str) -> dict:
    """Record source labels separately from shuffled candidate-slot labels."""
    rows = []
    counts = {str(i): 0 for i in range(len(bank))}
    for ex in examples:
        order = list(range(len(bank)))
        if path == "cross":
            random.Random(f"{seed}:{task}:{ex.index}").shuffle(order)
        counts[str(ex.gold)] += 1
        rows.append({"index": ex.index, "canonical_gold": ex.gold,
                     "context_sha256": hashlib.sha256(ex.context.encode()).hexdigest(),
                     "candidate_order": order, "packed_gold": order.index(ex.gold)})
    return {"rows": rows, "class_counts": counts,
            "unique_indices": len({ex.index for ex in examples}),
            "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}


def batch_schedule(n_rows: int, batch: int, steps: int, seed: int):
    """Deterministic stream of row indices, reshuffled each epoch.

    This is extracted and tested because it is the matched-budget claim: if the
    pretrained and random arms see different examples in a different order, any
    accuracy difference mixes initialisation with data order. Same seed, same
    schedule — asserted in tests/test_transfer_eval.py.
    """
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_rows, generator=gen)
    cursor = 0
    for _ in range(steps):
        if cursor + batch > n_rows:
            order = torch.randperm(n_rows, generator=gen)
            cursor = 0
        idx = order[cursor : cursor + batch]
        cursor += batch
        yield idx


def slice_batch(batch: DecisionBatch, idx: torch.Tensor) -> DecisionBatch:
    return DecisionBatch(
        ids=batch.ids[idx],
        pad_mask=batch.pad_mask[idx],
        spans=batch.spans[idx],
        cand_mask=batch.cand_mask[idx],
        gold=batch.gold[idx],
    )


def candidate_tensor(bank, tok) -> tuple[torch.Tensor, torch.Tensor]:
    if not bank.token_ids:
        bank.tokenize(tok)
    width = max(len(c) for c in bank.token_ids)
    ids = torch.full((len(bank), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(bank), width), dtype=torch.bool)
    for i, c in enumerate(bank.token_ids):
        ids[i, : len(c)] = torch.tensor(c, dtype=torch.long)
        mask[i, : len(c)] = True
    return ids, mask


def build_model(cfg: EncoderConfig, path: str, ckpt: str | None, device: str):
    encoder = Azurite(cfg)
    loaded = None
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        saved = EncoderConfig(**state["config"])
        if (saved.d_model, saved.n_layer, saved.max_seq, saved.vocab_size) != (
            cfg.d_model,
            cfg.n_layer,
            cfg.max_seq,
            cfg.vocab_size,
        ):
            raise SystemExit(
                f"checkpoint config {saved.to_dict()} does not match requested "
                f"{cfg.to_dict()}; refusing to load mismatched weights"
            )
        # strict=True: a partially loaded encoder would make the two arms
        # differ in ways the result does not disclose.
        report = encoder.load_state_dict(state["model"], strict=True)
        loaded = {
            "path": _rel(Path(ckpt)),
            "pretrain_step": state.get("step"),
            "pretrain_tokens": state.get("tokens_seen"),
            "missing": list(report.missing_keys),
            "unexpected": list(report.unexpected_keys),
        }
        if report.missing_keys or report.unexpected_keys:
            raise SystemExit(f"checkpoint does not match the model: {loaded}")
    model = AzuriteDecision(encoder) if path == "cross" else AzuriteDual(encoder)
    return model.to(device), loaded


def dual_part(batch, idx: torch.Tensor, device: str):
    return type(batch)(
        context_ids=batch.context_ids[idx].to(device),
        context_mask=batch.context_mask[idx].to(device),
        gold=batch.gold[idx].to(device),
    )


@torch.no_grad()
def evaluate(
    model, batch, device: str, cand_args=None, chunk: int = 64, keep_predictions: bool = False,
    keep_probabilities: bool = False,
) -> dict:
    """Accuracy, NLL, ECE and risk-at-coverage on a packed evaluation set.

    `keep_predictions` records per-example predictions and gold labels. The
    packed test set is identical across arms (same seed, same rows), so row i is
    the same example in both runs and the two arms can be compared with a paired
    McNemar test instead of two independent proportions — with 1,000 test rows
    the unpaired comparison has a ~1.5 point standard error and would not resolve
    the differences this study is looking for.
    """
    model.eval()
    logits_all = []
    gold_all = []
    n = batch.gold.shape[0]
    for start in range(0, n, chunk):
        idx = torch.arange(start, min(n, start + chunk))
        if cand_args is None:
            logits = model(slice_batch(batch, idx).to(device))
        else:
            part = dual_part(batch, idx, device)
            logits = model(part.context_ids, part.context_mask, *cand_args)
        logits_all.append(logits.float().cpu())
        gold_all.append(batch.gold[idx])
    logits = torch.cat(logits_all)
    gold = torch.cat(gold_all)

    probs = torch.softmax(logits, dim=-1)
    pred = logits.argmax(-1)
    acc = (pred == gold).float().mean().item()
    nll = torch.nn.functional.cross_entropy(logits, gold).item()

    conf, _ = probs.max(-1)
    correct = (pred == gold).float()
    bins = 10
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi) if b else (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(
                conf[m].mean().item() - correct[m].mean().item()
            )

    order = torch.argsort(conf, descending=True)
    def acc_at(cov: float) -> float:
        k = max(1, int(round(cov * len(order))))
        return correct[order[:k]].mean().item()

    out = {
        "n": int(len(gold)),
        "accuracy": round(acc, 4),
        "nll": round(nll, 4),
        "ece": round(ece, 4),
        "acc_at_50pct_coverage": round(acc_at(0.5), 4),
        "acc_at_100pct_coverage": round(acc, 4),
        "mean_confidence": round(conf.mean().item(), 4),
    }
    if keep_predictions:
        out["predictions"] = pred.tolist()
        out["gold"] = gold.tolist()
    if keep_probabilities:
        out["probabilities"] = probs.tolist()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--rung", default="tiny")
    ap.add_argument("--init", choices=["random", "pretrained"], required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--path", choices=["auto", "cross", "dual"], default="auto")
    ap.add_argument("--examples", type=int, default=4000)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--test-examples", type=int, default=1000)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-cap-gib", type=float, default=6.0)
    ap.add_argument("--stratify", action="store_true",
                    help=("class-balanced draw of the train/test subsets; the "
                          "Banking77 splits are label-sorted, so a prefix is a "
                          "few classes rather than a sample of the task"))
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", choices=["cpu"], default=None,
                    help="explicit CPU-only execution; default preserves prior device selection")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--evidence-dir", default="",
                    help="retain draw provenance, initial identity and final checkpoint")
    args = ap.parse_args()

    if args.init == "pretrained" and not args.ckpt:
        raise SystemExit("--init pretrained needs --ckpt")

    if args.threads is not None:
        if args.threads < 1:
            raise SystemExit("--threads must be positive")
        torch.set_num_threads(args.threads)
    if args.device == "cpu":
        device = "cpu"
    else:
        applied = configure_mps(args.mem_cap_gib)
        device = "mps" if applied["mps"] else "cpu"
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None
    if evidence_dir:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        if (evidence_dir / "final.pt").exists() or (args.out and Path(args.out).exists()):
            raise SystemExit("refusing to overwrite completed evidence")
    tok = load_tokenizer(args.arm)
    cfg = rung(args.rung, args.arm)

    spec = TASKS[args.task]
    bank = candidate_bank(spec)
    path = args.path
    if path == "auto":
        path = "cross" if len(bank) <= CROSS_ENCODER_MAX_K else "dual"

    if args.stratify:
        train_rows = stratified_examples(spec, bank, "train", args.examples, args.split_seed)
        test_rows = stratified_examples(spec, bank, "test", args.test_examples,
                                        args.split_seed + 1)
    else:
        train_rows = load_examples(spec, bank, split="train", limit=args.examples)
        test_rows = load_examples(spec, bank, split="test", limit=args.test_examples)

    if path == "cross":
        train_batch, pack_stats = pack_cross(
            tok, spec, bank, train_rows, max_seq=cfg.max_seq, seed=args.seed
        )
        test_batch, test_stats = pack_cross(
            tok, spec, bank, test_rows, max_seq=cfg.max_seq, seed=args.seed
        )
    else:
        train_batch, pack_stats = pack_dual(tok, spec, bank, train_rows, max_seq=cfg.max_seq)
        test_batch, test_stats = pack_dual(tok, spec, bank, test_rows, max_seq=cfg.max_seq)

    # Both arms must start from the same global RNG state, or the "random" arm
    # would be a different random model on every invocation and the comparison
    # would not be reproducible at all. This was missing in the first version.
    torch.manual_seed(args.seed)
    model, loaded = build_model(
        cfg, path, args.ckpt if args.init == "pretrained" else None, device
    )

    cand_args = None
    if path == "dual":
        cand_ids, cand_mask = candidate_tensor(bank, tok)
        cand_args = (
            cand_ids.to(device),
            cand_mask.to(device),
            torch.ones(len(bank), dtype=torch.bool, device=device),
        )

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    n_train = train_batch.gold.shape[0]
    steps_per_epoch = max(1, math.ceil(n_train / args.batch))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    print(
        f"task={args.task} path={path} K={len(bank)} init={args.init} "
        f"train={n_train} test={test_batch.gold.shape[0]} steps={total_steps} "
        f"device={device}",
        flush=True,
    )

    provenance = None
    if evidence_dir:
        if pack_stats["unpackable"] or test_stats["unpackable"]:
            raise SystemExit("evidence protocol requires no discarded rows")
        train_draw = draw_provenance(train_rows, bank, args.task, args.seed, path)
        test_draw = draw_provenance(test_rows, bank, args.task, args.seed, path)
        if test_batch.gold.tolist() != [r["packed_gold"] for r in test_draw["rows"]]:
            raise SystemExit("draw provenance does not reproduce packed test gold")
        if train_batch.gold.tolist() != [r["packed_gold"] for r in train_draw["rows"]]:
            raise SystemExit("draw provenance does not reproduce packed train gold")
        schedule = [idx.tolist() for idx in batch_schedule(n_train, args.batch, total_steps, args.seed)]
        provenance = {
            "stratified": args.stratify, "split_seed": args.split_seed,
            "test_split_seed": args.split_seed + 1, "train": train_draw, "test": test_draw,
            "candidate_labels": bank.labels, "candidate_raw_ids": bank.raw_ids,
            "dataset_fingerprints": {s: _load(spec, s)._fingerprint for s in ("train", "test")},
            "packing": {"train": pack_stats, "test": test_stats},
            "schedule_sha256": hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
            "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": 0.01,
                          "betas": [0.9, 0.999], "eps": 1e-8, "clip_grad_norm": 1.0},
            "initial_model_sha256": state_sha256(model.state_dict()),
            "initial_head_sha256": state_sha256({k: v for k, v in model.state_dict().items()
                                                  if not k.startswith("encoder.")}),
            "source_checkpoint": ({"path": args.ckpt, "sha256": file_sha256(Path(args.ckpt)),
                                    "bytes": Path(args.ckpt).stat().st_size} if args.ckpt else None),
            "source_sha256": {str(p.relative_to(PROJECT)): file_sha256(p)
                              for p in [Path(__file__), *sorted((PROJECT / "azurite").glob("*.py"))]},
            "runtime": {"device": device, "torch": torch.__version__, "python": sys.version,
                        "platform": platform.platform(), "threads": torch.get_num_threads()},
            "config": cfg.to_dict(),
        }
        (evidence_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    guard = MemoryGuard(cap_gib=args.mem_cap_gib)
    t0 = time.perf_counter()
    history = []

    for step, idx in enumerate(batch_schedule(n_train, args.batch, total_steps, args.seed)):
        model.train()
        opt.zero_grad(set_to_none=True)
        if path == "cross":
            batch = slice_batch(train_batch, idx).to(device)
            logits = model(batch)
            loss = AzuriteDecision.loss(logits, batch.gold)
        else:
            part = dual_part(train_batch, idx, device)
            logits = model(part.context_ids, part.context_mask, *cand_args)
            loss = torch.nn.functional.cross_entropy(logits, part.gold)
        if not torch.isfinite(loss):
            print(f"FATAL non-finite loss at step {step}", flush=True)
            return 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % max(1, total_steps // 6) == 0 or step == total_steps - 1:
            ev = evaluate(model, test_batch, device, cand_args)
            history.append({"step": step, **ev})
            print(
                f"  step {step:>4}/{total_steps}  loss {float(loss):.4f}  "
                f"test acc {ev['accuracy']:.4f}  ece {ev['ece']:.3f}",
                flush=True,
            )
        guard.check()
        if step % 100 == 99:
            guard.reclaim_if_needed()

    elapsed = time.perf_counter() - t0
    final = evaluate(model, test_batch, device, cand_args, keep_predictions=True,
                     keep_probabilities=bool(evidence_dir))
    retained = None
    if evidence_dir:
        final["canonical_gold"] = [ex.gold for ex in test_rows]
        final["canonical_predictions"] = [r["candidate_order"][pred]
                                            for r, pred in zip(provenance["test"]["rows"],
                                                               final["predictions"])]
        checkpoint = evidence_dir / "final.pt"
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "path": path,
                    "task": args.task, "seed": args.seed, "init": args.init,
                    "steps": total_steps}, checkpoint)
        retained = {"path": str(checkpoint), "sha256": file_sha256(checkpoint),
                    "state_sha256": state_sha256(model.state_dict()),
                    "bytes": checkpoint.stat().st_size,
                    "provenance_path": str(evidence_dir / "provenance.json"),
                    "provenance_sha256": file_sha256(evidence_dir / "provenance.json")}
    tokens_seen = total_steps * args.batch * pack_stats["mean_seq_len"]
    result = {
        "task": args.task,
        "arm": args.arm,
        "rung": args.rung,
        "path": path,
        "init": args.init,
        "loaded_from": loaded,
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in params),
        "freeze_encoder": bool(args.freeze_encoder),
        "k": len(bank),
        "train_examples": n_train,
        "train_unpackable": pack_stats["unpackable"],
        "test_examples": test_stats["packed"],
        "test_unpackable": test_stats["unpackable"],
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "steps": total_steps,
        "fine_tune_tokens": int(tokens_seen),
        "seconds": round(elapsed, 1),
        "final": final,
        "history": history,
        "note": (
            "learning rate is fixed across arms, not tuned per arm; a matched "
            "comparison, not each arm's best"
        ),
    }
    if evidence_dir:
        result["evidence"] = retained
        result["device"] = device
        result["stratified"] = bool(args.stratify)
        result["split_seed"] = args.split_seed
    out = (
        Path(args.out)
        if args.out
        else RESULTS / f"{args.task}_{args.rung}_{args.init}_s{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        f"final test acc {final['accuracy']:.4f}  ece {final['ece']:.4f}  "
        f"acc@50%cov {final['acc_at_50pct_coverage']:.4f}  {elapsed:.0f}s",
        flush=True,
    )
    print(f"wrote {_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
