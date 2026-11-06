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
import dataclasses
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
    pack_cross_subset,
    pack_dual,
    stratified_examples,
    _load,
)
from azurite.env import (  # noqa: E402
    MemoryGuard, amp_policy, configure_mps, select_device,
)
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


def draw_provenance(examples, bank, task: str, seed: int, path: str,
                    shuffle: bool = True) -> dict:
    """Record source labels separately from shuffled candidate-slot labels."""
    rows = []
    counts = {str(i): 0 for i in range(len(bank))}
    for ex in examples:
        order = list(range(len(bank)))
        if path == "cross" and shuffle:
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


def build_model(cfg: EncoderConfig, path: str, ckpt: str | None, device: str,
                allow_context_extension: bool = False, cross_head: str = "tanh"):
    encoder = Azurite(cfg)
    loaded = None
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        saved = EncoderConfig(**state["config"])
        saved_shape = (saved.d_model, saved.n_layer, saved.vocab_size)
        requested_shape = (cfg.d_model, cfg.n_layer, cfg.vocab_size)
        context_extension = False
        if saved_shape != requested_shape:
            raise SystemExit(
                f"checkpoint config {saved.to_dict()} does not match requested "
                f"{cfg.to_dict()}; refusing to load mismatched weights"
            )
        if saved.max_seq != cfg.max_seq:
            # RoPE has no learned position parameters. Extending the window is
            # therefore a config change only when every other dimension is
            # identical, and only when the caller explicitly asks for it. The
            # default stays strict so an accidental context change cannot be
            # mistaken for a controlled arm.
            if not allow_context_extension or cfg.max_seq < saved.max_seq:
                raise SystemExit(
                    f"checkpoint max_seq {saved.max_seq} does not match requested "
                    f"{cfg.max_seq}; pass --allow-context-extension to grow the "
                    "RoPE window explicitly"
                )
            context_extension = True
        # strict=True: a partially loaded encoder would make the two arms
        # differ in ways the result does not disclose.
        report = encoder.load_state_dict(state["model"], strict=True)
        loaded = {
            "path": _rel(Path(ckpt)),
            "pretrain_step": state.get("step"),
            "pretrain_tokens": state.get("tokens_seen"),
            "max_seq": saved.max_seq,
            "requested_max_seq": cfg.max_seq,
            "context_extension": context_extension,
            "missing": list(report.missing_keys),
            "unexpected": list(report.unexpected_keys),
        }
        if report.missing_keys or report.unexpected_keys:
            raise SystemExit(f"checkpoint does not match the model: {loaded}")
    if path == "cross":
        model = AzuriteDecision(encoder, head=cross_head)
    else:
        model = AzuriteDual(encoder)
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


def run_train_loop(model, path, train_batch, test_batch, device, cand_args, opt,
                   total_steps, batch_size, seed, guard, amp, log_prefix="") -> tuple:
    """Shared optimizer loop for the cross/dual/funnel training phases.

    Returns (history, elapsed_s, fatal). `fatal` is True on a non-finite loss;
    the caller turns that into its failure path. History cadence matches the
    historical inline loop exactly (6 checkpoints + final step).
    """
    n_train = train_batch.gold.shape[0]
    t0 = time.perf_counter()
    history = []
    for step, idx in enumerate(batch_schedule(n_train, batch_size, total_steps, seed)):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=amp[0] if amp else device,
                            dtype=amp[1] if amp else torch.float32,
                            enabled=amp is not None):
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
            return history, time.perf_counter() - t0, True
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % max(1, total_steps // 6) == 0 or step == total_steps - 1:
            ev = evaluate(model, test_batch, device, cand_args)
            history.append({"step": step, **ev})
            print(
                f"  {log_prefix}step {step:>4}/{total_steps}  loss {float(loss):.4f}  "
                f"test acc {ev['accuracy']:.4f}  ece {ev['ece']:.3f}",
                flush=True,
            )
        guard.check()
        if step % 100 == 99:
            guard.reclaim_if_needed()
    return history, time.perf_counter() - t0, False


@torch.no_grad()
def retrieve_topm(model, batch, cand_args, m: int, device: str, chunk: int = 64) -> tuple:
    """Dual-score every row against all K candidates; return top-M keep-lists.

    Returns (keep, recall) where keep[i] is the list of bank indices in rank
    order (best first) and recall is the fraction of rows whose gold is kept.
    """
    model.eval()
    scores_all = []
    n = batch.gold.shape[0]
    for start in range(0, n, chunk):
        idx = torch.arange(start, min(n, start + chunk))
        part = dual_part(batch, idx, device)
        scores_all.append(model(part.context_ids, part.context_mask, *cand_args).float().cpu())
    scores = torch.cat(scores_all)
    topm = scores.topk(min(m, scores.shape[1]), dim=-1).indices.tolist()
    gold = batch.gold.tolist()
    recall = sum(1 for g, k in zip(gold, topm) if g in k) / max(1, len(gold))
    return topm, round(recall, 4)


def funnel_map_predictions(subset_pred: list[int], keep_full: list[list[int]],
                           packed_indices: list[int], n_full: int) -> list[int | None]:
    """Map subset-slot predictions back to bank indices over all rows.

    Rows dropped as funnel misses map to None (an end-to-end error). Pure
    function so the miss-accounting is unit-testable without a model.
    """
    mapped: list[int | None] = [None] * n_full
    for pred, row in zip(subset_pred, packed_indices):
        mapped[row] = keep_full[row][pred]
    return mapped


def run_funnel(args, tok, cfg, spec, bank, train_rows, test_rows, device, amp,
               evidence_dir) -> int:
    """Retrieve-then-rerank funnel: dual retrieves top-M, cross decides among M.

    Phase A trains the dual scorer on the full K (exactly the dual path).
    Phase B trains a fresh cross head on the retrieved subsets (rank order, no
    shuffle — rank is signal). The decision metric is end-to-end test accuracy
    over ALL test rows with funnel misses counted as errors, never the
    retrieved-only accuracy alone.

    Phase-B head init is seeded with seed+1 (documented, deterministic): it is
    a different head from phase A by construction, not by accident.
    """
    m = args.funnel_m
    steps_per_epoch = max(1, math.ceil(len(train_rows) / args.batch))
    budgeted = max(1, int(steps_per_epoch * args.epochs))
    if args.steps:
        budgeted = args.steps
    dual_steps = args.funnel_dual_steps or budgeted
    cross_steps = args.funnel_cross_steps or budgeted
    print(f"task={args.task} path=funnel K={len(bank)} M={m} init={args.init} "
          f"train={len(train_rows)} test={len(test_rows)} "
          f"dual_steps={dual_steps} cross_steps={cross_steps} device={device}",
          flush=True)

    guard = MemoryGuard(cap_gib=args.mem_cap_gib)
    cand_ids, cand_mask = candidate_tensor(bank, tok)
    cand_args = (cand_ids.to(device), cand_mask.to(device),
                 torch.ones(len(bank), dtype=torch.bool, device=device))

    # Phase A: full-K dual retriever.
    torch.manual_seed(args.seed)
    dual_model, loaded = build_model(
        cfg, "dual", args.ckpt if args.init == "pretrained" else None, device,
        allow_context_extension=args.allow_context_extension,
        cross_head=args.cross_head)
    if args.freeze_encoder:
        for p in dual_model.encoder.parameters():
            p.requires_grad_(False)
    train_full, pack_full = pack_dual(tok, spec, bank, train_rows, max_seq=cfg.max_seq,
                                      field_aware=args.pack_fields)
    test_full, test_full_stats = pack_dual(tok, spec, bank, test_rows, max_seq=cfg.max_seq,
                                           field_aware=args.pack_fields)
    dual_params = [p for p in dual_model.parameters() if p.requires_grad]
    dual_opt = torch.optim.AdamW(dual_params, lr=args.lr, weight_decay=0.01)
    history_a, elapsed_a, fatal = run_train_loop(
        dual_model, "dual", train_full, test_full, device, cand_args, dual_opt,
        dual_steps, args.batch, args.seed, guard, amp, log_prefix="A/")
    if fatal:
        return 1
    final_a = evaluate(dual_model, test_full, device, cand_args)

    keep_train, recall_train = retrieve_topm(dual_model, train_full, cand_args, m, device)
    keep_test, recall_test = retrieve_topm(dual_model, test_full, cand_args, m, device)
    print(f"  funnel recall@{m}: train {recall_train:.4f} test {recall_test:.4f}", flush=True)

    train_sub, pack_sub = pack_cross_subset(
        tok, spec, bank, train_rows, keep_train, max_seq=cfg.max_seq)
    test_sub, test_sub_stats = pack_cross_subset(
        tok, spec, bank, test_rows, keep_test, max_seq=cfg.max_seq)
    # Evidence protocol, funnel-aware: the packed subset gold must be exactly
    # the remapped gold for the packed rows, and every packed row's gold must
    # resolve to its own candidate span.
    packed_gold_check = [keep_test[i].index(test_rows[i].gold)
                         for i in test_sub_stats["packed_indices"]]
    assert test_sub.gold.tolist() == packed_gold_check, "subset gold remap mismatch"

    # Phase B: fresh cross head over the retrieved subsets.
    torch.manual_seed(args.seed + 1)
    cross_model, _ = build_model(
        cfg, "cross", args.ckpt if args.init == "pretrained" else None, device,
        allow_context_extension=args.allow_context_extension,
        cross_head=args.cross_head)
    if args.freeze_encoder:
        for p in cross_model.encoder.parameters():
            p.requires_grad_(False)
    cross_params = [p for p in cross_model.parameters() if p.requires_grad]
    cross_opt = torch.optim.AdamW(cross_params, lr=args.lr, weight_decay=0.01)
    history_b, elapsed_b, fatal = run_train_loop(
        cross_model, "cross", train_sub, test_sub, device, None, cross_opt,
        cross_steps, args.batch, args.seed + 1, guard, amp, log_prefix="B/")
    if fatal:
        return 1
    for h in history_b:
        h["retrieved_only"] = True
    final_b = evaluate(cross_model, test_sub, device, None, keep_predictions=True)

    # End-to-end: map subset predictions to bank indices; misses are errors.
    gold_full = [ex.gold for ex in test_rows]
    mapped = funnel_map_predictions(final_b["predictions"], keep_test,
                                    test_sub_stats["packed_indices"], len(test_rows))
    correct = sum(1 for p, g in zip(mapped, gold_full) if p == g)
    e2e_acc = round(correct / len(gold_full), 4)
    print(f"  funnel end-to-end test acc {e2e_acc:.4f} "
          f"(retrieved-only {final_b['accuracy']:.4f}, misses "
          f"{test_sub_stats['funnel_miss']})  {elapsed_a + elapsed_b:.0f}s", flush=True)

    retained = None
    if evidence_dir:
        retr_path = evidence_dir / "retriever.pt"
        torch.save({"model": dual_model.state_dict(), "config": cfg.to_dict(),
                    "path": "dual", "task": args.task, "seed": args.seed,
                    "init": args.init, "steps": dual_steps}, retr_path)
        final_path = evidence_dir / "final.pt"
        torch.save({"model": cross_model.state_dict(), "config": cfg.to_dict(),
                    "path": "cross-funnel", "task": args.task, "seed": args.seed + 1,
                    "init": args.init, "steps": cross_steps}, final_path)
        provenance = {
            "stratified": args.stratify, "split_seed": args.split_seed,
            "cross_shuffle": "rank-order",
            "test_split_seed": args.split_seed + 1,
            "funnel": {"m": m, "dual_steps": dual_steps, "cross_steps": cross_steps,
                       "phase_b_seed": args.seed + 1,
                       "recall_train": recall_train, "recall_test": recall_test,
                       "keep_train": keep_train, "keep_test": keep_test,
                       "packed_indices": test_sub_stats["packed_indices"],
                       "packing": {"train": pack_sub, "test": test_sub_stats}},
            "initial_model_sha256": state_sha256(cross_model.state_dict()),
            "source_checkpoint": ({"path": args.ckpt, "sha256": file_sha256(Path(args.ckpt)),
                                    "bytes": Path(args.ckpt).stat().st_size} if args.ckpt else None),
            "source_sha256": {str(p.relative_to(PROJECT)): file_sha256(p)
                              for p in [Path(__file__), *sorted((PROJECT / "azurite").glob("*.py"))]},
            "runtime": {"device": device, "torch": torch.__version__, "python": sys.version,
                        "platform": platform.platform(), "threads": torch.get_num_threads()},
            "config": cfg.to_dict(),
            "cross_head": args.cross_head,
        }
        (evidence_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        retained = {"final": {"path": str(final_path), "sha256": file_sha256(final_path)},
                    "retriever": {"path": str(retr_path), "sha256": file_sha256(retr_path)},
                    "provenance_sha256": file_sha256(evidence_dir / "provenance.json")}

    result = {
        "task": args.task, "arm": args.arm, "rung": args.rung, "path": "funnel",
        "funnel_m": m, "init": args.init, "loaded_from": loaded,
        "cross_head": args.cross_head,
        "k": len(bank), "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "dual_steps": dual_steps, "cross_steps": cross_steps,
        "batch": args.batch, "lr": args.lr, "seed": args.seed,
        "seconds_retriever": round(elapsed_a, 1),
        "seconds_reranker": round(elapsed_b, 1),
        "seconds": round(elapsed_a + elapsed_b, 1),
        "funnel_recall_train": recall_train, "funnel_recall_test": recall_test,
        "funnel_miss_test": test_sub_stats["funnel_miss"],
        "retrieved_only": {k: final_b[k] for k in ("n", "accuracy", "nll", "ece")},
        "final": {"n": len(test_rows), "accuracy": e2e_acc,
                  "nll": None, "ece": None,
                  "note": ("end-to-end over ALL test rows, misses as errors; "
                             "NLL/ECE reported retrieved-only")},
        "history_retriever": history_a, "history_reranker": history_b,
        "note": ("learning rate is fixed across phases, not tuned; "
                 "phase-B head seeded seed+1, documented"),
    }
    if evidence_dir:
        result["evidence"] = retained
        result["device"] = device
        result["stratified"] = bool(args.stratify)
        result["split_seed"] = args.split_seed
    out = Path(args.out) if args.out else RESULTS / f"{args.task}_{args.rung}_funnel_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {_rel(out)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--rung", default="tiny")
    ap.add_argument("--init", choices=["random", "pretrained"], required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--path", choices=["auto", "cross", "dual", "funnel"], default="auto")
    ap.add_argument("--funnel-m", type=int, default=8,
                    help=("funnel path only: candidates retrieved per example; "
                          "must satisfy 2 <= M <= cross cap (16)"))
    ap.add_argument("--funnel-dual-steps", type=int, default=0,
                    help="funnel phase-A (retriever) steps; 0 = use --steps")
    ap.add_argument("--funnel-cross-steps", type=int, default=0,
                    help="funnel phase-B (reranker) steps; 0 = use --steps")
    ap.add_argument("--cross-head", choices=["tanh", "mlp"], default="tanh",
                    help=("cross/funnel phase-B head: 'tanh' (default, historical) "
                          "or 'mlp' (LayerNorm+GELU residual variant)"))
    ap.add_argument("--cross-shuffle", choices=["on", "off"], default="on",
                    help="cross path only: shuffle per-example candidate order; "
                    "off uses canonical bank order (matches dual)")
    ap.add_argument("--pack-fields", action="store_true",
                    help=("dual path only: split only overflowing multi-field rows "
                          "into equal shares so no field is silently dropped; "
                          "fitting rows keep their historical packed form"))
    ap.add_argument("--examples", type=int, default=4000)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=0,
                    help="exact optimizer steps; overrides the epoch-derived budget")
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
    ap.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default=None,
                    help="explicit device; default preserves the prior MPS/CPU selection")
    ap.add_argument("--max-seq", type=int, default=0,
                    help="override the rung context length when it differs from the checkpoint")
    ap.add_argument("--allow-context-extension", action="store_true",
                    help=("permit max_seq to grow above the checkpoint's RoPE table; "
                          "weights must otherwise match exactly"))
    ap.add_argument("--eval-lengths", nargs="+", type=int, default=[],
                    help=("after training, evaluate the same weights at each context "
                          "length; dual path only"))
    ap.add_argument("--threads", type=int, default=None)
    # Same policy as pretrain: bf16 autocast is measured on CUDA and off by
    # default everywhere else, so an eval cell cannot silently change dtype.
    ap.add_argument("--amp", choices=("off", "auto", "bf16", "fp16"), default="off")
    ap.add_argument("--evidence-dir", default="",
                    help="retain draw provenance, initial identity and final checkpoint")
    args = ap.parse_args()

    if args.init == "pretrained" and not args.ckpt:
        raise SystemExit("--init pretrained needs --ckpt")

    if args.threads is not None:
        if args.threads < 1:
            raise SystemExit("--threads must be positive")
        torch.set_num_threads(args.threads)
    if args.device in {"cpu", "mps", "cuda"}:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda was requested but CUDA is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("--device mps was requested but MPS is unavailable")
    elif args.device == "auto":
        device = select_device()
    else:
        applied = configure_mps(args.mem_cap_gib)
        device = "mps" if applied["mps"] else "cpu"
    amp = amp_policy(args.amp, device)
    print(
        f"amp {args.amp} -> "
        + (f"{str(amp[1]).replace('torch.', '')} autocast on {amp[0]}" if amp else "off"),
        flush=True,
    )
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None
    if evidence_dir:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        if (evidence_dir / "final.pt").exists() or (args.out and Path(args.out).exists()):
            raise SystemExit("refusing to overwrite completed evidence")
    tok = load_tokenizer(args.arm)
    cfg = rung(args.rung, args.arm)
    if args.max_seq:
        cfg = dataclasses.replace(cfg, max_seq=args.max_seq)

    spec = TASKS[args.task]
    bank = candidate_bank(spec)
    path = args.path
    if path == "auto":
        # The capability path is the dual context-candidate scorer for every K.
        # The cross encoder remains available via --path cross for the separate
        # mechanism/decode comparison, but it is not the default decision head:
        # its candidate-only span score was the context-blind path diagnosed on
        # DAIR Emotion.
        path = "dual"

    if args.pack_fields and path not in ("dual", "funnel"):
        raise SystemExit("--pack-fields is defined for the dual/funnel paths only")
    if path == "funnel":
        if not 2 <= args.funnel_m <= CROSS_ENCODER_MAX_K:
            raise SystemExit(
                f"--funnel-m must satisfy 2 <= M <= {CROSS_ENCODER_MAX_K}, "
                f"got {args.funnel_m}"
            )

    if args.stratify:
        train_rows = stratified_examples(spec, bank, "train", args.examples, args.split_seed)
        test_rows = stratified_examples(spec, bank, "test", args.test_examples,
                                        args.split_seed + 1)
    else:
        train_rows = load_examples(spec, bank, split="train", limit=args.examples)
        test_rows = load_examples(spec, bank, split="test", limit=args.test_examples)

    if path == "funnel":
        return run_funnel(args, tok, cfg, spec, bank, train_rows, test_rows,
                          device, amp, evidence_dir)

    if path == "cross":
        train_batch, pack_stats = pack_cross(
            tok, spec, bank, train_rows, max_seq=cfg.max_seq,
            shuffle=(args.cross_shuffle == "on"), seed=args.seed
        )
        test_batch, test_stats = pack_cross(
            tok, spec, bank, test_rows, max_seq=cfg.max_seq,
            shuffle=(args.cross_shuffle == "on"), seed=args.seed
        )
    else:
        train_batch, pack_stats = pack_dual(
            tok, spec, bank, train_rows, max_seq=cfg.max_seq,
            field_aware=args.pack_fields,
        )
        test_batch, test_stats = pack_dual(
            tok, spec, bank, test_rows, max_seq=cfg.max_seq,
            field_aware=args.pack_fields,
        )

    # Both arms must start from the same global RNG state, or the "random" arm
    # would be a different random model on every invocation and the comparison
    # would not be reproducible at all. This was missing in the first version.
    torch.manual_seed(args.seed)
    model, loaded = build_model(
        cfg, path, args.ckpt if args.init == "pretrained" else None, device,
        allow_context_extension=args.allow_context_extension,
        cross_head=args.cross_head,
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
    if args.steps:
        total_steps = args.steps
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
        train_draw = draw_provenance(train_rows, bank, args.task, args.seed, path,
                                     shuffle=(args.cross_shuffle == "on"))
        test_draw = draw_provenance(test_rows, bank, args.task, args.seed, path,
                                    shuffle=(args.cross_shuffle == "on"))
        if test_batch.gold.tolist() != [r["packed_gold"] for r in test_draw["rows"]]:
            raise SystemExit("draw provenance does not reproduce packed test gold")
        if train_batch.gold.tolist() != [r["packed_gold"] for r in train_draw["rows"]]:
            raise SystemExit("draw provenance does not reproduce packed train gold")
        schedule = [idx.tolist() for idx in batch_schedule(n_train, args.batch, total_steps, args.seed)]
        provenance = {
            "stratified": args.stratify, "split_seed": args.split_seed,
            "cross_shuffle": getattr(args, "cross_shuffle", "on"),
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
            "pack_fields": bool(args.pack_fields),
            "cross_head": args.cross_head,
        }
        (evidence_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    guard = MemoryGuard(cap_gib=args.mem_cap_gib)
    history, elapsed, fatal = run_train_loop(
        model, path, train_batch, test_batch, device, cand_args, opt,
        total_steps, args.batch, args.seed, guard, amp)
    if fatal:
        return 1
    final = evaluate(model, test_batch, device, cand_args, keep_predictions=True,
                     keep_probabilities=bool(evidence_dir))
    context_curve = []
    if args.eval_lengths:
        if path != "dual":
            raise SystemExit("--eval-lengths is defined for the dual path only")
        for length in args.eval_lengths:
            if length < 8:
                raise SystemExit(f"evaluation length must be >= 8, got {length}")
            curve_batch, curve_stats = pack_dual(
                tok, spec, bank, test_rows, max_seq=length,
                field_aware=args.pack_fields,
            )
            curve_eval = evaluate(
                model, curve_batch, device, cand_args, keep_predictions=True,
                keep_probabilities=False,
            )
            raw_lengths = [len(tok.encode(ex.context)) + 2 for ex in test_rows]
            curve_eval["full_context_fraction"] = round(
                sum(1 for n in raw_lengths if n <= length) / max(1, len(raw_lengths)), 4
            )
            context_curve.append({
                "max_seq": length,
                "pack": curve_stats,
                "eval": curve_eval,
            })
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
        "pack_fields": bool(args.pack_fields),
        "cross_head": args.cross_head,
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
        "epochs_effective": round(total_steps * args.batch / n_train, 3),
        "steps_override": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "steps": total_steps,
        "fine_tune_tokens": int(tokens_seen),
        "seconds": round(elapsed, 1),
        "final": final,
        "context_curve": context_curve,
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
