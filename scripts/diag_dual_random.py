"""Instrumented dual-random probe: one failing seed, unchanged recipe, 80 steps max.

Replicates scripts/transfer_eval.py exactly (task ag_news, byte arm, tiny rung,
path dual, init random, seed 0, split-seed 0, 2000/500 stratified rows, batch 16,
lr 5e-4 AdamW wd 0.01, fp32, no freeze) for the first 80 optimizer steps, and
records per-step gradient/update/probe evidence that the standard harness does
not store. No checkpoint is written. Stops at 80 steps, 180 s wall, or any
non-finite value, whichever comes first.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))

from transfer_eval import (  # noqa: E402
    TASKS,
    batch_schedule,
    build_model,
    candidate_bank,
    candidate_tensor,
    dual_part,
    load_tokenizer,
    pack_dual,
    rung,
    stratified_examples,
)

TASK = "ag_news"
ARM = "byte"
RUNGP = "tiny"
SEED = 0
SPLIT_SEED = 0
N_TRAIN = 2000
N_TEST = 500
BATCH = 16
LR = 5e-4
MAX_STEPS = 80
WALL_CAP_S = 180.0
PROBE_N = 64
DEVICE = "cuda"
OUT = Path("/marimo/azurite_ws/results/transfer/diag_dual_random_s0")


def split_params(model):
    enc, head = [], []
    for name, p in model.named_parameters():
        (enc if name.startswith("encoder.") else head).append((name, p))
    return enc, head


def grad_stats(named):
    tensors = len(named)
    none = sum(1 for _, p in named if p.grad is None)
    zero = sum(1 for _, p in named if p.grad is not None and not p.grad.any().item())
    nonzero = tensors - none - zero
    elem = sum(p.numel() for _, p in named)
    elem_zero = sum(
        int((p.grad == 0).sum().item()) for _, p in named if p.grad is not None
    )
    norm = math.sqrt(
        sum(float(p.grad.float().pow(2).sum().item()) for _, p in named if p.grad is not None)
    )
    return {
        "tensors": tensors,
        "grad_none": none,
        "grad_zero": zero,
        "grad_nonzero": nonzero,
        "elem": elem,
        "elem_zero": elem_zero,
        "grad_norm": norm,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tok = load_tokenizer(ARM)
    cfg = rung(RUNGP, ARM)
    spec = TASKS[TASK]
    bank = candidate_bank(spec)
    train_rows = stratified_examples(spec, bank, "train", N_TRAIN, SPLIT_SEED)
    test_rows = stratified_examples(spec, bank, "test", N_TEST, SPLIT_SEED + 1)
    train_batch, _ = pack_dual(tok, spec, bank, train_rows, max_seq=cfg.max_seq)
    test_batch, _ = pack_dual(tok, spec, bank, test_rows, max_seq=cfg.max_seq)

    torch.manual_seed(SEED)
    model, _ = build_model(cfg, "dual", None, DEVICE)
    cand_ids, cand_mask = candidate_tensor(bank, tok)
    cand_args = (
        cand_ids.to(DEVICE),
        cand_mask.to(DEVICE),
        torch.ones(len(bank), dtype=torch.bool, device=DEVICE),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)

    # Prefix-stability self-check: the first 80 batches of a 378-step schedule
    # must equal the 80-step schedule, or this is not the same trajectory.
    sched80 = [idx.tolist() for idx in batch_schedule(N_TRAIN, BATCH, MAX_STEPS, SEED)]
    sched378 = [idx.tolist() for idx in batch_schedule(N_TRAIN, BATCH, 378, SEED)]
    assert sched80 == sched378[:MAX_STEPS], "schedule prefix mismatch; not the same trajectory"

    probe_idx = torch.arange(min(PROBE_N, test_batch.gold.shape[0]))
    probe_ctx_ids = test_batch.context_ids[probe_idx].to(DEVICE)
    probe_ctx_mask = test_batch.context_mask[probe_idx].to(DEVICE)
    probe_gold = test_batch.gold[probe_idx]

    config = {
        "task": TASK, "arm": ARM, "rung": RUNGP, "path": "dual", "init": "random",
        "seed": SEED, "split_seed": SPLIT_SEED, "train": N_TRAIN, "test": N_TEST,
        "batch": BATCH, "lr": LR, "optimizer": "AdamW wd=0.01 betas=[0.9,0.999]",
        "max_steps": MAX_STEPS, "wall_cap_s": WALL_CAP_S, "probe_n": PROBE_N,
        "device": DEVICE, "dtype": "fp32", "schedule_prefix_of_378": True,
        "torch": torch.__version__,
    }
    (OUT / "CONFIG.json").write_text(json.dumps(config, indent=1) + "\n")

    enc, head = split_params(model)
    head_names = [n for n, _ in head]
    trace_path = OUT / "TRACE.jsonl"
    reason = "steps_exhausted"
    n_done = 0
    with open(trace_path, "w") as fh:
        for step, idx in enumerate(batch_schedule(N_TRAIN, BATCH, MAX_STEPS, SEED)):
            wall = time.perf_counter() - t0
            if wall > WALL_CAP_S:
                reason = "wall_cap"
                break
            model.train()
            opt.zero_grad(set_to_none=True)
            part = dual_part(train_batch, idx, DEVICE)
            logits = model(part.context_ids, part.context_mask, *cand_args)
            loss = F.cross_entropy(logits, part.gold)
            loss_f = float(loss)
            if not math.isfinite(loss_f):
                reason = f"nonfinite_loss@{step}"
                break
            loss.backward()

            gs_enc = grad_stats(enc)
            gs_head = grad_stats(head)
            total_norm = math.sqrt(gs_enc["grad_norm"] ** 2 + gs_head["grad_norm"] ** 2)
            if not math.isfinite(total_norm):
                reason = f"nonfinite_grad@{step}"
                break
            clip_factor = min(1.0, 1.0 / total_norm) if total_norm > 0 else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            before_enc = [p.detach().clone() for _, p in enc]
            before_head = [p.detach().clone() for _, p in head]
            opt.step()
            upd_enc = math.sqrt(
                sum(float((p.detach() - b).pow(2).sum().item())
                    for (_, p), b in zip(enc, before_enc))
            )
            upd_head = math.sqrt(
                sum(float((p.detach() - b).pow(2).sum().item())
                    for (_, p), b in zip(head, before_head))
            )
            if not (math.isfinite(upd_enc) and math.isfinite(upd_head)):
                reason = f"nonfinite_update@{step}"
                break

            log_scale = next(
                (float(p.detach().item()) for n, p in head if n == "log_scale"), None
            )
            model.eval()
            with torch.no_grad():
                plogits = model(probe_ctx_ids, probe_ctx_mask, *cand_args)
                if not bool(torch.isfinite(plogits).all().item()):
                    reason = f"nonfinite_probe@{step}"
                    break
                pprob = torch.softmax(plogits.float(), dim=-1)
                pred = plogits.argmax(-1).cpu()
                ctx_emb = model.embed(probe_ctx_ids, probe_ctx_mask)
                hidden = model.encoder(probe_ctx_ids, probe_ctx_mask)
                pooled = model._pool(hidden, probe_ctx_mask)
                cand_emb = model.candidate_embeddings(cand_ids.to(DEVICE),
                                                      cand_mask.to(DEVICE))
            rec = {
                "step": step,
                "wall_s": round(time.perf_counter() - t0, 3),
                "loss": loss_f,
                "lr": opt.param_groups[0]["lr"],
                "grad_enc": gs_enc,
                "grad_head": gs_head,
                "grad_total_norm": total_norm,
                "clip_factor": clip_factor,
                "update_enc_norm": upd_enc,
                "update_head_norm": upd_head,
                "log_scale": log_scale,
                "head_bias": None,  # AzuriteDual head: proj(bias=False) + log_scale; no bias exists
                "head_params": head_names,
                "probe": {
                    "logit_std_per_class": plogits.float().std(0).cpu().tolist(),
                    "prob_std_per_class": pprob.std(0).cpu().tolist(),
                    "pred_counts": {str(k): v for k, v in sorted(Counter(pred.tolist()).items())},
                    "probe_acc": float((pred == probe_gold).float().mean().item()),
                    "ctx_emb_std_mean": float(ctx_emb.std(0).mean().item()),
                    "pooled_std_mean": float(pooled.std(0).mean().item()),
                    "cand_emb_std_mean": float(cand_emb.std(0).mean().item()),
                },
                "mem_alloc_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done = step + 1
            if step % 10 == 0:
                print(f"step {step} loss {loss_f:.4f} genc {gs_enc['grad_norm']:.3e} "
                      f"ghead {gs_head['grad_norm']:.3e} uenc {upd_enc:.3e} "
                      f"uhead {upd_head:.3e} probe_std {rec['probe']['logit_std_per_class']}",
                      flush=True)
    print(f"DONE steps={n_done} reason={reason} wall={time.perf_counter()-t0:.1f}s", flush=True)
    (OUT / "RESULT.txt").write_text(f"steps={n_done} reason={reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
