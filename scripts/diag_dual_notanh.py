"""Tanh-ablation probe: THE single changed component is torch.tanh removed from
AzuriteDual.embed (identity in its place). Everything else replicates
diag_dual_random.py: seed 0, split-seed 0, same 2000 train rows, same first-80
schedule (asserted prefix of the 378 schedule), AdamW lr 5e-4 wd 0.01, fp32.

Added measurements only (eval-mode, no_grad: consume no RNG, perturb nothing):
- pre/post-proj activation statistics on the probe (would-be-saturation frac)
- fresh 200-row validation probe (split_seed+2, asserted disjoint from train)

NO test rows are loaded or evaluated in this run.
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
sys.path.insert(0, str(HERE / "scripts") if (HERE / "scripts").is_dir() else str(HERE))

import azurite.model as _M  # noqa: E402
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

# THE intervention: tanh -> identity. candidate_embeddings() calls self.embed(),
# so it follows automatically. Init path untouched (same seed => same weights).
_ORIG_EMBED = _M.AzuriteDual.embed


def _embed_notanh(self, ids, mask):
    pooled = _M.AzuriteDual._pool(self.encoder(ids, mask), mask)
    return F.normalize(self.proj(pooled), dim=-1)


_M.AzuriteDual.embed = _embed_notanh

TASK = "ag_news"
ARM = "byte"
RUNGP = "tiny"
SEED = 0
SPLIT_SEED = 0
N_TRAIN = 2000
N_VAL = 200
BATCH = 16
LR = 5e-4
MAX_STEPS = 80
WALL_CAP_S = 180.0
PROBE_N = 64
VAL_EVERY = 5
DEVICE = "cuda"
OUT = Path("/marimo/azurite_ws/results/transfer/diag_dual_notanh_s0")


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


def act_stats(pre: torch.Tensor) -> dict:
    f = pre.float()
    return {
        "std_mean": float(f.std(0).mean().item()),
        "mean_abs": float(f.abs().mean().item()),
        "sat_frac_gt2": float((f.abs() > 2).float().mean().item()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tok = load_tokenizer(ARM)
    cfg = rung(RUNGP, ARM)
    spec = TASKS[TASK]
    bank = candidate_bank(spec)
    train_rows = stratified_examples(spec, bank, "train", N_TRAIN, SPLIT_SEED)
    train_idx = {e.index for e in train_rows}
    val_seed = SPLIT_SEED + 2
    val_rows = [e for e in stratified_examples(spec, bank, "train", N_VAL, val_seed)
                if e.index not in train_idx]
    assert len(val_rows) >= 150, f"val too small after overlap filter: {len(val_rows)}"
    val_class_counts = {}
    for e in val_rows:
        val_class_counts[e.gold] = val_class_counts.get(e.gold, 0) + 1
    train_batch, _ = pack_dual(tok, spec, bank, train_rows, max_seq=cfg.max_seq)
    val_batch, _ = pack_dual(tok, spec, bank, val_rows, max_seq=cfg.max_seq)

    torch.manual_seed(SEED)
    model, _ = build_model(cfg, "dual", None, DEVICE)
    init_norm = math.sqrt(
        sum(float(p.detach().pow(2).sum().item()) for p in model.parameters())
    )
    cand_ids, cand_mask = candidate_tensor(bank, tok)
    cand_args = (
        cand_ids.to(DEVICE),
        cand_mask.to(DEVICE),
        torch.ones(len(bank), dtype=torch.bool, device=DEVICE),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)

    sched80 = [idx.tolist() for idx in batch_schedule(N_TRAIN, BATCH, MAX_STEPS, SEED)]
    sched378 = [idx.tolist() for idx in batch_schedule(N_TRAIN, BATCH, 378, SEED)]
    assert sched80 == sched378[:MAX_STEPS], "schedule prefix mismatch"

    probe_idx = torch.arange(min(PROBE_N, train_batch.gold.shape[0]))
    probe_ctx_ids = train_batch.context_ids[probe_idx].to(DEVICE)
    probe_ctx_mask = train_batch.context_mask[probe_idx].to(DEVICE)
    probe_gold = train_batch.gold[probe_idx]
    val_ctx_ids = val_batch.context_ids.to(DEVICE)
    val_ctx_mask = val_batch.context_mask.to(DEVICE)
    val_gold = val_batch.gold.to(DEVICE)
    val_gold_cpu = val_batch.gold

    config = {
        "task": TASK, "arm": ARM, "rung": RUNGP, "path": "dual", "init": "random",
        "seed": SEED, "split_seed": SPLIT_SEED, "train": N_TRAIN,        "val_draw": N_VAL,
        "val_split_seed": val_seed, "val_train_overlap": 0,
        "val_actual": len(val_rows), "val_class_counts": val_class_counts,
        "batch": BATCH, "lr": LR, "optimizer": "AdamW wd=0.01 betas=[0.9,0.999]",
        "max_steps": MAX_STEPS, "wall_cap_s": WALL_CAP_S, "probe_n": PROBE_N,
        "probe_source": "first 64 TRAIN rows (no test rows loaded in this run)",
        "device": DEVICE, "dtype": "fp32", "schedule_prefix_of_378": True,
        "intervention": "AzuriteDual.embed: torch.tanh removed (identity); "
                        "normalize + log_scale unchanged; monkeypatched in-process, "
                        "shared code untouched",
        "init_param_norm": init_norm,
        "loss_not_expected_to_match_diag": True,
        "torch": torch.__version__,
    }
    (OUT / "CONFIG.json").write_text(json.dumps(config, indent=1) + "\n")

    enc, head = split_params(model)
    reason = "steps_exhausted"
    n_done = 0
    with open(OUT / "TRACE.jsonl", "w") as fh, open(OUT / "VAL.jsonl", "w") as vh:
        for step, idx in enumerate(batch_schedule(N_TRAIN, BATCH, MAX_STEPS, SEED)):
            if time.perf_counter() - t0 > WALL_CAP_S:
                reason = "wall_cap"
                break
            model.train()
            opt.zero_grad(set_to_none=True)
            part = dual_part(train_batch, idx, DEVICE)
            logits = model(part.context_ids, part.context_mask, *cand_args)
            loss = F.cross_entropy(logits, part.gold)
            loss_f = float(loss.detach())
            if not math.isfinite(loss_f):
                reason = f"nonfinite_loss@{step}"
                break
            batch_acc = float((logits.argmax(-1) == part.gold).float().mean().item())
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
                # Activation evidence: pre-proj pooled -> proj output (what tanh
                # would have crushed) -> final normalized embedding.
                hid = model.encoder(probe_ctx_ids, probe_ctx_mask)
                pooled = _M.AzuriteDual._pool(hid, probe_ctx_mask)
                pre = model.proj(pooled)
                ctx_emb = F.normalize(pre, dim=-1)
                hid_c = model.encoder(cand_args[0], cand_args[1])
                pooled_c = _M.AzuriteDual._pool(hid_c, cand_args[1])
                pre_c = model.proj(pooled_c)
                if step % VAL_EVERY == 0 or step == MAX_STEPS - 1:
                    vlogits = model(val_ctx_ids, val_ctx_mask, *cand_args)
                    vloss = float(F.cross_entropy(vlogits, val_gold).item())
                    vacc = float((vlogits.argmax(-1).cpu() == val_gold_cpu).float().mean().item())
                    vh.write(json.dumps({"step": step, "val_loss": vloss,
                                         "val_acc": vacc}) + "\n")
                    vh.flush()
                else:
                    vloss, vacc = None, None
            rec = {
                "step": step,
                "wall_s": round(time.perf_counter() - t0, 3),
                "loss": loss_f,
                "batch_acc": batch_acc,
                "lr": opt.param_groups[0]["lr"],
                "grad_enc": gs_enc,
                "grad_head": gs_head,
                "grad_total_norm": total_norm,
                "clip_factor": clip_factor,
                "update_enc_norm": upd_enc,
                "update_head_norm": upd_head,
                "log_scale": log_scale,
                "head_bias": None,
                "probe": {
                    "logit_std_per_class": plogits.float().std(0).cpu().tolist(),
                    "prob_std_per_class": pprob.std(0).cpu().tolist(),
                    "pred_counts": {str(k): v for k, v in sorted(Counter(pred.tolist()).items())},
                    "probe_acc": float((pred == probe_gold).float().mean().item()),
                    "ctx_emb_std_mean": float(ctx_emb.std(0).mean().item()),
                    "pooled_std_mean": float(pooled.std(0).mean().item()),
                    "pre_proj_stats": act_stats(pre),
                    "cand_pre_proj_stats": act_stats(pre_c),
                    "val_loss": vloss,
                    "val_acc": vacc,
                },
                "mem_alloc_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done = step + 1
            if step % 10 == 0:
                print(f"step {step} loss {loss_f:.4f} bacc {batch_acc:.3f} "
                      f"pre_std {rec['probe']['pre_proj_stats']['std_mean']:.3e} "
                      f"sat {rec['probe']['pre_proj_stats']['sat_frac_gt2']:.3f} "
                      f"emb_std {rec['probe']['ctx_emb_std_mean']:.3e} "
                      f"val {vloss}/{vacc}", flush=True)
    print(f"DONE steps={n_done} reason={reason} wall={time.perf_counter()-t0:.1f}s", flush=True)
    (OUT / "RESULT.txt").write_text(f"steps={n_done} reason={reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
