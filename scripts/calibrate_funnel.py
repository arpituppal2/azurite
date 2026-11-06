#!/usr/bin/env python3
"""Validation-only temperature scaling for a funnel cell (development evidence).

Trains ONE funnel cell on development data only (default: banking77, split 0),
then fits a single temperature T on half of the dev rows and reports ECE / NLL /
accuracy before and after scaling on the fit half AND the held-out half.

Why this exists: the pplx-decider investigation exposed that we have never
measured calibration on our classification cells, while the competitor publishes
ECE 0.021. A confidence-gated product cannot be argued for without this number.
Temperature scaling is one scalar, fit on dev, with no architecture change and
no test-set peeking — the cheapest honest fix available.

What it does NOT do: touch confirmatory splits (99/123). Everything here is
split 0 and is labelled development evidence.

Usage (Molab):
    python scripts/calibrate_funnel.py --task banking77 --split-seed 0 \\
        --train-examples 4000 --dev-examples 1000 --device cuda \\
        --out results/transfer/envelope_v13/CALIB_banking77_dev.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import transfer_modernbert as tm  # noqa: E402
from azurite.decisions import candidate_bank, stratified_examples  # noqa: E402


def reliability_table(logits: torch.Tensor, gold: torch.Tensor, bins: int = 10):
    probs = torch.softmax(logits, -1)
    conf, pred = probs.max(-1)
    correct = (pred == gold).float()
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi) if b else (conf <= hi)
        rows.append({
            "bin": [round(lo, 2), round(hi, 2)],
            "n": int(m.sum().item()),
            "mean_conf": round(conf[m].mean().item(), 4) if m.any() else None,
            "acc": round(correct[m].mean().item(), 4) if m.any() else None,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="banking77")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--train-examples", type=int, default=4000)
    ap.add_argument("--dev-examples", type=int, default=1000)
    ap.add_argument("--funnel-m", type=int, default=8)
    ap.add_argument("--dual-steps", type=int, default=756)
    ap.add_argument("--cross-steps", type=int, default=756)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-cap-gib", type=float, default=6.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer, ModernBertModel

    tok = AutoTokenizer.from_pretrained(tm.MODEL_ID)
    assert tok.cls_token_id is not None and tok.sep_token_id is not None
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.sep_token

    spec = tm.TASKS[args.task]
    bank = candidate_bank(spec)
    # Funnel shortlist cannot exceed the candidate count (e.g. emotion K=6 < m=8).
    m_eff = min(args.funnel_m, len(bank))
    train_rows = stratified_examples(spec, bank, "train", args.train_examples, args.split_seed)
    dev_rows = stratified_examples(spec, bank, "test", args.dev_examples, args.split_seed + 1)
    dev_gold = torch.tensor([r.gold for r in dev_rows])
    print(f"train {len(train_rows)}  dev {len(dev_rows)} (SPLIT 0 ONLY — development evidence)",
          flush=True)

    device = args.device
    enc = ModernBertModel.from_pretrained(tm.MODEL_ID).to(device)
    d = enc.config.hidden_size
    torch.manual_seed(args.seed)
    dual = tm.DualHead(d).to(device)
    guard = tm.MemoryGuard(cap_gib=args.mem_cap_gib)

    train_ids, train_mask = tm.pack_dual_hf(tok, [r.context for r in train_rows], args.max_len)
    dev_ids, dev_mask = tm.pack_dual_hf(tok, [r.context for r in dev_rows], args.max_len)
    cand_ids, cand_mask = tm.pack_dual_hf(tok, bank.labels, args.max_len)
    train_gold = torch.tensor([r.gold for r in train_rows])
    with torch.no_grad():
        cand_h = enc(cand_ids.to(device),
                     attention_mask=cand_mask.to(device)).last_hidden_state
        cand_base = tm.DualHead._pool(cand_h, cand_mask.to(device))

    def dual_embed(ids, mask):
        h = enc(ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        return dual.embed(h, mask.to(device))

    @torch.no_grad()
    def dual_logits(ids, mask):
        c = F.normalize(torch.tanh(dual.proj(cand_base)), dim=-1)
        return tm.dual_scores(enc, dual, ids, mask, c, device)

    dopt = torch.optim.AdamW(list(enc.parameters()) + list(dual.parameters()),
                             lr=args.lr, weight_decay=0.01)
    enc.train()

    def dual_step(idx):
        c = F.normalize(torch.tanh(dual.proj(cand_base)), dim=-1)
        ctx = dual_embed(train_ids[idx], train_mask[idx])
        return F.cross_entropy(dual.log_scale.exp() * (ctx @ c.T),
                               train_gold[idx].to(device))

    def dual_eval():
        return tm.metrics_from_logits(dual_logits(dev_ids, dev_mask), dev_gold)

    t0 = time.perf_counter()
    hist_a, t_a, fatal = tm.train_phase(dopt, args.dual_steps, args.batch, len(train_rows),
                                        args.seed, dual_step, dual_eval, guard, "A/")
    if fatal:
        return 1
    with torch.no_grad():
        lg = dual_logits(dev_ids, dev_mask)
        tr_lg = dual_logits(train_ids, train_mask)
    keep_dev = lg.topk(m_eff, -1).indices.tolist()
    keep_train = tr_lg.topk(m_eff, -1).indices.tolist()
    rec = sum(g in k for g, k in zip(dev_gold.tolist(), keep_dev)) / len(keep_dev)
    print(f"  funnel recall@{m_eff}: dev {rec:.4f}", flush=True)

    tr_ids, tr_mask, tr_spans, tr_g, tr_packed, _ = tm.pack_cross_subset_hf(
        tok, [r.context for r in train_rows], bank.labels, keep_train,
        train_gold.tolist(), args.max_len)
    de_ids, de_mask, de_spans, de_g, de_packed, de_miss = tm.pack_cross_subset_hf(
        tok, [r.context for r in dev_rows], bank.labels, keep_dev,
        dev_gold.tolist(), args.max_len)
    torch.manual_seed(args.seed + 1)
    cross = tm.CrossHead(d).to(device)
    copt = torch.optim.AdamW(list(enc.parameters()) + list(cross.parameters()),
                             lr=args.lr, weight_decay=0.01)
    enc.train()

    def cross_logits(ids, mask, spans):
        outs = []
        with torch.no_grad():
            for s in range(0, ids.shape[0], 16):
                h = enc(ids[s:s + 16].to(device),
                        attention_mask=mask[s:s + 16].to(device)).last_hidden_state
                outs.append(cross(h, spans[s:s + 16].to(device),
                                  torch.ones(spans[s:s + 16].shape[:2],
                                             dtype=torch.bool, device=device)).float().cpu())
        return torch.cat(outs)

    def cross_eval():
        m = tm.metrics_from_logits(cross_logits(de_ids, de_mask, de_spans),
                                   torch.tensor(de_g))
        m["retrieved_only"] = True
        return m

    def cross_step(idx):
        h = enc(tr_ids[idx].to(device),
                attention_mask=tr_mask[idx].to(device)).last_hidden_state
        return F.cross_entropy(cross(h, tr_spans[idx].to(device),
                                     torch.ones(tr_spans[idx].shape[:2],
                                                dtype=torch.bool, device=device)),
                               torch.tensor(tr_g[idx], device=device))

    hist_b, t_b, fatal = tm.train_phase(copt, args.cross_steps, args.batch, tr_ids.shape[0],
                                        args.seed + 1, cross_step, cross_eval, guard, "B/")
    if fatal:
        return 1

    # Logits on ALL dev rows in packed order, mapped back to dev-row order.
    dev_logits_packed = cross_logits(de_ids, de_mask, de_spans)
    K = len(bank)
    full = torch.full((len(dev_rows), K), float("-inf"))
    for logit_vec, row in zip(dev_logits_packed, de_packed):
        kept = keep_dev[row]
        full[row, torch.tensor(kept)] = logit_vec[: len(kept)]
    # Misses (gold not retrieved) keep -inf everywhere except gold -> handled as
    # errors: give them a uniform logit so they count as maximally uncertain errors.
    miss = (full == float("-inf")).all(-1)
    full[miss] = 0.0
    print(f"  dev misses (gold outside top-{m_eff}): {int(miss.sum())}/{len(dev_rows)}",
          flush=True)

    # Split dev by index parity: even = temperature fit, odd = held-out report.
    idx = torch.arange(len(dev_rows))
    fit_m = (idx % 2 == 0)
    rep_m = ~fit_m
    g = dev_gold

    # Fit T on the fit half only (NLL) by exhaustive grid search.
    # Deliberately NOT gradient-based: an earlier Adam attempt diverged to
    # T=nan (T -> 0 sends logits/T to inf). A grid over [0.05, 10] cannot
    # diverge and is fully disclosed and reproducible.
    fl, fg = full[fit_m], g[fit_m]
    best_T, best_loss = 1.0, float("inf")
    with torch.no_grad():
        for Tc in torch.logspace(-1.3, 1.0, 120).tolist():
            loss = F.cross_entropy(fl / Tc, fg).item()
            if loss == loss and loss < best_loss:  # NaN-guard
                best_T, best_loss = Tc, loss
    T = best_T

    def report(mask, tag):
        base = tm.metrics_from_logits(full[mask], g[mask])
        scaled = tm.metrics_from_logits(full[mask] / T, g[mask])
        return {"n": int(mask.sum()), tag + "_raw": base, tag + "_scaled_T": scaled}

    out = {"task": args.task, "split": "DEVELOPMENT ONLY (split 0)",
           "confirmatory_splits_untouched": [99, 123],
           "method": "single temperature T fit by NLL on even-index dev rows; "
                     "reported on fit half and held-out odd half",
           "temperature": round(T, 4),
           "seconds_train": round(time.perf_counter() - t0, 1),
           "funnel_recall_dev": round(rec, 4),
           "dev_misses": int(miss.sum()),
           "fit_half": report(fit_m, "fit"),
           "heldout_half": report(rep_m, "heldout"),
           "reliability_heldout_raw": reliability_table(full[rep_m], g[rep_m]),
           "reliability_heldout_scaled": reliability_table(full[rep_m] / T, g[rep_m]),
           "dataset_fingerprints": {
               "train": tm._load(spec, "train")._fingerprint,
               "dev": tm._load(spec, "test")._fingerprint},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    p = Path(args.out) if args.out else (PROJECT / "results" / "transfer" / "envelope_v13" /
                                         f"CALIB_{args.task}_dev_s{args.seed}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    fh = out["heldout_half"]
    print(f"T={T:.4f}  heldout raw: acc {fh['heldout_raw']['accuracy']} "
          f"ece {fh['heldout_raw']['ece']} nll {fh['heldout_raw']['nll']}  -> scaled: acc "
          f"{fh['heldout_scaled_T']['accuracy']} ece {fh['heldout_scaled_T']['ece']} nll "
          f"{fh['heldout_scaled_T']['nll']}", flush=True)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
