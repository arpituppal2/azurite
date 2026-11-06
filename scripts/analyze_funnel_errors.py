#!/usr/bin/env python3
"""Prediction-error breakdown for a completed funnel cell (forward-only).

Reloads a funnel evidence dir (retriever.pt + final.pt + provenance.json)
and recomputes, without any training: gold-rank distribution among the
retrieved top-M, miss rate, confidence on correct vs error, retrieved-only
accuracy conditioned on gold rank, worst-confused label pairs, and context
length vs error. Answers "where does the 0.03 gap live" with errors, not
with singular values.

Usage:
    .venv/bin/python3 scripts/analyze_funnel_errors.py \\
        --result /tmp/az_dev/smoke_s0.json --evidence-dir /tmp/az_dev/ev_smoke \\
        --ckpt results/pretrain/l1_small_byte/ckpt.pt --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import rung  # noqa: E402
from azurite.decisions import TASKS, candidate_bank, stratified_examples  # noqa: E402
from azurite.model import Azurite, AzuriteDecision, AzuriteDual  # noqa: E402
from azurite.tokenizers import load as load_tokenizer  # noqa: E402
from scripts.transfer_eval import (  # noqa: E402
    build_model, candidate_tensor, dual_part, pack_cross_subset, pack_dual,
    retrieve_topm,
)

torch.manual_seed(0)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    res = json.loads(Path(args.result).read_text())
    prov = json.loads(Path(args.evidence_dir, "provenance.json").read_text())
    task, seed = res["task"], res["seed"]
    m = res["funnel_m"]
    cross_head = res.get("cross_head", "tanh")

    tok = load_tokenizer(res.get("arm", "byte"))
    cfg = rung(res["rung"], res.get("arm", "byte"))
    spec = TASKS[task]
    bank = candidate_bank(spec)
    train_rows = stratified_examples(spec, bank, "train", res["train_examples"], res["split_seed"])
    test_rows = stratified_examples(spec, bank, "test", res["test_examples"], res["split_seed"] + 1)

    train_full, _ = pack_dual(tok, spec, bank, train_rows, max_seq=cfg.max_seq)
    test_full, _ = pack_dual(tok, spec, bank, test_rows, max_seq=cfg.max_seq)
    device = args.device

    retr = torch.load(Path(args.evidence_dir) / "retriever.pt", map_location=device)
    dual_model = AzuriteDual(Azurite(cfg)).to(device)
    dual_model.load_state_dict(retr["model"], strict=True)
    dual_model.eval()
    cand_ids, cand_mask = candidate_tensor(bank, tok)
    cand_args = (cand_ids.to(device), cand_mask.to(device),
                 torch.ones(len(bank), dtype=torch.bool, device=device))
    keep_test, recall = retrieve_topm(dual_model, test_full, cand_args, m, device)

    fin = torch.load(Path(args.evidence_dir) / "final.pt", map_location=device)
    cross_model = AzuriteDecision(Azurite(cfg), head=cross_head).to(device)
    cross_model.load_state_dict(fin["model"], strict=True)
    cross_model.eval()

    test_sub, test_stats = pack_cross_subset(
        tok, spec, bank, test_rows, keep_test, max_seq=cfg.max_seq)
    logits_all, idx_all = [], []
    for s in range(0, test_sub.gold.shape[0], 32):
        idx = torch.arange(s, min(test_sub.gold.shape[0], s + 32))
        from scripts.transfer_eval import slice_batch
        logits_all.append(cross_model(slice_batch(test_sub, idx).to(device)).float().cpu())
        idx_all.append(idx)
    logits = torch.cat(logits_all)
    probs = torch.softmax(logits, -1)
    conf, pred = probs.max(-1)

    gold_rank, correct, err_conf, ok_conf, ctx_len_ok, ctx_len_err = [], [], [], [], [], []
    pair_counter = Counter()
    for i, row in enumerate(test_stats["packed_indices"]):
        g = test_rows[row].gold
        rank = keep_test[row].index(g) + 1
        gold_rank.append(rank)
        slot_pred = pred[i].item()
        ok = keep_test[row][slot_pred] == g
        correct.append(ok)
        (ok_conf if ok else err_conf).append(conf[i].item())
        clen = len(tok.encode(test_rows[row].context))
        (ctx_len_ok if ok else ctx_len_err).append(clen)
        if not ok:
            pair_counter[(bank.labels[g], bank.labels[keep_test[row][slot_pred]])] += 1

    n_miss = res["funnel_miss_test"]
    n_ret = len(correct)
    rank_hist = Counter(gold_rank)
    out = {
        "task": task, "seed": seed, "m": m, "cross_head": cross_head,
        "n_test": len(test_rows), "misses": n_miss,
        "miss_rate": round(n_miss / len(test_rows), 4),
        "retrieved_n": n_ret,
        "retrieved_acc": round(sum(correct) / max(1, n_ret), 4),
        "e2e_acc": res["final"]["accuracy"],
        "gold_rank_hist": {str(k): rank_hist.get(k, 0) for k in range(1, m + 1)},
        "acc_when_gold_rank1": round(
            sum(c for c, r in zip(correct, gold_rank) if r == 1) / max(1, sum(1 for r in gold_rank if r == 1)), 4),
        "acc_when_gold_rank_gt1": round(
            sum(c for c, r in zip(correct, gold_rank) if r > 1) / max(1, sum(1 for r in gold_rank if r > 1)), 4),
        "mean_conf_correct": round(sum(ok_conf) / max(1, len(ok_conf)), 4),
        "mean_conf_error": round(sum(err_conf) / max(1, len(err_conf)), 4),
        "mean_ctx_len_correct": round(sum(ctx_len_ok) / max(1, len(ctx_len_ok)), 1),
        "mean_ctx_len_error": round(sum(ctx_len_err) / max(1, len(ctx_len_err)), 1),
        "top_confused_pairs": [[a, b, n] for (a, b), n in pair_counter.most_common(10)],
    }
    print(json.dumps(out, indent=1))
    Path(args.evidence_dir, "error_breakdown.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
