#!/usr/bin/env python3
"""Recompute the context-use control correctly from saved heads.

Rationale: before v17 the harness permuted contexts AND golds by the same order,
so each row kept its true (passage, answer) pair. The "shuffled" arm was therefore
the same task reordered, and every real-minus-shuffled gap reported up to v17 was
batching noise. This script loads an arm's saved `heads.pt` and reports, on the
same rows:

  * real           -- original pairing (sanity check against the stored JSON)
  * shuffled_fixed -- passages permuted, golds in place: the intended control
  * shuffled_old   -- both permuted (the defective definition, for comparison)
  * prior          -- pick the more frequent of the retrieved candidates

Nothing is retrained, so this is a few seconds per arm and cannot alter a result.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.decisions import TASKS, candidate_bank, stratified_examples  # noqa: E402
import scripts.transfer_modernbert as tm  # noqa: E402
from scripts.transfer_modernbert import (  # noqa: E402
    CrossHead, DualHead, dual_scores, pack_cross_subset_hf, pack_dual_hf,
)


def score(enc, dual, cross, tok, bank, ctx, gold, m, max_len, device, cand_base):
    ids, mask = pack_dual_hf(tok, ctx, max_len)
    with torch.no_grad():
        c = F.normalize(torch.tanh(dual.proj(cand_base)), dim=-1)
        lg = dual_scores(enc, dual, ids, mask, c, device)
    keep = lg.topk(m, -1).indices.tolist()
    recall = sum(g in k for g, k in zip(gold, keep)) / len(gold)
    s_ids, s_mask, s_spans, s_gold, s_packed, miss = pack_cross_subset_hf(
        tok, ctx, bank.labels, keep, list(gold), max_len)
    outs = []
    with torch.no_grad():
        for st in range(0, s_ids.shape[0], 16):
            h = enc(s_ids[st:st + 16].to(device),
                    attention_mask=s_mask[st:st + 16].to(device)).last_hidden_state
            outs.append(cross(h, s_spans[st:st + 16].to(device),
                              torch.ones(s_spans[st:st + 16].shape[:2],
                                         dtype=torch.bool, device=device)).argmax(-1).cpu())
    preds = torch.cat(outs).tolist()
    mapped = [None] * len(ctx)
    for p, row in zip(preds, s_packed):
        mapped[row] = keep[row][p]
    e2e = sum(1 for p, g in zip(mapped, gold) if p == g) / len(gold)
    cond = (sum(1 for p, g in zip(mapped, gold) if p == g) /
            max(1, sum(1 for g, k in zip(gold, keep) if g in k)))
    return {"e2e": round(e2e, 4), "cond": round(cond, 4), "recall": round(recall, 4),
            "miss": int(miss), "keep": keep, "mapped": mapped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--pool", default="mean", choices=["mean", "max", "attn", "last"])
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--test-examples", type=int, default=500)
    ap.add_argument("--funnel-m", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--examples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tm.POOL = args.pool
    from transformers import AutoTokenizer, ModernBertModel
    tok = AutoTokenizer.from_pretrained(tm.MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    spec = TASKS[args.task]
    bank = candidate_bank(spec)
    tr_rows = stratified_examples(spec, bank, "train", args.examples, args.split_seed)
    te_rows = stratified_examples(spec, bank, "test", args.test_examples, args.split_seed + 1)
    ctx = [r.context for r in te_rows]
    gold = [r.gold for r in te_rows]

    enc = ModernBertModel.from_pretrained(tm.MODEL_ID).to(args.device).eval()
    d = enc.config.hidden_size
    dual, cross = DualHead(d).to(args.device), CrossHead(d).to(args.device)
    st = torch.load(args.heads, map_location=args.device)
    dual.load_state_dict(st["dual"])
    cross.load_state_dict(st["cross"])
    dual.eval(); cross.eval()

    c_ids, c_mask = pack_dual_hf(tok, bank.labels, args.max_len)
    with torch.no_grad():
        cand_h = enc(c_ids.to(args.device),
                     attention_mask=c_mask.to(args.device)).last_hidden_state
    cand_base = DualHead._pool(cand_h, c_mask.to(args.device))

    order = list(range(len(ctx)))
    random.Random(args.seed).shuffle(order)
    perm_ctx = [ctx[i] for i in order]

    real = score(enc, dual, cross, tok, bank, ctx, gold, args.funnel_m,
                 args.max_len, args.device, cand_base)
    fixed = score(enc, dual, cross, tok, bank, perm_ctx, gold, args.funnel_m,
                  args.max_len, args.device, cand_base)
    old = score(enc, dual, cross, tok, bank, perm_ctx, [gold[i] for i in order],
                args.funnel_m, args.max_len, args.device, cand_base)

    cnt = collections.Counter(r.gold for r in tr_rows)
    pri = [cnt[i] for i in range(len(bank))]
    hits = sum(int(max(c, key=lambda x: (pri[x], -x)) == g)
               for c, g in zip(real["keep"], gold) if g in c)
    prior = round(hits / len(gold), 4)

    res = {
        "task": args.task, "heads": str(args.heads), "pool": args.pool,
        "split_seed": args.split_seed, "n": len(gold), "funnel_m": args.funnel_m,
        "real": {k: real[k] for k in ("e2e", "cond", "recall", "miss")},
        "shuffled_fixed": {k: fixed[k] for k in ("e2e", "cond", "recall")},
        "shuffled_old": {k: old[k] for k in ("e2e", "cond", "recall")},
        "prior_baseline_cond": prior,
        "gap_fixed": round(real["e2e"] - fixed["e2e"], 4),
        "gap_fixed_cond": round(real["cond"] - fixed["cond"], 4),
        "gap_old": round(real["e2e"] - old["e2e"], 4),
        "cond_minus_prior": round(real["cond"] - prior, 4),
    }
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
