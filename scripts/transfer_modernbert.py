#!/usr/bin/env python3
"""External pretrained baseline through the same funnel protocol.

Runs a commercially compatible public checkpoint (default
answerdotai/ModernBERT-base, Apache-2.0) through the SAME decision protocol
as scripts/transfer_eval.py --path funnel:

    phase A: dual scorer over the full K (cosine-scale head, same init)
    phase B: fresh cross head over the retrieved top-M subsets (rank order,
             span mean-pool + tanh + linear, same math as AzuriteDecision)

Same rows (stratified split-seed), same batch schedule, same optimizer, same
seeds, same metric: end-to-end test accuracy over ALL rows with funnel
misses as errors. The comparison isolates the encoder (external pretrain on
web-scale data vs Azurite pretrain on 100M corpus tokens), not the protocol.

Usage:
    .venv/bin/python3 scripts/transfer_modernbert.py --seed 0 \\
        --out /tmp/az_dev/modbert_s0.json --evidence-dir /tmp/az_dev/ev_modbert_s0

Writes the same verdict fields as the funnel path: funnel_recall_test,
retrieved_only accuracy, end-to-end accuracy, histories, provenance.
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
import torch.nn.functional as F
from torch import nn

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.decisions import TASKS, candidate_bank, stratified_examples, load_examples, _load  # noqa: E402
from azurite.env import MemoryGuard, select_device  # noqa: E402
from scripts.transfer_eval import batch_schedule  # noqa: E402

MODEL_ID = "answerdotai/ModernBERT-base"
MODEL_LICENSE = "Apache-2.0"

# Evidence pooling over the passage. "mean" is the historical behaviour.
# "max" keeps the strongest activation per dimension (REJECTED: worse real
# accuracy, no wider real-minus-shuffled gap, on both seeds and both suites).
# "attn" is the next hypothesis and the competitor's actual mechanism class: a
# LEARNED weighting over the span (a query vector attending over tokens) rather
# than a fixed reduction. Zero-initialised so training starts at uniform weights
# (mean-like) and only moves if the data pays for it. This is an ARCHITECTURAL
# question and it transfers to a scratch-trained encoder; it is not a benefit
# inherited from pretraining.
# "last" is the competitor's own mechanism class (pooling: last): a single
# learned position that has attended to everything, rather than any reduction
# over tokens. It adds ZERO parameters, so a win here cannot be attributed to
# extra capacity, and it is testable on a scratch-trained encoder.
POOL = "mean"


class DualHead(nn.Module):
    """Mirror of AzuriteDual.embed: mean-pool, tanh(proj), normalize, scale."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.attn_q = nn.Parameter(torch.zeros(d))

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if POOL == "max":
            keep = mask.unsqueeze(-1)
            return hidden.masked_fill(~keep, torch.finfo(hidden.dtype).min).max(1).values
        if POOL == "last":
            t = hidden.shape[1]
            pos = (t - 1) - mask.flip(-1).to(torch.int64).argmax(-1)
            idx = pos.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
            return hidden.gather(1, idx).squeeze(1)
        w = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * w).sum(1) / w.sum(1).clamp(min=1.0)

    def embed(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if POOL == "attn":
            s = (hidden @ self.attn_q) / math.sqrt(hidden.shape[-1])
            a = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
            pooled = (hidden * a.unsqueeze(-1).to(hidden.dtype)).sum(1)
        else:
            pooled = self._pool(hidden, mask)
        return F.normalize(torch.tanh(self.proj(pooled)), dim=-1)


class CrossHead(nn.Module):
    """Mirror of AzuriteDecision tanh head: span mean-pool, tanh(proj), score."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.span_proj = nn.Linear(d, d, bias=False)
        self.score = nn.Linear(d, 1, bias=True)
        self.attn_q = nn.Parameter(torch.zeros(d))

    def forward(self, hidden: torch.Tensor, spans: torch.Tensor,
                cand_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = hidden.shape
        idx = torch.arange(t, device=hidden.device).view(1, 1, t)
        inside = (idx >= spans[..., 0].unsqueeze(-1)) & (idx < spans[..., 1].unsqueeze(-1))
        inside = inside & cand_mask.unsqueeze(-1)
        if POOL == "max":
            neg = torch.finfo(hidden.dtype).min
            pooled = hidden.unsqueeze(1).masked_fill(~inside.unsqueeze(-1), neg).max(2).values
            return self.score(torch.tanh(self.span_proj(pooled))).squeeze(-1).masked_fill(
                ~cand_mask, float("-inf"))
        if POOL == "attn":
            s = (hidden @ self.attn_q) / math.sqrt(d)  # [b,t]
            a = torch.softmax(s.unsqueeze(1).masked_fill(~inside, float("-inf")), -1)
            pooled = torch.einsum("bkt,btd->bkd", a.to(hidden.dtype), hidden)
            return self.score(torch.tanh(self.span_proj(pooled))).squeeze(-1).masked_fill(
                ~cand_mask, float("-inf"))
        if POOL == "last":
            # learned readout position inside each span: the final span token,
            # which under full attention has already seen the whole passage
            pos = (t - 1) - inside.flip(-1).to(torch.int64).argmax(-1)  # [b,k]
            pooled = hidden.gather(1, pos.unsqueeze(-1).expand(b, pos.shape[1], d))
            return self.score(torch.tanh(self.span_proj(pooled))).squeeze(-1).masked_fill(
                ~cand_mask, float("-inf"))
        w = inside.to(hidden.dtype)
        pooled = torch.einsum("bkt,btd->bkd", w, hidden) / w.sum(-1, keepdim=True).clamp(min=1.0)
        return self.score(torch.tanh(self.span_proj(pooled))).squeeze(-1).masked_fill(
            ~cand_mask, float("-inf"))


def pack_dual_hf(tok, contexts: list[str], max_len: int):
    ids, masks = [], []
    for c in contexts:
        t = tok(c, add_special_tokens=False)["input_ids"][: max(1, max_len - 2)]
        row = [tok.cls_token_id, *t, tok.sep_token_id]
        ids.append(row)
        masks.append([1] * len(row))
    w = max(len(r) for r in ids)
    pad = tok.pad_token_id
    return (torch.tensor([r + [pad] * (w - len(r)) for r in ids]),
            torch.tensor([m + [0] * (w - len(m)) for m in masks], dtype=torch.bool))


def pack_cross_subset_hf(tok, contexts: list[str], cand_texts: list[str],
                         keep: list[list[int]], golds: list[int], max_len: int):
    """Joint-encode [CLS] ctx [SEP] cand...; drop rows whose gold is unretrieved."""
    cand_toks = [tok(c, add_special_tokens=False)["input_ids"] for c in cand_texts]
    rows, spans, gold, packed_idx, miss = [], [], [], [], 0
    for pos, (ctx, kept) in enumerate(zip(contexts, keep)):
        if golds[pos] not in kept:
            miss += 1
            continue
        ctx_t = tok(ctx, add_special_tokens=False)["input_ids"]
        fixed = 3 + sum(len(cand_toks[k]) + 1 for k in kept)
        budget = max(8, max_len - fixed)
        ctx_t = ctx_t[:budget]
        row = [tok.cls_token_id, *ctx_t, tok.sep_token_id]
        sp = []
        for k in kept:
            sp.append((len(row), len(row) + len(cand_toks[k])))
            row += cand_toks[k] + [tok.sep_token_id]
        if len(row) > max_len:
            miss += 1
            continue
        rows.append(row)
        spans.append(sp)
        gold.append(kept.index(golds[pos]))
        packed_idx.append(pos)
    if not rows:
        raise SystemExit("no packable rows")
    w = max(len(r) for r in rows)
    pad = tok.pad_token_id
    ids = torch.tensor([r + [pad] * (w - len(r)) for r in rows])
    mask = torch.tensor([[1] * len(r) + [0] * (w - len(r)) for r in rows], dtype=torch.bool)
    return ids, mask, torch.tensor(spans), torch.tensor(gold), packed_idx, miss


@torch.no_grad()
def dual_scores(enc, head, ids, mask, cand_emb, device, chunk=32):
    out = []
    for s in range(0, ids.shape[0], chunk):
        h = enc(ids[s:s + chunk].to(device),
                attention_mask=mask[s:s + chunk].to(device)).last_hidden_state
        ctx = head.embed(h, mask[s:s + chunk].to(device))
        out.append((head.log_scale.exp() * (ctx @ cand_emb.T)).float().cpu())
    return torch.cat(out)


def train_phase(opt, steps, batch, n, seed, step_fn, eval_fn, guard, tag):
    hist = []
    t0 = time.perf_counter()
    for step, idx in enumerate(batch_schedule(n, batch, steps, seed)):
        loss = step_fn(idx)
        if not torch.isfinite(loss):
            print(f"FATAL non-finite loss at {tag} step {step}", flush=True)
            return hist, time.perf_counter() - t0, True
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 1.0)
        opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            ev = eval_fn()
            hist.append({"step": step, **ev})
            print(f"  {tag}step {step:>4}/{steps} loss {float(loss):.4f} "
                  f"test acc {ev['accuracy']:.4f}", flush=True)
        guard.check()
    return hist, time.perf_counter() - t0, False


def metrics_from_logits(logits, gold):
    probs = torch.softmax(logits, -1)
    pred = logits.argmax(-1)
    acc = (pred == gold).float().mean().item()
    nll = F.cross_entropy(logits, gold).item()
    conf, _ = probs.max(-1)
    correct = (pred == gold).float()
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        m = (conf > lo) & (conf <= hi) if b else (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    return {"n": len(gold), "accuracy": round(acc, 4), "nll": round(nll, 4),
            "ece": round(ece, 4), "mean_confidence": round(conf.mean().item(), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="banking77")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--funnel-m", type=int, default=8)
    ap.add_argument("--dual-steps", type=int, default=378)
    ap.add_argument("--cross-steps", type=int, default=378)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--examples", type=int, default=2000)
    ap.add_argument("--test-examples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--mem-cap-gib", type=float, default=6.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--evidence-dir", default="")
    ap.add_argument("--pool", choices=["mean", "max", "attn", "last"], default="mean",
                    help="evidence pooling over passage tokens (mean = historical)")
    ap.add_argument("--swap-margin", type=float, default=0.0,
                    help="hinge margin forcing score(gold|right passage) over "
                         "score(gold|foreign passage); 0 = off (historical)")
    ap.add_argument("--swap-lambda", type=float, default=1.0,
                    help="weight of the passage-swap margin term")
    ap.add_argument("--save-encoder", action="store_true",
                    help="also checkpoint the fine-tuned encoder so the arm can be "
                         "re-scored or served without retraining (~600 MB)")
    args = ap.parse_args()
    global POOL
    POOL = args.pool

    from transformers import AutoTokenizer, ModernBertModel

    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.cls_token_id is not None and tok.sep_token_id is not None, "need CLS/SEP"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.sep_token

    spec = TASKS[args.task]
    bank = candidate_bank(spec)
    train_rows = stratified_examples(spec, bank, "train", args.examples, args.split_seed)
    test_rows = stratified_examples(spec, bank, "test", args.test_examples, args.split_seed + 1)
    train_ctx = [r.context for r in train_rows]
    test_ctx = [r.context for r in test_rows]
    train_gold = torch.tensor([r.gold for r in train_rows])
    test_gold = torch.tensor([r.gold for r in test_rows])

    device = args.device
    enc = ModernBertModel.from_pretrained(args.model).to(device)
    d = enc.config.hidden_size
    torch.manual_seed(args.seed)
    dual = DualHead(d).to(device)

    guard = MemoryGuard(cap_gib=args.mem_cap_gib)
    train_ids, train_mask = pack_dual_hf(tok, train_ctx, args.max_len)
    test_ids, test_mask = pack_dual_hf(tok, test_ctx, args.max_len)
    cand_ids, cand_mask = pack_dual_hf(tok, bank.labels, args.max_len)

    @torch.no_grad()
    def cand_emb():
        h = enc(cand_ids.to(device),
                attention_mask=cand_mask.to(device)).last_hidden_state
        return DualHead._pool(h, cand_mask.to(device))

    with torch.no_grad():
        cand_h = enc(cand_ids.to(device),
                     attention_mask=cand_mask.to(device)).last_hidden_state
    cand_base = DualHead._pool(cand_h, cand_mask.to(device))

    def dual_embed(ids, mask):
        h = enc(ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        return dual.embed(h, mask.to(device))

    @torch.no_grad()
    def dual_eval():
        with torch.no_grad():
            c = F.normalize(torch.tanh(dual.proj(cand_base)), dim=-1)
            lg = dual_scores(enc, dual, test_ids, test_mask, c, device)
        return metrics_from_logits(lg, test_gold)

    dopt = torch.optim.AdamW(list(enc.parameters()) + list(dual.parameters()),
                             lr=args.lr, weight_decay=0.01)
    enc.train()

    def dual_step(idx):
        c = F.normalize(torch.tanh(dual.proj(cand_base)), dim=-1)
        ctx = dual_embed(train_ids[idx], train_mask[idx])
        logits = dual.log_scale.exp() * (ctx @ c.T)
        return F.cross_entropy(logits, train_gold[idx].to(device))

    hist_a, t_a, fatal = train_phase(dopt, args.dual_steps, args.batch, len(train_rows),
                                     args.seed, dual_step, dual_eval, guard, "A/")
    if fatal:
        return 1
    with torch.no_grad():
        c = F.normalize(torch.tanh(dual.proj(cand_base)), dim=-1)
        lg = dual_scores(enc, dual, test_ids, test_mask, c, device)
        tr_lg = dual_scores(enc, dual, train_ids, train_mask, c, device)
    keep_test = lg.topk(args.funnel_m, -1).indices.tolist()
    keep_train = tr_lg.topk(args.funnel_m, -1).indices.tolist()
    rec_tr = sum(g in k for g, k in zip(train_gold.tolist(), keep_train)) / len(keep_train)
    rec_te = sum(g in k for g, k in zip(test_gold.tolist(), keep_test)) / len(keep_test)
    print(f"  funnel recall@{args.funnel_m}: train {rec_tr:.4f} test {rec_te:.4f}", flush=True)

    # Phase B: fresh cross head over retrieved subsets.
    tr_ids, tr_mask, tr_spans, tr_gold, tr_packed, tr_miss = pack_cross_subset_hf(
        tok, train_ctx, bank.labels, keep_train, train_gold.tolist(), args.max_len)
    te_ids, te_mask, te_spans, te_gold, te_packed, te_miss = pack_cross_subset_hf(
        tok, test_ctx, bank.labels, keep_test, test_gold.tolist(), args.max_len)
    # Foreign-passage packs, built only when the margin arm is on so that every
    # historical cell stays byte-identical. Rotating by one guarantees each packed
    # row gets another row's passage (n > 1 always), and re-packing the SAME keep
    # lists keeps candidate sets and spans identical by construction.
    f_ids = None
    if args.swap_margin > 0:
        f_ctx = [train_ctx[tr_packed[(j + 1) % len(tr_packed)]] for j in range(len(tr_packed))]
        f_ids, f_mask, f_spans, _fg, f_packed, f_miss = pack_cross_subset_hf(
            tok, f_ctx, bank.labels, [keep_train[p] for p in tr_packed],
            [train_gold[p].item() for p in tr_packed], args.max_len)
        assert f_packed == list(range(len(tr_packed))), "foreign pack lost rows"
        assert f_spans.shape == tr_spans.shape, "foreign pack changed candidate spans"
        print(f"  swap-margin {args.swap_margin} lambda {args.swap_lambda}: "
              f"{len(tr_packed)} foreign packs aligned", flush=True)
    torch.manual_seed(args.seed + 1)
    cross = CrossHead(d).to(device)
    # Phase B fine-tunes the same encoder further (mirrors the Azurite funnel,
    # which shares the pretrained start and trains encoder+head jointly).
    copt = torch.optim.AdamW(list(enc.parameters()) + list(cross.parameters()),
                             lr=args.lr, weight_decay=0.01)

    def cross_eval():
        outs = []
        with torch.no_grad():
            for s in range(0, te_ids.shape[0], 16):
                h = enc(te_ids[s:s + 16].to(device),
                        attention_mask=te_mask[s:s + 16].to(device)).last_hidden_state
                outs.append(cross(h, te_spans[s:s + 16].to(device),
                                  torch.ones(te_spans[s:s + 16].shape[:2],
                                             dtype=torch.bool, device=device)).float().cpu())
        m = metrics_from_logits(torch.cat(outs), te_gold)
        m["retrieved_only"] = True
        return m

    def cross_step(idx):
        gold = tr_gold[idx].to(device)
        ones = torch.ones(tr_spans[idx].shape[:2], dtype=torch.bool, device=device)
        h = enc(tr_ids[idx].to(device),
                attention_mask=tr_mask[idx].to(device)).last_hidden_state
        logits = cross(h, tr_spans[idx].to(device), ones)
        loss = F.cross_entropy(logits, gold)
        if f_ids is not None:
            # Passage-swap margin. Same candidates, same spans, a different row's
            # passage: the only way to satisfy this term is to read the passage,
            # because the candidate set is identical between the two passes.
            hf = enc(f_ids[idx].to(device),
                     attention_mask=f_mask[idx].to(device)).last_hidden_state
            logits_f = cross(hf, tr_spans[idx].to(device), ones)
            sel = gold.view(-1, 1)
            right = logits.gather(1, sel).squeeze(1)
            foreign = logits_f.gather(1, sel).squeeze(1)
            loss = loss + args.swap_lambda * F.relu(args.swap_margin - (right - foreign)).mean()
        return loss

    hist_b, t_b, fatal = train_phase(copt, args.cross_steps, args.batch, tr_ids.shape[0],
                                     args.seed + 1, cross_step, cross_eval, guard, "B/")
    if fatal:
        return 1
    for h in hist_b:
        h["retrieved_only"] = True
    fin = cross_eval()
    gold_full = test_gold.tolist()
    with torch.no_grad():
        outs = []
        for s in range(0, te_ids.shape[0], 16):
            h = enc(te_ids[s:s + 16].to(device),
                    attention_mask=te_mask[s:s + 16].to(device)).last_hidden_state
            outs.append(cross(h, te_spans[s:s + 16].to(device),
                              torch.ones(te_spans[s:s + 16].shape[:2],
                                         dtype=torch.bool, device=device)).argmax(-1).cpu())
        preds = torch.cat(outs).tolist()
    mapped = [None] * len(test_rows)
    for p, row in zip(preds, te_packed):
        mapped[row] = keep_test[row][p]
    e2e = round(sum(1 for p, g in zip(mapped, gold_full) if p == g) / len(gold_full), 4)
    print(f"  funnel end-to-end test acc {e2e:.4f} (retrieved-only "
          f"{fin['accuracy']:.4f}, misses {te_miss})  {t_a + t_b:.0f}s", flush=True)

    # --- context-use control ---------------------------------------------
    # Score the SAME trained model again with the test contexts permuted, so
    # each question is paired with a different passage. If accuracy is unchanged
    # the model never read the context and was answering from priors; if it
    # collapses, the context is being used. This separates "weak" from
    # "ignores the input", which no amount of retuning can fix if it is the latter.
    #
    # DEFECT FIXED (v17): the first implementation permuted contexts AND gold by
    # the same order, so every row kept its true (passage, answer) pair and the
    # control only reordered rows -- it could not detect an ignored passage, and
    # all real-minus-shuffled gaps before this fix (v12, v17 attn/last) measured
    # batching noise, not evidence use. Gold must stay in place while the
    # passages move; `control_gold_fixed` records which definition produced the
    # number so old JSONs cannot be mistaken for this one.
    shuffled_e2e = None
    control_gold_fixed = False
    try:
        import random as _random
        order = list(range(len(test_ctx)))
        _random.Random(args.seed).shuffle(order)
        perm_ctx = [test_ctx[i] for i in order]
        perm_gold = test_gold.tolist()  # rows stay, passages move
        control_gold_fixed = True
        p_ids, p_mask = pack_dual_hf(tok, perm_ctx, args.max_len)
        with torch.no_grad():
            pc = F.normalize(torch.tanh(dual.proj(cand_base)), dim=-1)
            p_lg = dual_scores(enc, dual, p_ids, p_mask, pc, device)
        p_keep = p_lg.topk(args.funnel_m, -1).indices.tolist()
        s_ids, s_mask, s_spans, s_gold, s_packed, s_miss = pack_cross_subset_hf(
            tok, perm_ctx, bank.labels, p_keep, perm_gold, args.max_len)
        outs = []
        with torch.no_grad():
            for st in range(0, s_ids.shape[0], 16):
                hh = enc(s_ids[st:st + 16].to(device),
                         attention_mask=s_mask[st:st + 16].to(device)).last_hidden_state
                outs.append(cross(hh, s_spans[st:st + 16].to(device),
                                  torch.ones(s_spans[st:st + 16].shape[:2],
                                             dtype=torch.bool, device=device)).argmax(-1).cpu())
            sp = torch.cat(outs).tolist()
        smapped = [None] * len(perm_ctx)
        for pp, row in zip(sp, s_packed):
            smapped[row] = p_keep[row][pp]
        shuffled_e2e = round(sum(1 for a, b in zip(smapped, perm_gold) if a == b)
                             / len(perm_gold), 4)
        print(f"  context-control (shuffled passages): e2e {shuffled_e2e:.4f} "
              f"vs real {e2e:.4f}", flush=True)
    except Exception as exc:
        print(f"  context-control skipped: {type(exc).__name__}: {exc}", flush=True)

    try:
        rev = ModernBertModel.from_pretrained(args.model).config.to_dict().get("_commit_hash", "")
    except Exception:
        rev = ""
    result = {
        "task": args.task, "path": "funnel", "funnel_m": args.funnel_m,
        "model": args.model, "model_license": MODEL_LICENSE, "model_revision": rev,
        "hidden": d, "k": len(bank), "train_examples": len(train_rows),
        "test_examples": len(test_rows), "dual_steps": args.dual_steps,
        "cross_steps": args.cross_steps, "batch": args.batch, "lr": args.lr,
        "seed": args.seed, "split_seed": args.split_seed,
        "seconds_retriever": round(t_a, 1), "seconds_reranker": round(t_b, 1),
        "seconds": round(t_a + t_b, 1),
        "pool": args.pool,
        "swap_margin": args.swap_margin, "swap_lambda": args.swap_lambda,
        "funnel_recall_train": round(rec_tr, 4), "funnel_recall_test": round(rec_te, 4),
        "funnel_miss_test": te_miss,
        "retrieved_only": {k: fin[k] for k in ("n", "accuracy", "nll", "ece")},
        "final": {"n": len(test_rows), "accuracy": e2e, "nll": None, "ece": None,
                  "note": "end-to-end over ALL test rows, misses as errors"},
        "context_control_shuffled_e2e": shuffled_e2e,
        "context_control_gold_fixed": control_gold_fixed,
        "history_retriever": hist_a, "history_reranker": hist_b,
        "dataset_fingerprints": {s: _load(spec, s)._fingerprint for s in ("train", "test")},
        "runtime": {"device": device, "torch": torch.__version__,
                    "python": sys.version, "platform": platform.platform()},
    }
    out = Path(args.out) if args.out else PROJECT / "results" / "transfer" / \
        f"{args.task}_modernbert_funnel_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.evidence_dir:
        ev = Path(args.evidence_dir)
        ev.mkdir(parents=True, exist_ok=True)
        keep = {"keep_train": keep_train, "keep_test": keep_test,
                "packed_indices": te_packed, "predictions_bank": mapped,
                "gold": gold_full}
        (ev / "funnel_keep.json").write_text(json.dumps(keep) + "\n")
        # NOTE: the heads alone do NOT reproduce a result -- phase A/B also
        # fine-tune the encoder. A checkpoint without `encoder` is not
        # self-sufficient (found in v17: re-scoring a saved heads.pt with the
        # stock encoder gave 0.42 where the run reported 0.504). --save-encoder
        # writes it, at ~600 MB for ModernBERT-base, so mirroring is impractical;
        # it is for on-host re-scoring and deployment, not for transfer.
        ckpt = {"dual": dual.state_dict(), "cross": cross.state_dict(),
                "pool": args.pool, "max_len": args.max_len, "task": args.task}
        if args.save_encoder:
            ckpt["encoder"] = enc.state_dict()
        torch.save(ckpt, ev / "heads.pt")
        result["evidence"] = {"keep_sha256": hashlib.sha256(
            (ev / "funnel_keep.json").read_bytes()).hexdigest()}
        result["device"] = device
        result["stratified"] = True
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
