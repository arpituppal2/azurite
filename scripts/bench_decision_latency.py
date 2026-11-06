#!/usr/bin/env python3
"""Per-decision serving latency for the retrieve-then-rerank decision system.

Measures the two phases the product actually runs per request:

    retrieve  one context forward pass against K cached candidate embeddings
    rerank    one joint forward over [CLS] ctx [SEP] cand_1..cand_M [SEP]

Latency is a property of the architecture and shapes, not of the weight values,
so this runs with correctly-shaped heads and stock encoder weights. That is
stated in the output rather than hidden: this measures compute cost, not
accuracy.

Batch size 1, warmup + repeated trials, reported as p50/p95/p99. Also sweeps
context length to show how cost scales with the input the enterprise cares
about, and reports the GPU figure next to the recorded Jev per-decision claim
(283 ms) so the comparison is explicit rather than implied.

Usage:
    python scripts/bench_decision_latency.py --device cuda --reps 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.transfer_modernbert import pack_cross_subset_hf, pack_dual_hf  # noqa: E402

MODEL_ID = "answerdotai/ModernBERT-base"
JEV_RECORDED_MS = 283.0


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


@torch.no_grad()
def timeit(fn, reps: int, warmup: int = 12, device: str = "cuda") -> dict:
    """Time `fn`, synchronising only when the work is actually on that device.

    The first version tested `torch.cuda.is_available()` instead of the device
    being benchmarked, so a CPU run on a CUDA host issued a CUDA synchronise
    into a GPU this process was not using — and OOM'd against a busy device.
    """
    sync = device == "cuda" and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    xs: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": round(statistics.fmean(xs), 3),
        "p50_ms": round(pct(xs, 50), 3),
        "p95_ms": round(pct(xs, 95), 3),
        "p99_ms": round(pct(xs, 99), 3),
        "reps": reps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--candidate-text-len", type=int, default=4,
                    help="mean tokens per candidate label (measured on banking77 is ~2-4)")
    ap.add_argument("--context-tokens", type=int, default=64)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer, ModernBertModel

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = ModernBertModel.from_pretrained(args.model).to(args.device).eval()
    if args.dtype == "bf16" and args.device == "cuda":
        enc = enc.to(torch.bfloat16)
    d = enc.config.hidden_size

    k_list = [2, 3, 6, 77, 151]
    m_list = [2, 8]
    ctx = "banking question about a transaction " * 3
    cand_names = [f"intent_{i}" for i in range(max(k_list))]

    results = {
        "model": args.model, "device": args.device, "dtype": args.dtype,
        "batch": 1,        "weights": "stock encoder + randomly initialised heads "
                              "(latency depends on shapes, not values)",
        "candidate_embeddings": "cached once per checkpoint, excluded from the "
                                "per-request timing (matches serving design)",
        "params": sum(p.numel() for p in enc.parameters()),
        "jev_recorded_ms_per_decision": JEV_RECORDED_MS,
        "sweep": [], "context_curve": [],
    }

    for k in k_list:
        cand_ids, cand_mask = pack_dual_hf(tok, cand_names[:k], 64)
        cand_ids, cand_mask = cand_ids.to(args.device), cand_mask.to(args.device)

        # Candidate embeddings are computed ONCE PER CHECKPOINT, not per request.
        # Timing them inside the request loop (the first version of this script
        # did) charges the request for work the architecture never repeats, and
        # it made CPU cost look linear in K when it is not.
        with torch.no_grad():
            h = enc(cand_ids, attention_mask=cand_mask).last_hidden_state
            cw = cand_mask.unsqueeze(-1).to(h.dtype)
            cand_cached = ((h * cw).sum(1) / cw.sum(1).clamp(min=1.0)).detach()
        ctx_ids, ctx_mask = pack_dual_hf(tok, [ctx], 512)
        ctx_ids, ctx_mask = ctx_ids.to(args.device), ctx_mask.to(args.device)

        def retrieve() -> None:
            ch = enc(ctx_ids, attention_mask=ctx_mask).last_hidden_state
            cwm = ctx_mask.unsqueeze(-1).to(ch.dtype)
            q = (ch * cwm).sum(1) / cwm.sum(1).clamp(min=1.0)
            _ = q @ cand_cached.T

        r = timeit(retrieve, args.reps, device=args.device)
        entry = {"k": k, "retrieve": r}
        for m in m_list:
            if m > k:
                continue
            keep = [list(range(m))]
            ids, mask, spans, gold, packed, miss = pack_cross_subset_hf(
                tok, [ctx], cand_names[:k], keep, [0], 512)

            def rerank() -> None:
                h = enc(ids.to(args.device),
                        attention_mask=mask.to(args.device)).last_hidden_state
                _ = h[:, :m, :].mean(1)  # stand-in for span pooling + score

            rr = timeit(rerank, args.reps, device=args.device)
            entry[f"rerank_m{m}"] = rr
            entry[f"total_m{m}_p50_ms"] = round(r["p50_ms"] + rr["p50_ms"], 3)
            entry[f"total_m{m}_p99_ms"] = round(r["p99_ms"] + rr["p99_ms"], 3)
        results["sweep"].append(entry)
        print(f"K={k:3d}  retrieve p50 {r['p50_ms']:.2f} p99 {r['p99_ms']:.2f} ms"
              + "".join(f"  | rerank M{m} p50 {entry[f'rerank_m{m}']['p50_ms']:.2f}"
                        for m in m_list if m <= k), flush=True)

    # Context-length curve at fixed K=77, M=8: how cost scales with input size.
    cand_ids, cand_mask = pack_dual_hf(tok, cand_names[:77], 64)
    keep = [list(range(8))]
    for n_ctx in (32, 128, 256, 512):
        long_ctx = ("account transfer question " * (n_ctx // 4 + 1)).strip()
        ids, mask, spans, gold, packed, miss = pack_cross_subset_hf(
            tok, [long_ctx], cand_names[:77], keep, [0], 512)
        seq = int(ids.shape[1])

        def run() -> None:
            h = enc(ids.to(args.device),
                    attention_mask=mask.to(args.device)).last_hidden_state
            _ = h[:, :8, :].mean(1)

        rr = timeit(run, max(50, args.reps // 2), device=args.device)
        results["context_curve"].append({"target_ctx_tokens": n_ctx,
                                         "packed_seq": seq, **rr})
        print(f"ctx~{n_ctx:4d} packed {seq:5d}  p50 {rr['p50_ms']:.2f}  p99 {rr['p99_ms']:.2f} ms",
              flush=True)

    out = Path(args.out) if args.out else PROJECT / "results" / "diligence" / "v10_latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
