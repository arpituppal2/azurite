"""GPU serving-latency probe (forward-only; no training, no tuning, no test rows).

Builds tiny-byte dual + cross models on CUDA via the baselined harness,
packs real AGNews TRAIN rows (shapes only; latency is content-independent),
and times eval-mode forward passes at batch 1/16/64, K=4, fp32.
Writes LATENCY_GPU.json. Wall guard 180 s.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts") if (HERE / "scripts").is_dir() else str(HERE))

from transfer_eval import (  # noqa: E402
    TASKS, build_model, candidate_bank, candidate_tensor, load_tokenizer,
    pack_cross, pack_dual, rung, slice_batch, stratified_examples,
)

OUT = Path("/marimo/azurite_ws/results/diligence/v0/LATENCY_GPU.json")
WALL_CAP_S = 180.0
N_WARMUP = 10
N_ITERS = 50


def bench(fn):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": statistics.mean(ts), "p50_ms": statistics.median(ts),
            "min_ms": min(ts), "max_ms": max(ts), "iters": N_ITERS}


def main() -> int:
    t0 = time.perf_counter()
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    cfg = rung("tiny", "byte")
    tok = load_tokenizer("byte")
    spec = TASKS["ag_news"]
    bank = candidate_bank(spec)
    rows = stratified_examples(spec, bank, "train", 64, 0)
    dual_batch, _ = pack_dual(tok, spec, bank, rows, max_seq=cfg.max_seq)
    cross_batch, _ = pack_cross(tok, spec, bank, rows, max_seq=cfg.max_seq,
                                shuffle=False, seed=0)
    dual_model, _ = build_model(cfg, "dual", None, "cuda")
    cross_model, _ = build_model(cfg, "cross", None, "cuda")
    dual_model.eval()
    cross_model.eval()
    cand_ids, cand_mask = candidate_tensor(bank, tok)
    cand_ids, cand_mask = cand_ids.cuda(), cand_mask.cuda()
    dev = "cuda"
    dctx_ids = dual_batch.context_ids.to(dev)
    dctx_mask = dual_batch.context_mask.to(dev)
    ccb = cross_batch.to(dev) if hasattr(cross_batch, "to") else None

    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "config": {"rung": "tiny", "arm": "byte", "K": 4,
                       "max_seq": cfg.max_seq, "d_model": cfg.d_model,
                       "dtype": "fp32", "input": "real-packed-agnews-train-64-rows"},
           "arms": {}}
    with torch.no_grad():
        for name in ("dual", "cross"):
            per = {}
            for b in (1, 16, 64):
                if name == "dual":
                    cv = torch.ones(b, len(bank), dtype=torch.bool, device=dev)
                    fn = (lambda bb=b, ccv=cv: dual_model(dctx_ids[:bb], dctx_mask[:bb],
                                                          cand_ids, cand_mask, ccv))
                else:
                    sb = slice_batch(cross_batch, torch.arange(b)).to(dev)
                    fn = (lambda s=sb: cross_model(s))
                m = bench(fn)
                per[f"batch{b}"] = m
                per[f"per_decision_ms_batch{b}"] = m["mean_ms"] / b
                if time.perf_counter() - t0 > WALL_CAP_S:
                    raise TimeoutError("wall cap")
            res["arms"][name] = per
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk.startswith("per_")}
                      for k, v in res["arms"].items()}, indent=1))
    print("wall_s", res["wall_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
