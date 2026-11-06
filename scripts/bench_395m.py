#!/usr/bin/env python3
"""Measure 395M-class throughput and the batch-size ceiling on this GPU.

The 395M rung does not exist in `azurite.config.RUNG_DIMS`. This script builds
it at the ModernBERT-large width (24 layers, d_model 1,024, 16 heads, 4,096
inner) so the answer is about the card, not about a config table.

Faithfulness to the ladder: the 113M run trained in **fp32** — no autocast, no
GradScaler — with AdamW, a clipped backward pass, and 30% MLM. The default mode
reproduces that step exactly. `tf32` and `bf16` are measured as headroom, not as
the baseline, because switching precision is a code change with its own
numerical claim.

The step uses synthetic ids already on the device. The real loader adds a
memmap read and an H2D copy of ~32 KB per step at seq 1,024, which is noise at
these rates; the numbers here are therefore an upper bound on the real loop.

    python3 scripts/bench_395m.py --out results/scale_state/BENCH_395M.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from azurite.config import EncoderConfig  # noqa: E402
from azurite.model import Azurite  # noqa: E402

RUNG_395M = {"d_model": 1024, "n_layer": 24, "n_head": 16, "d_ff": 4096}
VOCAB = 260
SEQ = 1024
MODES = ("fp32", "tf32", "bf16")


def build(device: torch.device) -> Azurite:
    cfg = EncoderConfig(vocab_size=VOCAB, max_seq=SEQ, **RUNG_395M)
    return Azurite(cfg).to(device)


def set_precision(mode: str) -> None:
    if mode == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def one_step(
    model: Azurite,
    opt: torch.optim.Optimizer,
    ids: torch.Tensor,
    mode: str,
    gen: torch.Generator,
) -> float:
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(mode == "bf16")):
        loss, _ = model.mlm_loss(ids, generator=gen)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach())


def measure(
    model: Azurite,
    opt: torch.optim.Optimizer,
    batch: int,
    mode: str,
    device: torch.device,
    *, warmup: int, steps: int,
) -> dict:
    ids = torch.randint(1, VOCAB, (batch, SEQ), device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(warmup):
        one_step(model, opt, ids, mode, gen)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    start = time.perf_counter()
    for _ in range(steps):
        losses.append(one_step(model, opt, ids, mode, gen))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    tokens = batch * SEQ * steps
    return {
        "batch": batch,
        "seq": SEQ,
        "steps": steps,
        "seconds": round(seconds, 3),
        "s_per_step": round(seconds / steps, 4),
        "tokens_per_s": round(tokens / seconds, 1),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "mean_loss": round(sum(losses) / len(losses), 4),
        "finite": all(x == x for x in losses),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scale_state/BENCH_395M.json")
    ap.add_argument("--batches", default="8,16,32")
    ap.add_argument("--modes", default="fp32,tf32,bf16")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--bf16-batches", default="8,16,32,64")
    args = ap.parse_args()

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out.with_suffix(".jsonl")
    jsonl.write_text("", encoding="utf-8")

    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": props.name,
        "vram_gib": round(props.total_memory / 2**30, 2),
        "torch": torch.__version__,
        "config": {"vocab_size": VOCAB, "max_seq": SEQ, **RUNG_395M},
        "reference": {
            "note": "113M large rung measured at 28,134 tok/s end-to-end, fp32",
            "tokens_per_s": 28134.3,
            "params": 113465856,
        },
        "modes": {},
    }

    for mode in [m for m in args.modes.split(",") if m]:
        if mode not in MODES:
            raise SystemExit(f"unknown mode {mode!r}")
        set_precision(mode)
        model = build(device)
        params = model.num_parameters()
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95),
                                weight_decay=0.1)
        batches = (args.bf16_batches if mode == "bf16" else args.batches)
        records = []
        for batch in [int(b) for b in batches.split(",") if b]:
            try:
                rec = measure(model, opt, batch, mode, device,
                              warmup=args.warmup, steps=args.steps)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                rec = {"batch": batch, "seq": SEQ, "oom": True,
                       "error": str(exc).splitlines()[0][:200]}
                torch.cuda.empty_cache()
            rec["mode"] = mode
            records.append(rec)
            with jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if rec.get("oom"):
                break
        best = max((r for r in records if "tokens_per_s" in r),
                   key=lambda r: r["tokens_per_s"], default=None)
        summary["modes"][mode] = {
            "params": params,
            "records": records,
            "best": best,
            "hours_for_395m_tokens": (
                round(395_000_000 / best["tokens_per_s"] / 3600, 2) if best else None
            ),
            "hours_for_3_95b_tokens": (
                round(3_950_000_000 / best["tokens_per_s"] / 3600, 2) if best else None
            ),
        }
        del opt, model
        torch.cuda.empty_cache()

    set_precision("fp32")
    out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps({m: {"params": v["params"], "best": v["best"],
                          "h_3.95B": v["hours_for_3_95b_tokens"]}
                      for m, v in summary["modes"].items()}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
