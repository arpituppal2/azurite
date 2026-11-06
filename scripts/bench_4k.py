#!/usr/bin/env python3
"""GPU-only 4k training-step probe. Refuses CPU/MPS by design."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import EncoderConfig  # noqa: E402
from azurite.model import Azurite  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--out", default="results/parity/bench_4k.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; do not run this training probe on CPU/MPS")
    device = torch.device("cuda")
    cfg = EncoderConfig(vocab_size=32_004, max_seq=args.seq,
                        d_model=1024, n_layer=24, n_head=16, d_ff=4096)
    model = Azurite(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    ids = torch.randint(1, cfg.vocab_size, (args.batch, args.seq), device=device)
    gen = torch.Generator(device=device).manual_seed(0)

    def step() -> float:
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _ = model.mlm_loss(ids, generator=gen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.detach())

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = [step() for _ in range(args.steps)]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    rec = {
        "device": torch.cuda.get_device_name(0),
        "seq": args.seq, "batch": args.batch, "steps": args.steps,
        "dtype": args.dtype, "seconds": round(seconds, 3),
        "tokens": args.batch * args.seq * args.steps,
        "tokens_per_s": round(args.batch * args.seq * args.steps / seconds, 1),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "mean_loss": sum(losses) / len(losses),
        "finite": all(x == x for x in losses),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0 if rec["finite"] and rec["tokens_per_s"] >= 20_000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
