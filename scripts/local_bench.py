#!/usr/bin/env python3
"""Measure what this Mac can actually train, before anything gets trained.

This probe exists because every earlier compute estimate in this project was
arithmetic on paper, and paper arithmetic was wrong by 5x in both directions.
It trains a real model of each size for a bounded number of steps on MPS and
reports what happened:

    params, tokens/s, achieved TFLOP/s, peak MPS allocation, process RSS,
    page-outs, and the swap delta across the run

Three rules it does not break:

  * bounded. ~7 configurations x (5 warmup + 20 measured) steps. Minutes, not
    hours. It is a probe, not a training run.
  * memory-capped. Every step is checked against --mem-cap-gib; a config that
    exceeds it is recorded as a failed config, never allowed to push the
    machine into swap. Swapping to SSD is the failure mode this guards.
  * honest about failure. A config that OOMs or produces non-finite loss is
    written to the result with that outcome. It is not retried until it looks
    good and it is not dropped from the table.

Usage:
    .venv/bin/python3 scripts/local_bench.py --mem-cap-gib 6.0

The cap is enforced through torch.mps.set_per_process_memory_fraction, not the
PYTORCH_MPS_HIGH_WATERMARK_RATIO env var: setting that ratio alone makes the
allocator reject an inconsistent low-watermark value before any model is built.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results" / "local_bench"

# (label, dims) -- head_dim is fixed at 32: on this hardware other head dims
# fall off the fused SDPA kernel, measured earlier at 20x slower.
SIZES = {
    "tiny": dict(n_layer=4, d_model=192, n_head=6, d_ff=768),
    "small": dict(n_layer=6, d_model=384, n_head=12, d_ff=1536),
    "mid": dict(n_layer=8, d_model=512, n_head=16, d_ff=2048),
    "large": dict(n_layer=12, d_model=768, n_head=24, d_ff=3072),
}


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask, pad):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, key_padding_mask=pad, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    """Byte-level decoder LM: 256 embeddings + 8 reserved control symbols."""

    def __init__(self, vocab: int = 264, ctx: int = 1024, **dims):
        super().__init__()
        self.tok = nn.Embedding(vocab, dims["d_model"])
        self.pos = nn.Embedding(ctx, dims["d_model"])
        self.blocks = nn.ModuleList(
            Block(dims["d_model"], dims["n_head"], dims["d_ff"])
            for _ in range(dims["n_layer"])
        )
        self.norm_f = nn.RMSNorm(dims["d_model"])
        self.head = nn.Linear(dims["d_model"], vocab, bias=False)
        self.ctx = ctx

    def forward(self, idx):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), 1)
        pad = torch.zeros((b, t), dtype=torch.bool, device=idx.device)
        for blk in self.blocks:
            x = blk(x, mask, pad)
        return self.head(self.norm_f(x))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def swap_used_mb() -> float:
    out = subprocess.run(
        ["sysctl", "vm.swapusage"], capture_output=True, text=True
    ).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else math.nan


def pageouts() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"Pageouts:\s+(\d+)", out)
    return int(m.group(1)) if m else -1


def run_config(label: str, dims: dict, seq: int, batch: int, mode: str,
               steps: int, warmup: int, mem_cap_gib: float) -> dict:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    result = {
        "config": label,
        "dims": dims,
        "seq": seq,
        "batch": batch,
        "mode": mode,
        "device": str(dev),
    }
    torch.manual_seed(0)
    model = TinyLM(ctx=max(seq, 1024), **dims).to(dev)
    result["params"] = n_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dtype = torch.float16 if mode == "fp16" else torch.float32

    swap0, po0 = swap_used_mb(), pageouts()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    def one_step():
        idx = torch.randint(0, 256, (batch, seq + 1), device=dev)
        x, y = idx[:, :-1], idx[:, 1:]
        with torch.autocast(device_type=dev.type, dtype=dtype, enabled=mode != "fp32"):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach().float().item()

    try:
        t0 = time.perf_counter()
        losses = []
        for i in range(warmup + steps):
            loss = one_step()
            if i >= warmup:
                losses.append(loss)
            alloc = torch.mps.driver_allocated_memory() / 2**30 if dev.type == "mps" else 0.0
            if alloc > mem_cap_gib:
                raise MemoryError(
                    f"MPS allocation {alloc:.2f} GiB exceeded cap {mem_cap_gib} GiB"
                )
        wall = time.perf_counter() - t0
        torch.mps.synchronize()
        measured = wall * steps / (warmup + steps)
        toks = batch * seq * steps
        result.update(
            status="ok",
            loss_first=losses[0],
            loss_last=losses[-1],
            finite=bool(all(math.isfinite(v) for v in losses)),
            seconds=round(wall, 3),
            measured_seconds=round(measured, 3),
            tokens_per_s=round(toks / measured, 1),
            sec_per_step=round(measured / steps, 4),
            achieved_tflops=round(6 * n_params(model) * (toks / measured) / 1e12, 3),
            peak_mps_gib=round(torch.mps.driver_allocated_memory() / 2**30, 3),
            rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
            rss_delta_gib=round(
                (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6) - rss0, 3
            ),
            swap_delta_mb=round(swap_used_mb() - swap0, 1),
            pageouts_delta=pageouts() - po0,
        )
    except Exception as exc:  # OOM is a result, not a crash
        result.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}"[:300],
            swap_delta_mb=round(swap_used_mb() - swap0, 1),
            pageouts_delta=pageouts() - po0,
        )
    finally:
        del model, opt
        if dev.type == "mps":
            torch.mps.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--mem-cap-gib", type=float, default=6.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if torch.backends.mps.is_available():
        cap_bytes = int(args.mem_cap_gib * 2**30)
        recommended = torch.mps.recommended_max_memory()
        fraction = min(1.0, cap_bytes / recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
        print(
            f"mps recommended max memory {recommended / 2**30:.2f} GiB ->"
            f" hard cap fraction {fraction:.2f} ({args.mem_cap_gib} GiB)"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = [
        ("tiny", 256, 32, "fp32"),
        ("tiny", 256, 32, "fp16"),
        ("small", 256, 16, "fp32"),
        ("small", 256, 16, "fp16"),
        ("mid", 256, 8, "fp32"),
        ("mid", 256, 8, "fp16"),
        ("small", 1024, 4, "fp32"),
        ("large", 256, 4, "fp32"),
    ]

    print(f"torch {torch.__version__}  mps={torch.backends.mps.is_available()}")
    print(f"swap before: {swap_used_mb():.0f} MB   mem cap: {args.mem_cap_gib} GiB\n")
    hdr = f"{'config':22s} {'params':>10s} {'tok/s':>10s} {'TFLOP/s':>8s} {'peakGiB':>8s} {'swapMB':>8s} {'status':>8s}"
    print(hdr)
    print("-" * len(hdr))

    out = {
        "host": platform.node(),
        "machine": platform.machine(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "mps": torch.backends.mps.is_available(),
        "mps_memory_fraction": min(
            1.0,
            int(args.mem_cap_gib * 2**30)
            / torch.mps.recommended_max_memory()
            if torch.backends.mps.is_available()
            else 1.0,
        ),
        "mem_cap_gib": args.mem_cap_gib,
        "steps": args.steps,
        "warmup": args.warmup,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": [],
    }

    for name, seq, batch, mode in runs:
        r = run_config(
            name, SIZES[name], seq, batch, mode,
            args.steps, args.warmup, args.mem_cap_gib,
        )
        out["runs"].append(r)
        if r["status"] == "ok":
            print(
                f"{name+'/'+mode+'/'+str(seq):22s} {r['params']:>10,d}"
                f" {r['tokens_per_s']:>10,.0f} {r['achieved_tflops']:>8.2f}"
                f" {r['peak_mps_gib']:>8.2f} {r['swap_delta_mb']:>8.0f} {'ok':>8s}"
            )
        else:
            print(f"{name+'/'+mode+'/'+str(seq):22s} {'-':>10s} {'-':>10s} {'-':>8s} {'-':>8s} {'-':>8s} {'FAILED':>8s}")
            print(f"    {r['error']}")

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["swap_after_mb"] = swap_used_mb()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = RESULTS / f"bench_{stamp}{('_' + args.label) if args.label else ''}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {path.relative_to(PROJECT)}")
    print(f"swap after: {swap_used_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
