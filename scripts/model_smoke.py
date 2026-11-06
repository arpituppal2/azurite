#!/usr/bin/env python3
"""Measure the real azurite model, and check the ladder is runnable.

Why this exists rather than reusing `local_bench.py`'s numbers: that bench
(2026-10-27T030406) measured a **different architecture** — `nn.MultiheadAttention`
with a two-matrix GELU MLP — and `nn.MultiheadAttention` does not go through the
fused SDPA path, which its own source notes is worth up to 20x on non-32 head
dims. The numbers in `results/local_bench/` are therefore a floor measured on a
slower attention implementation, not a projection for this one. Every
wall-clock figure in PLAN.md §6.1 is derived from them, so they get replaced by
a measurement of the code that would actually be trained.

Same discipline as `local_bench.py`, deliberately, including reusing its swap
and page-out readers so the two are comparable:

  * bounded — a few steps per rung, minutes not hours
  * memory-capped — torch.mps.set_per_process_memory_fraction, and a rung that
    exceeds the cap is recorded as failed rather than tolerated
  * honest about failure — a non-finite loss or an OOM is a result
  * zero-swap is the operating condition, not a nicety

Usage:
    .venv/bin/python3 scripts/model_smoke.py --mem-cap-gib 6.0
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from azurite.config import MAX_SEQ, RUNG_DIMS, rung  # noqa: E402
from azurite.env import MemoryGuard, configure_mps, pageouts, swap_used_mb  # noqa: E402
from azurite.model import Azurite, AzuriteDecision, parameter_counts  # noqa: E402
from azurite.tokenizers import PAD_ID  # noqa: E402

RESULTS = PROJECT / "results" / "local_bench"

# (rung, arm, seq, batch) — batch chosen to fit the cap at each size, from the
# sizes that were measured before. Kept small enough that a failure is
# informative rather than a machine-level problem.
# Both arms at every rung: the tokenizer sub-study needs small sizes only, but
# the budget's ladder designs are priced per arm and refuse to interpolate a
# missing rung rather than quietly reusing a neighbouring measurement.
PLAN = [
    ("tiny", "byte", 256, 32),
    ("small", "byte", 256, 16),
    ("mid", "byte", 256, 8),
    ("large", "byte", 256, 4),
    ("tiny", "bpe", 256, 32),
    ("small", "bpe", 256, 16),
    ("mid", "bpe", 256, 8),
    ("large", "bpe", 256, 4),
]


def _step(model, opt, ids, gen) -> float:
    """One optimizer step.

    Module-level rather than a closure inside `run_rung`: a closure holds a
    reference to `model` and `opt`, so `del model` in the cleanup path frees a
    name while the objects stay alive, `empty_cache()` releases nothing, and the
    *next* rung inherits the previous rung's footprint. That is exactly how the
    BPE rung OOMed here at 4.2 GiB while the byte rung of the same size fitted in
    2.1 GiB.
    """
    opt.zero_grad(set_to_none=True)
    loss, _ = model.mlm_loss(ids, generator=gen)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach())


def run_rung(
    name: str,
    arm: str,
    seq: int,
    batch: int,
    steps: int,
    warmup: int,
    mem_cap_gib: float,
) -> dict:
    cfg = rung(name, arm)
    counts = parameter_counts(cfg)
    result = {
        "rung": name,
        "arm": arm,
        "seq": seq,
        "batch": batch,
        "dims": RUNG_DIMS[name],
        "max_seq_available": MAX_SEQ[arm],
        "params_analytic": counts["pretrain_total"],
        "params_decision": counts["decision_total"],
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
    }
    guard = MemoryGuard(cap_gib=mem_cap_gib)
    swap0, po0 = guard.swap_start_mb, guard.pageouts_start
    model = None
    try:
        model = Azurite(cfg).to(result["device"])
        counted = model.num_parameters()
        result["params_instantiated"] = counted
        if counted != counts["pretrain_total"]:
            raise AssertionError(
                f"accounting mismatch: {counted:,} != {counts['pretrain_total']:,}"
            )
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
        gen = torch.Generator(device=result["device"]).manual_seed(0)
        ids = torch.randint(
            5, cfg.vocab_size, (batch, seq), generator=gen, device=result["device"]
        )
        ids[:, -2:] = PAD_ID

        t0 = time.perf_counter()
        losses = []
        reclaims = 0
        for i in range(warmup + steps):
            value = _step(model, opt, ids, gen)
            if i >= warmup:
                losses.append(value)
            guard.check()
            if guard.reclaim_if_needed():
                reclaims += 1
        if result["device"] == "mps":
            torch.mps.synchronize()
        wall = time.perf_counter() - t0
        measured = wall * steps / (warmup + steps)
        toks = batch * seq * steps
        result.update(
            status="ok",
            loss_first=round(losses[0], 4),
            loss_last=round(losses[-1], 4),
            finite=bool(all(math.isfinite(v) for v in losses)),
            seconds=round(wall, 3),
            measured_seconds=round(measured, 3),
            tokens_per_s=round(toks / measured, 1),
            sec_per_step=round(measured / steps, 4),
            # 6ND is the whole-family estimate; using the instantiated count
            # keeps this comparable to the target-run arithmetic in budget.py.
            achieved_tflops=round(6 * counted * (toks / measured) / 1e12, 3),
            memory=guard.report(),
            reclaims_between_steps=reclaims,
        )
        result["peak_mps_gib"] = result["memory"]["peak_reserved_gib"]
        result["peak_live_gib"] = result["memory"]["peak_live_gib"]
        result["swap_delta_mb"] = result["memory"]["swap_delta_mb"]
        result["pageouts_delta"] = result["memory"]["pageout_delta"]
        guard.assert_zero_swap()
    except Exception as exc:  # OOM and swap are results, not crashes
        result.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}"[:300],
            memory=guard.report(),
        )
        result["swap_delta_mb"] = result["memory"]["swap_delta_mb"]
        result["pageouts_delta"] = result["memory"]["pageout_delta"]
    finally:
        model = None
        opt = None
        ids = None
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mem-cap-gib", type=float, default=6.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    applied = configure_mps(args.mem_cap_gib)
    if applied["mps"]:
        print(
            f"mps recommended {applied['recommended_gib']:.2f} GiB -> cap fraction "
            f"{applied['fraction']:.2f} ({applied['cap_gib']:.2f} GiB, live-memory cap)"
        )

    plan = PLAN
    if args.only:
        wanted = set(args.only.split(","))
        plan = [p for p in plan if f"{p[0]}/{p[1]}" in wanted]

    print(f"torch {torch.__version__}  mps={torch.backends.mps.is_available()}")
    print(f"swap before: {swap_used_mb():.0f} MB\n")
    hdr = (
        f"{'rung/arm':12s} {'params':>12s} {'tok/s':>10s} {'TFLOP/s':>8s} "
        f"{'live':>6s} {'rsvd':>6s} {'swapMB':>8s} {'status':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))

    out = {
        "kind": "azurite_model_smoke",
        "host": platform.node(),
        "machine": platform.machine(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "mps": torch.backends.mps.is_available(),
        "mem_cap_gib": args.mem_cap_gib,
        "steps": args.steps,
        "warmup": args.warmup,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": (
            "measures the azurite implementation, not the deleted bench binary in "
            "results/local_bench/bench_20261027T030406.json, which used "
            "nn.MultiheadAttention and a non-gated MLP"
        ),
        "runs": [],
    }

    for name, arm, seq, batch in plan:
        r = run_rung(name, arm, seq, batch, args.steps, args.warmup, args.mem_cap_gib)
        out["runs"].append(r)
        key = f"{r['rung']}/{r['arm']}"
        if r["status"] == "ok":
            print(
                f"{key:12s} {r['params_instantiated']:>12,d} {r['tokens_per_s']:>10,.0f}"
                f" {r['achieved_tflops']:>8.2f} {r['peak_live_gib']:>6.2f}"
                f" {r['peak_mps_gib']:>6.2f} {r['swap_delta_mb']:>8.0f} {'ok':>8s}"
            )
        else:
            print(f"{key:12s} {'-':>12s} {'-':>10s} {'-':>8s} {'-':>6s} {'-':>6s} {'-':>8s} {'FAILED':>8s}")
            print(f"    {r['error']}")

    # The decision head is what the transfer stage trains; smoke it too, at the
    # median option count, so a broken packing path is caught before L2.
    print("\ndecision-head smoke (K=6):")
    torch.manual_seed(0)
    model = AzuriteDecision(Azurite(rung("tiny", "byte")))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t = 48
    ids = torch.randint(5, 200, (4, t))
    spans = torch.zeros(4, 6, 2, dtype=torch.long)
    for b in range(4):
        for k in range(6):
            spans[b, k] = torch.tensor([2 + k * 6, 2 + k * 6 + 5])
    from azurite.model import DecisionBatch

    batch = DecisionBatch(
        ids=ids,
        pad_mask=torch.ones_like(ids, dtype=torch.bool),
        spans=spans,
        cand_mask=torch.ones(4, 6, dtype=torch.bool),
        gold=torch.arange(4) % 6,
    )
    losses = []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = AzuriteDecision.loss(logits, batch.gold)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    out["decision_smoke"] = {
        "first_loss": round(losses[0], 4),
        "last_loss": round(losses[-1], 4),
        "steps": len(losses),
        "note": "30 steps on one fixed batch: overfitting it is the expected result",
    }
    print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["swap_after_mb"] = swap_used_mb()
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = RESULTS / f"model_smoke_{stamp}{('_' + args.label) if args.label else ''}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {path.relative_to(PROJECT)}")
    print(f"swap after: {swap_used_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
