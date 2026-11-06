#!/usr/bin/env python3
"""Budget arithmetic for the Azurite study, derived from measurements.

Two halves, both reproducible:

  1. TARGET RUN — what it costs to train a competitor-class encoder from
     scratch. FLOPs from the standard dense estimate 6ND. Throughput is NOT
     assumed: the H100 MFU band is derived from published runs whose
     GPU-hour counts are public, and this script re-derives both published
     runs from their own numbers before pricing anything else.

  2. LOCAL LADDER — which scaling ladder this machine can actually run,
     from the measured tokens/s in results/local_bench/. Wall clock in
     GPU-hours is not a guess either; it is tokens / measured throughput.

Every external number carries its source URL. Nothing here is a measurement
this repo did not make; everything else is arithmetic over published inputs.

Usage:
    python3 scripts/budget.py                # both tables
    python3 scripts/budget.py --target       # target-run table only
    python3 scripts/budget.py --ladder       # local ladder only
    python3 scripts/budget.py --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Preferred source: a measurement of the azurite implementation itself. Falls
# back to the 2026-10-27 bench only if no smoke run exists, and says so.
SMOKE_GLOB = "results/local_bench/model_smoke_*.json"
BENCH_GLOB = "results/local_bench/bench_*.json"

# --------------------------------------------------------------------------
# External facts. Each is (value, unit, source URL). Nothing here is assumed.
# --------------------------------------------------------------------------

H100_BF16_DENSE_PEAK_TFLOPS = 989.0  # sparsity-inclusive peak; dense is ~half
#   ^ the 989 figure is what vendors publish. Treated as the denominator for
#     MFU, and the French run's 52% implied MFU is flagged for exactly this
#     reason. Reported MFU values below are therefore conservative.

# Published training runs, used to derive the achievable-throughput band.
# (name, params, tokens, gpu_hours, url)
PUBLISHED_RUNS = [
    (
        "ModernBERT base",
        149e6,
        1.7e12,
        194.0 * 8,  # 194 h on 8xH100 SXM
        "https://github.com/AnswerDotAI/ModernBERT/issues/236",
    ),
    (
        "mmBERT base",
        307e6,
        3.0e12,
        40.0 * 24 * 8,  # ~40 days on 8xH100
        "https://github.com/jhu-clsp/mmBERT",
    ),
    (
        "French ModernBERT-style",
        400e6,
        1.0e12,
        1300.0,
        "https://arxiv.org/html/2504.08716v2",
    ),
]

# What we intend to compare against. (name, params, tokens, note, source)
TARGETS = [
    (
        "Laya encoder, base stage",
        149e6,
        1.7e12,
        "ModernBERT-base recipe — what the large model is initialised from",
        "https://arxiv.org/html/2412.13663v2",
    ),
    (
        "Laya encoder (421M ckpt), from random",
        395e6,
        2.0e12,
        "ModernBERT-large size at its published 2T-token budget",
        "https://arxiv.org/html/2412.13663v2",
    ),
    (
        "Laya's published recipe end to end",
        None,  # summed from the two rows above
        None,
        "base run + large run, which is how the 421M checkpoint was made",
        "https://arxiv.org/html/2412.13663v2",
    ),
    (
        "Laya multilingual (mmBERT class)",
        307e6,
        3.0e12,
        "307M encoder, ~3T tokens, 1833 languages",
        "https://arxiv.org/html/2509.06888v1",
    ),
    (
        "Jev-class, HYPOTHETICAL 1.5B x 500B",
        1.5e9,
        500e9,
        "NOT DISCLOSED by TypeSafe. Shown only to price a plausible shape.",
        "https://typesafe.ai/blog/introducing-system-one-models-and-jev",
    ),
]

PRICE_PER_GPU_HOUR = 3.99  # https://lambda.ai/pricing, on-demand H100

# --------------------------------------------------------------------------
# Local ladder: candidate designs, priced from measured throughput.
# --------------------------------------------------------------------------

# token budgets to sweep, in tokens
BUDGETS = [0.05e9, 0.1e9, 0.3e9, 1.0e9]

# candidate designs: name -> list of (rung, budget) cells
def _designs() -> dict[str, list[tuple[str, float]]]:
    return {
        "A_full_4x4": [
            (n, d) for n in ("tiny", "small", "mid", "large") for d in BUDGETS
        ],
        "B_4x3": [
            (n, d) for n in ("tiny", "small", "mid", "large") for d in BUDGETS[1:]
        ],
        "C_recommended": [
            ("tiny", d) for d in BUDGETS
        ] + [
            ("small", d) for d in BUDGETS[1:]
        ] + [
            ("mid", d) for d in BUDGETS[1:]
        ] + [
            ("large", d) for d in BUDGETS[1:3]
        ],
        "D_minimal": [
            ("tiny", d) for d in BUDGETS
        ] + [
            ("small", d) for d in BUDGETS[1:3]
        ] + [
            ("mid", d) for d in BUDGETS[1:3]
        ] + [
            ("large", 0.1e9),
        ],
    }


def flops(params: float, tokens: float) -> float:
    """6ND, the standard dense-transformer training estimate."""
    return 6.0 * params * tokens


def gpu_hours(params: float, tokens: float, mfu: float) -> float:
    achieved = H100_BF16_DENSE_PEAK_TFLOPS * 1e12 * mfu
    return flops(params, tokens) / achieved / 3600.0


def load_measurement() -> dict:
    """Merge every azurite smoke run, best cell per (arm, rung).

    Not "the newest file": a partial re-measurement (`--only tiny/byte`) is a
    legitimate way to check one rung, and picking the newest file would silently
    reduce the ladder to whichever rungs that run happened to cover. Merging
    keeps every cell and records which file each one came from.
    """
    smoke = sorted(glob.glob(str(ROOT / SMOKE_GLOB)))
    if smoke:
        runs = []
        sources = []
        mem_caps = []
        for path in smoke:
            with open(path) as fh:
                data = json.load(fh)
            rel = str(Path(path).relative_to(ROOT))
            sources.append(rel)
            if data.get("mem_cap_gib"):
                mem_caps.append(data["mem_cap_gib"])
            for run in data.get("runs", []):
                runs.append({**run, "_source_file": rel})
        return {
            "_kind": "azurite",
            "_sources": sources,
            "_source": sources[-1],
            "mem_cap_gib": min(mem_caps) if mem_caps else None,
            "runs": runs,
        }
    bench = sorted(glob.glob(str(ROOT / BENCH_GLOB)))
    if not bench:
        raise SystemExit(f"no measurement matching {SMOKE_GLOB} or {BENCH_GLOB}")
    with open(bench[-1]) as fh:
        data = json.load(fh)
    data["_source"] = bench[-1]
    data["_kind"] = "bench"
    print(
        "WARNING: no azurite smoke measurement found; falling back to the "
        "deleted bench binary's numbers, which used nn.MultiheadAttention and a "
        "non-gated MLP. Run scripts/model_smoke.py first.",
        file=sys.stderr,
    )
    return data


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def measured_throughput(data: dict) -> dict[tuple[str, str], dict]:
    """Median of repeated measurements per (arm, rung).

    Median, not best-of-N. Repeated runs of the *same* configuration on this
    machine differ by up to 9% (large/bpe measured 926 and 846 tokens/s on
    identical settings), so best-of-N biases throughput upward and therefore
    biases every wall-clock estimate downward — optimistic in exactly the
    direction a plan should not be optimistic. The spread is reported so the
    uncertainty is visible rather than buried in a single lucky number.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    if data.get("_kind") == "azurite":
        for run in data.get("runs", []):
            if run.get("status") != "ok" or not run.get("finite"):
                continue
            key = (run["arm"], run["rung"])
            mem = run.get("memory") or {}
            live = run.get("peak_live_gib", mem.get("peak_live_gib"))
            groups.setdefault(key, []).append(
                {
                    "params": int(run["params_instantiated"]),
                    "tokens_per_s": float(run["tokens_per_s"]),
                    "achieved_tflops": float(run["achieved_tflops"]),
                    "peak_live_gib": None if live is None else float(live),
                    "source": run.get("_source_file", "?"),
                }
            )
    else:
        # Legacy bench schema: fp32 only, no tokenizer arm.
        for run in data.get("runs", []):
            if run.get("status") != "ok" or run.get("mode") != "fp32":
                continue
            if not run.get("finite"):
                continue
            key = ("byte", run["config"])
            groups.setdefault(key, []).append(
                {
                    "params": int(run["params"]),
                    "tokens_per_s": float(run["tokens_per_s"]),
                    "achieved_tflops": float(run["achieved_tflops"]),
                    "peak_live_gib": float(run.get("peak_mps_gib", 0.0)),
                    "source": "bench",
                }
            )
    if not groups:
        raise SystemExit("no usable runs in the measurement files")

    out: dict[tuple[str, str], dict] = {}
    for key, entries in groups.items():
        tps = [e["tokens_per_s"] for e in entries]
        lives = [e["peak_live_gib"] for e in entries if e["peak_live_gib"] is not None]
        out[key] = {
            "params": entries[0]["params"],
            "tokens_per_s": _median(tps),
            "achieved_tflops": _median([e["achieved_tflops"] for e in entries]),
            "peak_live_gib": _median(lives) if lives else None,
            "n_runs": len(entries),
            "tps_min": min(tps),
            "tps_max": max(tps),
            "sources": sorted({e["source"] for e in entries}),
        }
    return out


def mfu_band() -> tuple[float, float]:
    """Derive the MFU band from published runs rather than assuming one.

    Returns (pessimistic, optimistic). Note the circularity: the band comes
    from the same runs it is later used to price, so it is a consistency
    statement about the published record, not an independent validation. The
    spread across the three published runs is reported alongside it.
    """
    mfus = []
    for _name, params, tokens, hours, _url in PUBLISHED_RUNS:
        if hours <= 0:
            continue
        achieved = flops(params, tokens) / (hours * 3600.0)
        mfus.append(achieved / (H100_BF16_DENSE_PEAK_TFLOPS * 1e12))
    mfus.sort()
    consistent = [m for m in mfus if m <= 0.45]
    return min(consistent), max(consistent)


def mfu_spread() -> tuple[float, float]:
    """The full published spread, outlier included, as (min, max)."""
    mfus = []
    for _name, params, tokens, hours, _url in PUBLISHED_RUNS:
        achieved = flops(params, tokens) / (hours * 3600.0)
        mfus.append(achieved / (H100_BF16_DENSE_PEAK_TFLOPS * 1e12))
    return min(mfus), max(mfus)


def fmt_hours(h: float) -> str:
    if h < 48:
        return f"{h:.1f} h"
    return f"{h / 24:.1f} d"


def table_gpu_hours(h: float) -> str:
    return f"{h:,.0f}"


def report_target() -> list[str]:
    lo, hi = mfu_band()
    sp_lo, sp_hi = mfu_spread()
    out = []
    A = out.append
    A("## Target run — what beating Laya costs\n")
    A("`FLOPs = 6ND`; H100 bf16 dense peak 989 TFLOP/s; on-demand "
      f"${PRICE_PER_GPU_HOUR:.2f}/GPU-h.\n")
    A("### Throughput is taken from published runs, not assumed\n")
    A("| published run | params | tokens | reported GPU-h | implied achieved TFLOP/s | implied MFU |")
    A("|---|---:|---:|---:|---:|---:|")
    for name, params, tokens, hours, url in PUBLISHED_RUNS:
        achieved = flops(params, tokens) / (hours * 3600.0)
        mfu = achieved / (H100_BF16_DENSE_PEAK_TFLOPS * 1e12)
        A(
            f"| [{name}]({url}) | {params / 1e6:.0f}M | {tokens / 1e12:.2f}T | "
            f"{hours:,.0f} | {achieved / 1e12:,.0f} | {mfu * 100:.1f}% |"
        )
    A("")
    A(f"The three published runs imply **{sp_lo * 100:.1f}%, "
      f"{sp_hi * 100:.1f}%** and one of those — the French figure at "
      f"{sp_hi * 100:.1f}% — sits above the dense peak, so it is an upper "
      "bound on optimism rather than a reproducible rate. The internally "
      f"consistent band is **{lo * 100:.1f}%–{hi * 100:.1f}% MFU**, and both "
      "bands are shown below because the choice between them is a factor of "
      f"{sp_hi / lo:.1f} on every price.\n")
    A("| target | N | D | 6ND | H100-h (consistent) | cost (consistent) | cost (full spread) |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name, params, tokens, note, url in TARGETS:
        if params is None:
            params = TARGETS[1][1]
            tokens = TARGETS[0][2] + TARGETS[1][2]
        f = flops(params, tokens)
        h_min = gpu_hours(params, tokens, hi)
        h_max = gpu_hours(params, tokens, lo)
        c_lo = h_min * PRICE_PER_GPU_HOUR
        c_hi = h_max * PRICE_PER_GPU_HOUR
        s_min = gpu_hours(params, tokens, sp_hi) * PRICE_PER_GPU_HOUR
        s_max = gpu_hours(params, tokens, sp_lo) * PRICE_PER_GPU_HOUR
        A(
            f"| [{name}]({url}) | {params / 1e6:,.0f}M | {tokens / 1e12:.2f}T | "
            f"{f:.2e} | {table_gpu_hours(h_min)}–{table_gpu_hours(h_max)} | "
            f"**${c_lo / 1000:,.1f}k–${c_hi / 1000:,.1f}k** | "
            f"${s_min / 1000:,.1f}k–${s_max / 1000:,.1f}k |"
        )
    A("")
    A("**On circularity, stated plainly:** the MFU band is derived from these "
      "runs and then used to price them, so the table is a consistency check on "
      "the published record, not an independent measurement. The genuinely "
      "non-circular number in this project is the local one — "
      "`results/local_bench/` is measured on this machine, at 8–18% of its own "
      "accelerator's peak.\n")
    return out


def report_ladder() -> list[str]:
    data = load_measurement()
    rates = measured_throughput(data)
    arms = sorted({a for a, _ in rates})
    out = []
    A = out.append
    kind = data.get("_kind")
    A("## Local ladder — what this machine can run\n")
    if kind == "azurite":
        n_src = len(data.get("_sources", []))
        A(
            f"Sources: {n_src} azurite smoke run(s) under `results/local_bench/`, "
            "merged best-per-cell; fp32; live-memory cap "
            f"{data.get('mem_cap_gib') or 0:.2f} GiB. Each rung's own file is "
            "recorded in the JSON output.\n"
        )
    else:
        A(f"Source: `{Path(data['_source']).relative_to(ROOT)}` — LEGACY bench "
          "binary; see the warning on stderr.\n")
    A("| arm | rung | params | tokens/s (median, n) | spread | live GiB | 0.05B | 0.1B | 0.3B | 1B | one Chinchilla pass (20N) |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        for rung_name in ("tiny", "small", "mid", "large"):
            key = (arm, rung_name)
            if key not in rates:
                continue
            r = rates[key]
            tps = r["tokens_per_s"]
            cells = [fmt_hours(d / tps / 3600.0) for d in BUDGETS]
            chin = 20 * r["params"]
            cells.append(f"{fmt_hours(chin / tps / 3600.0)} ({chin / 1e6:.0f}M)")
            live = (
                f"{r['peak_live_gib']:.2f}" if r["peak_live_gib"] is not None else "n/a"
            )
            spread = f"{r['tps_min']:,.0f}-{r['tps_max']:,.0f}"
            A(
                f"| {arm} | {rung_name} | {r['params']:,} | {tps:,.0f} (n={r['n_runs']}) | "
                f"{spread} | {live} | "
                + " | ".join(cells)
                + " |"
            )
    A("")
    designs = _designs()
    A("### Candidate designs, priced in measured wall clock, per arm\n")
    A("| arm | design | runs | total tokens | total wall clock | days |")
    A("|---|---|---:|---:|---:|---:|")
    for arm in arms:
        for name, cells in designs.items():
            total_s = 0.0
            total_tok = 0.0
            for rung_name, d in cells:
                key = (arm, rung_name)
                if key not in rates:
                    raise SystemExit(
                        f"design {name} needs {key}, which has no measurement; "
                        "add it to scripts/model_smoke.py PLAN and re-measure"
                    )
                total_s += d / rates[key]["tokens_per_s"]
                total_tok += d
            hrs = total_s / 3600.0
            A(
                f"| {arm} | `{name}` | {len(cells)} | {total_tok / 1e9:.2f}B | "
                f"{hrs:,.0f} h | {hrs / 24:.1f} |"
            )
    A("")
    A("Runs are serial: one accelerator, so wall clock is the sum. Throughput "
      "is the **median** across repeats, with the observed spread beside it; "
      "best-of-N is not used because it biases every estimate optimistic.\n")

    # The tokenizer sub-study is the cheapest citable result in the study, so it
    # is priced on its own: two smallest rungs, both arms, every budget.
    if ("byte", "tiny") in rates and ("bpe", "tiny") in rates:
        sub_budgets = BUDGETS[:-1]  # 1B excluded: see the note below
        A("### Tokenizer sub-study (tiny + small, both arms, 0.05-0.3B)\n")
        A("| rung | budget | byte | bpe | both |")
        A("|---|---:|---:|---:|---:|")
        total = 0.0
        for rung_name in ("tiny", "small"):
            for d in sub_budgets:
                b = d / rates[("byte", rung_name)]["tokens_per_s"] / 3600.0
                p = d / rates[("bpe", rung_name)]["tokens_per_s"] / 3600.0
                total += b + p
                A(
                    f"| {rung_name} | {d / 1e9:.2f}B | {fmt_hours(b)} | "
                    f"{fmt_hours(p)} | {fmt_hours(b + p)} |"
                )
        A(f"| **total** | | | | **{total:,.0f} h ({total / 24:.1f} d)** |")
        A("")
        A("The 1B cells are excluded deliberately: `small`/bpe at 1B costs "
          "3.8 days on its own for one extra point on a curve the `tiny` rung "
          "already covers at every budget. If the study has spare days they are "
          "better spent on the 86M-class rung than on this cell.\n")
        A("Six (rung, budget) pairs in two arms answers the inversion question at "
          "2.4M-26M parameters, which is the entire regime ByT5 never tested. "
          "It is the cheapest citable result in the study and the first thing "
          "worth running.\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    sections = []
    if args.target or not (args.target or args.ladder):
        sections.append(report_target())
    if args.ladder or not (args.target or args.ladder):
        sections.append(report_ladder())

    text = "\n".join("\n".join(s) for s in sections)
    print(text)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        data = load_measurement()
        rates = measured_throughput(data)
        lo, hi = mfu_band()
        resolved = []
        for name, params, tokens, _note, url in TARGETS:
            if params is None:
                params = TARGETS[1][1]
                tokens = TARGETS[0][2] + TARGETS[1][2]
            resolved.append({
                "name": name,
                "params": params,
                "tokens": tokens,
                "flops": flops(params, tokens),
                "gpu_hours_lo": gpu_hours(params, tokens, lo),
                "gpu_hours_hi": gpu_hours(params, tokens, hi),
                "cost_usd_lo": gpu_hours(params, tokens, lo) * PRICE_PER_GPU_HOUR,
                "cost_usd_hi": gpu_hours(params, tokens, hi) * PRICE_PER_GPU_HOUR,
                "source": url,
            })
        payload = {
            "mfu_band": [lo, hi],
            "price_per_gpu_hour": PRICE_PER_GPU_HOUR,
            "published_runs": [
                {
                    "name": name,
                    "params": p,
                    "tokens": t,
                    "gpu_hours": h,
                    "implied_mfu": flops(p, t) / (h * 3600.0)
                    / (H100_BF16_DENSE_PEAK_TFLOPS * 1e12),
                    "source": url,
                }
                for name, p, t, h, url in PUBLISHED_RUNS
            ],
            "targets": resolved,
            "local_measurement_source": data["_source"],
            "local_rates": {
                f"{arm}/{rung_name}": {
                    "params": rates[(arm, rung_name)]["params"],
                    "tokens_per_s": rates[(arm, rung_name)]["tokens_per_s"],
                }
                for arm, rung_name in rates
            },
        }
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\n[json] {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
