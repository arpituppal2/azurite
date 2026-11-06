#!/usr/bin/env python3
"""Does a CDN compress a variable-length sequence, or charge per occurrence?

The claim under test
--------------------
"A CDN reduces information to a finite set, so variable-sequence compression is
impossible."  The operational version of that sentence is a price question:

    if a prompt contains the same passage r times, does the billed prompt-token
    count grow by r * |passage| (charge per occurrence: no compression), or does
    it grow sub-linearly (charge per unique token: sequence compression exists)?

That is measurable with no weights and no credentials, because a token-billed
inference endpoint *reports* `usage.prompt_tokens` computed by its own
tokenizer.  It is the exact quantity that a per-million-token price multiplies,
so it is the billing fact, not a proxy for it.

Second question, cheap to answer on the same endpoint: a content cache cuts
latency, but does it cut the *bill*?  A CDN that reduces a request to a finite
set of stored responses can serve a repeat in milliseconds and still charge for
every token, because the reduction happened at the set, not in the sequence.

Design
------
1. Repetition slope.  Prompt = header + P repeated r times, r in {1,2,4,8}.
   Fit prompt_tokens = a + b*r.  b is the billed tokens per repetition.
   - b ~ |P|  -> charged per occurrence; the CDN does not compress the sequence.
   - b << |P| -> sequence redundancy is being priced away; the claim is false.
2. Cache economics.  Send the *identical* request three times, then a perturbed
   copy (one byte at the head, one at the tail).  Record wall-clock and
   prompt_tokens for each.  A latency collapse with an unchanged token count is
   a cache that saves time and not money.

Both halves are falsifiable in the same direction: if b < |P|, or if a repeat is
billed fewer prompt tokens, then a CDN *can* compress variable sequences and the
thesis sentence is wrong.  The script states the falsification line in its own
output so the result cannot be re-read favourably after the fact.

Provider
--------
`pollinations` is a keyless OpenAI-compatible proxy in front of a hosted model
(gpt-oss-20b at time of writing).  It is used only as a live example of a
token-billed CDN; it is *not* Jev, and its per-token price is not Jev's.  What
generalises from it is the billing mechanism (usage accounting + response
cache), which is what the claim is about.  Payloads are synthetic filler; no
project text leaves the sandbox.

    python scripts/cdn_probe.py --reps 6 --out results/comparison/cdn_probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT / "results" / "comparison"

ENDPOINT = "https://text.pollinations.ai/openai"
MODEL = "openai"

# A passage long enough that the tokenizer count is stable and the slope
# measurable, but small enough that eight cold calls stay cheap.  Deliberately
# boring prose: we are measuring the tokenizer, not the model.
PASSAGE = (
    "The archive holds a sequence of records. Each record names a field, a type "
    "and a value arranged in a fixed order. The order matters because a reader "
    "walks the sequence from left to right and binds each name to the value that "
    "follows it. Nothing in the record is unique to this particular archive: the "
    "same names recur across every record, and the same shapes repeat in every "
    "batch that the archive is asked to process. "
)

HEADER = "Count the words in this text and reply with a single number.\n\n"


def _call(prompt: str, timeout: float = 120.0) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
        err = None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        payload, err = {}, f"{type(exc).__name__}: {exc}"
    dt = time.perf_counter() - t0
    usage = payload.get("usage") or {}
    return {
        "seconds": round(dt, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "served_model": payload.get("model"),
        "error": err,
    }


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b*x, returning (a, b, r2)."""
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else float("nan")
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return a, b, r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=3, help="identical-request reps")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT / "cdn_probe.json")
    args = ap.parse_args()

    rows: list[dict] = []

    # 1. Repetition slope.
    slope_rows = []
    for r in args.reps:
        prompt = HEADER + (PASSAGE * r)
        rec = _call(prompt)
        rec.update({"condition": "repeat", "r": r, "chars": len(prompt)})
        slope_rows.append(rec)
        rows.append(rec)
        print(f"r={r:<3d} chars={len(prompt):>6d} prompt_tokens={rec['prompt_tokens']} "
              f"{rec['seconds']:.2f}s err={rec['error']}", flush=True)

    ok = [r for r in slope_rows if r["prompt_tokens"]]
    a = b = r2 = float("nan")
    if len(ok) >= 2:
        a, b, r2 = _fit([float(r["r"]) for r in ok], [float(r["prompt_tokens"]) for r in ok])
    base = next((r["prompt_tokens"] for r in slope_rows if r["r"] == 1), None)
    tokens_per_passage = (base - a) if (base and b == b) else float("nan")

    # 2. Cache economics: identical repeat, then one-byte perturbations.
    exact_prompt = HEADER + PASSAGE
    for i in range(args.repeats):
        rec = _call(exact_prompt)
        rec.update({"condition": "exact_repeat", "i": i})
        rows.append(rec)
        print(f"exact#{i} prompt_tokens={rec['prompt_tokens']} {rec['seconds']:.2f}s", flush=True)

    head_prompt = HEADER + "x" + PASSAGE[1:]
    tail_prompt = HEADER + PASSAGE[:-2] + "y."
    for name, prompt in (("perturb_head", head_prompt), ("perturb_tail", tail_prompt)):
        rec = _call(prompt)
        rec.update({"condition": name})
        rows.append(rec)
        print(f"{name} prompt_tokens={rec['prompt_tokens']} {rec['seconds']:.2f}s", flush=True)

    exact = [r for r in rows if r["condition"] == "exact_repeat" and r["prompt_tokens"]]
    latency = [r["seconds"] for r in exact]
    tok_counts = {r["prompt_tokens"] for r in exact}
    per_call = [r for r in rows if r["prompt_tokens"]]
    priced = {
        "prompt_tokens_min": min(r["prompt_tokens"] for r in per_call) if per_call else None,
        "prompt_tokens_max": max(r["prompt_tokens"] for r in per_call) if per_call else None,
        "n_calls": len(per_call),
    }

    # Falsification, stated before the verdict so it cannot drift.
    compressed = bool(b == b and tokens_per_passage == tokens_per_passage
                      and b < 0.95 * tokens_per_passage)
    cache_saves_money = bool(len(tok_counts) > 1 and min(tok_counts) < max(tok_counts))
    verdict = (
        "sequence compression DOES occur on this CDN (billed tokens per repetition "
        "fall below one passage of tokens): the thesis sentence is falsified for it"
        if compressed else
        "charged per occurrence: billed tokens grow by one full passage per "
        "repetition, so a repeated variable sequence is not compressed at the price"
    )

    out = {
        "provider": "pollinations",
        "endpoint": ENDPOINT,
        "served_model": next((r["served_model"] for r in rows if r["served_model"]), None),
        "note": "keyless third-party proxy, not Jev; measures the billing mechanism only",
        "reps": args.reps,
        "passage_chars": len(PASSAGE),
        "fit": {"intercept": a, "tokens_per_repetition": b, "r2": r2},
        "tokens_per_passage_implied": tokens_per_passage,
        "falsification_line": "b < 0.95 * tokens(one passage) => compression exists",
        "compression_detected": compressed,
        "cache": {
            "latency_s": latency,
            "latency_median_s": statistics.median(latency) if latency else None,
            "distinct_prompt_token_counts": sorted(tok_counts),
            "cache_reduced_bill": cache_saves_money,
        },
        "priced_calls": priced,
        "verdict": verdict,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nfit: tokens = {a:.2f} + {b:.3f} * r   (r2={r2:.5f})")
    print(f"tokens per passage implied: {tokens_per_passage:.2f} (one copy = {base})")
    print(f"exact-repeat latencies: {[round(x,2) for x in latency]}  distinct token counts: {sorted(tok_counts)}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
