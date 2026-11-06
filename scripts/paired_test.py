#!/usr/bin/env python3
"""Paired comparison of two decision systems on identical test rows.

Two accuracy point estimates from different systems are not a comparison: the
only defensible test uses the same rows, so the disagreement cells can be
counted. This reports

  * exact McNemar (binomial on the discordant pairs) -- the test appropriate
    for a paired 0/1 outcome pair, no normal approximation,
  * a paired bootstrap CI for the accuracy difference,
  * the discordance breakdown (A right/B wrong vs A wrong/B right).

Inputs are JSON files each holding ``predictions`` and ``gold`` over the same
rows in the same order (the harness ``funnel_keep.json`` and
``scripts/run_pplx_baseline.py`` output both do).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_preds(spec: str) -> tuple[list[int], list[int], dict]:
    path, _, key = spec.partition(":")
    d = json.loads(Path(path).read_text())
    # the harness writes predictions_bank (label indices after the funnel
    # mapping); the competitor runner writes predictions -- accept either so a
    # typo cannot silently pick the wrong vector
    field = key or ("predictions_bank" if "predictions_bank" in d else "predictions")
    preds = d[field]
    gold = d["gold"]
    meta = {k: v for k, v in d.items() if k in ("task", "model", "pool", "seed", "split_seed",
                                                "n", "k", "accuracy", "adaptation")}
    return preds, gold, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="candidate preds JSON (optionally :bank)")
    ap.add_argument("--b", required=True, help="baseline preds JSON")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pa, ga, ma = load_preds(args.a)
    pb, gb, mb = load_preds(args.b)
    n = len(ga)
    if not (len(pa) == len(pb) == len(ga) == len(gb)):
        raise SystemExit(f"row-count mismatch: A preds {len(pa)}, B preds {len(pb)}, "
                         f"gold {len(ga)}/{len(gb)}")
    if ga != gb:
        raise SystemExit("gold vectors differ: the two systems were not scored on the "
                         "same rows in the same order, so a paired test is invalid")

    ca = [int(p == g) for p, g in zip(pa, ga)]
    cb = [int(p == g) for p, g in zip(pb, gb)]
    acc_a, acc_b = sum(ca) / n, sum(cb) / n
    # discordant cells
    n01 = sum(1 for x, y in zip(ca, cb) if x == 0 and y == 1)  # A wrong, B right
    n10 = sum(1 for x, y in zip(ca, cb) if x == 1 and y == 0)  # A right, B wrong
    m = n01 + n10
    if m:
        # exact two-sided McNemar via binomial; m <= a few hundred so this is cheap
        from math import comb
        k = min(n01, n10)
        tail = sum(comb(m, i) for i in range(k + 1)) / (2 ** m)
        p_mcnemar = min(1.0, 2 * tail)
    else:
        p_mcnemar = 1.0

    rng = random.Random(args.seed)
    diffs = []
    for _ in range(args.boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = sum(ca[i] for i in idx)
        sb = sum(cb[i] for i in idx)
        diffs.append((sa - sb) / n)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]

    res = {
        "n": n,
        "a": {"meta": ma, "accuracy": round(acc_a, 4), "correct": sum(ca)},
        "b": {"meta": mb, "accuracy": round(acc_b, 4), "correct": sum(cb)},
        "delta_a_minus_b": round(acc_a - acc_b, 4),
        "discordant": {"a_right_b_wrong": n10, "a_wrong_b_right": n01},
        "mcnemar_exact_p_two_sided": round(p_mcnemar, 6),
        "bootstrap": {"iters": args.boot, "seed": args.seed,
                      "ci95": [round(lo, 4), round(hi, 4)], "frac_gt_zero": round(
                          sum(1 for d in diffs if d > 0) / len(diffs), 4)},
        "sensitive": bool(p_mcnemar < 0.05 and lo > 0),
        "note": "paired on identical rows; discordant cells drive the test",
    }
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
