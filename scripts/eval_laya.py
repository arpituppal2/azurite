#!/usr/bin/env python3
"""Measure Laya on this project's frozen test splits.

Why this exists: "beats Laya" is only a claim about measurement if Laya is run
here, on these rows, with these labels. Laya's own published numbers come from
its own benchmark on its own splits; comparing against those would be comparing
two different experiments, which is how this project produced a number nobody
could reproduce before.

Protocol, fixed before the first call and identical for every model compared:

  * options are read from the dataset's own label space, never hand-written
  * state text is the task's input fields joined with a single space
  * one question per row, type `choice`, options in a fixed sorted order
  * prediction is Laya's returned choice; a response outside the option set
    counts as wrong and is recorded as a parse failure rather than discarded
  * metric is top-1 accuracy over every row of the split; latency is wall clock
    per call, reported as p50/p90, and includes the model's own preprocessing

Usage:
    .venv/bin/python3 scripts/eval_laya.py --task banking77 --limit 20
    .venv/bin/python3 scripts/eval_laya.py --task ag_news
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
RESULTS = PROJECT / "results" / "laya_baseline"

# task -> how to build a decision from one row.
#   path:        directory under data/raw (config, split)
#   state:       columns concatenated into the input text
#   options:     ("feature", col) for a ClassLabel, or ("column", col) for a
#                string column holding the label name
#   gold:        same shape as options
#   instruction: the question text, identical for every row and every model
TASKS: dict[str, dict] = {
    "banking77": {
        "path": ["default", "test"], "state": ["text"],
        "options": ("column", "label_text"), "gold": ("column", "label_text"),
        "instruction": "Which banking intent does this customer message express?",
    },
    "clinc_oos": {
        "path": ["plus", "test"], "state": ["text"],
        "options": ("feature", "intent"), "gold": ("feature", "intent"),
        "instruction": "Which virtual-assistant intent does this utterance express?",
    },
    "ag_news": {
        "path": ["default", "test"], "state": ["text"],
        "options": ("feature", "label"), "gold": ("feature", "label"),
        "instruction": "Which news topic is this headline about?",
    },
    "emotion": {
        "path": ["split", "test"], "state": ["text"],
        "options": ("feature", "label"), "gold": ("feature", "label"),
        "instruction": "Which emotion does this text express?",
    },
    "massive_intent": {
        "path": ["en", "test"], "state": ["text"],
        "options": ("column", "label_text"), "gold": ("column", "label_text"),
        "instruction": "Which virtual-assistant intent does this utterance express?",
    },
    "massive_scenario": {
        "path": ["en", "test"], "state": ["text"],
        "options": ("column", "label_text"), "gold": ("column", "label_text"),
        "instruction": "Which domain does this utterance belong to?",
    },
    "xnli": {
        "path": ["en", "test"], "state": ["premise", "hypothesis"],
        "options": ("feature", "label"), "gold": ("feature", "label"),
        "instruction": "Does the premise entail the hypothesis?",
    },
    "xnli_de": {
        "path": ["de", "test"], "state": ["premise", "hypothesis"],
        "options": ("feature", "label"), "gold": ("feature", "label"),
        "instruction": "Does the premise entail the hypothesis?",
    },
    "xnli_zh": {
        "path": ["zh", "test"], "state": ["premise", "hypothesis"],
        "options": ("feature", "label"), "gold": ("feature", "label"),
        "instruction": "Does the premise entail the hypothesis?",
    },
    "language_id": {
        "path": ["default", "test"], "state": ["text"],
        "options": ("feature", "labels"), "gold": ("feature", "labels"),
        "instruction": "Which language is this text written in?",
    },
    "financial_phrasebank": {
        "path": ["default", "test"], "state": ["text"],
        "options": ("column", "label_text"), "gold": ("column", "label_text"),
        "instruction": "What is the sentiment of this financial sentence?",
    },
    "massive_intent_de": {
        "path": ["de", "test"], "state": ["text"],
        "options": ("column", "label_text"), "gold": ("column", "label_text"),
        "instruction": "Which virtual-assistant intent does this utterance express?",
    },
}


def load_task(key: str):
    """Return the dataset, spec, option labels and a raw-value -> label converter.

    The converter is not optional detail. A ClassLabel column yields integers;
    comparing a predicted label string against an integer gold marks every row
    wrong and reports a clean 0.0000 with zero call failures. That is exactly
    the kind of number that gets mistaken for a result.
    """
    from datasets import load_from_disk

    spec = TASKS[key]
    ds = load_from_disk(str(RAW / key / Path(*spec["path"])))

    def options_for(side: str):
        kind, col = spec[side]
        if kind == "feature":
            feat = ds.features[col]
            if not hasattr(feat, "names") or not feat.names:
                raise SystemExit(f"{key}: feature {col} is not a ClassLabel")
            return list(feat.names)
        from collections import Counter

        return sorted(Counter(ds[col]))

    options = options_for("options")
    gold_kind, gold_col = spec["gold"]
    feat = ds.features[gold_col]
    names = list(getattr(feat, "names", []) or [])

    def to_label(raw) -> str:
        if isinstance(raw, str):
            return raw
        if names and isinstance(raw, int) and 0 <= raw < len(names):
            return names[raw]
        return str(raw)

    return ds, spec, options, to_label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="banking77")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="english")
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    from laya import Router

    RESULTS.mkdir(parents=True, exist_ok=True)
    ds, spec, options, to_label = load_task(args.task)
    n = min(args.limit or ds.num_rows, ds.num_rows)
    print(f"{args.task}: {ds.num_rows} rows, {len(options)} options, "
          f"evaluating {n}, device={args.device}")

    t0 = time.time()
    router = Router(device=args.device)
    print(f"Router built in {time.time() - t0:.1f}s "
          f"(checkpoints are fetched on first use)")

    # The package validates questions up front: a `choice` takes `criteria`
    # (a list of labels, or a dict of label -> description), not `options`.
    # Using the wrong key raised ValueError on every row of the first run and
    # the harness recorded 0.0000 accuracy for all 20 -- which is why the run
    # log is kept: a zero that means "the call failed" looks identical to a
    # zero that means "the model failed" in the result file alone.
    question = {
        "answer": {
            "type": "choice",
            "instructions": spec["instruction"],
            "criteria": options,
        }
    }
    gold = [to_label(v) for v in ds[spec["gold"][1]][:n]]
    states = [
        " ".join(str(row[f]) for f in spec["state"])
        for row in ds.select(range(n))
    ]

    correct = 0
    parse_failures = 0
    latencies: list[float] = []
    wrong: list[dict] = []
    for i, (state, want) in enumerate(zip(states, gold)):
        t1 = time.perf_counter()
        try:
            res = router.predict(state, question, model=args.model,
                                 max_len=args.max_len)
            pred = res["answers"]["answer"]["choice"]
        except Exception as exc:  # a failure is a result, not a crash
            parse_failures += 1
            pred = f"<{type(exc).__name__}: {str(exc)[:60]}>"
        dt = time.perf_counter() - t1
        latencies.append(dt)
        ok = pred == want
        correct += ok
        if not ok and len(wrong) < 10:
            wrong.append({"i": i, "want": str(want)[:60], "got": str(pred)[:60]})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}  acc={correct / (i + 1):.4f}  "
                  f"p50={statistics.median(latencies) * 1000:.0f}ms", flush=True)

    acc = correct / n if n else 0.0
    p50 = statistics.median(latencies) * 1000 if latencies else 0.0
    p90 = (
        statistics.quantiles(latencies, n=10)[8] * 1000
        if len(latencies) >= 10 else p50
    )
    print(f"\n{args.task}: accuracy {acc:.4f} on {n} rows "
          f"({correct} correct, {parse_failures} failures), "
          f"latency p50 {p50:.0f} ms, p90 {p90:.0f} ms")

    out = {
        "model": f"convaiinnovations/laya ({args.model})",
        "device": args.device,
        "task": args.task,
        "rows": n,
        "options": len(options),
        "instruction": spec["instruction"],
        "accuracy": acc,
        "correct": correct,
        "parse_failures": parse_failures,
        "latency_p50_ms": p50,
        "latency_p90_ms": p90,
        "total_seconds": round(sum(latencies), 1),
        "wrong_examples": wrong,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = RESULTS / f"laya_{args.task}_{n}rows.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
