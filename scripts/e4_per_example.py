#!/usr/bin/env python3
"""Write and verify every E4 per-example record from the stored result files.

The trainer stores the 2,000 predictions of each arm and a SHA-256 of the gold
label list (`test_gold_sha256`), but not the labels themselves.  The test split
is deterministic: `stratified_examples(spec, bank, "test", 2000, seed + 1)`
returns the same rows in the same order on every machine.  This script
reconstructs that split, refuses to continue unless its JSON-encoded label list
matches the digest in the result record, and pairs each label with the stored
prediction to write `per_example.jsonl`.

That makes every reported accuracy auditable row by row without trusting the
trainer's own correctness bits: the bit vector is checked against the
reconstruction as well.

Usage:
    python3 scripts/e4_per_example.py --root results/e4_candidate_sweep
    python3 scripts/e4_per_example.py --root results/e4_candidate_sweep --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import rung  # noqa: E402
from azurite.decisions import TASKS, candidate_bank, pack_dual, stratified_examples  # noqa: E402
from azurite.tokenizers import load as load_tokenizer  # noqa: E402

WIDTHS = {"banking77": [2, 8, 32, 77], "clinc_oos": [2, 8, 32, 151],
          "massive_intent": [2, 8, 32, 60]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs(rec: dict, gold: list[int]) -> list[dict]:
    pred = rec["predictions"]
    if len(pred) != len(gold) or len(gold) != rec["n"]:
        raise SystemExit(f"length mismatch: {len(pred)} predictions, {len(gold)} gold, "
                         f"{rec['n']} recorded")
    out = []
    for i, (g, p) in enumerate(zip(gold, pred)):
        if not 0 <= p < rec["k"]:
            raise SystemExit(f"prediction {p} outside bank at row {i}")
        out.append({"i": i, "gold": int(g), "pred": int(p),
                    "correct": int(p == g)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/e4_candidate_sweep")
    ap.add_argument("--check", action="store_true",
                    help="verify existing files instead of writing them")
    args = ap.parse_args()
    root = Path(args.root)
    tok = load_tokenizer("byte")
    written = checked = 0

    for task in WIDTHS:
        spec = TASKS[task]
        bank = candidate_bank(spec)
        for width in WIDTHS[task]:
            rp = root / task / f"w{width}" / "result.json"
            if not rp.exists():
                continue
            rec = json.loads(rp.read_text())
            if rec["protocol"] != "E4-width-training" or rec["width"] != width:
                raise SystemExit(f"{rp}: wrong protocol or width")
            cfg = rung(rec["rung"], "byte")
            rows = stratified_examples(spec, bank, "test", rec["n"], rec["seed"] + 1)
            batch, stats = pack_dual(tok, spec, bank, rows, max_seq=cfg.max_seq)
            if stats["unpackable"]:
                raise SystemExit(f"{task} w{width}: packing discarded rows")
            gold = batch.gold.tolist()
            digest = hashlib.sha256(json.dumps(gold).encode()).hexdigest()
            if digest != rec["test_gold_sha256"]:
                raise SystemExit(f"{task} w{width}: reconstructed gold does not match "
                                 f"record digest {rec['test_gold_sha256']}")
            rows_out = pairs(rec, gold)
            correct = sum(r["correct"] for r in rows_out)
            if correct != rec["correct"]:
                raise SystemExit(f"{task} w{width}: per-example correctness sums to "
                                 f"{correct}, record says {rec['correct']}")

            target = rp.parent / "per_example.jsonl"
            payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows_out)
            if args.check:
                if not target.exists() or target.read_text() != payload:
                    raise SystemExit(f"{task} w{width}: per_example.jsonl missing or stale")
                checked += 1
            else:
                target.write_text(payload)
                written += 1
            print(f"{task:15s} w{width:<3d} n={len(rows_out)} correct={correct} "
                  f"gold={digest[:12]} per_example={target.name}")

    print(f"{'verified' if args.check else 'wrote'} {checked or written} per-example files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
