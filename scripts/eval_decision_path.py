#!/usr/bin/env python3
"""Exercise the decision-time path on the frozen 10k battery with a stub reader.

Path, exactly as `SelectRead.decide` orders it: chunk -> BM25 top-8 -> reader
interface -> candidate scores -> conformal abstention.

The reader and head here are a deterministic lexical stub: BM25 chooses the
evidence, and each alternative is scored by weighted token overlap with the
selected text. The conformal calibrator (`azurite.abstain`) fits a 90% threshold
on a calibration half and the held-out half measures empirical coverage.

This is a control: the stub has no learned parameters, no checkpoint and no
capability claim. It proves the harness, the split and the abstention bookkeeping
work on all 10,000 rows so a real checkpoint can be dropped in unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.abstain import ConformalCalibrator  # noqa: E402
from azurite.battery import read_eval_battery  # noqa: E402
from azurite.selector import BM25ChunkSelector, chunk_text  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def stub_scores(question: str, selected, labels: list[str]) -> list[float]:
    """Deterministic lexical candidate scores over the selected evidence."""
    context_tokens = Counter(tokens(question + "\n" + "\n".join(c.text for c in selected)))
    raw: list[float] = []
    for label in labels:
        lt = Counter(tokens(str(label)))
        if not lt:
            raw.append(0.0)
            continue
        overlap = sum(min(count, context_tokens[tok]) for tok, count in lt.items())
        raw.append(overlap / sum(lt.values()))
    top = max(raw) if raw else 0.0
    exp = [math.exp((value - top) / 0.15) for value in raw]
    total = sum(exp) or 1.0
    return [value / total for value in exp]


def split_of(row_id: str) -> str:
    digest = hashlib.sha256(row_id.encode("utf-8")).digest()[0]
    return "calibration" if digest % 2 == 0 else "heldout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="results/parity/eval_battery_10k")
    ap.add_argument("--out-dir", default="results/parity/decision_path_control")
    ap.add_argument("--coverage", type=float, default=0.90)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    rows, manifest = read_eval_battery(args.battery)
    records: list[dict] = []
    for row in rows:
        chunks = chunk_text(row.context)
        selected = BM25ChunkSelector(chunks).select(row.question, top_k=args.top_k) if chunks else []
        selected = sorted(selected, key=lambda c: c.index)
        scores = stub_scores(row.question, selected, [str(a) for a in row.alternatives])
        choice = max(range(len(scores)), key=lambda i: scores[i]) if scores else -1
        records.append({
            "id": row.id,
            "family": row.slice,
            "split": split_of(row.id),
            "gold": row.gold,
            "choice": choice,
            "correct": choice == row.gold,
            "score": max(scores) if scores else 0.0,
            "selected": [{"index": c.index, "byte_start": c.byte_start, "byte_end": c.byte_end} for c in selected],
            "reader_chars": len(row.question) + sum(len(c.text) for c in selected),
        })

    cal = [r for r in records if r["split"] == "calibration"]
    held = [r for r in records if r["split"] == "heldout"]
    calibrator = ConformalCalibrator.fit(
        [r["score"] for r in cal], [r["correct"] for r in cal], coverage=args.coverage
    )
    for rec in records:
        rec["abstained"] = calibrator.abstain(rec["score"])
        rec["answered"] = not rec["abstained"]

    def family_stats(items: list[dict]) -> dict:
        out: dict[str, dict] = {}
        by: dict[str, list[dict]] = defaultdict(list)
        for r in items:
            by[r["family"]].append(r)
        for family, group in sorted(by.items()):
            answered = [r for r in group if not r["abstained"]]
            unanswerable = [r for r in group if r["family"] == "abstention" and r["gold"] == 1]
            out[family] = {
                "n": len(group),
                "abstained": sum(1 for r in group if r["abstained"]),
                "coverage": len(answered) / len(group),
                "answered_correct": sum(1 for r in answered if r["correct"]),
                "selective_accuracy": (sum(1 for r in answered if r["correct"]) / len(answered)) if answered else None,
                "unanswerable_abstain_recall": (
                    sum(1 for r in unanswerable if r["abstained"]) / len(unanswerable) if unanswerable else None
                ),
            }
        return out

    coverage = sum(1 for r in held if not r["abstained"]) / len(held)
    band = (args.coverage - 0.02, args.coverage + 0.02)
    unanswerable = [r for r in held if r["family"] == "abstention" and r["gold"] == 1]
    recall = sum(1 for r in unanswerable if r["abstained"]) / len(unanswerable) if unanswerable else None

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records_10k.jsonl.gz", "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    summary = {
        "battery": str(args.battery),
        "battery_sha256": manifest.get("battery_sha256"),
        "kind": "control_stub_reader",
        "note": "BM25 selector + lexical stub head + conformal abstention; no learned parameters, not a model result",
        "rows": len(records),
        "calibration": {"n": len(cal), **calibrator.to_dict()},
        "heldout": {
            "n": len(held),
            "coverage": coverage,
            "target": args.coverage,
            "target_band": list(band),
            "within_band": band[0] <= coverage <= band[1],
            "selective_accuracy": (sum(1 for r in held if r["answered"] and r["correct"]) / max(1, sum(1 for r in held if r["answered"]))),
            "unanswerable_abstain_recall": recall,
            "unanswerable_n": len(unanswerable),
        },
        "families_heldout": family_stats(held),
        "abstention_by_family_heldout": {f: v["abstained"] for f, v in family_stats(held).items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (out / "CALIBRATION.json").write_text(json.dumps({
        "calibrator": calibrator.to_dict(),
        "threshold_rule": "nonconformity = 1 - max score; ceil((n+1)*coverage)/n empirical quantile",
        "split_rule": "sha256(row id) first byte parity; disjoint by construction",
        "counts": {"calibration": len(cal), "heldout": len(held),
                   "families_calibration": dict(Counter(r["family"] for r in cal)),
                   "families_heldout": dict(Counter(r["family"] for r in held))},
    }, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(records), "threshold": calibrator.threshold,
        "heldout_coverage": round(coverage, 4), "target": args.coverage,
        "within_band": summary["heldout"]["within_band"],
        "selective_accuracy": round(summary["heldout"]["selective_accuracy"], 4),
        "unanswerable_abstain_recall": None if recall is None else round(recall, 4),
        "family_abstentions": summary["abstention_by_family_heldout"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
