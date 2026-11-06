#!/usr/bin/env python3
"""Score the frozen 10k battery and run the internal parity gate.

This script never calls Jev. It writes per-row records, a summary, and the gate
verdict. A live Jev run is a separate script and is refused unless the gate file
says it passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.api import Alternative, DecisionRequest, decide  # noqa: E402
from azurite.battery import read_eval_battery, summarize  # noqa: E402
from azurite.parity import gate, write_json  # noqa: E402
from azurite.select_read import SelectRead  # noqa: E402


def evidence_precision(records: list[dict], rows: dict[str, dict]) -> float | None:
    hits = 0
    total = 0
    for rec in records:
        gold = rows[rec["id"]].get("gold_evidence")
        if not gold:
            continue
        total += 1
        for span in rec.get("evidence", []):
            if not (span["byte_end"] <= gold[0] or span["byte_start"] >= gold[1]):
                hits += 1
                break
    return hits / total if total else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="results/parity/eval_battery_10k")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--engine", choices=["model", "control"], default="model",
                    help="control = majority-class dry run of this harness, never a parity number")
    ap.add_argument("--run-label", default=None,
                    help="label stored in RUN.json; control runs are always labelled DRY_RUN_CONTROL")
    ap.add_argument("--tokenizer", default="data/tokenizers/azurite_bpe_32004.json")
    ap.add_argument("--selector", default=None)
    ap.add_argument("--head", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="results/parity")
    ap.add_argument("--context-metrics", default="results/parity/context_metrics.json")
    ap.add_argument("--m1-profile", default="results/parity/M1_PROFILE.json")
    ap.add_argument("--regression", default="results/parity/regression.json")
    args = ap.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    rows, payload = read_eval_battery(args.battery)
    if args.limit:
        rows = rows[: args.limit]
    if args.engine == "model":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --engine model")
        engine = SelectRead.from_checkpoint(
            args.checkpoint, tokenizer_path=args.tokenizer,
            selector_path=args.selector, head_path=args.head, device=device,
        )
    else:
        from collections import Counter
        majority = {}
        for row in rows:
            majority.setdefault(row.slice, Counter()).update([row.gold])
        majority = {k: v.most_common(1)[0][0] for k, v in majority.items()}
        print(f"control engine: majority class per family, {len(majority)} families", flush=True)
    records = []
    t_all = time.perf_counter()
    for i, row in enumerate(rows):
        if args.engine == "control":
            records.append({
                "id": row.id, "slice": row.slice, "gold": row.gold,
                "choice": majority[row.slice], "abstained": False,
                "abstain_reason": None, "scores": {}, "evidence": [],
                "latency_ms": 0.0,
            })
        else:
            request = DecisionRequest(
                context=row.context, question=row.question,
                alternatives=[Alternative(id=str(j), label=a) for j, a in enumerate(row.alternatives)],
            )
            t0 = time.perf_counter()
            result = decide(request, engine)
            latency = (time.perf_counter() - t0) * 1000.0
            payload_result = result.to_dict()
            records.append({
                "id": row.id, "slice": row.slice, "gold": row.gold,
                "choice": int(payload_result["choice_id"]) if payload_result["choice_id"] is not None else -1,
                "abstained": payload_result["abstained"],
                "abstain_reason": payload_result["abstain_reason"],
                "scores": payload_result["scores"],
                "evidence": payload_result["evidence"],
                "latency_ms": round(latency, 3),
            })
        if (i + 1) % 500 == 0:
            print(f"scored {i + 1}/{len(rows)}", flush=True)
    summary = summarize(records)
    label = "DRY_RUN_CONTROL" if args.engine == "control" else (args.run_label or "MODEL_EVAL")
    summary["engine"] = args.engine
    summary["run_label"] = label
    summary["is_parity_result"] = args.engine != "control"
    by_id = {r.id: r.to_dict() for r in rows}
    ev = evidence_precision(records, by_id)
    context = json.loads(Path(args.context_metrics).read_text()) if Path(args.context_metrics).exists() else None
    m1 = json.loads(Path(args.m1_profile).read_text()) if Path(args.m1_profile).exists() else None
    regression = json.loads(Path(args.regression).read_text()) if Path(args.regression).exists() else None
    selector_recall = context.get("selector_recall") if context else None
    verdict = gate(
        summary, selector_recall=selector_recall, evidence_precision=ev,
        context=context, m1=m1, regression=regression,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records_10k.json"
    records_path.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    write_json(out / "summary.json", summary)
    write_json(out / "PARITY_GATE.json", verdict)
    run = {
        "schema": 1,
        "label": label,
        "is_parity_result": args.engine != "control",
        "engine": args.engine,
        "battery": args.battery,
        "battery_sha256": payload.get("battery_sha256"),
        "battery_rows": len(rows),
        "records": records_path.name,
        "records_count": len(records),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((out / "summary.json").read_bytes()).hexdigest(),
        "gate_sha256": hashlib.sha256((out / "PARITY_GATE.json").read_bytes()).hexdigest(),
        "gate_passed": bool(verdict["passed"]),
        "command": " ".join(sys.argv),
        "note": (
            "Dry run of the scoring harness with a majority-class control. Not a parity "
            "result: no model checkpoint was loaded and no gate was measured."
            if args.engine == "control" else
            "Model evaluation of the frozen battery; this is the parity-candidate run."
        ),
    }
    write_json(out / "RUN.json", run)
    print(json.dumps({"n": len(records), "accuracy": summary["accuracy"],
                      "wilson95": summary["wilson95"], "passed": verdict["passed"],
                      "failures": verdict["failures"], "engine": args.engine,
                      "label": label, "seconds": round(time.perf_counter() - t_all, 1)}, indent=1))
    if args.engine == "control":
        print(f"{label}: the gate verdict above is expected to fail and is not a model result")
        print(f"run record: {out / 'RUN.json'}")
        return 0
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
