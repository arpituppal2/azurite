#!/usr/bin/env python3
"""Evaluate the message-query head on all 3,080 official Banking77 rows.

Four inference-only arms on the same rows and the same frozen 24 examples:

  old_a0    existing mean-pool head, message only, 1,024 bytes
  old_a2c   existing mean-pool head, complete examples that fit, 1,024 bytes
  query_a0  new query-pool head, message only, 1,024 bytes
  query_a2c new query-pool head, complete examples that fit, 1,024 bytes

The primary P0 arm is `query_a2c`. `old_a2c` is the protocol-matched control for
the old head under the new complete-example packing; `query_a0` is the control
for the new head without examples. All protocol mechanics live in
`azurite.jev`; this file only chooses the arms and writes the record.

    python scripts/query_head_eval.py \
        --old-head results/e4_candidate_sweep/banking77/w77/final.pt \
        --query-head results/jev_query/final.pt \
        --out-dir results/jev_query/eval
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite import jev  # noqa: E402
from azurite.artifacts import sha256_file  # noqa: E402
from azurite.tokenizers import load as load_tokenizer  # noqa: E402

RULE = (
    "P0 exceeded if point estimate > 92.40% and exact McNemar p < 0.05; "
    "P0 met if point estimate >= 92.40% and the exact test does not reject "
    "equality in the losing direction; otherwise P0 missed. "
    "Non-inferiority at -2.0 pp is reported separately."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=ROOT / "results/jev_matched/inputs.json", type=Path)
    ap.add_argument("--old-head", default=ROOT / "results/e4_candidate_sweep/banking77/w77/final.pt", type=Path)
    ap.add_argument("--query-head", default=ROOT / "results/jev_query/final.pt", type=Path)
    ap.add_argument("--prereg", default=ROOT / "results/jev_query/PRE_REGISTRATION.md", type=Path)
    ap.add_argument("--prereg-sha", required=True)
    ap.add_argument("--out-dir", default=ROOT / "results/jev_query/eval", type=Path)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    prereg_sha = sha256_file(args.prereg)
    if prereg_sha != args.prereg_sha:
        raise SystemExit(f"pre-registration hash {prereg_sha} != {args.prereg_sha}")
    data = jev.load_inputs(args.inputs)
    rows = data["rows"]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    tok = load_tokenizer("byte")
    _, cand_ids, cand_mask = jev.candidate_bank(data["labels"], tok)

    arms = [
        jev.Arm("old_a0", "dual", 0),
        jev.Arm("old_a2c", "dual", jev.EXAMPLES, jev.NATIVE_BUDGET),
        jev.Arm("query_a0", "query", 0),
        jev.Arm("query_a2c", "query", jev.EXAMPLES, jev.NATIVE_BUDGET),
    ]
    packed = jev.pack_contexts(rows, arms, tok)
    print(
        f"packed contexts: a0={len(packed.contexts['old_a0'])} "
        f"a2c={len(packed.contexts['old_a2c'])} "
        f"mean_examples={np.mean(packed.effective['old_a2c']):.1f} "
        f"range={min(packed.effective['old_a2c'])}-{max(packed.effective['old_a2c'])}",
        flush=True,
    )

    old, old_state = jev.load_dual_head(args.old_head, device)
    query, query_state = jev.load_query_head(args.query_head, device)
    old_sha, query_sha = sha256_file(args.old_head), sha256_file(args.query_head)

    summaries: dict[str, dict] = {}
    correctness: dict[str, np.ndarray] = {}
    all_records: dict[str, list[dict]] = {}
    for kind, model, head_path, head_sha, prefix in (
        ("dual", old, str(args.old_head), old_sha, "old"),
        ("query", query, str(args.query_head), query_sha, "query"),
    ):
        for suffix in ("a0", "a2c"):
            arm = next(a for a in arms if a.name == f"{prefix}_{suffix}")
            print(f"arm {arm.name}", flush=True)
            summary, records = jev.run_arm(
                model,
                kind=kind,
                arm=arm,
                packed=packed,
                rows=rows,
                cand_ids=cand_ids,
                cand_mask=cand_mask,
                batch=args.batch,
                device=device,
                head_sha=head_sha,
                head_path=head_path,
            )
            summaries[arm.name] = summary
            correctness[arm.name] = np.asarray(
                [row["correct"] for row in records], dtype=np.int8
            )
            all_records[arm.name] = records
            print(
                f"arm {arm.name}: {summary['correct']}/{summary['n']} = "
                f"{summary['accuracy']:.6f} [{summary['wilson95'][0]:.4f}, "
                f"{summary['wilson95'][1]:.4f}]",
                flush=True,
            )

    jev_correct = np.asarray([row["jev_correct"] for row in rows], dtype=np.int8)
    if int(jev_correct.sum()) != jev.JEV_CORRECT:
        raise SystemExit("Jev vector mismatch")
    paired = jev.paired_stats(
        correctness,
        jev_correct,
        extra_pairs=[
            ("query_a2c_vs_old_a2c", "query_a2c", "old_a2c"),
            ("query_a2c_vs_query_a0", "query_a2c", "query_a0"),
        ],
    )
    decision = jev.decide(
        primary_arm="query_a2c",
        summaries=summaries,
        paired=paired,
        rule=RULE,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in all_records.items():
        path = args.out_dir / f"per_example_{name}.jsonl"
        with path.open("w") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        summaries[name]["per_example_sha256"] = sha256_file(path)

    summary = {
        "protocol": jev.NATIVE_PROTOCOL,
        "prereg_sha256": prereg_sha,
        "inputs_sha256": sha256_file(args.inputs),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "heads": {
            "old": {
                "path": str(args.old_head),
                "sha256": old_sha,
                "steps": old_state.get("steps"),
                "width": old_state.get("width"),
            },
            "query": {
                "path": str(args.query_head),
                "sha256": query_sha,
                "steps": query_state.get("steps"),
                "context_budget": query_state.get("context_budget"),
            },
        },
        "state": {
            "n": len(rows),
            "jev_correct": int(jev_correct.sum()),
            "jev_accuracy": float(jev_correct.mean()),
        },
        "arms": summaries,
        "paired": paired,
        "decision": decision,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")

    manifest_files = [
        args.prereg,
        args.inputs,
        summary_path,
        ROOT / "results/jev_query/final.pt",
        ROOT / "results/jev_query/train_meta.json",
        ROOT / "results/jev_query/train_records.jsonl",
        ROOT / "results/jev_query/support.jsonl",
        Path(__file__).resolve(),
        ROOT / "azurite/query.py",
        ROOT / "azurite/retrieval.py",
        ROOT / "scripts/query_head_train.py",
    ] + [args.out_dir / f"per_example_{name}.jsonl" for name in all_records]
    lines = [
        f"{sha256_file(path)}  ./{path.resolve().relative_to(ROOT).as_posix()}"
        for path in manifest_files
    ]
    (args.out_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

    print(f"wrote {summary_path}")
    print(
        f"DECISION {decision['verdict']} (acc {summaries['query_a2c']['accuracy']:.6f}, "
        f"diff {decision['diff_vs_jev']:+.6f}, p {decision['mcnemar_p']:.6g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
