#!/usr/bin/env python3
"""Evaluate the 5,120-byte joint-fine-tuned query head on all 3,080 rows.

Four arms on the same rows, the same gold labels and the same frozen 24 examples:

  old_a0     existing mean-pool head, message only, 1,024 bytes
  old_a2c    existing mean-pool head, complete examples that fit, 1,020 bytes
  len_a0     query head, message only, 5,120 bytes
  len_a2c    query head, all 24 examples at 5,118 bytes  (primary)

`old_a0` is the unchanged protocol control: it fixes the rows, labels, Jev
correctness vector and metric. `old_a2c` is the old-head control at its native
window. `len_a0` is the example control on the new head and is out of
distribution, because the head was trained on example-bearing contexts. The
protocol mechanics live in `azurite.jev`; this file chooses the arms, reuses the
stored old-head records when asked, and writes the record.

    python scripts/query_head_eval_5120.py \
        --old-head results/e4_candidate_sweep/banking77/w77/final.pt \
        --query-head results/jev_query_5120/final.pt \
        --out-dir results/jev_query_5120/eval
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
    "P0 met if point estimate >= 92.40%, with the exact test not rejecting "
    "equality in the losing direction; otherwise P0 missed. "
    "Non-inferiority at -2.0 pp is reported separately."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=ROOT / "results/jev_matched/inputs.json", type=Path)
    ap.add_argument("--old-head", default=ROOT / "results/e4_candidate_sweep/banking77/w77/final.pt", type=Path)
    ap.add_argument("--old-eval-dir", default=ROOT / "results/jev_query/eval", type=Path,
                    help="stored native-window evaluation whose old_a0/old_a2c records are reused")
    ap.add_argument("--reuse-old", action="store_true",
                    help="reuse the unchanged old-head per-example records instead of re-running the head")
    ap.add_argument("--query-head", default=ROOT / "results/jev_query_5120/final.pt", type=Path)
    ap.add_argument("--init-ckpt", default=ROOT / "results/pretrain/l1_large_byte_iso10v3/weights_only.pt", type=Path)
    ap.add_argument("--prereg", default=ROOT / "results/jev_query_5120/PRE_REGISTRATION.md", type=Path)
    ap.add_argument("--prereg-sha", required=True)
    ap.add_argument("--amendment", default=ROOT / "results/jev_query_5120/PRE_REGISTRATION_AMENDMENT_1.md", type=Path)
    ap.add_argument("--amendment-sha", required=True)
    ap.add_argument("--out-dir", default=ROOT / "results/jev_query_5120/eval", type=Path)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    prereg_sha = sha256_file(args.prereg)
    if prereg_sha != args.prereg_sha:
        raise SystemExit(f"pre-registration hash {prereg_sha} != {args.prereg_sha}")
    amendment_sha = sha256_file(args.amendment)
    if amendment_sha != args.amendment_sha:
        raise SystemExit(f"amendment hash {amendment_sha} != {args.amendment_sha}")
    data = jev.load_inputs(args.inputs, require_examples=jev.EXAMPLES)
    rows = data["rows"]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    tok = load_tokenizer("byte")
    _, cand_ids, cand_mask = jev.candidate_bank(data["labels"], tok)

    arms = [
        jev.Arm("old_a0", "dual", 0),
        jev.Arm("old_a2c", "dual", jev.EXAMPLES, jev.NATIVE_BUDGET),
        jev.Arm("len_a0", "query", 0),
        jev.Arm("len_a2c", "query", jev.EXAMPLES, jev.LONG_BUDGET, require_full=True),
    ]
    packed = jev.pack_contexts(rows, arms, tok)
    print(
        "packed contexts: "
        f"old_a0={len(packed.contexts['old_a0'])} "
        f"old_a2c={len(packed.contexts['old_a2c'])} "
        f"len_a0={len(packed.contexts['len_a0'])} "
        f"len_a2c={len(packed.contexts['len_a2c'])} "
        f"old_mean={np.mean(packed.effective['old_a2c']):.1f} "
        f"new_range={min(packed.effective['len_a2c'])}-{max(packed.effective['len_a2c'])}",
        flush=True,
    )
    for name in ("old_a0", "old_a2c", "len_a0", "len_a2c"):
        print(f"  {name} tensor={tuple(packed.ids[name].shape)}", flush=True)

    summaries: dict[str, dict] = {}
    correctness: dict[str, np.ndarray] = {}
    all_records: dict[str, list[dict]] = {}

    if args.reuse_old:
        old_summary = json.loads((args.old_eval_dir / "summary.json").read_text())
        for name in ("old_a0", "old_a2c"):
            path = args.old_eval_dir / f"per_example_{name}.jsonl"
            records = [json.loads(line) for line in path.open()]
            if len(records) != len(rows) or [r["id"] for r in records] != [r["id"] for r in rows]:
                raise SystemExit(f"{path}: rows do not match the official input order")
            for record, context in zip(records, packed.contexts[name]):
                if record["context_sha256"] != jev.sha256_text(context):
                    raise SystemExit(f"{path}: context hash differs from the reused protocol")
            summaries[name] = jev.summarize_records(
                name, records, seconds=0.0, reused_from=str(args.old_eval_dir)
            )
            correctness[name] = np.asarray(
                [row["correct"] for row in records], dtype=np.int8
            )
            all_records[name] = records
            print(
                f"arm {name}: reused {summaries[name]['correct']}/{len(records)} "
                f"= {summaries[name]['accuracy']:.6f}",
                flush=True,
            )
        old_meta = dict(old_summary["heads"]["old"])
        old_sha = old_meta["sha256"]
        print(f"reusing old-head records from {args.old_eval_dir} sha256={old_sha}", flush=True)
    else:
        old, old_state = jev.load_dual_head(args.old_head, device)
        old_sha = sha256_file(args.old_head)
        old_meta = {
            "path": str(args.old_head),
            "sha256": old_sha,
            "steps": old_state.get("steps"),
            "width": old_state.get("width"),
        }
        for name in ("old_a0", "old_a2c"):
            arm = next(a for a in arms if a.name == name)
            summary, records = jev.run_arm(
                old,
                kind="dual",
                arm=arm,
                packed=packed,
                rows=rows,
                cand_ids=cand_ids,
                cand_mask=cand_mask,
                batch=args.batch,
                device=device,
                head_sha=old_sha,
                head_path=str(args.old_head),
            )
            summaries[name] = summary
            correctness[name] = np.asarray(
                [row["correct"] for row in records], dtype=np.int8
            )
            all_records[name] = records

    query, query_state = jev.load_query_head(args.query_head, device, require_max_seq=5_120)
    query_sha = sha256_file(args.query_head)
    init_sha = sha256_file(args.init_ckpt)
    if query_state.get("init_checkpoint_sha256") != init_sha:
        raise SystemExit("query head does not name this 1,024-position init checkpoint")

    for name in ("len_a0", "len_a2c"):
        arm = next(a for a in arms if a.name == name)
        summary, records = jev.run_arm(
            query,
            kind="query",
            arm=arm,
            packed=packed,
            rows=rows,
            cand_ids=cand_ids,
            cand_mask=cand_mask,
            batch=args.batch,
            device=device,
            head_sha=query_sha,
            head_path=str(args.query_head),
        )
        summaries[name] = summary
        correctness[name] = np.asarray([row["correct"] for row in records], dtype=np.int8)
        all_records[name] = records
        print(
            f"arm {name}: {summary['correct']}/{summary['n']} = "
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
            ("len_a2c_vs_old_a2c", "len_a2c", "old_a2c"),
            ("len_a2c_vs_len_a0", "len_a2c", "len_a0"),
            ("len_a2c_vs_old_a0", "len_a2c", "old_a0"),
        ],
    )
    decision = jev.decide(
        primary_arm="len_a2c",
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

    meta = json.loads((ROOT / "results/jev_query_5120/train_meta.json").read_text())
    if meta["head_sha256"] != query_sha:
        raise SystemExit("train_meta head hash does not match evaluated head")
    separation = {
        "train_n": meta["train_n"],
        "train_correct": meta["train_correct"],
        "train_accuracy": meta["train_accuracy"],
        "train_loss": meta["train_loss"],
        "final_training_loss": meta["last_loss"],
        "mean_training_loss_last_50": meta["mean_loss_last_50"],
        "test_n": summaries["len_a2c"]["n"],
        "test_correct": summaries["len_a2c"]["correct"],
        "test_accuracy": summaries["len_a2c"]["accuracy"],
        "test_minus_train": summaries["len_a2c"]["accuracy"] - meta["train_accuracy"],
        "note": (
            "Descriptive separation only. No early stopping, checkpoint selection, "
            "or best-of-steps choice was used; the step-3,000 head is evaluated."
        ),
    }

    summary = {
        "protocol": jev.LONG_PROTOCOL,
        "prereg_sha256": prereg_sha,
        "inputs_sha256": sha256_file(args.inputs),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "heads": {
            "old": {
                **old_meta,
                "records_reused_from": str(args.old_eval_dir) if args.reuse_old else None,
            },
            "query": {
                "path": str(args.query_head),
                "sha256": query_sha,
                "steps": query_state.get("steps"),
                "context_budget": query_state.get("context_budget"),
            },
        },
        "amendment_sha256": amendment_sha,
        "init_checkpoint": {
            "path": str(args.init_ckpt),
            "sha256": init_sha,
            "role": "1,024-position pretraining weights, joint fine-tuned at 5,120",
        },
        "state": {
            "n": len(rows),
            "jev_correct": int(jev_correct.sum()),
            "jev_accuracy": float(jev_correct.mean()),
        },
        "arms": summaries,
        "paired": paired,
        "decision": decision,
        "train_test_separation": separation,
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
        args.amendment,
        args.inputs,
        summary_path,
        args.init_ckpt,
        args.query_head,
        ROOT / "results/jev_query_5120/train_meta.json",
        ROOT / "results/jev_query_5120/train_records.jsonl",
        ROOT / "results/jev_query/support.jsonl",
        ROOT / "results/jev_query/train_records.jsonl",
        Path(__file__).resolve(),
        ROOT / "azurite/query.py",
        ROOT / "azurite/retrieval.py",
        ROOT / "scripts/query_head_train_5120.py",
        ROOT / "azurite/model.py",
        ROOT / "azurite/config.py",
        ROOT / "scripts/transfer_eval.py",
    ] + [args.out_dir / f"per_example_{name}.jsonl" for name in all_records]
    if args.reuse_old:
        manifest_files += [
            args.old_eval_dir / "summary.json",
            args.old_eval_dir / "per_example_old_a0.jsonl",
            args.old_eval_dir / "per_example_old_a2c.jsonl",
        ]
    else:
        manifest_files.append(args.old_head)
    lines = [
        f"{sha256_file(path)}  ./{path.resolve().relative_to(ROOT).as_posix()}"
        for path in manifest_files
    ]
    (args.out_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

    print(f"wrote {summary_path}")
    print(
        f"DECISION {decision['verdict']} (acc {summaries['len_a2c']['accuracy']:.6f}, "
        f"diff {decision['diff_vs_jev']:+.6f}, p {decision['mcnemar_p']:.6g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
