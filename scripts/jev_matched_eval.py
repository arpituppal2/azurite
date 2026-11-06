#!/usr/bin/env python3
"""Protocol-matched Banking77 evaluation of the existing 113.47M head.

Runs only inference. The primary arm gives the model the exact 24 labelled
examples that Jev received for each official test row, at a 5,120-byte window;
two diagnostic arms isolate the row-set change and the context-length confound.
The protocol, row set, head and decision rule are fixed in
results/jev_matched/PRE_REGISTRATION.md before this script is run.

Usage on the Molab sandbox only:
    python scripts/jev_matched_eval.py \
        --inputs results/jev_matched/inputs.json \
        --head results/e4_candidate_sweep/banking77/w77/final.pt \
        --prereg results/jev_matched/PRE_REGISTRATION.md \
        --prereg-sha <hash> \
        --out-dir results/jev_matched

A smoke run must use a different --out-dir and --limit; it is never reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from azurite.config import EncoderConfig  # noqa: E402
from azurite.decisions import (  # noqa: E402
    Example,
    TASKS,
    CandidateBank,
    _pad,
    pack_dual,
)
from azurite.model import Azurite, AzuriteDual  # noqa: E402
from azurite.tokenizers import load as load_tokenizer  # noqa: E402
from e4_summary import bootstrap_lower, mcnemar_exact, wilson  # noqa: E402

PROTOCOL = "jev-matched-banking77-eval-v1"
PRIMARY = "examples24_5120"
MARGIN = -0.02
BOOT_SEED = 0
ARMS = {
    "no_examples_1024": {"max_seq": 1024, "examples": False, "truncate": False},
    "examples24_5120": {"max_seq": 5120, "examples": True, "truncate": False},
    "examples24_1024": {"max_seq": 1024, "examples": True, "truncate": True},
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def context_for(row: dict, use_examples: bool) -> str:
    if not use_examples:
        return row["text"]
    lines = ["Customer message: " + row["text"], "Labeled examples:"]
    for j, example in enumerate(row["examples"], 1):
        lines.append(f"{j}. {example['message']} => {example['intent']}")
    return "\n".join(lines)


def effective_examples(context: str, max_seq: int) -> int:
    prefix = context.encode("utf-8", errors="ignore")[: max(0, max_seq - 2)]
    return prefix.decode("utf-8", errors="ignore").count(" => ")


def load_head(path: Path, device: torch.device) -> tuple[AzuriteDual, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("task") != "banking77" or state.get("path") != "dual":
        raise SystemExit(f"{path}: not a Banking77 dual head")
    if int(state.get("width", -1)) != 77:
        raise SystemExit(f"{path}: width {state.get('width')} is not the full 77-bank")
    cfg = EncoderConfig(**state["config"])
    encoder = Azurite(cfg)
    model = AzuriteDual(encoder)
    model.load_state_dict(state["model"], strict=True)
    model.eval().to(device)
    return model, state


def run_arm(
    model: AzuriteDual,
    tok,
    bank: CandidateBank,
    spec,
    rows: list[dict],
    *,
    name: str,
    max_seq: int,
    use_examples: bool,
    truncate: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    cand_ids, cand_mask = _pad(bank.token_ids)
    cand_ids = cand_ids.to(device)
    cand_mask = cand_mask.to(device)
    # no_grad, not inference_mode: the RoPE tables are cached lazily on first
    # use, and tables created inside inference_mode poison every later arm.
    with torch.no_grad():
        cand_emb = model.embed(cand_ids, cand_mask)
    scale = float(model.log_scale.exp().item())

    records: list[dict] = []
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        contexts = [context_for(row, use_examples) for row in chunk]
        examples = [
            Example(index=row["i"], context=context, gold=row["gold_index"],
                    parts=(context,))
            for row, context in zip(chunk, contexts)
        ]
        if not truncate:
            too_long = [
                (row["id"], len(context.encode()))
                for row, context in zip(chunk, contexts)
                if len(context.encode()) + 2 > max_seq
            ]
            if too_long:
                raise SystemExit(f"{name}: context exceeds max_seq: {too_long[:3]}")
        batch, _stats = pack_dual(tok, spec, bank, examples, max_seq=max_seq)
        ids = batch.context_ids.to(device)
        mask = batch.context_mask.to(device)
        with torch.no_grad():
            ctx = model.embed(ids, mask)
            logits = scale * (ctx @ cand_emb.T)
            preds = logits.argmax(-1).tolist()
        for row, context, pred in zip(chunk, contexts, preds):
            pred = int(pred)
            gold = int(row["gold_index"])
            records.append({
                "arm": name,
                "id": row["id"],
                "i": int(row["i"]),
                "text_sha256": sha256_bytes(row["text"].encode()),
                "context_sha256": sha256_bytes(context.encode()),
                "context_bytes": len(context.encode()),
                "effective_examples": effective_examples(context, max_seq),
                "max_seq": max_seq,
                "gold": row["gold"],
                "gold_index": gold,
                "pred": bank.labels[pred],
                "pred_index": pred,
                "correct": int(pred == gold),
            })
        if start == 0 or (start // batch_size) % 100 == 0:
            done = len(records)
            print(f"  {name}: {done}/{len(rows)} rows "
                  f"({time.perf_counter() - started:.1f}s)", flush=True)

    seconds = time.perf_counter() - started
    correct = np.array([r["correct"] for r in records], dtype=np.int8)
    k = int(correct.sum())
    lo, hi = wilson(k, len(records))
    summary = {
        "arm": name,
        "n": len(records),
        "correct": k,
        "accuracy": k / len(records),
        "wilson95": [lo, hi],
        "seconds": seconds,
        "max_seq": max_seq,
        "use_examples": use_examples,
        "truncate": truncate,
        "mean_context_bytes": float(np.mean([r["context_bytes"] for r in records])),
        "mean_effective_examples": float(np.mean([r["effective_examples"] for r in records])),
        "min_effective_examples": int(min(r["effective_examples"] for r in records)),
        "max_effective_examples": int(max(r["effective_examples"] for r in records)),
    }
    return summary, records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=ROOT / "results/jev_matched/inputs.json", type=Path)
    ap.add_argument("--head", default=ROOT / "results/e4_candidate_sweep/banking77/w77/final.pt", type=Path)
    ap.add_argument("--prereg", default=ROOT / "results/jev_matched/PRE_REGISTRATION.md", type=Path)
    ap.add_argument("--prereg-sha", required=True)
    ap.add_argument("--out-dir", default=ROOT / "results/jev_matched", type=Path)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke only; never a reported result")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    prereg_sha = sha256_file(args.prereg)
    if prereg_sha != args.prereg_sha:
        raise SystemExit(f"pre-registration hash {prereg_sha} != --prereg-sha {args.prereg_sha}")
    inputs = json.loads(args.inputs.read_text())
    if inputs.get("protocol") != "jev-matched-banking77-inputs-v1":
        raise SystemExit("unexpected inputs protocol")
    rows = inputs["rows"]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no rows")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    tok = load_tokenizer("byte")
    labels = inputs["labels"]
    bank = CandidateBank(task="banking77", labels=labels, raw_ids=list(range(len(labels))))
    bank.tokenize(tok)
    spec = TASKS["banking77"]
    model, head_state = load_head(args.head, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")

    summaries: dict[str, dict] = {}
    correctness: dict[str, np.ndarray] = {}
    print(f"protocol={PROTOCOL} prereg={prereg_sha} inputs={sha256_file(args.inputs)}", flush=True)
    print(f"head={args.head} sha256={sha256_file(args.head)} device={device}", flush=True)
    for name in arm_names:
        spec_arm = ARMS[name]
        print(f"arm {name}: max_seq={spec_arm['max_seq']} examples={spec_arm['examples']}", flush=True)
        summary, records = run_arm(
            model, tok, bank, spec, rows,
            name=name, max_seq=spec_arm["max_seq"],
            use_examples=spec_arm["examples"], truncate=spec_arm["truncate"],
            batch_size=args.batch_size, device=device,
        )
        pe_path = args.out_dir / f"per_example_{name}.jsonl"
        with pe_path.open("w") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        summary["per_example_sha256"] = sha256_file(pe_path)
        summaries[name] = summary
        correctness[name] = np.array([r["correct"] for r in records], dtype=np.int8)
        print(f"arm {name}: {summary['correct']}/{summary['n']} = "
              f"{summary['accuracy']:.6f} [{summary['wilson95'][0]:.4f}, "
              f"{summary['wilson95'][1]:.4f}]", flush=True)

    jev = np.array([row["jev_correct"] for row in rows], dtype=np.int8)
    jev_correct = int(jev.sum())
    if not args.limit and jev_correct != int(inputs["jev_correct"]):
        raise SystemExit("Jev correctness vector does not match the input record")
    paired = {}
    for name in arm_names:
        a = correctness[name]
        m = mcnemar_exact(a, jev)
        lower = bootstrap_lower(a, jev, seed=BOOT_SEED)
        paired[name] = {
            "diff_vs_jev": float(a.mean() - jev.mean()),
            "mcnemar": m,
            "bootstrap_lower_95": lower,
        }

    primary = summaries.get(PRIMARY)
    decision = None
    if primary is not None:
        p = paired[PRIMARY]
        noninferior = bool(p["bootstrap_lower_95"] > MARGIN)
        superior = bool(p["diff_vs_jev"] >= 0 and p["mcnemar"]["p"] < 0.05)
        if superior:
            verdict = "superior"
        elif noninferior:
            verdict = "non-inferior"
        else:
            verdict = "gap survives"
        decision = {
            "primary_arm": PRIMARY,
            "margin_points": MARGIN * 100,
            "noninferior": noninferior,
            "superior": superior,
            "verdict": verdict,
            "rule": "lower bound of paired 95% bootstrap > -2.0 pp; superiority requires diff >= 0 and exact McNemar p < 0.05",
        }

    summary = {
        "protocol": PROTOCOL,
        "smoke": bool(args.limit),
        "prereg_sha256": prereg_sha,
        "inputs_sha256": sha256_file(args.inputs),
        "head": {"path": str(args.head), "sha256": sha256_file(args.head),
                 "task": head_state.get("task"), "width": head_state.get("width"),
                 "steps": head_state.get("steps")},
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "state": {"n": len(rows), "jev_correct": jev_correct,
                  "jev_accuracy": jev_correct / len(rows)},
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
    summary_path = args.out_dir / ("smoke_summary.json" if args.limit else "summary.json")
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")

    manifest_path = args.out_dir / ("smoke_MANIFEST.sha256" if args.limit else "MANIFEST.sha256")
    files = [args.inputs, args.prereg, summary_path]
    files += [args.out_dir / f"per_example_{name}.jsonl" for name in arm_names]
    lines = []
    for path in files:
        lines.append(f"{sha256_file(path)}  ./{path.relative_to(args.out_dir.parent)}")
    manifest_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {manifest_path}", flush=True)
    if decision:
        print(f"DECISION {decision['verdict']} "
              f"(diff {paired[PRIMARY]['diff_vs_jev']:+.6f}, "
              f"lower {paired[PRIMARY]['bootstrap_lower_95']:+.6f}, "
              f"p {paired[PRIMARY]['mcnemar']['p']:.6g})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
