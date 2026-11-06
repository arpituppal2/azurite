#!/usr/bin/env python3
"""Train the candidate-conditioned dual head for Azurite's decision path.

The old script trained only a frozen projection with one fixed batch per task and
no validation. This entry point keeps the same ``AzuriteDual`` head but wires it
to the task-7 controls in :mod:`azurite.decision_tuning`:

* candidate dropout with the gold candidate always retained;
* label paraphrases and BM25-selected hard negatives;
* a validation carve taken only from the source train split;
* reader freeze, warmup-then-unfreeze, or full unfreeze;
* resumable weights/optimizer/RNG state; and
* a held-out gate that requires every family to beat majority and BM25 controls
  and requires a stored evidence-precision measurement.

The final head is written as the existing ``.npz`` format consumed by
``SelectRead.from_checkpoint``. The resumable state is a sibling ``.state.pt``;
when the reader is unfrozen it contains the full dual model, not just the head.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.decision_tuning import (  # noqa: E402
    DEFAULT_TASKS,
    activate_reader,
    build_optimizer,
    evaluate_examples,
    load_train_validation,
    load_training_state,
    save_head_npz,
    save_training_state,
    train_batch,
    validation_gate,
    write_records,
)
from azurite.decisions import TASKS, candidate_bank  # noqa: E402
from azurite.model import AzuriteDual  # noqa: E402
from azurite.select_read import load_reader_checkpoint  # noqa: E402
from azurite.tokenizers import load as load_tokenizer  # noqa: E402


def read_evidence_precision(path: Path | None) -> float | None:
    """Compute precision from stored per-row evidence records.

    A record has ``id``, ``gold_evidence`` as a two-item byte span, and
    ``evidence`` as a list of ``{byte_start, byte_end}`` spans. A row is a hit
    when any predicted span overlaps its gold span. This is the same rule used
    by ``scripts/eval_parity.py``; an empty or malformed file is not accepted as
    a clean measurement.
    """
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"evidence rows not found: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    auditable = 0
    hit = 0
    for row in rows:
        gold = row.get("gold_evidence")
        if not gold or len(gold) != 2:
            continue
        auditable += 1
        gs, ge = int(gold[0]), int(gold[1])
        for span in row.get("evidence") or []:
            ss = int(span.get("byte_start", span.get("start", -1)))
            se = int(span.get("byte_end", span.get("end", -1)))
            if ss < ge and se > gs:
                hit += 1
                break
    if not auditable:
        return None
    return hit / auditable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--out", default="results/parity/decision/head.npz")
    ap.add_argument("--state", default="", help="resume state path; defaults beside --out")
    ap.add_argument("--tokens", type=float, default=1e9)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="accepted for compatibility")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--reader-lr", type=float, default=3e-5)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reader-mode", choices=("freeze", "warmup", "unfreeze"), default="warmup")
    ap.add_argument("--freeze-steps", type=int, default=500)
    ap.add_argument("--candidate-dropout", type=float, default=0.25)
    ap.add_argument("--hard-negatives", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=64)
    ap.add_argument("--paraphrase-prob", type=float, default=0.25)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--examples-per-task", type=int, default=4000)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--evidence-rows", default="",
                    help="per-row evidence JSONL; without it evidence precision is unmeasured")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for decision training; pass --device cpu only for a smoke test")
    device = torch.device(args.device)
    out = Path(args.out)
    state_path = Path(args.state) if args.state else out.with_suffix(".state.pt")
    names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    missing = [name for name in names if name not in TASKS]
    if missing:
        raise SystemExit(f"unknown task(s): {missing}")

    tok = load_tokenizer("bpe")
    reader = load_reader_checkpoint(args.reader, device=str(device))
    dual = AzuriteDual(reader).to(device)
    optimizer = build_optimizer(
        dual, head_lr=args.head_lr, reader_lr=args.reader_lr, reader_mode=args.reader_mode
    )

    data = {}
    for name in names:
        spec = TASKS[name]
        bank = candidate_bank(spec).tokenize(tok)
        train_rows, val_rows = load_train_validation(
            spec, bank,
            limit=args.examples_per_task if args.examples_per_task else None,
            fraction=args.val_fraction, seed=args.seed,
        )
        if not train_rows or not val_rows:
            raise SystemExit(f"{name}: train/validation carve is empty")
        data[name] = (spec, bank, train_rows, val_rows)
        print(f"{name}: train {len(train_rows)} val {len(val_rows)} labels {len(bank)}", flush=True)

    step = 0
    tokens_seen = 0
    rng = random.Random(args.seed)
    if args.resume:
        if not state_path.exists():
            raise SystemExit(f"--resume: no state at {state_path}")
        payload, rng = load_training_state(state_path, dual, optimizer, device=device)
        step = int(payload["step"])
        tokens_seen = int(payload["tokens_seen"])
        print(f"resumed decision state at step {step} / {tokens_seen:,} tokens", flush=True)

    dual.train()
    while tokens_seen < args.tokens:
        progressed = False
        for name in names:
            spec, bank, train_rows, _ = data[name]
            rows = list(train_rows)
            rng.shuffle(rows)
            for start in range(0, len(rows), args.batch):
                if tokens_seen >= args.tokens:
                    break
                batch = rows[start:start + args.batch]
                if not batch:
                    continue
                loss, stats = train_batch(
                    dual, tok, spec, bank, batch,
                    device=device, max_seq=args.max_seq, rng=rng,
                    candidate_dropout=args.candidate_dropout,
                    hard_negatives=args.hard_negatives,
                    max_candidates=args.max_candidates,
                    paraphrase_probability=args.paraphrase_prob,
                )
                if not torch.isfinite(loss):
                    raise SystemExit(f"non-finite decision loss at step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dual.parameters(), 1.0)
                optimizer.step()
                step += 1
                tokens_seen += int(stats["tokens"])
                progressed = True
                if activate_reader(
                    dual, optimizer, step=step, freeze_steps=args.freeze_steps,
                    reader_mode=args.reader_mode, reader_lr=args.reader_lr,
                ):
                    print(f"reader unfrozen at step {step}", flush=True)
                if step % 25 == 0:
                    print(
                        f"step {step} tokens {tokens_seen:,} loss {stats['loss']:.4f} "
                        f"kept {stats['kept_mean']:.1f}/{stats['candidates']}",
                        flush=True,
                    )
                if args.save_every and step % args.save_every == 0:
                    save_training_state(
                        state_path, dual, optimizer,
                        step=step, tokens_seen=tokens_seen, args=vars(args), rng=rng,
                    )
                if tokens_seen >= args.tokens:
                    break
        if not progressed:
            raise SystemExit("no training rows produced a step")

    dual.eval()
    records: list[dict] = []
    for name in names:
        spec, bank, _, val_rows = data[name]
        rows = evaluate_examples(
            dual, tok, spec, bank, val_rows, device=device, max_seq=args.max_seq
        )
        records.extend(rows)
        correct = sum(int(row["choice"] == row["gold"]) for row in rows)
        print(f"validation {name}: {correct}/{len(rows)}", flush=True)

    evidence_precision = read_evidence_precision(
        Path(args.evidence_rows) if args.evidence_rows else None
    )
    gate = validation_gate(records, evidence_precision=evidence_precision)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_records(out.parent / "validation_rows.jsonl", records)
    (out.parent / "VALIDATION.json").write_text(
        json.dumps({
            "n": len(records),
            "gate": gate,
            "tasks": names,
            "reader_mode": args.reader_mode,
            "reader_path": args.reader,
            "tokens_seen": tokens_seen,
            "step": step,
        }, indent=1) + "\n",
        encoding="utf-8",
    )
    save_head_npz(out, dual)
    save_training_state(
        state_path, dual, optimizer,
        step=step, tokens_seen=tokens_seen, args=vars(args), rng=rng,
    )
    print(json.dumps(gate, indent=1), flush=True)
    print(f"wrote {out} and {state_path}", flush=True)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
