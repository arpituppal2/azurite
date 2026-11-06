#!/usr/bin/env python3
"""Train the query-anchored Banking77 head at 5,120 bytes in one joint fine-tune.

One arm, fixed by `results/jev_query_5120/PRE_REGISTRATION.md` as amended by
`PRE_REGISTRATION_AMENDMENT_1.md`: the 1,220-step masked-LM length adaptation is
dropped and the encoder is fine-tuned jointly with the query head at 5,120
positions for 3,000 steps. Both hashes are verified before training starts.

* source encoder: the 1,024-position 113.47M pretraining checkpoint, weights
  only (`results/pretrain/l1_large_byte_iso10v3/weights_only.pt`, SHA-256
  `5294fd6856152c67515b3e0d5a0062adc0b3d7b1a0800a627a046881dffd9135`);
* the exact 4,000 class-balanced rows and the exact 24 BM25 example ids stored in
  `results/jev_query/train_records.jsonl`; no retrieval is redone;
* context budget 5,118 bytes, complete examples only; the pre-registered
  projection puts all 24 examples on every row;
* 3,000 steps, batch 8, AdamW LR 1e-4, weight decay 0.01, clip 1.0, seed 0;
* the full model is fine-tuned, as in the native-window head.

The step-3,000 head is the one evaluated. Train accuracy on all 4,000 rows is
measured and stored so interpolation is visible, not hidden behind a test number.

    python scripts/query_head_train_5120.py \
        --ckpt results/pretrain/l1_large_byte_iso10v3/weights_only.pt \
        --out-dir results/jev_query_5120 --prereg-sha <sha> --amendment-sha <sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from azurite.artifacts import sha256_file, sha256_text  # noqa: E402
from azurite.config import EncoderConfig, rung  # noqa: E402
from azurite.decisions import TASKS, _load, _pad, candidate_bank  # noqa: E402
from azurite.model import Azurite  # noqa: E402
from azurite.query import QueryDual  # noqa: E402
from azurite.retrieval import build_context  # noqa: E402
from azurite.tokenizers import load as load_tokenizer  # noqa: E402
from transfer_eval import batch_schedule  # noqa: E402

CLIP = 1.0
WEIGHT_DECAY = 0.01
CONTEXT_BUDGET = 5_118
EXAMPLES = 24
STEPS = 3_000
BATCH = 8
LR = 1e-4
SEED = 0
SAVE_EVERY = 250


def support_rows(spec, bank) -> list[dict]:
    rows = []
    for i, row in enumerate(_load(spec, "train")):
        text = spec.context(row)
        if not text.strip():
            continue
        rows.append({
            "id": i,
            "text": text,
            "message": text,
            "label": bank.labels[bank.resolve(row[spec.label_field])],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=ROOT / "results/pretrain/l1_large_byte_iso10v3/weights_only.pt", type=Path)
    ap.add_argument("--out-dir", default=ROOT / "results/jev_query_5120", type=Path)
    ap.add_argument("--old-records", default=ROOT / "results/jev_query/train_records.jsonl", type=Path)
    ap.add_argument("--support", default=ROOT / "results/jev_query/support.jsonl", type=Path)
    ap.add_argument("--prereg", default=ROOT / "results/jev_query_5120/PRE_REGISTRATION.md", type=Path)
    ap.add_argument("--prereg-sha", required=True)
    ap.add_argument("--amendment", default=ROOT / "results/jev_query_5120/PRE_REGISTRATION_AMENDMENT_1.md", type=Path)
    ap.add_argument("--amendment-sha", required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--resume", action="store_true",
                    help="continue from train_state.pt; the schedule and total step count are unchanged")
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY)
    ap.add_argument("--smoke", action="store_true", help="20 steps, /tmp output, never a reported result")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 20
        args.out_dir = Path("/tmp/query_head_5120_smoke")
    if not args.smoke and (args.steps, args.batch, args.lr, args.seed) != (STEPS, BATCH, LR, SEED):
        raise SystemExit("steps/batch/LR/seed are fixed by the pre-registration; do not change them")

    prereg_sha = sha256_file(args.prereg)
    if prereg_sha != args.prereg_sha:
        raise SystemExit(f"pre-registration hash {prereg_sha} != {args.prereg_sha}")
    amendment_sha = sha256_file(args.amendment)
    if amendment_sha != args.amendment_sha:
        raise SystemExit(f"amendment hash {amendment_sha} != {args.amendment_sha}")
    if (args.out_dir / "final.pt").exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {args.out_dir/'final.pt'}")
    if args.resume and not (args.out_dir / "train_state.pt").exists():
        raise SystemExit(f"--resume needs {args.out_dir/'train_state.pt'}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    tok = load_tokenizer("byte")
    spec = TASKS["banking77"]
    bank = candidate_bank(spec)
    bank.tokenize(tok)
    cand_ids, cand_mask = _pad(bank.token_ids)
    cand_ids = cand_ids.to(device)
    cand_mask = cand_mask.to(device)

    old_records = [json.loads(line) for line in args.old_records.read_text().splitlines()]
    if len(old_records) != 4_000:
        raise SystemExit(f"expected 4,000 source records, got {len(old_records)}")
    support = [json.loads(line) for line in args.support.read_text().splitlines()]
    support_by_id = {int(row["id"]): row for row in support}
    train_ds = _load(spec, "train")
    print(f"rows={len(old_records)} support={len(support)} BM25 records reused", flush=True)

    contexts: list[str] = []
    messages: list[str] = []
    records: list[dict] = []
    effective: list[int] = []
    gold: list[int] = []
    t_ret = time.perf_counter()
    for n, old in enumerate(old_records, 1):
        row_index = int(old["id"])
        row = train_ds[row_index]
        message = spec.context(row)
        gold_label = bank.labels[bank.resolve(row[spec.label_field])]
        if gold_label != old["gold"]:
            raise SystemExit(f"row {row_index}: gold {gold_label} != source record {old['gold']}")
        if sha256_text(message) != old["message_sha256"]:
            raise SystemExit(f"row {row_index}: message hash != source record")
        chosen = [support_by_id[int(e)] for e in old["examples"]]
        if [c["id"] for c in chosen] != [int(e) for e in old["examples"]]:
            raise SystemExit(f"row {row_index}: example ids do not resolve identically")
        context, used = build_context(message, chosen, budget=CONTEXT_BUDGET)
        if used != EXAMPLES:
            raise SystemExit(f"row {row_index}: {used} examples fit, expected {EXAMPLES}")
        contexts.append(context)
        messages.append(message)
        effective.append(used)
        gold.append(bank.resolve(row[spec.label_field]))
        records.append({
            "id": row_index,
            "gold": gold_label,
            "gold_index": bank.resolve(row[spec.label_field]),
            "examples": [int(e) for e in old["examples"]],
            "source_examples_sha256": hashlib.sha256(
                json.dumps(old["examples"], separators=(",", ":")).encode()
            ).hexdigest(),
            "effective_examples": used,
            "context_bytes": len(context.encode("utf-8")),
            "context_sha256": sha256_text(context),
            "message_sha256": sha256_text(message),
        })
        if n % 500 == 0:
            print(f"  packing {n}/{len(old_records)} ({time.perf_counter()-t_ret:.1f}s)", flush=True)
    packing_seconds = time.perf_counter() - t_ret

    ctx_ids, ctx_mask = _pad([tok.encode(t, add_specials=True) for t in contexts])
    msg_ids, msg_mask = _pad([tok.encode(t, add_specials=True) for t in messages])
    gold_t = torch.tensor(gold, dtype=torch.long)
    print(f"ctx={tuple(ctx_ids.shape)} msg={tuple(msg_ids.shape)} "
          f"mean_examples={sum(effective)/len(effective):.2f} range={min(effective)}-{max(effective)}", flush=True)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    source_cfg = EncoderConfig(**state["config"])
    if source_cfg.max_seq != 1_024:
        raise SystemExit(f"source must be the 1,024-position checkpoint, got {source_cfg.max_seq}")
    cfg = rung("large", "byte", max_seq=5_120)
    encoder = Azurite(cfg)
    # RoPE tables are non-persistent, so a 1,024-position state loads strictly
    # into the 5,120-position model; the buffer is grown at forward time.
    encoder.load_state_dict(state["model"], strict=True)
    init_sha = sha256_file(args.ckpt)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = QueryDual(encoder).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    schedule = list(batch_schedule(len(records), args.batch, args.steps, args.seed))
    print(f"steps={len(schedule)} batch={args.batch} lr={args.lr} "
          f"params={sum(p.numel() for p in params):,} device={device}", flush=True)

    t0 = time.perf_counter()
    losses: list[float] = []
    start_step = 0
    elapsed_before = 0.0
    if args.resume:
        saved = torch.load(args.out_dir / "train_state.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        start_step = int(saved["step"])
        losses = list(saved.get("losses", []))
        elapsed_before = float(saved.get("elapsed", 0.0))
        torch.set_rng_state(saved["torch_cpu"].cpu())
        if device.type == "cuda" and saved.get("torch_cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["torch_cuda"]])
        print(f"resumed at step {start_step}/{len(schedule)} after {elapsed_before/3600:.2f} h", flush=True)
    log_path = args.out_dir / "train.jsonl"
    log = log_path.open("a" if args.resume else "w", encoding="utf-8")
    for step in range(start_step, len(schedule)):
        idx_t = schedule[step]
        model.train()
        opt.zero_grad(set_to_none=True)
        idx = idx_t.tolist()
        ctx = model.embed_context(
            ctx_ids[idx].to(device), ctx_mask[idx].to(device),
            msg_ids[idx].to(device), msg_mask[idx].to(device),
        )
        cand = model.embed_candidates(cand_ids, cand_mask)
        logits = model.log_scale.exp() * (ctx @ cand.T)
        loss = F.cross_entropy(logits, gold_t[idx].to(device))
        value = float(loss.detach())
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        losses.append(value)
        if step % 25 == 0 or step == len(schedule) - 1:
            rate = (time.perf_counter() - t0) / (step + 1 - start_step)
            entry = {
                "step": step,
                "loss": value,
                "mean_loss_25": sum(losses[-25:]) / len(losses[-25:]),
                "grad_norm": float(grad_norm),
                "ms_per_step": 1000 * rate,
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(f"  step {step:>4}/{len(schedule)} loss {value:.4f} mean25 {entry['mean_loss_25']:.4f} "
                  f"{rate*1000:.0f} ms/step", flush=True)
        if (step + 1) % args.save_every == 0 and step + 1 < len(schedule):
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step + 1,
                "losses": losses,
                "elapsed": elapsed_before + (time.perf_counter() - t0),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()] if device.type == "cuda" else None,
                "config": cfg.to_dict(),
                "seed": args.seed,
                "context_budget": CONTEXT_BUDGET,
            }, args.out_dir / "train_state.pt")
            (args.out_dir / "train_state.json").write_text(json.dumps({
                "step": step + 1,
                "sha256": sha256_file(args.out_dir / "train_state.pt"),
                "elapsed": elapsed_before + (time.perf_counter() - t0),
            }, indent=1) + "\n")
            print(f"  state step {step + 1}/{len(schedule)}", flush=True)
    log.close()
    train_seconds = elapsed_before + (time.perf_counter() - t0)

    model.eval()
    correct = 0
    train_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(records), args.batch):
            sl = slice(start, start + args.batch)
            ctx = model.embed_context(
                ctx_ids[sl].to(device), ctx_mask[sl].to(device),
                msg_ids[sl].to(device), msg_mask[sl].to(device),
            )
            cand = model.embed_candidates(cand_ids, cand_mask)
            logits = model.log_scale.exp() * (ctx @ cand.T)
            train_loss += float(F.cross_entropy(logits, gold_t[sl].to(device))) * (sl.stop - sl.start)
            correct += int((logits.argmax(-1) == gold_t[sl].to(device)).sum())
    train_accuracy = correct / len(records)
    train_loss /= len(records)

    head_path = args.out_dir / "final.pt"
    torch.save({
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "path": "query",
        "task": "banking77",
        "seed": args.seed,
        "init": "pretrained_joint5120",
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "context_budget": CONTEXT_BUDGET,
        "init_checkpoint_sha256": init_sha,
        "prereg_sha256": prereg_sha,
        "amendment_sha256": amendment_sha,
        "train_accuracy": train_accuracy,
        "train_correct": correct,
        "train_n": len(records),
        "train_loss": train_loss,
        "final_loss": losses[-1],
        "mean_loss_last_50": sum(losses[-50:]) / len(losses[-50:]),
    }, head_path)

    records_path = args.out_dir / "train_records.jsonl"
    with records_path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    meta = {
        "protocol": "jev-query-head-5120-joint-train-v2",
        "prereg_sha256": prereg_sha,
        "amendment_sha256": amendment_sha,
        "schedule": "joint 5,120 fine-tune from the 1,024 checkpoint; 3,000 steps, exact resume at 250-step boundaries",
        "task": "banking77",
        "seed": args.seed,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "clip": CLIP,
        "examples": len(records),
        "support": len(support),
        "source_records": {"path": str(args.old_records), "sha256": sha256_file(args.old_records)},
        "source_examples": "reused verbatim from results/jev_query/train_records.jsonl; no retrieval",
        "context_budget": CONTEXT_BUDGET,
        "mean_effective_examples": sum(effective) / len(effective),
        "min_effective_examples": min(effective),
        "max_effective_examples": max(effective),
        "mean_context_bytes": sum(r["context_bytes"] for r in records) / len(records),
        "max_context_bytes": max(r["context_bytes"] for r in records),
        "packing_seconds": packing_seconds,
        "train_seconds": train_seconds,
        "ms_per_step": 1000 * train_seconds / max(1, len(schedule)),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_loss_last_50": sum(losses[-50:]) / len(losses[-50:]),
        "train_accuracy": train_accuracy,
        "train_correct": correct,
        "train_n": len(records),
        "train_loss": train_loss,
        "ckpt": str(args.ckpt),
        "init_checkpoint_sha256": init_sha,
        "head_sha256": sha256_file(head_path),
        "train_records_sha256": sha256_file(records_path),
        "support_sha256": sha256_file(args.support),
        "code": {
            "scripts/query_head_train_5120.py": sha256_file(Path(__file__).resolve()),
            "azurite/query.py": sha256_file(ROOT / "azurite/query.py"),
            "azurite/retrieval.py": sha256_file(ROOT / "azurite/retrieval.py"),
            "azurite/model.py": sha256_file(ROOT / "azurite/model.py"),
            "azurite/config.py": sha256_file(ROOT / "azurite/config.py"),
            "scripts/transfer_eval.py": sha256_file(ROOT / "scripts/transfer_eval.py"),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"wrote {head_path} sha256={meta['head_sha256']}")
    print(f"train_accuracy={train_accuracy:.6f} train_loss={train_loss:.6f} "
          f"last_loss={losses[-1]:.6f} ms/step={meta['ms_per_step']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
