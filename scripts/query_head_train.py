#!/usr/bin/env python3
"""Train the message-query attention-pooled Banking77 head on Molab.

One arm, fixed by `results/jev_query/PRE_REGISTRATION.md`:

* 4,000 class-balanced Banking77 training rows, seed 0, the same draw the
  suite/E4 heads used;
* 24 BM25 examples per row (words + adjacent pairs, at most four per label),
  self excluded;
* complete examples only, to the native 1,024-byte window;
* 3,000 steps, batch 32, AdamW LR 1e-4, weight decay 0.01, clip 1.0, seed 0;
* encoder initialised from the 113.47M pretrained checkpoint, new query/proj
  head randomly initialised under the same seed.

Inference-only evaluation is `scripts/query_head_eval.py`.

    python scripts/query_head_train.py \
        --ckpt results/pretrain/l1_large_byte_iso10v3/ckpt.pt \
        --out-dir results/jev_query
"""

from __future__ import annotations

import argparse
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
from azurite.config import EncoderConfig  # noqa: E402
from azurite.decisions import TASKS, _load, _pad, candidate_bank, stratified_examples  # noqa: E402
from azurite.model import Azurite  # noqa: E402
from azurite.query import QueryDual  # noqa: E402
from azurite.retrieval import BM25, build_context, select_examples  # noqa: E402
from azurite.tokenizers import load as load_tokenizer  # noqa: E402
from transfer_eval import batch_schedule  # noqa: E402

CLIP = 1.0
WEIGHT_DECAY = 0.01
CONTEXT_BUDGET = 1_020  # + CLS/SEP = 1,022 bytes, inside the 1,024 window
EXAMPLES = 24
PER_LABEL = 4


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
    ap.add_argument("--ckpt", default=ROOT / "results/pretrain/l1_large_byte_iso10v3/ckpt.pt", type=Path)
    ap.add_argument("--out-dir", default=ROOT / "results/jev_query", type=Path)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=4000)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--smoke", action="store_true", help="50 steps; never a reported result")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 50
        args.examples = 200

    if (args.out_dir / "final.pt").exists():
        raise SystemExit(f"refusing to overwrite {args.out_dir/'final.pt'}")
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

    train_rows = stratified_examples(spec, bank, "train", args.examples, args.seed)
    support = support_rows(spec, bank)
    retriever = BM25(support)
    print(f"train rows={len(train_rows)} support={len(support)} BM25 built", flush=True)

    contexts: list[str] = []
    messages: list[str] = []
    records: list[dict] = []
    effective: list[int] = []
    t_ret = time.perf_counter()
    for n, example in enumerate(train_rows, 1):
        chosen = select_examples(
            retriever, example.context, exclude_id=example.index,
            limit=EXAMPLES, per_label=PER_LABEL,
        )
        context, used = build_context(example.context, chosen, budget=CONTEXT_BUDGET)
        messages.append(example.context)
        contexts.append(context)
        effective.append(used)
        records.append({
            "id": example.index,
            "gold": bank.labels[example.gold],
            "examples": [row["id"] for row in chosen],
            "effective_examples": used,
            "context_bytes": len(context.encode("utf-8")),
            "context_sha256": sha256_text(context),
            "message_sha256": sha256_text(example.context),
        })
        if n % 500 == 0:
            print(f"  retrieval {n}/{len(train_rows)} ({time.perf_counter()-t_ret:.1f}s)", flush=True)
    retrieval_seconds = time.perf_counter() - t_ret

    ctx_ids, ctx_mask = _pad([tok.encode(t, add_specials=True) for t in contexts])
    msg_ids, msg_mask = _pad([tok.encode(t, add_specials=True) for t in messages])
    gold = torch.tensor([e.gold for e in train_rows], dtype=torch.long)
    print(f"tensors ctx={tuple(ctx_ids.shape)} msg={tuple(msg_ids.shape)} "
          f"mean_examples={sum(effective)/len(effective):.1f} "
          f"range={min(effective)}-{max(effective)}", flush=True)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**state["config"])
    if cfg.max_seq != 1024:
        raise SystemExit(f"checkpoint max_seq {cfg.max_seq} is not 1024")
    encoder = Azurite(cfg)
    encoder.load_state_dict(state["model"], strict=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = QueryDual(encoder).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    schedule = list(batch_schedule(len(train_rows), args.batch, args.steps, args.seed))
    print(f"steps={len(schedule)} batch={args.batch} lr={args.lr} "
          f"params={sum(p.numel() for p in params):,} device={device}", flush=True)

    t0 = time.perf_counter()
    losses: list[float] = []
    for step, idx_t in enumerate(schedule):
        model.train()
        opt.zero_grad(set_to_none=True)
        idx = idx_t.tolist()
        ctx = model.embed_context(
            ctx_ids[idx].to(device), ctx_mask[idx].to(device),
            msg_ids[idx].to(device), msg_mask[idx].to(device),
        )
        cand = model.embed_candidates(cand_ids, cand_mask)
        logits = model.log_scale.exp() * (ctx @ cand.T)
        loss = F.cross_entropy(logits, gold[idx].to(device))
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == len(schedule) - 1:
            rate = (time.perf_counter() - t0) / (step + 1)
            print(f"  step {step:>4}/{len(schedule)} loss {float(loss.detach()):.4f} "
                  f"{rate*1000:.0f} ms/step", flush=True)

    train_seconds = time.perf_counter() - t0
    head_path = args.out_dir / "final.pt"
    torch.save({
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "path": "query",
        "task": "banking77",
        "seed": args.seed,
        "init": "pretrained",
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "context_budget": CONTEXT_BUDGET,
    }, head_path)

    meta = {
        "protocol": "jev-query-head-train-v1",
        "task": "banking77",
        "seed": args.seed,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "clip": CLIP,
        "examples": len(train_rows),
        "support": len(support),
        "retrieval": "BM25 words + adjacent pairs, k1=1.5 b=0.75, max 4/label, 24 total, self excluded",
        "context_budget": CONTEXT_BUDGET,
        "mean_effective_examples": sum(effective) / len(effective),
        "min_effective_examples": min(effective),
        "max_effective_examples": max(effective),
        "retrieval_seconds": retrieval_seconds,
        "train_seconds": train_seconds,
        "ms_per_step": 1000 * train_seconds / max(1, len(schedule)),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "ckpt": str(args.ckpt),
        "ckpt_sha256": sha256_file(args.ckpt),
        "head_sha256": sha256_file(head_path),
        "train_records_sha256": "",
        "code": {
            "scripts/query_head_train.py": sha256_file(Path(__file__).resolve()),
            "azurite/query.py": sha256_file(ROOT / "azurite/query.py"),
            "azurite/retrieval.py": sha256_file(ROOT / "azurite/retrieval.py"),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    }
    records_path = args.out_dir / "train_records.jsonl"
    with records_path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    meta["train_records_sha256"] = sha256_file(records_path)
    (args.out_dir / "support.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in support) + "\n"
    )
    meta["support_sha256"] = sha256_file(args.out_dir / "support.jsonl")
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"wrote {head_path} sha256={meta['head_sha256']}", flush=True)
    print(f"ms/step={meta['ms_per_step']:.1f} last_loss={losses[-1]:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
