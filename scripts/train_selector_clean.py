#!/usr/bin/env python3
"""Resumable low-memory selector trainer for a source-clean pack view."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from azurite.config import rung  # noqa: E402
from azurite.model import Azurite  # noqa: E402
from azurite.selector import ChunkEncoder, chunk_text, tokenize_chunks  # noqa: E402
from azurite.tokenizers import PAD_ID, load as load_tokenizer  # noqa: E402
from train_selector import iter_packs  # noqa: E402


def pad(rows):
    width = max((len(r) for r in rows), default=1)
    ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, row in enumerate(rows):
        if row:
            ids[i, : len(row)] = torch.tensor(list(row), dtype=torch.long)
            mask[i, : len(row)] = True
    return ids, mask


def atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save(model, opt, cfg, step, out: Path, packs: str) -> None:
    atomic_save(
        {"config": cfg.to_dict(), "model": model.state_dict(), "opt": opt.state_dict(),
         "step": step, "packs": packs},
        out,
    )
    (out.parent / "selector_state.json").write_text(
        json.dumps({"step": step, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--out", default="results/parity/selector_clean/selector.pt")
    ap.add_argument("--log", default="results/parity/selector_clean/selector_rows.jsonl")
    ap.add_argument("--max-steps", type=int, default=31000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rung", default="mid")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if not (Path(args.packs) / "manifest.json").exists():
        raise SystemExit(f"no pack battery at {args.packs}")
    tok = load_tokenizer("bpe")
    cfg = rung(args.rung, "bpe", max_seq=args.max_tokens)
    model = ChunkEncoder(Azurite(cfg)).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    out = Path(args.out)
    step = 0
    if args.resume and out.exists():
        state = torch.load(out, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        if state.get("opt") is not None:
            opt.load_state_dict(state["opt"])
        step = int(state.get("step", 0))
        print(f"resumed selector at step {step}", flush=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log = open(args.log, "a", buffering=1)
    started = time.time()
    skipped = 0
    for pack in iter_packs(Path(args.packs)):
        if step >= args.max_steps:
            break
        if skipped < step:
            skipped += 1
            continue
        chunks = chunk_text(pack["context"])
        if not chunks:
            continue
        gold = min(int(pack.get("gold_chunk", 0)), len(chunks) - 1)
        q_ids, q_mask = pad([tok.encode(pack.get("target_question", pack["question"]))[:args.max_tokens]])
        c_ids, c_mask = pad(tokenize_chunks(tok, chunks, max_tokens=args.max_tokens))
        q = torch.utils.checkpoint.checkpoint(model.embed, q_ids.cuda(), q_mask.cuda(), use_reentrant=False)
        c = torch.utils.checkpoint.checkpoint(model.embed, c_ids.cuda(), c_mask.cuda(), use_reentrant=False)
        logits = model.log_scale.exp() * (q @ c.T)
        loss = F.cross_entropy(logits, torch.tensor([gold], device="cuda"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        rec = {"step": step, "pack": pack.get("id"), "chunks": len(chunks),
               "tokens": int(q_ids.numel() + c_ids.numel()), "loss": float(loss.detach()),
               "elapsed_s": round(time.time() - started, 1),
               "gpu_mem_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
               "nan": bool(torch.isnan(loss).any())}
        log.write(json.dumps(rec) + "\n")
        if step % 10 == 0 or step == 1:
            print(json.dumps(rec), flush=True)
        if step % args.save_every == 0:
            save(model, opt, cfg, step, out, args.packs)
        torch.cuda.empty_cache()
    save(model, opt, cfg, step, out, args.packs)
    log.close()
    print("DONE", step, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
