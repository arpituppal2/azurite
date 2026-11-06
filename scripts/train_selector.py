#!/usr/bin/env python3
"""Train the 30M-ish neural chunk selector with InfoNCE. GPU only."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import rung  # noqa: E402
from azurite.model import Azurite  # noqa: E402
from azurite.selector import ChunkEncoder, chunk_text, tokenize_chunks  # noqa: E402
from azurite.tokenizers import PAD_ID, load as load_tokenizer  # noqa: E402


def iter_packs(path: Path):
    """Yield packs from a sharded manifest directory or a legacy single file.

    A 100k-pack battery is ~6.7 GB of text, so it is stored as gzipped JSON
    lines in shards; loading it as one JSON array would need that much RAM. The
    legacy single-object format is still accepted so older batteries stay
    trainable.
    """
    path = Path(path)
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text())
        for shard in sorted(manifest["shards"], key=lambda s: s["index"]):
            with gzip.open(path / shard["file"], "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
    else:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from json.load(fh)["rows"]


def pad(rows):
    width = max((len(r) for r in rows), default=1)
    ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, row in enumerate(rows):
        if row:
            ids[i, : len(row)] = torch.tensor(list(row), dtype=torch.long)
            mask[i, : len(row)] = True
    return ids, mask


def run_batch(batch, tok, model, opt, args, step: int, epoch: int) -> int:
    q_rows, c_rows, golds = [], [], []
    for pack in batch:
        chunks = chunk_text(pack["context"])
        if not chunks:
            continue
        positive = min(int(pack.get("gold_chunk", 0)), len(chunks) - 1)
        q_rows.append(tok.encode(pack.get("target_question", pack["question"]))[: args.max_tokens])
        c_rows.append(tokenize_chunks(tok, chunks, max_tokens=args.max_tokens))
        golds.append(positive)
    if not c_rows:
        return step
    # One pack per forward keeps variable chunk counts simple; the selector is
    # small, so this is still fast.
    losses = []
    for q, chunks, gold in zip(q_rows, c_rows, golds):
        q_ids, q_mask = pad([q])
        c_ids, c_mask = pad(chunks)
        logits = model(
            q_ids.to(args.device), q_mask.to(args.device),
            c_ids.to(args.device), c_mask.to(args.device),
        )[0]
        losses.append(F.cross_entropy(logits[None], torch.tensor([gold], device=args.device)))
    loss = torch.stack(losses).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1
    if step % 50 == 0:
        print(f"epoch {epoch} step {step} loss {float(loss):.4f}", flush=True)
    return step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--out", default="results/parity/selector.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rung", default="mid")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; selector training is GPU-only")
    packs_path = Path(args.packs)
    if not (packs_path.is_file() or (packs_path / "manifest.json").exists()):
        raise SystemExit(f"no pack battery at {packs_path}")
    print(f"packs: {packs_path}", flush=True)
    tok = load_tokenizer("bpe")
    cfg = rung(args.rung, "bpe", max_seq=args.max_tokens)
    model = ChunkEncoder(Azurite(cfg)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        batch: list[dict] = []
        for pack in iter_packs(packs_path):
            batch.append(pack)
            if len(batch) < args.batch:
                continue
            step = run_batch(batch, tok, model, opt, args, step, epoch)
            batch = []
        if batch:
            step = run_batch(batch, tok, model, opt, args, step, epoch)
        print(f"epoch {epoch} done at step {step}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "model": model.state_dict()}, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
