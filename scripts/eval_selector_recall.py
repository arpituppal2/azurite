#!/usr/bin/env python3
"""Measure selector recall@8 against BM25 on the frozen dev packs.

This is the task-6 gate.  It reports any-overlap and strict (first gold chunk)
recall, the paired difference from BM25, and a deterministic paired bootstrap
95% interval.  A pass requires at least +3 percentage points AND a lower bound
above zero, matching the approved plan.

Usage:
    python scripts/eval_selector_recall.py \
      --checkpoint results/parity/selector_clean/selector.pt \
      --out results/parity/selector_recall_step500.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import EncoderConfig  # noqa: E402
from azurite.model import Azurite  # noqa: E402
from azurite.selector import (  # noqa: E402
    BM25ChunkSelector,
    ChunkEncoder,
    chunk_text,
    tokenize_chunks,
)
from azurite.tokenizers import PAD_ID, load as load_tokenizer  # noqa: E402

TOP_K = 8


def pad(rows):
    width = max((len(r) for r in rows), default=1)
    ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, row in enumerate(rows):
        if row:
            ids[i, : len(row)] = torch.tensor(list(row), dtype=torch.long)
            mask[i, : len(row)] = True
    return ids, mask


def hits(pack, selected_indices):
    picked = set(selected_indices)
    spans = pack.get("gold_spans") or []
    if not spans:
        return False, False
    any_hit = all(any(i in picked for i in span["chunks"]) for span in spans)
    strict_hit = all(span["chunks"][0] in picked for span in spans)
    return any_hit, strict_hit


def iter_packs(battery: Path):
    manifest = json.loads((battery / "manifest.json").read_text())
    for shard in manifest["shards"]:
        with gzip.open(battery / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--battery", default="results/parity/long_context_dev_noquestion")
    ap.add_argument("--out", default="results/parity/selector_recall.json")
    ap.add_argument("--rows", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tok = load_tokenizer("bpe")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = EncoderConfig(**payload["config"])
    model = ChunkEncoder(Azurite(cfg)).to(device).eval()
    model.load_state_dict(payload["model"])
    step = int(payload.get("step", -1))

    rows = []
    for pack in iter_packs(Path(args.battery)):
        chunks = chunk_text(pack["context"])
        if not chunks:
            continue
        q = str(pack.get("target_question") or pack.get("question") or "")
        q_ids, q_mask = pad([tok.encode(q)[: args.max_tokens]])
        c_ids, c_mask = pad(tokenize_chunks(tok, chunks, max_tokens=args.max_tokens))
        with torch.no_grad():
            qv = model.embed(q_ids.to(device), q_mask.to(device))
            cv = model.embed(c_ids.to(device), c_mask.to(device))
            scores = (qv @ cv.T)[0]
        order = torch.argsort(scores, descending=True).tolist()
        selected = [chunks[i].index for i in order[:TOP_K]]
        sel_any, sel_strict = hits(pack, selected)

        bm = BM25ChunkSelector(chunks).select(q, top_k=TOP_K)
        bm_any, bm_strict = hits(pack, [c.index for c in bm])
        rows.append({
            "id": pack.get("id"), "position": pack.get("position"),
            "selector_any": bool(sel_any), "selector_strict": bool(sel_strict),
            "bm25_any": bool(bm_any), "bm25_strict": bool(bm_strict),
            "selected": selected,
        })
        if args.limit and len(rows) >= args.limit:
            break
        if len(rows) % 100 == 0:
            print(f"scored {len(rows)}", flush=True)

    n = len(rows)
    if not n:
        raise SystemExit("no packs scored")
    sel = np.asarray([r["selector_any"] for r in rows], dtype=float)
    sel_s = np.asarray([r["selector_strict"] for r in rows], dtype=float)
    bm = np.asarray([r["bm25_any"] for r in rows], dtype=float)
    bm_s = np.asarray([r["bm25_strict"] for r in rows], dtype=float)
    delta = sel - bm
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(args.bootstrap, n))
    boot = sel[idx].mean(1) - bm[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    by_pos = {}
    for pos in sorted({r["position"] for r in rows}):
        mask = np.asarray([r["position"] == pos for r in rows])
        by_pos[pos] = {
            "n": int(mask.sum()),
            "selector": float(sel[mask].mean()),
            "bm25": float(bm[mask].mean()),
            "delta": float((sel[mask] - bm[mask]).mean()),
        }
    out = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": step,
        "battery": args.battery,
        "n": n,
        "top_k": TOP_K,
        "selector_recall": float(sel.mean()),
        "selector_recall_strict": float(sel_s.mean()),
        "bm25_recall": float(bm.mean()),
        "bm25_recall_strict": float(bm_s.mean()),
        "delta": float(delta.mean()),
        "delta_ci95": [float(lo), float(hi)],
        "bootstrap": args.bootstrap,
        "by_position": by_pos,
        "gate": {
            "criterion": "delta >= 0.03 and 95% paired-bootstrap lower bound > 0",
            "passed": bool(delta.mean() >= 0.03 and lo > 0.0),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    if args.rows:
        Path(args.rows).parent.mkdir(parents=True, exist_ok=True)
        Path(args.rows).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if out["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
