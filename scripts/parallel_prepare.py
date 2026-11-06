#!/usr/bin/env python3
"""Parallel, document-ordered BPE stream build for the GPU host.

Each shard is tokenised by one worker; the merge then takes one document from
each shard per round, preserving the source interleave used by pretrain.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from azurite.tokenizers import SEP_ID  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def worker(job: dict) -> dict:
    shard = Path(job["shard"])
    tok = Tokenizer.from_file(job["tokenizer"])
    quota = int(job["quota"])
    part = Path(job["part"])
    part.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(part, dtype=np.uint16, mode="w+", shape=(quota + 2_000_000,))
    filled = 0
    lengths: list[int] = []
    sources: dict[str, int] = {}
    with gzip.open(shard, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                text = json.loads(line).get("text")
            except json.JSONDecodeError:
                continue
            if not isinstance(text, str) or not text:
                continue
            ids = tok.encode(text).ids
            if not ids:
                continue
            if filled >= quota:
                break
            take = min(len(ids), quota - filled)
            mm[filled:filled + take] = ids[:take]
            lengths.append(take)
            filled += take
            sources[shard.name] = sources.get(shard.name, 0) + take
            if filled >= quota:
                break
    mm.flush()
    del mm
    os.truncate(part, filled * 2)
    docs = Path(job["docs"])
    docs.write_text(json.dumps({"lengths": lengths, "filled": filled,
                                "sources": sources}) + "\n")
    return {"part": str(part), "docs": str(docs), "filled": filled,
            "documents": len(lengths)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--tokenizer", default="data/tokenizers/azurite_bpe_32004.json")
    ap.add_argument("--tokens", type=float, default=6.2e9)
    ap.add_argument("--out", default="data/tokens/bpe_6200000000.parallel.u16")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    root = PROJECT
    corpus = root / args.corpus
    shards = sorted(corpus.glob("*.jsonl.gz")) or sorted(corpus.glob("*.jsonl"))
    if not shards:
        raise SystemExit(f"no corpus shards in {corpus}")
    tokenizer = str(root / args.tokenizer)
    want = int(args.tokens * 1.02)
    total = sum(p.stat().st_size for p in shards)
    parts = root / "results/parity/parallel_parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, shard in enumerate(shards):
        quota = int(want * shard.stat().st_size / total) + 2_000_000
        jobs.append({"shard": str(shard), "tokenizer": tokenizer, "quota": quota,
                     "part": str(parts / f"part_{i:02d}.u16"),
                     "docs": str(parts / f"docs_{i:02d}.json")})
    started = time.time()
    with mp.Pool(processes=min(args.workers, len(jobs))) as pool:
        results = pool.map(worker, jobs)
    print("workers", len(results), "seconds", round(time.time() - started, 1), flush=True)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(want,))
    parts_mm = [np.memmap(r["part"], dtype=np.uint16, mode="r") for r in results]
    lengths = [json.loads(Path(r["docs"]).read_text())["lengths"] for r in results]
    pos = [0] * len(results)
    di = [0] * len(results)
    filled = 0
    while filled < want:
        moved = False
        for i in range(len(results)):
            if di[i] >= len(lengths[i]):
                continue
            take = int(lengths[i][di[i]])
            di[i] += 1
            room = want - filled - (1 if filled else 0)
            take = min(take, room)
            if take <= 0:
                continue
            if filled:
                out[filled] = SEP_ID
                filled += 1
            out[filled:filled + take] = parts_mm[i][pos[i]:pos[i] + take]
            pos[i] += take
            filled += take
            moved = True
            if filled >= want:
                break
        if not moved:
            break
    out.flush()
    del out
    os.truncate(out_path, filled * 2)
    record = {
        "arm": "bpe", "tokens": int(filled),
        "training_tokens": int(filled * 0.98), "heldout_tokens": int(filled * 0.02),
        "documents": sum(len(x) for x in lengths), "path": str(out_path.relative_to(root)),
        "seconds": round(time.time() - started, 1), "parallel": True,
        "workers": len(results), "corpus_shards": len(shards),
    }
    out_path.with_suffix(".record.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1), flush=True)
    return 0 if filled >= int(args.tokens * 0.98) else 2


if __name__ == "__main__":
    raise SystemExit(main())
