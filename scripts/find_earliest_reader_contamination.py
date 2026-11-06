#!/usr/bin/env python3
"""Find the earliest reader step whose sampled windows touch a source pattern."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import ahocorasick
import numpy as np

SEP_ID = 3

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    patterns_doc = json.loads(Path(args.patterns).read_text())
    patterns = {bytes.fromhex(k): tuple(v) for k, v in patterns_doc["patterns"].items()}
    source = patterns_doc.get("source", {})
    auto = ahocorasick.Automaton()
    for raw, sids in patterns.items():
        auto.add_word(raw.decode("latin-1"), (raw, sids))
    auto.make_automaton()
    stream = np.memmap(args.stream, dtype="<u2", mode="r")
    rng = np.random.default_rng(args.seed)
    first = None
    hit_steps = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        starts = rng.integers(0, args.n_train - args.seq - 1, size=args.batch)
        step_hits = []
        for start in starts:
            raw = np.asarray(stream[int(start):int(start) + args.seq], dtype="<u2").tobytes()
            for end, (matched, sids) in auto.iter(raw.decode("latin-1")):
                off = end - len(matched) + 1
                if off % 2 == 0:
                    step_hits.append((int(start), off // 2, list(sids)))
        if step_hits:
            rec = {"step": step, "starts": [int(x) for x in starts], "hits": step_hits[:8],
                   "hit_count": len(step_hits)}
            hit_steps.append(rec)
            if first is None:
                first = rec
                print("EARLIEST", json.dumps(first)[:2000], flush=True)
                break
    rec = {
        "stream": str(args.stream),
        "stream_sha256": patterns_doc.get("stream_sha256"),
        "seed": args.seed, "seq": args.seq, "batch": args.batch,
        "steps_replayed": args.steps, "n_train": args.n_train,
        "earliest_contaminated_step": first,
        "first_16_hit_steps": hit_steps[:16],
        "scanned_windows": min(args.steps, args.steps) * args.batch,
        "elapsed_s": round(time.time() - t0, 1),
        "method": "replayed pretrain.batches() numpy default_rng(seed) from step 1",
    }
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)[:4000])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
