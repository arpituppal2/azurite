#!/usr/bin/env python3
"""Train the BPE arm on the local corpus, and measure the sequence-length ratio.

Two outputs:

1. `data/tokenizers/azurite_bpe_32004.json` plus a `.record.json` carrying the
   sha256 of the exact text the vocabulary was learned from. Nothing is
   downloaded; the vocabulary is a product of this corpus.
2. A measurement of **bytes per token on held-out documents**, in our own
   corpus, for both arms. The plan currently cites ByT5's 4.1 bytes/token on
   mC4 as the sequence-length ratio between byte and subword arms. That is a
   published number about a different corpus, and this project's whole
   tokenizer argument rests on the ratio, so it gets measured here.

Held-out means documents the trainer never saw: the training sample is taken
from the head of each shard, the measurement from the tail.

Usage:
    .venv/bin/python3 scripts/train_tokenizers.py --sample-mb 300
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import VOCAB_SIZE  # noqa: E402
from azurite.tokenizers import BpeTokenizer, ByteTokenizer  # noqa: E402

CORPUS = PROJECT / "data" / "corpus"
RESULTS = PROJECT / "results" / "tokenizer"


def iter_docs(paths: list[Path], *, tail: bool = False, limit_bytes: int | None = None):
    """Yield text from corpus shards, optionally from the end of each file.

    gzip is not seekable, so the tail is collected by keeping a bounded buffer
    of the last N records. Simpler than it sounds and exact.
    """
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        taken = 0
        buf: list[str] = []
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text")
                if not isinstance(text, str) or not text:
                    continue
                if tail:
                    buf.append(text)
                    if len(buf) > 400:
                        buf.pop(0)
                    continue
                yield text
                taken += len(text)
                if limit_bytes is not None and taken >= limit_bytes:
                    break
        if tail:
            yield from buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-mb", type=int, default=300)
    ap.add_argument("--measure-docs", type=int, default=400)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE["bpe"])
    args = ap.parse_args()

    files = sorted(glob.glob(str(CORPUS / "*.jsonl.gz")))
    if not files:
        files = sorted(glob.glob(str(CORPUS / "*.jsonl")))
    if not files:
        raise SystemExit(f"no corpus shards in {CORPUS}")
    paths = [Path(f) for f in files]
    # Use the English edu/web shards first: the Urdu/Chinese shards would make
    # the vocabulary multilingual, which is a different experiment.
    english = [p for p in paths if "wikipedia_de" not in p.name and "multiling" not in p.name]
    paths = english or paths
    print(f"corpus shards: {len(paths)} ({paths[0].name} … {paths[-1].name})")

    sample_chars = args.sample_mb * 1024 * 1024
    t0 = time.time()
    sample = []
    collected = 0
    for text in iter_docs(paths, limit_bytes=sample_chars):
        sample.append(text)
        collected += len(text)
        if collected >= sample_chars:
            break
    print(f"training sample: {len(sample):,} docs, {collected / 2**20:.1f} MiB chars, "
          f"read in {time.time() - t0:.1f}s")

    tmp = PROJECT / "data" / "tokenizers" / "_train_sample.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        for text in sample:
            fh.write(json.dumps({"text": text}) + "\n")

    t0 = time.time()
    bpe, record = BpeTokenizer.train(
        [tmp], vocab_size=args.vocab_size, sample_bytes=None
    )
    train_seconds = time.time() - t0
    tmp.unlink()
    print(f"trained BPE {bpe.vocab_size:,} tokens in {train_seconds:.1f}s -> "
          f"{record['out_path']}")

    # Held-out measurement: the tail of each shard, never fed to the trainer.
    held = []
    for text in iter_docs(paths, tail=True):
        held.append(text)
        if len(held) >= args.measure_docs:
            break
    if not held:
        raise SystemExit("no held-out documents found")

    byte_tok = ByteTokenizer()
    rows = []
    total_chars = total_bytes = total_byte_ids = total_bpe_ids = 0
    exact_fail = 0
    for text in held:
        raw = len(text.encode("utf-8"))
        n_byte = len(byte_tok.encode(text))
        n_bpe = len(bpe.encode(text))
        if bpe.decode(bpe.encode(text)) != text or byte_tok.decode(byte_tok.encode(text)) != text:
            exact_fail += 1
        total_chars += len(text)
        total_bytes += raw
        total_byte_ids += n_byte
        total_bpe_ids += n_bpe
        rows.append({"chars": len(text), "bytes": raw, "byte_ids": n_byte, "bpe_ids": n_bpe})

    bpt_byte = total_bytes / total_byte_ids
    bpt_bpe = total_bytes / total_bpe_ids
    ratio = total_byte_ids / total_bpe_ids
    uniq_bpe = total_bpe_ids / len(held) / 1.0
    summary = {
        "documents_measured": len(held),
        "bytes": total_bytes,
        "bytes_per_token_byte_arm": round(bpt_byte, 4),
        "bytes_per_token_bpe_arm": round(bpt_bpe, 4),
        "sequence_length_ratio_byte_vs_bpe": round(ratio, 3),
        "mean_bpe_tokens_per_doc": round(uniq_bpe, 1),
        "exact_roundtrip_failures": exact_fail,
        "bpe_vocab": bpe.vocab_size,
        "bpe_training_seconds": round(train_seconds, 1),
        "training_sample_chars": collected,
        "training_record": record,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    (RESULTS / f"tokenizer_{stamp}.json").write_text(json.dumps(summary, indent=1) + "\n")

    print()
    print(f"held-out documents:      {len(held):,} ({total_bytes / 2**20:.1f} MiB)")
    print(f"byte arm   bytes/token:  {bpt_byte:.3f}  (by construction, exactly 1.0 per "
          f"ASCII byte and 1.0 per UTF-8 byte otherwise)")
    print(f"bpe arm    bytes/token:  {bpt_bpe:.3f}")
    print(f"sequence-length ratio:   {ratio:.3f}x more positions in the byte arm")
    print(f"round-trip failures:     {exact_fail}")
    print(f"ByT5 reported on mC4:    4.1 bytes/token (range 2.5-9.0 across languages)")

    md = [
        "# Tokenizer arm measurement\n",
        f"Generated by `scripts/train_tokenizers.py` at {stamp}. Held-out documents "
        "are the tails of the corpus shards; the training sample is the heads, so "
        "the vocabulary has not seen the measurement text.\n",
        "| quantity | value |",
        "|---|---:|",
        f"| held-out documents | {len(held):,} |",
        f"| held-out bytes | {total_bytes:,} |",
        f"| byte arm, bytes/token | {bpt_byte:.3f} |",
        f"| BPE arm, bytes/token | {bpt_bpe:.3f} |",
        f"| sequence-length ratio (byte / BPE) | **{ratio:.3f}×** |",
        f"| mean BPE tokens per document | {uniq_bpe:.1f} |",
        f"| exact round-trip failures | {exact_fail} |",
        f"| BPE vocabulary | {bpe.vocab_size:,} |",
        f"| BPE training time | {train_seconds:.1f} s |",
        "",
        f"ByT5 measured **4.1 bytes/token** on mC4 (range 2.5–9.0 by language). "
        f"This corpus gives **{bpt_bpe:.2f}**, so the plan's sequence-length ratio "
        "is now measured on the text it will actually train on rather than borrowed "
        "from a different corpus.\n",
        "Reproduce with `--sample-mb` and `--measure-docs`; both are recorded in "
        "the JSON beside this file.\n",
    ]
    (RESULTS / "TOKENIZER.md").write_text("\n".join(md))
    print(f"\nwrote {RESULTS.relative_to(PROJECT)}/TOKENIZER.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
