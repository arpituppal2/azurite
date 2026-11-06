#!/usr/bin/env python3
"""Build a multi-billion-byte unique corpus as a u16 byte stream.

Why this exists
---------------
`scripts/pretrain.py --prepare` builds a stream from local `*.jsonl.gz` shards,
which in this sandbox hold 102M bytes. That is not enough to measure a scaling
law: at D/N = 10 the 113M rung needs 1.13B tokens, i.e. eleven passes over the
same bytes, and eleven passes is a memorisation experiment. The corpus, not the
GPU, was the binding constraint.

This script streams public datasets, writes bytes exactly as the training stack
expects them, and records enough provenance that the result can be re-derived.

Encoding, matching `azurite/tokenizers.py`
------------------------------------------
    specials 0..3   PAD, MASK, CLS, SEP
    byte b          id = b + 4
    between docs    SEP (3), never after the last one

Held-out is carved **by document**, not by byte offset
-------------------------------------------------------
`pretrain.py` reserves the tail of the stream, so the only way to make the tail a
document-level split is to choose the documents first and lay them out as
`[train docs][held-out docs]`. A document is held out iff
`sha1(doc) % 50 == 0`, which is deterministic, independent of source order, and
immune to the interleaving order changing between runs.

Because document sizes do not divide evenly, the boundary is *approximately* 2%
but never inside a document. The exact `n_train` is written to
`<stream>.split.json`, and `pretrain.py` / `eval_full.py` use it when present, so
no byte of a held-out document is ever trained on.

Deduplication
-------------
Exact-duplicate removal by sha1 of the normalised document. This is honest but
narrow: it catches reposts and repeated fixtures, not paraphrase. The manifest
says so, rather than implying a near-dup pipeline that does not exist.

Usage:
    python3 scripts/build_corpus.py --probe                 # can each source be read?
    python3 scripts/build_corpus.py --target-bytes 1.6e9    # build the stream
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
TOKENS = PROJECT / "data" / "tokens"
N_SPECIALS = 4
SEP_ID = 3
HELDOUT_MOD = 50  # 1 in 50 documents -> ~2% of documents

# (label, path, config, split, text field, share of the byte budget)
# Shares are of the *total*; the builder stops early if a source runs out and
# redistributes to the remaining ones rather than silently shrinking the corpus.
# Source list is the corpus composition, so it is data, not code. The code source
# (`smollm-corpus`/`python-edu`) is absent on purpose: resolving it hangs inside
# `load_dataset` before the first document, and the per-source deadline can only
# interrupt the iteration, not the resolution. Its 10% is redistributed to web
# text and encyclopedic text and the manifest records the omission, because a
# corpus description that silently drops a planned source is a false provenance
# claim.
SOURCES = [
    ("fineweb-edu",  "HuggingFaceFW/fineweb-edu",     "sample-10BT",   "train", "text", 0.50),
    ("cosmopedia",   "HuggingFaceTB/smollm-corpus",   "cosmopedia-v2", "train", "text", 0.16),
    ("open-web-math","open-web-math/open-web-math",   None,            "train", "text", 0.12),
    ("wikipedia",    "wikimedia/wikipedia",           "20231101.en",   "train", "text", 0.22),
]


def normalise(text: str) -> bytes:
    """Encode one document the way the training stream stores it.

    Strip outer whitespace only. Aggressive cleaning would change the byte
    distribution and make old and new streams incomparable; the point is a bigger
    corpus, not a different one.
    """
    t = text.strip()
    if not t:
        return b""
    return (t + "\n").encode("utf-8")


def docs_from(source, limit_docs: int, deadline_s: float = 900.0):
    """Yield document text, with a field fallback and a wall-clock cap.

    Both guards exist because a wrong field name does not fail loudly: the
    generator iterates the whole stream yielding nothing, so the byte quota is
    never reached and the build blocks on that one source forever. The probe
    caught it as "0 docs in 1.1s", which reads as an empty dataset rather than
    as a schema mismatch.
    """
    label, path, cfg, split, field, _ = source
    from datasets import load_dataset

    started = time.perf_counter()
    kw = {"split": split, "streaming": True}
    ds = load_dataset(path, cfg, **kw) if cfg else load_dataset(path, **kw)
    for i, ex in enumerate(ds):
        if i >= limit_docs or time.perf_counter() - started > deadline_s:
            return
        val = ex.get(field)
        if not isinstance(val, str):
            val = ex.get("text") or ex.get("content") or ex.get("code")
        if isinstance(val, str) and val.strip():
            yield val


def probe() -> int:
    ok = 0
    for src in SOURCES:
        label = src[0]
        try:
            t0 = time.perf_counter()
            got = 0
            nbytes = 0
            for text in docs_from(src, 25):
                b = normalise(text)
                got += 1
                nbytes += len(b)
            dt = time.perf_counter() - t0
            print(f"  {label:14s} OK   {got} docs, {nbytes:,} bytes, "
                  f"{dt:.1f}s ({nbytes / max(dt, 1e-9) / 1e6:.1f} MB/s)", flush=True)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - a source being unavailable is data
            print(f"  {label:14s} FAIL {type(exc).__name__}: {str(exc)[:110]}", flush=True)
    print(f"\n{ok}/{len(SOURCES)} sources reachable")
    return 0 if ok >= 3 else 1


def build(target_bytes: int, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    split_path = out.with_suffix("").with_suffix(".split.json")

    quota = {s[0]: int(target_bytes * s[5]) for s in SOURCES}
    seen: set[str] = set()
    stats = {s[0]: {"docs": 0, "bytes": 0, "dupes": 0} for s in SOURCES}
    heldout: list[bytes] = []
    # Count in u16 ELEMENTS, not bytes. The first version added `len(ids) + 2`
    # (elements, plus a guessed 2-byte separator) into a byte counter and then
    # divided by two, which reported a 2% held-out split as 51%. A split that is
    # wrong by a factor of two is the kind of error that silently trains on half
    # the corpus, so the unit is now in the name.
    train_elems = 0
    heldout_elems = 0

    fh = open(tmp, "wb")
    buf = bytearray()
    total_written = 0
    t0 = time.perf_counter()

    def flush(force: bool = False) -> None:
        nonlocal total_written
        if len(buf) >= 8 << 20 or (force and buf):
            fh.write(buf)
            total_written += len(buf)
            buf.clear()

    for src in SOURCES:
        label = src[0]
        budget = quota[label]
        got = 0
        try:
            for text in docs_from(src, 10_000_000, deadline_s=900.0):
                raw = normalise(text)
                if not raw:
                    continue
                digest = hashlib.sha1(raw).hexdigest()
                if digest in seen:
                    stats[label]["dupes"] += 1
                    continue
                seen.add(digest)
                ids = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16) + N_SPECIALS
                if int(digest[:8], 16) % HELDOUT_MOD == 0:
                    heldout.append(ids.tobytes())
                    heldout_elems += len(ids) + 1  # document + one SEP
                else:
                    buf.extend(ids.tobytes())
                    buf.extend(np.uint16(SEP_ID).tobytes())
                    train_elems += len(ids) + 1
                    got += len(raw)
                    flush()
                stats[label]["docs"] += 1
                stats[label]["bytes"] += len(raw)
                if got >= budget:
                    break
        except Exception as exc:  # noqa: BLE001
            print(f"  {label}: stopped after {got:,} bytes ({type(exc).__name__}: "
                  f"{str(exc)[:90]})", flush=True)
        print(f"  {label:14s} {got:>13,} bytes  ({time.perf_counter() - t0:.0f}s)", flush=True)

    # Held-out documents go at the very end, so the tail reserved by pretrain.py
    # is exactly this set and never straddles a document boundary.
    for blob in heldout:
        buf.extend(blob)
        buf.extend(np.uint16(SEP_ID).tobytes())
    flush(force=True)
    fh.close()
    tmp.replace(out)

    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()

    n_total = out.stat().st_size // 2
    n_train = train_elems
    assert n_train + heldout_elems == n_total, (
        f"split does not tile the stream: {n_train} + {heldout_elems} != {n_total}")
    split = {
        "stream": str(out),
        "n_total": int(n_total),
        "n_train": int(n_train),
        "heldout_bytes": int(n_total - n_train),
        "heldout_fraction": round(1 - n_train / max(1, n_total), 5),
        "split_rule": f"document held out iff sha1(doc) % {HELDOUT_MOD} == 0; "
                      "held-out documents appended after all training documents",
    }
    split_path.write_text(json.dumps(split, indent=1) + "\n", encoding="utf-8")

    manifest = {
        "stream": str(out),
        "sha256": digest,
        "bytes": out.stat().st_size,
        "tokens_u16": int(n_total),
        "encoding": {"specials": 4, "id": "byte + 4", "separator": "SEP(3) between documents"},
        "target_bytes": target_bytes,
        "sources": {k: v for k, v in stats.items()},
        "dedupe": "exact sha1 of normalised document; not near-duplicate",
        "omitted": {"python-edu": "load_dataset hangs during source resolution; "
                                "its 10% share moved to fineweb-edu and wikipedia."},
        "split": split,
        "seconds": round(time.perf_counter() - t0, 1),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out.with_suffix(".manifest.json")).write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    print(f"\nwrote {out} ({out.stat().st_size / 1e9:.2f} GB, {n_total:,} tokens)")
    print(f"  train {n_train:,}  held-out {n_total - n_train:,} "
          f"({split['heldout_fraction'] * 100:.2f}%)")
    print(f"  sha256 {digest}")
    print(f"  {len(seen):,} unique documents, {sum(v['dupes'] for v in stats.values()):,} dupes dropped")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--target-bytes", type=float, default=1.6e9)
    ap.add_argument("--out", default=str(TOKENS / "byte_corpus_v2.u16"))
    args = ap.parse_args()
    if args.probe:
        return probe()
    return build(int(args.target_bytes), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
