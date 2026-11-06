#!/usr/bin/env python3
"""Check the pretraining corpus slice for overlap with the evaluation sets.

A benchmark number is only meaningful if the model never saw the benchmark.
This project has already published a number once that could not be reproduced;
the corpus/eval boundary is exactly where that class of error lives, so it is
checked before any training, not after.

Two matching units, because one of them is the literature's and the other is
stricter in a different way:

  --unit window  (default) an exact 24-64 character window from the item's
                 opening text, whitespace-insensitive. Stricter than an n-gram
                 for the text it covers, but it only covers the opening.
  --unit ngram   13-word shingles (GPT-3's rule) taken from the item's start
                 and middle, after lowercasing and mapping punctuation to
                 spaces. This is the unit a paper is expected to report, and
                 it covers text deeper into the item.

Either way the probes go through one Aho-Corasick automaton over the whole
corpus. The 13-gram arm normalises the corpus identically before matching,
which is the only way a word-level shingle can match text that upstream
punctuation differences would otherwise hide.

Why not ripgrep: 900 fixed-string patterns against 500 MB timed out past 300 s
on this machine, and `rg` is not installed. An automaton is the correct tool
and runs in seconds.

Deliberate limitations, stated rather than hidden:
  * exact match only. Paraphrase, translation, reformatting and light editing
    are not detected. A clean result means "no verbatim leakage found", not
    "the benchmark is unseen".
  * probes are capped at 64 characters and floored at 24; short probes are
    weaker evidence and are reported separately so they can be inspected.
  * items whose text contains a quote or backslash are skipped whole rather
    than trimmed, because a trimmed probe is not the text that would have
    been memorised.

Usage:
    .venv/bin/python3 scripts/contam_check.py --per-task 100
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
CORPUS = PROJECT / "data" / "corpus"
RESULTS = PROJECT / "results" / "contam_check"

# key -> (path parts after data/raw/<key>/, fields concatenated into the probe)
TARGETS = {
    # primary suite
    "banking77": (["default", "test"], ["text"]),
    "clinc_oos": (["plus", "test"], ["text"]),
    "ag_news": (["default", "test"], ["text"]),
    "emotion": (["split", "test"], ["text"]),
    "massive_intent": (["en", "test"], ["text"]),
    "massive_scenario": (["en", "test"], ["text"]),
    "xnli": (["en", "test"], ["premise", "hypothesis"]),
    "language_id": (["default", "test"], ["text"]),
    "financial_phrasebank": (["default", "test"], ["text"]),
    # multilingual arm
    "xnli_de": (["de", "test"], ["premise", "hypothesis"]),
    "xnli_fr": (["fr", "test"], ["premise", "hypothesis"]),
    "xnli_es": (["es", "test"], ["premise", "hypothesis"]),
    "xnli_zh": (["zh", "test"], ["premise", "hypothesis"]),
    "massive_intent_de": (["de", "test"], ["text"]),
    # secondary suite
    "hellaswag": (["default", "validation"], ["ctx", "endings"]),
    "arc_easy": (["ARC-Easy", "test"], ["question", "choices"]),
    "arc_challenge": (["ARC-Challenge", "test"], ["question", "choices"]),
    "winogrande": (["winogrande_xl", "validation"], ["sentence"]),
    "lambada_openai": (["en", "test"], ["text"]),
    # wilitext-103 is built from Wikipedia, and the corpus contains Wikipedia:
    # this is the one target where overlap is expected rather than hypothetical
    "wikitext103": (["wikitext-103-raw-v1", "test"], ["text"]),
}

MAX_PROBE = 64
MIN_PROBE = 24
WEAK_PROBE = 40  # below this, a hit is reported for inspection, not as proof
NGRAM_N = 13  # GPT-3's contamination rule
NGRAM_MIN_CHARS = 40  # a 13-word shingle shorter than this is a punctuation artefact
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def iter_rows(split_dir: Path):
    """Stream a saved HF dataset directory (Arrow IPC files)."""
    import pyarrow as pa

    for f in sorted(split_dir.glob("*.arrow")):
        with pa.memory_map(str(f), "r") as source:
            for batch in pa.ipc.open_stream(source):
                yield from batch.to_pylist()


def strip_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def flatten(value) -> str:
    """Flatten nested lists/dicts into plain text.

    ARC stores its choices as {"text": [...], "label": [...]} and HellaSwag
    stores endings as a list. Stringifying those inserts brackets and quotes,
    which the probe filter then rejects, silently skipping the whole task.
    """
    if isinstance(value, dict):
        return " ".join(flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(flatten(v) for v in value)
    return str(value)


def normalize_words(text: str) -> str:
    """Lowercase, punctuation to spaces, collapse. One definition, used for
    both the probe and the corpus, so the two cannot drift."""
    return NON_ALNUM.sub(" ", text.lower()).strip()


def make_shingles(text: str, n: int = NGRAM_N, per_item: int = 2) -> list[str]:
    """Return up to `per_item` word n-grams from the start and middle of text."""
    words = normalize_words(text).split()
    if len(words) < n:
        return []
    starts = [0]
    if len(words) >= 2 * n:
        starts.append(len(words) // 2)
    out = []
    for s in starts[:per_item]:
        shingle = " ".join(words[s : s + n])
        if len(shingle) >= NGRAM_MIN_CHARS:
            out.append(shingle)
    return out


def make_probe(text: str) -> tuple[str, int] | None:
    """Return (probe, raw_length) or None if the text is unusable."""
    norm = re.sub(r"\s+", " ", text).strip()
    if len(norm) < MIN_PROBE:
        return None
    if any(c in norm for c in ('"', "\\")):
        return None
    probe = strip_ws(norm[:MAX_PROBE])
    if len(probe) < MIN_PROBE:
        return None
    return probe, len(norm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--unit", choices=["window", "ngram"], default="window")
    ap.add_argument("--n", type=int, default=NGRAM_N)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="positive and negative control: a shingle from a corpus document must "
        "be found in that document and must not be found in unrelated text",
    )
    args = ap.parse_args()

    import ahocorasick

    RESULTS.mkdir(parents=True, exist_ok=True)
    shards = sorted(CORPUS.glob("*.jsonl.gz"))
    if not shards:
        print("no corpus shards; nothing to check")
        return 1

    if args.self_test:
        # A checker that cannot find a probe it derived from a corpus document is
        # broken, and its zeros mean nothing. Same idea as the window arm's
        # positive control, adapted to the normalisation this arm applies.
        texts: list[str] = []
        with gzip.open(shards[0], "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    t = json.loads(line).get("text") or ""
                except json.JSONDecodeError:
                    continue
                if len(t) > 400:
                    texts.append(t)
                if len(texts) >= 3:
                    break
        if len(texts) < 2:
            print("self-test: not enough usable documents in the first shard")
            return 1
        shingles = make_shingles(texts[0], args.n)
        if not shingles:
            print("self-test: could not derive a shingle from the first document")
            return 1
        probe = shingles[0]
        a = ahocorasick.Automaton()
        a.add_word(probe, "CONTROL")
        a.make_automaton()
        in_own = any(pid == "CONTROL" for _e, pid in a.iter(normalize_words(texts[0])))
        in_other = any(
            pid == "CONTROL" for _e, pid in a.iter(normalize_words(" ".join(texts[1:])))
        )
        print(f"self-test probe ({len(probe.split())} words): {probe[:90]}…")
        print(f"  found in its own document: {in_own}")
        print(f"  found in unrelated text:   {in_other}")
        ok = in_own and not in_other
        print("self-test:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    rng = random.Random(args.seed)
    probes: dict[str, str] = {}
    raw_len: dict[str, int] = {}
    skipped: dict[str, int] = {}
    task_of: dict[str, str] = {}
    item_of: dict[str, str] = {}
    items: dict[str, set[str]] = {}

    for key, (parts, fields) in TARGETS.items():
        split_dir = RAW / key / Path(*parts)
        if not split_dir.exists():
            print(f"{key}: split dir missing ({split_dir})", file=sys.stderr)
            continue
        rows = list(iter_rows(split_dir))
        rng.shuffle(rows)
        skip = 0
        made = 0
        items[key] = set()
        for row in rows:
            if made >= args.per_task:
                break
            text = " ".join(flatten(row.get(f, "")) for f in fields)

            if args.unit == "window":
                got = make_probe(text)
                if got is None:
                    skip += 1
                    continue
                probe, rl = got
                pid = f"{key}:{made}"
                probes[pid] = probe
                raw_len[pid] = rl
                item_of[pid] = pid
            else:
                shingles = make_shingles(text, args.n)
                if not shingles:
                    skip += 1
                    continue
                item = f"{key}:{made}"
                for j, shingle in enumerate(shingles):
                    pid = f"{item}:{j}"
                    probes[pid] = shingle
                    raw_len[pid] = len(shingle)
                    item_of[pid] = item
            task_of[pid] = key
            items[key].add(item_of[pid])
            made += 1
        skipped[key] = skip
        unit_name = "probes" if args.unit == "window" else "items"
        print(f"{key:22s} {made:>4d} {unit_name} ({skip} skipped as too short/escaped)")

    automaton = ahocorasick.Automaton()
    for pid, probe in probes.items():
        automaton.add_word(probe, pid)
    automaton.make_automaton()

    patfile = RESULTS / "probes.txt"
    with patfile.open("w", encoding="utf-8") as fh:
        for probe in probes.values():
            fh.write(probe + "\n")

    normalizer = strip_ws if args.unit == "window" else normalize_words
    print(f"\nsearching {len(shards)} shard(s) for {len(probes)} "
          f"{args.unit} probes with Aho-Corasick...")
    t0 = time.time()
    hits: dict[str, int] = {}
    docs = 0
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    text = json.loads(line).get("text") or ""
                except json.JSONDecodeError:
                    continue
                docs += 1
                if not text:
                    continue
                for _end, pid in automaton.iter(normalizer(text)):
                    hits[pid] = hits.get(pid, 0) + 1
        print(f"  {shard.name}: done, {docs} docs scanned", flush=True)
    elapsed = time.time() - t0

    print(f"\n{len(probes)} probes vs {docs} corpus docs in {elapsed:.0f}s\n")
    print(f"{'task':22s} {'probes':>7s} {'hits':>6s} {'hit rate':>9s} {'weak hits':>10s}")
    print("-" * 60)

    report = {
        "seed": args.seed,
        "unit": args.unit,
        "n": args.n if args.unit == "ngram" else None,
        "max_probe_chars": MAX_PROBE if args.unit == "window" else None,
        "min_probe_chars": MIN_PROBE if args.unit == "window" else None,
        "ngram_min_chars": NGRAM_MIN_CHARS if args.unit == "ngram" else None,
        "per_task_requested": args.per_task,
        "shards": [
            {"path": str(s.relative_to(PROJECT)), "bytes": s.stat().st_size}
            for s in shards
        ],
        "method": (
            "Aho-Corasick automaton, whitespace-insensitive exact match"
            if args.unit == "window"
            else f"Aho-Corasick automaton over {args.n}-word shingles, "
            "lowercased with punctuation mapped to spaces on both sides"
        ),
        "limitations": "verbatim only; paraphrase/translation/editing not detected",
        "corpus_docs_scanned": docs,
        "seconds": round(elapsed, 1),
        "tasks": {},
    }
    total = 0
    total_items_hit = 0
    for key in TARGETS:
        tids = [pid for pid, t in task_of.items() if t == key]
        thits = [pid for pid in tids if pid in hits]
        weak = [pid for pid in thits if raw_len[pid] < WEAK_PROBE]
        total += len(thits)
        # Item-level: an item counts once however many of its shingles matched.
        hit_items = sorted({item_of[pid] for pid in thits})
        n_items = len(items.get(key, ()))
        total_items_hit += len(hit_items)
        rate = len(hit_items) / n_items if n_items else 0.0
        print(
            f"{key:22s} items={n_items:>4d} hit={len(hit_items):>4d} "
            f"{rate:>7.2%} shingles_hit={len(thits):>4d} weak={len(weak):>3d}"
        )
        report["tasks"][key] = {
            "items": n_items,
            "skipped": skipped.get(key, 0),
            "probes_or_shingles": len(tids),
            "items_hit": len(hit_items),
            "item_hit_rate": round(rate, 4),
            "probes_hit": len(thits),
            "weak_hits": len(weak),
            "hit_item_ids": hit_items[:50],
            "hit_probes": [probes[p] for p in thits[:5]],
            "hit_doc_counts": {p: hits[p] for p in thits[:5]},
        }
    report["total_hits"] = total
    report["total_items_hit"] = total_items_hit

    stamp = time.strftime("%Y%m%dT%H%M%S")
    suffix = args.out_suffix or ("_ngram" if args.unit == "ngram" else "")
    out = RESULTS / f"contam_{stamp}{suffix}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\ntotal hits: {total}\nwrote {out.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
