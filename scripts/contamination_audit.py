#!/usr/bin/env python3
"""Exact-overlap audit: did pretraining see the evaluation text?

The transfer claim is only interesting if the pretrained encoder learned
structure that transfers. A frozen pretraining corpus can also make a benchmark
easier by containing the benchmark text itself. This script measures that
directly rather than arguing about it:

  1. take the exact test rows used by the transfer run (indices and order come
     from the run's provenance, not a fresh sample);
  2. cut each normalised text into fixed 40-byte windows plus the whole text;
  3. search the decoded pretraining byte stream for every window with fixed
     string matching (`grep -F`), not a fuzzy n-gram estimate;
  4. join the match flag to the per-example correctness already stored in the
     result file and report accuracy on matched versus unmatched examples.

A 40-byte exact match is a conservative contamination signal: common phrases
are unlikely to collide by chance at that length, and a false negative is more
likely than a false positive. `strong` requires two distinct windows (or the
whole text), so one generic sentence cannot label an example contaminated.

The stream is stored as little-endian uint16 byte IDs with `id = byte + 4`;
decoding subtracts 4 and clips the four specials to 0. The decoded file is a
derived artifact and is verified by length, not treated as source evidence.

Usage:
    .venv/bin/python scripts/contamination_audit.py
    .venv/bin/python scripts/contamination_audit.py --skip-decode
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

TOKENS = PROJECT / "data" / "tokens"
STREAM = TOKENS / "byte_corpus_v2.u16"
DECODED = TOKENS / "byte_corpus_v2.bytes"
OUT = PROJECT / "results" / "contamination"
AG = PROJECT / "results" / "recovered_20261028" / "results" / "transfer" / "ag_news_balanced"
BANK = PROJECT / "results" / "decode_vs_score" / "banking77_tiny_byte_strat8000_s0.json"
WINDOW = 40


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def windows(text: str) -> list[str]:
    """Whole text plus evenly spaced fixed-length windows, de-duplicated."""
    text = norm(text)
    out: list[str] = []
    if len(text) >= 24:
        out.append(text)
    if len(text) < WINDOW:
        return out
    offsets = {0, max(0, len(text) // 3), max(0, 2 * len(text) // 3), len(text) - WINDOW}
    for off in sorted(offsets):
        w = text[off : off + WINDOW]
        if len(w) == WINDOW:
            out.append(w)
    return list(dict.fromkeys(out))


def decode_stream() -> Path:
    if DECODED.exists() and DECODED.stat().st_size == STREAM.stat().st_size // 2:
        return DECODED
    raw = np.memmap(STREAM, dtype="<u2", mode="r")
    n = raw.size
    tmp = DECODED.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        for start in range(0, n, 50_000_000):
            chunk = np.asarray(raw[start : start + 50_000_000], dtype=np.int16)
            # id 0..3 are specials; byte b is id b+4. Clipping keeps the
            # specials out of ASCII text without inventing new byte values.
            (chunk - 4).clip(0, 255).astype(np.uint8).tofile(fh)
    tmp.replace(DECODED)
    return DECODED


def ag_examples() -> list[dict]:
    from datasets import load_from_disk

    ds = load_from_disk(str(PROJECT / "data" / "raw" / "ag_news" / "default" / "test"))
    prov = json.loads((AG / "pretrained_s0" / "provenance.json").read_text())
    rows = prov["test"]["rows"]
    out = []
    for pos, rec in enumerate(rows):
        idx = int(rec["index"])
        out.append(
            {
                "dataset": "ag_news",
                "pos": pos,
                "index": idx,
                "text": norm(ds[idx]["text"]),
                "gold": int(rec["canonical_gold"]),
            }
        )
    return out


def banking_examples() -> list[dict]:
    from azurite.decisions import TASKS, candidate_bank, stratified_examples

    spec = TASKS["banking77"]
    rows = stratified_examples(spec, candidate_bank(spec), "test", 1000, 1)
    return [
        {
            "dataset": "banking77",
            "pos": pos,
            "index": int(e.index),
            "text": norm(e.context),
            "gold": int(e.gold),
        }
        for pos, e in enumerate(rows)
    ]


def search(patterns: list[tuple[str, str]], corpus: Path) -> Counter:
    """Return counts of each matched pattern string."""
    OUT.mkdir(parents=True, exist_ok=True)
    pat_path = OUT / "patterns.txt"
    seen = list(dict.fromkeys(p for _, p in patterns if "\n" not in p and "\t" not in p))
    pat_path.write_text("\n".join(seen) + "\n", encoding="utf-8")
    matches_path = OUT / "matches.txt"
    # BSD grep's multi-pattern matcher degenerates on thousands of long fixed
    # strings. ripgrep's Aho-Corasick pass is the same exact-string operation at
    # SIMD speed; the static binary is vendored under bin/ for reproducibility.
    rg = PROJECT / "bin" / "rg"
    if not rg.exists():
        rg = Path("rg")
    cmd = [
        str(rg),
        "-F",
        "-f",
        str(pat_path),
        "--text",
        "-o",
        "--no-line-number",
        "--no-filename",
        str(corpus),
    ]
    with open(matches_path, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, text=True)
    # grep exits 1 when nothing matches; that is a result, not a failure.
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"grep failed ({proc.returncode}): {proc.stderr[-500:]}")
    counts: Counter = Counter()
    with open(matches_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                counts[line] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--corpus", default=str(DECODED),
                    help="decoded corpus to search (default: full stream)")
    args = ap.parse_args()

    if not STREAM.exists():
        raise SystemExit(f"missing stream: {STREAM}")
    if not args.skip_decode:
        decode_stream()

    examples = ag_examples() + banking_examples()
    patterns: list[tuple[str, str]] = []
    per: dict[str, dict] = {}
    for ex in examples:
        eid = f"{ex['dataset']}:{ex['index']}"
        ws = windows(ex["text"])
        per[eid] = {**ex, "eid": eid, "n_windows": len(ws), "matched_windows": []}
        for j, w in enumerate(ws):
            patterns.append((eid, w))

    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"missing corpus: {corpus}")
    counts = search(patterns, corpus)
    for eid, w in patterns:
        if counts.get(w, 0):
            per[eid]["matched_windows"].append(w)
    for rec in per.values():
        n = len(rec["matched_windows"])
        rec["n_matched"] = n
        rec["any_match"] = n > 0
        rec["strong"] = n >= 2 or (n == 1 and len(rec["text"]) <= WINDOW)

    # Join AG News correctness, in the provenance order used above.
    ag_correct: dict[int, dict[int, int]] = defaultdict(dict)
    for seed in (0, 1, 2):
        p = AG / f"pretrained_s{seed}" / "result.json"
        if not p.exists():
            continue
        final = json.loads(p.read_text())["final"]
        for pos, c in enumerate(final["correct"] if "correct" in final else []):
            ag_correct[seed][pos] = int(c)
        if "correct" not in final:
            gold = final["canonical_gold"]
            pred = final["canonical_predictions"]
            for pos, (g, y) in enumerate(zip(gold, pred)):
                ag_correct[seed][pos] = int(g == y)

    bank = json.loads(BANK.read_text())
    bank_correct = [int(x) for x in bank["score"]["correct"]]

    summary: dict = {"window_bytes": WINDOW, "corpus": str(corpus),
                    "datasets": {}, "ag_by_seed": {}}
    for ds in ("ag_news", "banking77"):
        recs = [r for r in per.values() if r["dataset"] == ds]
        summary["datasets"][ds] = {
            "n": len(recs),
            "any_match": sum(r["any_match"] for r in recs),
            "strong": sum(r["strong"] for r in recs),
            "any_match_frac": round(sum(r["any_match"] for r in recs) / len(recs), 4),
            "strong_frac": round(sum(r["strong"] for r in recs) / len(recs), 4),
        }

    for seed, corr in ag_correct.items():
        rows = [r for r in per.values() if r["dataset"] == "ag_news"]
        buckets = {"matched": [], "unmatched": []}
        for r in rows:
            if r["pos"] in corr:
                buckets["matched" if r["any_match"] else "unmatched"].append(corr[r["pos"]])
        summary["ag_by_seed"][str(seed)] = {
            k: {
                "n": len(v),
                "accuracy": round(sum(v) / len(v), 4) if v else None,
                "correct": sum(v),
            }
            for k, v in buckets.items()
        }

    bank_rows = [r for r in per.values() if r["dataset"] == "banking77"]
    buckets = {"matched": [], "unmatched": []}
    for r in bank_rows:
        if r["pos"] < len(bank_correct):
            buckets["matched" if r["any_match"] else "unmatched"].append(bank_correct[r["pos"]])
    summary["banking77_accuracy"] = {
        k: {
            "n": len(v),
            "accuracy": round(sum(v) / len(v), 4) if v else None,
            "correct": sum(v),
        }
        for k, v in buckets.items()
    }
    summary["examples"] = sorted(per.values(), key=lambda r: (r["dataset"], r["pos"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(summary, indent=1) + "\n")
    (OUT / "AUDIT.md").write_text(render(summary))
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1))
    return 0


def render(s: dict) -> str:
    lines = [
        "# Contamination audit: exact overlap with the pretraining stream",
        "",
        f"Window: {s['window_bytes']} bytes, exact fixed-string match. `any` = at least",
        "one window; `strong` = two windows or a whole short text. A 40-byte match is",
        "evidence the corpus contains the text; it does not by itself prove the model",
        "used it, which is why accuracy is also split by match status.",
        "",
        "| dataset | n | any match | strong |",
        "|---|---:|---:|---:|",
    ]
    for ds, v in s["datasets"].items():
        lines.append(f"| {ds} | {v['n']} | {v['any_match']} ({v['any_match_frac']:.1%}) | {v['strong']} ({v['strong_frac']:.1%}) |")
    lines += ["", "## AG News accuracy by contamination", "", "| seed | matched n | matched acc | unmatched n | unmatched acc |", "|---:|---:|---:|---:|---:|"]
    for seed, v in s["ag_by_seed"].items():
        m, u = v["matched"], v["unmatched"]
        lines.append(f"| {seed} | {m['n']} | {fmt(m['accuracy'])} | {u['n']} | {fmt(u['accuracy'])} |")
    lines += ["", "## Banking77 accuracy by contamination", "", "| bucket | n | accuracy |", "|---|---:|---:|"]
    for k, v in s["banking77_accuracy"].items():
        lines.append(f"| {k} | {v['n']} | {fmt(v['accuracy'])} |")
    lines += ["", "If the matched subsets are too small, the comparison is underpowered; that is reported as unmeasured, not as zero."]
    return "\n".join(lines) + "\n"


def fmt(x) -> str:
    return "—" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
