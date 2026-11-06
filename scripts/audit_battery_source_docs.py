#!/usr/bin/env python3
"""Source-document contamination audit for the frozen 10k battery.

The earlier battery audit checks questions and gold answers. This one checks the
documents those items were built from, against the **actual BPE token stream the
reader is training on** (not the unrelated byte corpus). It tokenizes each source
document with the same BPE vocabulary, takes up to five 48-token windows (full
text for short documents), and scans the training prefix of the stream with an
Aho-Corasick automaton over packed uint16 token ids.

A source document is flagged if any audited token window occurs at a
token-aligned offset in the training prefix. The scan stops before the held-out
tail, so a document that appears only in the held-out split is not called
training contamination.

This is exact-match evidence. Paraphrase, translation and light editing are not
detected; the limitations are stored in the report rather than hidden.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from azurite.decisions import TASKS, candidate_bank, load_examples  # noqa: E402
from azurite.tokenizers import SEP_ID, load as load_tokenizer  # noqa: E402

BATTERY = PROJECT / "results" / "parity" / "eval_battery_10k"
STREAM = PROJECT / "data" / "tokens" / "bpe_6200000000.u16"
RECORD = PROJECT / "data" / "tokens" / "bpe_6200000000.record.json"
OUT = PROJECT / "results" / "parity" / "eval_battery_source_docs.json"
WINDOW_TOKENS = 48
MIN_WINDOW_TOKENS = 8
CHUNK_BYTES = 64 * 1024 * 1024
CONTROL = "__stream_control__"


def rows() -> list[dict]:
    manifest = json.loads((BATTERY / "manifest.json").read_text())
    out: list[dict] = []
    for shard in manifest["shards"]:
        with gzip.open(BATTERY / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
    return out


def _source_examples(names: list[str], split: str = "test") -> dict[str, dict[int, str]]:
    cache: dict[str, dict[int, str]] = {}
    for name in names:
        spec = TASKS[name]
        bank = candidate_bank(spec)
        examples = load_examples(spec, bank, split=split, limit=None)
        cache[name] = {
            int(ex.index): (ex.parts[0].strip() if ex.parts else ex.context.strip())
            for ex in examples
        }
    return cache


def source_texts(data: list[dict]) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """Return source text by id, row ids per source, and the source scope."""
    doc_names = sorted({r["task"] for r in data if r["family"] == "document"})
    robust_names = sorted(
        {r["provenance"]["dataset"] for r in data if r["family"] == "robustness"}
    )
    doc_sources = _source_examples(doc_names)
    robust_sources = _source_examples(
        robust_names, split="test"
    )
    by_text: dict[str, str] = {}
    row_ids: dict[str, list[str]] = defaultdict(list)
    scope: dict[str, str] = {}

    def add(text: str, rid: str, kind: str) -> None:
        text = str(text).strip()
        if not text:
            return
        # The source id is a content address, so repeated passages collapse.
        sid = "src_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
        if sid not in by_text:
            by_text[sid] = text
            scope[sid] = kind
        if rid not in row_ids[sid]:
            row_ids[sid].append(rid)

    for row in data:
        family = row["family"]
        if family == "document":
            task = row["task"]
            source = doc_sources.get(task, {}).get(int(row["provenance"]["row"]))
            if source:
                add(source, row["id"], "source_passage")
            else:
                # Fail closed: a missing source must be visible in the report.
                add(row.get("context", ""), row["id"], "pack_context_fallback")
        elif family == "abstention":
            sentence = (row.get("meta") or {}).get("sentence") or row["question"]
            add(sentence, row["id"], "heldout_sentence")
        elif family == "robustness":
            dataset = row["provenance"]["dataset"]
            source = robust_sources.get(dataset, {}).get(int(row["provenance"]["row"]))
            if source:
                add(source, row["id"], "base_context")
            else:
                add(row.get("context", ""), row["id"], "perturbed_context_fallback")
        else:
            add(row.get("context", ""), row["id"], "source_context")
    return by_text, row_ids, scope


def token_windows(ids: list[int], n: int = WINDOW_TOKENS) -> list[list[int]]:
    ids = [int(i) for i in ids if int(i) > SEP_ID]
    if len(ids) < MIN_WINDOW_TOKENS:
        return []
    if len(ids) <= n:
        return [ids]
    offsets = {0, len(ids) // 4, len(ids) // 2, (3 * len(ids)) // 4, len(ids) - n}
    return [ids[o : o + n] for o in sorted(offsets)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-bytes", type=int, default=0, help="debug: scan only this many bytes")
    ap.add_argument("--limit-docs", type=int, default=0, help="debug: audit only this many source docs")
    args = ap.parse_args()

    started = time.time()
    battery = rows()
    counts = Counter(r["family"] for r in battery)
    texts, row_ids, scope = source_texts(battery)
    sids = sorted(texts)
    if args.limit_docs:
        sids = sids[: args.limit_docs]

    tok = load_tokenizer("bpe")
    patterns: dict[bytes, list[str]] = defaultdict(list)
    doc_windows: dict[str, int] = {}
    short_docs = 0
    skipped_docs = 0
    for sid in sids:
        ids = tok.encode(texts[sid])
        wins = token_windows(ids)
        if not wins:
            skipped_docs += 1
            continue
        doc_windows[sid] = len(wins)
        if len(ids) < WINDOW_TOKENS:
            short_docs += 1
        for win in wins:
            raw = np.asarray(win, dtype="<u2").tobytes()
            # Identical windows from repeated passages share one pattern but
            # retain every source id that contributed it.
            if sid not in patterns[raw]:
                patterns[raw].append(sid)
    print(f"battery rows {len(battery)}; source docs {len(sids)}; patterns {len(patterns)}", flush=True)

    if not patterns:
        raise SystemExit("no auditable source documents")

    record = json.loads(RECORD.read_text())
    n_train = int(record["training_tokens"])
    stream_bytes = STREAM.stat().st_size
    if n_train * 2 > stream_bytes:
        raise SystemExit(f"training prefix exceeds stream: {n_train*2} > {stream_bytes}")

    automaton = ahocorasick.Automaton()
    for raw in patterns:
        # pyahocorasick keys are strings; latin-1 is a lossless one-byte map.
        automaton.add_word(raw.decode("latin-1"), raw)
    automaton.make_automaton()

    # Positive control: take a token-aligned window from the middle of the
    # training prefix, add it as a pattern, and require the scan to find it.
    control_offset = (n_train // 2) * 2
    with open(STREAM, "rb") as fh:
        fh.seek(control_offset)
        control_raw = fh.read(WINDOW_TOKENS * 2)
    if len(control_raw) != WINDOW_TOKENS * 2:
        raise SystemExit("could not read stream control")
    automaton.add_word(control_raw.decode("latin-1"), CONTROL)
    automaton.make_automaton()

    scan_bytes = n_train * 2 if not args.max_bytes else min(args.max_bytes, n_train * 2)
    hits: dict[str, list[int]] = defaultdict(list)
    control_found = False
    scanned = 0
    with open(STREAM, "rb") as fh:
        carry = b""
        while scanned < scan_bytes:
            take = min(CHUNK_BYTES, scan_bytes - scanned)
            chunk = fh.read(take)
            if not chunk:
                break
            data = carry + chunk
            base = scanned - len(carry)
            for end, matched in automaton.iter(data.decode("latin-1")):
                if matched == CONTROL:
                    control_found = True
                    continue
                start = end - len(matched) + 1
                if start < 0:
                    continue
                absolute = base + start
                if absolute % 2 == 0 and 0 <= absolute < scan_bytes:
                    for sid in patterns[matched]:
                        hits[sid].append(absolute // 2)
            carry = data[-(WINDOW_TOKENS * 2 - 2):]
            scanned += len(chunk)
            if scanned % (512 * 1024 * 1024) < CHUNK_BYTES:
                print(f"scanned {scanned/1e9:.2f} GB; hits {len(hits)}", flush=True)

    per_family: dict[str, dict] = {
        fam: {"rows": counts[fam], "hit_rows": 0, "source_docs": 0, "hit_sources": 0}
        for fam in counts
    }
    row_family = {r["id"]: r["family"] for r in battery}
    battery_by_id = {r["id"]: r for r in battery}
    hit_rows: list[dict] = []
    for sid in sids:
        if sid not in doc_windows:
            continue
        fams = Counter(row_family[rid] for rid in row_ids[sid])
        for fam in fams:
            per_family[fam]["source_docs"] += 1
        if sid in hits:
            for rid in row_ids[sid]:
                row = battery_by_id[rid]
                hit_rows.append({
                    "id": rid,
                    "family": row["family"],
                    "source_id": sid,
                    "scope": scope[sid],
                    "stream_token_offsets": hits[sid][:5],
                    "window_count": doc_windows[sid],
                })
            for fam in fams:
                per_family[fam]["hit_rows"] += fams[fam]
                per_family[fam]["hit_sources"] += 1

    report = {
        "battery": str(BATTERY),
        "battery_sha256": json.loads((BATTERY / "manifest.json").read_text())["battery_sha256"],
        "stream": str(STREAM),
        "stream_sha256": "e8da0a60483e45ab194ad861979456abd50d65a7e728feca9de727732dd4deec",
        "stream_bytes": stream_bytes,
        "training_prefix_tokens": n_train,
        "training_prefix_bytes": n_train * 2,
        "heldout_tokens_excluded": int(record["tokens"]) - n_train,
        "method": "BPE-tokenize source documents; exact 48-token windows; Aho-Corasick over packed uint16; token-aligned offsets only",
        "window_tokens": WINDOW_TOKENS,
        "source_docs": len(sids),
        "source_docs_auditable": len(doc_windows),
        "source_docs_short": short_docs,
        "source_docs_skipped_too_short": skipped_docs,
        "patterns": len(patterns),
        "rows": len(battery),
        "families": per_family,
        "hit_rows": len(hit_rows),
        "hit_source_docs": len(hits),
        "hit_examples": hit_rows[:25],
        "control_found": control_found,
        "control_probe": control_raw.hex(),
        "scanned_bytes": scan_bytes,
        "elapsed_s": round(time.time() - started, 1),
        "limitations": "exact token match only; paraphrase, translation and light editing are not detected; short documents use the whole token sequence",
        "gate": "PASS" if (not hits and control_found) else "FAIL",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("source_docs", "rows", "hit_rows", "hit_source_docs", "control_found", "gate")}, indent=1))
    return 0 if report["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
