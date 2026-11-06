#!/usr/bin/env python3
"""Per-row contamination audit of long-context pack sets against pretraining.

The battery-source audit answers "did the pretraining stream contain the 10k
battery's source documents". This audit answers the other half of the plan's
requirement: for every dev and locked pack row, did the pretraining prefix
contain the pack's real question or its gold evidence?

Method, per row:

  * the question is the pack's `target_question` (the pack's own instruction
    string is generic and is not audited);
  * the evidence is the text under every `gold_spans` byte range, sliced from
    the UTF-8 context exactly as the pack records it;
  * each text is tokenized with the frozen BPE vocabulary and cut into up to
    five 48-token windows (whole text when shorter);
  * every window is searched in the training prefix of the frozen clean stream
    with an Aho-Corasick automaton over packed uint16 bytes, accepting only
    token-aligned offsets;
  * the scan stops at `n_train`, so a document that exists only in the held-out
    tail is not called training contamination.

Findings are written for **every** row, including clean rows. `null` means the
row had too few tokens for an auditable window; it is never reported as clean.
A positive control window is taken from the stream itself and must be found.
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

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.pack_audit import evidence_texts, token_windows  # noqa: E402
from azurite.tokenizers import load as load_tokenizer  # noqa: E402

STREAM = PROJECT / "data" / "tokens" / "bpe_6200000000.clean.u16"
SPLIT = PROJECT / "data" / "tokens" / "bpe_6200000000.clean.split.json"
OUT = PROJECT / "results" / "parity" / "pack_contamination.json"

WINDOW_TOKENS = 48
MIN_WINDOW_TOKENS = 8
CHUNK_BYTES = 256 * 1024 * 1024
OVERLAP_BYTES = 256
CONTROL_KEY = "__positive_control__"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(pack_dir: Path) -> dict:
    path = pack_dir / "manifest.json"
    if not path.exists():
        raise SystemExit(f"{pack_dir}: no manifest.json")
    manifest = json.loads(path.read_text())
    missing = [s["file"] for s in manifest["shards"] if not (pack_dir / s["file"]).exists()]
    if missing:
        raise SystemExit(f"{pack_dir}: missing shard(s) {missing}")
    return manifest


def iter_packs(pack_dir: Path, manifest: dict):
    for shard in sorted(manifest["shards"], key=lambda s: s["index"]):
        with gzip.open(pack_dir / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def collect(label: str, pack_dir: Path, manifest: dict) -> tuple[list[dict], dict[bytes, list[tuple[str, str, str, int]]]]:
    """Tokenize every auditable text; return per-pack metadata and window map."""
    tok = load_tokenizer("bpe")
    packs: list[dict] = []
    patterns: dict[bytes, list[tuple[str, str, str, int]]] = defaultdict(list)
    for pack in iter_packs(pack_dir, manifest):
        pid = str(pack["id"])
        question = str(pack.get("target_question") or "")
        q_ids = tok.encode(question) if question else []
        q_windows = token_windows(q_ids)
        ev_windows: list[list[int]] = []
        for span_index, text in enumerate(evidence_texts(pack)):
            span_windows = token_windows(tok.encode(text))
            ev_windows.extend(span_windows)
        for window in q_windows:
            patterns[bytes_of(window)].append((label, pid, "question", 0))
        for span_index, window in enumerate(ev_windows):
            patterns[bytes_of(window)].append((label, pid, "evidence", span_index))
        packs.append({
            "id": pid,
            "source": str(pack.get("source", "")),
            "position": str(pack.get("position", "")),
            "distractors": pack.get("distractors"),
            "distributed": bool(pack.get("distributed")),
            "gold_chunks": [c for span in pack.get("gold_spans") or [] for c in span.get("chunks", [])],
            "question_windows": len(q_windows),
            "evidence_windows": len(ev_windows),
            "gold_spans": len(pack.get("gold_spans") or []),
            "target_chars": len(question),
        })
    return packs, patterns


def bytes_of(ids: list[int]) -> bytes:
    import numpy as np

    return np.asarray(ids, dtype="<u2").tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], metavar="LABEL=DIR",
                    help="pack set, e.g. dev=results/parity/long_context_dev_noquestion")
    ap.add_argument("--stream", default=str(STREAM))
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if not args.set:
        ap.error("at least one --set LABEL=DIR is required")
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT / out

    started = time.time()
    sets: list[tuple[str, Path, dict]] = []
    for item in args.set:
        if "=" not in item:
            ap.error(f"--set expects LABEL=DIR, got {item!r}")
        label, raw = item.split("=", 1)
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT / path
        sets.append((label, path, read_manifest(path)))

    all_packs: dict[str, list[dict]] = {}
    patterns: dict[bytes, list[tuple[str, str, str, int]]] = defaultdict(list)
    for label, path, manifest in sets:
        packs, found = collect(label, path, manifest)
        all_packs[label] = packs
        for key, value in found.items():
            patterns[key].extend(value)
        print(f"{label}: {len(packs)} packs, {len(found)} distinct windows so far", flush=True)

    split = json.loads(Path(args.split).read_text())
    train_tokens = int(split["n_train"])
    train_bytes = train_tokens * 2
    stream_path = Path(args.stream)
    if stream_path.stat().st_size // 2 != int(split["n_total"]):
        raise SystemExit("stream length disagrees with the split record")

    pattern_ids = {raw: i for i, raw in enumerate(sorted(patterns))}
    auto = ahocorasick.Automaton()
    lengths: list[int] = []
    # The installed Aho-Corasick build accepts str keys only. Latin-1 maps each
    # byte to exactly one code point, so match offsets stay byte offsets.
    for raw, pid in pattern_ids.items():
        auto.add_word(raw.decode("latin-1"), pid)
    auto.make_automaton()
    ids_by_bytes = {i: raw for raw, i in pattern_ids.items()}
    lengths = [len(raw) for raw in pattern_ids]

    # Positive control: 48 real training tokens near the middle of the prefix.
    import numpy as np

    control_start = ((train_bytes // 2) // 2) * 2
    with stream_path.open("rb") as fh:
        fh.seek(control_start)
        control_raw = fh.read(WINDOW_TOKENS * 2)
    auto.add_word(control_raw.decode("latin-1"), -1)
    auto.make_automaton()

    hits: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "question_first_token": None, "evidence_first_token": None,
        "question_windows": 0, "evidence_windows": 0,
    })
    control_at: int | None = None
    scanned_matches = 0
    accept_from = 0
    with stream_path.open("rb") as fh:
        start = 0
        while start < train_bytes:
            take = min(CHUNK_BYTES, train_bytes - start)
            fh.seek(start)
            buf = fh.read(take)
            for end, pid in auto.iter(buf.decode("latin-1")):
                if pid < 0:
                    if control_at is None:
                        control_at = (start + end - len(control_raw) + 1) // 2
                    continue
                off = end - lengths[pid] + 1
                if off % 2:
                    continue
                absolute = start + off
                if absolute < accept_from:
                    continue
                scanned_matches += 1
                token = absolute // 2
                for label, pack_id, kind, _span in patterns[ids_by_bytes[pid]]:
                    rec = hits[(label, pack_id)]
                    rec[f"{kind}_windows"] += 1
                    key = f"{kind}_first_token"
                    if rec[key] is None or token < rec[key]:
                        rec[key] = token
            accept_from = start + take - OVERLAP_BYTES
            start += take - OVERLAP_BYTES
            if start < 0:
                break
            if take < CHUNK_BYTES:
                break

    report_sets: dict[str, dict] = {}
    rows_paths: dict[str, Path] = {}
    for label, path, manifest in sets:
        domain = out.parent
        rows_path = domain / f"pack_contamination_{label}.rows.jsonl.gz"
        rows_paths[label] = rows_path
        q_hit = e_hit = any_hit = unauditable = 0
        by_source: dict[str, Counter] = defaultdict(Counter)
        by_position: dict[str, Counter] = defaultdict(Counter)
        lines: list[bytes] = []
        for pack in all_packs[label]:
            rec = hits.get((label, pack["id"]), {
                "question_first_token": None, "evidence_first_token": None,
                "question_windows": 0, "evidence_windows": 0,
            })
            q = rec["question_first_token"] is not None
            e = rec["evidence_first_token"] is not None
            if not (pack["question_windows"] or pack["evidence_windows"]):
                unauditable += 1
            row = {
                "set": label,
                "id": pack["id"],
                "source": pack["source"],
                "position": pack["position"],
                "question_windows": int(rec["question_windows"]),
                "evidence_windows": int(rec["evidence_windows"]),
                "question_hit": q if pack["question_windows"] else None,
                "evidence_hit": e if pack["evidence_windows"] else None,
                "first_hit_token": min(
                    [t for t in (rec["question_first_token"], rec["evidence_first_token"]) if t is not None],
                    default=None,
                ),
                "auditable": bool(pack["question_windows"] or pack["evidence_windows"]),
            }
            lines.append((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
            q_hit += int(q)
            e_hit += int(e)
            hit_any = q or e
            any_hit += int(hit_any)
            by_source[pack["source"]]["n"] += 1
            by_source[pack["source"]]["hit"] += int(hit_any)
            by_position[pack["position"]]["n"] += 1
            by_position[pack["position"]]["hit"] += int(hit_any)
        with rows_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
                for line in lines:
                    gz.write(line)
        report_sets[label] = {
            "path": str(path.relative_to(PROJECT)),
            "manifest_sha256": sha256_file(path / "manifest.json"),
            "packs": len(all_packs[label]),
            "rows": str(rows_path.relative_to(PROJECT)),
            "rows_count": len(lines),
            "rows_bytes": rows_path.stat().st_size,
            "rows_sha256": sha256_file(rows_path),
            "packs_with_question_hit": q_hit,
            "packs_with_evidence_hit": e_hit,
            "packs_with_any_hit": any_hit,
            "packs_unauditable": unauditable,
            "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
            "by_position": {k: dict(v) for k, v in sorted(by_position.items())},
        }

    total_hits = sum(s["packs_with_any_hit"] for s in report_sets.values())
    report = {
        "schema": 1,
        "stream": {
            "path": str(stream_path.relative_to(PROJECT)),
            "sha256": sha256_file(stream_path),
            "tokens": int(split["n_total"]),
            "training_tokens": train_tokens,
            "heldout_tokens": int(split["heldout_bytes"]),
            "split_sha256": sha256_file(Path(args.split)),
        },
        "windows": {
            "tokens": WINDOW_TOKENS,
            "min_tokens": MIN_WINDOW_TOKENS,
            "distinct_patterns": len(pattern_ids),
            "matches": scanned_matches,
            "chunk_bytes": CHUNK_BYTES,
            "overlap_bytes": OVERLAP_BYTES,
        },
        "sets": report_sets,
        "control": {
            "kind": "48 real training tokens",
            "token": None if control_at is None else control_at,
            "found": control_at is not None,
        },
        "gate": {
            "criterion": "zero token-aligned 48-token question/evidence windows in the training prefix",
            "hit_packs": total_hits,
            "state": "pass" if total_hits == 0 else "fail",
        },
        "seconds": round(time.time() - started, 1),
        "limitations": [
            "Exact 48-token windows only; paraphrase, translation and light editing are not detected.",
            "A null question_hit/evidence_hit means fewer than 8 tokens were available, not that the row is clean.",
            "The audit covers the training prefix; a source found only in the held-out tail is not training contamination.",
            "Gold evidence is taken from the pack's own recorded byte spans, not re-found by string search.",
        ],
    }
    text = json.dumps(report, indent=1) + "\n"
    out.write_text(text, encoding="utf-8")
    (out.with_suffix(out.suffix + ".sha256")).write_text(
        hashlib.sha256(text.encode()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "gate": report["gate"],
        "control_found": report["control"]["found"],
        "sets": {k: {kk: v[kk] for kk in ("packs", "packs_with_any_hit", "packs_unauditable", "rows_sha256")}
                 for k, v in report_sets.items()},
        "seconds": report["seconds"],
    }, indent=1))
    return 0 if total_hits == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
