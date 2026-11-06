#!/usr/bin/env python3
"""Remove every BPE document that contains an audited battery-source window.

The battery is not touched. The input is the existing uint16 token stream and a
portable pattern map produced by `audit_battery_source_docs.py`. A document is
the bytes between SEP_ID tokens; if any source-document window hits inside it,
the whole document is excluded. Training and held-out membership is preserved
by the old document-level boundary, and the output gets a new `.split.json`.

The script also rescans the entire corrected stream (not only the training
prefix) with the same exact 48-token patterns and a fresh stream-derived
positive control. It fails unless the corrected stream has zero hits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from array import array
from pathlib import Path

import ahocorasick
import numpy as np

SEP_ID = 3
TOKEN_CHUNK = 32_000_000
OVERLAP_BYTES = 94  # two 48-token patterns are 96 bytes; keep one token slack


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(
    stream: Path,
    patterns: dict[bytes, list[str]],
    *,
    max_bytes: int | None = None,
    control: bool = True,
) -> dict:
    """Scan a whole uint16 stream; return hits, separator positions, control."""
    size = stream.stat().st_size
    limit = size if max_bytes is None else min(size, max_bytes)
    max_len = max((len(k) for k in patterns), default=96)
    automaton = ahocorasick.Automaton()
    for raw, sids in patterns.items():
        automaton.add_word(raw.decode("latin-1"), (raw, tuple(sids)))
    automaton.make_automaton()

    control_raw = None
    if control:
        with stream.open("rb") as fh:
            fh.seek(max(0, (size // 2) // 2 * 2))
            control_raw = fh.read(max_len)
        if not control_raw:
            raise SystemExit("could not read stream control")
        automaton.add_word(control_raw.decode("latin-1"), (control_raw, ("__CONTROL__",)))
        automaton.make_automaton()

    hits: dict[str, list[int]] = {}
    hit_positions: set[int] = set()
    seps = array("Q")
    scanned = 0
    carry = b""
    found_control = False
    t0 = time.time()
    with stream.open("rb") as fh:
        while scanned < limit:
            take = min(TOKEN_CHUNK * 2, limit - scanned)
            chunk = fh.read(take)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype="<u2")
            seps.extend((np.flatnonzero(arr == SEP_ID) + scanned // 2).tolist())
            data = carry + chunk
            base = scanned - len(carry)
            for end, (matched, sids) in automaton.iter(data.decode("latin-1")):
                start = end - len(matched) + 1
                if start < 0:
                    continue
                absolute = base + start
                if absolute % 2:
                    continue
                tok = absolute // 2
                if tok < 0 or tok >= limit // 2:
                    continue
                if "__CONTROL__" in sids:
                    found_control = True
                    continue
                hit_positions.add(tok)
                for sid in sids:
                    hits.setdefault(sid, []).append(tok)
            carry = data[-OVERLAP_BYTES:] if len(data) > OVERLAP_BYTES else data
            scanned += len(chunk)
            if scanned % (512 * 1024 * 1024) < len(chunk):
                print(f"scan {scanned/1e9:.2f} GB; hits {len(hit_positions)}", flush=True)
    return {
        "hits": hits,
        "hit_positions": hit_positions,
        "seps": np.asarray(seps, dtype=np.int64),
        "control_found": found_control,
        "control_hex": control_raw.hex() if control_raw else None,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--old-n-train", type=int, required=True)
    ap.add_argument("--max-bytes", type=int, default=0, help="debug only")
    args = ap.parse_args()

    stream = Path(args.stream)
    out = Path(args.out)
    patterns_doc = json.loads(Path(args.patterns).read_text())
    patterns: dict[bytes, list[str]] = {
        bytes.fromhex(k): list(v) for k, v in patterns_doc["patterns"].items()
    }
    source = patterns_doc.get("source", {})
    print(f"scanning old stream {stream} with {len(patterns)} patterns", flush=True)
    old = scan(stream, patterns, max_bytes=args.max_bytes or None)
    seps = old["seps"]
    n_old = stream.stat().st_size // 2
    doc_starts = np.concatenate(([0], seps + 1)) if len(seps) else np.array([0])
    doc_ends = np.concatenate((seps, [n_old])) if len(seps) else np.array([n_old])
    if len(doc_starts) != len(doc_ends):
        raise SystemExit("document boundary length mismatch")
    drop = np.zeros(len(doc_starts), dtype=bool)
    for pos in old["hit_positions"]:
        # SEP at s means doc k ends at s; side='right' maps a token before s to k.
        drop[int(np.searchsorted(seps, pos, side="right"))] = True

    old_train = int(args.old_n_train)
    # The old stream's boundary was a byte offset and can fall inside a
    # document. Hold out that whole crossing document; never cut it at the
    # byte offset. This is the document-level split the protocol requires.
    crossing = np.flatnonzero((doc_starts < old_train) & (doc_ends > old_train))
    is_train = doc_ends <= old_train

    kept_train = np.flatnonzero(~drop & is_train)
    kept_held = np.flatnonzero(~drop & ~is_train)
    if not len(kept_train) or not len(kept_held):
        raise SystemExit("repair would leave an empty split")

    def runs(indices: np.ndarray) -> list[tuple[int, int]]:
        """Contiguous runs of kept document indices."""
        if not len(indices):
            return []
        out_runs = []
        first = prev = int(indices[0])
        for value in indices[1:]:
            value = int(value)
            if value != prev + 1:
                out_runs.append((first, prev))
                first = value
            prev = value
        out_runs.append((first, prev))
        return out_runs

    train_runs = runs(kept_train)
    held_runs = runs(kept_held)
    new_total = sum(
        int(doc_ends[b] - doc_starts[a]) for a, b in train_runs + held_runs
    )
    new_total += max(0, len(train_runs) - 1) + max(0, len(held_runs) - 1)
    if train_runs and held_runs:
        new_total += 1
    print(
        f"old docs {len(doc_starts):,}; dropped {int(drop.sum()):,}; "
        f"kept train {len(kept_train):,} held {len(kept_held):,}; "
        f"new tokens {new_total:,}",
        flush=True,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    dst = np.memmap(tmp, dtype="<u2", mode="w+", shape=(new_total,))
    src = np.memmap(stream, dtype="<u2", mode="r")
    pos = 0
    new_train = 0

    def copy_runs(group: list[tuple[int, int]], *, mark_train: bool) -> None:
        nonlocal pos, new_train
        for ri, (a, b) in enumerate(group):
            if ri:
                dst[pos] = SEP_ID
                pos += 1
            start = int(doc_starts[a])
            end = int(doc_ends[b])
            dst[pos:pos + (end - start)] = src[start:end]
            pos += end - start
            if mark_train:
                new_train = pos

    copy_runs(train_runs, mark_train=True)
    if train_runs and held_runs:
        dst[pos] = SEP_ID
        pos += 1
    copy_runs(held_runs, mark_train=False)
    if pos != new_total:
        raise SystemExit(f"wrote {pos} tokens, expected {new_total}")
    dst.flush()
    del dst, src
    tmp.replace(out)
    split_path = out.with_suffix(".split.json")
    split_path.write_text(
        json.dumps(
            {
                "stream": str(out),
                "n_total": new_total,
                "n_train": new_train,
                "heldout_bytes": new_total - new_train,
                "heldout_fraction": round((new_total - new_train) / new_total, 6),
                "old_n_train": old_train,
                "old_boundary_crossing_documents": int(len(crossing)),
                "old_training_tokens_after_document_fix": int(doc_ends[is_train].sum()),
                "excluded_documents": int(drop.sum()),
                "split_rule": "document held out iff sha1(doc) % 50 == 0; old byte boundary promoted to the enclosing document boundary; every document containing a battery-source window removed",
            },
            indent=1,
        )
        + "\n"
    )

    new_sha = sha256_file(out)
    print(f"corrected stream sha256 {new_sha}; auditing whole stream", flush=True)
    clean = scan(out, patterns)
    clean_hits = {sid: offsets for sid, offsets in clean["hits"].items()}
    # Build the exclusion list from every source that hit the old stream.
    excluded_sids = sorted(old["hits"])
    family_hits: dict[str, dict] = {}
    for sid in excluded_sids:
        meta = source.get(sid, {})
        fam = meta.get("family", "unknown")
        rec = family_hits.setdefault(fam, {"source_docs": 0, "rows": 0})
        rec["source_docs"] += 1
        rec["rows"] += len(meta.get("ids", []))
    record = {
        "old_stream": str(stream),
        "old_stream_sha256": patterns_doc.get("stream_sha256"),
        "new_stream": str(out),
        "new_stream_sha256": new_sha,
        "old_tokens": n_old,
        "new_tokens": new_total,
        "old_training_tokens": old_train,
        "old_boundary_crossing_documents": int(len(crossing)),
        "old_training_tokens_after_document_fix": int(doc_ends[is_train].sum()),
        "new_training_tokens": new_train,
        "new_heldout_tokens": new_total - new_train,
        "excluded_documents": int(drop.sum()),
        "excluded_source_docs": len(excluded_sids),
        "excluded_source_ids": excluded_sids,
        "old_hit_rows": sum(len(source.get(sid, {}).get("ids", [])) for sid in excluded_sids),
        "family_hits_old": family_hits,
        "audit": {
            "patterns": len(patterns),
            "whole_stream_hit_source_docs": len(clean_hits),
            "whole_stream_hit_positions": len(clean["hit_positions"]),
            "control_found": clean["control_found"],
            "control_hex": clean["control_hex"],
            "elapsed_s": clean["elapsed_s"],
            "gate": "PASS" if not clean_hits and clean["control_found"] else "FAIL",
        },
        "split": str(split_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in (
        "new_stream_sha256", "new_tokens", "new_training_tokens",
        "excluded_documents", "excluded_source_docs",
        "old_hit_rows")}, indent=1))
    print("AUDIT", json.dumps(record["audit"], indent=1))
    return 0 if record["audit"]["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
