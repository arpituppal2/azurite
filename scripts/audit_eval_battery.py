#!/usr/bin/env python3
"""Contamination audit for the frozen 10k evaluation battery.

Two match levels:

  full    the whole question string, and the whole gold answer when it is
          contentful (>=24 chars), whitespace removed. This is the gate.
  window  up to three 40-character windows per field. Diagnostic only: it
          reports phrase-level overlap (Wikipedia boilerplate) without claiming
          the item itself was seen.

Matching is done by ripgrep (`-F`, fixed strings) over a whitespace-stripped
copy of the training corpus. The Python Aho-Corasick pass over the same text
was killed twice while scanning the 100k packs; ripgrep's matcher is the same
exact-string operation with bounded memory, and the project already vendors it.

Training data: the training portion of `byte_corpus_v2.u16` (first `n_train`
ids; the held-out tail is excluded) plus every context in the 100k selector
battery. For the packs the window scan is the search: a full question string
cannot occur there if none of its 40-character windows does.

Positive control: a window taken from the stripped training text must be found
by the same rg invocation; a search that cannot find its own control reports a
zero that means nothing.

Limitation, stated rather than hidden: verbatim only. Paraphrase, translation
and light editing are not detected.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

TOKENS = PROJECT / "data" / "tokens"
DEFAULT_PACKS = PROJECT / "results" / "parity" / "long_context_train_100k"
TMP = PROJECT / "results" / "parity" / ".audit_tmp"
WINDOW = 40
MIN_ANSWER = 24


def strip_ws(text: str) -> str:
    return re.sub(r"\s+", "", text)


def windows(text: str) -> list[str]:
    text = strip_ws(text)
    if len(text) < WINDOW:
        return []
    offsets = {0, max(0, (len(text) - WINDOW) // 2), len(text) - WINDOW}
    return list(dict.fromkeys(text[o: o + WINDOW] for o in sorted(offsets)))


def rg_binary() -> str:
    local = PROJECT / "bin" / "rg"
    if local.exists():
        return str(local)
    found = shutil.which("rg")
    if not found:
        raise SystemExit("ripgrep not found; vendored binary missing and no rg on PATH")
    return found


def strip_corpus(n_train: int, out_path: Path) -> int:
    import numpy as np

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path.stat().st_size
    raw = np.memmap(TOKENS / "byte_corpus_v2.u16", dtype="<u2", mode="r")
    written = 0
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        step = 16_000_000
        for start in range(0, n_train, step):
            end = min(start + step, n_train)
            chunk = np.asarray(raw[start:end], dtype=np.int16)
            text = (chunk - 4).clip(0, 255).astype(np.uint8).tobytes().decode("utf-8", "replace")
            data = strip_ws(text).encode("utf-8")
            fh.write(data)
            written += len(data)
            del chunk, text, data
    tmp.replace(out_path)
    return written


def rg_counts(patterns: list[str], target: Path) -> Counter:
    """Exact fixed-string search; returns how many times each pattern matched."""
    TMP.mkdir(parents=True, exist_ok=True)
    pat_file = TMP / "patterns.txt"
    pat_file.write_text("\n".join(patterns) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [rg_binary(), "-F", "-f", str(pat_file), "--text", "-o", "--no-filename", str(target)],
        capture_output=True, text=True,
    )
    if proc.returncode not in (0, 1):
        raise SystemExit(f"rg failed ({proc.returncode}): {proc.stderr[-400:]}")
    counts: Counter = Counter()
    for line in proc.stdout.splitlines():
        if line:
            counts[line] += 1
    return counts


def iter_pack_contexts(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    for shard in manifest["shards"]:
        with gzip.open(root / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)["context"]


def strip_packs(root: Path, out_path: Path) -> int:
    """Write one whitespace-stripped line per pack, streaming; return the count."""
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, "rb") as fh:
            return sum(1 for _ in fh)
    manifest = json.loads((root / "manifest.json").read_text())
    tmp = out_path.with_suffix(".tmp")
    n = 0
    with open(tmp, "wb") as out:
        for shard in manifest["shards"]:
            with gzip.open(root / shard["file"], "rt", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    out.write(strip_ws(row["context"]).encode("utf-8") + b"\n")
                    n += 1
    tmp.replace(out_path)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="results/parity/eval_battery_10k")
    ap.add_argument("--packs", default=str(DEFAULT_PACKS))
    ap.add_argument("--out", default="results/parity/eval_battery_contamination.json")
    args = ap.parse_args()

    root = Path(args.battery)
    manifest = json.loads((root / "manifest.json").read_text())

    full: dict[str, str] = {}
    wins: dict[str, str] = {}
    rows_by_probe: dict[str, set[str]] = {}
    row_meta: dict[str, dict] = {}
    rows_total = 0
    for shard in manifest["shards"]:
        with gzip.open(root / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows_total += 1
                rid = row["id"]
                question = strip_ws(row["question"])
                answer = row.get("answer")
                auditable = bool(answer) and len(strip_ws(answer)) >= MIN_ANSWER
                if len(question) >= 12:
                    full[question] = question
                    rows_by_probe.setdefault(question, set()).add(rid)
                if auditable:
                    a = strip_ws(answer)
                    full[a] = a
                    rows_by_probe.setdefault(a, set()).add(rid)
                for text in (row["question"], answer if auditable else ""):
                    for w in windows(text):
                        wins[w] = w
                        rows_by_probe.setdefault(w, set()).add(rid)
                row_meta[rid] = {
                    "family": row["family"],
                    "answer_auditable": auditable,
                    "short_answer": bool(answer) and not auditable,
                }
    print(f"rows {rows_total}; full probes {len(full)}; window probes {len(wins)}", flush=True)

    split = json.loads((TOKENS / "byte_corpus_v2.split.json").read_text())
    n_train = int(split["n_train"])
    TMP.mkdir(parents=True, exist_ok=True)
    stripped = TMP / "corpus_train.stripped"
    t0 = time.time()
    size = strip_corpus(n_train, stripped)
    print(f"stripped corpus: {size/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)

    # Positive control: 40 chars from the stripped training text itself.
    control = None
    with open(stripped, "rb") as fh:
        fh.seek(max(0, size // 2))
        chunk = fh.read(4096).decode("utf-8", "ignore")
    if len(chunk) >= 200:
        control = chunk[40:80]
    t1 = time.time()
    full_counts = rg_counts(list(full), stripped)
    print(f"full scan: {time.time()-t1:.0f}s, matched probes {len(full_counts)}", flush=True)
    t2 = time.time()
    win_counts = rg_counts(list(wins), stripped)
    print(f"window scan: {time.time()-t2:.0f}s, matched probes {len(win_counts)}", flush=True)
    control_found = False
    if control:
        control_counts = rg_counts([control], stripped)
        control_found = control_counts.get(control, 0) > 0

    pack_counts: Counter = Counter()
    pack_full_counts: Counter = Counter()
    n_packs = 0
    packs = Path(args.packs)
    t3 = time.time()
    if packs.exists():
        packs_stripped = TMP / "packs.stripped"
        n_packs = strip_packs(packs, packs_stripped)
        pack_full_counts = rg_counts(list(full), packs_stripped) if full else Counter()
        pack_counts = rg_counts(list(wins), packs_stripped) if wins else Counter()
    print(f"packs: {n_packs} in {time.time()-t3:.0f}s; full matches {len(pack_full_counts)}; window matches {len(pack_counts)}", flush=True)

    families: dict[str, dict] = {}
    full_hit_rows: list[dict] = []
    full_hit_probes: set[str] = set()
    window_hit_rows = 0
    for probe, rids in rows_by_probe.items():
        in_full = bool(full_counts.get(probe) or pack_full_counts.get(probe)) if probe in full else False
        in_win = bool(win_counts.get(probe)) or bool(pack_counts.get(probe))
        for rid in rids:
            meta = row_meta[rid]
            fam = families.setdefault(meta["family"], {
                "n": 0, "full_hits": 0, "window_hits": 0,
                "auditable_answers": 0, "short_answers_not_auditable": 0,
            })
            if in_full:
                fam["full_hits"] += 1
                if probe in full:
                    full_hit_probes.add(probe)
                if len(full_hit_rows) < 25:
                    full_hit_rows.append({
                        "id": rid, "family": meta["family"], "probe": probe[:80],
                        "corpus_matches": full_counts.get(probe, 0),
                    })
            if in_win:
                fam["window_hits"] += 1
                window_hit_rows += 1
    for rid, meta in row_meta.items():
        fam = families.setdefault(meta["family"], {
            "n": 0, "full_hits": 0, "window_hits": 0,
            "auditable_answers": 0, "short_answers_not_auditable": 0,
        })
        fam["n"] += 1
        fam["auditable_answers"] += int(meta["answer_auditable"])
        fam["short_answers_not_auditable"] += int(meta["short_answer"])

    report = {
        "battery": str(root),
        "battery_sha256": manifest.get("battery_sha256"),
        "gate": "full question and full gold-answer string, whitespace removed, exact fixed-string match",
        "diagnostic": "40-character windows, exact fixed-string match",
        "limitations": "verbatim only; paraphrase/translation/editing not detected",
        "min_auditable_answer_chars": MIN_ANSWER,
        "training_corpus": {
            "stream": "data/tokens/byte_corpus_v2.u16",
            "n_train_ids": n_train,
            "heldout_excluded": True,
            "stripped_bytes": size,
            "stream_control_found": control_found,
            "control_probe": control,
        },
        "packs": {
            "path": str(packs),
            "rows_scanned": n_packs,
            "full_probe_matches": sum(pack_full_counts.values()),
            "window_probe_matches": sum(pack_counts.values()),
        },
        "items": rows_total,
        "full_item_hits": sum(f["full_hits"] for f in families.values()),
        "window_item_hits": window_hit_rows,
        "families": families,
        "full_hit_examples": full_hit_rows,
        "full_hit_probes": sorted(full_hit_probes),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("items", "full_item_hits", "window_item_hits")}, indent=1))
    print("control found:", control_found)
    return 0 if report["full_item_hits"] == 0 and control_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
