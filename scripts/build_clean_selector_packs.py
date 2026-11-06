#!/usr/bin/env python3
"""Build a selector training view containing only source-clean packs.

Reads packs in manifest order, tokenizes each context, and scans it with the
same exact 48-token source-document patterns used by the stream audit. A pack
with any hit is excluded whole. The first `--max-packs` clean packs are written
to a new sharded directory; every written pack has therefore been checked.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, sys, time
from pathlib import Path
import ahocorasick
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from azurite.tokenizers import load  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iter_manifest(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    for shard in sorted(manifest["shards"], key=lambda s: s["index"]):
        with gzip.open(path / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-packs", type=int, default=31000)
    ap.add_argument("--packs-per-shard", type=int, default=1000)
    ap.add_argument("--record", required=True)
    args = ap.parse_args()
    src_manifest = json.loads((Path(args.packs) / "manifest.json").read_text())
    pat = json.loads(Path(args.patterns).read_text())
    patterns = {bytes.fromhex(k): tuple(v) for k, v in pat["patterns"].items()}
    source = pat.get("source", {})
    auto = ahocorasick.Automaton()
    for raw, sids in patterns.items():
        auto.add_word(raw.decode("latin-1"), (raw, sids))
    auto.make_automaton()
    tok = load("bpe")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    buf = []
    clean = 0
    scanned = 0
    hit_packs = 0
    hit_sources = set()
    family_hits: dict[str, int] = {}
    first_hit = None
    t0 = time.time()

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        idx = len(shards)
        name = f"shard-{idx:05d}.jsonl.gz"
        p = out_dir / name
        with gzip.open(p, "wt", encoding="utf-8", compresslevel=6) as fh:
            for row in buf:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        shards.append({"index": idx, "file": name, "packs": len(buf),
                       "gzip_bytes": p.stat().st_size, "sha256": sha256(p)})
        buf = []

    for pack in iter_manifest(Path(args.packs)):
        if clean >= args.max_packs:
            break
        scanned += 1
        text = str(pack.get("context", ""))
        for key in ("target_question", "question"):
            if pack.get(key):
                text += "\n" + str(pack[key])
        ids = np.asarray(tok.encode(text), dtype="<u2")
        local = set()
        for end, (matched, sids) in auto.iter(ids.tobytes().decode("latin-1")):
            if (end - len(matched) + 1) % 2 == 0:
                local.update(sids)
        if local:
            hit_packs += 1
            hit_sources.update(local)
            if first_hit is None:
                first_hit = {"scan_index": scanned, "pack_id": pack.get("id"),
                             "source_docs": sorted(local)}
            for sid in local:
                fam = source.get(sid, {}).get("family", "unknown")
                family_hits[fam] = family_hits.get(fam, 0) + 1
            continue
        buf.append(pack)
        clean += 1
        if len(buf) >= args.packs_per_shard:
            flush()
        if clean % 1000 == 0:
            print(f"clean {clean:,} / scanned {scanned:,}; hits {hit_packs:,}", flush=True)
    flush()
    manifest = {
        "schema": 3, "format": "jsonl.gz",
        "config": {**src_manifest.get("config", {}),
                   "packs": clean, "packs_per_shard": args.packs_per_shard,
                   "battery": "noquestion_source_clean",
                   "source_audit_patterns": len(patterns)},
        "shards": shards,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    record = {
        "source_packs": str(args.packs), "out_dir": str(out_dir),
        "source_packs_scanned": scanned, "clean_packs": clean,
        "excluded_packs": hit_packs, "excluded_source_docs": len(hit_sources),
        "family_hits": family_hits, "first_hit": first_hit,
        "pattern_count": len(patterns), "manifest_sha256": sha256(out_dir / "manifest.json"),
        "shards": len(shards), "elapsed_s": round(time.time() - t0, 1),
        "gate": "PASS" if clean >= args.max_packs else "FAIL",
    }
    Path(args.record).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("source_packs_scanned", "clean_packs", "excluded_packs", "excluded_source_docs", "gate")}, indent=1))
    return 0 if record["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
