#!/usr/bin/env python3
"""Audit selector training packs against the battery source-document patterns."""
from __future__ import annotations
import argparse, gzip, json, sys, time
from pathlib import Path
import ahocorasick
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from azurite.tokenizers import load  # noqa: E402

def packs(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    for shard in sorted(manifest["shards"], key=lambda s: s["index"]):
        with gzip.open(path / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-packs", type=int, default=0)
    args = ap.parse_args()
    pat = json.loads(Path(args.patterns).read_text())
    patterns = {bytes.fromhex(k): tuple(v) for k, v in pat["patterns"].items()}
    source = pat.get("source", {})
    auto = ahocorasick.Automaton()
    for raw, sids in patterns.items():
        auto.add_word(raw.decode("latin-1"), (raw, sids))
    auto.make_automaton()
    tok = load("bpe")
    scanned = 0
    hit_packs = []
    hit_sources = set()
    family = {}
    t0 = time.time()
    for pack in packs(Path(args.packs)):
        if args.max_packs and scanned >= args.max_packs:
            break
        scanned += 1
        text = str(pack.get("context", ""))
        for key in ("target_question", "question"):
            if pack.get(key):
                text += "\n" + str(pack[key])
        ids = np.asarray(tok.encode(text), dtype="<u2")
        raw = ids.tobytes()
        local = set()
        for end, (matched, sids) in auto.iter(raw.decode("latin-1")):
            off = end - len(matched) + 1
            if off % 2 == 0:
                local.update(sids)
        if local:
            hit_sources.update(local)
            hit_packs.append({"index": scanned, "id": pack.get("id"),
                              "source_docs": sorted(local)})
            for sid in local:
                fam = source.get(sid, {}).get("family", "unknown")
                family[fam] = family.get(fam, 0) + 1
        if scanned % 1000 == 0:
            print(f"scanned {scanned:,} packs; hit packs {len(hit_packs):,}", flush=True)
    rec = {
        "packs": str(args.packs), "packs_scanned": scanned,
        "patterns": len(patterns), "hit_pack_count": len(hit_packs),
        "hit_source_docs": len(hit_sources),
        "family_hits": family,
        "first_hit": hit_packs[0] if hit_packs else None,
        "hit_packs": hit_packs,
        "elapsed_s": round(time.time() - t0, 1),
        "gate": "PASS" if not hit_packs else "FAIL",
    }
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("packs_scanned", "hit_pack_count", "hit_source_docs", "gate")}, indent=1))
    print(json.dumps(rec, indent=1)[:3000])
    return 0 if not hit_packs else 1

if __name__ == "__main__":
    raise SystemExit(main())
