#!/usr/bin/env python3
"""One screen of what this checkout actually has.

Reads the manifest and the filesystem, prints counts, sizes, machine headroom
and the measured local envelope. No claims, only what is on disk.

Usage:
    .venv/bin/python3 scripts/status.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "data" / "MANIFEST.json"
BENCH_DIR = PROJECT / "results" / "local_bench"
BYTES_PER_TOKEN = 4.0


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    for part in out.split():
        pass
    import re

    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else float("nan")


def main() -> int:
    if not MANIFEST.exists():
        print("no manifest: nothing pulled yet")
        return 1
    m = json.loads(MANIFEST.read_text())
    ds = m["datasets"]
    tasks = {k: v for k, v in ds.items() if v.get("kind") == "task"}
    corpus = {k: v for k, v in ds.items() if v.get("kind") == "corpus"}
    models = m.get("models", {})

    rows = sum(s["rows"] for r in tasks.values() for s in r.get("splits", {}).values())
    c_docs = sum(r.get("docs", 0) for r in corpus.values())
    c_text = sum(r.get("bytes_text", r.get("bytes_uncompressed", 0)) for r in corpus.values())
    c_disk = sum(r.get("bytes_compressed", 0) for r in corpus.values())
    c_shards = sum(len(r.get("shards", [])) for r in corpus.values())
    incomplete = [k for k, r in corpus.items() if not r.get("shards")]

    print(f"manifest written: {m['created']}")
    print(f"\ntasks ({len(tasks)}): {rows:,} rows")
    for fam in ("primary", "multilingual", "secondary", None):
        keys = [k for k, r in tasks.items() if r.get("family") == fam]
        if keys:
            label = fam or "unclassified"
            print(f"  {label:13s} {len(keys):2d}  " + ", ".join(sorted(keys))[:150])

    print(f"\ncorpus ({len(corpus)} sources, {c_shards} shards): "
          f"{c_docs:,} docs, {c_text / 1e9:.2f} GB text "
          f"(~{c_text / BYTES_PER_TOKEN / 1e9:.1f}B tokens), {c_disk / 1e9:.2f} GB on disk")
    for k, r in sorted(corpus.items()):
        state = "" if r.get("shards") else "  <-- EMPTY"
        print(f"  {k:22s} {r.get('docs', 0):>10,d} docs  "
              f"{r.get('bytes_text', r.get('bytes_uncompressed', 0)) / 1e9:6.2f}/"
              f"{r.get('cap_bytes', 0) / 1e9:5.2f} GB  {len(r.get('shards', []))} shards{state}")

    if models:
        print(f"\ncheckpoints ({len(models)}):")
        for k, r in sorted(models.items()):
            if r.get("error"):
                print(f"  {k:22s} FAILED: {r['error'][:60]}")
                continue
            print(f"  {k:22s} {r.get('params_total', 0) / 1e6:7.1f}M  "
                  f"{r.get('bytes_total', 0) / 1e9:5.2f} GB  {r.get('license')}")

    print(f"\ndisk: {shutil.disk_usage(PROJECT).free / 1e9:.1f} GB free; "
          f"data/ {dir_size(PROJECT / 'data') / 1e9:.2f} GB, "
          f".archive/ {dir_size(PROJECT / '.archive') / 1e9:.2f} GB, "
          f".venv/ {dir_size(PROJECT / '.venv') / 1e9:.2f} GB")
    print(f"swap in use: {swap_used_mb():.0f} MB")

    benches = sorted(BENCH_DIR.glob("bench_*.json"))
    if benches:
        latest = json.loads(benches[-1].read_text())
        print(f"\nmeasured envelope ({benches[-1].name}):")
        for r in latest["runs"]:
            if r["status"] == "ok":
                print(f"  {r['params']:>10,d} params  {r['tokens_per_s']:>9,.0f} tok/s  "
                      f"{r['achieved_tflops']:.2f} TFLOP/s  {r['peak_mps_gib']:.2f} GiB peak")
            else:
                print(f"  {r['params'] if 'params' in r else '?':>10} FAILED: "
                      f"{r.get('error', '')[:60]}")
    if incomplete:
        print(f"\nINCOMPLETE: {', '.join(incomplete)} — pull not finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
