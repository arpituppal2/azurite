#!/usr/bin/env python3
"""Render data/MANIFEST.json into DATA.md.

The manifest is the machine record; this is the human one. It is generated, not
hand-written, so a stale claim in DATA.md is impossible: if the manifest changes
and DATA.md does not, the difference shows up in git.

Licences are rendered exactly as the source states them. Where a source states
none, the row says UNRESOLVED. That is not formatting: several evaluation tasks
state no licence at all, and that fact has to survive into the record rather
than being smoothed over.

Usage:
    .venv/bin/python3 scripts/data_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "data" / "MANIFEST.json"
OUT = PROJECT / "DATA.md"

UNRESOLVED = {"", "none", "unknown", "other", "other-", "[]", "null"}
BYTES_PER_TOKEN = 4.0  # English web text, GPT-2-class tokenizer; an estimate


def lic(value) -> str:
    if value is None:
        return "UNRESOLVED"
    if isinstance(value, list):
        if not value:
            return "UNRESOLVED"
        value = ", ".join(str(v) for v in value)
    text = str(value).strip()
    if text.lower() in UNRESOLVED or not text:
        return "UNRESOLVED"
    return f"`{text}`"


def used_list(rec: dict) -> list[dict]:
    used = rec.get("used", [])
    return [used] if isinstance(used, dict) else list(used)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    env = manifest["environment"]
    all_recs = manifest["datasets"]
    tasks = {k: v for k, v in all_recs.items() if v.get("kind") == "task"}
    corpus = {k: v for k, v in all_recs.items() if v.get("kind") == "corpus"}
    models = manifest.get("models", {})

    primary = [k for k, v in tasks.items() if v.get("family") == "primary"]
    multi = [k for k, v in tasks.items() if v.get("family") == "multilingual"]
    secondary = [k for k, v in tasks.items() if v.get("family") == "secondary"]
    unmatched = [
        k for k, v in tasks.items() if v.get("family") not in
        ("primary", "multilingual", "secondary")
    ]

    def text_bytes(rec: dict) -> int:
        """Text payload, which is what the byte cap is written in.

        bytes_uncompressed is the whole decompressed file and is larger by the
        cost of JSON syntax and escaping; the two must not be mixed.
        """
        return rec.get("bytes_text", rec.get("bytes_uncompressed", 0))

    corpus_bytes = sum(text_bytes(r) for r in corpus.values())
    corpus_docs = sum(r.get("docs", 0) for r in corpus.values())
    corpus_disk = sum(r.get("bytes_compressed", 0) for r in corpus.values())
    task_rows = sum(
        s["rows"] for r in tasks.values() for s in r.get("splits", {}).values()
    )

    L: list[str] = []
    A = L.append
    A("# Data record\n\n")
    A(
        "Generated from `data/MANIFEST.json` by `scripts/data_report.py`. Every "
        "row was written to disk by `scripts/pull_datasets.py`; nothing here is "
        "transcribed by hand.\n\n"
    )
    A(
        f"- pulled: {manifest['created']}\n"
        f"- host: {env['machine']}, {env['os']}\n"
        f"- python {env['python']}, torch {env['torch']}, "
        f"MPS available: {env['mps_available']}\n"
        f"- datasets {env['datasets']}, huggingface_hub {env['huggingface_hub']}\n"
        f"- cache `{env['hf_home']}` and corpus shards in `data/` "
        "(project-local, gitignored)\n\n"
    )
    A(
        f"**Totals:** {len(tasks)} evaluation tasks, {task_rows:,} rows; "
        f"{len(corpus)} corpus sources, {corpus_docs:,} documents, "
        f"{corpus_bytes / 1e9:.2f} GB of text "
        f"(~{corpus_bytes / BYTES_PER_TOKEN / 1e9:.1f}B tokens at "
        f"{BYTES_PER_TOKEN:.0f} bytes/token) on {corpus_disk / 1e9:.2f} GB of disk; "
        f"{len(models)} competitor checkpoints.\n\n"
    )

    A("## Tasks: primary (a competitor publishes a number)\n\n")
    A("| task | measures | published comparison | rows |\n|---|---|---|---:|\n")
    for key in sorted(primary):
        rec = tasks[key]
        rows = sum(s["rows"] for s in rec["splits"].values())
        A(f"| `{key}` | {rec['task']} | {rec['competitor_basis']} | {rows:,} |\n")

    A("\n## Tasks: multilingual arm\n\n")
    A("| task | measures | published comparison | rows |\n|---|---|---|---:|\n")
    for key in sorted(multi):
        rec = tasks[key]
        rows = sum(s["rows"] for s in rec["splits"].values())
        A(f"| `{key}` | {rec['task']} | {rec['competitor_basis']} | {rows:,} |\n")

    A("\n## Tasks: secondary sanity suite\n\n")
    A(
        "No competitor number exists for these. They are reported in a separate "
        "table and are never folded into a headline claim.\n\n"
    )
    A("| task | measures | rows |\n|---|---|---:|\n")
    for key in sorted(secondary):
        rec = tasks[key]
        rows = sum(s["rows"] for s in rec["splits"].values())
        A(f"| `{key}` | {rec['task']} | {rows:,} |\n")
    if unmatched:  # pragma: no cover - defensive
        A(f"\nUnclassified tasks: {', '.join(sorted(unmatched))}\n")

    A("\n## Provenance per task\n")
    for key, rec in sorted(tasks.items()):
        canon = rec.get("canonical") or {}
        A(f"\n### `{key}` — {rec['task']}\n")
        A(f"- canonical: `{canon.get('repo', '?')}` "
          f"(licence as published: {lic(canon.get('license'))})\n")
        for u in used_list(rec):
            A(f"- pulled from: `{u['repo']}`"
              + (f", config `{u['config']}`" if u.get("config") else "")
              + f" (licence as published: "
                f"{lic((u.get('source_meta') or {}).get('license'))}), "
                f"commit `{str((u.get('source_meta') or {}).get('sha'))[:12]}`\n")
        if not used_list(rec):
            A("- pulled from: **NOTHING — every source failed**\n")
        if rec.get("mirror_note"):
            A(f"- caveat: {rec['mirror_note']}\n")
        A("\n| split | rows | expected | sha256 (first 12) |\n|---|---:|---|---|\n")
        for split, s in rec["splits"].items():
            if s.get("matches_expected") is True:
                verdict = "match"
            elif s.get("matches_expected") is False:
                verdict = "**MISMATCH**"
            else:
                verdict = "no published count"
            digest = ""
            if s.get("files"):
                digest = f"`{sorted(s['files'].items())[0][1]['sha256'][:12]}`"
            A(f"| {split} | {s['rows']:,} | {s['expected_rows'] if s['expected_rows'] is not None else '—'} "
              f"({verdict}) | {digest} |\n")
        A(f"- local path: `{rec['path']}`\n")

    A("\n## Pretraining corpus\n\n")
    A(
        "Sized against what the local machine can train, not against what the "
        "target run needs. A 30–90M parameter model is Chinchilla-optimal at "
        "0.6–1.8B tokens, so this corpus is already several times more text "
        "than any locally-trainable model can consume in a single pass; the "
        "binding constraint locally is compute. The target run's corpus would "
        "be streamed on the training node, because 300B tokens is ~600 GB "
        "tokenised and this machine has 64 GB free.\n"
    )
    for key, rec in sorted(corpus.items()):
        src = rec.get("source") or {}
        A(f"\n### `{key}`\n")
        A(f"- source `{src.get('repo')}`, configs {rec.get('configs')}, "
          f"commit `{str(src.get('sha'))[:12]}`\n")
        A(f"- licence as published: {lic(src.get('license'))}\n")
        A(f"- role: {rec.get('role')} (group `{rec.get('group')}`)\n")
        tb = text_bytes(rec)
        A(f"- **{rec.get('docs', 0):,} documents**, "
          f"{tb / 1e9:.2f} GB text, "
          f"~{tb / BYTES_PER_TOKEN / 1e9:.2f}B tokens, "
          f"cap {rec.get('cap_bytes', 0) / 1e9:.2f} GB, "
          f"{rec.get('bytes_compressed', 0) / 1e9:.2f} GB on disk, "
          f"{len(rec.get('shards', []))} shards\n")
        A("\n| shard | docs | GB text | GB file | MB disk | sha256 (first 12) |\n")
        A("|---|---:|---:|---:|---:|---|\n")
        for s in rec.get("shards", []):
            A(f"| `{Path(s['path']).name}` | {s['docs']:,} | "
              f"{s.get('bytes_text', s['bytes_uncompressed']) / 1e9:.2f} | "
              f"{s['bytes_uncompressed'] / 1e9:.2f} | "
              f"{s['bytes_compressed'] / 1e6:.0f} | `{s['sha256'][:12]}` |\n")

    if models:
        A("\n## Competitor checkpoints\n\n")
        A(
            "Parameter counts are read from the safetensors headers, not from "
            "the model card.\n\n"
        )
        A("| key | repo | licence | parameters | size | commit |\n")
        A("|---|---|---|---:|---:|---|\n")
        for key, rec in sorted(models.items()):
            if rec.get("error"):
                A(f"| `{key}` | `{rec.get('repo')}` | — | — | — | failed to pull |\n")
                continue
            A(f"| `{key}` | `{rec['repo']}` | {lic(rec.get('license'))} | "
              f"{rec.get('params_total', 0) / 1e6:.1f}M | "
              f"{rec.get('bytes_total', 0) / 1e9:.2f} GB | "
              f"`{str(rec.get('sha'))[:12]}` |\n")

    A("\n## Contamination\n\n")
    contam_files = sorted((PROJECT / "results" / "contam_check").glob("contam_*.json"))
    if contam_files:
        c = json.loads(contam_files[-1].read_text())
        A(
            f"Every test split is checked against the corpus before any training:"
            f" an exact {c.get('min_probe_chars')}-{c.get('max_probe_chars')}"
            f" character window from each sampled item, whitespace-insensitive,"
            f" run through one Aho-Corasick automaton over"
            f" {c.get('corpus_docs_scanned', 0):,} corpus documents"
            f" ({c.get('seconds', 0):.0f}s).\n\n"
        )
        A(
            f"**Total hits: {c.get('total_hits', 0)}** across"
            f" {len(c.get('tasks', {}))} tasks"
            f" ({sum(t['probes'] for t in c.get('tasks', {}).values()):,} probes)."
            f" Method: {c.get('method')}. Limitation: {c.get('limitations')}.\n\n"
        )
        A("| task | probes | hits | hit rate | weak hits (short probes) |\n")
        A("|---|---:|---:|---:|---:|\n")
        for key, t in sorted(c.get("tasks", {}).items()):
            rate = t["hits"] / t["probes"] if t["probes"] else 0.0
            A(
                f"| `{key}` | {t['probes']:,} | {t['hits']:,} | {rate:.2%} |"
                f" {t.get('weak_hits', 0):,} |\n"
            )
        A(
            f"\nFull record: `{contam_files[-1].relative_to(PROJECT)}`. The checker"
            " is itself verified by positive control: a probe taken from a known"
            " corpus document is recovered, so a zero is a result rather than a"
            " broken search.\n"
        )
    else:
        A("Not yet run. `scripts/contam_check.py` must run before any training.\n")

    A("\n## Licence status\n\n")
    unresolved = []
    for key, rec in sorted(tasks.items()):
        stated = [lic((rec.get("canonical") or {}).get("license"))]
        stated += [lic((u.get("source_meta") or {}).get("license"))
                   for u in used_list(rec)]
        if all(s == "UNRESOLVED" for s in stated):
            unresolved.append(key)
    A(
        f"{len(unresolved)} of {len(tasks)} tasks state **no licence** in either "
        "the canonical source or the mirror they were pulled from: "
        + ", ".join(f"`{k}`" for k in unresolved)
        + ".\n\nThis is recorded as unresolved rather than resolved by "
        "assumption. Before any result from these sets is shipped or published, "
        "each needs its terms established from the originating paper or "
        "repository. XNLI and Financial Phrasebank are known non-commercial.\n"
    )

    OUT.write_text("".join(L))
    print(f"wrote {OUT.relative_to(PROJECT)} ({len(''.join(L))} bytes)")
    print(
        f"tasks: {len(tasks)} (primary {len(primary)}, multilingual {len(multi)}, "
        f"secondary {len(secondary)}), corpus {len(corpus)} "
        f"({corpus_bytes / 1e9:.2f} GB, ~{corpus_bytes / BYTES_PER_TOKEN / 1e9:.1f}B tokens), "
        f"models {len(models)}, unresolved licences {len(unresolved)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
