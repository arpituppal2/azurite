#!/usr/bin/env python3
"""Pull every public dataset the Azurite study needs, into ./data.

Rules this script enforces so the study stays honest and stays local:

  * Nothing leaves the project. HF_HOME and the datasets cache are set to
    ./data/.hf before huggingface_hub is imported, so no cache lands in $HOME
    and no artefact is invisible to the manifest.
  * Every artefact is counted and hashed. The manifest is the provenance
    record: HF commit sha, licence as published, rows per split, sha256 of
    every file written, and which mirror was actually used.
  * Canonical repos that only ship a loading script are served through a
    parquet mirror (datasets>=4 refuses scripts). The mirror is named in the
    manifest, never hidden -- a mirror can differ from the canonical release,
    so row counts are checked against published canonical counts and
    mismatches are reported, not swallowed.
  * Corpus text is written in ~1 GiB shards, not one giant file: a single
    11 GB file cannot be partially read, verified or resumed.
  * RAM discipline. One dataset is materialised at a time and released; the
    corpus is streamed and written, never loaded.

Usage:
    .venv/bin/python3 scripts/pull_datasets.py --only corpus-english
    .venv/bin/python3 scripts/pull_datasets.py            # everything
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RAW = DATA / "raw"
CORPUS = DATA / "corpus"
MANIFEST = DATA / "MANIFEST.json"

os.environ.setdefault("HF_HOME", str(DATA / ".hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DATA / ".hf" / "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import huggingface_hub  # noqa: E402
import datasets as hf_datasets  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

SHARD_TARGET_BYTES = 1 << 30  # ~1 GiB of uncompressed text per shard
WINDOW_FILES = 3  # parquet files downloaded in parallel per window; these
# corpora ship ~2 GB files, so a large window means downloading gigabytes that
# the byte cap will never consume


# --------------------------------------------------------------------------
# Evaluation and supervision tasks.
#
# `primary` tasks are the ones a competitor publishes a number on, so a
# like-for-like comparison exists. `secondary` tasks are standard sanity
# checks with no competitor number; they are reported separately and never
# folded into a headline. `expect` holds canonical row counts where published.
# --------------------------------------------------------------------------
TASKS: list[dict] = [
    {
        "key": "banking77",
        "canonical": "PolyAI/banking77",
        "sources": [{"repo": "mteb/banking77", "config": None}],
        "splits": ["train", "test"],
        "task": "77-way single-domain intent",
        "family": "primary",
        "competitor_basis": "Laya (Banking77), Jev (Banking77 79.71)",
        "expect": {"train": 10003, "test": 3080},
        "mirror_note": "canonical ships only a loading script; datasets>=4 refuses it",
    },
    {
        "key": "clinc_oos",
        "canonical": "clinc/clinc_oos",
        "sources": [{"repo": "clinc/clinc_oos", "config": "plus"}],
        "splits": ["train", "validation", "test"],
        "task": "150-way intent + out-of-scope",
        "family": "primary",
        "competitor_basis": "Jev (Clinc150 89.45)",
        "expect": {"train": 15250, "validation": 3100, "test": 5500},
        "mirror_note": None,
    },
    {
        "key": "ag_news",
        "canonical": "fancyzhx/ag_news",
        "sources": [{"repo": "fancyzhx/ag_news", "config": None}],
        "splits": ["train", "test"],
        "task": "4-way news topic",
        "family": "primary",
        "competitor_basis": "Laya (AG News), Jev (AG News 88.51)",
        "expect": {"train": 120000, "test": 7600},
        "mirror_note": None,
    },
    {
        "key": "emotion",
        "canonical": "dair-ai/emotion",
        "sources": [{"repo": "dair-ai/emotion", "config": "split"}],
        "splits": ["train", "validation", "test"],
        "task": "6-way emotion",
        "family": "primary",
        "competitor_basis": "Laya (DAIR Emotion), Jev (Emotion 58.50)",
        "expect": {"train": 16000, "validation": 2000, "test": 2000},
        "mirror_note": None,
    },
    {
        "key": "massive_intent",
        "canonical": "AmazonScience/massive",
        "sources": [{"repo": "mteb/amazon_massive_intent", "config": "en"}],
        "splits": ["train", "validation", "test"],
        "task": "60-way assistant intent (en-US)",
        "family": "primary",
        "competitor_basis": "Laya (MASSIVE intent)",
        "expect": {"train": 11514, "validation": 2033, "test": 2974},
        "mirror_note": "canonical ships only a loading script; datasets>=4 refuses it",
    },
    {
        "key": "massive_scenario",
        "canonical": "AmazonScience/massive",
        "sources": [{"repo": "mteb/amazon_massive_scenario", "config": "en"}],
        "splits": ["train", "validation", "test"],
        "task": "18-way assistant scenario (en-US)",
        "family": "primary",
        "competitor_basis": "secondary axis of the same corpus",
        "expect": {"train": 11514, "validation": 2033, "test": 2974},
        "mirror_note": "canonical ships only a loading script; datasets>=4 refuses it",
    },
    {
        "key": "xnli",
        "canonical": "facebook/xnli",
        "sources": [{"repo": "facebook/xnli", "config": "en"}],
        "splits": ["train", "validation", "test"],
        "task": "3-way natural language inference (en)",
        "family": "primary",
        "competitor_basis": "Laya (XNLI); non-commercial licence",
        "expect": {"train": 392702, "validation": 2490, "test": 5010},
        "mirror_note": None,
    },
    {
        "key": "language_id",
        "canonical": "papluca/language-identification",
        "sources": [{"repo": "papluca/language-identification", "config": None}],
        "splits": ["train", "validation", "test"],
        "task": "20-way language identification",
        "family": "primary",
        "competitor_basis": "Jev (Language ID 99.64)",
        "expect": {"train": 70000, "validation": 10000, "test": 10000},
        "mirror_note": None,
    },
    {
        "key": "financial_phrasebank",
        "canonical": "takala/financial_phrasebank",
        "sources": [{"repo": "FinanceMTEB/financial_phrasebank", "config": None}],
        "splits": ["train", "test"],
        "task": "3-way financial sentiment (all-agree subset)",
        "family": "primary",
        "competitor_basis": "Jev (Financial Phrasebank 72.99); non-commercial licence",
        "expect": {"train": 1264, "test": 1000},
        "mirror_note": (
            "canonical has four agreement-level configs behind a loading script;"
            " this mirror's level is established by arithmetic: 1,264 + 1,000 ="
            " 2,264 = canonical sentences_allagree"
        ),
    },
    # --- multilingual arm: Laya is multilingual, so the comparison needs it ---
    *[
        {
            "key": f"xnli_{lang}",
            "canonical": "facebook/xnli",
            "sources": [{"repo": "facebook/xnli", "config": lang}],
            "splits": ["validation", "test"],
            "task": f"3-way NLI ({lang})",
            "family": "multilingual",
            "competitor_basis": "Laya (multilingual XNLI)",
            "expect": {"validation": 2490, "test": 5010},
            "mirror_note": None,
        }
        for lang in ("de", "fr", "es", "zh")
    ],
    {
        "key": "massive_intent_de",
        "canonical": "AmazonScience/massive",
        "sources": [{"repo": "mteb/amazon_massive_intent", "config": "de"}],
        "splits": ["validation", "test"],
        "task": "60-way assistant intent (de-DE)",
        "family": "multilingual",
        "competitor_basis": "Laya (multilingual MASSIVE)",
        "expect": {"validation": 2033, "test": 2974},
        "mirror_note": "canonical ships only a loading script; datasets>=4 refuses it",
    },
    # --- secondary sanity suite: no competitor number, reported separately ---
    {
        "key": "hellaswag",
        "canonical": "Rowan/hellaswag",
        "sources": [{"repo": "Rowan/hellaswag", "config": None}],
        "splits": ["train", "validation"],
        "task": "10-way commonsense completion",
        "family": "secondary",
        "competitor_basis": "none — standard LM sanity check",
        "expect": {"train": 39905, "validation": 10042},
        "mirror_note": "test labels are not public; validation is the usable split",
    },
    {
        "key": "arc_easy",
        "canonical": "allenai/ai2_arc",
        "sources": [{"repo": "allenai/ai2_arc", "config": "ARC-Easy"}],
        "splits": ["train", "validation", "test"],
        "task": "multiple-choice science questions (easy)",
        "family": "secondary",
        "competitor_basis": "none — standard LM sanity check",
        "expect": None,
        "mirror_note": (
            "one key per config: two configs under one key overwrite each"
            " other's split records, which is how a silent data swap happens"
        ),
    },
    {
        "key": "arc_challenge",
        "canonical": "allenai/ai2_arc",
        "sources": [{"repo": "allenai/ai2_arc", "config": "ARC-Challenge"}],
        "splits": ["train", "validation", "test"],
        "task": "multiple-choice science questions (challenge)",
        "family": "secondary",
        "competitor_basis": "none — standard LM sanity check",
        "expect": None,
        "mirror_note": None,
    },
    {
        "key": "winogrande",
        "canonical": "allenai/winogrande",
        "sources": [{"repo": "allenai/winogrande", "config": "winogrande_xl"}],
        "splits": ["train", "validation"],
        "task": "2-choice coreference resolution",
        "family": "secondary",
        "competitor_basis": "none — standard LM sanity check",
        "expect": {"train": 40398, "validation": 1267},
        "mirror_note": None,
    },
    {
        "key": "lambada_openai",
        "canonical": "EleutherAI/lambada_openai",
        "sources": [{"repo": "EleutherAI/lambada_openai", "config": "en"}],
        "splits": ["test"],
        "task": "last-word prediction (LM)",
        "family": "secondary",
        "competitor_basis": "none — standard LM sanity check",
        "expect": {"test": 5153},
        "mirror_note": None,
    },
    {
        "key": "wikitext103",
        "canonical": "Salesforce/wikitext",
        "sources": [{"repo": "Salesforce/wikitext", "config": "wikitext-103-raw-v1"}],
        "splits": ["train", "validation", "test"],
        "task": "held-out perplexity corpus",
        "family": "secondary",
        "competitor_basis": "none — the standard perplexity reference",
        "expect": None,
        "mirror_note": (
            "expectations left empty on purpose: the commonly quoted 4,358/"
            "4,358 counts belong to wikitext-103-v1 (tokenised). raw-v1 has"
            " different row counts and asserting the wrong one produces a"
            " false MISMATCH flag"
        ),
    },
]

# --------------------------------------------------------------------------
# Pretraining corpus. Caps are uncompressed text bytes and sum to the
# --corpus-gib budget. The point of the size is that a local study with 10-90M
# parameter models has more tokens than it can consume, so the bottleneck is
# compute rather than data, and the same pipeline can serve a larger run
# without a data scramble.
# --------------------------------------------------------------------------
CORPUS_SOURCES: list[dict] = [
    {
        "key": "fineweb_edu",
        "repo": "HuggingFaceFW/fineweb-edu",
        "configs": ["sample-10BT"],
        "split": "train",
        "text_field": "text",
        "cap_gib": 12.0,
        "role": "primary pretraining text (ODC-BY, filtered educational web)",
        "group": "english",
    },
    {
        "key": "fineweb",
        "repo": "HuggingFaceFW/fineweb",
        "configs": ["sample-10BT"],
        "split": "train",
        "text_field": "text",
        "cap_gib": 3.0,
        "role": "general unfiltered web text to dilute the edu filter",
        "group": "english",
    },
    {
        "key": "wikipedia_en",
        "repo": "wikimedia/wikipedia",
        "configs": ["20231101.en"],
        "split": "train",
        "text_field": "text",
        "cap_gib": 3.0,
        "role": "encyclopaedic prose, clean register",
        "group": "english",
    },
    {
        "key": "finemath",
        "repo": "HuggingFaceTB/finemath",
        "configs": ["finemath-3plus"],
        "split": "train",
        "text_field": "text",
        "cap_gib": 0.6,
        "role": "math reasoning text (ablation lever: does it help intent tasks?)",
        "group": "english",
    },
    {
        "key": "wikipedia_multilingual",
        "repo": "wikimedia/wikipedia",
        "configs": ["20231101.de", "20231101.fr", "20231101.es", "20231101.zh"],
        "split": "train",
        "text_field": "text",
        "cap_gib": 2.0,
        "role": "multilingual arm: de/fr/es/zh prose",
        "group": "multilingual",
    },
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dir_hashes(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(path.rglob("*")):
        if f.is_file():
            out[str(f.relative_to(path))] = {
                "bytes": f.stat().st_size,
                "sha256": sha256_file(f),
            }
    return out


def repo_meta(api: HfApi, repo: str) -> dict:
    try:
        info = api.dataset_info(repo)
    except Exception as exc:  # pragma: no cover - network dependent
        return {"repo": repo, "error": f"{type(exc).__name__}: {exc}"}
    card = getattr(info, "cardData", None) or {}
    return {
        "repo": repo,
        "sha": getattr(info, "sha", None),
        "last_modified": str(getattr(info, "lastModified", "")),
        "license": card.get("license"),
        "license_name": card.get("license_name"),
        "gated": getattr(info, "gated", None),
    }


def pull_task(api: HfApi, entry: dict, out: dict, force: bool = False) -> None:
    key = entry["key"]
    dest = RAW / key
    record: dict = {
        "kind": "task",
        "task": entry["task"],
        "family": entry["family"],
        "competitor_basis": entry["competitor_basis"],
        "canonical": repo_meta(api, entry["canonical"]),
        "mirror_note": entry["mirror_note"],
        "attempts": [],
        "splits": {},
        "path": str(dest.relative_to(PROJECT)),
    }
    print(f"\n=== {key} (canonical {entry['canonical']}) ===", flush=True)

    def take(repo: str, cfg: str | None, split: str):
        split_dir = dest / (cfg or "default") / split
        if split_dir.exists() and not force:
            ds = hf_datasets.load_from_disk(str(split_dir))
            rows, cols = ds.num_rows, ds.column_names
            del ds
            return rows, cols, True, None
        t0 = time.time()
        ds = hf_datasets.load_dataset(repo, cfg, split=split)
        rows, cols = ds.num_rows, ds.column_names
        split_dir.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(split_dir))
        del ds
        return rows, cols, False, round(time.time() - t0, 2)

    loaded = False
    for source in entry["sources"]:
        repo, cfg = source["repo"], source["config"]
        attempt = {"repo": repo, "config": cfg, "splits": {}, "errors": {}}
        try:
            configs = hf_datasets.get_dataset_config_names(repo)
            if cfg is not None and cfg not in configs:
                attempt["errors"]["config"] = f"{cfg!r} not in {configs[:8]}"
                record["attempts"].append(attempt)
                continue
            for split in entry["splits"]:
                try:
                    rows, cols, cached, secs = take(repo, cfg, split)
                    expected = (entry.get("expect") or {}).get(split)
                    attempt["splits"][split] = {
                        "rows": rows,
                        "columns": cols,
                        "cached": cached,
                        "seconds": secs,
                        "expected_rows": expected,
                        "matches_expected": None if expected is None else rows == expected,
                        "files": dir_hashes(dest / (cfg or "default") / split),
                    }
                    flag = "" if expected in (None, rows) else f"  <-- expected {expected}"
                    print(f"  {cfg or 'default'}/{split}: {rows} rows{flag}")
                except Exception as exc:
                    attempt["errors"][split] = f"{type(exc).__name__}: {exc}"[:300]
                    print(f"  {cfg or 'default'}/{split}: FAILED {type(exc).__name__}")
            if attempt["splits"] and not attempt["errors"]:
                record["splits"].update(attempt["splits"])
                record.setdefault("used", []).append(
                    {
                        "repo": repo,
                        "config": cfg,
                        "source_meta": repo_meta(api, repo),
                    }
                )
                loaded = True
            record["attempts"].append(attempt)
        except Exception as exc:
            attempt["errors"]["configs"] = f"{type(exc).__name__}: {exc}"[:300]
            record["attempts"].append(attempt)

    if not loaded:
        print(f"  !! {key}: every source failed; nothing written")
    out[key] = record


def gzip_uncompressed_bytes(shard: Path, chunk: int = 1 << 22) -> int:
    """Exact uncompressed payload of a gzipped file.

    Streams the file through zlib. A first attempt read the ISIZE field of the
    gzip trailer instead; that reported 51,356 GB for a 406 MB shard, because
    finding the next gzip member requires real deflate parsing, not scanning
    for a zero byte. The slow correct version is the only version kept.
    """
    import zlib

    size = 0
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with shard.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            size += len(dec.decompress(block))
    return size


def corpus_record_ok(rec: dict | None) -> bool:
    """A corpus record is complete only if it reached its cap or ran out of data.

    Checking that the shard files exist is not enough: a source that stopped at
    1.29 GB of a 3.00 GB budget looked 'cached' on the next run and kept the
    shortfall forever, while the manifest still advertised the larger cap.
    """
    if not rec or not rec.get("shards"):
        return False
    for shard in rec["shards"]:
        p = PROJECT / shard["path"]
        if not p.exists() or p.stat().st_size != shard["bytes_compressed"]:
            return False
    text = rec.get("bytes_text", rec.get("bytes_uncompressed", 0))
    return bool(text >= rec.get("cap_bytes", 0) * 0.98 or rec.get("exhausted"))


def _norm(s: str) -> str:
    """Normalise a config name or a path prefix for comparison.

    The hub does not spell configs the way the filesystem does: FineWeb's
    config `sample-10BT` lives under `sample/10BT`, so a literal prefix match
    silently returns nothing. Comparing alphanumerics only makes both spellings
    the same string.
    """
    return "".join(ch for ch in s.lower() if ch.isalnum())


def config_files(api: HfApi, repo: str, config: str, revision: str | None) -> list:
    """Parquet files belonging to one config of a HF dataset repo, sorted."""
    info = api.dataset_info(repo, files_metadata=True, revision=revision)
    files = [s for s in info.siblings if s.rfilename.endswith(".parquet")]
    if not files:
        return []
    target = _norm(config)

    # 1. directory prefixes of every parquet file, matched after normalisation
    prefixes: set[str] = set()
    for s in files:
        parts = s.rfilename.split("/")[:-1]
        for depth in (1, 2, 3):
            if len(parts) >= depth:
                prefixes.add("/".join(parts[:depth]))
    hits = [p for p in prefixes if target and target in _norm(p)]
    if hits:
        best = max(hits, key=len)
        sel = [s for s in files if s.rfilename.startswith(best + "/")]
        if sel:
            return sorted(sel, key=lambda s: s.rfilename)

    # 2. fall back to a literal substring match on the file name
    sel = [s for s in files if config in s.rfilename]
    return sorted(sel, key=lambda s: s.rfilename)


class ShardWriter:
    """Writes ~1 GiB gzip shards of {"text": ...} JSONL, counting as it goes.

    compresslevel=1 trades ~20% disk for roughly double the write speed. The
    uncompressed count is tracked in process rather than recovered from the
    file afterwards: recovering it correctly means decompressing the shard
    again, which verify_data.py does, and doing it twice is waste.
    """

    def __init__(self, key: str, record: dict):
        self.key = key
        self.record = record
        self.fh = None
        self.path: Path | None = None
        self.docs = 0
        self.bytes_text = 0  # payload of the text fields; this is what the cap counts
        self.bytes_file = 0  # whole decompressed file, JSON syntax and escaping included
        self.start = time.time()

    def write(self, text: str) -> int:
        if self.fh is None:
            self.path = CORPUS / f"{self.key}.{len(self.record['shards']):03d}.jsonl.gz"
            self.fh = gzip.open(self.path, "wt", encoding="utf-8", compresslevel=1)
            self.start = time.time()
            self.docs = 0
            self.bytes_text = 0
            self.bytes_file = 0
        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
        self.fh.write(line)
        n = len(text.encode("utf-8"))
        self.docs += 1
        self.bytes_text += n
        self.bytes_file += len(line.encode("utf-8"))
        if self.bytes_file >= SHARD_TARGET_BYTES:
            self.close()
        return n

    def close(self) -> None:
        if self.fh is None:
            return
        self.fh.close()
        rec = _shard_record(
            self.path, self.docs, time.time() - self.start,
            self.bytes_file, self.bytes_text,
        )
        self.record["shards"].append(rec)
        print(
            f"    closed {self.path.name}: {self.docs:,} docs, "
            f"{rec['bytes_uncompressed'] / 1e9:.2f} GB file "
            f"({rec['bytes_text'] / 1e9:.2f} GB text), "
            f"{rec['bytes_compressed'] / 1e6:.0f} MB on disk",
            flush=True,
        )
        self.fh = None


def pull_corpus(api: HfApi, sources: list[dict], out: dict) -> None:
    import pyarrow.parquet as pq

    CORPUS.mkdir(parents=True, exist_ok=True)
    for src in sources:
        key = src["key"]
        cap_bytes = int(src["cap_gib"] * 1e9)
        per_config = cap_bytes // len(src["configs"])
        revision = (repo_meta(api, src["repo"]) or {}).get("sha")
        record: dict = {
            "kind": "corpus",
            "role": src["role"],
            "group": src["group"],
            "source": repo_meta(api, src["repo"]),
            "configs": src["configs"],
            "text_field": src["text_field"],
            "cap_bytes": cap_bytes,
            "shards": [],
        }
        print(
            f"\n=== corpus {key} (cap {cap_bytes / 1e9:.2f} GB, "
            f"{len(src['configs'])} config(s), revision {str(revision)[:8]}) ===",
            flush=True,
        )

        if corpus_record_ok(out.get(key)) and out[key].get("cap_bytes") == cap_bytes:
            record = out[key]
            record["cached"] = True
            print(
                f"  cached: {record['docs']:,} docs, "
                f"{record['bytes_uncompressed'] / 1e9:.2f} GB in "
                f"{len(record['shards'])} shards"
            )
            out[key] = record
            continue

        for stale in CORPUS.glob(f"{key}.*.jsonl.gz"):
            stale.unlink()

        writer = ShardWriter(key, record)
        docs = written = 0
        t0 = time.time()
        for cfg in src["configs"]:
            entries = config_files(api, src["repo"], cfg, revision)
            if not entries:
                print(f"  !! {cfg}: no parquet files found")
                continue
            stage = DATA / ".hf" / "parquet" / key / cfg
            stage.mkdir(parents=True, exist_ok=True)
            total_gb = sum((e.size or 0) for e in entries) / 1e9
            print(
                f"  -- config {cfg}: {len(entries)} parquet files, "
                f"{total_gb:.2f} GB in the repo for this config",
                flush=True,
            )

            # Download and convert in windows, adaptively, until the cap is
            # actually met. An earlier version estimated each file's text size
            # up front and downloaded a fixed set: the estimate was 2.4x high
            # on Wikipedia (article lengths vary), so the source stopped at
            # 1.29 GB of a 3.00 GB budget. It also downloaded a whole 2 GB
            # parquet file just to sample its first rows. Windows measured
            # from what the first file actually yields make the cap real and
            # cost no wasted bandwidth.
            cfg_written = 0
            i = 0
            ratio: float | None = None  # text bytes per parquet byte
            parquet_bytes = 0
            while cfg_written < per_config and i < len(entries):
                if ratio is None:
                    width = WINDOW_FILES
                else:
                    est_next = ratio * (entries[i].size or 1)
                    width = max(
                        1, min(WINDOW_FILES,
                               int((per_config - cfg_written) / max(est_next, 1)) + 1)
                    )
                batch = entries[i : i + width]
                i += width
                t_dl = time.time()
                snapshot_download(
                    src["repo"], repo_type="dataset", revision=revision,
                    allow_patterns=[e.rfilename for e in batch],
                    local_dir=str(stage), max_workers=8,
                )
                print(
                    f"  -- window {len(batch)} files downloaded in "
                    f"{time.time() - t_dl:.0f}s ({i}/{len(entries)} files, "
                    f"{cfg_written / 1e9:.2f}/{per_config / 1e9:.2f} GB)",
                    flush=True,
                )
                for e in batch:
                    parquet_bytes += e.size or (stage / e.rfilename).stat().st_size
                    pf = pq.ParquetFile(stage / e.rfilename)
                    for b in pf.iter_batches(
                        batch_size=1000, columns=[src["text_field"]]
                    ):
                        col = b.column(0).to_pylist()
                        for text in col:
                            if not text:
                                continue
                            n = writer.write(text)
                            written += n
                            cfg_written += n
                            docs += 1
                            if docs % 50000 == 0:
                                print(
                                    f"    {docs:,} docs, {written / 1e9:.2f} GB,"
                                    f" {time.time() - t0:.0f}s",
                                    flush=True,
                                )
                            if cfg_written >= per_config:
                                break
                        if cfg_written >= per_config:
                            break
                    if parquet_bytes:
                        ratio = cfg_written / parquet_bytes
                    if cfg_written >= per_config:
                        break
                for e in batch:  # staging is scratch; the shards are the record
                    p = stage / e.rfilename
                    if p.exists():
                        p.unlink()
        writer.close()
        record.update(
            docs=docs,
            bytes_text=written,
            bytes_uncompressed=sum(
                s["bytes_uncompressed"] for s in record["shards"]
            ),
            bytes_compressed=sum(s["bytes_compressed"] for s in record["shards"]),
            cached=False,
            exhausted=written < cap_bytes,  # ran out of files before the cap
            seconds=round(time.time() - t0, 1),
        )
        print(
            f"  wrote {docs:,} docs, {written / 1e9:.2f} GB in "
            f"{len(record['shards'])} shards ({time.time() - t0:.0f}s)"
        )
        out[key] = record


def _shard_record(
    path: Path, docs: int, seconds: float, payload_file: int, payload_text: int
) -> dict:
    """Record a closed shard.

    Two byte counts are kept, and they are not interchangeable. `bytes_text` is
    the concatenated text payload and is what the byte cap is written in.
    `bytes_uncompressed` is the whole decompressed file -- JSON braces, keys,
    escaping -- and is what an independent verifier can re-derive from the file
    alone. Recording one and checking the other is how the first version
    produced a 0.76% systematic mismatch on every shard.
    """
    return {
        "path": str(path.relative_to(PROJECT)),
        "docs": docs,
        "bytes_uncompressed": payload_file,
        "bytes_text": payload_text,
        "bytes_compressed": path.stat().st_size,
        "sha256": sha256_file(path),
        "seconds": round(seconds, 1),
    }


def environment() -> dict:
    import torch

    return {
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "datasets": hf_datasets.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "hf_home": os.environ["HF_HOME"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of keys, or a group name")
    ap.add_argument("--skip-tasks", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    only = {k.strip() for k in args.only.split(",") if k.strip()}
    api = HfApi()

    previous = {}
    if MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text())
    out = {} if args.force else previous.get("datasets", {})

    if not args.skip_tasks:
        for entry in TASKS:
            if only and entry["key"] not in only and entry["family"] not in only:
                continue
            pull_task(api, entry, out, force=args.force)

    for src in CORPUS_SOURCES:
        if only and src["key"] not in only and src["group"] not in only:
            continue
        pull_corpus(api, [src], out)

    # Start from what is already recorded and update only this script's keys.
    # Writing a fresh dict dropped the `models` section every time a pull ran,
    # which meant the Laya checkpoints vanished from the record the next time
    # any dataset was added.
    manifest = dict(previous) if previous else {}
    manifest.update(
        {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "environment": environment(),
            "datasets": out,
            "policy": {
                "cache": "data/.hf (project-local)",
                "shard_target_bytes": SHARD_TARGET_BYTES,
                "note": "no artefact outside ./data; hashes are of the files written",
            },
        }
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nmanifest -> {MANIFEST.relative_to(PROJECT)}")

    rows_total = 0
    for key, rec in sorted(out.items()):
        if rec.get("kind") == "task":
            rows = sum(s["rows"] for s in rec["splits"].values())
            rows_total += rows
            mismatch = [
                f"{sp}({s['rows']}!={s['expected_rows']})"
                for sp, s in rec["splits"].items()
                if s.get("matches_expected") is False
            ]
            flag = ("  MISMATCH " + ",".join(mismatch)) if mismatch else ""
            used = rec.get("used", [])
            if isinstance(used, dict):  # records written by the first version
                used = [used]
            repos = ",".join(sorted({str(u.get("repo", "?")) for u in used}))
            print(f"  {key:24s} {rows:>9,d} rows  {rec.get('family', '?'):12s} {repos}{flag}")
        else:
            print(f"  {key:24s} {rec.get('bytes_uncompressed', 0) / 1e9:>9.2f} GB "
                  f"{rec.get('docs', 0):>10,d} docs  {len(rec.get('shards', []))} shards")
    print(f"  {'TOTAL task rows':24s} {rows_total:>9,d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
