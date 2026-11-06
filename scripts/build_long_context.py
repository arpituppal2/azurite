#!/usr/bin/env python3
"""Build long-context selector packs. Data preparation only; no model step.

A pack is one 32k-token document as the selector sees it: 64 chunks of up to 512
tokens. A target passage sits among 0/2/8 same-task distractors at an
early/middle/late position, and the rest of the 64 chunks is background filler.
Without the filler a pack holds nine chunks, and selecting eight of nine scores
~89% recall by construction.

Two batteries exist:

* ``embedded`` — the earlier build, where the task context was passage+question,
  so the question string also occurred inside the target passage. BM25 scored
  1.000 recall@8 on it and the "+3 pp" selector gate had no headroom.
* ``noquestion`` — this file's default. Only the passage is used, filler that
  contains the question is rejected, and every pack is checked for the property
  before it is written.

Gold is tracked by construction: each target's byte range is recorded as it is
placed, and chunks are mapped from that range. It is never recovered by
searching for anchor text, which matched duplicate paragraphs in the filler.

Output is always sharded JSON lines plus a manifest, written incrementally, so a
100k-pack build needs memory for one shard rather than for the whole corpus.

Usage:
    .venv/bin/python3 scripts/build_long_context.py --packs 2000 \
        --out-dir results/parity/long_context_dev_noquestion \
        --metrics results/parity/context_metrics_dev_noquestion.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.decisions import TASKS, candidate_bank, load_examples  # noqa: E402
from azurite.selector import BM25ChunkSelector, chunk_text  # noqa: E402

SEP = "\n\n---\n\n"
CHUNK_BYTES = 2_048
STRIDE = 1_024
TOP_K = 8
FILLER_DOCS = 160
POSITION_FRACTION = {"early": 0.0, "middle": 0.45, "late": 0.90}
BUILD_REASONS: list[str] = []


def window_bytes(chunks_per_pack: int) -> int:
    return STRIDE * (chunks_per_pack - 1) + CHUNK_BYTES


def length_bounds(chunks_per_pack: int) -> tuple[int, int]:
    """Text lengths that yield exactly `chunks_per_pack` chunks."""
    low = STRIDE * (chunks_per_pack - 2) + CHUNK_BYTES
    high = STRIDE * (chunks_per_pack - 1) + CHUNK_BYTES
    return low + 1, high


def _find_gap(occupied: list[tuple[int, int]], size: int, limit: int) -> int | None:
    """First offset with `size` free bytes, given sorted reserved intervals."""
    cursor = 0
    for start, end in sorted(occupied):
        if start - cursor >= size:
            return cursor
        cursor = max(cursor, end)
    return cursor if limit - cursor >= size else None


def _pad_to(out: bytearray, target: int, fillers: list[str], cursor: int) -> int:
    """Append filler until `out` reaches `target` bytes, never overshooting."""
    sep = SEP.encode("utf-8")
    while len(out) < target and cursor < len(fillers):
        piece = fillers[cursor].encode("utf-8")
        cursor += 1
        if out:
            piece = sep + piece
        room = target - len(out)
        if len(piece) > room:
            piece = piece[: _safe_cut(piece, room)]
        if not piece:
            break
        out += piece
    return cursor


def _safe_cut(raw: bytes, at: int) -> int:
    """Largest k <= at that does not split a UTF-8 sequence."""
    at = max(0, min(at, len(raw)))
    while 0 < at < len(raw) and (raw[at] & 0xC0) == 0x80:
        at -= 1
    return at


def compose_pack(
    content: list[str],
    fillers: list[str],
    *,
    positions: list[str],
    gold: tuple[int, ...] = (0,),
    chunks_per_pack: int = 64,
) -> tuple[str, list[dict]]:
    """Place content documents at their positions; return gold spans only.

    Documents are appended whole, so every byte range is valid UTF-8 by
    construction. Placement order is sorted by target offset, and the gold
    documents get first claim on their region: an earlier version sorted all
    distractors ahead of the target, which labelled a distractor as gold and let
    long distractors push the real target out of the window.
    """
    if len(content) != len(positions):
        raise ValueError("every content document needs a position")
    low, high = length_bounds(chunks_per_pack)

    # Content shares the window: nine long passages would otherwise overflow it
    # together and push the gold document out of the pack.
    room = max(CHUNK_BYTES, min(high // 3, (high // 2) // len(content)))
    raws: dict[int, bytes] = {}
    for index, doc in enumerate(content):
        raw = doc.encode("utf-8")
        raws[index] = raw[: _safe_cut(raw, room)] if len(raw) > room else raw

    # Gold claims its region first, then distractors fill the gaps. A distractor
    # must never be able to displace the target or be labelled as it.
    plan: list[tuple[int, int, bytes]] = []
    occupied: list[tuple[int, int]] = []
    for index in sorted(gold):
        raw = raws[index]
        position = positions[index]
        if position == "early":
            start = 0
        elif position == "middle":
            start = max(0, (high - len(raw)) // 2)
        else:
            start = max(0, high - len(raw) - CHUNK_BYTES)
        plan.append((start, index, raw))
        occupied.append((start, start + len(raw)))
    for index in range(len(content)):
        if index in gold:
            continue
        raw = raws[index]
        # The separator that will precede this document is part of its footprint.
        start = _find_gap(occupied, len(raw) + len(SEP), high)
        if start is None:
            continue
        plan.append((start, index, raw))
        occupied.append((start, start + len(raw)))
    plan.sort(key=lambda item: (item[0], item[1]))

    out = bytearray()
    spans: dict[int, tuple[int, int]] = {}
    cursor = 0
    for start, index, raw in plan:
        # Land the document exactly on its planned offset; the separator counts.
        target = start - (len(SEP) if out else 0)
        if target > len(out):
            cursor = _pad_to(out, target, fillers, cursor)
        if out:
            out += SEP.encode("utf-8")
        spans[index] = (len(out), len(out) + len(raw))
        out += raw

    if len(out) < low + 1:
        cursor = _pad_to(out, low + 1, fillers, cursor)
    text = bytes(out).decode("utf-8", errors="ignore")
    chunks = chunk_text(text)
    # `_pad_to` counts UTF-8 bytes while text can contain multibyte characters:
    # a passage with curly quotes can sit above the byte floor and still be one
    # chunk short. Pad until the chunker agrees, never past the byte ceiling.
    while len(chunks) != chunks_per_pack and len(out) < high and cursor < len(fillers):
        cursor = _pad_to(out, min(high, len(out) + 8_192), fillers, cursor)
        text = bytes(out).decode("utf-8", errors="ignore")
        chunks = chunk_text(text)
    tail_floor = max([end for _, end in spans.values()] + [0])
    if len(out) > high:
        cut = max(_safe_cut(bytes(out), high), tail_floor)
        out = out[:cut]
        text = bytes(out).decode("utf-8", errors="ignore")

    if not (low <= len(text.encode("utf-8")) <= high):
        raise ValueError(f"composed {len(text.encode('utf-8'))} bytes, outside {low}..{high}")
    chunks = chunk_text(text)
    if len(chunks) != chunks_per_pack:
        raise ValueError(f"expected {chunks_per_pack} chunks, got {len(chunks)}")

    spans_out = []
    for index in gold:
        if index not in spans:
            raise ValueError(f"gold index {index} was never placed")
        start, end = spans[index]
        indices = [c.index for c in chunks if c.byte_start < end and c.byte_end > start]
        if not indices:
            raise ValueError("gold span covers no chunk")
        spans_out.append({"start": int(start), "end": int(end), "chunks": indices})
    return text, spans_out


def sample_fillers(pool: list[str], question: str, rng: random.Random, need: int) -> list[str]:
    """Draw filler documents that do not contain the question string."""
    picked: list[str] = []
    tries = 0
    while len(picked) < FILLER_DOCS and tries < FILLER_DOCS * 4:
        tries += 1
        doc = pool[rng.randrange(len(pool))]
        if question and question in doc:
            continue
        picked.append(doc)
    if not picked:
        raise ValueError("no filler documents free of the question string")
    size = 0
    while size < need:
        chunk = picked[size % len(picked)]
        picked.append(chunk)
        size += len(chunk.encode("utf-8")) + len(SEP.encode("utf-8"))
    return picked


def document_of(example) -> str:
    return example.parts[0] if example.parts else example.context


def question_of(name: str, example) -> str:
    if len(example.parts) > 1:
        return example.parts[1]
    return TASKS[name].instruction


def build_pack(index: int, data, pool: list[str], args, rng: random.Random) -> dict | None:
    """Build one pack, or None if the question could not be kept out of it."""
    for _ in range(8):
        name, target, bank = data[rng.randrange(len(data))]
        question = question_of(name, target)
        doc = document_of(target)
        if question and question in doc:
            BUILD_REASONS.append(f"pack {index}: question appears in the target passage")
            continue

        n_distractors = [0, 2, 8][index % 3]
        candidates = [d for d in data if d[2].task == bank.task and d[1].index != target.index]
        rng.shuffle(candidates)
        distractors = []
        for _, example, _ in candidates:
            if len(distractors) >= n_distractors:
                break
            text = document_of(example)
            if question and question in text:
                continue
            distractors.append(text)

        distributed = index % 10 == 0
        if distributed:
            middle = len(doc) // 2
            content = [doc[:middle], doc[middle:]]
            positions = ["early", "late"]
            gold = (0, 1)
            position = "distributed"
        else:
            content = [doc]
            positions = [["early", "middle", "late"][index % 3]]
            gold = (0,)
            position = positions[0]
        content += distractors
        positions += ["middle"] * len(distractors)

        fillers = sample_fillers(pool, question, rng, window_bytes(args.chunks_per_pack))
        try:
            text, spans = compose_pack(
                content, fillers, positions=positions, gold=gold,
                chunks_per_pack=args.chunks_per_pack,
            )
        except ValueError as exc:
            BUILD_REASONS.append(f"pack {index}: compose failed: {exc}")
            continue
        if question and question in text:
            BUILD_REASONS.append(f"pack {index}: question survived into the pack")
            continue
        return {
            "id": f"pack.{index:06d}",
            "battery": args.battery,
            "source": name,
            "context": text,
            "question": TASKS[name].instruction,
            "target_question": question,
            "alternatives": list(bank.labels),
            "gold": int(target.gold),
            "gold_chunk": spans[0]["chunks"][0],
            "gold_chunk_end": spans[0]["chunks"][-1],
            "gold_spans": spans,
            "distributed": distributed,
            "distractors": n_distractors,
            "position": position,
            "chunks": len(chunk_text(text)),
            "label_kind": "document_span",
        }
    return None


def score_pack(pack: dict) -> tuple[bool, bool]:
    """(any-overlap recall, strict first-chunk recall) for one pack."""
    chunks = chunk_text(pack["context"])
    picked = {c.index for c in BM25ChunkSelector(chunks).select(pack["target_question"], top_k=TOP_K)}
    spans = pack["gold_spans"]
    any_hit = all(any(i in picked for i in span["chunks"]) for span in spans)
    strict_hit = all(span["chunks"][0] in picked for span in spans)
    return any_hit, strict_hit


def shard_file(index: int) -> str:
    return f"shard-{index:05d}.jsonl.gz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_shard(path: Path, rows: list[dict]) -> int:
    tmp = path.with_name(path.name + ".tmp")
    # mtime=0 keeps the gzip header constant, so rebuilding a shard reproduces
    # the same bytes and its recorded sha256 stays meaningful across runs.
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(tmp, path)
    return path.stat().st_size


def merge_shards(out_dir: Path, config: dict, expected: int | None = None) -> dict | None:
    """Assemble the manifest from per-shard metadata files.

    Workers write only their own shard and its metadata, so parallel builds
    never race on a shared manifest; this merge runs once at the end.
    """
    entries = [json.loads(p.read_text()) for p in sorted(out_dir.glob("shard-*.meta.json"))]
    if not entries:
        return None
    entries.sort(key=lambda e: e["index"])
    if expected is not None and len(entries) < expected:
        return None
    return {
        "schema": 3, "format": "jsonl.gz", "config": config,
        "shards": entries, "packs": sum(e["packs"] for e in entries),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_source(name: str, split: str, limit: int):
    spec = TASKS[name]
    bank = candidate_bank(spec)
    return load_examples(spec, bank, split=split, limit=limit), bank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", type=int, default=100_000)
    ap.add_argument("--limit-per-source", type=int, default=2_000)
    ap.add_argument("--chunks-per-pack", type=int, default=64)
    ap.add_argument("--packs-per-shard", type=int, default=2_000)
    ap.add_argument("--shard-stride", type=int, default=1, help="workers: build every Nth shard")
    ap.add_argument("--shard-offset", type=int, default=0, help="worker index within the stride")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metrics", default="results/parity/context_metrics.json")
    ap.add_argument("--battery", default="noquestion")
    ap.add_argument("--probe", type=int, default=2_000)
    ap.add_argument("--sources", default="squad2_answerability_ctx,contract_nli_ctx,boolq_ctx")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    config = {
        "battery": args.battery,
        "packs": args.packs,
        "chunks_per_pack": args.chunks_per_pack,
        "packs_per_shard": args.packs_per_shard,
        "seed": args.seed,
        "sources": args.sources,
        "split": args.split,
        "chunk_bytes": CHUNK_BYTES,
        "stride": STRIDE,
        "position_fraction": POSITION_FRACTION,
    }

    data = []
    for name in [s for s in args.sources.split(",") if s]:
        examples, bank = load_source(name, args.split, args.limit_per_source)
        for ex in examples:
            data.append((name, ex, bank))
    if not data:
        raise SystemExit("no source rows")
    pool = [document_of(ex) for _, ex, _ in data if document_of(ex).strip()]
    if len(pool) < 20:
        raise SystemExit(f"not enough filler documents ({len(pool)})")

    known: dict[int, dict] = {}
    for meta_path in sorted(out_dir.glob("shard-*.meta.json")):
        meta = json.loads(meta_path.read_text())
        path = out_dir / meta["file"]
        if path.exists() and sha256_file(path) == meta.get("sha256"):
            known[meta["index"]] = meta
    resumed = 0

    per = args.packs_per_shard
    n_shards = -(-args.packs // per)
    hits = strict = measured = 0
    by_position: dict[str, list[int]] = {}
    question_in_pack = 0
    shards: list[dict] = []
    started = time.time()

    for shard_index in range(args.shard_offset, n_shards, args.shard_stride):
        start = shard_index * per
        count = min(per, args.packs - start)
        path = out_dir / shard_file(shard_index)
        entry = known.get(shard_index)
        if entry and path.exists() and entry.get("packs") == count and sha256_file(path) == entry.get("sha256"):
            shards.append(entry)
            resumed += 1
            continue

        rng = random.Random(args.seed * 1_000_003 + shard_index)
        rows: list[dict] = []
        for j in range(count):
            index = start + j
            row = build_pack(index, data, pool, args, rng)
            if row is None:
                raise SystemExit(
                    "could not build pack " + str(index) + ": " + "; ".join(BUILD_REASONS[-8:])
                )
            if row["target_question"] and row["target_question"] in row["context"]:
                question_in_pack += 1
            if len(rows) < args.probe:
                any_hit, strict_hit = score_pack(row)
                measured += 1
                hits += int(any_hit)
                strict += int(strict_hit)
                bucket = by_position.setdefault(row["position"], [0, 0, 0])
                bucket[0] += int(any_hit)
                bucket[1] += int(strict_hit)
                bucket[2] += 1
            rows.append(row)
        size = write_shard(path, rows)
        meta = {
            "index": shard_index,
            "file": path.name,
            "packs": len(rows),
            "gzip_bytes": size,
            "sha256": sha256_file(path),
        }
        write_manifest(out_dir / f"shard-{shard_index:05d}.meta.json", meta)
        print(f"shard {shard_index} {len(rows)} packs, {size/1e6:.1f} MB gz", flush=True)

    merged = merge_shards(out_dir, config, expected=n_shards)
    if merged is None:
        print(f"battery incomplete ({len(known)} of {n_shards} shards); rerun to resume", flush=True)
        return 0
    write_manifest(manifest_path, merged)
    if not measured:
        print("this worker built no probe packs; metrics belong to the probe shard", flush=True)
        return 0

    position_recall = {
        pos: {"any": v[0] / v[2], "strict": v[1] / v[2], "n": v[2]}
        for pos, v in sorted(by_position.items())
    }
    chance_any = TOP_K / args.chunks_per_pack
    mean_chance = sum(
        (chance_any ** 2 if pos == "distributed" else chance_any) * v["n"]
        for pos, v in position_recall.items()
    ) / measured
    worst_gap = None
    singles = [v["any"] for pos, v in position_recall.items() if pos != "distributed"]
    if len(singles) > 1:
        worst_gap = round((max(singles) - min(singles)) * 100, 2)
    metrics = {
        "battery": args.battery,
        "n": measured,
        "packs": merged["packs"],
        "chunks_per_pack": args.chunks_per_pack,
        "mean_pack_bytes": window_bytes(args.chunks_per_pack),
        "chance_recall_at_8": round(chance_any, 4),
        "mean_chance_recall": round(mean_chance, 4),
        "selector_recall": hits / measured,
        "selector_recall_strict": strict / measured,
        "selector_recall_start_chunk": strict / measured,
        "by_position": position_recall,
        "worst_position_gap_pp": worst_gap,
        "question_in_pack": question_in_pack,
        "selector": "BM25",
        "label_kind": "document_span",
        "manifest": str(manifest_path),
        "shards": len(merged["shards"]),
        "resumed_shards": resumed,
        "elapsed_s": round(time.time() - started, 1),
        "niah_32k": None,
        "multi_hop": None,
        "gate": {
            "criterion": "BM25 recall@8 >= 0.85 on dev packs (any-overlap)",
            "state": "pass" if hits / measured >= 0.85 else "fail",
        },
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    print(f"manifest {manifest_path} ({len(merged['shards'])} shards, {metrics['packs']} packs)")
    print(f"BM25 recall@8 {metrics['selector_recall']:.3f} strict {metrics['selector_recall_strict']:.3f} "
          f"(chance {chance_any:.3f}) gate {metrics['gate']['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
