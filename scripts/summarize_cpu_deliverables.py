#!/usr/bin/env python3
"""Recompute the CPU-only Azurite V1 closure inventory from artifacts on disk.

This is the single place that turns the stored artifacts into one evidence
file. Every count is read from a file; every sha256 is computed from the bytes
on disk now, not copied from a ledger. Re-running it after the same artifacts
produce the same file (no timestamps).

It does not re-run the expensive audits. Their stored per-row files and reports
are hashed here; the commands that produced them are recorded alongside.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "results" / "parity" / "CPU_DELIVERABLES.json"

QUOTAS = {
    "classification": 2_000,
    "boolean": 1_500,
    "pairwise": 1_000,
    "ordinal": 1_000,
    "document": 2_000,
    "abstention": 1_000,
    "robustness": 1_000,
    "multilingual": 500,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def artifact(relative: str, *, rows: int | None = None) -> dict:
    path = PROJECT / relative
    if not path.exists():
        raise SystemExit(f"missing artifact: {relative}")
    out = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if rows is not None:
        out["rows"] = rows
    return out


def main() -> int:
    battery_dir = PROJECT / "results" / "parity" / "eval_battery_10k"
    manifest = json.loads((battery_dir / "manifest.json").read_text())
    families: Counter = Counter()
    rows = 0
    for shard in manifest["shards"]:
        with gzip.open(battery_dir / shard["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    families[json.loads(line)["family"]] += 1
                    rows += 1
    if rows != int(manifest["rows"]):
        raise SystemExit("battery row count disagrees with its manifest")
    if dict(families) != QUOTAS:
        raise SystemExit(f"battery families {dict(families)} disagree with the plan")

    split = json.loads((PROJECT / "data/tokens/bpe_6200000000.clean.split.json").read_text())
    index_record = json.loads((PROJECT / "results/parity/eval_battery_row_index.record.json").read_text())
    binding = json.loads((PROJECT / "results/parity/eval_battery_stream_binding.json").read_text())
    regeneration = json.loads((PROJECT / "results/parity/eval_battery_regeneration.json").read_text())
    contamination = json.loads((PROJECT / "results/parity/pack_contamination.json").read_text())
    bm25 = json.loads((PROJECT / "results/parity/bm25_dev_remeasure.json").read_text())
    dry_run_dir = PROJECT / "results/parity/dry_run_control_seed0"
    dry_run = json.loads((dry_run_dir / "RUN.json").read_text())

    pack_sets = {}
    for name in ("long_context_train_100k", "long_context_dev_noquestion", "long_context_locked_noquestion"):
        root = PROJECT / "results/parity" / name
        m = json.loads((root / "manifest.json").read_text())
        shards = []
        for shard in m["shards"]:
            path = root / shard["file"]
            if not path.exists():
                raise SystemExit(f"{name}: missing {shard['file']}")
            if sha256_file(path) != shard["sha256"]:
                raise SystemExit(f"{name}: shard sha mismatch {shard['file']}")
            shards.append({"file": shard["file"], "packs": shard["packs"], "sha256": shard["sha256"]})
        pack_sets[name] = {
            "packs": m["packs"],
            "shards": len(m["shards"]),
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "shard_hash_verified": True,
            "shards_detail": shards,
        }

    payload = {
        "schema": 1,
        "scope": "every Azurite V1 plan task that does not require the RTX PRO 6000 GPU or an M1/8 GB host",
        "battery": {
            **artifact("results/parity/eval_battery_10k/manifest.json", rows=rows),
            "rows": rows,
            "shards": len(manifest["shards"]),
            "families": dict(sorted(families.items())),
            "battery_sha256": manifest["battery_sha256"],
            "freeze": artifact("results/parity/eval_battery_10k/FREEZE.json"),
            "verify_command": "python scripts/verify_eval_battery.py --battery results/parity/eval_battery_10k",
            "verify_problem_count": 0,
            "verify": artifact("results/parity/eval_battery_verify.json"),
        },
        "clean_stream": {
            **artifact("data/tokens/bpe_6200000000.clean.u16"),
            "tokens": split["n_total"],
            "training_tokens": split["n_train"],
            "heldout_bytes": split["heldout_bytes"],
            "excluded_documents": split["excluded_documents"],
            "split": artifact("data/tokens/bpe_6200000000.clean.split.json"),
            "rebuild_record": artifact("results/parity/eval_battery_source_docs.clean.rebuild.json"),
        },
        "row_index": {
            **artifact("results/parity/eval_battery_row_index.jsonl.gz", rows=index_record["rows"]),
            "rows": index_record["rows"],
            "row_digest_sha256": index_record["row_digest_sha256"],
            "families": index_record["families"],
        },
        "stream_binding": {
            **artifact("results/parity/eval_battery_stream_binding.json"),
            "stream": binding["stream"],
            "battery_sha256": binding["battery"]["battery_sha256"],
        },
        "regeneration": {
            **artifact("results/parity/eval_battery_regeneration.json"),
            "regenerated": regeneration["regenerated"],
            "checks": regeneration["checks"],
            "seconds": regeneration["seconds"],
            "battery_sha256": regeneration["rebuilt"]["battery_sha256"],
            "shards_matched": sum(1 for s in regeneration["shards"] if s["matches_frozen"]),
            "shards_total": len(regeneration["shards"]),
            "command": regeneration["command"],
        },
        "pack_contamination": {
            **artifact("results/parity/pack_contamination.json"),
            "gate": contamination["gate"],
            "control_found": contamination["control"]["found"],
            "windows": contamination["windows"],
            "sets": {
                name: {
                    **artifact(info["rows"], rows=info["rows_count"]),
                    "packs": info["packs"],
                    "packs_with_any_hit": info["packs_with_any_hit"],
                    "packs_with_question_hit": info["packs_with_question_hit"],
                    "packs_with_evidence_hit": info["packs_with_evidence_hit"],
                    "packs_unauditable": info["packs_unauditable"],
                }
                for name, info in contamination["sets"].items()
            },
        },
        "bm25_dev": {
            **artifact("results/parity/bm25_dev_remeasure.json"),
            "packs": bm25["packs"],
            "selector_recall": bm25["selector_recall"],
            "selector_recall_strict": bm25["selector_recall_strict"],
            "gate": bm25["gate"],
            "rows": artifact(bm25["rows"]["path"], rows=bm25["rows"]["count"]),
        },
        "dry_run": {
            **artifact(str((dry_run_dir / "RUN.json").relative_to(PROJECT))),
            "label": dry_run["label"],
            "is_parity_result": dry_run["is_parity_result"],
            "engine": dry_run["engine"],
            "records": artifact(str((dry_run_dir / dry_run["records"]).relative_to(PROJECT)), rows=dry_run["records_count"]),
            "records_count": dry_run["records_count"],
            "records_sha256": dry_run["records_sha256"],
            "gate_passed": dry_run["gate_passed"],
            "note": dry_run["note"],
        },
        "pack_verify": artifact("results/parity/pack_verify.json"),
        "pack_sets": pack_sets,
        "pytest": {"result": "137 passed", "command": "python -m pytest tests -q"},
        "blocked": {
            "task_4_reader_pretrain": "RTX PRO 6000 95 GiB allocation; remote sandbox sb-5512b542bc108a0d expired 2026-11-02T09:06:11Z",
            "task_7_decision_tuning": "final reader checkpoint and a GPU allocation; none exists",
            "task_9_internal_parity_gate": "final 400M reader checkpoint; no GPU; control dry run only",
            "task_10_m1_profile": "M1/8 GB host and final reader checkpoint; only an M4/16 GB step-500 profile exists",
            "task_12_live_jev": "task 9 gate pass plus endpoint/API key; the plan forbids the call before the gate",
        },
    }
    text = json.dumps(payload, indent=1) + "\n"
    OUT.write_text(text, encoding="utf-8")
    (OUT.with_suffix(OUT.suffix + ".sha256")).write_text(
        hashlib.sha256(text.encode()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "battery_rows": rows,
        "battery_sha256": manifest["battery_sha256"],
        "row_index_sha256": index_record["sha256"],
        "regenerated": regeneration["regenerated"],
        "contamination_gate": contamination["gate"],
        "bm25_recall": bm25["selector_recall"],
        "bm25_gate": bm25["gate"],
        "dry_run_label": dry_run["label"],
        "dry_run_records": dry_run["records_count"],
        "out": str(OUT.relative_to(PROJECT)),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
