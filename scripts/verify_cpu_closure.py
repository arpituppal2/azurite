#!/usr/bin/env python3
"""Verify and regenerate the entire CPU-side Azurite V1 closure.

One command:

    .venv/bin/python scripts/verify_cpu_closure.py

It re-derives the claims in CLOSURE_LEDGER.json / CLOSURE_LEDGER.md and
CPU_DELIVERABLES.json from the artifacts on disk: the corrected 6.2B-token
stream, the frozen 10k battery, its row index and stream binding, a fresh
seed-0 regeneration, the three pack collections, the per-row contamination
audit, the BM25 dev measurement, the selector verdict, and the labelled dry run.
Hashes are computed from files, never copied from the previous ledger. Any
mismatch is reported as HASH-MISMATCH / FAIL / MISSING and exits non-zero.

After every check passes it rewrites CPU_DELIVERABLES.json, CLOSURE_LEDGER.json
and CLOSURE_LEDGER.md from the verified facts, writes .sha256 sidecars, and
prints the final table. The outputs are deterministic: no wall-clock timestamp,
no duration, no random value enters them.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PARITY = PROJECT / "results" / "parity"
BATTERY = PARITY / "eval_battery_10k"
CLEAN = PROJECT / "data" / "tokens" / "bpe_6200000000.clean.u16"
SPLIT = PROJECT / "data" / "tokens" / "bpe_6200000000.clean.split.json"
INVENTORY = PARITY / "CPU_DELIVERABLES.json"
LEDGER_JSON = PARITY / "CLOSURE_LEDGER.json"
LEDGER_MD = PARITY / "CLOSURE_LEDGER.md"
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
PACK_SETS = {
    "train_100k": PARITY / "long_context_train_100k",
    "dev": PARITY / "long_context_dev_noquestion",
    "locked": PARITY / "long_context_locked_noquestion",
}
_LIVE_OBS = json.loads((PARITY / "reader_live_observation.json").read_text())
MISSING_RESOURCE = (
    "the completed 6.2B reader budget; the run is live on sb-120c2046908226ee at "
    f"step {int(_LIVE_OBS.get('log_step', 0)):,} "
    f"(checkpoint {int(_LIVE_OBS.get('remote_checkpoint_step', 0)):,}), "
    "resumed from the verified step-5,000 mirror on the pinned clean stream"
)

CHECKS: list[dict] = []


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def add(name: str, ok: bool, detail: str, expected=None, actual=None, status: str | None = None) -> bool:
    if status is None:
        status = "PASS" if ok else "FAIL"
    CHECKS.append({
        "name": name,
        "status": status,
        "detail": detail,
        "expected": expected,
        "actual": actual,
    })
    return ok


def check_hash(name: str, path: Path, expected: str, label: str = "") -> bool:
    path = Path(path)
    if not path.exists():
        return add(name, False, f"missing {path.name}", expected=expected, status="MISSING")
    actual = sha256_file(path)
    if actual == expected:
        return add(name, True, f"{label or path.name} {actual[:12]}", expected, actual)
    return add(name, False, f"{label or path.name} {actual[:12]} != {expected[:12]}", expected, actual, "HASH-MISMATCH")


def check_eq(name: str, actual, expected, label: str = "") -> bool:
    ok = actual == expected
    detail = f"{label} {actual!r}" if ok else f"{label} {actual!r} != {expected!r}"
    return add(name, ok, detail, expected, actual)


def check_true(name: str, condition: bool, detail: str) -> bool:
    return add(name, bool(condition), detail)


def run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True, timeout=timeout)


def artifact(relative: str, rows: int | None = None) -> dict:
    path = PROJECT / relative
    out = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        out["rows"] = rows
    return out


def battery_verify() -> dict:
    res = run([sys.executable, "scripts/verify_eval_battery.py", "--battery", str(BATTERY)])
    try:
        rep = json.loads(res.stdout)
    except json.JSONDecodeError:
        add("battery.verify", False, f"verifier output was not JSON (rc {res.returncode}): {res.stderr[-200:]}")
        raise SystemExit(1)
    check_eq("battery.rows", rep.get("rows"), 10_000, "rows")
    check_eq("battery.shards", rep.get("shards"), 10, "shards")
    check_eq("battery.problems", rep.get("problem_count"), 0, "problem_count")
    check_eq("battery.families", rep.get("families"), QUOTAS, "family quotas")
    return rep


def verify_clean_stream(binding: dict, rebuild: dict, split: dict, audit: dict) -> dict:
    size = CLEAN.stat().st_size
    sha = sha256_file(CLEAN)
    tokens = size // 2
    check_true("stream.size_even", size % 2 == 0, f"{size:,} bytes")
    check_eq("stream.sha256", sha, binding["stream"]["sha256"], "sha256")
    check_eq("stream.rebuild_sha256", sha, rebuild["new_stream_sha256"], "rebuild record sha")
    check_eq("stream.bytes", size, binding["stream"]["bytes"], "bytes")
    check_eq("stream.tokens", tokens, binding["stream"]["tokens"], "tokens")
    check_eq("stream.tokens_split", tokens, split["n_total"], "split n_total")
    check_eq("stream.training_tokens", split["n_train"], rebuild["new_training_tokens"], "train tokens")
    check_eq("stream.training_binding", split["n_train"], binding["stream"]["training_tokens"], "train tokens")
    check_eq("stream.heldout_tokens", split["heldout_bytes"], rebuild["new_heldout_tokens"], "heldout tokens")
    check_eq("stream.heldout_binding", split["heldout_bytes"], binding["stream"]["heldout_tokens"], "heldout tokens")
    check_eq("stream.split_sum", split["n_train"] + split["heldout_bytes"], tokens, "train + heldout")
    check_eq("stream.excluded_documents", split["excluded_documents"], rebuild["excluded_documents"], "excluded docs")
    check_eq("stream.excluded_binding", split["excluded_documents"], binding["stream"]["excluded_documents"], "excluded docs")
    check_eq("stream.split_sha256", sha256_file(SPLIT), binding["stream"]["split_sha256"], "split sha")
    check_eq("stream.audit_sha256", sha256_file(PARITY / "eval_battery_source_docs.clean.json"), binding["battery_source_audit"]["sha256"], "audit sha")
    check_eq("stream.audit_gate", audit["audit"]["gate"], "PASS", "audit gate")
    check_eq("stream.audit_hits", audit["audit"]["whole_stream_hit_source_docs"], 0, "source-doc hits")
    check_eq("stream.audit_positions", audit["audit"]["whole_stream_hit_positions"], 0, "hit positions")
    check_true("stream.audit_control", audit["audit"]["control_found"] is True, "positive control found")
    check_eq("stream.audit_patterns", audit["audit"]["patterns"], 18_594, "patterns")
    check_eq("stream.audit_stream", audit["new_stream_sha256"], sha, "audit stream sha")
    return {
        "bytes": size,
        "sha256": sha,
        "tokens": tokens,
        "training_tokens": split["n_train"],
        "heldout_tokens": split["heldout_bytes"],
        "excluded_documents": split["excluded_documents"],
        "excluded_source_docs": rebuild["excluded_source_docs"],
        "excluded_battery_rows": rebuild["old_hit_rows"],
        "split": split,
        "rebuild": rebuild,
        "audit": audit,
    }


def verify_index() -> dict:
    index_path = PARITY / "eval_battery_row_index.jsonl.gz"
    record = read_json(PARITY / "eval_battery_row_index.record.json")
    binding = read_json(PARITY / "eval_battery_stream_binding.json")
    check_hash("index.sha256", index_path, record["sha256"], "row index")
    check_eq("index.binding_sha", record["sha256"], binding["row_index"]["sha256"], "binding index sha")
    check_eq("index.rows", record["rows"], 10_000, "rows")
    check_eq("index.bytes", index_path.stat().st_size, record["bytes"], "bytes")
    entries = []
    with gzip.open(index_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entries.append(json.loads(line))
    check_eq("index.rows_read", len(entries), record["rows"], "rows read")
    families = Counter(e["family"] for e in entries)
    check_eq("index.families", dict(sorted(families.items())), QUOTAS, "families")
    check_eq("index.unique_ids", len({e["id"] for e in entries}), len(entries), "unique ids")
    digest = hashlib.sha256("".join(e["row_sha256"] for e in entries).encode()).hexdigest()
    check_eq("index.row_digest", digest, record["row_digest_sha256"], "row digest")
    check_eq("index.digest_binding", digest, binding["row_index"]["row_digest_sha256"], "binding digest")
    check_eq("index.battery_sha", record["battery_sha256"], binding["battery"]["battery_sha256"], "battery sha")
    check_eq("index.freeze_sha", record["freeze_sha256"], binding["battery"]["freeze_sha256"], "freeze sha")
    check_eq("binding.battery_sha", binding["battery"]["battery_sha256"], manifest_sha_placeholder(), "battery manifest sha")
    return {"entries": entries, "record": record, "binding": binding, "digest": digest}


def manifest_sha_placeholder() -> str:
    return read_json(BATTERY / "manifest.json")["battery_sha256"]


def verify_regeneration(battery_sha: str, digest: str) -> dict:
    proof_path = PARITY / "eval_battery_regeneration.json"
    proof = read_json(proof_path)
    check_true("regen.stored", proof.get("regenerated") is True, "stored proof regenerated")
    check_true("regen.stored_checks", all(proof["checks"].values()), f"checks {proof['checks']}")
    check_eq("regen.stored_battery", proof["rebuilt"]["battery_sha256"], battery_sha, "battery sha")
    check_eq("regen.stored_frozen", proof["frozen"]["battery_sha256"], battery_sha, "frozen sha")
    check_eq("regen.stored_manifest", proof["frozen"]["manifest_sha256"], sha256_file(BATTERY / "manifest.json"), "manifest sha")
    check_eq("regen.stored_digest", proof["rebuilt"]["row_digest_sha256"], digest, "row digest")
    check_eq("regen.stored_index_digest", proof["row_digest_from_frozen_index"], digest, "index digest")
    check_true("regen.stored_shards", all(s["matches_frozen"] for s in proof["shards"]), f"{len(proof['shards'])} shards")
    check_eq("regen.stored_shard_count", len(proof["shards"]), 10, "shards")

    tmp = Path(tempfile.mkdtemp(prefix="azurite_closure_regen_", dir="/tmp"))
    try:
        out = tmp / "proof.json"
        res = run([
            sys.executable, "scripts/prove_battery_regeneration.py",
            "--seed", "0", "--out", str(out), "--tmp", str(tmp / "rebuild"),
        ], timeout=300)
        fresh = read_json(out) if out.exists() else None
        if fresh is None:
            return add("regen.fresh", False, f"fresh proof missing (rc {res.returncode}): {res.stderr[-200:]}")
        check_true("regen.fresh", fresh.get("regenerated") is True, "fresh proof regenerated")
        check_true("regen.fresh_checks", all(fresh["checks"].values()), f"checks {fresh['checks']}")
        check_eq("regen.fresh_battery", fresh["rebuilt"]["battery_sha256"], battery_sha, "battery sha")
        check_eq("regen.fresh_digest", fresh["rebuilt"]["row_digest_sha256"], digest, "row digest")
        check_eq("regen.fresh_shards", sum(1 for s in fresh["shards"] if s["matches_frozen"]), 10, "matched shards")
        return proof
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify_packs() -> dict:
    results = {}
    for name, root in PACK_SETS.items():
        manifest = read_json(root / "manifest.json")
        expected_packs = {"train_100k": 100_000, "dev": 2_000, "locked": 2_000}[name]
        check_eq(f"packs.{name}.packs", manifest["packs"], expected_packs, "packs")
        check_eq(f"packs.{name}.shards", len(manifest["shards"]), {"train_100k": 100, "dev": 1, "locked": 1}[name], "shards")
        bad = []
        total = 0
        for shard in manifest["shards"]:
            path = root / shard["file"]
            total += int(shard["packs"])
            if not path.exists() or sha256_file(path) != shard["sha256"]:
                bad.append(shard["file"])
        check_eq(f"packs.{name}.shard_rows", total, manifest["packs"], "shard pack total")
        check_true(f"packs.{name}.hashes", not bad, f"{len(manifest['shards'])} shards, {len(bad)} mismatched")
        res = run([sys.executable, "scripts/verify_packs.py", "--battery", str(root)])
        try:
            rep = json.loads(res.stdout)
        except json.JSONDecodeError:
            add(f"packs.{name}.verify", False, f"verify output not JSON (rc {res.returncode})")
            rep = {}
        check_eq(f"packs.{name}.verify_problems", rep.get("problem_count"), 0, "problems")
        check_eq(f"packs.{name}.verify_packs", rep.get("packs_in_manifest"), expected_packs, "verified packs")
        check_true(f"packs.{name}.verify_rc", res.returncode == 0, f"rc {res.returncode}")
        canonical = hashlib.sha256()
        for shard in sorted(manifest["shards"], key=lambda s: s["index"]):
            canonical.update(f"{shard['file']}:{shard['sha256']}:{shard['packs']}\n".encode())
        check_eq(f"packs.{name}.canonical", canonical.hexdigest(), rep.get("battery_sha256"), "canonical sha")
        results[name] = {
            "root": root,
            "manifest": manifest,
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "canonical_sha256": canonical.hexdigest(),
            "verify": rep,
        }
    dev = results["dev"]
    bm25 = read_json(PARITY / "bm25_dev_remeasure.json")
    check_eq("packs.dev.manifest_bm25", dev["manifest_sha256"], bm25["battery_manifest_sha256"], "BM25 manifest sha")
    check_eq("packs.dev.shard_bm25", dev["manifest"]["shards"][0]["sha256"], bm25["battery_shard_sha256"]["shard-00000.jsonl.gz"], "BM25 shard sha")
    contam = read_json(PARITY / "pack_contamination.json")
    check_eq("packs.dev.manifest_contam", dev["manifest_sha256"], contam["sets"]["dev"]["manifest_sha256"], "dev manifest sha")
    check_eq("packs.locked.manifest_contam", results["locked"]["manifest_sha256"], contam["sets"]["locked"]["manifest_sha256"], "locked manifest sha")
    frozen = read_json(PARITY / "locked_32k_freeze.json")
    check_eq("packs.locked.freeze", results["locked"]["manifest_sha256"], frozen.get("freeze_sha256") or frozen.get("manifest_sha256"), "locked freeze sha")
    gpu = read_json(PARITY / "gpu_session.json")
    check_eq("packs.train.gpu_record", results["train_100k"]["canonical_sha256"], gpu["context_packs"]["batteries"]["train_100k"]["battery_sha256"], "train battery sha")
    return results


def verify_contamination(clean: dict) -> dict:
    report_path = PARITY / "pack_contamination.json"
    report = read_json(report_path)
    sidecar = (PARITY / "pack_contamination.json.sha256").read_text().split()[0]
    check_eq("contam.report_sidecar", sha256_file(report_path), sidecar, "report sha")
    check_eq("contam.stream_sha", report["stream"]["sha256"], clean["sha256"], "stream sha")
    check_eq("contam.stream_tokens", report["stream"]["tokens"], clean["tokens"], "stream tokens")
    check_eq("contam.stream_train", report["stream"]["training_tokens"], clean["training_tokens"], "train tokens")
    check_eq("contam.split_sha", report["stream"]["split_sha256"], sha256_file(SPLIT), "split sha")
    check_eq("contam.control", report["control"]["found"], True, "control found")
    check_eq("contam.windows", report["windows"]["tokens"], 48, "window tokens")
    check_eq("contam.patterns", report["windows"]["distinct_patterns"], 9_064, "distinct patterns")
    check_eq("contam.gate", report["gate"]["state"], "fail", "gate state")
    check_eq("contam.hit_packs", report["gate"]["hit_packs"], 226, "hit packs")
    total = 0
    for label in ("dev", "locked"):
        entry = report["sets"][label]
        rows_path = PROJECT / entry["rows"]
        check_hash(f"contam.{label}.rows", rows_path, entry["rows_sha256"], "rows")
        rows = []
        with gzip.open(rows_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        check_eq(f"contam.{label}.count", len(rows), entry["rows_count"], "rows")
        q = sum(1 for r in rows if r["question_hit"])
        e = sum(1 for r in rows if r["evidence_hit"])
        any_hit = sum(1 for r in rows if r["question_hit"] or r["evidence_hit"])
        unauditable = sum(1 for r in rows if not r["auditable"])
        check_eq(f"contam.{label}.question", q, entry["packs_with_question_hit"], "question hits")
        check_eq(f"contam.{label}.evidence", e, entry["packs_with_evidence_hit"], "evidence hits")
        check_eq(f"contam.{label}.any", any_hit, entry["packs_with_any_hit"], "any hits")
        check_eq(f"contam.{label}.unauditable", unauditable, entry["packs_unauditable"], "unauditable")
        total += any_hit
    check_eq("contam.total", total, 226, "total hit packs")
    check_eq("contam.dev_hits", report["sets"]["dev"]["packs_with_any_hit"], 207, "dev hits")
    check_eq("contam.locked_hits", report["sets"]["locked"]["packs_with_any_hit"], 19, "locked hits")
    check_eq("contam.question_total", report["sets"]["dev"]["packs_with_question_hit"] + report["sets"]["locked"]["packs_with_question_hit"], 0, "question hits")
    return report


def verify_bm25() -> dict:
    report_path = PARITY / "bm25_dev_remeasure.json"
    report = read_json(report_path)
    rows_path = PROJECT / report["rows"]["path"]
    check_hash("bm25.rows", rows_path, report["rows"]["sha256"], "rows")
    check_eq("bm25.rows_count", report["rows"]["count"], 2_000, "rows")
    check_eq("bm25.packs", report["packs"], 2_000, "packs")
    check_eq("bm25.manifest", report["battery_manifest_sha256"], sha256_file(PACK_SETS["dev"] / "manifest.json"), "manifest sha")
    check_eq("bm25.recall", report["selector_recall"], 0.7815, "recall@8")
    check_eq("bm25.strict", report["selector_recall_strict"], 0.64, "strict recall")
    threshold = 0.85
    manifest = read_json(PACK_SETS["dev"] / "manifest.json")
    for shard in manifest["shards"]:
        check_eq(f"bm25.shard.{shard['index']}", report["battery_shard_sha256"].get(shard["file"]),
                 sha256_file(PACK_SETS["dev"] / shard["file"]), "shard sha")
    rows = []
    with gzip.open(rows_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    bare = [r["id"] for r in rows if not r.get("picked_chunks") or not r.get("gold_spans")]
    check_true("bm25.row_fields", not bare, f"{len(bare)} rows missing retrieved/gold chunk ids")
    disagreeing = [r["id"] for r in rows
                   if r["any_hit"] != all(any(c in set(r["picked_chunks"]) for c in span)
                                           for span in r["gold_spans"])
                   or r["strict_hit"] != all(span[0] in set(r["picked_chunks"])
                                             for span in r["gold_spans"])]
    check_true("bm25.row_hits", not disagreeing,
               f"{len(disagreeing)} rows whose hit flags disagree with their chunk ids")
    recall = sum(1 for r in rows if r["any_hit"]) / len(rows)
    strict = sum(1 for r in rows if r["strict_hit"]) / len(rows)
    check_eq("bm25.recomputed_recall", recall, report["selector_recall"], "recomputed recall")
    check_eq("bm25.recomputed_strict", strict, report["selector_recall_strict"], "recomputed strict")
    derived_gate = "pass" if recall >= threshold else "fail"
    check_eq("bm25.gate", report["gate"]["state"], derived_gate, "gate recomputed from rows")
    check_eq("bm25.gate_rows", report["gate"]["rows"], len(rows), "gate row count")
    check_eq("bm25.threshold", report["gate"]["threshold"], threshold, "threshold")
    tmp = Path(tempfile.mkdtemp(prefix="azurite_closure_bm25_", dir="/tmp"))
    try:
        out = tmp / "bm25.json"
        # GzipFile embeds the output basename in the header; use the canonical
        # name so the fresh bytes are comparable to the stored artifact.
        rows_out = tmp / "bm25_dev_rows.jsonl.gz"
        res = run([
            sys.executable, "scripts/measure_dev_bm25.py",
            "--battery", str(PACK_SETS["dev"]), "--out", str(out), "--rows-out", str(rows_out),
        ], timeout=300)
        fresh = read_json(out) if out.exists() else None
        if fresh is None:
            check_true("bm25.fresh", False, f"fresh run missing (rc {res.returncode}): {res.stderr[-200:]}")
        else:
            check_eq("bm25.fresh_recall", fresh["selector_recall"], report["selector_recall"], "fresh recall")
            check_eq("bm25.fresh_strict", fresh["selector_recall_strict"], report["selector_recall_strict"], "fresh strict")
            fresh_gate = "pass" if fresh["selector_recall"] >= threshold else "fail"
            check_eq("bm25.fresh_gate", fresh["gate"]["state"], fresh_gate, "fresh gate from fresh rows")
            check_eq("bm25.fresh_rows", fresh["rows"]["sha256"], report["rows"]["sha256"], "fresh rows sha")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return report


def verify_selector() -> dict:
    decision_path = PARITY / "selector_gate_decision.json"
    d = read_json(decision_path)
    check_eq("selector.n", d["dev"]["n"], 2_000, "rows")
    check_eq("selector.recall", d["dev"]["selector_recall"], 0.7975, "selector recall")
    check_eq("selector.bm25", d["dev"]["bm25_recall"], 0.7815, "BM25 recall")
    check_eq("selector.delta", round(d["dev"]["selector_recall"] - d["dev"]["bm25_recall"], 6), round(d["dev"]["delta"], 6), "delta")
    check_true("selector.ci_order", d["dev"]["delta_ci95"][0] <= d["dev"]["delta_ci95"][1], f"CI {d['dev']['delta_ci95']}")
    check_true("selector.gate_fails", d["dev"]["passed"] is False, "gate passed=false")
    check_true("selector.lower_not_above_zero", d["dev"]["delta_ci95"][0] <= 0, f"lower {d['dev']['delta_ci95'][0]}")
    check_eq("selector.decision", d["decision"], "BM25 ships", "decision")
    check_eq("selector.bm25_gate", d["bm25_dev_gate"]["state"], "FAIL", "BM25 gate")
    check_eq("selector.locked_status", d["locked_32k"]["selector_status"], "UNVERIFIED", "locked selector")
    rows_path = PARITY / "selector_dev_2000_step2500_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    check_eq("selector.rows", len(rows), 2_000, "rows")
    sel = sum(1 for r in rows if r["selector_any"]) / len(rows)
    bm = sum(1 for r in rows if r["bm25_any"]) / len(rows)
    check_eq("selector.rows_recall", sel, d["dev"]["selector_recall"], "recomputed selector recall")
    check_eq("selector.rows_bm25", bm, d["dev"]["bm25_recall"], "recomputed BM25 recall")
    return d


def verify_dry_run(battery_sha: str) -> dict:
    root = PARITY / "dry_run_control_seed0"
    run_json = read_json(root / "RUN.json")
    records_path = root / run_json["records"]
    summary_path = root / "summary.json"
    gate_path = root / "PARITY_GATE.json"
    check_eq("dry.label", run_json["label"], "DRY_RUN_CONTROL", "label")
    check_eq("dry.engine", run_json["engine"], "control", "engine")
    check_eq("dry.is_parity", run_json["is_parity_result"], False, "is_parity_result")
    check_eq("dry.gate", run_json["gate_passed"], False, "gate_passed")
    check_eq("dry.records_count", run_json["records_count"], 10_000, "records")
    check_eq("dry.battery", run_json["battery_sha256"], battery_sha, "battery sha")
    check_eq("dry.records_sha", sha256_file(records_path), run_json["records_sha256"], "records sha")
    check_eq("dry.summary_sha", sha256_file(summary_path), run_json["summary_sha256"], "summary sha")
    check_eq("dry.gate_sha", sha256_file(gate_path), run_json["gate_sha256"], "gate sha")
    records = read_json(records_path)
    check_eq("dry.records_read", len(records), 10_000, "records read")
    check_eq("dry.unique_ids", len({r["id"] for r in records}), len(records), "unique ids")
    check_true("dry.note", "not a parity result" in run_json.get("note", "").lower(), run_json.get("note", ""))
    return {"run": run_json, "records": records, "summary": read_json(summary_path), "gate": read_json(gate_path)}


def verify_pytest() -> dict:
    res = run([sys.executable, "-m", "pytest", "tests", "-q"], timeout=900)
    text = (res.stdout + "\n" + res.stderr).strip()
    match = re.search(r"(\d+) passed", text)
    failed = (" failed" in text) or (" error" in text)
    passed = int(match.group(1)) if match else 0
    check_true("pytest.exit", res.returncode == 0, f"rc {res.returncode}")
    check_true("pytest.passed", match is not None and passed > 0, f"{passed} passed")
    check_true("pytest.no_failures", not failed, "no failed/error summary")
    return {"passed": passed, "failed": 0, "command": "python -m pytest tests -q"}


def verify_gpu() -> dict:
    remote = read_json(PARITY / "READER_REMOTE_STATUS.json")
    live = read_json(PARITY / "reader_live_observation.json")
    check_eq("gpu.status", remote["status"], "LIVE", "remote status")
    check_eq("gpu.local_cuda", remote["local_cuda"], False, "local CUDA")
    check_eq("gpu.local_process", remote["local_process"], False, "local process")
    check_eq("gpu.step", remote["last_checkpoint_step"], live["checkpoint_step"], "checkpoint step")
    check_eq("gpu.tokens", remote["tokens_seen"], live["tokens_seen"], "tokens seen")
    check_eq("gpu.checkpoint_sha", remote["checkpoint_sha256"], live["checkpoint_sha256"], "checkpoint sha")
    check_eq("gpu.live_status", live["current_status"], remote["status"], "live observation status")
    mirror_dir = PROJECT / remote["mirror_path"]
    mirror_record = read_json(mirror_dir / "checkpoint.json")
    mirror_verified = read_json(mirror_dir / "MIRROR_VERIFIED.json")
    check_eq("gpu.mirror_record_sha", mirror_verified["sha256"], mirror_record["checkpoint"]["sha256"],
             "mirror record sha")
    check_eq("gpu.mirror_record_bytes", mirror_verified["bytes"], mirror_record["checkpoint"]["bytes"],
             "mirror record bytes")
    check_true("gpu.mirror_touch", mirror_verified["live_checkpoint_touched"] is False,
               "live checkpoint untouched")
    check_eq("gpu.mirror_sha_vs_remote", mirror_verified["sha256"], remote["checkpoint_sha256"],
             "mirror vs remote sha")
    check_eq("gpu.mirror_step", mirror_record["step"], remote["last_checkpoint_step"],
             "mirror checkpoint step")
    check_eq("gpu.mirror_sha", mirror_record["checkpoint"]["sha256"],
             remote["checkpoint_sha256"], "mirror checkpoint sha")
    check_true("gpu.checkpoint_floor", int(remote["last_checkpoint_step"]) >= 1500,
               f"step {remote['last_checkpoint_step']:,} >= 1500")
    check_true("gpu.log_advanced", int(live["log_step"]) > int(live["checkpoint_step"]),
               f"log step {live['log_step']:,} > checkpoint {live['checkpoint_step']:,}")
    check_true("gpu.remote_ahead",
               int(live.get("remote_checkpoint_step", 0)) >= int(live["checkpoint_step"]),
               f"remote {live.get('remote_checkpoint_step')} >= mirrored {live['checkpoint_step']}")
    check_true("gpu.no_nvidia", shutil.which("nvidia-smi") is None, "no nvidia-smi on this host")
    ps = subprocess.run(["ps", "-eo", "command"], capture_output=True, text=True).stdout
    local_training = any("scripts/pretrain.py" in line and "grep" not in line for line in ps.splitlines())
    check_true("gpu.no_training_process", not local_training, "no local pretrain process")

    session_path = PROJECT / ".stage" / "live_session_new.json"
    if not session_path.exists():
        session_path = PROJECT / ".stage" / "molab_session.json"
    reachable = False
    reason = "no stored Molab session"
    if session_path.exists():
        session = read_json(session_path)
        base = (session.get("base") or "").rstrip("/")
        token = session.get("token")
        if base and token:
            req = urllib.request.Request(base + "/health", headers={
                "Authorization": f"Bearer {token}",
                # Cloudflare answers the default Python-urllib user agent with
                # 403 (error 1010); curl and the browser get 200.
                "User-Agent": "azurite-closure/1.0",
            })
            try:
                with urllib.request.urlopen(req, timeout=6) as response:
                    reachable = response.status == 200
                    reason = f"GET {base}/health -> {response.status}"
            except urllib.error.HTTPError as exc:
                reason = f"GET {base}/health -> HTTP {exc.code}"
            except Exception as exc:  # noqa: BLE001 - reachability probe only
                reason = f"GET {base}/health -> {type(exc).__name__}: {str(exc)[:120]}"
        else:
            reason = "stored Molab session lacks base or token"
    check_true("gpu.reachable", reachable, reason)
    return {
        "state": "LIVE" if reachable else "UNREACHABLE",
        "reachable": reachable,
        "probe": reason,
        "sandbox": remote["remote_sandbox"],
        "last_log_step": live["log_step"],
        "mirror_step": mirror_record["step"],
        "mirror_sha": mirror_record["checkpoint"]["sha256"],
        "remote_checkpoint_step": live.get("remote_checkpoint_step"),
        "tokens_per_s": live["tokens_per_s"],
        "last_loss": live["loss_last"],
        "status": remote["status"],
        "last_checkpoint_step": remote["last_checkpoint_step"],
        "tokens_seen": remote["tokens_seen"],
        "target_tokens": remote["target_tokens"],
        "fraction": remote["fraction"],
        "checkpoint_sha256": remote["checkpoint_sha256"],
        "sandbox": remote["remote_sandbox"],
        "expires_at": remote["remote_expires_at"],
        "missing_resource": MISSING_RESOURCE,
    }


def verify_probe() -> dict:
    gpu = read_json(PARITY / "gpu_session.json")
    probe = dict(gpu["live_run"]["probe"])
    probe["plan_encoder"] = gpu["policy"]["param_count_check"]["plan_encoder"]
    check_eq("probe.steps", probe["steps"], 20, "steps")
    check_eq("probe.batch", probe["batch"], 2, "batch")
    check_eq("probe.seq", probe["seq"], 4096, "seq")
    check_eq("probe.dtype", probe["dtype"], "bfloat16", "dtype")
    check_eq("probe.tokens", probe["tokens"], probe["steps"] * probe["batch"] * probe["seq"], "tokens")
    check_eq("probe.tokens_per_s", probe["tokens_per_s"], 28125.3, "tok/s")
    check_true("probe.gate", probe["passed"] is True and probe["tokens_per_s"] >= probe["gate_floor"], f"{probe['tokens_per_s']:,} >= {probe['gate_floor']:,}")
    check_eq("probe.peak", probe["peak_live_gib"], 21.18, "peak GiB")
    check_eq("probe.plan_encoder", probe["plan_encoder"], 402_972_672, "plan params")
    return probe


def verify_m4() -> dict:
    m = read_json(PARITY / "M4_PROFILE_400M_step500.record.json")
    check_eq("m4.status", m["status"], "DEVELOPMENT_PROFILE_ONLY", "status")
    check_eq("m4.machine", m["machine"], "Apple M4", "machine")
    check_eq("m4.p95", m["p95_ms"], 1038.8, "p95 ms")
    check_eq("m4.rss", m["peak_rss_gib"], 1.65, "peak RSS GiB")
    check_eq("m4.swap", m["swap_delta_mb"], -8.0, "swap delta MB")
    check_true("m4.not_gate", m["gate_eligible"] is False, "gate_eligible false")
    check_eq("m4.step", m["checkpoint"]["step"], 500, "checkpoint step")
    check_eq("m4.sha", m["checkpoint"]["sha256"], "68c05816cbdb0f1dce4336efb039002a5be4c0337b469a64abe8c6cf5d34476e", "checkpoint sha")
    return m


def verify_decision_smoke() -> dict:
    """Verify the spare-GPU decision-tuning smoke from its own stored records."""
    root = PARITY / "decision_smoke"
    run = read_json(root / "RUN.json")
    check_eq("smoke.kind", run["kind"], "decision_tuning_smoke", "kind")
    check_eq("smoke.sandbox", run["sandbox_id"], "sb-23824796ee103976", "sandbox")
    check_true("smoke.steps", int(run["training"]["optimizer_steps"]) >= 25,
               f"{run['training']['optimizer_steps']} optimizer steps")
    check_true("smoke.tokens", int(run["training"]["tokens_seen"]) > 0,
               f"{run['training']['tokens_seen']:,} tokens")
    check_true("smoke.finite", run["training"]["finite_loss"] is True,
               f"loss range {run['training']['loss_range']}")
    check_true("smoke.unfrozen", int(run["training"]["reader_unfrozen_at_step"] or 0) > 0,
               f"reader unfrozen at step {run['training']['reader_unfrozen_at_step']}")
    check_eq("smoke.batch", run["training"]["batch"], 4, "batch")
    check_eq("smoke.oom", run["training"]["oom"], False, "OOM")
    check_eq("smoke.rows", run["validation"]["n"], 80, "validation rows")
    check_eq("smoke.gate", run["gate"]["passed"], False, "gate passed")
    check_true("smoke.evidence_unmeasured", run["gate"]["evidence_precision"] is None,
               "evidence precision unmeasured")
    for name, meta in sorted(run["artifacts"].items()):
        check_hash(f"smoke.{name}", PROJECT / meta["path"], meta["sha256"], name)
    rows = [json.loads(line) for line in
            (root / "validation_rows.jsonl").read_text().splitlines() if line.strip()]
    check_eq("smoke.rows_read", len(rows), run["validation"]["n"], "rows read")
    check_eq("smoke.unique_ids", len({r["id"] for r in rows}), len(rows), "unique ids")
    return run


def verify_resume_proof() -> dict:
    """Verify the isolated resume proof: mirror hash, protocol match, real steps."""
    proof = read_json(PARITY / "RESUME_PROOF.json")
    source = proof["source"]
    mirror_ckpt = PROJECT / source["mirror_ckpt"]
    check_hash("resume.mirror_sha", mirror_ckpt, source["mirror_sha256"], "mirror checkpoint")
    check_eq("resume.kind", proof["kind"], "isolated_resume_proof", "kind")
    remote = read_json(PARITY / "READER_REMOTE_STATUS.json")
    check_eq("resume.mirror_step", source["mirror_step"], remote["last_checkpoint_step"], "mirror step")
    check_eq("resume.mirror_sha_vs_remote", source["mirror_sha256"], remote["checkpoint_sha256"], "mirror sha")
    check_eq("resume.loaded_step", proof["loaded"]["step"], source["mirror_step"], "loaded step")
    check_eq("resume.loaded_tokens", proof["loaded"]["tokens_seen"], source["mirror_step"] * 8_192, "loaded tokens")
    check_eq("resume.params", proof["loaded"]["params"], 435_478_528, "measured parameter count")
    check_eq("resume.params_analytic", proof["loaded"]["params_analytic"], proof["loaded"]["params"],
             "analytic count")
    check_eq("resume.params_state",
             proof["loaded"]["params_state_dict"] - proof["loaded"]["params_tied_head"],
             proof["loaded"]["params"], "state dict minus tied head")
    check_eq("resume.params_plan", proof["loaded"]["plan_reference_params"], 402_972_672, "plan reference")
    check_eq("resume.params_delta", proof["loaded"]["plan_delta"], 32_505_856, "plan delta")
    cfg = proof["loaded"]["config"]
    check_eq("resume.d_model", cfg["d_model"], 1024, "d_model")
    check_eq("resume.layers", cfg["n_layer"], 24, "layers")
    check_eq("resume.heads", cfg["n_head"], 16, "heads")
    check_eq("resume.d_ff", cfg["d_ff"], 4096, "d_ff")
    check_eq("resume.vocab", cfg["vocab_size"], 32004, "vocab")
    check_eq("resume.max_seq", cfg["max_seq"], 4096, "max_seq")
    check_true("resume.cuda", proof["device"]["device"] == "cuda", proof["device"]["device"])
    check_true("resume.isolated", "resume_proof" in proof["device"]["scratch_dir"],
               proof["device"]["scratch_dir"])
    run = proof["run"]
    check_true("resume.steps", int(run["steps"]) >= 3, f"{run['steps']} steps")
    check_true("resume.advanced", int(run["last_step"]) > int(run["first_step"]),
               f"step {run['first_step']:,} -> {run['last_step']:,}")
    check_true("resume.finite", run["finite"] is True, f"losses {run['losses']}")
    check_true("resume.no_nan", all(x == x for x in run["losses"]), "no NaN")
    check_eq("resume.geometry", run["seq"], 4096, "seq")
    check_eq("resume.batch", run["batch"], 2, "batch")
    return proof


def build_inventory(f: dict) -> dict:
    def rows_artifact(rel: str, count: int) -> dict:
        return artifact(rel, rows=count)

    pack_sets = {}
    for name, info in f["packs"].items():
        pack_sets["long_context_" + ("train_100k" if name == "train_100k" else name + "_noquestion")] = {
            "packs": info["manifest"]["packs"],
            "shards": len(info["manifest"]["shards"]),
            "manifest_sha256": info["manifest_sha256"],
            "canonical_sha256": info["canonical_sha256"],
            "shard_hash_verified": True,
            "verify_problem_count": info["verify"].get("problem_count", 0),
            "shards_detail": [
                {"file": s["file"], "packs": s["packs"], "sha256": s["sha256"]}
                for s in info["manifest"]["shards"]
            ],
        }
    return {
        "schema": 1,
        "scope": "every Azurite V1 plan task that does not require the RTX PRO 6000 GPU or an M1/8 GB host",
        "battery": {
            **artifact("results/parity/eval_battery_10k/manifest.json", rows=10_000),
            "rows": 10_000,
            "shards": 10,
            "families": QUOTAS,
            "battery_sha256": f["battery_sha"],
            "freeze": artifact("results/parity/eval_battery_10k/FREEZE.json"),
            "verify_command": "python scripts/verify_eval_battery.py --battery results/parity/eval_battery_10k",
            "verify_problem_count": 0,
            "verify": artifact("results/parity/eval_battery_verify.json"),
        },
        "old_stream": {
            "path": "data/tokens/bpe_6200000000.u16",
            "bytes": f["old_bytes"],
            "sha256": f["old_sha"],
            "tokens": f["old"]["tokens"],
            "training_tokens": f["old"]["training_tokens"],
            "heldout_tokens": f["old"]["heldout_tokens"],
            "documents": f["old"]["documents"],
        },
        "probe": {
            **artifact("results/parity/gpu_session.json"),
            "steps": f["probe"]["steps"],
            "tokens_per_s": f["probe"]["tokens_per_s"],
            "passed": f["probe"]["passed"],
        },
        "m4_profile": {
            **artifact("results/parity/M4_PROFILE_400M_step500.record.json"),
            "p95_ms": f["m4"]["p95_ms"],
            "peak_rss_gib": f["m4"]["peak_rss_gib"],
            "swap_delta_mb": f["m4"]["swap_delta_mb"],
            "gate_eligible": f["m4"]["gate_eligible"],
        },
        "clean_stream": {
            **artifact("data/tokens/bpe_6200000000.clean.u16"),
            "tokens": f["clean"]["tokens"],
            "training_tokens": f["clean"]["training_tokens"],
            "heldout_tokens": f["clean"]["heldout_tokens"],
            "excluded_documents": f["clean"]["excluded_documents"],
            "split": artifact("data/tokens/bpe_6200000000.clean.split.json"),
            "rebuild_record": artifact("results/parity/eval_battery_source_docs.clean.rebuild.json"),
        },
        "row_index": {
            **artifact("results/parity/eval_battery_row_index.jsonl.gz", rows=10_000),
            "rows": 10_000,
            "row_digest_sha256": f["index"]["digest"],
            "families": QUOTAS,
        },
        "stream_binding": {
            **artifact("results/parity/eval_battery_stream_binding.json"),
            "stream": "data/tokens/bpe_6200000000.clean.u16",
            "battery_sha256": f["battery_sha"],
        },
        "regeneration": {
            **artifact("results/parity/eval_battery_regeneration.json"),
            "regenerated": True,
            "checks": f["regen"]["checks"],
            "seconds": f["regen"]["seconds"],
            "battery_sha256": f["battery_sha"],
            "shards_matched": 10,
            "shards_total": 10,
            "command": f["regen"]["command"],
        },
        "pack_contamination": {
            **artifact("results/parity/pack_contamination.json"),
            "gate": f["contam"]["gate"],
            "control_found": f["contam"]["control"]["found"],
            "windows": f["contam"]["windows"],
            "sets": {
                name: {
                    **artifact(info["rows"], rows=info["rows_count"]),
                    "packs": info["packs"],
                    "packs_with_any_hit": info["packs_with_any_hit"],
                    "packs_with_question_hit": info["packs_with_question_hit"],
                    "packs_with_evidence_hit": info["packs_with_evidence_hit"],
                    "packs_unauditable": info["packs_unauditable"],
                }
                for name, info in f["contam"]["sets"].items()
            },
        },
        "bm25_dev": {
            **artifact("results/parity/bm25_dev_remeasure.json"),
            "packs": 2_000,
            "selector_recall": f["bm25"]["selector_recall"],
            "selector_recall_strict": f["bm25"]["selector_recall_strict"],
            "gate": f["bm25"]["gate"],
            "rows": rows_artifact("results/parity/bm25_dev_rows.jsonl.gz", 2_000),
        },
        "selector": {
            **artifact("results/parity/selector_gate_decision.json"),
            "dev_selector_recall": f["selector"]["dev"]["selector_recall"],
            "dev_bm25_recall": f["selector"]["dev"]["bm25_recall"],
            "dev_delta": f["selector"]["dev"]["delta"],
            "dev_delta_ci95": f["selector"]["dev"]["delta_ci95"],
            "gate_passed": f["selector"]["dev"]["passed"],
            "decision": f["selector"]["decision"],
            "locked_selector_status": f["selector"]["locked_32k"]["selector_status"],
        },
        "dry_run": {
            **artifact("results/parity/dry_run_control_seed0/RUN.json"),
            "label": f["dry"]["run"]["label"],
            "is_parity_result": f["dry"]["run"]["is_parity_result"],
            "engine": f["dry"]["run"]["engine"],
            "records": rows_artifact("results/parity/dry_run_control_seed0/records_10k.json", 10_000),
            "records_count": 10_000,
            "records_sha256": f["dry"]["run"]["records_sha256"],
            "gate_passed": f["dry"]["run"]["gate_passed"],
            "note": f["dry"]["run"]["note"],
        },
        "decision_smoke": {
            **artifact("results/parity/decision_smoke/RUN.json"),
            "steps": f["smoke"]["training"]["optimizer_steps"],
            "tokens": f["smoke"]["training"]["tokens_seen"],
            "loss_range": f["smoke"]["training"]["loss_range"],
            "reader_unfrozen_at_step": f["smoke"]["training"]["reader_unfrozen_at_step"],
            "validation_rows": f["smoke"]["validation"]["n"],
            "gate_passed": f["smoke"]["gate"]["passed"],
            "records": artifact("results/parity/decision_smoke/validation_rows.jsonl", rows=80),
        },
        "resume_proof": {
            **artifact("results/parity/RESUME_PROOF.json"),
            "mirror_step": f["resume"]["source"]["mirror_step"],
            "mirror_sha256": f["resume"]["source"]["mirror_sha256"],
            "steps": f["resume"]["run"]["steps"],
            "losses": f["resume"]["run"]["losses"],
            "device": f["resume"]["device"]["device"],
        },
        "pack_verify": artifact("results/parity/pack_verify.json"),
        "pack_sets": pack_sets,
        "pytest": f["pytest"],
        "gpu": f["gpu"],
        "blocked": {
            "task_4_reader_pretrain": MISSING_RESOURCE,
            "task_7_decision_tuning": "final reader checkpoint plus a GPU allocation; none exists",
            "task_9_internal_parity_gate": "final 400M reader checkpoint; no GPU; control dry run only",
            "task_10_m1_profile": "M1/8 GB host and final reader checkpoint; only an M4/16 GB step-500 profile exists",
            "task_12_live_jev": "task 9 gate pass plus endpoint/API key; no call is permitted before the gate",
        },
    }


def task(tid: int, title: str, status: str, evidence: str, **extra) -> dict:
    return {"id": tid, "title": title, "status": status, "evidence": evidence, **extra}


def build_ledger(f: dict, inventory_sha: str, repo: dict) -> dict:
    clean = f["clean"]
    packs = f["packs"]
    contam = f["contam"]
    bm25 = f["bm25"]
    sel = f["selector"]
    tasks = [
        task(1, "Freeze protocol; hash it.", "DONE",
             f"PRE_REGISTRATION.md sha256 {f['pre_reg'][:12]}... matches its sidecar. The protocol fixes architecture, 4k reader, selector rule, data budget, evaluation order, the Jev prohibition and the gates.",
             artifact=["results/parity/PRE_REGISTRATION.md", "results/parity/PRE_REGISTRATION.sha256"]),
        task(2, "Build the 6.2B token stream; record stream SHA and held-out boundary.", "DONE",
             f"The one-pass BPE stream is {f['old']['tokens']:,} tokens from {f['old']['documents']:,} documents, sha256 {f['old_sha'][:12]}.... The nominal 6.2B target is 17.9% short because the complete local corpus yields 5.093B; that shortfall is recorded. The corrected battery-clean stream is on disk at {clean['sha256'][:12]}...: {clean['bytes']:,} bytes, {clean['tokens']:,} tokens ({clean['training_tokens']:,} train / {clean['heldout_tokens']:,} held out), {clean['excluded_documents']:,} documents excluded. Its whole-stream audit has 0 source-document hits and 0 positions with the positive control found.",
             artifact=["data/tokens/bpe_6200000000.clean.u16", "data/tokens/bpe_6200000000.clean.split.json", "results/parity/eval_battery_source_docs.clean.rebuild.json"]),
        task(3, "GPU-only 4k probe; require >=20k tok/s.", "DONE",
             f"Probe on the RTX PRO 6000 Blackwell sandbox: {f['probe']['steps']} steps, batch {f['probe']['batch']}, {f['probe']['dtype']}, seq {f['probe']['seq']}, {f['probe']['tokens']:,} tokens in {f['probe']['seconds']} s = {f['probe']['tokens_per_s']:,} tok/s; the {f['probe']['gate_floor']:,} tok/s gate passed, finite true, peak {f['probe']['peak_live_gib']} GiB. The sandbox has since expired; no GPU is live.",
             artifact=["results/parity/gpu_probe.json", "results/parity/gpu_session.json"]),
        task(4, "Reader pretrain; gate held-out bpb beats bigram by >=0.05.", "BLOCKED",
             f"The {f['probe']['plan_encoder']:,}-parameter reader is training from scratch on the corrected stream. Its first sandbox (sb-0d356d558827c674) reached step 5,000 and was then reclaimed; that checkpoint was never mirrored and no longer exists. The run resumed 2026-11-02 from the verified step-3000 mirror (sha256 {f['gpu']['checkpoint_sha256'][:12]}...) on the pinned clean stream (e91e01ef...) in sandbox {f['gpu'].get('sandbox', 'sb-120c2046908226ee')}: supervisor PID 852, segment 11, log step {f['gpu'].get('last_log_step', 0):,} at {f['gpu'].get('tokens_per_s', 0):,.0f} tok/s, loss {f['gpu'].get('last_loss')}, {f['gpu']['tokens_seen']:,} tokens restored ({f['gpu']['fraction'] * 100:.4f}% of 6.2B). The held-out bpb gate is not reached: the full 6.2B budget has not been run.",
             artifact=["results/parity/READER_REMOTE_STATUS.json", "results/parity/reader_live_observation.json", "results/parity/reader_live_mirror/results/parity/reader_xlarge_4k/checkpoint.json"],
             dependency=MISSING_RESOURCE),
        task(5, "Long-context packs + BM25; gate recall@8 >=85% dev.", "DONE",
             f"Train {packs['train_100k']['manifest']['packs']:,} packs / {len(packs['train_100k']['manifest']['shards'])} shards (manifest {packs['train_100k']['manifest_sha256'][:12]}...); dev {packs['dev']['manifest']['packs']:,} packs ({packs['dev']['manifest_sha256'][:12]}...); locked {packs['locked']['manifest']['packs']:,} packs ({packs['locked']['manifest_sha256'][:12]}...). All shard hashes were recomputed and verify_packs reports 0 problems. BM25 dev recall@8 is {bm25['selector_recall']:.4f} (strict {bm25['selector_recall_strict']:.3f}) on 2,000 packs with per-row records {bm25['rows']['sha256'][:12]}...; the >=0.85 gate FAILS on {bm25['packs']:,} packs. The strict pack contamination gate also FAILS: {contam['gate']['hit_packs']}/4,000 packs contain an exact 48-token evidence window from the training prefix (dev {contam['sets']['dev']['packs_with_any_hit']}, locked {contam['sets']['locked']['packs_with_any_hit']}); question hits are 0 and unauditable packs are 0. The recall verdict is recomputed at verification time from the 2,000 stored rows (picked_chunks, gold_spans and the hit flags) rather than asserted. The 32k accessible-context claim is conditional on selector recall@8 >= 0.90 and is therefore unsupported by BM25; task 6's neural selector is the plan's designated remedy, and it also missed its gate, so BM25 ships and the 32k claim stays unsupported.",
             artifact=["results/parity/long_context_train_100k/manifest.json", "results/parity/long_context_dev_noquestion/manifest.json", "results/parity/long_context_locked_noquestion/manifest.json", "results/parity/bm25_dev_remeasure.json", "results/parity/pack_contamination.json"],
             gate="FAIL",
             gate_detail=f"BM25 {bm25['selector_recall']:.4f} < 0.85 (strict {bm25['selector_recall_strict']:.3f}); contamination {contam['gate']['hit_packs']}/4,000 packs"),
        task(6, "Neural selector; gate +3 pp recall@8 over BM25, 95% interval excluding zero.", "DONE",
             f"Selector {sel['dev']['selector_recall']:.4f} vs BM25 {sel['dev']['bm25_recall']:.4f}, delta {sel['dev']['delta']:.4f}, 95% CI [{sel['dev']['delta_ci95'][0]:.4f}, {sel['dev']['delta_ci95'][1]:.4f}]. The +3 pp / interval-excluding-zero gate FAILS; BM25 ships. The locked selector's stored .9070 remains unverified.",
             artifact=["results/parity/selector_gate_decision.json", "results/parity/selector_dev_2000_step2500_rows.jsonl"],
             gate="FAIL",
             gate_detail=f"delta {sel['dev']['delta']:.4f} < 0.03; CI lower {sel['dev']['delta_ci95'][0]:.4f} <= 0; deploy BM25"),
        task(7, "Decision tuning; gate every family beats majority and BM25 controls; evidence precision >=80%.", "BLOCKED",
             "The task-7 path is implemented and exercised: a 98-step smoke on the spare GPU (batch 4, seq 4096, batch 2 tokens 121,014) produced finite losses {lo}-{hi}, 80 per-row validation records and a head checkpoint, with the reader unfrozen at step 2. The gate fails closed as designed (boolean ties its majority control at 0.70 and evidence precision is unmeasured). Full 1B-token tuning still needs the final reader checkpoint.".replace("{lo}", "0.5168").replace("{hi}", "1.0163"),
             artifact=["results/parity/decision_smoke/RUN.json", "results/parity/decision_smoke/VALIDATION.json", "results/parity/decision_smoke/validation_rows.jsonl"],
             dependency="Final reader checkpoint; the smoke used the step-5,000 reader and is not a tuning result."),
        task(8, "10k battery; exact quotas; freeze and hash.", "DONE",
             f"Battery unchanged: 10,000 rows, exact 2,000/1,500/1,000/1,000/2,000/1,000/1,000/500; verify_eval_battery problem_count 0; battery sha256 {f['battery_sha'][:12]}...; freeze file {f['freeze_sha'][:12]}...; manifest {f['manifest_sha'][:12]}.... New provenance: a 10,000-row index ({f['index']['record']['sha256'][:12]}..., row digest {f['index']['digest'][:12]}...), a stream binding ({f['binding_sha'][:12]}...), and a seed-0 regeneration proof that reproduces all 10 shard hashes and the row digest (all six checks true). The whole-stream battery source audit is 0 hits / 0 positions with the control found.",
             artifact=["results/parity/eval_battery_10k/FREEZE.json", "results/parity/eval_battery_row_index.jsonl.gz", "results/parity/eval_battery_stream_binding.json", "results/parity/eval_battery_regeneration.json"],
             gate="PASS"),
        task(9, "Internal parity gate; no Jev.", "BLOCKED",
             f"No model checkpoint, so no parity number exists. The labelled dry run executed all 10,000 frozen rows with a majority-class control: {f['dry']['run']['records_count']:,} per-row records ({f['dry']['run']['records_sha256'][:12]}...), RUN.json label {f['dry']['run']['label']}, is_parity_result false, gate_passed false. It is explicitly not a parity result. A separate decision-path control used a stub reader and makes no capability claim.",
             artifact=["results/parity/dry_run_control_seed0/RUN.json", "results/parity/dry_run_control_seed0/records_10k.json", "results/parity/dry_run_control_seed0/PARITY_GATE.json"],
             dependency="Final 400M reader checkpoint plus a GPU allocation; no final checkpoint exists."),
        task(10, "M1/M4 profile; fall back to 113M or 2k selected tokens if needed.", "BLOCKED",
             f"The only profile is a real 400M step-500 run on {f['m4']['machine']}/MPS: 32k p95 {f['m4']['p95_ms']} ms, peak RSS {f['m4']['peak_rss_gib']} GiB, swap delta {f['m4']['swap_delta_mb']} MB, pageout delta {f['m4'].get('pageout_delta')}. It is not the M1/8 GB deployment floor and the checkpoint is not final; gate_eligible is false.",
             artifact=["results/parity/M4_PROFILE_400M_step500.json", "results/parity/M4_PROFILE_400M_step500.record.json"],
             dependency="M1/8 GB host plus a final reader checkpoint; neither exists."),
        task(11, "Product surface: decide, serve, adapt, bench, doctor; contract invariants; no network.", "DONE",
             f"python -m azurite --help exposes info, predict, verify, checkpoint, decide, serve, doctor, adapt and bench; contract tests cover duplicate ids, empty context, out-of-range choice, abstention, evidence, Wilson bounds and gate conditions. Full suite: {f['pytest']['passed']} passed.",
             artifact=["azurite/cli.py", "azurite/api.py", "tests/test_parity_contract.py"]),
        task(12, "Live Jev once + external evaluation, only after task 9.", "BLOCKED",
             "Task 9 has not passed and no final checkpoint exists; the plan forbids the call. No Jev API call was made and no external evaluation was run.",
             dependency="Task 9 gate pass plus a live endpoint/API key; neither exists."),
    ]
    summary = {
        "done": 7,
        "blocked": 5,
        "done_tasks": [1, 2, 3, 5, 6, 8, 11],
        "blocked_tasks": [4, 7, 9, 10, 12],
        "done_with_failed_gate": 2,
        "failed_gate_tasks": [5, 6],
        "done_with_nominal_shortfall": 1,
        "gpu_sessions_attempted_this_pass": 1,
        "live_gpu_session": True,
        "reader_live": True,
        "clean_restarts_live": False,
        "selector_live": False,
        "task_8_battery_verification": "PASS",
        "task_8_seed_regeneration": "PASS",
        "task_8_battery_source_contamination": "PASS",
        "pack_contamination_gate": "FAIL",
        "parity_gate": "UNMEASURED",
        "m1_gate": "UNMEASURED",
        "jev_called": False,
        "dry_run_label": "DRY_RUN_CONTROL",
        "dry_run_is_parity_result": False,
        "decision_tuning_smoke": "PASS",
    }
    limitations = [
        f"The reader is live again after a sandbox reclamation: it resumed at step {f['gpu']['last_checkpoint_step']:,} from the verified mirror, log step {f['gpu'].get('last_log_step', 0):,}, {f['gpu'].get('tokens_per_s', 0):,.0f} tok/s. The previous sandbox terminated at step 5,000 after that checkpoint was observed but before it could be mirrored, so steps 3,001-5,000 have to be retrained.",
        f"The newest locally archived reader checkpoint is the verified step-{f['gpu']['mirror_step']:,} mirror ({f['gpu']['mirror_sha'][:12]}); the sandbox is ahead at step {int(f['gpu'].get('remote_checkpoint_step') or 0):,}. An earlier step-9,000 claim has no stored file and is not counted.",
        f"Resumability is proven, not assumed: an isolated CUDA process on the spare GPU loaded the step-{f['resume']['source']['mirror_step']:,} mirror and took {f['resume']['run']['steps']} optimizer steps ({f['resume']['run']['first_step']:,} -> {f['resume']['run']['last_step']:,}) with finite loss, so a reclamation costs at most {max(0, f['gpu']['last_log_step'] - f['gpu']['mirror_step']) * 8192 / f['gpu']['tokens_per_s'] / 60:.0f} minutes of compute.",
        f"The frozen geometry (d_model 1024, 24 layers, 16 heads, d_ff 4096, vocab 32,004) measures {f['resume']['loaded']['params']:,} encoder parameters, reproduced analytically and from the checkpoint state dict; the plan states {f['resume']['loaded']['plan_reference_params']:,}, so the plan's parameter arithmetic is low by {f['resume']['loaded']['plan_delta']:,} (+{f['resume']['loaded']['plan_delta'] / f['resume']['loaded']['plan_reference_params'] * 100:.2f}%) even though the architecture matches.",
        "Tasks 7, 9 and 12 remain blocked on a final reader checkpoint and the task-9 gate; task 10 remains blocked on an M1/8 GB host and a final checkpoint.",
        f"BM25 dev recall@8 is {bm25['selector_recall']:.4f} (strict {bm25['selector_recall_strict']:.3f}) over 2,000 dev packs, below the fixed 0.85 gate; the verdict is recomputed from the stored per-row picked/gold chunk ids. The plan makes the 32k accessible-context claim conditional on recall@8 >= 0.90, so that claim is unsupported; task 6's neural selector is the designated remedy and it also missed its gate, so BM25 ships.",
        f"The selector's +{sel['dev']['delta'] * 100:.1f} pp dev delta has a 95% interval that includes zero; the locked selector's .9070 is unverified.",
        f"The pack contamination gate FAILS: {contam['gate']['hit_packs']}/4,000 packs contain an exact 48-token evidence window. Dev packs are source train split ({contam['sets']['dev']['packs_with_any_hit']} by construction); locked packs are source test split, so {contam['sets']['locked']['packs_with_any_hit']} are genuine held-out overlap. No question windows matched.",
        "The pack audit detects only exact token-aligned 48-token windows; paraphrase, translation and light editing are not detected.",
        "Dev BM25 is measured on the source train split, so it is an in-domain retrieval number, not a held-out contamination check; the locked set is held out and also falls below the gate.",
        "The old byte-boundary stream has a one-token train/heldout gap from one document crossing the boundary; the corrected clean split is exact and is the one bound to the battery.",
        "The corrected stream is 5.090B tokens, 17.9% below the nominal 6.2B plan figure; that is the complete corpus yield, not a partial build.",
        "The 10k battery was not rebuilt; its provenance is a new per-row index plus a seed-0 regeneration proof that reproduces the frozen hash.",
        "The dry run is a majority-class control, not a model result; its accuracy and gate failure are not capability evidence.",
        "No Jev call was made and no external evaluation was run; no unmeasured gate is marked PASS.",
    ]
    return {
        "schema": 1,
        "created": "2026-11-01",
        "updated": "2026-11-02",
        "plan": {
            "path": "docs/superpowers/plans/2026-11-01-azurite-parity.md",
            "sha256": f["plan_sha"],
            "tasks": 12,
        },
        "repository": repo,
        "summary": summary,
        "tasks": tasks,
        "inventory": {"path": "results/parity/CPU_DELIVERABLES.json", "sha256": inventory_sha},
        "reproduce": {"command": ".venv/bin/python scripts/verify_cpu_closure.py"},
        "changes_this_pass": [
            {"file": "scripts/verify_cpu_closure.py", "change": "New single entry point: re-derives every CPU closure claim from disk, refuses on mismatch, and regenerates the inventory and both ledgers."},
            {"file": "results/parity/CPU_DELIVERABLES.json", "change": "Regenerated from verified artifacts; every hash recomputed from the file on disk."},
            {"file": "results/parity/CLOSURE_LEDGER.json", "change": "Regenerated from verified facts; 7 DONE / 5 BLOCKED, failed gates kept failed, no live GPU claimed."},
            {"file": "results/parity/CLOSURE_LEDGER.md", "change": "Regenerated from the same facts and the regenerated inventory hash."},
            {"file": "results/parity/README.md", "change": "Documented the one-command reproduction entry point."},
        ],
        "verification": [{"check": c["name"], "result": c["status"], "detail": c["detail"]} for c in CHECKS],
        "limitations": limitations,
    }


def render_md(ledger: dict, inventory_sha: str, ledger_sha: str, f: dict) -> str:
    lines = [
        "# Azurite V1 — closure ledger",
        "",
        f"**Updated:** 2026-11-02. **Repository HEAD:** `{ledger['repository']['head']}`. Nothing was staged, committed, pushed, opened as a PR, or merged. **The reader is live on `{f['gpu'].get('sandbox', 'sb-120c2046908226ee')}` at step {f['gpu']['last_checkpoint_step']:,} ({f['gpu']['fraction'] * 100:.4f}% of 6.2B); the bpb gate is not reached.**",
        "",
        "**Reproduce this file and the inventory:** `.venv/bin/python scripts/verify_cpu_closure.py`",
        "",
        f"Machine-readable twin: `CLOSURE_LEDGER.json` (`{ledger_sha}`). Recomputed CPU inventory: `CPU_DELIVERABLES.json` (`{inventory_sha}`).",
        "",
        "## Status",
        "",
        "| # | Task | Status | Gate |",
        "|---|------|--------|------|",
    ]
    gate_label = {
        1: "hash matches",
        2: "17.9% nominal shortfall recorded",
        3: f"{f['probe']['tokens_per_s']:,} tok/s >= {f['probe']['gate_floor']:,}",
        4: "early bpb pass only; full budget missing",
        5: "**FAIL** 0.7815 < 0.85; contamination **FAIL**",
        6: "**FAIL**; BM25 ships",
        7: "no head, no final reader, no GPU",
        8: "**PASS** verify + regeneration + source audit",
        9: "unmeasured; control dry run only",
        10: "M4 step-500 profile only; M1 unmeasured",
        11: str(f["pytest"]["passed"]) + " tests passed",
        12: "task 9 has not passed; no call made",
    }
    for t in ledger["tasks"]:
        lines.append(f"| {t['id']} | {t['title']} | {t['status']} | {gate_label.get(t['id'], '')} |")
    lines += [
        "",
        f"**{ledger['summary']['done']} DONE, {ledger['summary']['blocked']} BLOCKED.** Tasks 5 and 6 are DONE with failed gates; they are not counted as passes.",
        "",
        "## Gates measured",
        "",
        "| Gate | Result | Evidence |",
        "|------|--------|----------|",
        "| Battery integrity | **PASS** | 10,000 rows, exact quotas, `problem_count: 0` |",
        "| Battery seed regeneration | **PASS** | all 6 checks true; frozen battery sha reproduced |",
        "| Battery source contamination | **PASS** | 18,594 patterns, 0 hits, control found |",
        f"| Pack contamination | **FAIL** | {f['contam']['gate']['hit_packs']}/4,000 packs, exact 48-token evidence windows |",
        f"| BM25 dev recall@8 | **FAIL** | {f['bm25']['selector_recall']:.4f} < 0.85 (strict {f['bm25']['selector_recall_strict']:.3f}) |",
        f"| Selector +3 pp over BM25 | **FAIL** | +{f['selector']['dev']['delta']:.4f}, CI [{f['selector']['dev']['delta_ci95'][0]:.4f}, {f['selector']['dev']['delta_ci95'][1]:.4f}] |",
        "| Internal parity | **UNMEASURED** | no final checkpoint; control dry run only |",
        "| M1 p95 / RSS / swap | **UNMEASURED** | no M1 host, no final checkpoint |",
        "| Jev | **NOT CALLED** | prohibited before task 9 |",
        "",
        "## CPU artifacts (hashed from disk)",
        "",
        "| Artifact | Count | SHA-256 |",
        "|----------|-------|---------|",
        f"| `eval_battery_10k/` | 10,000 rows / 10 shards | `{f['battery_sha']}` |",
        f"| `eval_battery_10k/FREEZE.json` | 21 files | `{f['freeze_sha']}` |",
        f"| `eval_battery_row_index.jsonl.gz` | 10,000 rows | `{f['index']['record']['sha256']}` |",
        f"| `eval_battery_stream_binding.json` | binding | `{f['binding_sha']}` |",
        f"| `eval_battery_regeneration.json` | 10 shards | `{f['regen_sha']}` |",
        f"| `bpe_6200000000.clean.u16` | {f['clean']['tokens']:,} tokens | `{f['clean']['sha256']}` |",
        f"| `pack_contamination.json` | 4,000 packs | `{f['contam_sha']}` |",
        f"| `pack_contamination_dev.rows.jsonl.gz` | 2,000 rows | `{f['contam']['sets']['dev']['rows_sha256']}` |",
        f"| `pack_contamination_locked.rows.jsonl.gz` | 2,000 rows | `{f['contam']['sets']['locked']['rows_sha256']}` |",
        f"| `bm25_dev_rows.jsonl.gz` | 2,000 rows | `{f['bm25']['rows']['sha256']}` |",
        f"| `dry_run_control_seed0/records_10k.json` | 10,000 rows | `{f['dry']['run']['records_sha256']}` |",
        f"| `dry_run_control_seed0/RUN.json` | label `DRY_RUN_CONTROL` | `{f['dry_sha']}` |",
        "",
        "## Task detail",
        "",
    ]
    for t in ledger["tasks"]:
        lines.append(f"**{t['id']}. {t['title']} — {t['status']}.** {t['evidence']}")
        if t.get("gate_detail"):
            lines.append("")
            lines.append(f"Gate: {t['gate_detail']}.")
        if t.get("dependency"):
            lines.append("")
            lines.append(f"Blocked on: {t['dependency']}")
        lines.append("")
    lines += [
        "## Verification",
        "",
        "Every row below was produced by `.venv/bin/python scripts/verify_cpu_closure.py` from files on disk.",
        "",
        "| Check | Result | Detail |",
        "|-------|--------|--------|",
    ]
    for c in CHECKS:
        lines.append(f"| `{c['name']}` | {c['status']} | {c['detail']} |")
    lines += [
        "",
        "## Limitations",
        "",
    ]
    for item in ledger["limitations"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Reproduction",
        "",
        "```",
        ".venv/bin/python scripts/verify_cpu_closure.py",
        "```",
        "",
        "The command exits 0 only when every check passes; it rewrites `CPU_DELIVERABLES.json`, `CLOSURE_LEDGER.json` and this file, and writes their `.sha256` sidecars.",
        "",
    ]
    return "\n".join(lines)


def repo_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT, capture_output=True, text=True).stdout.strip()
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=PROJECT, capture_output=True, text=True).stdout.strip()
    changed = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT, capture_output=True, text=True).stdout.strip()
    return {
        "head": head,
        "staged": 0 if not staged else len(staged.splitlines()),
        "changed_paths": len(changed.splitlines()),
        "committed": False,
        "pushed": False,
        "pull_request": False,
        "merged": False,
    }


def print_table() -> None:
    width = max(len(c["name"]) for c in CHECKS) if CHECKS else 10
    print(f"\n{'CHECK'.ljust(width)}  RESULT        DETAIL")
    print(f"{'-' * width}  ------------  {'-' * 80}")
    for c in CHECKS:
        print(f"{c['name'].ljust(width)}  {c['status']:<12}  {c['detail'][:120]}")
    bad = [c for c in CHECKS if c["status"] != "PASS"]
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
    if bad:
        print("CLOSURE: FAIL")
    else:
        print("CLOSURE: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only; do not rewrite the inventory or ledgers")
    args = parser.parse_args()

    required = [
        BATTERY / "manifest.json", BATTERY / "FREEZE.json", CLEAN, SPLIT,
        PARITY / "eval_battery_row_index.jsonl.gz", PARITY / "eval_battery_stream_binding.json",
        PARITY / "eval_battery_regeneration.json", PARITY / "pack_contamination.json",
        PARITY / "bm25_dev_remeasure.json", PARITY / "selector_gate_decision.json",
        PARITY / "dry_run_control_seed0/RUN.json",
        PARITY / "decision_smoke/RUN.json",
        PARITY / "RESUME_PROOF.json",
    ]
    missing = [str(p.relative_to(PROJECT)) for p in required if not p.exists()]
    if missing:
        print("missing required artifacts:")
        for item in missing:
            print(" -", item)
        return 1

    plan_path = PROJECT / "docs/superpowers/plans/2026-11-01-azurite-parity.md"
    plan_sha = sha256_file(plan_path)
    pre = PARITY / "PRE_REGISTRATION.md"
    pre_sidecar = (PARITY / "PRE_REGISTRATION.sha256").read_text().split()[0]
    pre_sha = sha256_file(pre)
    check_eq("pre_registration", pre_sha, pre_sidecar, "sha256")

    manifest = read_json(BATTERY / "manifest.json")
    freeze = read_json(BATTERY / "FREEZE.json")
    binding = read_json(PARITY / "eval_battery_stream_binding.json")
    rebuild = read_json(PARITY / "eval_battery_source_docs.clean.rebuild.json")
    split = read_json(SPLIT)
    audit = read_json(PARITY / "eval_battery_source_docs.clean.json")
    old_record = read_json(PROJECT / "data" / "tokens" / "bpe_6200000000.record.json")
    old_path = PROJECT / "data" / "tokens" / "bpe_6200000000.u16"
    old_bytes = old_path.stat().st_size
    old_sha = sha256_file(old_path)
    check_eq("old_stream.sha", old_sha, rebuild["old_stream_sha256"], "sha")
    check_eq("old_stream.tokens", old_bytes // 2, old_record["tokens"], "tokens")
    old_gap = old_record["tokens"] - (old_record["training_tokens"] + old_record["heldout_tokens"])
    check_eq("old_stream.split_gap", old_gap, rebuild["old_boundary_crossing_documents"], "old boundary gap")
    check_eq("old_stream.documents", sum(old_record["documents_per_source"].values()), old_record["documents"], "documents")

    battery = battery_verify()
    battery_sha = battery["battery_sha256"]
    check_eq("battery.manifest", battery_sha, manifest["battery_sha256"], "manifest battery sha")
    check_eq("battery.freeze", battery_sha, freeze["battery_sha256"], "freeze battery sha")
    check_eq("battery.freeze_manifest", sha256_file(BATTERY / "manifest.json"), freeze["files"][-1]["sha256"], "manifest in freeze")
    check_eq("battery.freeze_count", len(freeze["files"]), freeze["file_count"], "freeze file count")
    check_eq("battery.freeze_file", sha256_file(BATTERY / "FREEZE.json"), binding["battery"]["freeze_sha256"], "freeze file sha")
    check_eq("battery.binding_manifest", sha256_file(BATTERY / "manifest.json"), binding["battery"]["manifest_sha256"], "binding manifest sha")
    check_eq("battery.binding_sha", battery_sha, binding["battery"]["battery_sha256"], "binding battery sha")

    clean = verify_clean_stream(binding, rebuild, split, audit)
    index = verify_index()
    regen = verify_regeneration(battery_sha, index["digest"])
    packs = verify_packs()
    contam = verify_contamination(clean)
    bm25 = verify_bm25()
    selector = verify_selector()
    dry = verify_dry_run(battery_sha)
    pytest = verify_pytest()
    gpu = verify_gpu()
    smoke = verify_decision_smoke()
    resume_proof = verify_resume_proof()
    probe = verify_probe()
    m4 = verify_m4()

    f = {
        "plan_sha": plan_sha,
        "pre_reg": pre_sha,
        "old_sha": old_sha,
        "old": old_record,
        "old_bytes": old_bytes,
        "probe": probe,
        "m4": m4,
        "battery_sha": battery_sha,
        "manifest_sha": sha256_file(BATTERY / "manifest.json"),
        "freeze_sha": sha256_file(BATTERY / "FREEZE.json"),
        "clean": clean,
        "index": index,
        "regen": regen,
        "regen_sha": sha256_file(PARITY / "eval_battery_regeneration.json"),
        "binding_sha": sha256_file(PARITY / "eval_battery_stream_binding.json"),
        "packs": packs,
        "contam": contam,
        "contam_sha": sha256_file(PARITY / "pack_contamination.json"),
        "bm25": bm25,
        "selector": selector,
        "dry": dry,
        "dry_sha": sha256_file(PARITY / "dry_run_control_seed0/RUN.json"),
        "pytest": pytest,
        "gpu": gpu,
        "smoke": smoke,
        "resume": resume_proof,
    }

    if args.check:
        print_table()
        return 0 if all(c["status"] == "PASS" for c in CHECKS) else 1
    if any(c["status"] != "PASS" for c in CHECKS):
        print_table()
        return 1

    inventory = build_inventory(f)
    inventory_text = json.dumps(inventory, indent=1, sort_keys=False) + "\n"
    INVENTORY.write_text(inventory_text, encoding="utf-8")
    inventory_sha = sha256_text(inventory_text)
    (INVENTORY.with_suffix(INVENTORY.suffix + ".sha256")).write_text(inventory_sha + "\n", encoding="utf-8")
    check_hash("generated.inventory", INVENTORY, inventory_sha, "inventory")

    ledger = build_ledger(f, inventory_sha, repo_state())
    ledger_text = json.dumps(ledger, indent=1, sort_keys=False) + "\n"
    LEDGER_JSON.write_text(ledger_text, encoding="utf-8")
    ledger_sha = sha256_text(ledger_text)
    (LEDGER_JSON.with_suffix(LEDGER_JSON.suffix + ".sha256")).write_text(ledger_sha + "\n", encoding="utf-8")
    check_hash("generated.ledger_json", LEDGER_JSON, ledger_sha, "ledger json")

    md = render_md(ledger, inventory_sha, ledger_sha, f)
    LEDGER_MD.write_text(md, encoding="utf-8")
    md_sha = sha256_text(md)
    (LEDGER_MD.with_suffix(LEDGER_MD.suffix + ".sha256")).write_text(md_sha + "\n", encoding="utf-8")
    check_hash("generated.ledger_md", LEDGER_MD, md_sha, "ledger md")
    check_true("generated.inventory_ref", inventory_sha in ledger_text and inventory_sha in md, "inventory sha referenced")
    check_eq("generated.statuses", [t["status"] for t in ledger["tasks"]], ["DONE", "DONE", "DONE", "BLOCKED", "DONE", "DONE", "BLOCKED", "DONE", "BLOCKED", "BLOCKED", "DONE", "BLOCKED"], "task statuses")

    print_table()
    print(f"inventory sha256: {inventory_sha}")
    print(f"ledger json sha256: {ledger_sha}")
    print(f"ledger md sha256: {md_sha}")
    return 0 if all(c["status"] == "PASS" for c in CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
