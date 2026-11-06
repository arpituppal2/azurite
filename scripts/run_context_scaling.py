#!/usr/bin/env python3
"""Run the pre-registered fixed-checkpoint context-scaling curve.

Each task trains one head at 256 positions, then evaluates the same weights at
256/512/1024/2048/4096. No locked row is read.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "context_scaling"
CKPT = PROJECT / "results" / "pretrain" / "l1_small_byte_ctx256" / "ckpt.pt"
CKPT_SHA = "8fd8bedee794ee4aa464af7c11fb06eb1d8dcfbfe82f529f8106a11a798ff131"
LENGTHS = [256, 512, 1024, 2048, 4096]
STEPS = 500
BATCH = 16
LR = 1e-4
TRAIN_ROWS = 2_500
TEST_ROWS = 750
THREADS = 2

TASKS = {
    "banking77_ctx": {"family": "classification", "pack_fields": False},
    "clinc_oos_ctx": {"family": "classification", "pack_fields": False},
    "boolq_ctx": {"family": "document_grounded", "pack_fields": True},
    "squad2_answerability_ctx": {"family": "document_grounded_abstention", "pack_fields": True},
    "contract_nli_ctx": {"family": "policy_rule", "pack_fields": True},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not CKPT.exists():
        raise SystemExit(f"missing checkpoint {CKPT}")
    if sha256_file(CKPT) != CKPT_SHA:
        raise SystemExit("checkpoint hash does not match the pre-registration")
    OUT.mkdir(parents=True, exist_ok=True)
    pre = PROJECT / "results" / "context_scaling" / "PRE_REGISTRATION.md"
    amendment = PROJECT / "results" / "context_scaling" / "PRE_REGISTRATION_AMENDMENT_1.md"
    (OUT / "PRE_REGISTRATION.sha256").write_text(sha256_file(pre) + "\n")
    amendment_sha = sha256_file(amendment)
    (OUT / "PRE_REGISTRATION_AMENDMENT_1.sha256").write_text(amendment_sha + "\n")

    runs = []
    for task, meta in TASKS.items():
        out = OUT / f"{task}.json"
        evidence = OUT / f"{task}_evidence"
        log = OUT / f"{task}.log"
        if out.exists():
            print(f"[skip existing] {task}", flush=True)
            runs.append({"task": task, "status": "skipped-existing", "path": str(out)})
            continue
        cmd = [
            sys.executable, "-u", str(PROJECT / "scripts" / "transfer_eval.py"),
            "--task", task, "--arm", "byte", "--rung", "small", "--path", "dual",
            "--init", "pretrained", "--ckpt", str(CKPT), "--max-seq", "256",
            "--examples", str(TRAIN_ROWS), "--test-examples", str(TEST_ROWS),
            "--steps", str(STEPS), "--batch", str(BATCH), "--lr", str(LR),
            "--seed", "0", "--device", "cpu", "--threads", str(THREADS),
            "--out", str(out), "--evidence-dir", str(evidence),
            "--eval-lengths", *map(str, LENGTHS),
        ]
        if meta["pack_fields"]:
            cmd.append("--pack-fields")
        print(f"[run] {task}: {' '.join(cmd)}", flush=True)
        started = time.time()
        with log.open("w") as fh:
            fh.write(" ".join(cmd) + "\n\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        elapsed = round(time.time() - started, 1)
        if proc.returncode != 0:
            print(f"[failed] {task} rc={proc.returncode} log={log}", flush=True)
            runs.append({"task": task, "status": "failed", "returncode": proc.returncode,
                         "seconds": elapsed, "log": str(log)})
            continue
        record = json.loads(out.read_text())
        runs.append({
            "task": task,
            "family": meta["family"],
            "status": "ok",
            "seconds": elapsed,
            "n": record["final"]["n"],
            "accuracy_at_training_length": record["final"]["accuracy"],
            "curve": [
                {
                    "max_seq": c["max_seq"],
                    "accuracy": c["eval"]["accuracy"],
                    "nll": c["eval"]["nll"],
                    "ece": c["eval"]["ece"],
                    "full_context_fraction": c["eval"]["full_context_fraction"],
                    "mean_packed_len": c["pack"]["mean_seq_len"],
                    "unpackable": c["pack"]["unpackable"],
                }
                for c in record.get("context_curve", [])
            ],
            "result": str(out),
            "log": str(log),
        })
        (OUT / "summary.json").write_text(json.dumps({
            "protocol": "context-scaling-fixed-checkpoint-v1",
            "checkpoint": {"path": str(CKPT.relative_to(PROJECT)), "sha256": CKPT_SHA},
            "lengths": LENGTHS,
            "steps": STEPS,
            "batch": BATCH,
            "lr": LR,
            "train_rows": TRAIN_ROWS,
            "test_rows": TEST_ROWS,
            "seed": 0,
            "device": "cpu",
            "pre_registration_sha256": sha256_file(pre),
            "amendment_sha256": amendment_sha,
            "pack_fields_for": [t for t, m in TASKS.items() if m["pack_fields"]],
            "runs": runs,
        }, indent=2) + "\n")
    print("[done] all runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
