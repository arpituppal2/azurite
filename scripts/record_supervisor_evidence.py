#!/usr/bin/env python3
"""Read the GPU host's pretrain evidence and write it into gpu_session.json.

One probe prints a single JSON object between markers; everything else (record
merge, GPU-hour arithmetic, recovery assertions) happens locally, so a dropped
remote stream cannot leave a half-written record.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EXEC = PROJECT / ".agents/skills/marimo-pair/scripts/execute-code.sh"
SESSION = PROJECT / ".stage/live_session.json"
RECORD = PROJECT / "results/parity/gpu_session.json"

TARGET = 6_200_000_000
UNIQUE = 5_093_196_562
TRAIN = int(UNIQUE * 0.98)
PARAMS = 435_478_528
PLAN_PARAMS = 402_972_672
UNMANAGED_UP_S = 656.0  # 13:57:53Z launch -> 14:08:49Z handover

PROBE = r'''
import json, pathlib, subprocess, time
ws = pathlib.Path("/marimo/azurite_ws"); out = ws/"results/parity"; rd = out/"reader_xlarge_4k"
def ck(): return json.loads((rd/"checkpoint.json").read_text())
t = time.time() + 90
while time.time() < t and int(ck().get("step") or 0) < 2000: time.sleep(5)
st = json.loads((out/"gpu"/"supervisor_state.json").read_text())
rows = [json.loads(l) for l in (rd/"train.jsonl").read_text().splitlines() if l.strip()]
pid = st.get("pid"); rss = sw = None
try:
    for line in pathlib.Path("/proc/%s/status" % pid).read_text().splitlines():
        if line.startswith("VmRSS:"): rss = int(line.split()[1])
        if line.startswith("VmSwap:"): sw = int(line.split()[1])
except Exception: pass
sup = (out/"gpu"/"supervisor.log").read_text().strip().splitlines()[-4:]
hist = [json.loads(l) for l in (out/"gpu"/"supervisor.jsonl").read_text().splitlines() if l.strip()]
print("BEGIN_EVIDENCE")
print(json.dumps({"ck": ck(), "st": st, "rows": rows[-4:], "n": len(rows), "sup": sup,
 "hist": hist, "gpu": subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
 "rss": rss, "sw": sw}))
print("END_EVIDENCE")
'''


def fetch(attempts: int = 3) -> dict:
    ses = json.loads(SESSION.read_text())
    err = ""
    for i in range(attempts):
        proc = subprocess.run(
            ["bash", str(EXEC), "--url", ses["base"], "--token", ses["token"], "-c", PROBE],
            capture_output=True, text=True, timeout=300,
        )
        text = proc.stdout
        if "BEGIN_EVIDENCE" in text and "END_EVIDENCE" in text:
            body = text.split("BEGIN_EVIDENCE", 1)[1].split("END_EVIDENCE", 1)[0].strip()
            return json.loads(body)
        err = f"rc={proc.returncode} out={text[-200:]!r} err={proc.stderr[-200:]!r}"
        time.sleep(10)
    raise SystemExit(f"probe failed after {attempts} attempts: {err}")


def main() -> int:
    ev = fetch()
    ck, st = ev["ck"], ev["st"]
    rows = ev["rows"]
    rate = statistics.median(r["tokens_per_s"] for r in rows)
    left = max(0, TARGET - int(ck["tokens_seen"]))
    left_h = left / rate / 3600
    seg_tok = int(rate * 36000)
    consumed_h = round((UNMANAGED_UP_S + st["total_up_s"]) / 3600, 3)

    rec = json.loads(RECORD.read_text())
    rec["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["device"] = ev["gpu"]
    rec["pretrain"].update({
        "step": ck["step"], "tokens_seen": ck["tokens_seen"], "fraction": ck["fraction"],
        "status": ck["status"], "loss_last": rows[-1]["loss"], "tok_per_s": rate,
        "log_rows": ev["n"], "last_row": rows[-1],
        "checkpoint": {"path": "results/parity/reader_xlarge_4k/ckpt.pt",
                       "bytes": ck["checkpoint"]["bytes"],
                       "sha256": ck["checkpoint"]["sha256"], "status": ck["status"]},
    })
    # Keep proofs captured earlier: a supervisor generation change must not erase
    # the restart history it is evidence for.
    prior = rec.get("supervisor") or {}
    rec["supervisor"] = {
        "script": "scripts/gpu_supervisor.py",
        "pid": st["supervisor_pid"],
        "detached": "start_new_session; stdout -> results/parity/gpu/supervisor.log; outlives notebook and stream drops",
        "attempts": st["attempts_total"], "restarts": st["restarts"],
        "total_child_up_s": st["total_up_s"], "total_gap_s": st["total_gap_s"],
        "policy": {
            "backoff_base_s": 5, "backoff_max_s": 60, "max_consecutive_no_progress": 6,
            "fatal_give_up_only_on": [
                "refusing to start a fresh run", "refusing to restart from zero",
                "does not match --tokens", "do not match this invocation",
                "is not a loadable checkpoint"],
            "rc0_relaunch": "immediate (10h segment timeout is a clean exit, not a failure)",
        },
        "history": ev["hist"],
        "timeline": ev["sup"],
        "history_file": "results/parity/gpu/supervisor.jsonl",
    }
    for key in ("recovery_proof", "segment_relaunch_proof", "scope", "generations", "durability"):
        if key in prior:
            rec["supervisor"][key] = prior[key]
    # The recovery proof describes a specific past interruption (child 19668 at
    # 14:10:56Z). Never rebuild it from the *current* supervisor state: that
    # rewrites the gap and the resume step with an unrelated generation's
    # numbers, which reads as a fresh recovery that never happened.
    rec["throughput"] = {
        "measured_tok_per_s_median_recent": rate,
        "gpu_hours_consumed": consumed_h,
        "tokens_remaining": left,
        "gpu_hours_remaining": round(left_h, 1),
        "tokens_per_10h_segment": seg_tok,
        "segments_projected": int(-(-left // seg_tok)),
    }
    rec["resources"] = {
        "gpu_util_mem": ev["gpu"], "peak_live_gib": rows[-1]["peak_live_gib"],
        "swap_mb": rows[-1]["swap_mb"], "process_rss_kb": ev["rss"],
        "process_vmswap_kb": ev["sw"], "card_total_gib": 94.97,
    }
    rec["reconciliations"] = {
        "token_budget": {
            "frozen_target_tokens": TARGET,
            "stream_unique_tokens": UNIQUE,
            "stream_train_tokens": TRAIN,
            "heldout_tokens": UNIQUE - TRAIN,
            "passes_over_unique_train": round(TARGET / TRAIN, 4),
            "tokens_per_param_run": round(TARGET / PARAMS, 2),
            "tokens_per_param_plan": round(TARGET / PLAN_PARAMS, 2),
            "unique_tokens_per_param": round(TRAIN / PARAMS, 2),
            "explanation": (
                "The plan sized 6.2B tokens from 20.82 GB of text at ~0.30 tokens/byte. The shipped "
                "corpus yields 5,093,196,562 BPE tokens (0.563 per compressed byte), so the frozen 6.2e9 "
                "budget runs as 1.2422 passes over 4,991,332,630 unique training tokens. Tokens seen per "
                "parameter is 14.24 at the BPE parameter count (15.39 against the plan's byte-arm count). "
                "Budget and architecture left frozen; no re-prepare and no restart."
            ),
        },
        "parameters": {
            "plan_402_972_672": (
                "byte arm: 24 x 16,779,392 = 402,705,408 block params + byte embeddings 260 x 1024 = "
                "266,240 + final RMSNorm 1,024 = 402,972,672"
            ),
            "run_435_478_528": (
                "BPE arm: same 402,705,408 block params + BPE embeddings 32,004 x 1024 = 32,772,096 "
                "+ final RMSNorm 1,024 = 435,478,528"
            ),
            "delta": PARAMS - PLAN_PARAMS,
            "delta_factorisation": "(32,004 - 260) x 1,024 = 31,744 x 1,024 = 32,505,856",
            "mlm_head": "tied (tie_mlm_head=True), so pretrain_total == encoder; no separate head",

            "correction": (
                "An earlier note in this record called 266,240 an unexplained discrepancy; that note counted "
                "the byte vocabulary's embedding twice. The full decomposition above accounts for every "
                "parameter in both arms."
            ),
            "architecture_changed": False,
        },
    }
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({
        "step": ck["step"], "tokens_seen": ck["tokens_seen"], "loss": rows[-1]["loss"],
        "rate": rate, "gap_s": st["gap_s"], "attempts": st["attempts_total"],
        "segments_projected": rec["throughput"]["segments_projected"],
        "gpu_hours_remaining": rec["throughput"]["gpu_hours_remaining"],
        "swap_mb": rows[-1]["swap_mb"], "peak_live_gib": rows[-1]["peak_live_gib"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
