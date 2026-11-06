#!/usr/bin/env python3
"""Wait for the two verification records, then restart both training runs clean.

Reader restart is gated on the full corrected-stream audit PASS. Selector restart
is gated on the clean-pack build PASS. The old reader is stopped only after the
stream record exists and passes; both new processes are detached under their
respective supervisors. This script records every PID and checkpoint it starts.
"""
from __future__ import annotations
import json, os, signal, subprocess, sys, time
from pathlib import Path

WS = Path("/marimo/azurite_ws")
PY = "/tmp/uv-venv/bin/python"
STATE = WS / "results/parity/gpu_clean/restart_coordinator.json"
LOG = WS / "results/parity/gpu_clean/coordinator.log"


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write(rec: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    tmp.replace(STATE)


def main() -> int:
    rec = {"started_at": stamp(), "status": "waiting", "reader": {}, "selector": {}}
    write(rec)
    deadline = time.time() + 3600
    while time.time() < deadline:
        audit = load(WS / "results/parity/eval_battery_source_docs.clean.json")
        if audit:
            rec["clean_audit"] = {k: audit.get(k) for k in
                                  ("new_stream_sha256", "new_tokens", "new_training_tokens",
                                   "excluded_documents", "excluded_source_docs", "old_hit_rows")}
            rec["clean_audit"]["gate"] = (audit.get("audit") or {}).get("gate")
            rec["clean_audit"]["control_found"] = (audit.get("audit") or {}).get("control_found")
            rec["clean_audit"]["whole_stream_hits"] = (audit.get("audit") or {}).get("whole_stream_hit_source_docs")
            if (audit.get("audit") or {}).get("gate") != "PASS":
                rec["status"] = "clean_audit_failed"
                write(rec)
                print("clean audit failed; not restarting", flush=True)
                return 2
            if not rec["reader"]:
                rec["status"] = "stopping_old_reader"
                write(rec)
                old = load(WS / "results/parity/reader_xlarge_4k/checkpoint.json")
                (WS / "results/parity/gpu_clean/old_reader_stop.json").write_text(
                    json.dumps({"stopped_at": stamp(), "old_supervisor_pid": 3669,
                                "old_child_pid": 6943, "old_checkpoint": old}, indent=1) + "\n")
                try:
                    os.kill(3669, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(6)
                try:
                    os.kill(6943, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                (WS / "results/parity/gpu_clean").mkdir(parents=True, exist_ok=True)
                cmd = [PY, "-u", "scripts/gpu_supervisor.py",
                       "--out", "results/parity/reader_xlarge_4k_clean",
                       "--log-dir", "results/parity/gpu_clean",
                       "--start-segment-index", "1",
                       "--args-file", "results/parity/reader_xlarge_4k_clean.args.json"]
                p = subprocess.Popen(cmd, cwd=str(WS), stdin=subprocess.DEVNULL,
                                     stdout=(WS / "results/parity/gpu_clean/supervisor.log").open("a"),
                                     stderr=subprocess.STDOUT, start_new_session=True)
                rec["reader"] = {"supervisor_pid": p.pid, "started_at": stamp(),
                                 "stream_sha256": audit["new_stream_sha256"],
                                 "stream": "data/tokens/bpe_6200000000.clean.u16"}
                rec["status"] = "reader_started_waiting_for_clean_packs"
                write(rec)
                print("reader supervisor", p.pid, flush=True)
        packs = load(WS / "results/parity/selector_pack_clean_record.json")
        if packs and not rec["selector"]:
            rec["clean_packs"] = {k: packs.get(k) for k in
                                  ("clean_packs", "excluded_packs", "excluded_source_docs",
                                   "manifest_sha256", "gate", "first_hit")}
            if packs.get("gate") != "PASS":
                rec["status"] = "clean_packs_failed"
                write(rec)
                print("clean pack build failed; not starting selector", flush=True)
                return 3
            out_dir = WS / "results/parity/selector_clean"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [PY, "-u", "scripts/selector_supervisor.py",
                   "--packs", "results/parity/long_context_train_31k_clean",
                   "--out", "results/parity/selector_clean/selector.pt",
                   "--log", "results/parity/selector_clean/selector_rows.jsonl",
                   "--max-steps", "31000"]
            p = subprocess.Popen(cmd, cwd=str(WS), stdin=subprocess.DEVNULL,
                                 stdout=(out_dir / "selector_supervisor.log").open("a"),
                                 stderr=subprocess.STDOUT, start_new_session=True)
            rec["selector"] = {"supervisor_pid": p.pid, "started_at": stamp(),
                               "packs": "results/parity/long_context_train_31k_clean"}
            rec["status"] = "both_started"
            write(rec)
            print("selector supervisor", p.pid, flush=True)
        if rec["reader"] and rec["selector"]:
            rec["status"] = "both_started"
            write(rec)
            print("both supervisors launched", flush=True)
            return 0
        time.sleep(10)
    rec["status"] = "timeout"
    write(rec)
    print("coordinator timeout", flush=True)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
