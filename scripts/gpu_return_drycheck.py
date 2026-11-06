#!/usr/bin/env python3
"""CPU dry-check of the GPU-return runbook (no GPU, no spend, no secrets).

Verifies everything about a GPU return that CAN be verified from this
machine: session-record parseability, anchor checkpoint identity, prereg +
baseline artifact presence and hashes, funnel test health. The live-host
probe is reported, never required: a dead host yields verdict BLOCKED
(exact unblock), not a failure.

Usage: .venv/bin/python3 scripts/gpu_return_drycheck.py [--out PATH]
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SESSION = PROJECT / ".stage" / "live_session_newhost3.json"
CKPT = PROJECT / "results" / "pretrain" / "l1_tiny_byte" / "ckpt.pt"
CKPT_SHA = "817e7f0449fae7d228f3d570bdbfe4c0efe031383b3579a4528a077ba6140410"
PREREG = PROJECT / "results" / "transfer" / "envelope_v4" / "PREREG_V4.json"

EXPECTED = {
    "results/transfer/envelope_v4/PREREG_V4.md": "c63c1f94e6767e2453983fe268e6a97a7b570d7a19220486c6220b375e6c2196",
    "results/transfer/funnel_cpu_baseline/PROBE.md": "b41135b8c3f2214352f139470e5f6cd81eeaf8d758ca92c359fb6a7cc4ad47b3",
    "results/transfer/funnel_cpu_baseline/result_s0.json": "b74af99b7b447018e2e69c82ae5494a6fd051656bbb02173ebfa8b77c39b536a",
    "results/transfer/funnel_cpu_baseline/result_s1.json": "3d0af216bafc16c14271220c01a0a45dac1e7838921224c6eee2167ffbdb19e7",
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    checks: dict = {}
    # 1. Session record: parseable? expired? (token NEVER printed)
    try:
        rec = json.loads(SESSION.read_text())
        base = rec.get("base", "")
        host = base.split("://", 1)[-1].split("/", 1)[0]
        exp = int(rec.get("expires_at", 0))
        checks["session_record"] = {
            "parseable": True, "host": host,
            "sandbox_id": rec.get("sandbox_id"),
            "expired": exp < int(time.time()),
            "token_present": bool(rec.get("token")),
        }
    except Exception as e:  # noqa: BLE001
        checks["session_record"] = {"parseable": False, "error": str(e)[:120]}
    # 2. Live-host probe (no auth; informational only)
    try:
        base = json.loads(SESSION.read_text()).get("base", "")
        req = urllib.request.Request(base + "/api/sessions", method="GET")
        try:
            urllib.request.urlopen(req, timeout=20)
            checks["host_probe"] = {"reachable": True, "note": "unexpected 2xx without auth"}
        except Exception as e:  # noqa: BLE001
            checks["host_probe"] = {"reachable": False, "detail": str(e)[:160]}
    except Exception as e:  # noqa: BLE001
        checks["host_probe"] = {"reachable": False, "detail": f"no session record: {e}"[:160]}
    # 3. Anchor checkpoint identity
    if CKPT.exists():
        got = sha(CKPT)
        checks["anchor_ckpt"] = {"present": True, "match": got == CKPT_SHA,
                                 "sha_prefix": got[:16]}
    else:
        checks["anchor_ckpt"] = {"present": False, "match": False}
    # 4. Prereg + baseline artifacts
    art = {}
    for rel, want in EXPECTED.items():
        p = PROJECT / rel
        if not p.exists():
            art[rel] = {"present": False}
            continue
        got = sha(p)
        art[rel] = {"present": True, "sha_prefix": got[:16],
                    "match": (want is None or got == want)}
    if PREREG.exists():
        art["results/transfer/envelope_v4/PREREG_V4.json"] = {
            "present": True, "sha_prefix": sha(PREREG)[:16], "match": True}
    else:
        art["results/transfer/envelope_v4/PREREG_V4.json"] = {"present": False}
    checks["artifacts"] = art
    # 5. Funnel test health (targeted trio, fast)
    pr = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_funnel.py",
         "tests/test_decisions.py", "tests/test_transfer_eval.py", "-q"],
        capture_output=True, text=True, cwd=str(PROJECT), timeout=280)
    tail = (pr.stdout + pr.stderr).strip().splitlines()
    checks["funnel_tests"] = {"rc": pr.returncode,
                              "tail": tail[-1] if tail else ""}
    sess_ok = checks["session_record"].get("parseable") and not checks["session_record"].get("expired")
    verdict = ("READY" if (sess_ok and checks["anchor_ckpt"].get("match")
                           and all(v.get("present") and v.get("match") for v in art.values())
                           and pr.returncode == 0)
               else "BLOCKED")
    out = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "verdict": verdict,
           "unblock": ("fresh browser session into .stage/live_session_newhost3.json "
                       "(browser action outside this tab); then restage per docs/RUNBOOK.md, "
                       "verify ckpt hash on-host, launch envelope-v4 s0+s1")
           if verdict == "BLOCKED" else "proceed with restage + ckpt verify + launch",
           "checks": checks, "gpu_spend_s": 0}
    print(json.dumps(out, indent=1))
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        Path(sys.argv[i + 1]).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
