#!/usr/bin/env python3
"""Prove the supervisor's rc=0 (segment timeout) path without a GPU.

The live run's crash path is proven end-to-end on the host (SIGKILL -> resume).
The 10-hour segment exit is rc=0, which takes a different branch: no backoff,
immediate relaunch, and the `done` guard must not fire on a truncated segment.
Waiting 10 hours to see it is not an option, so this drives the real Supervisor
class against a fake child that mimics pretrain.py's exit behaviour:

  segment 1: exit 0, status "truncated", step 900   -> must relaunch at once
  segment 2: exit 0, status "truncated", step 1800  -> must relaunch at once
  segment 3: exit 0, status "complete",  fraction 1 -> must stop, rc 0

No model, no data, no CUDA: only checkpoint.json / supervisor.jsonl writes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_supervisor import Supervisor  # noqa: E402


class FakeProc:
    seq = 0

    def __init__(self, owner: "FakeSupervisor") -> None:
        FakeProc.seq += 1
        self.pid = 900000 + FakeProc.seq
        self._owner = owner

    def wait(self) -> int:
        return self._owner.complete_segment()

    def poll(self):
        return 0

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class FakeSupervisor(Supervisor):
    """Supervisor whose child is a scripted pretrain exit."""

    def __init__(self, args: argparse.Namespace, script: list[dict]) -> None:
        super().__init__(args)
        self.script = script
        self.launches: list[float] = []

    def spawn(self, segment: int):  # type: ignore[override]
        self.launches.append(time.time())
        return FakeProc(self)

    def complete_segment(self) -> int:
        i = len(self.launches) - 1
        step = self.script[i]
        rec = {
            "schema": 2, "status": step["status"], "step": step["step"],
            "tokens_seen": step["step"] * 8192, "target_tokens": 6_200_000_000,
            "fraction": step["fraction"], "segment_index": i + 1, "checkpoint": {},
        }
        tmp = self.ckpt_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(self.ckpt_json)
        return 0


def main() -> int:
    # Must live under the project: Supervisor.state() records out relative to it.
    work = Path(__file__).resolve().parent.parent / ".stage" / "sup_relaunch_test"
    if work.exists():
        import shutil
        shutil.rmtree(work)
    out = work / "reader_xlarge_4k"
    out.mkdir(parents=True)
    (out / "checkpoint.json").write_text(json.dumps({"step": 0, "tokens_seen": 0}))
    (work / "pretrain.log").write_text("")

    script = [
        {"status": "truncated", "step": 900, "fraction": 0.0012},
        {"status": "truncated", "step": 1800, "fraction": 0.0024},
        {"status": "complete", "step": 2_700, "fraction": 1.0},
    ]
    args = argparse.Namespace(
        out=str(out), log_dir=str(work / "gpu"), start_segment_index=1,
        mem_cap_gib=60.0, segment_seconds=36000.0, max_no_progress=6,
        backoff_base=5.0, backoff_max=60.0,
    )
    sup = FakeSupervisor(args, script)
    rc = sup.run()

    hist = [json.loads(l) for l in (work / "gpu" / "supervisor.jsonl").read_text().splitlines() if l.strip()]
    gaps = [h["gap_s"] for h in hist]
    reasons = [h["reason"] for h in hist]
    ok = (
        rc == 0
        and len(hist) == 3
        and [h["rc"] for h in hist] == [0, 0, 0]
        and [h["end_step"] for h in hist] == [900, 1800, 2_700]
        and [h["resume_step"] for h in hist] == [0, 900, 1800]
        and max(gaps) == 0.0          # rc=0 must not back off
        and reasons == ["exited 0"] * 3
    )
    print(json.dumps({"rc": rc, "entries": len(hist), "gaps": gaps,
                      "resume_steps": [h["resume_step"] for h in hist],
                      "end_steps": [h["end_step"] for h in hist],
                      "reasons": reasons, "pass": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
