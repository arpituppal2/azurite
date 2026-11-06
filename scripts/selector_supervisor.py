#!/usr/bin/env python3
"""Detached supervisor for the clean-packs selector trainer."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Supervisor:
    def __init__(self, args) -> None:
        self.args = args
        self.out = (PROJECT / args.out).resolve()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out.parent / "selector_supervisor_state.json"
        self.history_path = self.out.parent / "selector_supervisor.jsonl"
        self.pid_path = self.out.parent / "selector_supervisor.pid"
        self.child = None

    def state(self, **kw) -> None:
        rec = {"supervisor_pid": os.getpid(), "updated": stamp(),
               "out": str(self.out), **kw}
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        tmp.replace(self.state_path)

    def read_step(self) -> int:
        try:
            return int(json.loads((self.out.parent / "selector_state.json").read_text()).get("step", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            return 0

    def entry(self, rec: dict) -> None:
        with self.history_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def run(self) -> int:
        self.pid_path.write_text(str(os.getpid()) + "\n")
        step = self.read_step()
        self.state(status="starting", step=step, attempts=0, restarts=0)
        attempts = 0
        no_progress = 0
        previous = step
        while True:
            attempts += 1
            before = self.read_step()
            cmd = [
                sys.executable, "-u", "scripts/train_selector_clean.py",
                "--packs", self.args.packs, "--out", str(self.out),
                "--log", self.args.log, "--max-steps", str(self.args.max_steps),
                "--save-every", str(self.args.save_every), "--resume",
            ]
            log = (self.out.parent / "selector_run.log").open("a")
            log.write(f"\n=== selector supervisor attempt {attempts} {stamp()} ===\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=str(PROJECT), stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            self.child = proc
            log.close()
            self.state(status="running", pid=proc.pid, step=before,
                       attempts=attempts, restarts=attempts - 1)
            rc = proc.wait()
            after = self.read_step()
            done = after >= self.args.max_steps
            rec = {"attempt": attempts, "pid": proc.pid, "started": stamp(),
                   "rc": rc, "step_before": before, "step_after": after,
                   "done": done, "gave_up": False}
            self.entry(rec)
            if done:
                self.state(status="complete", pid=None, step=after,
                           attempts=attempts, restarts=attempts - 1, last=rec)
                print(f"[selector-sup] complete at step {after}", flush=True)
                return 0
            if after > previous:
                no_progress = 0
                previous = after
            else:
                no_progress += 1
            if no_progress >= self.args.max_no_progress:
                rec["gave_up"] = True
                self.entry(rec)
                self.state(status="gave_up_no_progress", pid=None, step=after,
                           attempts=attempts, restarts=attempts - 1, last=rec)
                print(f"[selector-sup] giving up at step {after}", flush=True)
                return 2
            backoff = min(60.0, 5.0 * 2 ** max(0, no_progress))
            self.state(status="backoff", pid=None, step=after, backoff_s=backoff,
                       attempts=attempts, restarts=attempts - 1, last=rec)
            print(f"[selector-sup] attempt {attempts} rc {rc} step {before}->{after}; "
                  f"relaunch in {backoff:.0f}s", flush=True)
            time.sleep(backoff)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--out", default="results/parity/selector_clean/selector.pt")
    ap.add_argument("--log", default="results/parity/selector_clean/selector_rows.jsonl")
    ap.add_argument("--max-steps", type=int, default=31000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-no-progress", type=int, default=4)
    args = ap.parse_args()
    return Supervisor(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
