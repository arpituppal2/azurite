#!/usr/bin/env python3
"""Detached supervisor for the frozen Azurite reader pretrain.

Runs ON the GPU host. Owns the frozen pretrain command and relaunches it with
`--resume` after ANY exit -- segment timeout, OOM, crash, or signal -- with
bounded exponential backoff. Gives up only on a non-retryable error (a resume
that cannot proceed: missing/mismatched checkpoint, changed geometry, existing
run state) or after `--max-no-progress` consecutive attempts that fail to move
the checkpoint step.

Every transition is appended to <log-dir>/supervisor.jsonl (one JSON object per
attempt: reason, resume step, wall time, gap) and mirrored into
<log-dir>/supervisor_state.json, so the restart history survives the supervisor.

    setsid nohup python3 scripts/gpu_supervisor.py --out ... \
        --log-dir results/parity/gpu --start-segment-index 1 \
        >> results/parity/gpu/supervisor.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parent

FROZEN = [
    "-u", "scripts/pretrain.py",
    "--rung", "xlarge", "--arm", "bpe", "--seq", "4096",
    "--tokens", "6.2e9", "--batch", "2", "--lr", "3e-4", "--amp", "bf16",
    "--ckpt-every", "500",
]


def frozen(segment_seconds: float, base: list[str] | None = None) -> list[str]:
    """The pretrain command; only the segment cap is chosen per attempt.

    `base` exists so a scratch run can use the same control flow with a smaller
    model. Production passes nothing and gets the frozen reader command.
    """
    return [*(base or FROZEN), "--max-seconds", str(segment_seconds)]

# A resume that cannot proceed will fail identically on every retry. Retrying it
# just burns the wall clock while looking healthy.
FATAL_PATTERNS = (
    "refusing to start a fresh run",
    "refusing to restart from zero",
    "does not match --tokens",
    "do not match this invocation",
    "is not a loadable checkpoint",
    "--eval-only: no checkpoint",
)
OOM_PATTERNS = ("out of memory", "MemoryError", "CUDA error: out of memory")


def now() -> float:
    return time.time()


def stamp(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now() if ts is None else ts))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = (PROJECT / args.out).resolve()
        self.ckpt_json = self.out / "checkpoint.json"
        self.log_dir = (PROJECT / args.log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = self.log_dir / "supervisor.jsonl"
        self.state_path = self.log_dir / "supervisor_state.json"
        self.pid_path = self.log_dir / "supervisor.pid"
        self.train_log = self.out.parent / "pretrain.log"
        self.child: subprocess.Popen | None = None

    # -- records -----------------------------------------------------------
    def checkpoint(self) -> dict:
        try:
            return json.loads(self.ckpt_json.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def state(self, **kw) -> None:
        rec = {
            "supervisor_pid": os.getpid(),
            "updated": stamp(),
            "out": str(self.out.relative_to(PROJECT)),
            **kw,
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        tmp.replace(self.state_path)

    def entry(self, rec: dict) -> None:
        with self.history.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    # -- child -------------------------------------------------------------
    def log_tail(self, before: int) -> str:
        try:
            with self.train_log.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 65536))
                text = fh.read().decode("utf-8", "replace")
        except OSError:
            return ""
        return text[before:] if before > 0 else text

    def classify(self, rc: int, tail: str) -> tuple[str, bool]:
        """(reason, fatal)."""
        text = tail[-8000:]
        for pat in FATAL_PATTERNS:
            if pat in text:
                return f"fatal resume: {pat}", True
        if any(p in text for p in OOM_PATTERNS):
            return "oom", False
        if rc == 0:
            return "exited 0", False
        if rc < 0:
            return f"signal {-rc}", False
        if rc > 128:
            return f"signal {rc - 128}", False
        return f"exit {rc}", False

    def spawn(self, segment: int) -> subprocess.Popen:
        base = (json.loads(pathlib.Path(self.args.args_file).read_text())
                if self.args.args_file else None)
        cmd = [sys.executable, *frozen(self.args.segment_seconds, base),
               "--mem-cap-gib", str(self.args.mem_cap_gib),
               "--segment-index", str(segment),
               "--resume", "--out", str(self.out)]
        fh = self.train_log.open("a", encoding="utf-8")
        fh.write(f"\n=== supervisor attempt {segment} {stamp()} {sys.executable} {' '.join(cmd[1:])} ===\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(PROJECT), stdin=subprocess.DEVNULL,
                                stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        fh.close()
        return proc

    def terminate(self) -> None:
        if self.child and self.child.poll() is None:
            try:
                self.child.terminate()
                self.child.wait(timeout=30)
            except Exception:
                self.child.kill()

    # -- main loop ---------------------------------------------------------
    def run(self) -> int:
        self.pid_path.write_text(str(os.getpid()) + "\n")
        me = self.checkpoint()
        self.state(status="starting", step=me.get("step"), tokens_seen=me.get("tokens_seen"),
                   attempts=0, restarts=0, total_up_s=0.0, total_gap_s=0.0)

        attempts = 0
        no_progress = 0
        total_up = 0.0
        total_gap = 0.0
        last_end: float | None = None
        segment = self.args.start_segment_index - 1
        previous_step = int(me.get("step") or 0)

        while True:
            attempts += 1
            segment += 1
            before = self.checkpoint()
            step_before = int(before.get("step") or 0)
            tokens_before = int(before.get("tokens_seen") or 0)
            gap = 0.0 if last_end is None else max(0.0, now() - last_end)
            total_gap += gap
            log_size = self.train_log.stat().st_size if self.train_log.exists() else 0

            proc = self.spawn(segment)
            self.child = proc
            started = now()
            self.state(status="running", pid=proc.pid, attempt=attempts,
                       segment_index=segment, step=step_before, tokens_seen=tokens_before,
                       attempts_total=attempts, restarts=attempts - 1,
                       total_up_s=round(total_up, 1), total_gap_s=round(total_gap, 1),
                       gap_s=round(gap, 1))
            print(f"[sup] attempt {attempts} seg {segment} pid {proc.pid} "
                  f"resume_step {step_before} gap {gap:.1f}s", flush=True)

            rc = proc.wait()
            ended = now()
            wall = ended - started
            total_up += wall
            last_end = ended
            after = self.checkpoint()
            step_after = int(after.get("step") or 0)
            tokens_after = int(after.get("tokens_seen") or 0)
            reason, fatal = self.classify(rc, self.log_tail(log_size))
            fraction = float(after.get("fraction") or 0.0)
            done = fraction >= 1.0 and str(after.get("status")) == "complete"

            rec = {
                "attempt": attempts, "segment_index": segment, "pid": proc.pid,
                "started": stamp(started), "ended": stamp(ended),
                "wall_s": round(wall, 1), "rc": rc, "reason": reason,
                "resume_step": step_before, "resume_tokens": tokens_before,
                "end_step": step_after, "end_tokens": tokens_after,
                "fraction": fraction, "gap_s": round(gap, 1), "fatal": fatal,
                "loss_progress": step_after > previous_step,
            }
            self.entry(rec)

            if done:
                self.state(status="complete", pid=None, attempts_total=attempts,
                           restarts=attempts - 1, total_up_s=round(total_up, 1),
                           total_gap_s=round(total_gap, 1), step=step_after,
                           tokens_seen=tokens_after, last=rec)
                print(f"[sup] budget complete at step {step_after}", flush=True)
                return 0
            if fatal:
                self.state(status="gave_up_fatal", pid=None, attempts_total=attempts,
                           restarts=attempts - 1, total_up_s=round(total_up, 1),
                           total_gap_s=round(total_gap, 1), step=step_after,
                           tokens_seen=tokens_after, last=rec)
                print(f"[sup] fatal, giving up: {reason}", flush=True)
                return 2

            if step_after > previous_step:
                no_progress = 0
                previous_step = step_after
            else:
                no_progress += 1
            if no_progress >= self.args.max_no_progress:
                self.state(status="gave_up_no_progress", pid=None, attempts_total=attempts,
                           restarts=attempts - 1, total_up_s=round(total_up, 1),
                           total_gap_s=round(total_gap, 1), step=step_after,
                           tokens_seen=tokens_after, last=rec)
                print(f"[sup] {no_progress} attempts without step progress; giving up", flush=True)
                return 3

            backoff = 0.0 if rc == 0 else min(self.args.backoff_max,
                                              self.args.backoff_base * 2 ** max(0, no_progress))
            self.state(status="backoff", pid=None, attempt=attempts, backoff_s=backoff,
                       attempts_total=attempts, restarts=attempts - 1,
                       total_up_s=round(total_up, 1), total_gap_s=round(total_gap, 1),
                       step=step_after, tokens_seen=tokens_after, last=rec)
            print(f"[sup] attempt {attempts} rc {rc} ({reason}) step {step_before}->{step_after} "
                  f"wall {wall:.0f}s; relaunch in {backoff:.0f}s", flush=True)
            if backoff:
                time.sleep(backoff)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", default="results/parity/gpu")
    ap.add_argument("--start-segment-index", type=int, default=1)
    ap.add_argument("--mem-cap-gib", type=float, default=60.0)
    ap.add_argument("--segment-seconds", type=float, default=36000.0,
                    help="--max-seconds for one pretrain segment (default 10h)")
    ap.add_argument("--args-file", default="",
                    help="JSON list replacing the frozen argv (scratch runs only)")
    ap.add_argument("--max-no-progress", type=int, default=6)
    ap.add_argument("--backoff-base", type=float, default=5.0)
    ap.add_argument("--backoff-max", type=float, default=60.0)
    args = ap.parse_args()

    sup = Supervisor(args)

    def stop(signum, _frame):
        print(f"[sup] signal {signum}; stopping child", flush=True)
        sup.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return sup.run()


if __name__ == "__main__":
    raise SystemExit(main())
