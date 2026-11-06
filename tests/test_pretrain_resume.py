"""Regression tests for the checkpoint/segment path.

These are the failure modes that made the 113M rung unrecoverable:
a partial run with a log but no weights, a checkpoint written non-atomically,
and a per-segment driver whose only durable record was overwritten.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

import pretrain  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stream(path: Path, tokens: int = 80_000) -> Path:
    """A byte-alphabet stream with no specials except document boundaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = (np.arange(tokens, dtype=np.uint32) % 254 + 4).astype(np.uint16)
    path.write_bytes(ids.tobytes())
    return path


def _run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "pretrain.py"), *args],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_atomic_save_keeps_previous_checkpoint_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ckpt.pt"
    torch.save({"step": 1, "tokens_seen": 8}, path)
    before = _sha256(path)

    def fail(obj, handle):  # noqa: ANN001
        Path(handle).write_bytes(b"half a checkpoint")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError):
        pretrain._atomic_torch_save({"step": 2, "tokens_seen": 16}, path)

    assert _sha256(path) == before
    assert torch.load(path, weights_only=True)["step"] == 1
    assert not path.with_suffix(".pt.tmp").exists(), "a failed save left a temp file"


def test_truncated_segment_leaves_a_resumable_checkpoint(tmp_path: Path) -> None:
    stream = _stream(tmp_path / "stream.u16")
    out = tmp_path / "large"
    proc = _run([
        "--rung", "tiny", "--arm", "byte", "--stream", str(stream),
        "--tokens", "1000000", "--seq", "32", "--batch", "4",
        "--out", str(out), "--max-seconds", "0.001",
        "--ckpt-every", "1", "--log-every", "1", "--mem-cap-gib", "2.0",
        "--segment-index", "0",
    ])
    assert proc.returncode == 0, proc.stderr[-2000:] + proc.stdout[-2000:]

    assert (out / "ckpt.pt").exists()
    assert (out / "checkpoint.json").exists()
    assert (out / "segment.json").exists()
    assert not (out / "summary.json").exists(), "a partial rung must not be graded complete"

    ckpt = json.loads((out / "checkpoint.json").read_text())
    seg = json.loads((out / "segment.json").read_text())
    assert seg["status"] == "truncated"
    assert seg["complete"] is False
    assert ckpt["segment_index"] == 0
    assert ckpt["checkpoint"]["sha256"] == _sha256(out / "ckpt.pt")
    assert seg["checkpoint"]["sha256"] == ckpt["checkpoint"]["sha256"]
    assert ckpt["tokens_seen"] == seg["tokens_seen"]

    # The rolling ckpt.pt is overwritten by the next segment. The boundary copy
    # is what makes a segment checkpoint survive that overwrite.
    archive = out / "checkpoints" / f"seg_000_step_{ckpt['step']}"
    assert (archive / "ckpt.pt").exists()
    assert _sha256(archive / "ckpt.pt") == ckpt["checkpoint"]["sha256"]
    archived = json.loads((archive / "checkpoint.json").read_text())
    assert archived["segment_index"] == 0
    assert archived["archive_path"].endswith("ckpt.pt")


def test_resume_continues_from_the_checkpoint_to_the_same_budget(tmp_path: Path) -> None:
    stream = _stream(tmp_path / "stream.u16")
    out = tmp_path / "run"
    base = [
        "--rung", "tiny", "--arm", "byte", "--stream", str(stream),
        "--tokens", "1024", "--seq", "32", "--batch", "4",
        "--out", str(out), "--ckpt-every", "1", "--log-every", "1",
        "--mem-cap-gib", "2.0",
    ]
    first = _run([*base, "--max-seconds", "0.001", "--segment-index", "0"])
    assert first.returncode == 0, first.stderr[-2000:]
    assert not (out / "summary.json").exists()

    before = json.loads((out / "checkpoint.json").read_text())
    second = _run([*base, "--resume", "--max-seconds", "0", "--segment-index", "1"])
    assert second.returncode == 0, second.stderr[-4000:] + second.stdout[-2000:]

    summary = json.loads((out / "summary.json").read_text())
    assert summary["tokens_seen"] >= 1024
    assert summary["steps"] >= 8
    assert summary["resumed_from"] is not None
    assert summary["resumed_from"]["tokens_seen"] == before["tokens_seen"]
    assert (out / "segment.json").read_text()
    assert json.loads((out / "segment.json").read_text())["status"] == "complete"


def test_resume_without_weights_refuses_and_does_not_erase_the_log(tmp_path: Path) -> None:
    stream = _stream(tmp_path / "stream.u16")
    out = tmp_path / "run"
    first = _run([
        "--rung", "tiny", "--arm", "byte", "--stream", str(stream),
        "--tokens", "1000000", "--seq", "32", "--batch", "4",
        "--out", str(out), "--max-seconds", "0.001",
        "--ckpt-every", "1", "--log-every", "1", "--mem-cap-gib", "2.0",
    ])
    assert first.returncode == 0, first.stderr[-2000:]
    log = out / "train.jsonl"
    before = _sha256(log)
    (out / "ckpt.pt").unlink()

    second = _run([
        "--rung", "tiny", "--arm", "byte", "--stream", str(stream),
        "--tokens", "1000000", "--seq", "32", "--batch", "4",
        "--out", str(out), "--resume", "--max-seconds", "0",
        "--ckpt-every", "1", "--log-every", "1", "--mem-cap-gib", "2.0",
    ])
    assert second.returncode != 0
    assert "refusing to restart from zero" in second.stderr
    assert _sha256(log) == before, "a refused resume must not append a fresh run to the log"


def test_archive_copy_is_stable_after_the_rolling_checkpoint_is_replaced(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "ckpt.pt").write_bytes(b"weights-at-step-10")
    record = {
        "step": 10,
        "segment_index": 0,
        "checkpoint": {
            "path": "ckpt.pt",
            "bytes": len(b"weights-at-step-10"),
            "sha256": hashlib.sha256(b"weights-at-step-10").hexdigest(),
        },
    }
    pretrain._archive_segment_checkpoint(out, record, 0)
    archived = out / "checkpoints" / "seg_000_step_10" / "ckpt.pt"
    assert archived.read_bytes() == b"weights-at-step-10"

    # Simulate the next segment overwriting the rolling file.
    (out / "ckpt.pt").write_bytes(b"weights-at-step-20")
    assert archived.read_bytes() == b"weights-at-step-10"
    again = json.loads((archived.parent / "checkpoint.json").read_text())
    assert again["checkpoint"]["sha256"] == record["checkpoint"]["sha256"]


def test_fresh_run_refuses_to_overwrite_existing_state(tmp_path: Path) -> None:
    stream = _stream(tmp_path / "stream.u16")
    out = tmp_path / "run"
    first = _run([
        "--rung", "tiny", "--arm", "byte", "--stream", str(stream),
        "--tokens", "1000000", "--seq", "32", "--batch", "4",
        "--out", str(out), "--max-seconds", "0.001",
        "--ckpt-every", "1", "--log-every", "1", "--mem-cap-gib", "2.0",
    ])
    assert first.returncode == 0, first.stderr[-2000:]
    log = out / "train.jsonl"
    before = _sha256(log)

    second = _run([
        "--rung", "tiny", "--arm", "byte", "--stream", str(stream),
        "--tokens", "1000000", "--seq", "32", "--batch", "4",
        "--out", str(out), "--max-seconds", "0",
        "--ckpt-every", "1", "--log-every", "1", "--mem-cap-gib", "2.0",
    ])
    assert second.returncode != 0
    assert "refusing to start a fresh run" in second.stderr
    assert _sha256(log) == before, "the refused run overwrote the existing log"


def test_segment_driver_keeps_an_append_only_history(tmp_path: Path) -> None:
    stream = _stream(tmp_path / "stream.u16")
    out = tmp_path / "segments"
    proc = subprocess.run(
        [
            sys.executable, str(PROJECT / "scripts" / "train_segments.py"),
            "--rung", "tiny", "--arm", "byte", "--stream", str(stream),
            "--tokens", "8192", "--seq", "32", "--batch", "4",
            "--out-dir", str(out), "--segment-seconds", "0.02",
            "--max-segments", "1", "--ckpt-every", "1", "--log-every", "1",
            "--mem-cap-gib", "2.0",
        ],
        cwd=PROJECT, capture_output=True, text=True, timeout=180,
    )
    # One incomplete segment is expected: the driver reports that explicitly
    # rather than calling a short run complete.
    assert proc.returncode == 1, proc.stderr[-2000:] + proc.stdout[-2000:]
    history = out / "segments.jsonl"
    assert history.exists(), proc.stdout[-2000:]
    rows = [json.loads(line) for line in history.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["index"] == 0
    assert rows[0]["tokens_seen"] > 0
    assert (out / "checkpoint.json").exists()
    driver = json.loads((out / "driver.json").read_text())
    assert driver["status"] in {"segments-exhausted", "stalled"}
    assert driver["segments"], "the driver lost the segment record"
