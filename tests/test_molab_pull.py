"""The puller must not lose a checkpoint just because no segment marker landed.

The 113M run wrote ckpt.pt at every segment boundary, but the old puller only
pulled weights after a segment/summary/result marker. The sandbox then died with
no marker and the weights were unrecoverable. checkpoint.json is now sufficient.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

import molab_pull  # noqa: E402


def test_checkpoint_json_alone_licenses_a_verified_pull(
    tmp_path: Path, monkeypatch
) -> None:
    dest = tmp_path / "mirror"
    monkeypatch.setattr(molab_pull, "DEST", dest)
    monkeypatch.setattr(molab_pull, "LOCAL_STATE", dest / ".pull_state.json")

    payload = b"weights" * 3
    digest = "0" * 64
    entries = {
        "results/pretrain/run/checkpoint.json": {
            "path": "results/pretrain/run/checkpoint.json", "size": 200, "mtime": 10,
        },
        "results/pretrain/run/ckpt.pt": {
            "path": "results/pretrain/run/ckpt.pt", "size": len(payload), "mtime": 10,
        },
    }
    monkeypatch.setattr(molab_pull, "manifest", lambda: entries)

    calls: list[tuple[str, str | None]] = []

    def fake_pull(rel: str, entry: dict, expected_sha256: str | None = None) -> Path:
        calls.append((rel, expected_sha256))
        local = dest / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("checkpoint.json"):
            local.write_text(json.dumps({
                "status": "running",
                "step": 7,
                "tokens_seen": 896,
                "checkpoint": {"path": "ckpt.pt", "bytes": len(payload),
                               "sha256": digest},
            }))
        else:
            local.write_bytes(payload)
        return local

    monkeypatch.setattr(molab_pull, "pull_file", fake_pull)

    assert molab_pull.main() == 0
    assert ("results/pretrain/run/ckpt.pt", digest) in calls
    verified = json.loads((dest / "MIRROR_VERIFIED.json").read_text())
    record = verified["checkpoints"]["results/pretrain/run/ckpt.pt"]
    assert record["step"] == 7
    assert record["tokens_seen"] == 896
    assert record["status"] == "running"
    assert (dest / "results/pretrain/run/ckpt.pt").read_bytes() == payload
