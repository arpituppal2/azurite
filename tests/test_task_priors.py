"""A task that cannot be loaded must still appear in the priors table.

The first version of this table silently dropped tasks whose split would not
load, which is indistinguishable from a task that does not exist — the exact
failure mode the contamination audit's "skipped" counter exists to avoid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

import task_priors  # noqa: E402


def test_unloadable_split_is_recorded_not_dropped(tmp_path, monkeypatch) -> None:
    out = tmp_path / "PRIORS.json"
    monkeypatch.setattr(task_priors, "OUT", out)
    monkeypatch.setattr(task_priors, "load_examples", lambda *a, **k: [])
    monkeypatch.setattr(task_priors, "candidate_bank", lambda spec: [1, 2, 3])
    monkeypatch.setattr(
        task_priors, "TASKS", {"ghost_task": type("S", (), {"name": "ghost_task"})()}
    )
    monkeypatch.setattr(sys, "argv", ["task_priors.py"])

    assert task_priors.main() == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert "ghost_task" in written, "the task vanished from the table instead of failing loudly"
    assert written["ghost_task"]["error"] == "no test split"


def test_priors_are_bounded_probabilities(tmp_path, monkeypatch) -> None:
    """Majority accuracy is a probability; a value outside [0, 1] would mean the
    gold indices and the label space had drifted apart."""
    out = tmp_path / "PRIORS.json"
    monkeypatch.setattr(task_priors, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["task_priors.py"])
    assert task_priors.main() == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written, "no tasks measured"
    for name, row in written.items():
        if "error" in row:
            continue
        assert 0.0 <= row["majority_accuracy"] <= 1.0, (name, row)
        assert row["test_rows"] > 0, (name, row)
        # `uniform_accuracy` is 1/K rounded to 4 decimals, so the product is only
        # 1 to within 0.5e-4 * K — which at K=151 is 0.008, not 1e-6.
        assert abs(row["uniform_accuracy"] * row["k"] - 1.0) < 1e-2, (name, row)
