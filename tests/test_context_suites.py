from __future__ import annotations

import json
from pathlib import Path

from azurite.decisions import TASKS


ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "research" / "suites"


def load(name: str) -> dict:
    return json.loads((SUITES / name).read_text())


def test_dev_suite_is_capped_and_frozen() -> None:
    dev = load("dev_suite.json")
    assert dev["status"] == "frozen"
    assert len(dev["tasks"]) == 5
    assert abs(sum(t["weight"] for t in dev["tasks"]) - 1.0) < 1e-6
    assert max(t["weight"] for t in dev["tasks"]) <= 0.25
    assert all(t["train_rows"] >= 2_499 for t in dev["tasks"])
    assert all(t["test_rows"] >= 748 for t in dev["tasks"])


def test_locked_suite_is_sealed_and_capped() -> None:
    locked = load("locked_suite.json")
    assert locked["status"] == "sealed"
    assert locked["rule"]["opened"] is False
    assert len(locked["tasks"]) == 12
    assert max(t["weight"] for t in locked["tasks"]) <= 0.25
    assert all(t["rows"] >= 493 for t in locked["tasks"])


def test_context_tasks_use_the_uniform_schema() -> None:
    for task in [
        "banking77_ctx",
        "clinc_oos_ctx",
        "boolq_ctx",
        "squad2_answerability_ctx",
        "contract_nli_ctx",
    ]:
        spec = TASKS[task]
        assert spec.train_rel == "train"
        assert spec.split_rel == "test"
        assert spec.text_fields[0] == "context"


def test_comparator_classes_admit_no_live_reproduction() -> None:
    registry = json.loads((ROOT / "research" / "benchmark_registry.json").read_text())
    classes = registry["comparator_classes"]
    assert classes["counts"]["reproduced_live"] == 0
    assert classes["counts"]["local_reconstruction"] == 1
    assert classes["counts"]["published_reference"] == 13
