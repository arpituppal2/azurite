"""The installed surface: resources, checkpoint identity and the CLI parser."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from azurite import bundle as bundle_module
from azurite import checkpoints
from azurite.cli import build_parser, _sample


def test_bundle_has_the_measured_task_and_support():
    bundle = bundle_module.load_bundle()
    assert bundle.task == "banking77"
    assert len(bundle.labels) == 77
    assert len(set(bundle.labels)) == 77
    assert len(bundle.support) == 9_968
    for row in bundle.support:
        assert row["label"] in bundle.labels
        assert row["message"].strip()


def test_provenance_hash_matches_the_packaged_bundle():
    provenance = bundle_module.load_provenance()
    raw = bundle_module._resource_bytes("banking77.json.gz")
    assert provenance["bundle"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert provenance["bundle"]["support_rows"] == 9_968
    assert provenance["sources"]["support"]["removed_rows"] == 25
    assert provenance["sources"]["support"]["residue_rows"] == 0
    assert gzip.decompress(raw).startswith(b"{")


def test_registry_points_at_the_measured_checkpoints():
    registry = bundle_module.load_registry()
    assert registry.default == "query-native"
    native = registry.get("query-native")
    assert native.sha256 == (
        "7992c97362b3c6b19ad39cf94792218ebb9c5508a8d3f3969f3526f8d15cb723"
    )
    assert native.measured["accuracy"] == pytest.approx(0.8905844155844156)
    assert native.measured["correct"] == 2_743
    assert native.measured["bank"] == "decontaminated"
    assert native.measured["support_rows"] == 9_968
    assert native.frozen["accuracy"] == pytest.approx(0.8902597402597403)
    assert native.frozen["correct"] == 2_742
    assert native.frozen["contamination"]["support_messages"] == 25
    assert native.frozen["contamination"]["affected_rows"] == 255
    assert native.max_seq == 1_024
    assert registry.get("7992c97362b3").id == "query-native"
    long = registry.get("query-joint-5120")
    assert long.measured["accuracy"] == pytest.approx(0.8711038961038962)
    assert long.max_seq == 5_120


def test_bundled_model_card_is_the_repository_card():
    root = Path(__file__).resolve().parents[1] / "MODEL_CARD.md"
    assert bundle_module._resource_bytes("MODEL_CARD.md") == root.read_bytes()


def test_registry_refuses_an_unknown_checkpoint():
    with pytest.raises(bundle_module.BundleError):
        bundle_module.load_registry().get("not-a-checkpoint")


def test_missing_checkpoint_names_the_expected_hash(tmp_path, monkeypatch):
    monkeypatch.delenv("AZURITE_CHECKPOINT_DIR", raising=False)
    monkeypatch.setenv("AZURITE_HOME", str(tmp_path / "home"))
    with pytest.raises(checkpoints.CheckpointError) as excinfo:
        checkpoints.resolve("query-native")
    message = str(excinfo.value)
    assert "7992c973" in message
    assert "azurite checkpoint add" in message


def test_wrong_file_at_the_expected_name_is_refused(tmp_path, monkeypatch):
    native = bundle_module.load_registry().get("query-native")
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / f"{native.sha256}.pt").write_bytes(b"not the weights")
    monkeypatch.setenv("AZURITE_CHECKPOINT_DIR", str(directory))
    monkeypatch.delenv("AZURITE_HOME", raising=False)
    with pytest.raises(checkpoints.CheckpointError, match="has SHA-256"):
        checkpoints.resolve("query-native")


def test_add_refuses_a_file_that_is_not_the_published_checkpoint(tmp_path):
    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"not a checkpoint")
    with pytest.raises(checkpoints.CheckpointError, match="not a published checkpoint"):
        checkpoints.add(fake, home=tmp_path / "home")


def test_cli_parser_exposes_the_product_surface():
    parser = build_parser()
    args = parser.parse_args(["predict", "my card was swallowed"])
    assert args.command == "predict"
    assert args.top == 5
    args = parser.parse_args(
        ["verify", "--records", "r.jsonl", "--inputs", "i.json", "--n", "2"]
    )
    assert args.n == 2
    assert args.device == "cpu"


def test_verify_sampler_is_even_and_deterministic():
    records = [{"id": f"test-{i:05d}"} for i in range(100)]
    picked = _sample(records, 5, None)
    assert [row["id"] for row in picked] == [
        "test-00000",
        "test-00025",
        "test-00050",
        "test-00074",
        "test-00099",
    ]
    assert _sample(records, 2, "test-00007,test-00002") == [
        {"id": "test-00007"},
        {"id": "test-00002"},
    ]
