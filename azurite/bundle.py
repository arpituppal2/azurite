"""What the installed package carries, independent of the repository.

An installed wheel has no `data/raw`, no `results/`, and no Hugging Face cache.
Everything needed to turn a customer message into a Banking77 intent therefore
ships inside the package:

    resources/banking77.json.gz   77 labels in candidate order, 9,968 support rows
    resources/checkpoints.json    checkpoint ids, content hashes, measured scores
    resources/provenance.json     the source hashes the two files were built from

The support bank is decontaminated: the 25 evaluation texts found in the
9,993-row source split are removed at build time, and the build refuses to emit
a bundle in which a normalised evaluation text survives.

The checkpoint weights are the one thing that does not fit; `azurite.checkpoints`
resolves those by SHA-256 from a local cache. This module never imports torch, so
`azurite info` and `azurite checkpoint` stay fast and can run before a model is
installed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Mapping


class BundleError(RuntimeError):
    """The packaged data is missing or internally inconsistent."""


def _resource_bytes(name: str) -> bytes:
    path = files("azurite").joinpath("resources", name)
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise BundleError(
            f"packaged resource {name!r} is missing; reinstall the wheel"
        ) from exc


@dataclass(frozen=True)
class Bundle:
    task: str
    labels: tuple[str, ...]
    support: tuple[Mapping[str, object], ...]
    schema: int = 1

    def label_index(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise BundleError(f"{label!r} is not one of the 77 Banking77 labels") from exc


@dataclass(frozen=True)
class Checkpoint:
    id: str
    sha256: str
    task: str
    head: str
    max_seq: int
    context_budget: int
    examples: int
    measured: Mapping[str, object]
    source_path: str
    frozen: Mapping[str, object] | None = None

    def require_task(self, task: str) -> None:
        if self.task != task:
            raise BundleError(
                f"checkpoint {self.id!r} is for {self.task!r}, not {task!r}"
            )


@dataclass(frozen=True)
class Registry:
    default: str
    checkpoints: tuple[Checkpoint, ...]
    schema: int = 1

    def get(self, ident: str | None = None) -> Checkpoint:
        """Resolve an id, a full content hash, or an unambiguous hash prefix."""
        wanted = (ident or self.default).strip().lower()
        exact = [c for c in self.checkpoints if c.id == wanted or c.sha256 == wanted]
        if exact:
            return exact[0]
        prefix = [c for c in self.checkpoints if c.sha256.startswith(wanted)]
        if len(prefix) == 1 and len(wanted) >= 12:
            return prefix[0]
        if len(prefix) > 1 and len(wanted) >= 12:
            raise BundleError(f"checkpoint hash prefix {wanted!r} is ambiguous")
        known = ", ".join(c.id for c in self.checkpoints)
        raise BundleError(f"unknown checkpoint {ident!r}; known ids: {known}")


@lru_cache(maxsize=1)
def load_bundle() -> Bundle:
    compressed = _resource_bytes("banking77.json.gz")
    expected = load_provenance()["bundle"]["sha256"]
    actual = hashlib.sha256(compressed).hexdigest()
    if actual != expected:
        raise BundleError(
            f"packaged bundle SHA-256 is {actual}, provenance requires {expected}"
        )
    try:
        data = json.loads(gzip.decompress(compressed))
    except (gzip.BadGzipFile, json.JSONDecodeError) as exc:
        raise BundleError("packaged Banking77 bundle is not valid gzip JSON") from exc
    if data.get("schema") != 1:
        raise BundleError(f"unsupported bundle schema {data.get('schema')!r}")
    labels = tuple(str(label) for label in data["labels"])
    if len(labels) != 77:
        raise BundleError(f"bundle has {len(labels)} labels, expected 77")
    if len(set(labels)) != len(labels):
        raise BundleError("bundle labels are not unique")
    support = tuple(data["support"])
    if not support:
        raise BundleError("bundle has no support rows")
    if len(support) != int(load_provenance()["bundle"]["support_rows"]):
        raise BundleError("bundle support count does not match its provenance record")
    for row in support:
        if row.get("label") not in labels:
            raise BundleError(
                f"support row {row.get('id')!r} has unknown label {row.get('label')!r}"
            )
        if not str(row.get("message", "")).strip():
            raise BundleError(f"support row {row.get('id')!r} has an empty message")
    return Bundle(task=str(data["task"]), labels=labels, support=support)


@lru_cache(maxsize=1)
def load_registry() -> Registry:
    try:
        data = json.loads(_resource_bytes("checkpoints.json"))
    except json.JSONDecodeError as exc:
        raise BundleError("packaged checkpoint registry is not valid JSON") from exc
    if data.get("schema") != 1:
        raise BundleError(f"unsupported registry schema {data.get('schema')!r}")
    entries = []
    seen_ids: set[str] = set()
    for raw in data["checkpoints"]:
        digest = str(raw["sha256"]).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise BundleError(f"checkpoint {raw.get('id')!r} has an invalid SHA-256")
        if raw["id"] in seen_ids:
            raise BundleError(f"duplicate checkpoint id {raw['id']!r}")
        seen_ids.add(raw["id"])
        entries.append(
            Checkpoint(
                id=str(raw["id"]),
                sha256=digest,
                task=str(raw["task"]),
                head=str(raw["head"]),
                max_seq=int(raw["max_seq"]),
                context_budget=int(raw["context_budget"]),
                examples=int(raw["examples"]),
                measured=raw["measured"],
                source_path=str(raw["source_path"]),
                frozen=raw.get("frozen"),
            )
        )
    registry = Registry(default=str(data["default"]), checkpoints=tuple(entries))
    registry.get(registry.default)
    return registry


@lru_cache(maxsize=1)
def load_provenance() -> Mapping[str, object]:
    try:
        return json.loads(_resource_bytes("provenance.json"))
    except json.JSONDecodeError as exc:
        raise BundleError("packaged provenance record is not valid JSON") from exc
