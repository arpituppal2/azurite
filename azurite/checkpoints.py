"""Checkpoint identity: resolve weights by content hash, not by file path.

A 437 MB checkpoint does not belong in a wheel, and a hard-coded path into this
repository does not survive `pip install`. The package instead ships a registry
of published checkpoint ids and their SHA-256 digests. Resolution looks in two
places for a file named `<sha256>.pt`:

    $AZURITE_CHECKPOINT_DIR/<sha256>.pt
    $AZURITE_HOME/checkpoints/<sha256>.pt   (default: platform cache dir)

`azurite checkpoint add` verifies a source file against the registry and puts it
in the cache under its digest. Every load re-hashes the file, so a truncated
download or a swapped checkpoint fails loudly rather than producing plausible
wrong intents.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from azurite.artifacts import sha256_file
from azurite.bundle import BundleError, Checkpoint, load_registry


class CheckpointError(RuntimeError):
    """A checkpoint is absent, mismatched, or not published."""


def cache_root() -> Path:
    if value := os.environ.get("AZURITE_HOME"):
        return Path(value).expanduser()
    if value := os.environ.get("XDG_CACHE_HOME"):
        return Path(value).expanduser() / "azurite"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "azurite"
    return Path.home() / ".cache" / "azurite"


def checkpoint_dirs(home: Path | None = None) -> list[Path]:
    dirs = []
    if value := os.environ.get("AZURITE_CHECKPOINT_DIR"):
        dirs.append(Path(value).expanduser())
    dirs.append((home or cache_root()) / "checkpoints")
    unique = []
    for path in dirs:
        resolved = path.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


@dataclass(frozen=True)
class Resolved:
    checkpoint: Checkpoint
    path: Path

    @property
    def sha256(self) -> str:
        return self.checkpoint.sha256


def _require_digest(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise CheckpointError(
            f"{path} has SHA-256 {actual}, but checkpoint identity requires {expected}"
        )


def find(checkpoint: Checkpoint, *, home: Path | None = None) -> Path | None:
    for directory in checkpoint_dirs(home):
        candidate = directory / f"{checkpoint.sha256}.pt"
        if candidate.is_file():
            return candidate
    return None


def resolve(
    ident: str | None = None, *, home: Path | None = None
) -> Resolved:
    try:
        checkpoint = load_registry().get(ident)
    except BundleError as exc:
        raise CheckpointError(str(exc)) from exc
    path = find(checkpoint, home=home)
    if path is None:
        locations = "\n".join(f"  {d}" for d in checkpoint_dirs(home))
        raise CheckpointError(
            f"checkpoint {checkpoint.id!r} is not installed.\n"
            f"  expected SHA-256: {checkpoint.sha256}\n"
            f"  searched:\n{locations}\n"
            "Install it from a weights file with:\n"
            f"  azurite checkpoint add /path/to/weights.pt --id {checkpoint.id}"
        )
    _require_digest(path, checkpoint.sha256)
    return Resolved(checkpoint=checkpoint, path=path)


def add(
    source: Path,
    *,
    ident: str | None = None,
    home: Path | None = None,
    link: bool = False,
) -> Resolved:
    source = source.expanduser()
    if not source.is_file():
        raise CheckpointError(f"{source} is not a file")
    digest = sha256_file(source)
    registry = load_registry()
    if ident is not None:
        checkpoint = registry.get(ident)
        if digest != checkpoint.sha256:
            raise CheckpointError(
                f"{source} has SHA-256 {digest}, not {checkpoint.sha256} "
                f"required by {checkpoint.id!r}"
            )
    else:
        matches = [c for c in registry.checkpoints if c.sha256 == digest]
        if not matches:
            known = ", ".join(c.id for c in registry.checkpoints)
            raise CheckpointError(
                f"{source} is not a published checkpoint (SHA-256 {digest}); "
                f"known ids: {known}"
            )
        checkpoint = matches[0]

    destination = (home or cache_root()) / "checkpoints" / f"{checkpoint.sha256}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _require_digest(destination, checkpoint.sha256)
        return Resolved(checkpoint=checkpoint, path=destination)
    if link:
        try:
            os.link(source, destination)
        except OSError as exc:
            raise CheckpointError(
                f"cannot hard-link {source} to {destination}: {exc}; "
                "omit --link to copy instead"
            ) from exc
    else:
        shutil.copy2(source, destination)
        _require_digest(destination, checkpoint.sha256)
    return Resolved(checkpoint=checkpoint, path=destination)


def installed() -> list[tuple[Checkpoint, Path | None]]:
    return [(c, find(c)) for c in load_registry().checkpoints]
