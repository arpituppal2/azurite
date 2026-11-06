"""Path helpers shared by the scripts.

`Path.relative_to` raises when the argument is not a subpath — and legitimately
so: `--out /tmp/x.json` and a pytest `tmp_path` are both outside the project.
Five scripts independently hit this while *printing* a success message, each
after the artifact had already been written, which turns a completed run into a
non-zero exit and makes anything chaining on exit status think a finished job
failed. One helper, one place to fix it.
"""

from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def rel(path: "Path | str") -> str:
    """Project-relative when possible, absolute otherwise. Never raises."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT))
    except ValueError:
        return str(resolved)
