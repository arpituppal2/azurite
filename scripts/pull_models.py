#!/usr/bin/env python3
"""Record the competitor checkpoints used as local baselines.

"Beats Laya" is only a claim about measurement if Laya is run here, on these
splits, with this harness. Its published numbers were produced on unknown
splits with an unknown harness; quoting them would be a comparison of two
different experiments, which is how this project got into trouble before.

This downloads the Laya checkpoints and records, per repo: commit sha, licence,
every file with its sha256 and size, and the parameter count read from the
safetensors headers -- not from the model card, which states whatever it likes.

Usage:
    .venv/bin/python3 scripts/pull_models.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
MODELS = DATA / "models"
MANIFEST = DATA / "MANIFEST.json"

os.environ.setdefault("HF_HOME", str(DATA / ".hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

REPOS = [
    {"key": "laya", "repo": "convaiinnovations/laya",
     "role": "primary baseline: the model the project claims to beat"},
    {"key": "laya_multilingual", "repo": "convaiinnovations/laya-multilingual",
     "role": "multilingual arm baseline"},
    {"key": "laya_typed_decisions", "repo": "convaiinnovations/laya-typed-decisions",
     "role": "baseline for the typed-decisions interface"},
]

IGNORE = ["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.ot", "*.pth", "*.gguf"]


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def param_count(model_dir: Path) -> dict:
    """Parameters per safetensors file, from the headers alone."""
    from safetensors import safe_open

    out = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        total = 0
        dtypes: dict[str, int] = {}
        with safe_open(str(f), framework="pt") as fh:
            for key in fh.keys():
                shape = fh.get_slice(key).get_shape()
                n = 1
                for dim in shape:
                    n *= dim
                total += n
                dtype = str(fh.get_slice(key).get_dtype())
                dtypes[dtype] = dtypes.get(dtype, 0) + n
        out[f.name] = {"params": total, "by_dtype": dtypes}
    return out


def main() -> int:
    api = HfApi()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"datasets": {}}
    models = manifest.setdefault("models", {})

    for entry in REPOS:
        key, repo = entry["key"], entry["repo"]
        print(f"\n=== {key} ({repo}) ===", flush=True)
        t0 = time.time()
        try:
            info = api.model_info(repo)
        except Exception as exc:
            print(f"  !! {type(exc).__name__}: {exc}")
            models[key] = {"kind": "model", "repo": repo, "error": str(exc)[:300]}
            continue
        card = getattr(info, "cardData", None) or {}
        local = MODELS / key
        path = snapshot_download(
            repo_id=repo, repo_type="model",
            local_dir=str(local), ignore_patterns=IGNORE,
        )
        root = Path(path)
        files = []
        for f in sorted(root.rglob("*")):
            if f.is_file() and ".cache" not in f.parts:
                files.append(
                    {
                        "path": str(f.relative_to(root)),
                        "bytes": f.stat().st_size,
                        "sha256": sha256_file(f),
                    }
                )
        params = param_count(root)
        total_params = sum(v["params"] for v in params.values())
        models[key] = {
            "kind": "model",
            "role": entry["role"],
            "repo": repo,
            "sha": info.sha,
            "license": card.get("license"),
            "local_path": str(local.relative_to(PROJECT)),
            "params_total": total_params,
            "params_by_file": params,
            "files": files,
            "bytes_total": sum(f["bytes"] for f in files),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  sha {str(info.sha)[:8]}  licence {card.get('license')}")
        print(f"  {total_params / 1e6:.1f}M parameters, "
              f"{models[key]['bytes_total'] / 1e9:.2f} GB, {len(files)} files")
        for name, v in params.items():
            print(f"    {name}: {v['params'] / 1e6:.1f}M {v['by_dtype']}")

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nmanifest -> {MANIFEST.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
