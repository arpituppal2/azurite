#!/usr/bin/env python3
"""Build the resources the installed package carries.

Three files are emitted, all deterministic so a rebuild either reproduces them
byte for byte or fails:

    azurite/resources/banking77.json.gz   labels + 9,968 decontaminated support messages
    azurite/resources/checkpoints.json    checkpoint ids -> content hashes
    azurite/resources/provenance.json     source files, hashes, licence

The bundle is what makes a prediction possible after `pip install`: no repo
root, no `data/` directory, no Hugging Face download. The 25 evaluation texts
found in the 9,993-row source support split are removed here, and the build
fails if any normalised evaluation text survives. Checkpoints are *not*
bundled at 437 MB each; the registry names the exact SHA-256 the resolver looks
for, and `azurite checkpoint add` puts the file where the resolver expects it.

    .venv/bin/python scripts/build_package_resources.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "azurite" / "resources"
INPUTS = ROOT / "results" / "jev_matched" / "inputs.json"
CLEAN_SUPPORT = ROOT / "results" / "product_path_clean" / "clean_support.jsonl.gz"
ORIGINAL_SUPPORT = ROOT / "results" / "jev_query" / "support.jsonl"
REMOVED_SUPPORT = ROOT / "results" / "product_path_clean" / "removed_support.json"
CLEAN_SUMMARY = ROOT / "results" / "product_path_clean" / "summary.json"
NATIVE_SUMMARY = ROOT / "results" / "jev_query" / "eval" / "summary.json"
LONG_SUMMARY = ROOT / "results" / "jev_query_5120" / "eval" / "summary.json"
NATIVE_HEAD = ROOT / "results" / "jev_query" / "final.pt"
LONG_HEAD = ROOT / "results" / "jev_query_5120" / "final.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gzip_bytes(data: bytes) -> bytes:
    # mtime=0 and an empty filename keep the gzip container reproducible.
    return gzip.compress(data, compresslevel=9, mtime=0)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def normalize(text: str) -> str:
    """The decontamination normalisation used by the clean rerun."""
    folded = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def build_bundle() -> dict:
    data = json.loads(INPUTS.read_text())
    labels = data["labels"]
    if len(labels) != 77:
        raise SystemExit(f"expected 77 labels, got {len(labels)}")
    if data.get("task") != "banking77":
        raise SystemExit(f"inputs task is {data.get('task')!r}, not banking77")
    if len(data["rows"]) != 3080:
        raise SystemExit(f"expected 3,080 evaluation rows, got {len(data['rows'])}")

    removed = json.loads(REMOVED_SUPPORT.read_text())
    if len(removed) != 25:
        raise SystemExit(f"expected 25 removed support rows, got {len(removed)}")
    evaluation_texts = {normalize(row["text"]) for row in data["rows"]}
    clean_rows = read_gzip_jsonl(CLEAN_SUPPORT)
    residue = [row for row in clean_rows if normalize(row["message"]) in evaluation_texts]
    if residue:
        raise SystemExit(
            f"clean support still contains {len(residue)} evaluation texts; "
            "refusing to build a contaminated wheel"
        )

    support = []
    seen_ids = set()
    for row in clean_rows:
        message = str(row["message"]).strip()
        label = str(row["label"])
        if not message:
            raise SystemExit(f"support row {row['id']} has an empty message")
        if label not in labels:
            raise SystemExit(f"support row {row['id']} has unknown label {label!r}")
        if row["id"] in seen_ids:
            raise SystemExit(f"duplicate support id {row['id']}")
        seen_ids.add(row["id"])
        support.append({"id": int(row["id"]), "message": message, "label": label})

    bundle = {
        "schema": 1,
        "task": "banking77",
        "labels": labels,
        "support": support,
    }
    raw = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    packed = gzip_bytes(raw)
    RESOURCES.mkdir(parents=True, exist_ok=True)
    out = RESOURCES / "banking77.json.gz"
    out.write_bytes(packed)

    provenance = {
        "schema": 1,
        "bundle": {
            "path": "azurite/resources/banking77.json.gz",
            "sha256": sha256_bytes(packed),
            "uncompressed_sha256": sha256_bytes(raw),
            "bytes": len(packed),
            "uncompressed_bytes": len(raw),
            "labels": len(labels),
            "support_rows": len(support),
        },
        "sources": {
            "labels_and_eval_rows": {
                "path": INPUTS.relative_to(ROOT).as_posix(),
                "sha256": sha256(INPUTS),
                "labels_sha256": sha256_bytes(
                    json.dumps(labels, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ),
                "rows": len(data["rows"]),
            },
            "support": {
                "path": CLEAN_SUPPORT.relative_to(ROOT).as_posix(),
                "sha256": sha256(CLEAN_SUPPORT),
                "uncompressed_sha256": sha256_bytes(
                    gzip.decompress(CLEAN_SUPPORT.read_bytes())
                ),
                "rows": len(support),
                "original_path": ORIGINAL_SUPPORT.relative_to(ROOT).as_posix(),
                "original_sha256": sha256(ORIGINAL_SUPPORT),
                "original_rows": 9_993,
                "removed_path": REMOVED_SUPPORT.relative_to(ROOT).as_posix(),
                "removed_sha256": sha256(REMOVED_SUPPORT),
                "removed_rows": len(removed),
                "removed_exact_text_matches": json.loads(
                    CLEAN_SUMMARY.read_text()
                )["support"]["removed_exact_text_matches"],
                "normalisation": (
                    "NFKC, casefold, non-alphanumeric runs collapsed to spaces"
                ),
                "residue_rows": 0,
            },
        },
        "licence": {
            "code_and_weights": "MIT",
            "banking77_support_text": "CC-BY-4.0",
            "attribution": (
                "Banking77 (Casanueva et al., 2020), PolyAI. The bundled support "
                "messages are a decontaminated subset of the training split as "
                "pulled by this project: 25 evaluation texts removed from 9,993 "
                "source rows; see DATA.md and THIRD_PARTY_NOTICES.md."
            ),
        },
        "generated_by": "scripts/build_package_resources.py",
    }
    (RESOURCES / "provenance.json").write_text(
        json.dumps(provenance, indent=1, sort_keys=True) + "\n"
    )
    return provenance


def build_registry() -> dict:
    native = json.loads(NATIVE_SUMMARY.read_text())
    long = json.loads(LONG_SUMMARY.read_text())
    clean = json.loads(CLEAN_SUMMARY.read_text())
    native_arm = native["arms"]["query_a2c"]
    long_arm = long["arms"]["len_a2c"]
    clean_arm = clean["clean"]
    contamination = clean["frozen_examples"]
    source_support = clean["support"]
    jev = native["state"]
    decision = native["decision"]
    native_head = native["heads"]["query"]
    long_head = long["heads"]["query"]
    if native_head["sha256"] != sha256(NATIVE_HEAD):
        raise SystemExit("native head hash does not match its evaluation summary")
    if long_head["sha256"] != sha256(LONG_HEAD):
        raise SystemExit("5120 head hash does not match its evaluation summary")
    if source_support["clean_rows"] != 9_968:
        raise SystemExit("clean support record does not have 9,968 rows")
    if contamination["rows_with_any_evaluation_text"] != 255:
        raise SystemExit("clean support record disagrees on affected rows")

    def entry(
        *,
        ident: str,
        path: Path,
        head: dict,
        measured: dict,
        max_seq: int,
        context_budget: int,
        examples: int,
        frozen: dict | None = None,
    ) -> dict:
        item = {
            "id": ident,
            "sha256": head["sha256"],
            "source_path": path.relative_to(ROOT).as_posix(),
            "task": "banking77",
            "head": "query",
            "max_seq": max_seq,
            "context_budget": context_budget,
            "examples": examples,
            "measured": measured,
        }
        if frozen is not None:
            item["frozen"] = frozen
        return item

    clean_measured = {
        "n": clean_arm["n"],
        "correct": clean_arm["correct"],
        "accuracy": clean_arm["accuracy"],
        "wilson95": clean_arm["wilson95"],
        "bank": "decontaminated",
        "support_rows": source_support["clean_rows"],
        "support_source_rows": source_support["original_rows"],
        "jev_accuracy": jev["jev_accuracy"],
        "gap_vs_jev": clean_arm["accuracy"] - jev["jev_accuracy"],
        "verdict": decision["verdict"],
    }
    frozen_measured = {
        "n": native_arm["n"],
        "correct": native_arm["correct"],
        "accuracy": native_arm["accuracy"],
        "wilson95": native_arm["wilson95"],
        "bank": "original",
        "support_rows": source_support["original_rows"],
        "contamination": {
            "support_messages": contamination["unique_evaluation_texts_in_support"],
            "support_rows": source_support["original_rows"],
            "exact": contamination["instances_exact_text"],
            "affected_rows": contamination["rows_with_any_evaluation_text"],
            "affected_share": round(
                100 * contamination["rows_with_any_evaluation_text"] / native_arm["n"], 2
            ),
            "instances": contamination["instances_matching_an_evaluation_text"],
        },
    }
    long_measured = {
        "n": long_arm["n"],
        "correct": long_arm["correct"],
        "accuracy": long_arm["accuracy"],
        "wilson95": long_arm["wilson95"],
        "bank": "frozen-examples",
        "jev_accuracy": jev["jev_accuracy"],
        "gap_vs_jev": long["decision"]["diff_vs_jev"],
        "mcnemar_p": long["decision"]["mcnemar_p"],
        "verdict": long["decision"]["verdict"],
    }

    registry = {
        "schema": 1,
        "task": "banking77",
        "default": "query-native",
        "checkpoints": [
            entry(
                ident="query-native",
                path=NATIVE_HEAD,
                head=native_head,
                measured=clean_measured,
                frozen=frozen_measured,
                max_seq=1024,
                context_budget=1020,
                examples=24,
            ),
            entry(
                ident="query-joint-5120",
                path=LONG_HEAD,
                head=long_head,
                measured=long_measured,
                max_seq=5120,
                context_budget=5118,
                examples=24,
            ),
        ],
    }
    (RESOURCES / "checkpoints.json").write_text(
        json.dumps(registry, indent=1, sort_keys=True) + "\n"
    )
    return registry


def copy_documents() -> dict:
    """Ship the model card and attribution inside the wheel."""
    documents = {}
    for name in ("MODEL_CARD.md", "THIRD_PARTY_NOTICES.md"):
        source = ROOT / name
        if not source.is_file():
            raise SystemExit(f"missing {name}; cannot build the wheel")
        data = source.read_bytes()
        (RESOURCES / name).write_bytes(data)
        documents[name] = sha256_bytes(data)
    return documents


def main() -> int:
    bundle_provenance = build_bundle()
    provenance = json.loads((RESOURCES / "provenance.json").read_text())
    provenance["documents"] = copy_documents()
    (RESOURCES / "provenance.json").write_text(
        json.dumps(provenance, indent=1, sort_keys=True) + "\n"
    )
    registry = build_registry()
    print(
        f"bundle sha256 {bundle_provenance['bundle']['sha256']} "
        f"({bundle_provenance['bundle']['support_rows']} support rows, "
        f"{bundle_provenance['bundle']['bytes']} bytes)"
    )
    for item in registry["checkpoints"]:
        print(
            f"{item['id']}: {item['sha256']} "
            f"{item['measured']['accuracy']:.6f} on {item['measured']['n']} rows"
        )
    print(f"wrote {RESOURCES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
