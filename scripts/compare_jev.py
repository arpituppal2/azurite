#!/usr/bin/env python3
"""Join Azurite's Jev-overlap runs to the published Jev points.

The output is deliberately a table of gaps, not a single mean. Nine of the
datasets use accuracy; four use F1, one uses AUPRC, four use Spearman. A mean
over those would be a manufactured number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "research" / "jev_benchmarks.json"
OUT = ROOT / "results" / "jev_comparison"

NAME = {
    "ag_news": "AG News", "banking77": "Banking77", "clinc_oos": "CLINC150",
    "emotion": "Emotion", "financial_phrasebank": "Financial PhraseBank",
    "language_id": "Language ID", "imdb": "IMDB",
    "rotten_tomatoes": "Rotten Tomatoes", "sst2": "SST-2",
    "sms_spam": "SMS Spam", "paws": "PAWS", "anli": "ANLI", "boolq": "BoolQ",
    "toxigen": "ToxiGen", "toxic_chat": "ToxicChat",
    "prompt_injections": "Prompt Injections", "agb_de": "AGB-DE",
}


def wilson(correct: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_runs(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("cells", {})


def main() -> int:
    inv = json.loads(INVENTORY.read_text())
    points = {}
    for rows in inv["families"].values():
        for row in rows:
            points[row["dataset"]] = row
    runs = {}
    for path in [
        ROOT / "results/jev_full_20261030/summary.json",
        ROOT / "results/jev_extra_20261030/summary.json",
    ]:
        runs.update(load_runs(path))

    rows = []
    for task, name in NAME.items():
        run = runs.get(task, {})
        point = points.get(name)
        if not point:
            continue
        row = {
            "dataset": name,
            "task": task,
            "jev": point["score"],
            "jev_metric": point["metric"],
            "jev_n": point["n"],
            "jev_ci": point.get("ci"),
            "azurite_status": run.get("status", "not_run"),
            "azurite_n": run.get("n"),
            "azurite_accuracy": run.get("accuracy"),
            "azurite_seconds": run.get("seconds"),
            "split_note": (
                "local split differs" if point["n"] != run.get("n") else ""
            ),
        }
        acc = run.get("accuracy")
        if acc is not None and point["metric"] == "accuracy":
            row["gap_pp"] = round(100 * (acc - point["score"]), 3)
            correct = round(acc * run["n"])
            row["azurite_wilson95"] = [round(x, 4) for x in wilson(correct, run["n"])]
        elif acc is not None:
            row["gap_pp"] = None
            row["gap_note"] = f"local metric is accuracy, Jev metric is {point['metric']}"
        rows.append(row)

    rows.sort(key=lambda r: (r["azurite_status"] != "ok", r["dataset"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps({
        "note": "Published Jev 1.13.0 points, not paired runs. Only accuracy cells have a gap.",
        "source": inv["source"],
        "rows": rows,
    }, indent=2) + "\n")

    lines = [
        "# Jev overlap: Azurite on the same published benchmarks",
        "",
        "Jev numbers are the published `jev-1.13.0` points from the Bonn study "
        "([arXiv:2609.37647v1](https://arxiv.org/html/2609.37647v1), Table 2). "
        "Azurite numbers are local 113.47M byte-encoder runs on the full official "
        "evaluation splits, fine-tuned on each task's train split. **This is not a "
        "paired run**: Jev is zero-shot and hosted; Azurite fine-tunes. A gap is a "
        "gap to a published point, not a head-to-head win.",
        "",
        "| dataset | n (Jev / local) | metric | Jev | Azurite | gap (pp) | status |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        n = f"{row['jev_n']:,}" if row["jev_n"] else "—"
        local_n = f"{row['azurite_n']:,}" if row["azurite_n"] else "—"
        acc = "—" if row["azurite_accuracy"] is None else f"{row['azurite_accuracy']:.4f}"
        gap = "—" if row.get("gap_pp") is None else f"{row['gap_pp']:+.2f}"
        status = row["azurite_status"]
        if row["split_note"]:
            status += f" ({row['split_note']})"
        lines.append(
            f"| {row['dataset']} | {n} / {local_n} | {row['jev_metric']} | "
            f"{row['jev']:.3f} | {acc} | {gap} | {status} |"
        )
    lines += [
        "",
        "## What this does and does not show",
        "",
        "- Only accuracy cells have a numeric gap. F1 / AUPRC / Spearman cells are "
        "listed but not differenced, because the local run reports accuracy.",
        "- Financial PhraseBank is 1,000 local rows against Jev's 970; SMS Spam is "
        "a fixed 80/20 local split because HF ships one 5,574-row split. Both are "
        "marked, not silently pooled.",
        "- The six core runs (AG News, Banking77, CLINC150, Emotion, Financial "
        "PhraseBank, Language ID) use the full official splits. The extra batch "
        "adds IMDB, Rotten Tomatoes, SST-2, SMS Spam, PAWS, ANLI, BoolQ, ToxiGen, "
        "ToxicChat, Prompt Injections and AGB-DE.",
        "- Per-row multiple choice (MMLU, HellaSwag, ARC, WinoGrande, "
        "CommonsenseQA, alphaNLI, Belebele, BIG-bench), multi-label (GoEmotions, "
        "OpenAI Moderation, UNFAIR-ToS) and ordinal scoring (STS-B, SST-5, "
        "SummEval, HelpSteer2) are not represented yet; they need a different "
        "harness, not a different checkpoint.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "RESULTS.md")
    print("rows", len(rows), "ok", sum(r["azurite_status"] == "ok" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
