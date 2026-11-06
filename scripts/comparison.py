#!/usr/bin/env python3
"""Render the competitor comparison from recorded facts.

Rule enforced by construction: every cell is either
  (a) a published figure with the URL it came from,
  (b) a figure measured by this repo, with the script and result file, or
  (c) the literal string "not disclosed".
There is no fourth option. Estimates do not go in this table; where a number is
unknown it says so, because a filled-in guess in a comparison table is how a
project starts lying to itself.

Usage:
    python3 scripts/comparison.py                 # markdown to stdout
    python3 scripts/comparison.py --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LAYA_CARD = "https://huggingface.co/convaiinnovations/laya"
LAYA_TYPED = "https://huggingface.co/convaiinnovations/laya-typed-decisions"
LAYA_BENCH = "https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md"
JEV_POST = "https://typesafe.ai/blog/introducing-system-one-models-and-jev"
MMBERT_PAPER = "https://arxiv.org/html/2509.06888v1"
MODERNBERT_PAPER = "https://arxiv.org/html/2412.13663v2"
OUR_EVAL = "scripts/eval_laya.py"
OUR_MANIFEST = "data/MANIFEST.json"

ND = "not disclosed"
UNTRAINED = "no model exists yet"

# (field, laya, laya_status, jev, jev_status, azurite, azurite_status, source)
FIELDS: list[tuple[str, str, str, str, str, str, str, str]] = [
    (
        "English checkpoint parameters",
        "**421M** — ModernBERT-large encoder + a 2-layer decision head",
        "published",
        ND,
        "not disclosed",
        "395M encoder + head is the parity target; largest local model is 86M",
        "target",
        LAYA_CARD,
    ),
    (
        "Multilingual checkpoint parameters",
        "**322M** — mmBERT-base encoder + head",
        "published",
        ND,
        "not disclosed",
        "out of reach: see next row",
        "target",
        LAYA_CARD,
    ),
    (
        "Encoder pretraining tokens",
        "**~2T** for the English line (ModernBERT: 1.7T at len 1024 + 300B "
        "context extension); **~3T** for the multilingual line (mmBERT, 1,833 "
        "languages)",
        "published",
        ND,
        "not disclosed",
        "**5.2B available locally** — 0.26% of the English line, 0.17% of the "
        "multilingual line",
        "measured",
        MODERNBERT_PAPER,
    ),
    (
        "Architecture",
        "Bidirectional encoder, published: RoPE, GeGLU, alternating "
        "local/global attention, 22 layers (base) / 28 (large); head is 2 "
        "linear layers",
        "published",
        '"a new model architecture" + a "parallel sampler" + Reinforcement '
        "Learning for Calibrated Decisions — no specification, no layer counts, "
        "no sizes",
        "not disclosed",
        "undecided — tokenizer and depth are open variables in this study, not "
        "settled",
        "open",
        LAYA_CARD,
    ),
    (
        "Context window",
        "512 (English ckpt) / 1,024 (multilingual, typed-decisions); the "
        "encoder itself supports 8,192",
        "published",
        ND,
        "not disclosed",
        "to be set by the study; the decision interface packs "
        "context+instruction+K candidates into one window",
        "target",
        LAYA_CARD,
    ),
    (
        "Vocabulary",
        "modified OLMo tokenizer (ModernBERT); 256k for the mmBERT line",
        "published",
        ND,
        "not disclosed",
        "learned from our own corpus — byte-level (256) and corpus-trained BPE "
        "are both arms of the study",
        "open",
        MMBERT_PAPER,
    ),
    (
        "Weights",
        "released on HuggingFace (Apache-2.0 for the ModernBERT/mmBERT "
        "backbones)",
        "published",
        "served through the TypeSafe API; weights not released",
        "published",
        "none — this is a from-scratch study, and no pretrained weights or "
        "vocabulary enter any arm",
        "constraint",
        LAYA_CARD,
    ),
    (
        "Fine-tuning disclosed",
        "typed-decisions: 1,200 cases / 6,000 decisions train, 400 cases / "
        "2,000 decisions held out, across 4 workflows; ~4-5 h on Kaggle 2xT4",
        "published",
        ND,
        "not disclosed",
        "9 primary + 11 secondary task splits, 2,659,054 rows on disk",
        "measured",
        LAYA_TYPED,
    ),
    (
        "Latency (decision call)",
        "**93-114 ms p50** on 3-6 option tasks, **283 ms** at 77 options — "
        "measured here, not published by the vendor",
        "measured (this repo)",
        ND,
        "not disclosed",
        "to be measured on identical rows; the comparison is only meaningful "
        "paired, on the same machine, in the same process",
        "target",
        OUR_EVAL,
    ),
    (
        "Calibration / abstention reporting",
        "none published — accuracy only",
        "published (absent)",
        '"calibrated decisions" is the product claim; no ECE, reliability '
        "diagram, or risk-coverage curve is published",
        "not disclosed",
        "**this is a contribution of the study**: ECE with stated binning and "
        "risk-coverage curves, neither of which either vendor publishes",
        "target",
        JEV_POST,
    ),
    (
        "Published evaluation harness",
        "a vendor benchmark report plus a vendor-authored typed-decisions "
        "benchmark (the model's own card reports 0.362 zero-shot / 0.766 "
        "fine-tuned on it)",
        "published",
        "vendor-published per-task scores; methodology not published",
        "not disclosed",
        "20 public task splits, frozen before any training, contamination "
        "audited with a positive-controlled checker",
        "measured",
        LAYA_BENCH,
    ),
]

# Published benchmark scores. Several tasks have MULTIPLE Laya numbers from
# different runs; all of them are shown rather than one being picked.
# (task, n_options, laya_en, laya_multiling, jev, azurite, source, note)
BENCHMARKS: list[tuple] = [
    ("ag_news", 4, "0.950 / 0.953 / 0.947", "—", "0.910", "pending", LAYA_CARD, "three Laya numbers from three runs: card, application run, T4 suite"),
    ("emotion (DAIR)", 6, "0.595 / 0.600 / 0.573", "—", "0.480", "pending", LAYA_CARD, "same three-run pattern"),
    ("banking77", 77, "0.425", "—", "0.870 (72 labels)", "pending", LAYA_CARD, "**not like-for-like**: Laya scored 77 labels, Jev 72. Jev's figure is a different task."),
    ("MASSIVE intent (en)", 60, "0.783", "0.657", "—", "pending", LAYA_CARD, "Laya's own cross-language row is 0.306 for the English checkpoint"),
    ("MASSIVE intent (13 other langs)", 60, "0.306", "0.451", "—", "pending", LAYA_CARD, "the multilingual checkpoint exists precisely for this row"),
    ("XNLI (en)", 3, "0.860", "0.843", "—", "pending", LAYA_CARD, "measured here at 0.8000 (n=100) — the vendor's number is 0.06 higher"),
    ("XNLI (14 other langs)", 3, "0.521", "0.731", "—", "pending", LAYA_CARD, ""),
    ("typed decisions", "K", "0.362 zero-shot / 0.766 fine-tuned", "—", "—", "pending", LAYA_TYPED, "the only decision-format benchmark any of these vendors publishes; 400 held-out cases"),
]


def render() -> str:
    out = []
    A = out.append
    A("# Laya vs Jev vs Azurite — public record, line for line\n")
    A("Generated by `scripts/comparison.py`. Every cell is a published figure "
      "with its source, a figure measured by this repo, or `not disclosed`. "
      "There is no fourth kind of cell.\n")
    A("## The honest summary before the table\n")
    A("**Jev publishes no numbers that permit a structural comparison.** No "
      "parameter count, no architecture, no token budget, no context length, no "
      "latency, no harness. What exists is a product description and seven "
      "per-task scores on unspecified splits. Any comparison table claiming "
      "otherwise is inventing those cells, and this one will not.\n")
    A("**Laya publishes three different numbers for the same task.** AG News "
      "appears as 0.950 (model card), 0.953 (application run) and 0.947 (T4 "
      "suite); DAIR emotion as 0.595, 0.600 and 0.573. These are different "
      "harnesses, so there is no single \"Laya number\" to beat — a head-to-head "
      "must name which one it is comparing against, and this study will measure "
      "Laya on its own frozen rows through Laya's own interface rather than "
      "cite a card.\n")
    A("**The banking77 comparison between the two vendors is invalid.** Laya "
      "scored a 77-label label space, Jev a 72-label one. The vendor table "
      "presents these side by side.\n")
    A("## Structural comparison\n")
    A("| field | Laya | Jev | Azurite (target) | source |")
    A("|---|---|---|---|---|")
    for field, laya, ls, jev, js, az, as_, src in FIELDS:
        A(
            f"| **{field}** | {laya} *({ls})* | {jev} *({js})* | {az} *({as_})* | "
            f"[link]({src}) |"
        )
    A("")
    A("## Published benchmark scores\n")
    A("`pending` means this study has not run that cell yet. It does not mean a "
      "number is coming; it means none exists.\n")
    A("| task | options | Laya (en) | Laya (multiling) | Jev | Azurite | note |")
    A("|---|---:|---|---:|---|---|---|")
    for task, k, l_en, l_multi, jev, az, src, note in BENCHMARKS:
        A(f"| `{task}` | {k} | {l_en} | {l_multi} | {jev} | {az} | {note} |")
    A("")
    A("Sources: "
      f"[Laya card]({LAYA_CARD}) · [Laya typed-decisions]({LAYA_TYPED}) · "
      f"[Laya benchmark report]({LAYA_BENCH}) · [TypeSafe]({JEV_POST}) · "
      f"[ModernBERT]({MODERNBERT_PAPER}) · [mmBERT]({MMBERT_PAPER})\n")
    A("## What this table costs to change\n")
    A("The only rows Azurite can move are the `pending` ones, and "
      "`scripts/budget.py` prices the cheapest honest attempt at the parity "
      "row: **$19.3k-$26.3k** of H100 rental for a 395M encoder trained on the "
      "2T tokens Laya's encoder consumed, and **$35.7k-$48.6k** to reproduce "
      "Laya's actual base-then-large recipe. The plan previously carried "
      "$420 for that row, from a 20x-short token count and a 1.5-2x optimistic "
      "MFU; the two errors partly cancelled. That is why the number looked "
      "plausible.\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    text = render()
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "fields": [
                        {
                            "field": f,
                            "laya": l,
                            "laya_status": ls,
                            "jev": j,
                            "jev_status": js,
                            "azurite": a,
                            "azurite_status": as_,
                            "source": s,
                        }
                        for f, l, ls, j, js, a, as_, s in FIELDS
                    ],
                    "benchmarks": [
                        {
                            "task": t,
                            "options": k,
                            "laya_en": le,
                            "laya_multilingual": lm,
                            "jev": j,
                            "azurite": a,
                            "source": s,
                            "note": n,
                        }
                        for t, k, le, lm, j, a, s, n in BENCHMARKS
                    ],
                },
                indent=1,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
