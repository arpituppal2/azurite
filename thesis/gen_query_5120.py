#!/usr/bin/env python3
"""Generate the 5,120-byte joint-fine-tune table and macros from stored records.

Reads `results/jev_query_5120/eval/summary.json`, its per-example files and the
trained-head metadata. The run follows `PRE_REGISTRATION_AMENDMENT_1.md`: one
joint fine-tune at 5,120 positions instead of a separate masked-LM adaptation.
Every arm total and Wilson interval is recomputed before a cell is written. A
missing record is an error, never a `pending` placeholder.

    python thesis/gen_query_5120.py            # write fragments
    python thesis/gen_query_5120.py --check    # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.stats import wilson  # noqa: E402

SRC = ROOT / "results" / "jev_query_5120" / "eval"
NATIVE = ROOT / "results" / "jev_query" / "eval" / "summary.json"
INIT = ROOT / "results" / "pretrain" / "l1_large_byte_iso10v3" / "weights_only.pt"
OUT = ROOT / "thesis" / "tables"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def num(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def signed(x: float) -> str:
    text = f"{100 * x:+.2f}"
    return f"${text}$" if text.startswith("-") else text


def science(p: float) -> str:
    if p == 0:
        return r"$<10^{-300}$"
    exp = int(math.floor(math.log10(p)))
    mant = p / 10 ** exp
    return rf"${mant:.2f}\times10^{{{exp}}}$"


def load() -> tuple[dict, dict]:
    summary = json.loads((SRC / "summary.json").read_text())
    for name, arm in summary["arms"].items():
        path = SRC / f"per_example_{name}.jsonl"
        if sha256(path) != arm["per_example_sha256"]:
            raise SystemExit(f"{name}: per-example hash mismatch")
        rows = [json.loads(line) for line in path.open()]
        k = sum(int(r["correct"]) for r in rows)
        if len(rows) != arm["n"] or k != arm["correct"]:
            raise SystemExit(f"{name}: recomputed {k}/{len(rows)} != stored")
        lo, hi = wilson(k, len(rows))
        if abs(lo - arm["wilson95"][0]) > 1e-12 or abs(hi - arm["wilson95"][1]) > 1e-12:
            raise SystemExit(f"{name}: Wilson interval mismatch")
    meta = json.loads((ROOT / "results/jev_query_5120/train_meta.json").read_text())
    if meta["head_sha256"] != summary["heads"]["query"]["sha256"]:
        raise SystemExit("head hash does not match summary")
    init_sha = sha256(INIT)
    if init_sha != summary["init_checkpoint"]["sha256"]:
        raise SystemExit("init checkpoint hash does not match summary")
    if meta["init_checkpoint_sha256"] != init_sha:
        raise SystemExit("train_meta init hash does not match the stored checkpoint")
    if summary["prereg_sha256"] != meta["prereg_sha256"]:
        raise SystemExit("pre-registration hash differs between train and eval")
    if summary["amendment_sha256"] != meta["amendment_sha256"]:
        raise SystemExit("amendment hash differs between train and eval")
    return summary, meta


def projection(sec_per_step: float) -> dict[str, float]:
    """Cost of the 16k/32k arms from the measured 5,120 step time.

    Step time is modelled as linear in sequence length plus a quadratic
    attention term, with the attention share at 5,120 taken between 0.3 and
    0.5. Each arm keeps 3,000 head steps and the dropped protocol's 1,220-step
    adaptation. This is an arithmetic allowance, not a measurement; GAPS.md
    states the formula.
    """
    out: dict[str, float] = {}
    for seq, tag in ((16_384, "Short"), (32_768, "Long")):
        ratio = seq / 5_120
        lo = 3_000 * sec_per_step * (ratio * 0.7 + ratio ** 2 * 0.3) / 3_600
        hi = 3_000 * sec_per_step * (ratio * 0.5 + ratio ** 2 * 0.5) / 3_600
        adapt_lo = 1_220 * sec_per_step * (ratio * 0.7 + ratio ** 2 * 0.3) / 3_600
        adapt_hi = 1_220 * sec_per_step * (ratio * 0.5 + ratio ** 2 * 0.5) / 3_600
        out[f"{tag}Lo"] = lo + adapt_lo
        out[f"{tag}Hi"] = hi + adapt_hi
    return out


def macros(s: dict, meta: dict) -> list[str]:
    a, p, d = s["arms"], s["paired"], s["decision"]
    sep = s["train_test_separation"]
    jev = s["state"]["jev_accuracy"]
    native = json.loads(NATIVE.read_text())["arms"]["query_a2c"]
    proj = projection(meta["ms_per_step"] / 1000)
    met = 1 if d["verdict"] != "P0 missed" else 0
    m = [
        rf"\newcommand{{\LenJevPct}}{{{pct(jev)}}}",
        rf"\newcommand{{\LenN}}{{{num(s['state']['n'])}}}",
        rf"\newcommand{{\LenPrereg}}{{{s['prereg_sha256'][:12]}}}",
        rf"\newcommand{{\LenAmendment}}{{{s['amendment_sha256'][:12]}}}",
        rf"\newcommand{{\LenInitSha}}{{{s['init_checkpoint']['sha256'][:12]}}}",
        rf"\newcommand{{\LenHeadSha}}{{{s['heads']['query']['sha256'][:12]}}}",
        rf"\newcommand{{\LenVerdict}}{{{d['verdict']}}}",
        rf"\newcommand{{\LenPZeroMet}}{{{met}}}",
        rf"\newcommand{{\LenEff}}{{{a['len_a2c']['mean_effective_examples']:.1f}}}",
        rf"\newcommand{{\LenEffMin}}{{{a['len_a2c']['min_effective_examples']}}}",
        rf"\newcommand{{\LenEffMax}}{{{a['len_a2c']['max_effective_examples']}}}",
        rf"\newcommand{{\LenContextMax}}{{{num(meta['max_context_bytes'])}}}",
        rf"\newcommand{{\LenNonInferior}}{{{'yes' if d['noninferior_at_minus_2pp'] else 'no'}}}",
        rf"\newcommand{{\LenHeadHours}}{{{meta['train_seconds'] / 3600:.2f}}}",
        rf"\newcommand{{\LenTotalHours}}{{{meta['train_seconds'] / 3600:.2f}}}",
        rf"\newcommand{{\LenSteps}}{{{num(meta['steps'])}}}",
        rf"\newcommand{{\LenBatch}}{{{meta['batch']}}}",
        rf"\newcommand{{\LenSecStep}}{{{meta['ms_per_step'] / 1000:.2f}}}",
        rf"\newcommand{{\LenTrainPct}}{{{pct(sep['train_accuracy'])}}}",
        rf"\newcommand{{\LenTrainCorrect}}{{{num(sep['train_correct'])}}}",
        rf"\newcommand{{\LenTrainLoss}}{{{sep['train_loss']:.4f}}}",
        rf"\newcommand{{\LenLastLoss}}{{{sep['final_training_loss']:.4f}}}",
        rf"\newcommand{{\LenMeanLastFifty}}{{{sep['mean_training_loss_last_50']:.4f}}}",
        rf"\newcommand{{\LenGapTrainTest}}{{{signed(sep['test_minus_train'])}}}",
    ]
    for key, label in (("old_a0", "OldZero"), ("old_a2c", "OldTwo"),
                       ("len_a0", "LenZero"), ("len_a2c", "LenTwo")):
        arm, pair = a[key], p[key]
        m += [
            rf"\newcommand{{\LenCorrect{label}}}{{{num(arm['correct'])}}}",
            rf"\newcommand{{\LenPct{label}}}{{{pct(arm['accuracy'])}}}",
            rf"\newcommand{{\LenLo{label}}}{{{pct(arm['wilson95'][0])}}}",
            rf"\newcommand{{\LenHi{label}}}{{{pct(arm['wilson95'][1])}}}",
            rf"\newcommand{{\LenGap{label}}}{{{signed(pair['diff_vs_jev'])}}}",
            rf"\newcommand{{\LenLower{label}}}{{{signed(pair['bootstrap_lower_95'])}}}",
            rf"\newcommand{{\LenP{label}}}{{{science(pair['mcnemar']['p'])}}}",
        ]
    m += [
        rf"\newcommand{{\LenDiff}}{{{signed(d['diff_vs_jev'])}}}",
        rf"\newcommand{{\LenDiffAbs}}{{{abs(d['diff_vs_jev']) * 100:.2f}}}",
        rf"\newcommand{{\LenLowerPrimary}}{{{signed(d['bootstrap_lower_95'])}}}",
        rf"\newcommand{{\LenPValue}}{{{science(d['mcnemar_p'])}}}",
        rf"\newcommand{{\LenPValueText}}{{p = {science(d['mcnemar_p']).strip('$')}}}",
        rf"\newcommand{{\LenMechGain}}{{{signed(p['len_a2c_vs_old_a2c']['diff'])}}}",
        rf"\newcommand{{\LenMechGainAbs}}{{{abs(p['len_a2c_vs_old_a2c']['diff']) * 100:.2f}}}",
        rf"\newcommand{{\LenMechLower}}{{{signed(p['len_a2c_vs_old_a2c']['bootstrap_lower_95'])}}}",
        rf"\newcommand{{\LenMechP}}{{{science(p['len_a2c_vs_old_a2c']['mcnemar']['p'])}}}",
        rf"\newcommand{{\LenVsLenZeroGain}}{{{signed(p['len_a2c_vs_len_a0']['diff'])}}}",
        rf"\newcommand{{\LenVsOldZeroGain}}{{{signed(p['len_a2c_vs_old_a0']['diff'])}}}",
        rf"\newcommand{{\LenVsOldZeroAbs}}{{{abs(p['len_a2c_vs_old_a0']['diff']) * 100:.2f}}}",
        rf"\newcommand{{\LenVsOldZeroLower}}{{{signed(p['len_a2c_vs_old_a0']['bootstrap_lower_95'])}}}",
        rf"\newcommand{{\LenVsOldZeroP}}{{{science(p['len_a2c_vs_old_a0']['mcnemar']['p'])}}}",
        rf"\newcommand{{\LenVsOldZeroPText}}{{p = {science(p['len_a2c_vs_old_a0']['mcnemar']['p']).strip('$')}}}",
        rf"\newcommand{{\LenNativePct}}{{{pct(native['accuracy'])}}}",
        rf"\newcommand{{\LenNativeCorrect}}{{{num(native['correct'])}}}",
        rf"\newcommand{{\LenNativeDiff}}{{{signed(a['len_a2c']['accuracy'] - native['accuracy'])}}}",
        rf"\newcommand{{\LenNativeGapAbs}}{{{abs(a['len_a2c']['accuracy'] - native['accuracy']) * 100:.2f}}}",
        rf"\newcommand{{\LenNextShortLo}}{{{proj['ShortLo']:.1f}}}",
        rf"\newcommand{{\LenNextShortHi}}{{{proj['ShortHi']:.1f}}}",
        rf"\newcommand{{\LenNextLongLo}}{{{proj['LongLo']:.1f}}}",
        rf"\newcommand{{\LenNextLongHi}}{{{proj['LongHi']:.1f}}}",
    ]
    return m


def table(s: dict) -> str:
    a, p = s["arms"], s["paired"]
    jev = s["state"]
    jlo, jhi = wilson(jev["jev_correct"], jev["n"])

    def row(label: str, arm: dict, examples: str, pair: dict | None) -> str:
        lo, hi = arm["wilson95"]
        diff = "---" if pair is None else signed(pair["diff_vs_jev"])
        pv = "---" if pair is None else science(pair["mcnemar"]["p"])
        body = (rf"{label} & {examples} & {num(arm['n'])} & {num(arm['correct'])} & "
                rf"{pct(arm['accuracy'])} & {pct(lo)}--{pct(hi)} & {diff} & {pv}")
        return body + " \\\\"

    rows = [
        row("Old head, message only (control)", a["old_a0"], "0.0", p["old_a0"]),
        row("Old head, examples that fit (control)", a["old_a2c"],
            f"{a['old_a2c']['mean_effective_examples']:.1f}", p["old_a2c"]),
        row("5{,}120 query head, message only", a["len_a0"], "0.0", p["len_a0"]),
        row("5{,}120 query head, all 24 (primary)", a["len_a2c"],
            f"{a['len_a2c']['mean_effective_examples']:.1f}", p["len_a2c"]),
        row("Jev 1.13, cited", {"n": jev["n"], "correct": jev["jev_correct"],
                                "accuracy": jev["jev_accuracy"],
                                "wilson95": (jlo, jhi)}, "24", None),
    ]
    return r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{The 5{,}120-byte query head on all 3{,}080 official Banking77 rows.
Following amendment 1 to the pre-registration, the encoder starts from the
1{,}024-position 113.47M checkpoint and is fine-tuned jointly with the query
head for \LenSteps{} steps at sequence 5{,}120 and batch \LenBatch{}
(\LenHeadHours{} GPU-hours); the separate masked-LM length adaptation was not
run. The head uses the same 4{,}000 rows and the exact stored 24 BM25 example
ids as the native-window head. ``Examples'' is the mean number of complete lines
in the 5{,}118-byte context budget. The old head and its native 1{,}020-byte
packing are unchanged controls, reused from the native-window evaluation with
their per-example context hashes checked. The P0 bar is Jev's \LenJevPct\%.
Wilson intervals are 95\%; McNemar is exact and two-sided; the bootstrap lower
bound is the 5th percentile of 10{,}000 paired resamples, seed 0. Read from
\texttt{results/jev\_query\_5120/eval/summary.json}.}
\label{tab:len5120}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Arm & Ex. & $n$ & Correct & Acc. & 95\% Wilson & $\Delta$ vs Jev & McNemar $p$ \\
 & & & & (\%) & & (pp) & \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    summary, meta = load()
    if args.check:
        print(f"checked {len(summary['arms'])} arms; verdict {summary['decision']['verdict']}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "len5120.tex").write_text(table(summary), encoding="utf-8")
    (OUT / "macros_len5120.tex").write_text(
        "% generated by thesis/gen_query_5120.py — do not edit by hand\n"
        + "\n".join(macros(summary, meta)) + "\n", encoding="utf-8")
    print(f"wrote len5120.tex and macros_len5120.tex to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
