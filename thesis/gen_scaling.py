#!/usr/bin/env python3
"""Generate the row-scaling and risk-coverage tables for the comparator.

    thesis/tables/scaling.tex       1.6k vs 8k fine-tune rows, per rung
    thesis/tables/risk_coverage.tex accuracy on the test-set confidence ranking
    thesis/tables/macros_scaling.tex the numbers the prose quotes

Reads the score-only Banking77 77-way records written by
`scripts/decode_vs_score.py --arms score --stratify --tag strat{1600,8000}`. Those records
carry the scorer's own per-row confidence and correctness, which is what makes
the risk-coverage curve a measurement rather than a second training run.

The parity verdict is computed from the pooled Wilson bounds against the cited
78.8% (interval 73.1-83.6). Nothing here is transcribed by hand: if a run is
missing the fragment says `pending`, and if the interval moves the sentence
moves with it.

    python thesis/gen_scaling.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.stats import wilson  # noqa: E402

OUT = ROOT / "thesis" / "tables"
DVS = ROOT / "results" / "decode_vs_score"
PENDING = r"\emph{pending}"

JEV_POINT = 0.788
JEV_LO, JEV_HI = 0.731, 0.836

ARMS = [("tiny", 1600), ("tiny", 8000), ("small", 1600), ("small", 8000)]
COVERAGES = (0.5, 0.7, 0.9, 1.0)


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def runs(rung: str, rows: int) -> list[dict]:
    out = []
    # `strat` tags are the class-balanced draws; the pre-fix `rows` records are
    # quarantined in _sortedprefix_bug/ and must never be picked up here.
    for p in sorted(DVS.glob(f"banking77_{rung}_byte_strat{rows}_s*.json")):
        d = load(p)
        if d and d.get("steps", 0) >= 100:
            out.append(d)
    return out


def z_two_prop(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p for a difference in proportions. Seeds are independent draws,
    so this is the honest test for 8k vs 1.6k; a paired test would need the same
    test rows scored by both arms, which the run protocol deliberately does not
    do (each seed is a fresh init)."""
    if not (n1 and n2):
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p2 - p1) / se
    # normal CDF via erf
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def pooled(rs: list[dict]) -> dict | None:
    if not rs:
        return None
    k = n = 0
    accs = []
    conf: list[float] = []
    hit: list[int] = []
    lat = []
    for r in rs:
        s = r["score"]
        # `correct` is authoritative; accuracy alone cannot be pooled across seeds.
        if "correct" in s:
            k += sum(s["correct"])
            n += len(s["correct"])
            conf.extend(s["confidence"])
            hit.extend(s["correct"])
        else:
            k += round(s["accuracy"] * s["n"])
            n += s["n"]
        accs.append(s["accuracy"])
        lat.append(s.get("latency_p50_ms"))
    lo, hi = wilson(k, n)
    order = sorted(range(len(hit)), key=lambda i: -conf[i]) if hit else []
    curve = {}
    for c in COVERAGES:
        m = max(1, int(round(c * len(order))))
        curve[c] = (sum(hit[i] for i in order[:m]) / m) if order else float("nan")
    return {"seeds": len(rs), "k": k, "n": n, "acc": k / n if n else float("nan"),
            "mean_seed_acc": sum(accs) / len(accs), "lo": lo, "hi": hi,
            "curve": curve,
            "lat": sorted(x for x in lat if x)[len(lat) // 2] if lat else None,
            "params": rs[0]["score"].get("params"),
            "bytes": rs[0]["score"].get("checkpoint_bytes"),
            "test_n": rs[0].get("test_examples")}


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {a: pooled(runs(*a)) for a in ARMS}

    # ---------------- scaling table
    rows = []
    for rung, base, wide in (("tiny", ("tiny", 1600), ("tiny", 8000)),
                             ("small", ("small", 1600), ("small", 8000))):
        b, w = data[base], data[wide]
        if not b:
            rows.append(rf"{rung} & {PENDING} & {PENDING} & {PENDING} & {PENDING} & {PENDING} \\")
            continue
        if not w:
            # show the measured base rather than hiding it behind a pending wide cell
            rows.append(rf"{rung} & {b['k']}/{b['n']} & {fmt_pct(b['acc'])} & "
                        rf"{PENDING} & {PENDING} & {PENDING} \\")
            continue
        p = z_two_prop(b["k"], b["n"], w["k"], w["n"])
        rows.append(
            rf"{rung} & {b['k']}/{b['n']} & {fmt_pct(b['acc'])} & "
            rf"\textbf{{{fmt_pct(w['acc'])}}} & {100 * (w['acc'] - b['acc']):+.1f} pt & "
            rf"$p={p:.3f}$ \\"
        )
    body = r"""\begin{table}[htbp]
\centering
\caption{Fine-tune rows at the Banking77 77-way interface, exhaustive scoring,
same pretrained checkpoint, order, steps-per-example, batch, learning rate and
seeds. The two columns differ only in how many distinct training rows the five
epochs cover. Generated by \texttt{thesis/gen\_scaling.py}.}
\label{tab:rows}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}
\toprule
Rung & Correct & 1.6k rows & 8k rows & $\Delta$ & Test \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "scaling.tex").write_text(body)

    # ---------------- risk-coverage table
    cols = [a for a in ARMS if data[a]]
    head = " & ".join(
        rf"\shortstack{{{a[0]}\\{a[1] // 1000}k}}" for a in cols) if cols else PENDING
    lines = []
    for c in COVERAGES:
        vals = " & ".join(fmt_pct(data[a]["curve"][c]) for a in cols) if cols else PENDING
        lines.append(rf"{int(c * 100)}\% & {vals} \\")
    rc_body = r"""\begin{table}[htbp]
\centering
\caption{Accuracy at coverage: the scorer ranked by its own maximum softmax
probability on this test set. Read down a column and accuracy rises as coverage
falls; read across and the row-scaling gain is visible. This is a test-set
ranking curve, not a deployment abstention guarantee. Pooled over three seeds
and the 1000-row test split.}
\label{tab:risk-coverage}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l""" + "r" * len(cols) + r"""}
\toprule
Coverage & """ + head + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "risk_coverage.tex").write_text(rc_body)

    # ---------------- macros
    m = ["% generated by thesis/gen_scaling.py — do not edit by hand"]
    for key, (rung, rows_) in {
        "TinyBase": ("tiny", 1600), "TinyWide": ("tiny", 8000),
        "SmallBase": ("small", 1600), "SmallWide": ("small", 8000),
    }.items():
        d = data[(rung, rows_)]
        if not d:
            for suf in ("Acc", "Lo", "Hi", "Seeds", "Rows", "Lat"):
                m.append(rf"\newcommand{{\Row{key}{suf}}}{{{PENDING}}}")
            continue
        m.append(rf"\newcommand{{\Row{key}Acc}}{{{100 * d['acc']:.1f}}}")
        m.append(rf"\newcommand{{\Row{key}Lo}}{{{100 * d['lo']:.1f}}}")
        m.append(rf"\newcommand{{\Row{key}Hi}}{{{100 * d['hi']:.1f}}}")
        m.append(rf"\newcommand{{\Row{key}Seeds}}{{{d['seeds']}}}")
        m.append(rf"\newcommand{{\Row{key}Rows}}{{{d['n']}}}")
        m.append(rf"\newcommand{{\Row{key}Lat}}{{{d['lat']:.1f}}}")

    # the arm the argument rests on: best measured accuracy, ties to the smaller rung
    best = max((a for a in data if data[a]), key=lambda a: (data[a]["acc"], -data[a]["params"]),
               default=None)
    if best:
        d = data[best]
        m.append(rf"\newcommand{{\ShipRung}}{{{best[0]}}}")
        m.append(rf"\newcommand{{\ShipRows}}{{{best[1]}}}")
        m.append(rf"\newcommand{{\ShipAcc}}{{{100 * d['acc']:.1f}}}")
        m.append(rf"\newcommand{{\ShipLo}}{{{100 * d['lo']:.1f}}}")
        m.append(rf"\newcommand{{\ShipHi}}{{{100 * d['hi']:.1f}}}")
        m.append(rf"\newcommand{{\ShipParams}}{{{d['params']:,}}}".replace(",", "{,}"))
        m.append(rf"\newcommand{{\ShipMB}}{{{d['bytes'] / 1e6:.1f}}}")
        m.append(rf"\newcommand{{\ShipP}}{{{d['lat']:.1f}}}")
        m.append(rf"\newcommand{{\ShipTestN}}{{{d['test_n']}}}")
        # one thousand batch-1 decisions at p50 occupy p50 seconds of the machine,
        # so the break-even rate is 144/p50 dollars per GPU-hour at the cited $0.04.
        m.append(rf"\newcommand{{\ShipGpuSec}}{{{d['lat']:.1f}}}")
        m.append(rf"\newcommand{{\ShipBreakEven}}{{{144.0 / d['lat']:.1f}}}")
        above = d["acc"] >= JEV_POINT
        if d["lo"] > JEV_HI:
            verdict = "lies above the cited interval"
        elif d["hi"] < JEV_LO:
            verdict = "lies below the cited interval"
        elif above:
            verdict = ("overlaps the cited interval; no win demonstrated")
        else:
            verdict = ("overlaps the cited interval; below it on the point estimate")
        m.append(rf"\newcommand{{\ShipVerdict}}{{{verdict}}}")
        gap = abs(JEV_POINT - d["acc"]) * 100
        m.append(rf"\newcommand{{\ShipGap}}{{{gap:.1f}}}")
        # signed gap, so a win is not silently printed as a loss
        m.append(rf"\newcommand{{\ShipSignedGap}}{{{(d['acc'] - JEV_POINT) * 100:+.1f}}}")
        if d["acc"] >= JEV_POINT:
            gapsent = (rf"It sits {gap:.1f} points above the cited point estimate, but the "
                       r"intervals overlap, so this is not a demonstrated win; the "
                       r"cross-harness caveat below governs what it can be called")
        else:
            gapsent = (rf"It sits {gap:.1f} points below the cited point estimate, and that "
                       r"shortfall is a result rather than a rounding error")
        m.append(rf"\newcommand{{\ShipGapSentence}}{{{gapsent}}}")
        m.append(rf"\newcommand{{\ShipAtJev}}{{{'yes' if d['hi'] >= JEV_POINT else 'no'}}}")
        # One computed headline, so the prose cannot quietly soften a loss.
        if d["lo"] > JEV_HI:
            head = (rf"Azurite lies above the cited Jev interval: {100 * d['acc']:.1f}\% "
                    rf"(95\% Wilson {100 * d['lo']:.1f}--{100 * d['hi']:.1f}) against 78.8\%")
        elif d["hi"] < JEV_LO:
            head = (rf"Azurite lies below the cited Jev interval: {100 * d['acc']:.1f}\% "
                    rf"(95\% Wilson {100 * d['lo']:.1f}--{100 * d['hi']:.1f}) against 78.8\%")
        elif d["acc"] >= JEV_POINT:
            head = (rf"Azurite is above the cited Jev point estimate but its interval "
                    rf"overlaps the cited one: {100 * d['acc']:.1f}\% "
                    rf"(95\% Wilson {100 * d['lo']:.1f}--{100 * d['hi']:.1f}) against "
                    rf"78.8\% (73.1--83.6); this is not a demonstrated win")
        else:
            head = (rf"Azurite overlaps the cited Jev interval from below: "
                    rf"{100 * d['acc']:.1f}\% (95\% Wilson {100 * d['lo']:.1f}--"
                    rf"{100 * d['hi']:.1f}) against 78.8\% (73.1--83.6)")
        m.append(rf"\newcommand{{\ShipHeadline}}{{{head}}}")
        if d["acc"] >= JEV_POINT:
            fut = (r"The remaining work is a paired harness on the same rows, which "
                   r"this comparison cannot provide")
        else:
            fut = (r"Closing the gap is a data and scale question, and it leads the "
                   r"future work rather than being argued around here")
        m.append(rf"\newcommand{{\ShipFuture}}{{{fut}}}")
        m.append(rf"\newcommand{{\ShipAtNinety}}{{{100 * d['curve'][0.9]:.1f}}}")
        m.append(rf"\newcommand{{\ShipAtFifty}}{{{100 * d['curve'][0.5]:.1f}}}")
    else:
        for k in ("Rung", "Rows", "Acc", "Lo", "Hi", "Params", "MB", "P", "Verdict",
                  "Gap", "SignedGap", "AtJev", "TestN", "GpuSec", "BreakEven",
                  "AtNinety", "AtFifty", "Headline", "Future",
                  "GapSentence", "RungBase", "RungTiny", "RungGainWide"):
            m.append(rf"\newcommand{{\Ship{k}}}{{{PENDING}}}")

    # data lever: the gain from 8k rows, as a number rather than an adjective
    tb, tw = data[("tiny", 1600)], data[("tiny", 8000)]
    sb, sw = data[("small", 1600)], data[("small", 8000)]
    if tb and tw:
        m.append(rf"\newcommand{{\RowTinyGain}}{{{100 * (tw['acc'] - tb['acc']):+.1f}}}")
    else:
        m.append(rf"\newcommand{{\RowTinyGain}}{{{PENDING}}}")
    if sb and sw:
        m.append(rf"\newcommand{{\RowSmallGain}}{{{100 * (sw['acc'] - sb['acc']):+.1f}}}")
    else:
        m.append(rf"\newcommand{{\RowSmallGain}}{{{PENDING}}}")
    # The rung lever is measured at matched rows: 1,600 is where both rungs have
    # landed, and quoting an 8,000-row small arm before it exists would be a
    # projection. A second macro carries the 8,000-row contrast if it lands.
    if sb and tb:
        m.append(rf"\newcommand{{\RowRungGain}}{{{100 * (sb['acc'] - tb['acc']):+.1f}}}")
        m.append(rf"\newcommand{{\RowRungBase}}{{{100 * sb['acc']:.1f}}}")
        m.append(rf"\newcommand{{\RowRungTiny}}{{{100 * tb['acc']:.1f}}}")
    else:
        m.append(rf"\newcommand{{\RowRungGain}}{{{PENDING}}}")
        m.append(rf"\newcommand{{\RowRungBase}}{{{PENDING}}}")
        m.append(rf"\newcommand{{\RowRungTiny}}{{{PENDING}}}")
    if tw and sw:
        m.append(rf"\newcommand{{\RowRungGainWide}}{{{100 * (sw['acc'] - tw['acc']):+.1f}}}")
    else:
        m.append(rf"\newcommand{{\RowRungGainWide}}{{{PENDING}}}")

    (OUT / "macros_scaling.tex").write_text("\n".join(m) + "\n")

    for a in ARMS:
        d = data[a]
        if d:
            print(f"{a[0]:5s} {a[1]:>5} rows  acc={d['acc']:.4f} "
                  f"[{d['lo']:.3f},{d['hi']:.3f}] n={d['n']} "
                  f"p50={d['lat']}ms")
        else:
            print(f"{a[0]:5s} {a[1]:>5} rows  pending")
    print("best arm:", best, "->", (data[best]["acc"] if best else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
