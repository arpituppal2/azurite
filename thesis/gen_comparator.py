#!/usr/bin/env python3
"""Generate the comparator tables and macros from result files.

    thesis/tables/comparator.tex   decode-vs-score, per task, averaged over seeds
    thesis/tables/cdn.tex          billed tokens vs repetition on a live CDN
    thesis/tables/macros_cmp.tex   the numbers the prose quotes

Separate from gen_tables.py on purpose: that script reads the training ladder,
this one reads the comparator, and a build that has not run a comparator leaves
these fragments at `pending` rather than failing.

    python thesis/gen_comparator.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.stats import wilson  # noqa: E402

OUT = ROOT / "thesis" / "tables"
CMP = ROOT / "results" / "decode_vs_score"
CDN = ROOT / "results" / "comparison" / "cdn_probe.json"
PENDING = r"\emph{pending}"
MIN_STEPS = 100  # smoke runs write 10-step files; they must never be quoted


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def runs() -> dict:
    out, strat = {}, {}
    if not CMP.exists():
        return out
    # Two draw protocols live here: the original sorted-prefix draw and the
    # class-balanced `strat` draw that replaces it for Banking77. A task with
    # any stratified record uses only those; mixing them would average an
    # 8-class task into a 77-class one.
    seen: set = set()
    for pattern in ("*_tiny_byte_s*.json", "*_tiny_byte_strat*_s*.json"):
        for p in sorted(CMP.glob(pattern)):
            # `*_tiny_byte_s*.json` also matches `_tiny_byte_strat1600_s0`, so
            # the two patterns overlap; counting a record twice would double
            # every pooled n in this chapter.
            if p in seen:
                continue
            seen.add(p)
            d = load(p)
            if not d or d.get("steps", 0) < MIN_STEPS:
                continue
            bucket = strat if d.get("stratified") else out
            bucket.setdefault(d["task"], []).append(d)
    out.update(strat)
    return out


def stats() -> dict:
    per_task = {}
    agg = {"score": [], "decode": [], "invalid": [], "lat_score": [], "lat_decode": [],
           "params": None, "bytes": None, "n": 0}
    for task, rs in sorted(runs().items()):
        # The score arm runs on the full test split and the decode arm on a
        # 300-row subset of the same draw. Average each over the records that
        # actually measured it, rather than mixing test sizes in one mean.
        n_max = max((r.get("test_examples") or 0) for r in rs)
        rows_max = max((r.get("train_examples") or 0) for r in rs)
        # The score arm is quoted at the largest fine-tune budget measured
        # (that is the arm a deployment would ship); decode is quoted wherever
        # it was measured, because its failure is the point of that row.
        score_rs = [r for r in rs
                    if (r.get("test_examples") or 0) == n_max
                    and (r.get("train_examples") or 0) == rows_max]
        dec_rs = [r for r in rs if r.get("decode_full")]
        # Pool the decisions when the records carry them, so the interval is
        # computed on 3,000 rows rather than on the mean of three 1,000-row runs.
        pooled = [r for r in score_rs if r["score"].get("correct")]
        if pooled:
            k_tot = sum(sum(r["score"]["correct"]) for r in pooled)
            n_tot = sum(len(r["score"]["correct"]) for r in pooled)
            score_val = k_tot / n_tot
        else:
            k_tot, n_tot = None, None
            score_val = mean([r["score"]["accuracy"] for r in score_rs])
        v = {"n": len(score_rs), "k": rs[0]["k"],
             "score": score_val, "score_k": k_tot, "score_n": n_tot,
             "decode": mean([r["decode_full"]["accuracy"] for r in dec_rs]),
             "invalid": mean([r["decode_full"]["invalid_rate"] for r in dec_rs]),
             "lat_score": mean([r["score"]["latency_p50_ms"] for r in score_rs]),
             "lat_decode": mean([r["decode_full"]["latency_p50_ms"] for r in dec_rs])}
        per_task[task] = v
        for k in ("score", "decode", "invalid", "lat_score", "lat_decode"):
            agg[k].append(v[k])
        # score runs on the full stratified test set, decode on a 300-row
        # subset; the table's interval belongs to the accuracy it quotes, so
        # take the largest test set rather than an average across arms.
        v["n_rows"] = n_tot if n_tot else max(r["score"].get("n") or 0 for r in score_rs)
        agg["n_rows"] = v["n_rows"]
        # Encoder params (decode arm) are the size axis; the score arm's count
        # includes the classifier head and would overstate it slightly.
        agg["params"] = ((rs[0].get("decode_full") or {}).get("params")
                         or rs[0]["score"].get("params") or agg["params"])
        agg["bytes"] = ((rs[0].get("decode_full") or {}).get("checkpoint_bytes")
                        or rs[0]["score"].get("checkpoint_bytes") or agg["bytes"])
        agg["n"] += len(rs)
    for k in ("score", "decode", "invalid", "lat_score", "lat_decode"):
        agg[k] = mean(agg[k])
    return {"tasks": per_task, **agg}


def cdn_stats():
    d = load(CDN)
    if not d:
        return None
    cache = d.get("cache") or {}
    rows = [r for r in d.get("rows", []) if r.get("condition") == "repeat"]
    # The cold number is the FIRST billed call, which lives in the r=1 row. The
    # cache.latency_s list holds the warm repeats, so taking its max as "cold"
    # understates the speed-up by an order of magnitude.
    cold = next((r.get("seconds") for r in rows if r.get("r") == 1), None)
    lat = [x for x in (cache.get("latency_s") or []) if x is not None]
    return {"rows": rows,
            "fit": d.get("fit") or {},
            "tpp": d.get("tokens_per_passage_implied"),
            "compressed": bool(d.get("compression_detected")),
            "lat_warm": statistics.median(lat) if lat else None,
            "lat_cold": cold,
            "bills": sorted(set(cache.get("distinct_prompt_token_counts") or []))}


def macros() -> list:
    m = []
    s = stats()
    if s["n"]:
        m.append(rf"\newcommand{{\CmpSeeds}}{{{s['n']}}}")
        m.append(rf"\newcommand{{\CmpTasks}}{{{len(s['tasks'])}}}")
        for name, key, fmt in (("ScoreAcc", "score", "{:.4f}"), ("DecodeAcc", "decode", "{:.4f}"),
                               ("DecodeInvalid", "invalid", "{:.3f}"),
                               ("LatScore", "lat_score", "{:.1f}"), ("LatDecode", "lat_decode", "{:.1f}")):
            val = s[key]
            m.append(rf"\newcommand{{\Cmp{name}}}{{{fmt.format(val) if val is not None else PENDING}}}")
        m.append(rf"\newcommand{{\CmpLatRatio}}{{{s['lat_decode'] / s['lat_score']:.1f}}}"
                 if s["lat_score"] and s["lat_decode"] else rf"\newcommand{{\CmpLatRatio}}{{{PENDING}}}")
        if s["params"]:
            m.append(rf"\newcommand{{\CmpParams}}{{{s['params']:,}}}".replace(",", "{,}"))
        if s["bytes"]:
            m.append(rf"\newcommand{{\CmpMB}}{{{s['bytes'] / 1e6:.1f}}}")
        # Break-even hourly rate: at what GPU price does 1k local decisions cost
        # what Jev charges for the same payload? Rate-free by construction, so
        # the verdict does not depend on a price this repo never recorded.
        if s["lat_score"]:
            hours = 1000 * s["lat_score"] / 1000 / 3600
            for name, toks in (("BreakEvenRate", 900), ("BreakEvenHigh", 1600)):
                m.append(rf"\newcommand{{\{name}}}{{{toks * 0.042 / 1e6 * 1000 / hours:.1f}}}")
    else:
        for k in ("Seeds", "Tasks", "ScoreAcc", "DecodeAcc", "DecodeInvalid",
                  "LatScore", "LatDecode", "LatRatio", "Params", "MB"):
            m.append(rf"\newcommand{{\Cmp{k}}}{{{PENDING}}}")
        for k in ("BreakEvenRate", "BreakEvenHigh"):
            m.append(rf"\newcommand{{\{k}}}{{{PENDING}}}")
    c = cdn_stats()
    if c and c["fit"].get("tokens_per_repetition") is not None:
        f = c["fit"]
        m.append(rf"\newcommand{{\CdnSlope}}{{{f['tokens_per_repetition']:.1f}}}")
        m.append(rf"\newcommand{{\CdnRTwo}}{{{f['r2']:.4f}}}")
        m.append(rf"\newcommand{{\CdnPassageTokens}}{{{c['tpp']:.0f}}}")
        if c["lat_cold"] and c["lat_warm"]:
            m.append(rf"\newcommand{{\CdnLatCold}}{{{c['lat_cold']:.1f}}}")
            m.append(rf"\newcommand{{\CdnLatWarm}}{{{c['lat_warm']:.2f}}}")
            m.append(rf"\newcommand{{\CdnLatSpeedup}}{{{c['lat_cold'] / max(c['lat_warm'], 1e-6):.0f}}}")
        else:
            for k in ("LatCold", "LatWarm", "LatSpeedup"):
                m.append(rf"\newcommand{{\Cdn{k}}}{{{PENDING}}}")
        if c["bills"]:
            m.append(rf"\newcommand{{\CdnBillTokens}}{{{c['bills'][0]:,}}}".replace(",", "{,}"))
        m.append(rf"\newcommand{{\CdnVerdict}}{{{'compressed' if c['compressed'] else 'per occurrence'}}}")
    else:
        for k in ("Slope", "RTwo", "PassageTokens", "LatCold", "LatWarm", "LatSpeedup",
                  "BillTokens", "Verdict"):
            m.append(rf"\newcommand{{\Cdn{k}}}{{{PENDING}}}")
    return m


PREFIX = {"banking77": "Bank", "ag_news": "News"}


def task_macros(s: dict) -> list:
    """Per-task macros. Aggregates across tasks with different K are meaningless:
    a 0.50 on 4-way and a 0.20 on 77-way do not average into anything a reader
    can interpret, so the prose quotes tasks by name."""
    m = []
    for task, v in s["tasks"].items():
        p = PREFIX.get(task, task.title().replace("_", ""))
        m += [
            rf"\newcommand{{\Cmp{p}Seeds}}{{{v['n']}}}",
            rf"\newcommand{{\Cmp{p}K}}{{{v['k']}}}",
            rf"\newcommand{{\Cmp{p}Score}}{{{v['score']:.4f}}}",
            # percent form for prose that quotes an accuracy next to a cited one
            rf"\newcommand{{\Cmp{p}ScorePct}}{{{100 * v['score']:.1f}}}",
            # A task with score-only records has no decode arm yet; printing
            # `pending` is honest, formatting None is a crash.
            (rf"\newcommand{{\Cmp{p}Decode}}{{{v['decode']:.4f}}}"
             if v["decode"] is not None else
             rf"\newcommand{{\Cmp{p}Decode}}{{{PENDING}}}"),
            (rf"\newcommand{{\Cmp{p}DecodePct}}{{{100 * v['decode']:.1f}}}"
             if v["decode"] is not None else
             rf"\newcommand{{\Cmp{p}DecodePct}}{{{PENDING}}}"),
            (rf"\newcommand{{\Cmp{p}Invalid}}{{{v['invalid']:.3f}}}"
             if v["invalid"] is not None else
             rf"\newcommand{{\Cmp{p}Invalid}}{{{PENDING}}}"),
            rf"\newcommand{{\Cmp{p}LatScore}}{{{v['lat_score']:.1f}}}",
            (rf"\newcommand{{\Cmp{p}LatDecode}}{{{v['lat_decode']:.1f}}}"
             if v["lat_decode"] is not None else
             rf"\newcommand{{\Cmp{p}LatDecode}}{{{PENDING}}}"),
        ]
        # 1,000 decisions x p50 ms = p50 GPU-seconds per 1,000 decisions.
        sec_per_1k = 1000 * v["lat_score"] / 1000
        m.append(rf"\newcommand{{\Cmp{p}GpuSec}}{{{sec_per_1k:.1f}}}")
        n = v.get("n_rows") or 0
        k = v.get("score_k")
        lo, hi = wilson(int(k) if k is not None else int(round(v["score"] * n)), int(n))
        m += [
            rf"\newcommand{{\Cmp{p}Rows}}{{{int(n)}}}",
            rf"\newcommand{{\Cmp{p}ScoreLo}}{{{lo:.3f}}}",
            rf"\newcommand{{\Cmp{p}ScoreHi}}{{{hi:.3f}}}",
        ]
        if task == "banking77":
            gap = 0.788 - v["score"]
            m.append(rf"\newcommand{{\CmpBankGap}}{{{gap * 100:.1f}}}")
            # The verdict is computed, not chosen: a gap inside the intervals is
            # noise, a gap that clears them is a deficit, and the middle case is
            # reported as unresolved rather than argued either way.
            if lo > 0.836:
                verdict = "exceeds the cited interval"
            elif hi < 0.731:
                verdict = "falls below the cited interval"
            elif v["score"] >= 0.788:
                verdict = "overlaps the cited interval; no win demonstrated"
            else:
                verdict = "overlaps the cited interval; below the point estimate"
            m.append(rf"\newcommand{{\CmpBankVerdict}}{{{verdict}}}")
        if v["lat_score"]:
            hours = sec_per_1k / 3600
            # Billed tokens per decision as the third party measured them, so the
            # price list times their count must reproduce their invoice.
            toks = {"banking77": 952, "ag_news": 360}.get(task, 900)
            m.append(rf"\newcommand{{\Cmp{p}JevTokens}}{{{toks}}}")
            m.append(rf"\newcommand{{\Cmp{p}JevCost}}{{{toks * 0.042 / 1e6 * 1000:.4f}}}")
            m.append(rf"\newcommand{{\Cmp{p}BreakEven}}{{{toks * 0.042 / 1e6 * 1000 / hours:.1f}}}")
    # missing tasks still need their macros, or the build dies on an undefined one
    for task, p in PREFIX.items():
        if task in s["tasks"]:
            continue
        for k in ("Seeds", "K", "Score", "Decode", "Invalid", "LatScore", "LatDecode",
                  "GpuSec", "BreakEven", "BreakEvenHigh", "Rows", "ScoreLo", "ScoreHi"):
            m.append(rf"\newcommand{{\Cmp{p}{k}}}{{{PENDING}}}")
        if p == "Bank":
            for k in ("Gap", "Verdict"):
                m.append(rf"\newcommand{{\Cmp{p}{k}}}{{{PENDING}}}")
    return m


def comparator_table() -> str:
    s = stats()
    rows = []
    for task, v in s["tasks"].items():
        f = lambda x, fmt="{:.4f}": fmt.format(x) if x is not None else PENDING
        sec = (1000 * v["lat_score"] / 1000) if v["lat_score"] else None
        toks = {"banking77": 952, "ag_news": 360}.get(task, 900)
        be = (toks * 0.042 / 1e6 * 1000 / (sec / 3600)) if sec else None
        rows.append(rf"{task.replace('_', '-')} & {v['k']} & {v['n']} & {f(v['score'])} & "
                    rf"{f(v['decode'])} & {f(v['invalid'], '{:.2f}')} & "
                    rf"{f(v['lat_score'], '{:.1f}')} & {f(v['lat_decode'], '{:.1f}')} & "
                    rf"{f(sec, '{:.1f}')} & {f(be, '{:.1f}')} \\")
    if not rows:
        rows = [rf"{PENDING} & & & & & & & & & \\"]
    return r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{Decode versus score with everything else fixed: same encoder, same
initialisation, same rows in the same order, same steps, batch, learning rate
and seed. \emph{Score} evaluates every candidate in one pass and cannot emit an
out-of-bank label; \emph{decode} fills the label's byte slots iteratively and is
matched against the bank, so it can. Latency is batch-1 wall clock on CPU over
the same test rows, and therefore a lower bound on what an autoregressive
decoder of the same label length would cost. GPU seconds per 1,000 decisions
come from $1000\times p_{50}$ and are CPU seconds here, so they overstate an
accelerator. The last column is the GPU rate at which 1,000 local decisions
cost what Jev charges for that task's billed payload at \$0.042/MTok; it is rate-free
because no price for this machine is recorded anywhere in the repository. Read
from \texttt{results/decode\_vs\_score/*.json}.}
\label{tab:comparator}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrrr}
\toprule
Task & $K$ & Seats & Score acc. & Decode acc. & Invalid & $p_{50}$ score & $p_{50}$ decode & GPU s & break-even \\
 & & & & & rate & (ms) & (ms) & per 1k & \$/GPU-h \\
\midrule
""" + "\n".join(rows) + "\n" + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def cdn_table() -> str:
    c = cdn_stats()
    rows = []
    if c:
        for r in sorted(c["rows"], key=lambda x: x.get("r") or 0):
            tok, secs = r.get("prompt_tokens"), r.get("seconds")
            bill = f"{tok:,}" if tok else PENDING + r" (quota)"
            rows.append(rf"{r.get('r')} & {r.get('chars', 0):,} & {bill} & "
                        rf"{f'{secs:.2f}' if secs is not None else PENDING} \\")
    if not rows:
        rows = [rf"{PENDING} & & & \\"]
    fit = ""
    if c and c["fit"].get("tokens_per_repetition") is not None:
        f = c["fit"]
        fit = (rf"Fit: $\mathrm{{billed}} = {f['intercept']:.0f} + "
               rf"{f['tokens_per_repetition']:.1f}\,r$, $R^2={f['r2']:.4f}$; one passage is "
               rf"{c['tpp']:.0f} tokens, so every repetition is billed in full.")
    return r"""\begin{table}[htbp]
\centering
\caption{Billed prompt tokens against repetition of one passage on a live
token-priced CDN, read from the provider's own \texttt{usage} field --- the
exact quantity a per-million-token price multiplies. If a CDN compressed
variable-length sequences, the slope would be below one passage.}
\label{tab:cdn}
\begin{tabular}{rrrr}
\toprule
Repetitions $r$ & Characters & Billed prompt tokens & Seconds \\
\midrule
""" + "\n".join(rows) + "\n" + r"""
\bottomrule
\end{tabular}
\\[0.4em]
""" + fit + "\n" + r"""
\end{table}
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparator.tex").write_text(comparator_table(), encoding="utf-8")
    (OUT / "cdn.tex").write_text(cdn_table(), encoding="utf-8")
    s = stats()
    (OUT / "macros_cmp.tex").write_text(
        "% generated by thesis/gen_comparator.py — do not edit by hand\n"
        + "\n".join(macros() + task_macros(s)) + "\n", encoding="utf-8")
    print(f"comparator: {s['n']} seed-runs over {len(s['tasks'])} tasks "
          f"(score {s['score']}, decode {s['decode']})")
    print(f"wrote comparator.tex, cdn.tex, macros_cmp.tex to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
