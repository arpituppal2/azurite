@app.cell
def _():
    import json as _s_json
    from pathlib import Path as _s_Path
    import subprocess as _s_sp

    import marimo as _s_mo

    _s_ws = _s_Path("/marimo/azurite_ws")
    _s_dvs = _s_ws / "results" / "decode_vs_score"
    _s_jev, _s_jev_lo, _s_jev_hi = 0.788, 0.731, 0.836

    def _s_wilson(k, n, z=1.959964):
        if not n:
            return float("nan"), float("nan")
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
        return (c - h) / d, (c + h) / d

    def _s_pool(rung, tag):
        runs = []
        for p in sorted(_s_dvs.glob(f"banking77_{rung}_byte_{tag}_s*.json")):
            d = _s_json.loads(p.read_text())
            if d.get("steps", 0) >= 100 and d["score"].get("correct"):
                runs.append(d)
        if not runs:
            return None
        # decode runs on a 300-row subset; pooling it with the 1,000-row score
        # runs would weight some seeds less than others.
        big = max((r.get("test_examples") or 0) for r in runs)
        runs = [r for r in runs if (r.get("test_examples") or 0) == big]
        k = sum(sum(r["score"]["correct"]) for r in runs)
        n = sum(len(r["score"]["correct"]) for r in runs)
        pairs = [(c, h) for r in runs
                 for c, h in zip(r["score"]["confidence"], r["score"]["correct"])]
        pairs.sort(key=lambda x: -x[0])
        curve = {}
        for cov in (0.5, 0.7, 0.9, 1.0):
            m = max(1, int(round(cov * len(pairs))))
            curve[cov] = sum(h for _, h in pairs[:m]) / m
        hi_cov, run = 0.0, 0
        for j, (_, h) in enumerate(pairs, start=1):
            run += h
            if run / j >= 0.99:
                hi_cov = j / len(pairs)
        lo, hi = _s_wilson(k, n)
        return {"runs": len(runs), "k": k, "n": n, "acc": k / n,
                "lo": lo, "hi": hi, "curve": curve, "hi_cov": hi_cov,
                "p50": runs[0]["score"]["latency_p50_ms"],
                "params": runs[0]["score"].get("params"),
                "mb": (runs[0]["score"].get("checkpoint_bytes") or 0) / 1e6}

    _s_arms = [(r, t) for r in ("tiny", "small") for t in ("strat1600", "strat8000")]
    _s_data = {(r, t): _s_pool(r, t) for r, t in _s_arms}
    _s_live = [k for k, v in _s_data.items() if v]

    _s_lines = [
        "### Where the accuracy is: rows, rung, and coverage",
        "",
        "Class-balanced draw across all 77 labels (`--stratify`). The shipped",
        "splits are label-sorted, so a prefix is a few classes: the earlier",
        "`limit=1600` fine-tune had seen 13 of 77 labels and the 300-row test",
        "set covered 8. Those records are quarantined under",
        "`results/decode_vs_score/_sortedprefix_bug/`.",
        "",
        "| rung | rows | seeds | accuracy | 95% Wilson | 50% cov | 70% cov | 90% cov | 100% cov | p50 ms |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (r, t), v in _s_data.items():
        if not v:
            _s_lines.append(f"| {r} | {t.replace('strat', '')} | _pending_ | | | | | | | |")
            continue
        _s_lines.append(
            f"| {r} | {t.replace('strat', '')} | {v['runs']} | {v['acc']:.3f} | "
            f"{v['lo']:.3f}--{v['hi']:.3f} | {v['curve'][0.5]:.3f} | {v['curve'][0.7]:.3f} | "
            f"{v['curve'][0.9]:.3f} | {v['curve'][1.0]:.3f} | {v['p50']:.1f} |")
    if not _s_live:
        _s_lines.append("| _no stratified run has completed yet_ | | | | | | | | | |")

    _s_best = max((v for v in _s_data.values() if v), key=lambda v: v["acc"], default=None)
    if _s_best:
        gap = (_s_jev - _s_best["acc"]) * 100
        if _s_best["lo"] > _s_jev_hi:
            verdict = "lies above the cited interval"
        elif _s_best["hi"] < _s_jev_lo:
            verdict = "lies below the cited interval"
        elif _s_best["acc"] >= _s_jev:
            verdict = "overlaps the cited interval, above it on the point estimate"
        else:
            verdict = "overlaps the cited interval, below it on the point estimate"
        _s_lines += [
            "",
            f"**Best measured arm:** {_s_best['acc']:.1%} "
            f"({_s_best['lo']:.1%}--{_s_best['hi']:.1%}, n={_s_best['n']}) against the cited "
            f"78.8% (73.1--83.6) on 231 rows from a different harness. Signed difference "
            f"{-gap:+.1f} points; the measured interval **{verdict}**. A win is not claimed: "
            "the intervals overlap and the harnesses differ.",
            "",
            f"**Coverage.** Reaching 99% requires abstaining down to {_s_best['hi_cov']:.0%} "
            f"coverage; at 90% coverage the arm is {_s_best['curve'][0.9]:.1%} accurate, at 50% "
            f"it is {_s_best['curve'][0.5]:.1%}. The product claim is a coverage claim, and this "
            "is the coverage.",
        ]
    else:
        _s_lines.append("")
        _s_lines.append("**Verdict:** no stratified run has completed, so nothing is claimed.")

    _s_st = _s_ws / "results" / "pretrain" / "GPU_QUEUE_STATE.json"
    _s_jobs = _s_json.loads(_s_st.read_text())["jobs"] if _s_st.exists() else {}
    _s_live_jobs = [n for n, j in _s_jobs.items() if j.get("status") in ("running", "starting", "scoring")]
    _s_idle = sum(float(j.get("idle_s") or 0) for j in _s_jobs.values())
    _s_gpu = _s_sp.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    _s_lines += [
        "",
        f"**Card, untouched by this work:** {', '.join(_s_live_jobs) or 'idle'} · "
        f"GPU {_s_gpu or '?'} · idle between jobs {_s_idle:.1f} s. The sweep above runs on CPU.",
    ]
    _s_mo.md("\n".join(_s_lines))
    return
