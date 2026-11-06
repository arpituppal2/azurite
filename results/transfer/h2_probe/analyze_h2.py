"""H2 observational probe (CPU-only, zero GPU).

Regenerates batch_schedule(2000,16,80,seed 0) + the stratified train label
order, computes per-step batch class-imbalance, and correlates it against the
residual grad norms stored in both dual-random TRACE files (tanh + notanh).

H2 prediction: at uniform outputs, residual gradient scale tracks batch class
imbalance (balanced batches cancel, imbalanced batches leave a residual).
Support: strong positive Pearson r on plateau steps 10-79 in BOTH traces.
Weaken: r ~ 0 in either trace (with enough imbalance variance to test).
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "scripts"))

from transfer_eval import TASKS, batch_schedule, candidate_bank, stratified_examples

ROOT = HERE.parents[1]
TRACE_A = ROOT / "diag_dual_random_s0" / "TRACE.jsonl"
TRACE_B = ROOT / "diag_dual_notanh_s0" / "TRACE.jsonl"


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def main():
    spec = TASKS["ag_news"]
    bank = candidate_bank(spec)
    train_rows = stratified_examples(spec, bank, "train", 2000, 0)
    labels = [e.gold for e in train_rows]
    assert len(labels) == 2000
    sched = [idx.tolist() for idx in batch_schedule(2000, 16, 80, 0)]

    imb, maxfreq, entropy_def = [], [], []
    for idx in sched:
        counts = [0, 0, 0, 0]
        for i in idx:
            counts[labels[i]] += 1
        freqs = [c / 16 for c in counts]
        imb.append(math.sqrt(sum((f - 0.25) ** 2 for f in freqs)))
        maxfreq.append(max(freqs))
        ent = -sum(f * math.log(f) for f in freqs if f > 0)
        entropy_def.append(math.log(4) - ent)

    out = {
        "schedule": {"n_train": 2000, "batch": 16, "steps": 80, "seed": 0,
                      "label_classes": sorted(set(labels))},
        "imbalance_summary": {
            "mean_l2": sum(imb) / len(imb),
            "min_l2": min(imb), "max_l2": max(imb),
            "n_perfectly_balanced": sum(1 for v in imb if v == 0.0),
            "n_maxfreq_ge_0375": sum(1 for v in maxfreq if v >= 0.375),
        },
        "traces": {},
    }
    for name, path in (("tanh", TRACE_A), ("notanh", TRACE_B)):
        T = [json.loads(line) for line in open(path)]
        assert len(T) == 80, (name, len(T))
        g = [math.sqrt(r["grad_enc"]["grad_norm"] ** 2 + r["grad_head"]["grad_norm"] ** 2)
             for r in T]
        all_r = pearson(imb, g)
        plat = slice(10, 80)
        plat_r = pearson(imb[plat], g[plat])
        out["traces"][name] = {
            "steps": len(T),
            "pearson_all_steps": all_r,
            "pearson_plateau_10_79": plat_r,
            "plateau_grad_mean": sum(g[plat]) / 70,
            "plateau_grad_min": min(g[plat]),
            "plateau_grad_max": max(g[plat]),
            "plateau_imb_mean": sum(imb[plat]) / 70,
        }

    rs = [out["traces"][k]["pearson_plateau_10_79"] for k in ("tanh", "notanh")]
    if all(r is not None and r >= 0.5 for r in rs):
        verdict = "H2 SUPPORTED"
    elif all(r is not None and abs(r) < 0.3 for r in rs):
        verdict = "H2 WEAKENED"
    else:
        verdict = "H2 MIXED"
    out["verdict"] = verdict
    out["rule"] = ("SUPPORTED iff plateau r >= 0.5 in BOTH traces; WEAKENED iff "
                   "|r| < 0.3 in BOTH; else MIXED. Thresholds fixed before computing.")
    (ROOT / "h2_probe" / "H2_VERDICT.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out["traces"][k] for k in out["traces"]}, indent=1))
    print("imbalance:", out["imbalance_summary"])
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
