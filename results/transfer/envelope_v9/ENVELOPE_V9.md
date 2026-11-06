# Envelope v9 — verdict: BELOW, decisively (executed 2026-11-04, RTX 6000)

## Pre-registration compliance

4 cells ran the locked v9 recipe (`PREREG_V9.md` `56dbf7aa…`): native-1024
mid anchor (`a5ed3ed8…`, summary + independent re-hash matched), native 1024
window, A (M8/378) + B (M16/756), seeds 0+1. No deviation.

## Results (misses as errors, Wilson 95% LB vs recorded 0.797, n=500)

| cell | recall test | misses /500 | retrieved-only | end-to-end | Wilson LB |
|---|---|---|---|---|---|
| A_s0 | 0.316 | 342 | 0.2152 | 0.0680 | 0.0491 |
| A_s1 | 0.390 | 305 | 0.2667 | 0.1040 | 0.0802 |
| B_s0 | 0.522 | 239 | 0.1686 | 0.0880 | 0.0662 |
| B_s1 | 0.850 | 75 | 0.2800 | 0.2380 | 0.2028 |

## Verdict: BELOW — the native-1024 anchor is transfer-dead

Phase-A dual test accuracy sat at 0.02-0.07 (chance ≈ 0.013) across all 378
steps with flat loss — the encoder learns nothing in transfer, while its
held-out LM (1.0686 bpb) looks merely mediocre. Rank diagnostic (32-sentence
probe, mean-pooled last layer): all anchors are anisotropic (eff_rank ~7),
but mid-1024's top singular values are 2-3× larger (23.0 vs 9.6) with
mean cosine 0.86 vs 0.78 — a sharper collapse signature. The seq-1024 run
with tiny-run hparams (lr 6e-4, warmup 200) produced a transfer-dead
anchor. This is a finding, not a failure to report: long-context training
needs its own hparams.

## Artifacts (mirrored byte-identical, all 8 hashes match)

- `A_s0.json` `40bb9028…`, `A_s1.json` `97b7f709…`,
  `B_s0.json` `095b7ec5…`, `B_s1.json` `53d9431c…`
- `ev_*/provenance.json` (`ac575b06…`, `a4471cec…`, `b053b7fe…`,
  `fc105f34…`), cell logs.

## Next (already running)

Retrain mid-1024 with lr 3e-4 + warmup 500 (seed 1, same 100M tokens).
If its rank signature normalizes AND a transfer smoke learns → v10.
If it also dies → mid-rung transfer needs a different recipe (or rung),
and banking77 closes at small-rung 0.804 (LB 0.767) vs 0.797.

## Addendum — hparam-fix retrain also transfer-dead (close-out)
- Retrain mid-1024 (lr 3e-4, warmup 500, seed 1, same 100M tokens, ~15 min):
  ckpt `751fe91e…` (host-side), held-out 1.0622 bpb. Rank signature improved
  (top singular 17.4 vs 23.0) but mean cosine WORSE (0.902 vs 0.86).
- Transfer smoke (M8, 150+150 steps): recall 0.222, e2e 0.034 — DEAD,
  worse than v1's 0.32-0.39. Lower LR does not fix it. A third guess
  (lr 1.5e-4, …) would be flailing, not science — STOPPING the mid-1024 line.
- Honest ledger for banking77 funnel vs 0.797: tiny 0.68 → small 0.745 →
  small M16/756 0.804 (LB 0.767, measured ceiling) → mid+RoPE 0.77 →
  mid-native transfer-dead. The gap (LB 0.767 vs 0.797) needs a new idea
  (different head, different rung with native window, or different task),
  not more budget on known-dead paths.
