# Envelope v4 — verdict (executed 2026-11-04 on RTX PRO 6000 Blackwell, ~50 s GPU)

## Pre-registration compliance

- Cells ran EXACTLY the locked recipe (`PREREG_V4.md` `c63c1f94…`): banking77,
  anchor ckpt verified on-host byte-identical (`817e7f04…40`), staged code
  hashes matched pins (`f5adb34b`/`683e884c`/`73087712`), staged-code trio
  33/33 pass on host, M=8, stratified split-seed 0, 2000/500, 378+378 steps,
  batch 16, lr 5e-4, cuda, seeds 0+1. No deviation; the pre-reg stands.
- Failure + recovery on record: first launch died on missing
  `data/raw/banking77` (host had `banking77.tgz` unextracted); extracted on
  host (xattr warnings only), relaunched. Recipe untouched.

## Results (misses as errors, Wilson 95% LB, recorded Jev 0.797)

| seed | recall@8 test | misses /500 | retrieved-only | end-to-end | Wilson LB | wall |
|---|---|---|---|---|---|---|
| 0 | 0.936 | 32 | 0.7265 (n=468) | **0.6800** | 0.6379 | 24.8 s |
| 1 | 0.938 | 31 | 0.7186 (n=469) | **0.6740** | 0.6317 | 24.8 s |

## Verdict: BELOW — claim fails under the unchanged rule

Both lower bounds < 0.797. The funnel runs K=77 cross decisions at scale on
GPU (recall@8 ≈ 0.94, ~25 s/cell vs ~300–425 s CPU) but does NOT beat the
recorded Jev point. No re-seeding, no re-analysis, no criterion change.

## What this means

- The K-cap lift is real as mechanism (retrieval covers the label ~94%, the
  head is honest about the rest via miss accounting), but at tiny rung +
  378-step budget the end-to-end number is far from 0.797. Closing that gap
  needs a different lever (budget, rung, M, head capacity) — a NEW pre-reg,
  not a reinterpretation of this one.
- CPU baseline (0.668/0.632) and GPU (0.680/0.674) agree within seed noise:
  the recipe is stable across devices; device is not the bottleneck.

## Artifacts (sha256, mirrored byte-identical from host, verified locally)

- `result_s0.json` `b6b8c626…`, `result_s1.json` `eb1b5c42…`
- `evidence_s0/provenance.json` `79685251…` (keep_test 500, packed 468,
  ckpt `817e7f04…`), `evidence_s1/provenance.json` `4ecf225d…`
  (keep_test 500, packed 469). Weights (`final.pt`, `retriever.pt`) remain
  on host only — no local GPU-weight need; noted, not hidden.
- `cell_s0.log`, `cell_s1.log` (full transcripts).

## Limitations (carried + new)

- Pre-reg limitations stand (tiny rung, 378 steps, one split-seed, two
  seeds, recorded-not-live Jev). New: two cells shared one GPU concurrently;
  walls are engineering numbers, not benchmarks. Host weights not mirrored.
