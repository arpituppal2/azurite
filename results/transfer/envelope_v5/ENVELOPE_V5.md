# Envelope v5 — verdict (executed 2026-11-04, RTX PRO 6000, ~32 s GPU)

## Pre-registration compliance

Cells ran the locked v5 recipe (`PREREG_V5.md` `bbe0cabe…`): identical to
superiority-v1 Execution-1 (546/116, 378 steps, batch 16, lr 5e-4,
stratified split-seed 0, anchor ckpt verified on-host, pinned code).
Data staged from the local licensed tree via tarball with matching
local/remote sha (`90d3c8fd…`) before extract — same precedent as v1.
No deviation.

## Results (Wilson 95% LB vs recorded Jev 0.741, n=116)

| cell | acc | LB | verdict |
|---|---|---|---|
| cross s1 | 0.8879 (103/116) | 0.8176 | SUPERIOR — replicates v1 cross s0 (LB 0.7486) |
| dual s1 | 0.8966 (104/116) | 0.8279 | SUPERIOR — replicates v1 dual s0 (LB 0.8381) |

## Judgment

The single-task superiority on prompt_injections now holds on TWO seeds
per path (s0 v1 + s1 here), all four cells LB > 0.741 under the unchanged
rule. This strengthens exactly one bounded claim — prompt_injections
transfer under the 378-step recipe — and nothing else. Emotion,
financial_phrasebank, banking77 (dual, funnel-CPU, funnel-GPU) remain
losses; banking77-cross-funnel stays BELOW per v4.

## Artifacts (mirrored byte-identical)

- `cross_s1.json` `be8b23a4…`, `dual_s1.json` `e2e582b2…`, cell logs.
- Pre-reg `PREREG_V5.md` `bbe0cabe…`.

## Limitations

- n=116 wide intervals; two seeds is replication, not robustness.
- Jev recorded, not live. No funnel on this task (small-K, cross serves
  it directly). v4 probe plateau (~0.72 tiny-rung funnel ceiling on
  banking77) recorded separately in `envelope_v5_probe/`.
