# AG News balanced transfer: completed verdict

The six pre-registered cells finished on the replacement sandbox and were
verified locally. Full evidence:
`results/recovered_20261028/results/transfer/ag_news_balanced/` (`summary.json`,
`validation.json`, six `result.json`/`final.pt` pairs, per-example predictions).

## Per-seed

| seed | pretrained | random | Δ pp | exact McNemar p |
|---:|---:|---:|---:|---:|
| 0 | 0.512 | 0.510 | +0.20 | 1.000 |
| 1 | 0.518 | 0.490 | +2.80 | 0.247 |
| 2 | 0.490 | 0.486 | +0.40 | 0.931 |

## Pooled

- pretrained: **760/1500 = 0.5067** (descriptive Wilson 0.481–0.532)
- random: **743/1500 = 0.4953** (descriptive Wilson 0.470–0.521)
- paired difference: **+1.13 pp**
- crossed seed/item bootstrap 95%: **−2.60 to +4.80 pp**
- verdict: **inconclusive** (interval spans zero)
- historical +6.4 pp: **above** the new interval; **withdrawn** as a
  class-balanced result.

The pre-registered decision rule was: positive if the paired interval is wholly
above zero, adverse if wholly below, otherwise inconclusive. This is
inconclusive, not adverse. It is also not evidence for +6.4 pp.

## Controls

- permutation control (77-way Banking77): 2.1/2.6/1.2% against a 1.30% chance
  floor, so the transfer score is not a label-leakage artifact;
- second draw: 87.1/84.6/85.3% on split seed 2;
- contamination: 3/500 AG News examples share a 40-byte window with the
  train corpus, and those 3 are not easier (2/3 correct vs 51.1% on the rest).

The source checkpoint for these six cells is the amended `817e7f04…0410`, not
the lost `3cf421…bed8`; the protocol amendment is recorded in `PROTOCOL.md`.
