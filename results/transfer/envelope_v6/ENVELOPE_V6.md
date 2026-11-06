# Envelope v6 — verdict (executed 2026-11-04, RTX PRO 6000, ~73 s/cell)

## Pre-registration compliance

Cells ran the locked v6 recipe (`PREREG_V6.md` `5ae944bb…`): identical to
v4 except rung tiny→small (anchor `93427b07…`, staged via molab_put_direct,
verified byte-identical on-host BEFORE launch). Pinned code, M=8,
stratified split-seed 0, 2000/500, 378+378, batch 16, lr 5e-4, cuda,
seeds 0+1. No deviation.

## Results (misses as errors, Wilson 95% LB vs recorded 0.797, n=500)

| seed | recall@8 test | misses /500 | retrieved-only | end-to-end | Wilson LB | wall |
|---|---|---|---|---|---|---|
| 0 | 0.956 | 22 | 0.7845 (n=478) | **0.7500** | 0.7102 | 72.8 s |
| 1 | 0.952 | 24 | 0.7794 (n=476) | **0.7420** | 0.7019 | 72.9 s |

## Verdict: BELOW — claim fails under the unchanged rule

Rung scale-up is a real lever (+0.07 e2e over tiny: 0.68→0.75; head
0.7265→0.7845; recall 0.936→0.956) but the gap to 0.797 remains ~0.05
with LB ~0.71. No re-seeding, no reinterpretation.

## Artifacts (mirrored byte-identical, verified locally)

- `result_s0.json` `35becaba…`, `result_s1.json` `e0bea4eb…`
- `evidence_s0/provenance.json` `2836a596…`, `evidence_s1/provenance.json`
  `faae0c12…`, cell logs. Host weights stay on host (noted).

## Limitations

- Single rung step (14M); same split-seed, two seeds, recorded Jev.
- Combined-lever probe (small + M16 + 756/756) running separately in
  `envelope_v6_probe/` — diagnostic only, not part of this verdict.
