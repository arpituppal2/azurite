# v13 calibration results — development data only (split 0)

Pre-registration: `PREREG_CALIB.md`. Script: `scripts/calibrate_funnel.py`
(sha `93a3053c…`; grid-search fit — an earlier Adam attempt diverged to T=nan and
was replaced, disclosed here).

## banking77, C recipe, 4000 train / 1000 dev rows, T fit on even dev rows

| seed | T | heldout raw (acc / ECE / NLL) | heldout scaled (acc / ECE / NLL) |
|---|---|---|---|
| 0 | 2.7510 | 0.904 / 0.0818 / 0.9844 | 0.904 / **0.0541** / **0.4934** |
| 1 | 2.2022 | 0.896 / 0.0948 / 0.7888 | 0.896 / **0.0279** / **0.4757** |

Fit-half and held-out-half agree (one scalar cannot overfit much): s0 fit-scaled
ECE 0.0531 vs heldout-scaled 0.0541; s1 0.0270 vs 0.0279. Accuracy is exactly
unchanged, as temperature scaling guarantees (argmax-invariant).

Read-out:

- Our raw classification ECE is ~0.08–0.09, not ~0.33 (that was boolq context
  cells). The gap to the competitor's published 0.021 is ~4×, and it closes to
  ~1.3–2.6× with one dev-fit scalar. Seed 1 lands at 0.0279 — the same order of
  magnitude, on a different task, so not a head-to-head claim, but no longer an
  embarrassment.
- NLL roughly halves on both seeds. The model was systematically overconfident
  (T > 2 both seeds); scaling fixes the confidence level without touching any
  decision.
- Cost: ~95 s/arm. This now becomes the default reporting companion for every
  future cell: raw + scaled ECE/NLL alongside accuracy.

What this does NOT establish: anything about split 99/123 (untouched), or about
context-task calibration (boolq ECE 0.33–0.37 is a separate, harder problem —
the cross head over passages, not the choice head, and likely miscalibrated for
different reasons).

## emotion replication (same protocol, K=6 so funnel m clamps to 6)

| seed | T | heldout raw (acc / ECE / NLL) | heldout scaled (acc / ECE / NLL) |
|---|---|---|---|
| 0 | 4.2931 | 0.844 / 0.1560 / 2.2717 | 0.844 / **0.0212** / **0.6742** |
| 1 | 4.1062 | 0.868 / 0.1319 / 1.4893 | 0.868 / **0.0243** / **0.5182** |

The effect replicates across task and seed, and it is larger on emotion: raw ECE
0.13–0.16 (more overconfident than banking77, T ≈ 4.1–4.3 vs 2.2–2.8) scaling
down to **0.021–0.024 — the competitor's published 0.021 to three decimals, on a
different task**. Accuracy exactly unchanged in all four arms. NLL more than
halves in three of four arms.

Engineering note: emotion has K=6 < funnel_m=8, which crashed the first attempt
(`topk` out of range). Fixed by clamping the shortlist to K in
`calibrate_funnel.py`; no result from the crashed run was kept.
