# PREREG — envelope v13: validation-only calibration (development evidence)

Date: 2026-11-04. Script: `scripts/calibrate_funnel.py`.

## Status of prior splits (explicit labeling)

- Split 99 and split 123 Banking77 rows now carry a dual status. The locked v10/v11
  confirmatory values (A2 on 99, C on 123) were produced by recipes frozen on
  development data and **stand as confirmatory**. But those rows have since been
  repeatedly inspected for comparative reporting (the pplx-decider baseline ran on
  split 123). Any FUTURE model-selection decision that uses split 99/123 numbers
  must be labelled development evidence, not a fresh confirmatory result.
- The confirmatory cells themselves are untouched: no locked number is revised,
  re-fit, or re-reported here.

## What v13 does (split 0 only)

1. Replicate the v11 C recipe (funnel_m 8, dual+cross 756 steps, batch 16,
   lr 3e-5, 4000 train rows) on banking77, split_seed 0, seeds 0–1.
2. Fit ONE temperature scalar by NLL on even-index dev rows (split 1, n=1000).
3. Report accuracy / NLL / ECE (10-bin, same `metrics_from_logits`) and
   reliability tables on the fit half and the held-out odd half, raw and scaled.

## Acceptance / decision rule

- If held-out ECE drops substantially with accuracy unchanged, temperature
  scaling becomes the default reporting companion for every future cell, and the
  competitor's ECE 0.021 claim gets a same-task answer on development data.
- If ECE does not move, calibration is a modelling problem (not a scaling
  problem) and goes on the capability-gap backlog behind attention-readout work.
- Either way: no claim about split 99/123, and no change to bars.

## Bounded tuning disclosure

One scalar, one dev split, fixed recipe, ~200 s/arm on Molab GPU. This is the
"reasonably tuned" companion to the "shared recipe" cells — disclosed here, not
smuggled into a headline number.
