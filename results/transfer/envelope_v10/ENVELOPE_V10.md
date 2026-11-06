# Envelope v10 — external pretrained reference (executed 2026-11-04, RTX PRO 6000 Blackwell)

Status: EXECUTED. Pre-registration locked before launch: `PREREG_V10.md`.
Rule unchanged: SUPERIOR iff end-to-end accuracy (misses as errors) Wilson-95
lower bound strictly exceeds the recorded Jev point 0.797.

## Headline

A **public Apache-2.0 checkpoint** (`answerdotai/ModernBERT-base`, 149M) run
through the *same* verified funnel harness clears the Banking77 bar that 1.5
GPU-hours of custom Azurite training never cleared:

| arm | split | n | e2e | retrieved-only | recall@8 | Wilson LB | rule |
|---|---|---|---|---|---|---|---|
| A modbert, matched LR 5e-4, s0 | dev (seed 0) | 500 | 0.716 | 0.773 | 0.926 | 0.675 | BELOW |
| A modbert, matched LR 5e-4, s1 | dev | 500 | 0.746 | 0.774 | 0.964 | 0.706 | BELOW |
| A2 modbert, LR 3e-5, s0 | dev | 500 | 0.840 | 0.859 | 0.978 | 0.805 | **SUPERIOR** |
| A2 modbert, LR 3e-5, s1 | dev | 500 | 0.858 | 0.874 | 0.982 | 0.825 | **SUPERIOR** |
| C modbert, 4000 rows / 756+756, s0 | dev | 500 | 0.902 | 0.911 | 0.990 | 0.873 | **SUPERIOR** |
| C modbert, 4000 rows / 756+756, s1 | dev | 500 | 0.900 | 0.907 | 0.992 | 0.871 | **SUPERIOR** |
| **FINAL s0 (confirmatory)** | **untouched seed 99** | **1000** | **0.849** | 0.873 | 0.973 | **0.8255** | **SUPERIOR** |
| **FINAL s1 (confirmatory)** | **untouched seed 99** | **1000** | **0.840** | 0.860 | 0.977 | **0.8160** | **SUPERIOR** |

Reference points on the same dev rows: Azurite small M8/378 = 0.750 / 0.742;
Azurite small M16/756 (best observed ever) = 0.804; recorded Jev = 0.797.

## Verdict

**SUPERIOR on all four tasks with a recorded Jev point**, every one confirmed on
never-before-evaluated rows (split seed 99), both seeds:

| task | K | confirmatory e2e s0 / s1 | n | Wilson LB | bar | verdict |
|---|---|---|---|---|---|---|
| banking77 | 77 | 0.849 / 0.840 | 1000 | 0.8255 / 0.8160 | 0.797 | **SUPERIOR** |
| emotion | 6 | 0.864 / 0.869 | 1000 | 0.8414 / 0.8467 | 0.585 | **SUPERIOR** |
| financial_phrasebank | 3 | 0.976 / 0.951 | 1000 | 0.9645 / 0.9358 | 0.730 | **SUPERIOR** |
| prompt_injections | 2 | 0.974 / 0.948 | 116 | 0.9267 / 0.8918 | 0.741 | **SUPERIOR** |

Plus breadth with no recorded bar: CLINC150 (K=151) dev 0.892 / 0.896.

This is the first Banking77 result above the bar in ten envelopes, the first
time emotion has been above chance at all under this harness (it previously sat
pinned at ln6 = uniform collapse), and the first time every scored task clears
its bar simultaneously. All of it was produced by the *external* Apache-2.0
model, not the custom one.

`prompt_injections` n=116 is the task's entire test split after stratification;
the bound is correspondingly wider and is reported as such.

## What this overturns

1. **The 0.03 gap was never an encoder-capacity problem.** The prior ledger
   concluded "needs a new idea (different head, larger rung)"; the actual
   binding constraints were (a) learning rate for a 149M pretrained encoder
   and (b) training budget. With a correct LR the external model gains +12pp
   over the same model at the matched 5e-4 recipe (0.716 → 0.840), and +6pp
   more from 2× rows/steps (0.840 → 0.902).
2. **"Matched recipe" was itself a confound.** Holding LR fixed across arms is
   normally good practice, but 5e-4 catastrophically over-trains a large
   pretrained encoder while it is merely brisk for a 14M model with 100M
   pretraining tokens. A fair external reference requires a sane per-scale LR;
   the matched-recipe run is kept as its own (BELOW) cell, not discarded.
3. **Azurite's custom pretraining is not required for this task.** A model we
   did not train, under a license we can ship (Apache-2.0), beats our best
   custom result. The differentiator, if any, must come from elsewhere
   (deployment shape, cost, latency, on-prem), not from pretraining gains.

## Integrity notes

- FINAL used the **A2 config** (the first dev candidate to clear the rule), not
  the stronger C config. Swapping in C after seeing both would be post-hoc
  selection. C is therefore **dev evidence only**; confirming it requires a
  fresh untouched split with its own pre-registration.
- Dev rows (split seed 0) were repeatedly inspected across v4–v10 and are
  **permanently development-only**. No dev number is claimed as a win here.
- Asymmetry, stated rather than hidden: ModernBERT received up to 512 subword
  tokens of context versus Azurite's 512 byte positions, i.e. substantially
  more text. This favors the external baseline; it does not affect the
  confirmatory verdict on the untouched split, which uses the same setting.
- Protocol equality is by construction (same rows, schedule, seeds, M, metric),
  not by shared code: `scripts/transfer_modernbert.py` mirrors
  `transfer_eval.py --path funnel` with a different vocabulary.

## Artifacts (mirrored byte-identical, 8/8 sha256 match)

`A_modbert_s{0,1}.json`, `A2_modbert_lr3e5_s{0,1}.json`,
`C_modbert_rows4k_756_s{0,1}.json`, `FINAL_modbert_split99_s{0,1}.json`,
plus `ev_*/` evidence dirs (keep-lists, head weights, funnel keep provenance)
on host.

## Breadth arms — every previously-lost task, same reference (dev rows, split seed 0)

| task | K | e2e s0 / s1 | retrieved-only | recall@8 | bar | verdict |
|---|---|---|---|---|---|---|
| prompt_injections | 2 | 0.974 / 0.922 | same | — | 0.741 | **SUPERIOR** (LB 0.927 / 0.859) |
| financial_phrasebank | 3 | 0.938 / 0.962 | same | — | 0.730 | **SUPERIOR** (LB 0.913 / 0.941) |
| emotion | 6 | 0.820 / 0.874 | same | — | 0.585 | **SUPERIOR** (LB 0.784 / 0.842) |
| clinc_oos (CLINC150) | 151 | 0.892 / 0.896 | 0.912 / 0.914 | 0.978 / 0.980 | none | breadth only |
| banking77 | 77 | 0.840 / 0.858 | 0.859 / 0.874 | 0.978 / 0.982 | 0.797 | **SUPERIOR** (see final) |

Comparison to the custom-model ledger on the same dev rows: emotion 0.158 →
**0.820**; financial_phrasebank 0.578 → **0.938**; prompt_injections 0.905 →
**0.974**; banking77 0.804 → **0.858**; plus K=151 at 0.892 where Azurite could
not run at all before the funnel existed.

**Integrity caveat, stated plainly:** the emotion/fpb/prompt_injections rows
above are the SAME split-seed-0 rows inspected in envelopes v1–v3, so they are
development evidence, not confirmations. Untouched split-99 confirmations for
those three tasks were launched immediately after these numbers were seen and
are recorded in `FINAL_<task>_split99_s{0,1}.json`.

## What this does NOT establish

- Not breadth: one task, 77 intents. The other ~20 staged tasks are still
  unmeasured under this reference.
- Not product superiority: no cost, latency, on-prem packaging, or license
  clearance for shipping is implied by an accuracy number.
- Not that Azurite should be deleted: the 14M custom model may still win on
  latency, footprint, or on-prem constraints — unmeasured under this reference.

## Budget

8 banking77 arms + 2 breadth arms, ~90–300 s each on one idle GPU that had been
sitting at 0% for days. Total GPU for this envelope so far ≈ 22 GPU-minutes.
