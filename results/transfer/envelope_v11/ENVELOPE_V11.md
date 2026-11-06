# Envelope v11 — the budget configuration, confirmed on a fresh split

Status: EXECUTED. Pre-registration locked before launch: `PREREG_V11.md`.
Rule unchanged: SUPERIOR iff end-to-end accuracy (misses as errors) Wilson-95
lower bound strictly exceeds the recorded Jev point 0.797.

## Why this envelope exists

Envelope v10 left one honest hole: the confirmatory run used A2 (2000 rows,
378+378 steps) because it was the *first* dev candidate to clear the bar, while
the stronger C config (4000 rows, 756+756) had only dev evidence. Substituting C
after seeing both would have been post-hoc selection. This envelope closes that
hole on a split (seed 123) never used by any prior envelope.

## Results

| cell | split | n | e2e | recall@8 | Wilson LB | rule |
|---|---|---|---|---|---|---|
| banking77 C, s0 | **untouched 123** | 1000 | **0.910** | 0.996 | 0.8907 | **SUPERIOR** |
| banking77 C, s1 | **untouched 123** | 1000 | **0.919** | 0.993 | 0.9004 | **SUPERIOR** |
| clinc150 C, s0 | untouched 123 | 1000 | 0.921 | 0.985 | 0.9030 | no bar (breadth) |
| clinc150 C, s1 | untouched 123 | 1000 | 0.918 | 0.988 | 0.8998 | no bar (breadth) |

## Verdict: C CONFIRMED, and it supersedes A2

Pre-registered interpretation was: both seeds above the bar → C is confirmed and
supersedes A2. Both seeds clear it by ~9–10 points, with LB ≈ 0.89–0.90 against
a bar of 0.797.

Against the recorded Jev point of 0.797, Banking77 now stands at **0.910 / 0.919**
(+11.3 / +12.2 points), on rows no model has seen before.

The budget lever also holds at label scale: CLINC150 moved 0.892 → 0.921 with the
same change (2× rows, 2× steps), so the gain is not a Banking77 artefact.

## What changed, plainly

For ten envelopes the deficit was attributed to encoder capacity: custom
pretraining, bigger rungs, native context windows, head geometry. None of it
moved the number past 0.804. Two changes did:

1. **A pretrained external encoder at its correct learning rate** (3e-5, not the
   matched 5e-4) — +9 points on its own.
2. **Training budget: 2× rows and 2× steps** — a further +6 to +7.

Both are ordinary, and both were available from the start. The expensive search
was for capacity that was never the constraint.

## Artifacts (mirrored, per-file sha256 verified)

`C_split123_s0.json`, `C_split123_s1.json`,
`clinc150_C_split123_s0.json`, `clinc150_C_split123_s1.json`, plus `ev_*/`
evidence directories (retrieval keep-lists, head weights, packing provenance)
on host.

## Limits

- Banking77 and CLINC150 are intent classification. Neither exercises reasoning
  over long documents; that is the context track and is measured separately.
- Two seeds each. Seed spread on CLINC150 is 0.3 points, on Banking77 0.9.
- The batch-16, lr 3e-5, M=8 recipe is fixed by pre-registration; no per-task
  tuning was done, so these are not each task's best.
