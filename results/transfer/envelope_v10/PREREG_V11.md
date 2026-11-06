# Envelope v11 — pre-registration: confirming the budget arm on a fresh split

Status: LOCKED before launch. No v11 cell has been executed at the time of writing.

## Why

Envelope v10 produced two dev findings on split seed 0:

- A2 (ModernBERT, lr 3e-5, 2000 rows, 378+378): dev 0.840 / 0.858 — this was the
  config carried into the split-99 confirmation, because it was the *first* to
  clear the rule.
- C (ModernBERT, lr 3e-5, 4000 rows, 756+756): dev 0.902 / 0.900 — stronger,
  but seen only after A2 had already been chosen.

Choosing C for the confirmatory run after seeing both would be post-hoc
selection. So C is currently **dev evidence only**, and this envelope exists to
close that hole honestly: a single confirmation of C on rows that have never
been evaluated or inspected.

## Cells

Task banking77, funnel M8, ModernBERT-base, lr 3e-5, batch 16, **4000 train
rows, 756 dual + 756 cross steps**, seeds 0 and 1, evaluated on
**split seed 123** (test split seed 124), **n = 1000** rows.

Split seed 123 has never been used by any envelope. Split seed 0 remains
permanently development-only; split seed 99 stays reserved as the v10
confirmation record and is not reused.

## Decision rule (unchanged)

SUPERIOR iff end-to-end accuracy (misses as errors) Wilson-95 lower bound
strictly exceeds the recorded Jev point 0.797, per seed.

## Pre-stated interpretation

- If both seeds exceed 0.797 with LB, C is confirmed and supersedes A2 as the
  recommended configuration.
- If only one seed does, C is NOT confirmed; A2's split-99 result stands as the
  claim and the discrepancy is reported as seed sensitivity.
- If neither does, the 0.902 dev figure does not replicate and must be treated
  as split-specific; this would itself be the more important finding.

No re-seeding, no config changes after results, wins and losses both reported.
