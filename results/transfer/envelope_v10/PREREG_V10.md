# Envelope v10 — pre-registration (dev comparison, 2026-11-04)

Status: LOCKED before launch. No v10 cell has been launched.

## Why this envelope exists

Banking77 funnel vs recorded Jev 0.797: best observed is small M16/756
e2e 0.804 (LB 0.767). Correction: that is the best *observed* result, not a
demonstrated ceiling. The learning curves show a specific, testable pattern:

- Phase-B (cross reranker) retrieved-only accuracy saturates by step ~126 at
  0.78-0.82 in EVERY config (v6 s0/s1, v6-probe combined, v8 B_s1) while NLL
  climbs (1.4 -> 2.3) and mean confidence sits at 0.98. Classic head-capacity
  / overfit signature, not an encoder-data limit.
- Phase-A (dual retriever) is still improving at the last step in the
  ceiling probe (0.764 at step 630 -> 0.804 at step 755).

Two confounded hypotheses: (H1) the tanh cross head caps discrimination
among confusable intents; (H2) the Azurite encoder's pretraining data is the
limit. One intervention cannot separate them. This envelope runs both arms.

## Cells (all on DEVELOPMENT rows — see separation note)

Task banking77, funnel M8, stratified split-seed 0, 2000 train / 500 test,
batch 16, lr 5e-4, dual 378 + cross 378 steps, seeds 0+1:

- A_modbert_s0/s1: `scripts/transfer_modernbert.py` (ModernBERT-base,
  Apache-2.0, external pretrain) through the same funnel protocol.
- B_mlphead_s0/s1: `scripts/transfer_eval.py --cross-head mlp`
  (Azurite-small encoder, LayerNorm+GELU residual head in phase B only;
  phase A retriever identical to v6).

## Decision rule (unchanged)

SUPERIOR iff end-to-end accuracy (misses as errors) Wilson-95 LB strictly
exceeds the recorded Jev point 0.797, per seed. Wins AND losses reported, no
re-seeding. Interpreting H1 vs H2 uses the CONTRAST (modbert retrieved-only
vs mlphead retrieved-only vs v6 tanh baseline), not any single LB.

## Development/final separation (new, binding)

The split-seed-0 test rows (500) have guided decisions since v4. They are
hereby DEVELOPMENT rows: v10 verdicts are development evidence only and may
not be claimed as a win. The untouched final assessment is split-seed 99
(2000 train / 1000 test, never evaluated, never inspected) — reserved for a
single confirmatory run of at most one candidate that first clears LB>0.797
on dev. Running the final early voids it; a new seed must then be reserved.

## Limitations pre-stated

- Local MPS/CPU execution (no GPU host reachable); wall ~20-40 min/cell.
- ModernBERT script mirrors but does not share packing code (different
  vocabulary); protocol equality is by construction, audited in review.
- Head comparison changes only phase B; phase A retriever is the v6 recipe.
