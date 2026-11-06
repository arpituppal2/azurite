# Funnel CPU probe — stratified banking77 (2026-11-04, 0 s GPU)

## Why this probe exists

The prefix toy smoke (64 train rows, dataset indices 0–63) could not
distinguish *positions* from *global dataset ids*: both coincide. A
stratified draw (385 train / 154 test, non-contiguous global indices) broke
the funnel end-to-end with `IndexError: list index out of range` — two real
plumbing bugs, both invisible to the 32-test suite:

1. `pack_cross_subset` stored `ex.index` (global dataset id) in
   `packed_indices`; consumers need positions into the rows/keep lists.
   Fixed: `packed_indices.append(pos)` via `enumerate`.
2. `funnel_map_predictions` zipped `subset_pred` positionally against the
   full-length `keep_full`, misaligning every prediction after the first
   miss. Fixed: resolve each prediction through its own row's keep-list,
   `mapped[row] = keep_full[row][pred]`.

Both fixes are covered by a new regression test with non-contiguous global
ids (`test_packed_indices_are_positions_not_global_ids`) plus a corrected
mapping test; the old mapping test had encoded the buggy zip semantics and
was fixed to the correct expectation ([2, None, 2, 1]).

## Cell (random init = mechanism probe, NOT a superiority claim)

- Recipe: banking77, byte tiny, random init, funnel M=8, stratified
  (split-seed 0, 385 train / 154 test), dual 24 steps + cross 24 steps,
  batch 32, seed 0, CPU. Wall 26.6 s.
- Result: recall@8 train 0.3688 / test 0.3117; misses 106/154;
  retrieved-only acc 0.1458 (n=48); **end-to-end acc 0.0455**.
- Interpretation: a random-init retriever at 24 steps retrieves the gold in
  its top-8 only ~31% of the time on a class-balanced 77-way draw, so the
  miss counter dominates — exactly the honest behavior the funnel metric was
  designed to report (misses as errors). This says nothing about the
  pretrained funnel (the GPU-blocked envelope-v4 question).
- Determinism: rerun to /tmp is metric-identical across final, recalls,
  misses, retrieved-only, and both histories.
- Files: `result_s0.json` (`e7c34110…`), `evidence_s0/` (retriever.pt,
  final.pt, provenance.json `9d30f1ce…` with per-row keep-lists).

## Code state

- `azurite/decisions.py` (`f5adb34b…`), `scripts/transfer_eval.py`
  (`683e884c…`), `tests/test_funnel.py` (`73087712…`).
- Full suite: 158 passed (excl. test_eval_battery).
