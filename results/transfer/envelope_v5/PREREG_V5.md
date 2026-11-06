# Envelope v5 — pre-registration (LOCKED 2026-11-04, before any v5 cell)

## Question

Does the single-task superiority on prompt_injections (v1: cross s0 AND
dual s0, both LB > 0.741) replicate on a fresh seed under the identical
recipe? Two confirmatory cells, no exploration inside this envelope.

## Fixed recipe (identical to superiority-v1 Execution-1)

- Task prompt_injections, byte tiny rung, pretrained anchor
  `results/pretrain/l1_tiny_byte/ckpt.pt`, sha256
  `817e7f0449fae7d228f3d570bdbfe4c0efe031383b3579a4528a077ba6140410`
  (verified on-host byte-identical this pass).
- Stratified split-seed 0, train 546 / test 116, 378 steps, batch 16,
  lr 5e-4, device cuda, cross-shuffle off.
- Cells: cross seed 1, dual seed 1. (v1 measured seeds 0; this envelope
  adds exactly one fresh seed per path, no funnel — small-K task, cross
  serves it directly.)
- Staged code pinned to post-fix hashes (`f5adb34b`/`683e884c`).

## Decision rule (unchanged since v0/v1)

Per path, SUPERIOR iff fresh accuracy Wilson-95 LB strictly exceeds
RECORDED Jev 0.741. Reported per path; a failing path fails only that
path. This envelope can at most replicate the two v1 winning cells on a
second seed — it cannot create a new winning task.

## Limitations (pre-stated)

- n=116: LBs are wide; replication on one seed is weak evidence.
- Jev side recorded, not live (founder-spend blocker).
- banking77 funnel (v4) stays BELOW; nothing here re-opens it.
