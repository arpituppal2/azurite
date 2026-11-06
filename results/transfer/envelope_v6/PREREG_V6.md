# Envelope v6 — pre-registration (LOCKED 2026-11-04, before any v6 cell)

## Question

Does rung scale-up break the tiny-rung funnel ceiling (~0.72 e2e, v4+v5
probe) on banking77 against the recorded 0.797?

## Fixed recipe (v4 recipe, rung swapped tiny→small; one lever only)

- Task banking77, byte SMALL rung, pretrained anchor
  `results/pretrain/l1_small_byte/ckpt.pt`, sha256
  `93427b07049ac1098cedb193d71497847f9ab049a033cf81d9b101ec0e1eea0c`
  (local sha; staged to host via molab_put_direct, verified byte-identical
  on-host via execute channel BEFORE launch).
- Path funnel, M=8 (unchanged — isolates rung); stratified split-seed 0,
  train 2000 / test 500; dual 378 + cross 378 steps, batch 16, lr 5e-4,
  device cuda; cross-shuffle off (matches v4 CPU/GPU cells).
- Seeds 0 and 1. Per-cell wall cap 300 s (bigger model).
- Staged code pinned (`f5adb34b`/`683e884c`); small-rung load smoke-passed
  on CPU (8 steps, 9 s).

## Decision rule (unchanged)

SUPERIOR iff end-to-end (misses as errors) Wilson-95 LB strictly exceeds
RECORDED Jev 0.797, per seed. One failing seed fails the claim.

## Limitations (pre-stated)

- Single rung step (14M params); still far below Jev-scale capacity.
- Same split-seed, two seeds, recorded-not-live Jev.
- v4/v5 results stand (tiny BELOW, pi replication holds); this envelope
  targets banking77 only.
