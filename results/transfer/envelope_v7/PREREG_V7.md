# Envelope v7 — pre-registration (LOCKED 2026-11-04, before any v7 cell)

## Question

Does the mid-rung anchor lift the funnel past the recorded 0.797, and
does the combined recipe still dominate the base recipe at the new rung?

## New anchor (trained this session, verified)

- Mid rung byte, 33,696,768 params, 100,003,840 tokens, 24,415 steps,
  835.4 s on RTX PRO 6000, final train loss 0.6861, held-out 0.9167
  bits/byte (vs small 1.0098 — capacity confirmed before any transfer).
- `results/pretrain/l1_mid_byte_gpu/ckpt.pt` (404,458,153 B), sha256
  `b78146e6d5f19f63471eed9aa27220861edbeb003742726ef6946b306e32fffd`
  — summary-recorded AND independently re-hashed on-host (match).
- Stream `data/tokens/byte_100000000.u16` sha-verified pre-launch
  (`67809404…` both sides). Staged code pinned (`f5adb34b`/`683e884c`).

## Cells (4, all banking77, stratified split-seed 0, 2000/500, batch 16,
lr 5e-4, cuda, cross-shuffle off)

- Config A (isolation: rung effect only): funnel M=8, dual 378 + cross
  378, seeds 0+1 — byte-identical recipe to v6 except the anchor.
- Config B (best-shot): funnel M=16, dual 756 + cross 756, seeds 0+1 —
  the small-rung winner (0.804) ported to the new rung.
- Per-cell wall cap 300 s.

## Decision rule (unchanged)

Per config, SUPERIOR iff BOTH seeds' end-to-end (misses as errors)
Wilson-95 LB strictly exceeds RECORDED Jev 0.797. Configs judged
independently; a failing seed fails only its config.

## Limitations (pre-stated)

- One rung step (33.7M); same split-seed; recorded-not-live Jev.
- v4/v5/v6 stand as recorded. This envelope cannot rescue them.
- If config A wins but B does not (or vice versa), the judgment reports
  exactly that — no pooling across configs.
