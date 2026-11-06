# Envelope v9 — pre-registration (LOCKED 2026-11-04, before any v9 cell)

## Question

Same as v8, but on the NATIVE-1024 mid anchor: does removing RoPE
extrapolation lift the funnel past the recorded 0.797?

## New anchor (trained this session, verified)

- Mid rung byte, 33,696,768 params, seq 1024 NATIVE, 100,007,936 tokens,
  6,103 steps, 928.5 s on RTX 6000, final train loss 0.7311.
- `results/pretrain/l1_mid_byte_seq1024/ckpt.pt` (404,458,153 B), sha256
  `a5ed3ed8ae6a4cff9828ae4e8931d9eebee8c66a929801b7cb3dbc1592bba6f6` —
  summary-recorded AND independently re-hashed on-host (match).
- Window: native 1024, NO `--allow-context-extension`, NO `--max-seq`
  (harness default 1024 = ckpt-native — the load check passes cleanly).

## Cells (all banking77, stratified split-seed 0, 2000/500, batch 16,
lr 5e-4, cuda, cross-shuffle off, pinned code)

- Config A: M=8, dual 378 + cross 378, seeds 0+1.
- Config B: M=16, dual 756 + cross 756, seeds 0+1.
- Per-cell wall cap 600 s.

## Decision rule (unchanged)

Per config, SUPERIOR iff BOTH seeds' end-to-end (misses as errors)
Wilson-95 LB strictly exceeds RECORDED Jev 0.797. Configs independent.

## Limitations (pre-stated)

- Native-1024 held-out (1.0686 bpb) is worse than seq-256 mid (0.9167) —
  longer context is harder; if the head still caps ~0.78, capacity-per-seat
  is the wall, not the window.
- Same split-seed, two seeds, recorded-not-live Jev. v7 void, v8 below stand.
