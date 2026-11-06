# Envelope v8 — pre-registration (LOCKED 2026-11-04, before any v8 cell)

## Question

Mid-rung funnel on banking77 vs recorded 0.797, under the ACTUAL v4/v6
window conditions (max_seq 1024; v4/v6 provenance-confirmed).

## Fixed recipe (explicit where v7 was silent)

- Task banking77, byte MID rung, anchor
  `results/pretrain/l1_mid_byte_gpu/ckpt.pt` (33.7M params, 100M tokens,
  sha `b78146e6…`, independently re-hashed on-host).
- Window: max_seq 1024 via `--allow-context-extension` (ckpt is 256-native;
  the extension is the documented cost of matching v4/v6 conditions —
  v4 tiny and v6 small both packed at 1024, mean len ~283).
- NO `--max-seq` flag. Stratified split-seed 0, 2000/500, batch 16,
  lr 5e-4, cuda, cross-shuffle off. Staged code pinned.
- Config A: M=8, dual 378 + cross 378, seeds 0+1 (v4/v6 configuration).
- Config B: M=16, dual 756 + cross 756, seeds 0+1 (small-rung winner).
- Per-cell wall cap 600 s.

## Decision rule (unchanged)

Per config, SUPERIOR iff BOTH seeds' end-to-end (misses as errors)
Wilson-95 LB strictly exceeds RECORDED Jev 0.797. Configs independent.

## Limitations (pre-stated)

- RoPE extrapolation 256→1024 is unvalidated on this anchor; if the
  extended window degrades the encoder, that appears as a loss, not a
  re-run.
- Same split-seed, two seeds, recorded-not-live Jev. v7 void rationale in
  ENVELOPE_V7.md.
