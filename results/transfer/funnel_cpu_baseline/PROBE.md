# Funnel CPU baseline — banking77, pretrained, stratified (2026-11-04, 0 s GPU)

## Config (fixed before launch; transfer recipe mirrored)

- Task banking77 (K=77), byte tiny rung, **pretrained anchor**
  `results/pretrain/l1_tiny_byte/ckpt.pt` (sha `817e7f04…`, matches the
  recorded anchor — verified locally this pass).
- Path funnel, M=8; stratified split-seed 0, train 2000 / test 500;
  dual 378 steps + cross 378 steps, batch 16, lr 5e-4, device cpu.
- Seeds 0 and 1. Command shape (seed s):
  `transfer_eval --task banking77 --arm byte --rung tiny --init pretrained
  --ckpt results/pretrain/l1_tiny_byte/ckpt.pt --path funnel --funnel-m 8
  --stratify --split-seed 0 --examples 2000 --test-examples 500
  --funnel-dual-steps 378 --funnel-cross-steps 378 --batch 16 --lr 5e-4
  --seed s --device cpu --out result_s{s}.json --evidence-dir evidence_s{s}`

## Results (mechanism evidence, NOT a superiority claim)

| seed | recall@8 test | misses /500 | retrieved-only | end-to-end | Wilson LB | wall |
|---|---|---|---|---|---|---|
| 0 | 0.9420 | 29 | 0.7091 (n=471) | **0.6680** | 0.6256 | 293 s |
| 1 | 0.9100 | 45 | 0.6945 (n=455) | **0.6320** | 0.5889 | 425 s |

- Both lower bounds < recorded Jev 0.797 → BELOW under the locked rule.
  The funnel is proven to *run* the K=77 cross decision at scale
  (recall@8 > 0.9, misses counted as errors), but this CPU baseline does
  not win and does not close envelope-v4 — that needs the GPU cell with
  matched recipe/timing, pre-registered prospectively.
- Pretrained retriever vs random probe: recall@8 0.91–0.94 here vs 0.31 at
  24 random steps — init matters, as the closed dual-random line predicts.
- Seed spread (0.668 vs 0.632) is consistent with known ±15pp seed variance.
- Determinism: seed-0 rerun to /tmp is IDENTICAL across final, recalls,
  misses, retrieved-only, and both full histories.

## Files (hashes sha256 prefix)

- `result_s0.json` `b74af99b…`, `evidence_s0/` (retriever.pt, final.pt,
  provenance.json `722b9889…` with per-row keep-lists + ckpt sha).
- `result_s1.json` `3d0af216…`, `evidence_s1/` (provenance `35ba02d8…`).
- `cell_s0.log`, `cell_s1.log` (full training transcripts).
- Code: `azurite/decisions.py`, `scripts/transfer_eval.py`,
  `tests/test_funnel.py` unchanged since the stratified-probe fix;
  full suite 158 passed (excl. test_eval_battery).

## Limitations

- CPU timings (293/425 s) are laptop references, not comparable to GPU
  cell walls; wall variance between seeds is machine noise + eval cadence.
- Random-init caveat does not apply (pretrained init used); the caveat that
  does apply: tiny rung, 378-step budget, single split-seed.
- No claim against Jev beyond the recorded-point comparison above (BELOW).
