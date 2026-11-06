# Addendum 2026-11-03 — GPU execution (smoke passed, session lost, cap set)

## Authorization change
User lifted the CPU-only restriction: GPU is available and free. CPU-only
planning assumption retired — no six CPU-hours will be spent to preserve it.
Scope unchanged: faster execution only. No extra cells, no resumed
pretraining, no broader campaign.

## Precision policy (same for all 12 cells, matched to history)
`--amp` default is `off` (verified in `scripts/transfer_eval.py`: "bf16
autocast is measured on CUDA and off by default everywhere else"). All cells
run `--device cuda` with default `--amp off`: fp32 throughout, no autocast —
the identical numerical policy as the historical CPU cells. Historical CPU
results stay in `results/transfer/ag_news_balanced/`; the new matched
comparison lives separately in `results/transfer/ag_news_shared_order/`.

## GPU + software record (probed on sb-58ad8e167881cecb before loss)
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB (97887 MiB),
  capability 12.0, driver 595.71.05.
- torch 2.11.0+cu130, CUDA build 13.0, cuDNN 91900.
- Peak memory (smoke, 400 train / 100 test / 40 steps): cross-canonical
  2345 MiB, dual 2735 MiB.

## Smoke test (PASSED, both paths, on the GPU)
- `cross --cross-shuffle off` seed 0 pretrained 40 steps: rc 0, 8.6 s wall.
  Provenance records `cross_shuffle: "off"`; all 100 test candidate orders
  canonical `[0,1,2,3]`; packed_gold == canonical_gold on all rows (the
  harness's own packed-gold assertions passed under canonical packing).
- `dual` seed 0 pretrained 40 steps: rc 0, 9.0 s wall; all orders canonical.
  (Dual provenance carries the inert default `cross_shuffle: "on"`; the flag
  is cross-only and dual ignores it.)
- Evidence retained on the host in `ag_news_shared_order/_smoke_cross/` and
  `_smoke_dual/` (provenance.json + final.pt; result.json was omitted by a
  missing `--out` in the smoke commands — corrected in all 12 cell commands,
  and the driver writes training.log from captured stdout).
- Extrapolation to full cells (378 steps, 2000/500 rows, eval every 6 steps
  on 500): ~2–5 min/cell.

## Cap (set from measured smoke runtime)
- Per-cell cap: 15 min (3–7x the extrapolated full-cell time).
- Total cap: 12 cells sequential => 3 GPU-hours.
- Calibration rule: the first full cell's measured wall time resets the
  per-cell cap to max(15 min, 4x measured slower-path cell); any cell
  exceeding its cap stops the run and blocks pooling.

## Session loss (blocker, no spend incurred)
- After the smoke verification, sb-58ad8e167881cecb began returning HTTP 410
  on `/`, `/api/status`, and code execution (three consecutive failures).
  The sandbox is gone, same as the earlier sessions.
- The 12-cell launch NEVER STARTED: `_run12.py` was never written to the
  host, no full cell ran, no GPU time was consumed beyond the two ~9 s
  smoke runs. Nothing was left running; there is nothing to stop.
- Staged host files (42-file comparator set, manifest, FROZEN_PROTOCOL.md,
  H0_VERDICT.json, amendment copy, smoke evidence) are presumed lost with
  the sandbox.
- Recovery path: on the next live molab session, re-stage from the local
  manifest (`STAGING_MANIFEST.json`, 42 files + itself, all hashes local),
  plus the patched `scripts/transfer_eval.py` (sha
  3f438183089c253ad86a467bb5c4be4cf893f17f574f5747d63971e54b233e8f —
  verified identical to the reviewed molab patch), rerun H0, then launch.
  The amendment text above is the authoritative spec; no re-derivation needed.
