# Envelope v4 — pre-registration (LOCKED 2026-11-04, CPU-written, GPU-blocked)

## Status

Pre-registered BEFORE any v4 GPU cell runs. No v4 cell has launched; the GPU
is blocked (stored sessions expired, HTTP 410). This document + PREREG_V4.json
freeze the comparison. Unchanged after results — any deviation voids the claim.

## Question

Does the retrieve-then-rerank funnel let the cross path decide banking77
(K=77, structurally N/A to plain cross) at or above the recorded Jev point?

## Fixed recipe (identical to the CPU baseline anchor)

- Task banking77, byte tiny rung, pretrained anchor
  `results/pretrain/l1_tiny_byte/ckpt.pt`, sha256
  `817e7f0449fae7d228f3d570bdbfe4c0efe031383b3579a4528a077ba6140410`.
  Stage 1 of every GPU return is re-verifying this hash on-host before
  anything launches.
- Path funnel, M=8; stratified split-seed 0, train 2000 / test 500;
  dual 378 steps + cross 378 steps, batch 16, lr 5e-4, device cuda.
- Seeds 0 and 1 (the baseline's two seeds). No re-seeding, no extras.
- Per-cell wall cap 180 s (GPU); evidence-guard no-overwrite applies.

## Metrics (all reported, none optional)

- recall@8 train/test (retriever coverage).
- Retrieved-only accuracy (cross head on packed rows only; diagnostic).
- **Decision metric: end-to-end accuracy over ALL 500 test rows with funnel
  misses counted as errors**, Wilson 95% lower bound.

## Decision rule (unchanged from v0 + superiority pre-regs)

- SUPERIOR iff end-to-end Wilson-95 LB strictly exceeds RECORDED Jev 0.797.
- Wins AND losses reported per seed; a single failing seed fails the claim.
- The CPU baseline (s0 0.6680 / s1 0.6320) is orientation only: it is NOT a
  success bar, NOT part of the verdict, and beating it means nothing.

## Anchor (CPU baseline, for comparability — not a criterion)

- `results/transfer/funnel_cpu_baseline/PROBE.md`
  (`b41135b8…c4ad47b3`); `result_s0.json` (`b74af99b…`);
  `result_s1.json` (`3d0af216…`); provenance with per-row keep-lists
  `evidence_s0` (`722b9889…`) / `evidence_s1` (`35ba02d8…`).
- Rerun identity proven on CPU (s0 rerun identical on all fields).
- Code pinned: `azurite/decisions.py` (`f5adb34b…`),
  `scripts/transfer_eval.py` (`683e884c…`), `tests/test_funnel.py`
  (`73087712…`) — the post positional-zip-fix state. Any code change
  re-opens the protocol, it does not silently inherit this pre-reg.

## GPU-return procedure (see scripts/gpu_return_drycheck.py)

1. Fresh browser session → decode URL → update
   `.stage/live_session_newhost3.json` (never commit tokens).
2. Restage per `docs/RUNBOOK.md` (`molab_sync.sh --all`); hash-verify the
   delivered bytes through the execute channel before launch.
3. Verify anchor ckpt hash on-host first; abort on mismatch.
4. Launch the two cells; poll; mirror artifacts back; confirm hashes
   locally before concluding anything.

## Limitations (pre-stated)

- Tiny rung, 378-step budget, single split-seed, two result seeds; seed
  variance (±15pp historically) can dominate.
- GPU walls are not comparable to CPU baseline walls.
- Jev side is the recorded 0.797 point (no live re-query: founder-spend
  blocker). Beating it would bound one task, not license a general claim.
- Dual-random from-scratch stays CLOSED; nothing here reopens it.
