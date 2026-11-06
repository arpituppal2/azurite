# Amendment 2026-11-03 — shared candidate ordering (dual-vs-cross)

Status: SPECIFICATION ONLY. No cells run. FROZEN_PROTOCOL.md untouched.
Supersedes nothing; the frozen six-cell dual protocol is ON HOLD, not replaced.
LOCAL MIRROR of the copy recorded on molab sb-58ad8e167881cecb (session
later lost; this file is the authoritative text).

## Problem
H0 found the path comparison confounded with candidate ordering: the six
staged cross cells use per-example seeded shuffle; dual uses canonical bank
order. Canonical remapping aligns scoring but not experimental comparability.
Question: does changing the path help under ONE shared ordering policy?

## Compatibility finding
Identical ordering is technically COMPATIBLE with both paths, so this is NOT
a bundled-configuration comparison — ordering is held constant, path varies:
- `azurite/decisions.py::pack_cross(..., shuffle=True, seed=0)` already
  supports `shuffle=False`, which leaves `order = list(range(len(bank)))`
  (canonical), the same order `pack_dual` always uses.
- Guard: `scripts/transfer_eval.py::draw_provenance` shuffles
  whenever `path == "cross"`. With canonical packing the harness's own
  assertion (`packed test/train gold == draw packed_gold`) would abort, so
  the shuffle flag MUST be threaded into `draw_provenance` too (default
  True preserves all historical records byte-for-byte).

## Required harness diff (apply at execution time, with diff review)
Applied 2026-11-03 on molab (base 300d9ada.. → 3f438183..) and mirrored to
the local checkout (same sha verified). Changes only:
1. `transfer_eval.py`, arg parser: added
   `ap.add_argument("--cross-shuffle", choices=["on", "off"], default="on")`.
2. Both `pack_cross(...)` call sites (train + test): added
   `shuffle=(args.cross_shuffle == "on")`.
3. `draw_provenance(..., path: str)` gained `shuffle: bool = True`; the guard
   is now `if path == "cross" and shuffle:`.
4. Both `draw_provenance(...)` call sites pass
   `shuffle=(args.cross_shuffle == "on")`.
5. Provenance dict records `"cross_shuffle"` (default `"on"` when absent).
Diff review performed on molab before/after: only these lines changed.

## Design: smallest matched comparison (12 cells, one shared ordering)
- Ordering policy: canonical bank order for every example, every seed, both
  paths (`--cross-shuffle off` for cross; dual is canonical by construction).
- Everything else frozen-identical: AG News tiny byte, 2000/500 balanced
  draws with --split-seed 0 for every cell (verified on host 2026-11-03: all
  six staged comparator provenances use split-seed 0; row sets identical
  within seed), seeds 0-2 (vary batch schedule), 378 steps, batch 16,
  lr 0.0005, AdamW, init pretrained (ckpt 817e7f04..0410) vs random.
- New evidence dir `results/transfer/ag_news_shared_order/` (never touches
  `ag_news_balanced/` or `ag_news_balanced_dual/`).
- Decision rule (carried over): paired crossed-bootstrap 95% CI on pooled
  accuracy; ordering effect now excluded by construction.

## Reusability ruling
- REUSABLE AS-IS: ckpt, AG News data, all six existing cross cells (they
  remain the shuffled-order reference for their own claim).
- MUST RERUN (6): all cross cells under canonical order — shuffled cells
  cannot answer the shared-ordering question (slot-gold mapping differs).
- NEW (6): dual cells (never run).
- New cells total: 12. No LR sweep, no other tasks, no extra seeds: not a
  factorial.

## Device/cap revision (see ADDENDUM_2026-11-03_gpu.md)
CPU-only restriction lifted by the user 2026-11-03. All 12 cells run on ONE
device (GPU), `--device cuda`, default `--amp off` (fp32, same policy as the
historical CPU cells). Provisional cap pending first full-cell timing:
15 min/cell, 3 GPU-hours total, 0 extra cells.

## Exact commands (run from workspace root, GPU, one at a time)
Cross-canonical (6):
```
python3 scripts/transfer_eval.py --task ag_news --path cross --cross-shuffle off --init pretrained --seed 0 --examples 2000 --test-examples 500 --epochs 3 --steps 378 --batch 16 --lr 0.0005 --stratify --split-seed 0 --device cuda --ckpt results/pretrain/l1_tiny_byte/ckpt.pt --evidence-dir results/transfer/ag_news_shared_order/cross_s0 --out results/transfer/ag_news_shared_order/cross_s0/result.json
python3 scripts/transfer_eval.py --task ag_news --path cross --cross-shuffle off --init random     --seed 0 --examples 2000 --test-examples 500 --epochs 3 --steps 378 --batch 16 --lr 0.0005 --stratify --split-seed 0 --device cuda --evidence-dir results/transfer/ag_news_shared_order/random_cross_s0 --out results/transfer/ag_news_shared_order/random_cross_s0/result.json
# repeat with --seed 1 -> cross_s1 / random_cross_s1
# repeat with --seed 2 -> cross_s2 / random_cross_s2
```
Dual (6):
```
python3 scripts/transfer_eval.py --task ag_news --path dual --init pretrained --seed 0 --examples 2000 --test-examples 500 --epochs 3 --steps 378 --batch 16 --lr 0.0005 --stratify --split-seed 0 --device cuda --ckpt results/pretrain/l1_tiny_byte/ckpt.pt --evidence-dir results/transfer/ag_news_shared_order/dual_pretrained_s0 --out results/transfer/ag_news_shared_order/dual_pretrained_s0/result.json
python3 scripts/transfer_eval.py --task ag_news --path dual --init random     --seed 0 --examples 2000 --test-examples 500 --epochs 3 --steps 378 --batch 16 --lr 0.0005 --stratify --split-seed 0 --device cuda --evidence-dir results/transfer/ag_news_shared_order/dual_random_s0 --out results/transfer/ag_news_shared_order/dual_random_s0/result.json
# repeat with --seed 1, --seed 2
```
Per-cell acceptance (else cell fails, is retained, and blocks pooling):
steps=378, test_unpackable=0, packed-gold assertions pass, protocol asserts
pass, provenance records `cross_shuffle` value and new source hashes.

## Scope label
Path comparison under one shared ordering policy. Residual bundle is
path-inherent only (cross-attention over concatenated pairs vs separate
context/candidate encoding + respective head + packing geometry); ordering is
NOT part of the bundle. No claim beyond this task/rung/device/mount.

## Execution gate
Separate go required. Nothing in this amendment authorizes running.
