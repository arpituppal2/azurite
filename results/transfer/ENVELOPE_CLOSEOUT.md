# Evaluation Envelope — Consolidated Close-out (v1–v9 + probes, 2026-11-04)

Versioned, durable record of what was proven, what failed, and what remains
open. Success criteria were NEVER changed after results: every envelope below
used the locked rule "SUPERIOR iff end-to-end (misses as errors) Wilson-95 LB
strictly exceeds the RECORDED Jev point". No historical evidence was deleted
or overwritten; void/launcher-error cells are labeled, not hidden.

## Proven (the single bounded win)

**Prompt-injections transfer, 2 seeds × 2 paths, all LB > recorded 0.741:**
- v1 s0 (`results/transfer/superiority_v1/SUPERIORITY_V1.json` `0b17623a…`):
  cross 0.8276 LB 0.7486; dual 0.9052 LB 0.8381.
- v5 s1 (`results/transfer/envelope_v5/`, pre-reg `bbe0cabe…`):
  cross 0.8879 LB 0.8176 (`be8b23a4…`); dual 0.8966 LB 0.8279 (`e2e582b2…`).
- Losses elsewhere (emotion, financial_phrasebank, banking77-dual) stand.

## Banking77 progression (bar: recorded 0.797), all BELOW except the void

| step | config | e2e (LB) | verdict / artifact |
|---|---|---|---|
| CPU baseline | tiny funnel M8 378/378 s0+s1 | 0.668 / 0.632 (0.626/0.589) | anchor, BELOW — `funnel_cpu_baseline/PROBE.md` `b41135b8…` |
| v4 GPU | tiny funnel M8 s0+s1 | 0.680 / 0.674 (0.638/0.632) | BELOW — `envelope_v4/ENVELOPE_V4.md` `170ada6f…`, results `b6b8c626…`/`eb1b5c42…` |
| v5-probe | tiny M16 / cross756 / combined | 0.686 / 0.686 / 0.724 | tiny ceiling ~0.72; head flat from step 126 (overfit) |
| v6 GPU | small M8 s0+s1 | 0.750 / 0.742 (0.710/0.702) | BELOW, rung lever +0.07 — `envelope_v6/` results `35becaba…`/`e0bea4eb…` |
| v6-probe | small M16/756 (+rows4k) | 0.804 / 0.804 (LB 0.767) | **measured ceiling**; rows 2000→4000 no gain |
| v7 | mid, my `--max-seq 256` flag | as-run 0.228/0.204, B hard-fail | **VOID** (launch-condition error: v4+v6 provably ran at 1024) — `ENVELOPE_V7.md` `2c655ea4…`, host-side only |
| v8 GPU | mid + RoPE-1024, A/B × s0/s1 | A 0.730/0.724, B 0.760/0.770 (best LB 0.731) | BELOW — `ENVELOPE_V8.md`, 12 files mirrored ALL-MATCH |
| v9 GPU | mid NATIVE-1024 anchor | A 0.068/0.104, B 0.088/0.238 | BELOW — **transfer-dead anchor** (dual stuck at chance); 8 files mirrored ALL-MATCH |
| retrain | mid-1024, lr 3e-4/wu 500 | smoke recall 0.22, e2e 0.034 | ALSO dead — line STOPPED, no third guess |

Mechanism findings: (1) small-rung cross head caps ~0.82 retrieved-only regardless of
M/steps/rows — capacity, not budget; (2) seq-1024 mid pretraining with tiny hparams
yields a transfer-dead anchor (rank diagnostic: top singular 23.0 vs 9.6, mean cosine
0.86 vs 0.78) while its seq-256 twin transfers — long-context training needs its own
hparams; (3) packing needs max_seq 1024 for M8 on banking77 (mean packed len ~283) —
verified from v4/v6 provenance, the lesson that voided v7.

## Byte-confirmation (independent on-host re-hash, this pass)

- v6 4/4, v8 12/12, v9 8/8 files: every local sha256 EQUALS the on-host
  re-hash (full 64-char equality, e.g. v6 `35becaba…e7e5acf93`).
- Mid ckpts host-side: `b78146e6…` (seq-256), `a5ed3ed8…` (seq-1024 v1),
  `751fe91e…` (seq-1024 v2); token stream `67809404…` matches locally byte-for-byte.
- GPU weights remain host-side (no local GPU-weight need; stated, not hidden).

## The 0.03 gap (declared, not solved)

Small-rung ceiling LB 0.767 vs bar 0.797. Needs a NEW idea (different head,
native-window larger rung with working hparams, or a different task) — not
more budget on known-dead paths. Criteria for any future envelope unchanged.

## Budgets consumed (measured, RTX PRO 6000 Blackwell, cuda True)

- Mid seq-256 pretrain 100M tok: 835 s. Native-1024 v1: 928 s. v2: ~15 min.
- Transfer cells: tiny ~25 s, small ~73–85 s, mid ~160–480 s (4 concurrent OK).
- Total GPU this workstream ≈ 1.5 h free allocation. Zero spend, zero charges.

## Blockers (founder-only)

1. Azurite license selection (UNRESOLVED — legal/founder decision).
2. Live-Jev re-query (founder-prepaid spend + endpoint auth) — all Jev sides recorded.

## Next executable actions

- New idea for the 0.03 banking77 gap, OR envelope on a new task (fresh pre-reg).
- Publish/deploy/push only on explicit founder authorization (none given).
- Resume files: `MISSION_RESUME.md`, `TASK_GRAPH.json` (this envelope + both).
