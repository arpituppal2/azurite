# Scorecard v1 — what is TRUE for an enterprise buyer (2026-11-03)

## Capability (fresh Azurite cells vs recorded Jev points, transfer recipe)

| Task | Azurite best (fresh) | Jev (recorded) | Verdict |
|---|---|---|---|
| prompt_injections | dual 0.9052 [0.8381,0.9462]; cross 0.8276 [0.7486,0.8855] | 0.741 | **AZURITE WINS (both paths)** |
| emotion | cross 0.1700 | 0.585 | BELOW |
| financial_phrasebank | dual 0.5920 | 0.73 | BELOW |
| banking77 | dual 0.6800 | 0.797 | BELOW (cross: harness K-cap, N/A) |

Rule (fixed before runs): SUPERIOR iff Wilson 95% lower bound > Jev point.
Scope: single seed s0, transfer recipe (378 steps, tiny byte), Jev as-recorded —
no live Jev re-query (founder-spend blocker). One task won, three lost.

## Operating (measured, RTX PRO 6000 fp32 unless noted)

- prompt_injections winner: dual 3.87 ms / cross 2.20 ms per decision batch-1
  (0.35/0.22 ms amortized batch-16); forward peak 37.1/34.4 MiB.
- AGNews reference: dual 3.96 ms / cross 2.08 ms batch-1.
- Full train cells (exec1): cross ~13.3 s wall / 2345 MiB; dual ~19.7 s / 3015 MiB.
- Not SLAs; no p99, no end-to-end serving, no K>>100 scaling curve.

## Deployability & license

- Azurite: self-hostable (local code+weights, no API). License UNRESOLVED —
  founder decision; blocks enterprise use. No SOC2/ISO.
- Jev: API-only (no on-prem evidence), proprietary + anti-distillation terms,
  $42/Btok in / free out, ZDR + SOC 2, no published region selector.

## Failure modes (declared, measured)

1. From-scratch dual collapses (6 variants, collapse-or-freeze); dual needs pretrained init.
2. Seed variance ±15pp dominates path effects.
3. Cross path K≤16 in this harness; K>>100 is dual/funnel territory (unvalidated at scale here).
4. Sweep-harness highs do not reproduce under the transfer recipe.

## Bottom line

Azurite beats Jev on exactly one measured task (prompt_injections, both paths)
with self-hosting and no per-token meter; it trails on three tasks, cannot run
cross at K=77, has no license, no certs, and no serving SLAs. That is the
envelope. Everything else is future work, not a claim.
