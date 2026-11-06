# Envelope v2 — seed-replicated prompt_injections + financial_phrasebank probe

## Pre-registration (locked before launch, unchanged after results)
`PREREG_V2.json` (host sha `0b505c22…`, host mtime 2026-11-03T23:39:38Z, stamped
23:39:21Z — before any cell ran): 6 cells, exact CLI recipe, locked rule
(SUPERIOR iff Wilson-95 LB strictly exceeds the recorded Jev point), both wins
and losses reported, no re-seeding, per-cell timeout 180 s.

## Results — all 6 cells rc-logged, result.json valid, 7-row histories
| cell | acc | LB(95%) | Jev | verdict |
|---|---|---|---|---|
| pi cross s1 | 0.8276 | 0.7487 | 0.741 | SUPERIOR (thin, +0.0077) |
| pi dual s1 | 0.9224 | 0.8591 | 0.741 | SUPERIOR |
| pi cross s2 | 0.8448 | 0.7680 | 0.741 | SUPERIOR |
| pi dual s2 | 0.8879 | 0.8176 | 0.741 | SUPERIOR |
| fpb cross s1 | 0.5520 | 0.5082 | 0.73 | BELOW |
| fpb dual s1 | 0.5920 | 0.5484 | 0.73 | BELOW |

## Envelope state (TRUE under the locked rule)
prompt_injections is SUPERIOR at 3 result seeds × both paths (s0 verdict cells +
preserved-win reruns + s1/s2 — 6/6 fresh measurements above the bar, one thin).
financial_phrasebank BELOW at s0+s1 both paths; emotion BELOW (collapse);
banking77-dual BELOW (0.68; 0.722 fix still below 0.797); banking77-cross
STRUCTURAL (K-cap 16 < 77). The winning envelope is prompt_injections only —
now seed-replicated rather than single-seed.

## Incident log (no data harmed)
The run script was launched twice ~100 s apart. transfer_eval's evidence guard
refused every overwrite on the second pass (rc=1), so all result.json are
single-clean-run products. Second-pass log walls (1.8–1.9 s) describe the
refusal; first-run walls were overwritten — GPU spend is a labeled estimate
(~65–115 s from the measured 10.9–19.0 s/cell range), not a measurement.

## Hashes (host-authoritative)
Cell result.json shas: pi-cross-s1 `654028a3…`, pi-dual-s1 `095c1929…`,
pi-cross-s2 `51711d5f…`, pi-dual-s2 `797d9c15…`, fpb-cross-s1 `2c6bd9eb…`,
fpb-dual-s1 `87418927…`. Traces: scorecard_v1 `3794ecbe…`, recovery_v1
`56e80752…`, superiority pi cells `a3f088c5…` / `418a6d50…`.
