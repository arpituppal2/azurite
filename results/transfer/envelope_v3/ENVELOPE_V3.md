# Envelope v3 — emotion collapse is seed-stable

## Pre-registration (locked before launch, unchanged after results)
`PREREG_V3.json` (host sha `887711e6…`, host mtime 2026-11-03T23:44:46Z, stamped
23:44:43Z — before any cell ran): 4 cells (emotion s1/s2 × cross/dual), exact
CLI recipe, locked rule (SUPERIOR iff Wilson-95 LB strictly exceeds recorded
Jev 0.585), hypothesis stated in advance (seed-stable collapse → expect BELOW;
a SUPERIOR would reopen the diagnosis). Single launch only.

## Results — all rc=0, measured walls, 7-row histories
| cell | acc | NLL | LB(95%) | Jev | wall | verdict |
|---|---|---|---|---|---|---|
| emotion cross s1 | 0.158 | 1.7917 | 0.1287 | 0.585 | 10.0 s | BELOW |
| emotion dual s1 | 0.206 | 1.7908 | 0.1729 | 0.585 | 12.4 s | BELOW |
| emotion cross s2 | 0.144 | 1.7912 | 0.1159 | 0.585 | 9.8 s | BELOW |
| emotion dual s2 | 0.134 | 1.7918 | 0.1069 | 0.585 | 12.1 s | BELOW |

NLL 1.7908–1.7917 = ln 6 (1.791759) at all seeds, both paths; accuracy at
chance. The recovery-v1 collapse diagnosis replicates at s1/s2: seed-stable,
not a seed-0 accident. No-fix decision stands; nothing reopens.

## Envelope state (TRUE under the locked rule)
prompt_injections SUPERIOR at 3 seeds × both paths; emotion BELOW at 3 seeds ×
both paths; financial_phrasebank BELOW at s0+s1; banking77-dual BELOW (incl.
0.722 fix); banking77-cross STRUCTURAL (K-cap 16 < 77). Winning envelope
remains prompt_injections only — v3 adds losing-envelope evidence, which is
what a truthful envelope version does.

## Timing repair for v2 (from authoritative host artifacts, no re-runs)
v2 wall logs were lost to the duplicate-launch overwrite. Measured training
compute from per-cell result.json `seconds` fields (first-run products,
evidence intact): 5.0+7.5+6.1+6.1+12.3+12.3 = **49.3 s**. v3 measured this
turn: wall 44.3 s (10.0+12.4+9.8+12.1), training compute 24.6 s
(5.0+4.8+7.4+7.4). No large sweeps in either version.

## Hashes (host-authoritative)
v3 cells: cross-s1 `67206a45…`, cross-s2 `11dfe75c…`, dual-s1 `3e7161cf…`,
dual-s2 `4b03c3e8…`. Traces: envelope_v2 `bf44e094…`, recovery_v1
`56e80752…`, scorecard_v1 `3794ecbe…`.
