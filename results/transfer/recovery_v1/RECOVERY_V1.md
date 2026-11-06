# Recovery v1 — staged-transfer loss triage + one bounded fix attempt

## Diagnosis (from pulled train histories of scorecard_v1 losers)

1. **emotion, both paths — collapse.** Train NLL pins *exactly* at ln 6 = 1.792,
   the uniform-output signature (same family as dual-random). Dead end at the
   378-step budget; no cheap fix exists. No attempt made.
2. **financial_phrasebank, both paths — underfit plateau.** Train NLL flat at
   ~0.86. Not collapse, not rising: 2x steps is unlikely to add the ~14pp
   needed versus the recorded Jev point. No attempt made.
3. **banking77-dual — budget truncation (the fixable one).** At step 377 the
   curve is still rising (train NLL 1.499 → 1.186, acc climbing). Fix: 2x steps.
4. **banking77-cross — structural, not a model loss.** Harness cross K-cap is
   16 < banking77 K=77. Fixing it means changing the harness, out of scope for
   a bounded model-fix spike. Recorded, not attempted.

## Fix attempt (pre-registered, criteria locked before launch)

- Cell: `banking77_dual_s0_x2`, 756 steps, otherwise identical recipe.
- Fix-success bar: final acc ≥ 0.71 AND train NLL descending from step 0.
- Win bar: Wilson LB(95%) > recorded Jev 0.797.
- Result: acc **0.722** (up from 0.68), train NLL 1.6331 → 0.3102 descending,
  wall 19.0 s, rc 0. **FIX SUCCESS.** Wilson LB = 0.6812 < 0.797 → **NO WIN**.
- Winning envelope is unchanged (prompt_injections only).

## Preserved-win checks (identical-recipe reruns, seed 0)

- `pi_cross_s0_check`: acc 0.8534 (orig 0.8276), LB **0.7777** > 0.741 — PRESERVED.
- `pi_dual_s0_check`: acc 0.8362 (orig 0.9052; delta is seed/GPU noise),
  LB **0.7583** > 0.741 — PRESERVED.

## Hashes (host-authoritative)

| artifact | sha256 |
|---|---|
| banking77_dual_s0_x2/result.json | a2078150…ab67e449 |
| pi_cross_s0_check/result.json | 0894e681…7ca282f |
| pi_dual_s0_check/result.json | 09d1b427…ad1ed79f |
| scorecard_v1/SCORECARD_V1.json (local match ✓) | 3794ecbe…1988d97b |
| superiority_v1 pi_cross result.json | a3f088c5…e9ecaaa39 |
| superiority_v1 pi_dual result.json | 418a6d50…8e377bfdc |

Provenance flag: local copies of the two superiority_v1 pi result.json differ
from host; host hashes above are authoritative and local files were NOT
overwritten.

## Budget

3 cells, 46.9 s GPU total (19.0 + 10.9 + 17.0). No large sweeps.
Dual-random-from-scratch stays CLOSED.
