# Envelope v7 — verdict: VOID (wrong window), numbers recorded as-run

## What happened

The 4 cells ran with `--max-seq 256` (my flag, NOT in the pre-reg). At
256, M8 packing needs mean length ~241+ and only 146/500 test rows fit —
326 structural drops on top of retrieval misses. As-run numbers:

- A_s0 e2e 0.2280 (recall 0.944, miss 354), A_s1 e2e 0.2040 (recall 0.938,
  miss 371). B_s0/B_s1: hard failure, "no packable examples at max_seq=256".

## Why VOID, not BELOW

Post-mortem of local provenance showed v4 (tiny) and v6 (small) BOTH ran
at max_seq 1024 (mean packed len ~283 > 256 — the funnel NEEDS >256 for M8
on banking77). My 256 flag did not reproduce "v4 conditions except rung";
it created a condition where the funnel structurally cannot run. The
pre-reg's silence on max_seq does not cover this: the stated intent
("identical to v4 except rung/anchor") was violated by the flag.

## Artifacts (kept, labeled void)

- Host `results/transfer/envelope_v7/` A_s0/A_s1 JSONs + evidence + logs
  (B logs show the packing failure). NOT mirrored locally — void cells do
  not enter the evidence record. (Saved host-side only for the post-mortem.)

## Follow-up

v8 pre-registers the ACTUAL v4-matched condition explicitly: mid anchor,
1024 window via `--allow-context-extension`, configs A (M8/378) and B
(M16/756), seeds 0+1, rule unchanged.
