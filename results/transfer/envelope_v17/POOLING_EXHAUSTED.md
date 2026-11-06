# Pooling is exhausted as an explanation for the context deficit

> **CORRECTION (same session):** the argument below rested on the shuffled control,
> which was defective — it permuted passages and answers together, so its gaps were
> noise (see `CONTROL_DEFECT.md`). Under the fixed control the model turns out to read
> the passage substantially: on dev split 0, attn scores 0.524 real vs **0.402**
> shuffled, with the shuffled score landing on the 0.418 class-prior baseline. So
> "the reranker is not reading the passage" below is **wrong**; what survives is the
> *real-accuracy* ordering (attn > last, both > max) and the fact that the absolute
> level, not the presence of context use, is the deficit. Read the measurements below
> as real-accuracy results only.

Envelope **v17**, Molab RTX PRO 6000, 2026-11-05. Artifacts:
`LAST_contract_nli_s0.json` (`c4c65336…`), `ATTN_contract_nli_s0/s1.json`
(`81004e06…`, `ed852102…`), heads `6708c5a7…` (last s0), `70747bad…`, `f5f390a6…`
(attn s0/s1) — all mirrored and sha-verified locally.

## What was tested

`--pool last` — the competitor's own mechanism class (`pooling: last`): a single
learned readout position inside each span that has already attended over the whole
passage, rather than any reduction over span tokens. It adds **zero parameters**
(confirmed: same 4,730,871-byte head file as every other arm), so any difference
cannot be attributed to capacity.

Tested on `contract_nli_ctx` first, because that is the one suite where pooling
had something to prove: the attn readout's shuffled-control gap had already
collapsed to +0.016/+0.002 there.

## Results — all three readouts, same recipe, same rows

| arm | recall | cond acc (n=419) | e2e (n=500) | shuffled e2e | e2e gap | implied cond (shuffled) | cond gap |
|---|---|---|---|---|---|---|---|
| attn s0 | 0.838 | 0.6301 | 0.528 | 0.512 | +0.016 | 0.6110 | **+0.019** |
| attn s1 | 0.838 | 0.5943 | 0.498 | 0.496 | +0.002 | 0.5919 | **+0.002** |
| last s0 | 0.838 | 0.6014 | 0.504 | 0.502 | +0.002 | 0.5990 | **+0.002** |
| last s1 | 0.838 | 0.6062 | 0.508 | 0.508 | **+0.000** | 0.6062 | **+0.000** |

The "implied cond" column divides the shuffled end-to-end score by the same recall
(the permutation preserves row count and recall is computed within the permuted
pairing), giving conditional accuracy when the passage is **wrong**. That is the
decisive column: the model scores 0.594–0.630 with the right passage and
0.592–0.611 with the wrong one.

## Verdict

**Pooling architecture is exhausted as an explanation. Change the readout and the
behaviour does not change.**

* Mean → max: rejected earlier (worse accuracy, no wider gap).
* Mean → attn: selectivity gains on the K=2 suites, none here.
* attn → last (the competitor's mechanism): real accuracy 0.528 → 0.504, gap
  +0.016 → +0.002. Slightly *worse*, and the confidence-weighted readout that
  helped answerability tasks does nothing for evidence reading.

Two independent readout families, two seeds each, now leave conditional accuracy
within 2 points of its wrong-passage value — and the `last` readout lands at exactly
0.000 on seed 1 (0.508 real vs 0.508 shuffled). The model is not weighting the passage badly; it is
barely reading it. Since the same recipe *does* show a real passage dependence on
boolq/squad2 (+0.03 to +0.17), the deficit is task-shaped, and it lives in what
the reranker is trained to do — scoring K candidates independently against a
shared passage, where the passage is constant across candidates and therefore
carries no discriminative gradient. Nothing in the loss ever rewards attending to
it.

## Next intervention (designed, not launched)

**Passage-swap margin loss.** Keep the current architecture and add, per training
step, a second forward pass of the same K candidates against a *different* row's
passage, with a margin term requiring `score(gold | right passage) >
score(gold | foreign passage) + m`. This is the cheapest objective that can only
be satisfied by reading the passage: the candidate set is identical between the
two passes, so any score difference is attributable to the passage alone. No new
parameters, no new head, no distillation, and it is the direct complement of the
shuffled control that already measures the failure.

Alternative if that fails: raise `funnel_m` so the reranker must discriminate among
more than 2 candidates, where priors alone cannot reach 0.60.

## Backlog (capability gaps, ranked by expected information)

1. **Context-use absolute levels** — cond 0.59–0.63 (contract_nli) and 0.59–0.70
   (boolq/squad2). Binding deficit; pooling is ruled out; training signal is next.
2. **Conditional-accuracy ceiling under correct retrieval** — e2e cannot exceed
   recall 0.838 on contract_nli, but that ceiling is not what is being hit.
3. **Deployment/serving** — latency already measured (GPU 21.4 ms p50 / 24.2 p99 at
   K=77; CPU 49.4 ms vs Jev 283 ms) and untouched this session.
4. **Banking77 intent classification** — **closed**. Paired test vs
   `pplx-decider-flash-v1.2` is a tie with a negative point estimate
   (`EXTERNAL_BASELINE_PAIRED.md`); further tuning there has no product value.
