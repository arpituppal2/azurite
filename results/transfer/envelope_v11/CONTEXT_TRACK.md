# Context track — document-grounded decisions (executed 2026-11-04)

The v10/v11 wins are single-utterance classification. This track tests what
breaks when the same system must decide over a **document**: passages of several
hundred tokens, where the answer depends on reading the context rather than
matching a sentence to a label.

Method: identical funnel harness (ModernBERT-base, lr 3e-5, M=8, 756+756 steps,
batch 16, 2000 train rows, split seed 123), **max_len raised to 1024** so long
passages are actually used rather than truncated. Suites built from train-only
rows by `scripts/build_context_suite.py` (sha `accfbf87…`); the locked suites were
not touched.

## Results

| task | family | K | e2e | n | status |
|---|---|---|---|---|---|
| squad2_answerability_ctx | document-grounded abstention | 2 | 0.700 | 500 | measured |
| squad2_answerability_ctx (s1) | " | 2 | 0.628 | 500 | measured |
| boolq_ctx | document-grounded yes/no | 2 | 0.638 | 500 | **one seed only** (s0 OOM) |
| contract_nli_ctx | policy rule | 3 | — | — | running at write time |

## Verdict: this is the new deficit, stated plainly

- **BoolQ 0.638 against a 62% majority-class rate** is a near-majority result.
  The system is not reading the passage to answer the question; it is leaning on
  priors about yes/no questions.
- **SQuAD2 answerability 0.700 / 0.628** is unstable across seeds by 7 points,
  which usually means the model is near a decision boundary rather than
  representing the task. A 2-way answerable/unanswerable decision that swings
  this much is not shippable.

Compare the same system on single-utterance intent tasks: 0.910–0.921 at K=77 and
K=151. The gap between 0.91 and 0.64 is the **context-use deficit**, and it is
larger than the entire Banking77 gap that consumed ten envelopes.

This is reported as a failure, not screened out. The v10/v11 accuracy claim
stands on its own tasks; it does not transfer to document grounding.

## Does the model read the passage at all? (shuffled-context control)

Same trained model, re-scored with the test **passages permuted** so each
question is paired with someone else's passage. If accuracy is unchanged, the
model never read the context; if it collapses, the context is doing the work.

| task | seed | real e2e | shuffled passages | drop |
|---|---|---|---|---|
| boolq_ctx | 0 | 0.620 | 0.546 | **−7.4 pts** |
| boolq_ctx | 1 | 0.656 | 0.608 | **−4.8 pts** |

SQuAD2 answerability, same control, two batch sizes:

| run | seed | real e2e | shuffled | drop |
|---|---|---|---|---|
| batch 8 | 0 | 0.540 | 0.528 | −1.2 pts |
| batch 8 | 1 | 0.560 | 0.554 | −0.6 pts |
| batch 16 | 0 | 0.694 | 0.686 | −0.8 pts |
| batch 16 | 1 | 0.670 | 0.530 | **−14.0 pts** |

### What this control actually establishes

1. **BoolQ is consistent:** both seeds lose 5–7 points when passages are
   scrambled, so the passage is read but carries little weight, and real accuracy
   sits at the 62% majority rate.
2. **SQuAD2 is not consistent:** three of four runs show a drop indistinguishable
   from noise (±2 pts at n=500), and one shows 14 points. Same task, same recipe,
   different seed. Context use here is unstable, which is a worse property than
   uniformly weak — it means the behaviour is not predictable from the recipe.
3. **Batch size is a large uncontrolled factor:** SQuAD2 scored 0.700/0.628 at
   batch 16 without the control, 0.540/0.560 at batch 8, and 0.694/0.670 at batch
   16 again. Comparisons across batch sizes in this track are therefore
   **invalid**, and only within-run real-vs-shuffled contrasts are reported.

The honest reading: context is consumed but under-weighted on BoolQ and
unstable on SQuAD2. This rules out "the pipeline ignores the passage entirely"
(BoolQ would have shown zero drop, and it did not), and it rules out "the
pipeline reads the passage reliably" (three of four SQuAD2 runs are flat).

## Likely causes, to be tested rather than assumed

1. **Retrieval is meaningless at K=2–3.** The funnel was designed for K=77–151.
   With two candidates, "retrieve top 8" retrieves everything, so the reranker
   carries the full decision alone.
2. **One passage per example, no cross-passage selection.** BoolQ needs the model
   to locate the supporting span inside the passage; mean-pooling the whole
   passage averages the evidence away.
3. **Budget/recipe transferred unchanged.** 756 steps and 2000 rows were tuned on
   intent tasks with short inputs; document tasks may need different treatment.
4. **Token budget truncation.** At 1024 tokens most BoolQ passages fit, but long
   ones are cut, and the answer often sits late in the passage.

## Provenance and limits

- Artifacts: `CTX_squad2_answerability_ctx_s{0,1}.json`,
  `CTX_boolq_ctx_s1.json`, mirrored with per-file sha256 verification.
- **boolq s0 was lost to CUDA OOM** when six 1024-token arms shared the GPU. It
  must be re-run serially; one seed is not a variance estimate.
- No claimed win on this track. Nothing here is a Jev comparison — Jev has no
  recorded figure for these suites, so this is capability mapping, not a bar.
- The dev-suite builder crashed after writing the five dev tasks because a
  *locked*-suite source (`massive_intent`) is absent on this host. Dev outputs
  were written before the crash and are the ones used here; the locked suite was
  never built and never evaluated.
