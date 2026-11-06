# Pivot review — stop scaling the legacy fixed-label byte path

**Date:** 2026-10-30.
**Status:** decision record and pre-registration for a development screen.
**Scope:** research direction, not a capability claim. No new model has been run
for this document.

This review is triggered by three substantial failures in one direction:
trying to close the Jev gap by adding capacity and bytes to the legacy
text-only, fixed-label byte encoder. The failures are not independent, and
repeating the same experiment with another head would not be progress.

## 1. The three failures

| # | Direction | Result | Source |
|---|---|---|---|
| F1 | More parameters on the legacy suite | 113.47M is 7.95x the 14.4M rung but wins only 4 of 9 cells, loses XNLI by 9.95 pp, and leaves 4 cells statistically unmoved. XNLI is a uniform predictor (94.7% one class). | `results/suite_large_113m/RESULTS.md` |
| F2 | Retrieval/query head and a longer packed window on Banking77 | Native query head 89.03% misses the 92.40% bar by 3.38 pp. The 5,120-byte joint fine-tune is 87.11%, 5.29 pp below the bar and 1.92 pp below the native head. The examples help (+29.90 over mean-pool) but do not buy parity. | `results/jev_query/`, `results/jev_query_5120/` |
| F3 | Extending the evaluation window without length adaptation | 256→4,096 fixed-checkpoint curve: **0 context-additive cells**, **4 degraded cells**. ContractNLI falls 69.79% → 37.03%; BoolQ and SQuAD 2.0 sit at majority. | `results/context_scaling/RESULTS.md` |

Supporting evidence, not counted as a fourth failure: candidate width matters.
E4 found width 2 much better than the full candidate set, so the decision
interface and candidate construction are live variables, not just encoder
capacity. See `results/e4_candidate_sweep/`.

## 2. Common cause

The failures point at the same bottleneck:

1. **Capacity is not the constraint.** A 7.95x larger encoder did not close the
   suite, and XNLI degenerated rather than improving.
2. **A longer window is not usable without training for it.** The 4,096-byte
   arm evaluates weights trained at 256; the curve is flat or worse, not
   evidence that a length-adapted model would fail.
3. **The fixed-label head cannot test the product contract.** Every
   Banking77-centric arm answers “which of 77 labels?” It cannot score an
   arbitrary caller-supplied alternative set, produce an abstention, or expose
   evidence spans.
4. **Single-seed, repeatedly inspected evidence cannot support a capability
   claim.** The dev test is development evidence; the locked suite is the final
   test and has not been opened.

## 3. Decision

**Stop the current direction.** No further work is authorised on:

- more parameters on the legacy fixed-`D` suite;
- another Banking77-only head or another 5,120-byte packing variant;
- 16k/32k full-window training until a length-adapted 4k arm passes the
  ContractNLI non-inferiority gate;
- opening the locked suite to diagnose a development result.

**Continue only with the hypothesis below.** Banking77 may appear only as one
control task. A Banking77 gain alone does not justify the next phase.

## 4. New falsifiable hypothesis (H1)

> On the five-task dev suite, a candidate-conditioned cross-encoder initialised
> from a strong pretrained text encoder will beat both TF-IDF and a
> same-interface byte cross-encoder by at least **3.0 points macro accuracy**,
> with a paired 95% bootstrap lower bound above zero, and will beat the majority
> class on ContractNLI by at least **10 points**. The gain must include the
> document-grounded and policy rows, not only Banking77.

H1 is about the **decision interface plus a pretrained encoder**, not about
adding parameters to the current fixed head. It is falsified if the gain is
Banking77-only, if no pretrained arm clears the byte cross-encoder, or if the
confidence interval includes zero.

This is a **development screen**, not a confirmatory result. The v1 dev test has
already been inspected for the context experiment and is legacy development
evidence. Any capability claim from H1 requires a new suite version with a
disjoint calibration partition and a fresh final evaluation; the locked suite
stays sealed.

## 5. Controlled comparison

All arms use the same five tasks, the same train/dev rows, the same candidate
construction, and the same maximum byte budget. Candidate construction is fixed
before training: for each row, the gold label plus a deterministic sample of
distractors, with the order permuted by a frozen seed.

| arm | role | provenance |
|---|---|---|
| Majority / class prior | simple baseline | measured from train labels only |
| TF-IDF (1–2 grams) + logistic regression | lexical baseline | same rows and budgets as the neural arms |
| Byte cross-encoder, 14.4M | previous Azurite version, same candidate interface | `results/pretrain/l1_small_byte_ctx256/ckpt.pt`, SHA-256 `8fd8bedee794ee4aa464af7c11fb06eb1d8dcfbfe82f529f8106a11a798ff131` |
| MiniLM frozen embeddings + logistic regression | cheap pretrained alternative | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0 |
| ModernBERT-base cross-encoder | strong pretrained alternative | `answerdotai/ModernBERT-base`, revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`, Apache-2.0 |

The 113.47M byte model is not a same-interface arm; it remains a legacy
reference on the old suite. It is not silently compared across splits.

### Tasks

- Banking77 and CLINC150: short-context controls.
- BoolQ: document-grounded boolean decision.
- SQuAD 2.0 answerability: document-grounded abstention.
- ContractNLI: policy/rule-conditioned decision and the long-document stress
  row.

### Metrics

- Primary: macro accuracy over the five tasks.
- Per-task accuracy with 95% Wilson intervals.
- Paired McNemar against the byte cross-encoder; paired bootstrap 95% interval
  for the macro delta.
- ContractNLI majority margin and SQuAD answerability majority margin.
- ECE and AURC are descriptive only. **No threshold or abstention policy is
  fitted**, because the v1 suite has no calibration partition.
- CPU and GPU p50 latency at the evaluation budget, if the arm is a plausible
  product candidate.

### Seeds and uncertainty

- Stage 1: one seed (0), all arms. This is a screen, not a capability estimate.
- Stage 2: only if Stage 1 clears the gate, rerun the winner and the byte
  cross-encoder at seeds 0/1/2 and report the run-to-run range.
- A single-seed win that disappears under the repeat is a failed screen.

## 6. Success, failure, and decision

**H1 is supported only if all of the following hold:**

1. A pretrained arm beats TF-IDF by at least 3.0 pp macro.
2. The same arm beats the byte cross-encoder by at least 3.0 pp macro, with a
   paired 95% bootstrap lower bound above zero.
3. ContractNLI is at least 10 pp above its majority class.
4. SQuAD 2.0 answerability is at least 5 pp above its majority class.
5. No task is more than 2.0 pp below the byte cross-encoder.
6. The Stage 2 three-seed range does not reverse the sign of the macro gain.

**If H1 fails:** stop the encoder-replacement direction. Do not run another
encoder sweep. The next proposal must be a different falsifiable mechanism,
such as retrieval-first decision or an explicit abstention head, with its own
baselines and gates. A failure of H1 is a result, not a reason to tune the
screen until it passes.

**If H1 passes:** proceed to a candidate-conditioned head and a suite v2 with a
disjoint calibration partition. Passing H1 is permission to run the next
experiment, not a product claim.

## 7. Compute budget

Projection, not a measurement. The screen uses 2,500 train rows per task, three
epochs, and the existing 256-byte packing for the byte arm; pretrained arms use
their native tokenizer with a matched byte cap.

| stage | arms | seeds | projected GPU-hours |
|---|---|---:|---:|
| Stage 1 | 3 learned arms × 5 tasks | 1 | 4–8 |
| Stage 2 | 2 arms × 5 tasks | 3 | 6–10 |
| **total cap** | | | **18** |

If the cap is exceeded, stop and report the partial screen. Do not silently
extend the budget. Data download, tokenizer checks, and failed starts count
against the cap.

## 8. Stop rules

- No locked-suite access for development, calibration, or diagnosis.
- No 16k/32k training until a length-adapted 4k arm passes ContractNLI.
- No Banking77-only claim; Banking77 is one control among five tasks.
- No calibration threshold without suite v2 and a disjoint calibration split.
- No “best of seeds” selection; report the pre-registered seed set.
- No cosmetic rerun of a failed arm under a new name.

## 9. What would change this decision

- A length-adapted 4k ContractNLI arm that is non-inferior to 256 would reopen
  the context direction, but it is a separate pre-registered experiment and
  does not authorise 16k.
- A retrieval-first baseline that beats the cross-encoder would redirect H1,
  not rescue the encoder path.
- A demonstrated licence or contamination block for the pretrained candidates
  would stop H1 before training; the byte cross-encoder remains the fallback
  baseline.

## 10. Provenance

- Dev suite: `research/suites/dev_suite.json`, SHA-256
  `43737b16b3ec85d3ffcdc264932c33499bc00417779a14eb8a526b4d49797630`.
- Locked suite: `research/suites/locked_suite.json`, SHA-256
  `1f2115ffab5ee0ca1959d534b424d99db1e3f286dce6d11e534b92fb058c87b2`;
  sealed, not opened.
- Context result: `results/context_scaling/RESULTS.md`, SHA-256
  `113a14db6543939b29f5c6658d4313e895866e0d01b7a23922298b934d3e2704`.
- Comparator classes: 13 published references, 1 local reconstruction,
  **0 live reproduced Jev runs**. No comparator row is used to set H1's
  thresholds.

## 9. Outcome (added after the run, 2026-10-30)

Stage 1 was executed once under the pre-registered protocol; Stage 2 was not
run. Result: **H1 fails.**

- ModernBERT macro 0.7750; TF-IDF 0.7545; byte 0.5740. ModernBERT clears byte
  by +20.11 pp (bootstrap 95% CI 18.64–21.65) but clears TF-IDF by only
  **+2.05 pp**, short of the +3.0 pp gate.
- SQuAD answerability is **0.6600**, 0.13 pp below the majority class; the +5 pp
  gate fails.
- ContractNLI is **0.6471**, 2.81 pp below the byte arm; the −2.0 pp floor
  fails. ContractNLI is +14.17 pp over majority, so that gate passes.
- The byte arm is best only on ContractNLI and is far behind on Banking77 and
  CLINC150. The candidate interface plus a strong pretrained encoder is a real
  improvement over the 14.4M byte arm, but not over TF-IDF at the pre-registered
  margin.

Per §6, the encoder-replacement direction stops. No Stage 2, no encoder sweep,
no seed selection. The 256-byte cap leaves both document-grounded rows
floor-limited; that is a limitation to diagnose separately, not a reason to
reinterpret the gate. Full records:
[`results/h1_stage1/RESULTS.md`](../results/h1_stage1/RESULTS.md).
