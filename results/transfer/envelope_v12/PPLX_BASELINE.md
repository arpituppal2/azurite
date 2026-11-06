# MEASURED BASELINE — pplx-decider-flash-v1.2 on our harness

Executed on Molab (RTX PRO 6000 Blackwell, 95 GiB), 2026-11-04.
Script: `scripts/run_pplx_baseline.py` (sha256 `8e90f275c99a2f9a8535ca23ad599083e9d60bcb5a7bf794e4c5997c0fdf1d1b`).
Artifacts: `pplx_banking77_s123.json`, `pplx_prompt_injections_s123.json`, `SMOKE_pplx.json`.

## What was run

The competitor's **own decision inference path** (`pplx_decider.Decider.from_pretrained(...)`,
`decider.decide([...])`), not a text-generation shim — the model card states ordinary
causal generation does not reproduce its decision behaviour, so a `generate()` call
would have measured a different object. Requests used the documented `choice`
primitive with our task's instruction and the full candidate bank as `criteria`.

Environment actually used, versus what the repo pins:

| | repo pins (`requirements.txt`) | used |
|---|---|---|
| torch | 2.14.0 | **2.11.0+cu130** |
| transformers | 5.17.0 | **5.14.1** |
| safetensors | 0.8.0 | host default |
| flash-linear-attention | 0.5.2 (`requirements-fast.txt`) | **absent** |

The package imported and the model loaded under the skew, and the card's decision
path produced sane outputs, but this is **unvalidated against the pinned stack**.
Its own log states: *"The fast path is not available because one of the required
library is not installed. Falling back to torch implementation."* So every number
below is the **reference path**, and the fast path was not measured.

## result — banking77, split seed 123, n = 1000, K = 77

| model | params | adaptation | accuracy | Wilson LB | s/request |
|---|---|---|---|---|---|
| **pplx-decider-flash-v1.2** | **4.54 B** | **zero-shot** | **0.906** | **0.8863** | **0.151** (GPU, batch 8) |
| Azurite-track C (v11 confirmatory) | ~14 M | fine-tuned, 2,000 rows | 0.910 / 0.919 | 0.8907 / 0.9004 | 0.0214 GPU p50 / 0.0494 CPU |
| Bar to clear | — | — | 0.797 | — | — |

Read this carefully, because it cuts both ways:

1. **On raw end-product accuracy the competitor ties us.** 0.906 sits inside our
   two-seed spread (0.910–0.919) and its Wilson lower bound overlaps ours. A 4.54 B
   pretrained, teacher-distilled model with a non-causal readout head does not beat
   a ~14 M model at this task; it also does not lose to it.
2. **It did that with zero adaptation.** We fine-tuned on 2,000 in-domain rows.
   Zero-shot-versus-fine-tuned is a real advantage in the competitor's column and
   it is the single most uncomfortable fact in this document: our deployment story
   currently requires per-task labelled data, and theirs does not.
3. **That advantage cannot be separated from contamination.** A 4.54 B Qwen3.5
   derivative scoring 0.906 on 77 fine-grained banking intents with no training is
   either an extraordinary zero-shot result or a memory of the task family. Qwen3.5's
   pretraining corpus almost certainly contains Banking77-style intent text, and the
   card's distillation teachers are undisclosed. We did not establish which it is —
   so this number is an **upper-bounded, contamination-unknown** figure and must
   never be cited as clean generalization.
4. **Cost of the tie.** ~324× the parameters (4.54 B vs ~14 M), CUDA required
   (~9 GiB BF16; the reference figures in the card are 39–54 GiB for long
   requests), 3–7× our per-request latency (0.151 s vs 0.021 s GPU / 0.049 s CPU),
   and it cannot run on the target CPU-only deployment at all in this form.

## result — prompt_injections, split 123, n = 116, K = 2

This suite is **our own generated data**, so contamination is far less plausible
than on Banking77. It was run precisely as a contamination probe.

| model | adaptation | accuracy | Wilson LB | s/request |
|---|---|---|---|---|
| **pplx-decider-flash-v1.2** | **zero-shot** | **0.9397** | **0.8807** | **0.022** (GPU, batch 8) |
| our v10/v11 fine-tuned cells | 2,000 rows | 0.974 / 0.948 | — | — |
| bar to clear | — | 0.741 | — | — |

**This partially refutes my own contamination hypothesis and I am recording it as
such.** Clean recall of a pretraining set cannot explain 0.94 on rows we
synthesised. So on this suite the competitor is genuinely deciding, not
remembering. It still sits below both of our fine-tuned seeds, and still needs
~324× the parameters and a CUDA GPU — but the case against it is now weaker than
the Banking77 result alone suggested, and the honest summary is "strong zero-shot
decider, contamination unresolved on public suites, clearly not contaminating on
ours".

## Calibration — the gap this measurement exposed on *us*

Our existing context cells already log ECE; the competitor publishes ECE 0.021
(their dev set). Ours, on boolq (`envelope_v11/CTXC_boolq_ctx_s{0,1}.json`):

| arm | n | accuracy | ECE |
|---|---|---|---|
| CTXC boolq s0 | 500 | 0.620 | **0.3698** |
| CTXC boolq s1 | 500 | 0.656 | **0.3334** |

An ECE of ~0.33–0.37 is not a rounding difference against 0.021; it is an order of
magnitude, on a different task and split, so it is not a head-to-head comparison —
but it tells us we cannot currently make any calibration claim at all, and that
confidence-gated routing (our own product argument) is unsupported until we fix it.
That is now the top queued experiment, ahead of any further architecture search.

## Negative result — evidence pooling is not the mechanism

Hypothesis under test: our passage encoder averages over all span tokens, so one
decisive sentence is diluted by hundreds of irrelevant ones; mean→max pooling
should recover evidence use. Implementation adds `--pool {mean,max}` to
`scripts/transfer_modernbert.py` (sha `825a6276c4fa585af19039cce32bd13735ab6daa076d4f212796b76c257f687c`),
applied to both the dual-encoder passage pooling and the cross-encoder span pooling.

`POOLMAX_*` = max pooling; `CTXC_*`/`CTX_*` = mean (historical). The shuffled
control feeds a random passage in place of the retrieved one, end-to-end.

| arm | batch | pool | real acc | shuffled acc | real − shuffled |
|---|---|---|---|---|---|
| CTXC boolq s0 | 8 | mean | 0.620 | 0.546 | **+0.074** |
| CTXC boolq s1 | 8 | mean | 0.656 | 0.608 | **+0.048** |
| MEAN boolq s0 | 16 | mean | 0.588 | 0.592 | −0.004 |
| POOLMAX boolq s0 | 16 | max | 0.576 | 0.548 | +0.028 |
| POOLMAX boolq s1 | 16 | max | 0.608 | 0.554 | +0.054 |
| CTX boolq s1 | 16 | mean | 0.638 | — | — |
| POOLMAX squad2 s0 | 16 | max | 0.652 | 0.570 | +0.082 |
| CTX squad2 s0/s1 | 16 | mean | 0.694 / 0.686 | 0.670 / 0.530 | +0.024 / +0.156 (unstable) |

**Rejected, and it replicates off boolq.** At matched batch 16, mean pooling scores
0.638 (boolq) and 0.694/0.686 (squad2); max scores 0.608/0.576 and 0.652 — max is
*worse on real accuracy on both suites*. The real-minus-shuffled gap does not widen
reliably: the squad2 mean arm's own gap swings from +0.024 to +0.156 between seeds,
so the max arm's +0.082 carries no signal. Max pooling also makes the model *more*
robust to shuffled evidence on boolq, which is the opposite of attention to
evidence: it latches onto strong activations regardless of whether they came from
the right passage. The matched batch-16 pair for seed 0 has since been completed
(mean 0.588/shuffled 0.592 vs max 0.576/0.548): max is worse on real accuracy by
1.2 pts and its gap (+0.028) is no better than mean's (−0.004). The s0 mean arm
itself shows no context effect at all, which further confirms that context use on
these suites is unstable across seeds rather than systematically improvable by
fixed pooling.

## Attention readout: ~~PASS on the pre-registered rule~~ — WITHDRAWN (see v17 CONTROL_DEFECT.md)

> **The PASS below is withdrawn.** Its acceptance rule was "real-minus-shuffled gap
> consistently positive", and the shuffled control was defective: it permuted
> passages *and* answers by the same index, so no row was ever mismatched and the
> control could not detect an ignored passage. Every gap in this section measured
> batch-composition noise. Attn's *real-accuracy* results below stand (it ties mean
> and beats max); the **selectivity claim does not**. See
> `envelope_v17/CONTROL_DEFECT.md` for the fix, the evidence, and the corrected re-run.

`--pool attn` (zero-init learned span weighting, both heads; starts mean-like).
Same batch-16 protocol, same shuffled control. Three-way split is reported
because recall, conditional accuracy and end-to-end accuracy are different claims:
here funnel_m=2=K, so recall is trivially 1.0 and misses are 0 — retrieval is
non-binding by construction on these K=2 suites, and conditional accuracy
coincides with end-to-end accuracy. That coincidence is itself a finding: on K=2
tasks the funnel cannot separate retrieval failure from decision failure.

| arm | batch | pool | recall | cond acc (= e2e, miss 0) | shuffled | gap |
|---|---|---|---|---|---|---|
| MEAN boolq s0 | 16 | mean | 1.0 | 0.588 | 0.592 | −0.004 |
| MEAN boolq s1 | 16 | mean | 1.0 | 0.638 | — | — |
| MAX boolq s0 | 16 | max | 1.0 | 0.576 | 0.548 | +0.028 |
| MAX boolq s1 | 16 | max | 1.0 | 0.608 | 0.554 | +0.054 |
| **ATTN boolq s0** | 16 | attn | 1.0 | **0.590** | 0.556 | **+0.034** |
| **ATTN boolq s1** | 16 | attn | 1.0 | **0.618** | 0.508 | **+0.110** |
| MAX squad2 s0 | 16 | max | 1.0 | 0.652 | 0.570 | +0.082 |
| **ATTN squad2 s0** | 16 | attn | 1.0 | **0.688** | 0.518 | **+0.170** |
| **ATTN squad2 s1** | 16 | attn | 1.0 | **0.700** | 0.636 | **+0.064** |
| **ATTN contract_nli s0** | 16 | attn | 0.838 | **0.528** e2e (0.630 cond) | 0.512 | **+0.016** |
| **ATTN contract_nli s1** | 16 | attn | 0.838 | **0.498** e2e (0.594 cond) | 0.496 | **+0.002** |

Judgment against the rule (accept = real-minus-shuffled gap consistently
positive): **PASS, bounded — replication pair closed 2026-11-05.** All four
attn gaps are positive (+0.034/+0.110 boolq, +0.170/+0.064 squad2); the s1
replication was lost to a sandbox outage and rerun clean on a fresh host
(sb-7df510dd4d142dc1, 225.7 s, identical recipe). But real accuracy only ties
mean (boolq 0.590/0.618 vs 0.588/0.638; squad2 0.688/0.700 vs mean s0 0.694)
— it does not beat it. So the mechanism is validated for
*evidence-selectivity*, not for accuracy: the learned weighting gates the
passage, but on these suites the gated evidence still does not convert to more
correct decisions. Adopt attn as the default pooling for context cells
(strictly dominates max, ties mean on accuracy with a better selectivity
signal, same ~230–330 s/arm cost); do NOT claim context-use is fixed. Absolute
levels (0.59–0.70) remain the binding deficit.

### Third suite (contract_nli): the rule does not extend — gap collapses to zero

contract_nli is a 3-class task with `funnel_m=2 < K=3`, so for the first time the
three claims separate instead of coinciding. Both seeds completed on a fresh host
(640.8 s / 640.3 s, identical recipe, `--pool attn`):

| arm | recall | cond acc (n=419) | miss | e2e (n=500) | shuffled e2e | gap |
|---|---|---|---|---|---|---|
| contract_nli s0 | 0.838 | 0.6301 | 81 | 0.528 | 0.512 | +0.016 |
| contract_nli s1 | 0.838 | 0.5943 | 81 | 0.498 | 0.496 | +0.002 |

Judgment: **the attention readout does not deliver context selectivity here.**
Both gaps are positive in sign, so a literal reading of "consistently positive"
is not violated, but +0.016 and +0.002 are zero in substance: the shuffled
control reproduces the real score to within 1.6 points on n=500, and the
difference between the two seeds' *real* accuracies (3.0 pts) is larger than
either gap. Pooling was never the binding constraint on this suite, and this arm
is the first direct evidence of that: the model answers nearly from priors
regardless of which passage it is shown. The earlier ATTN PASS is therefore
scoped to the K=2 answerability suites and must not be cited as a general
context-use result.

What the three-way split buys (the arithmetic of the deficit):

    e2e = recall x cond = 0.838 x 0.630 = 0.528  (s0)
                      0.838 x 0.594 = 0.498  (s1)

Retrieval is **not** the bottleneck: recall is 0.838 on both seeds, so 81/500
rows are unrecoverable errors by construction and e2e can never exceed 0.838.
The lost accuracy is in the conditional decision: 0.63/0.59 on rows whose gold
passage *was* retrieved. Raising recall cannot move e2e beyond 0.838; raising
conditional accuracy is the only lever. Combined with the near-zero shuffled
gap this points at the reranker's evidence reading, not the funnel, as the
binding constraint — and it is the first measurement in the programme that
prices the two stages apart on a non-degenerate split.

Provenance: `ATTN_contract_nli_s0.json` (sha256 `81004e06…`), `ATTN_contract_nli_s1.json`
(`ed852102…`), heads mirrored and sha-verified locally
(`70747bad…`, `f5f390a6…`). Exploratory, not confirmatory: split-123 train, test
accuracy printed per fixed eval step, no early stopping.

Hygiene note: these mechanism arms train on split-123 train and the training
log prints test accuracy each eval (fixed steps, no early stopping on test).
They are exploratory evidence, not confirmatory — no locked value depends on
them.

Consequence for the mission: the "mean pooling dilutes the decisive span"
explanation is wrong or at least not dominant, so the next architecture test is
**not** another fixed reduction. The competitor's actual mechanism is different
from either: `pooling: last` on a non-causal full-attention readout — a *learned
position* that can attend to everything, rather than any reduction over tokens.
That is the next thing worth testing, and it is a pooling-architecture question
that transfers to a scratch-trained encoder because it involves no pretrained
weights.

## Reproducing

```bash
python scripts/run_pplx_baseline.py --task banking77 --split-seed 123 \
    --test-examples 1000 --batch 8 --out results/transfer/envelope_v12/pplx_banking77_s123.json
```

The checkpoint resolves unauthenticated to
`~/.cache/huggingface/hub/models--perplexity-ai--pplx-decider-flash-v1.2/snapshots/9ff7cf98…`;
no token, no agreement, no charge was used.
