# Harness defect: the context-use control never permuted anything

Found and repaired 2026-11-05 (envelope v17). This is a **retraction of every
real-minus-shuffled gap reported before this fix**, and a correction to the verdicts
that were built on them.

## The defect

The control was meant to answer one question: does the model read the passage, or is
it answering from priors? It does that by scoring the same model with each question
paired to a *different* passage — accuracy must collapse if the passage matters.

What the code actually did:

```python
order = list(range(len(test_ctx)))
_random.Random(args.seed).shuffle(order)
perm_ctx  = [test_ctx[i] for i in order]      # passages moved
perm_gold = [test_gold.tolist()[i] for i in order]   # answers moved the SAME way
```

Passage and answer were permuted by the same index, so row `j` still held its own
true `(passage, answer)` pair. The "shuffled" arm was the same task with its rows
reordered: nothing was mismatched, and no amount of accuracy collapse could occur.
The only differences it could detect were batch-composition and retrieval-tie
effects. Every gap in the +0.002…+0.170 range is exactly that.

The two pieces of evidence that exposed it, both internal to this session:

1. The passage-swap margin arm scored **higher** on the control (0.536) than on the
   real pairing (0.500) — a genuine passage swap cannot help.
2. Implied conditional accuracy in the control (shuffled_e2e / recall) was 0.592–0.611
   against 0.594–0.630 real: a model that is genuinely reading an 11k-character legal
   passage does not land within half a point of its wrong-passage score. That
   implausibility is what prompted reading the control code instead of trusting the
   number.

Independent confirmation on split 0 with the class-prior baseline computed on the host:
prior-only conditional accuracy is **0.418**, the trained model reaches **0.592** —
so the model is *not* prior-bound (it beats priors by 17 pts) and *is* doing something
with the input. The defective control had been reporting that it did nothing.

## What this invalidates

* `PPLX_BASELINE.md` **"Attention readout: PASS on the pre-registered rule"** — the rule
  was "real-minus-shuffled gap consistently positive", measured with this control.
  The PASS is **withdrawn**. (Attn's *real* accuracy numbers are unaffected: it ties
  mean on accuracy and beats max. Only the selectivity claim fails.)
* `POOLING_EXHAUSTED.md`'s framing that both readouts sit "within 2 points of their
  shuffled control" — the comparison is void, so "pooling is exhausted" is no longer
  established by that argument. The real-accuracy ordering (attn 0.528 > last 0.504 on
  contract_nli s0, both above max) still stands.
* Every `context_control_shuffled_e2e` field in JSONs written before this fix
  (`ATTN_*`, `LAST_*`, `POOLMAX_*`, `MEAN_*`, `DEV_*`, the context-track cells).

## What survives

Everything that does not depend on the control: recall, conditional and end-to-end
accuracies, the three-way split on contract_nli, the paired pplx comparison
(`EXTERNAL_BASELINE_PAIRED.md`, which is a paired McNemar on real predictions), the
latency numbers, and the calibration results.

## Second defect found while repairing it: checkpoints are not self-sufficient

`heads.pt` saved only the two heads. Phases A and B also fine-tune the encoder, so a
saved arm **cannot re-score itself**: loading `heads.pt` against the stock pretrained
encoder reproduced 0.42 end-to-end where the run had reported 0.504 (verified on
`LAST_contract_nli_s0`). A checkpoint that cannot reproduce its own number is not a
durable artifact, and it also cannot be served.

## Repairs

1. `scripts/transfer_modernbert.py`: the control now permutes passages with **gold in
   place** (`perm_gold = test_gold.tolist()`), and every new JSON records
   `context_control_gold_fixed` so a pre-fix number can never be mistaken for a
   post-fix one.
2. `scripts/rescore_control.py`: re-scores an arm's saved heads under all three
   definitions (real / corrected / old) plus the class-prior baseline, so the size of
   the defect is measurable rather than asserted.
3. `--save-encoder`: the encoder is checkpointed with the heads (~600 MB, on-host
   only — too large to mirror; recorded so no one assumes a 4.7 MB `heads.pt` is a
   backup).
4. A corrected re-run (`CTRLFIX_attn_contract_nli_s0`, dev split 0) measures the
   context gap under the fixed control for the first time.

## What the corrected control says (dev split 0, contract_nli_ctx, seed 0)

Re-run end to end with the fixed control, so nothing depends on the defective field:

| pool | real e2e | cond acc | shuffled (fixed) | **gap** | old (defective) gap | prior baseline |
|---|---|---|---|---|---|---|
| attn | 0.524 | 0.6253 | **0.402** | **+0.122** | +0.016 (split 123) | 0.418 |
| mean | **0.526** | 0.6232 | **0.402** | **+0.124** | −0.006 | 0.418 |
| last | 0.518 | 0.6181 | 0.414 | **+0.104** | +0.002 | 0.418 |

All three readouts are within 0.8 pts of each other and share a ~+0.10…+0.12 gap;
the differences are smaller than seed-to-seed spread measured earlier. **Default stays
mean** (fewest parameters, no tuning, marginally best here); attn and last are retained
as options, and the v12 "adopt attn" decision is formally reversed.

Three things follow, and all three are the opposite of what the defective control
suggested:

1. **The model does read the passage.** Destroying the pairing costs 12.2–12.4 points
   end-to-end. The earlier conclusion that the reranker was ignoring its input was an
   artifact of the broken shuffled multi-set.
2. **The fixed control behaves as a control should**: shuffled 0.402 lands on the
   class-prior baseline 0.418 (slightly below, which is right — a random passage
   actively misleads). That coincidence is the sanity check the old control failed.
3. **Pooling still does not matter**: mean and attn are 0.526/0.524 real and the same
   0.402/0.402 shuffled. The v12 preference for attn over mean was never supported by
   evidence, and it should not be cited as a mechanism result. Attn vs max (max was
   genuinely worse on real accuracy) is the only pooling claim that survives.

The real deficit is therefore **absolute level, not mechanism**: 0.52 e2e / 0.62
conditional, with +12 pts of genuine passage dependence already present. Raising the
level means better use of an 11k-character passage that is currently truncated to
~1024 tokens (mean 11,414 chars, max 35,310 — roughly two-thirds of each document is
never seen). That is the next arm, and it is a data-side hypothesis with a cheap A/B:
`--max-len` 2048 vs 1024 on the same dev rows.

## Standing rule going forward

A control must be shown to *fail when it should*. Before any mechanism claim rests on
a real-minus-control gap, the control must be sanity-checked by confirming that the
marginal-accuracy identity holds (shuffled ≈ chance or prior baseline for a
passage-dependent task) — not merely that the gap has the right sign.
