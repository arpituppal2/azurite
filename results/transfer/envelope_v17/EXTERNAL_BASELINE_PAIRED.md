# External baseline, measured paired — and why the earlier claim does not survive

Executed on Molab (RTX PRO 6000 Blackwell 96 GiB), 2026-11-05, envelope **v17**.
Artifacts: `PAIR_b77_s123_s0/s1.json`, `pplx_banking77_ctx_s123.json`,
`PAIRED_b77_s0/s1_vs_pplx.json`, `PREDS_b77_ctx_s0/s1.json` (all sha-verified
locally; the full evidence dirs `ev_PAIR_*` stay host-side because they are
large, but their per-row predictions are captured in `PREDS_*`).

## The defect this closes

The competitor numbers in `envelope_v12/PPLX_BASELINE.md` (b77 0.906, prompt
injections 0.9397) were compared to our 0.910/0.919 as **two independent point
estimates**. That is not a comparison: without the same rows there are no
discordant pairs, so no test exists and the gap could be entirely row-to-row
noise. Worse, a provenance check this session showed the two were also measured
on **different data builds**:

| | v11/v12 cells (old sandbox) | this session's artifact (fresh host) |
|---|---|---|
| train fingerprint | `3b28f7873574ab94` | `a238d69cbec57b9d` |
| test fingerprint | `e3aef745a26bd9d6` | `ef7c4853ab30d803` |
| train rows available | 4000 | 2500 |
| test rows available | 1000 | 750 |

The old build is gone (sandbox reclaimed, evidence dirs lost, only hashes
ledgered), so it cannot be re-scored. Every cross-build b77 comparison in the
programme is therefore **void as a paired statement** and must be read as two
unrelated measurements of two different datasets.

## Provenance of the artifact that *is* reproducible

`data/raw/banking77_ctx` as staged to the host, measured on the host:

* train 2500 rows, test 750 rows; 77 classes in each.
* `source_split` is **`train` for every row in both splits** — the ctx artifact
  is carved from the official banking77 *train* split, not its test split.
* Leakage checked, not assumed: **0** shared `id`s between the splits and **0**
  shared `(context, label_text)` pairs.

So the artifact is clean but is a *reduced held-out slice of the train split*
(3250 of ~13k rows), not the official test set. Numbers from it are internally
comparable and reproducible; they are not comparable to a published Banking77
test score, and the earlier v12 table was not either.

## Method

Both systems scored on **identical rows in identical order**: both scripts call
`stratified_examples(spec, bank, "test", 1000, split_seed + 1)` with
`split_seed=123`, so rows, order and gold vector coincide — `scripts/paired_test.py`
refuses to run if the gold vectors differ, which is what makes the pairing sound
rather than assumed.

* candidate: `scripts/transfer_modernbert.py`, funnel m=8, K=77, 2500 train rows,
  756/756 steps, batch 16, lr 3e-5, `--pool mean`, seeds 0 and 1.
* baseline: `scripts/run_pplx_baseline.py`, `pplx-decider-flash-v1.2` via its own
  `Decider` path, **zero-shot** (no task training), 750/750 rows answered,
  0.106 s/request.
* statistic: exact McNemar on the discordant cells (no normal approximation),
  plus a 5000-iteration paired bootstrap CI.

## Results

| system | seed | n | accuracy | correct |
|---|---|---|---|---|
| Azurite funnel (fine-tuned, 2500 rows) | 0 | 750 | 0.8707 | 653 |
| Azurite funnel (fine-tuned, 2500 rows) | 1 | 750 | 0.8613 | 646 |
| **pplx-decider-flash-v1.2 (zero-shot)** | — | 750 | **0.8787** | **659** |

| contrast | delta (ours − pplx) | ours right / pplx wrong | ours wrong / pplx right | McNemar p | 95% CI | P(ours better) |
|---|---|---|---|---|---|---|
| seed 0 vs pplx | **−0.0080** | 40 | 46 | 0.590 | [−0.0333, +0.0160] | 0.244 |
| seed 1 vs pplx | **−0.0173** | 39 | 52 | 0.208 | [−0.0413, +0.0080] | 0.081 |

## Verdict

**No measured advantage over the external baseline on banking77. The two systems
are statistically indistinguishable, and the point estimate is negative on both
seeds.** The discordant cells are almost balanced (86 vs 98 across both seeds),
which is what a tie looks like on 750 rows — not a large-sample near-miss.

This retracts, explicitly:

* "Azurite 0.910/0.919 beats pplx 0.906 at banking77." It was never paired, and
  the artifact behind our side is unrecoverable. On the only artifact both
  systems can be scored on together, the ordering is **pplx ≥ Azurite**.
* Any reading of the v12 table as evidence of external superiority. That table's
  *internal* comparisons (our arms against each other, same build) remain valid;
  its use as a cross-system claim does not.

What survives: the measured envelope against **Jev** is a separate, paired
comparison on its own rows and is untouched by this. What does not survive is
any claim that we are ahead of the stronger external baseline.

Two caveats stated rather than buried: n=750 with a small effect means this
test has little power — the CI still admits a +1.6 pt advantage — so the honest
summary is "no evidence of advantage", not "proven worse". And pplx is zero-shot
here while we fine-tuned 2500 rows on the same pool: matching its accuracy while
requiring task training is a *cost* disadvantage even at parity of accuracy.

## Consequence for the next development action

The banking77 intent-classification axis is now closed as a source of product
advantage: it is at parity with a zero-shot competitor, it has consumed several
envelopes already, and further tuning there buys nothing the mission needs. The
capability-gap backlog still has two live items with real headroom —
**context-use absolute levels (0.49–0.70 across boolq/squad2/contract_nli, with a
near-zero shuffled gap on contract_nli)** and **deployment/serving**. Work
continues on the former (see `LAST_` arms), not on Banking77.
