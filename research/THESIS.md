# Azurite — Doctoral Research Plan

**Status:** working plan, revised against measurement. Every number below is measured on
the target hardware unless marked otherwise. Where a number is uncertain it is labelled.

---

## 1. The claim under test

TypeSafe's Jev is valued on the proposition that a *decider* architecture — a model that
reads a long context and selects among a constrained candidate set — is the path to
reliable, near-perfect machine judgement. That proposition is worth roughly $10B.

A valuation is a claim about a mechanism. It asserts that something specific about this
architecture produces the reliability, and that the reliability will survive scale, wider
candidate sets, longer contexts, and new domains. **This thesis tests the mechanism, not
the company.** The scientific question is:

> Does constrained selection produce reliability because the model has learned a
> transferable decision function, or because the evaluation constrains the output space
> so tightly that reliability is a property of the test rather than the model?

If the second, the valuation is unsupported: the accuracy is real but it does not
generalise to the conditions under which the product must eventually operate (open
candidate sets, long context, distribution shift), and the cost curve is worse than the
headline suggests. That is a falsifiable claim about a mechanism, and it is the spine of
the thesis.

**Azurite** is the constructive half: a byte-level decider with a context window larger
than Jev's, native multimodality (bytes are bytes — text, code, and quantised audio share
one input space with no tokeniser to break on), and a characterised scaling law rather
than an assumed one.

---

## 2. What is already measured

All runs: masked byte-level model, 30% mask (MosaicBERT rate), seq 1024, batch 16,
100M-token budget, seed 0, AdamW lr 6e-4, cosine to 10%, grad clip 1.0.
Device: RTX PRO 6000 Blackwell, 94.97 GiB, sm_120.

### 2.1 The fixed-data size sweep

| rung | params | D/N | train loss (nats) | held-out bits/byte | gap |
|---|---|---|---|---|---|
| tiny | 2,411,200 | 41.5 | 0.952 | **1.3600** | 1.0× |
| small | 14,260,992 | 7.0 | 0.727 | **1.1158** | 1.1× |
| mid | 33,696,768 | 3.0 | 0.571 | **1.2223** | 1.5× |
| large | 113,465,856 | 0.88 | 0.051 | **2.8169** | **39×** |

Scored by exhaustive coverage of the 2.04M-byte held-out tail (611,055 masked positions,
repeatability ±0.002 bits/byte).

This is **not** a flat or inverted scaling law. It is a U-shaped slice through a 2-D
surface, and the mechanism is visible in the train/held-out gap: at D/N = 0.88 a
113M-parameter model has ~1.15 parameters per byte of unique text, so the cheapest way
to reduce the objective is to store the corpus. Train loss falls to 0.074 bits/byte while
held-out sits at 2.9 — a 39× gap. The tiny rung has no capacity to memorise, so it is
forced to learn structure that transfers, and it beats the large rung by 1.5 bits/byte.

**Consequence for the thesis:** any comparison of architectures at fixed token budget
compares models at different distances from the compute-optimal line. Architecture claims
made this way are not interpretable. This is a methodological result in its own right and
it applies to published comparisons, not just to this project.

### 2.2 The measurement-protocol result

The first version of this table was scored with 64 random windows (0.66% coverage) and a
fresh random mask per window. On one frozen checkpoint, six draws spanned
**2.168 → 2.520** bits/byte: a spread of 0.352, larger than every effect the ladder was
supposed to measure. Seeding the mask fixed reproducibility but not accuracy — a seed is
one arbitrary draw of a small sample.

Replacing sampling with exhaustive coverage (every held-out window once, in order, mask
derived from position) reduces the spread to **0.0017** — 200× tighter — with no seed to
choose. The residual is CUDA kernel reduction order.

**Consequence:** the "1.2242 vs 1.2988 on the same checkpoint" discrepancy that motivated
this work was not a bug in one number; it was the noise floor of the protocol. Any result
at this model scale that reports a difference below ~0.02 bits/byte without exhaustive
evaluation is reporting its own sampling error.

### 2.3 Throughput ledger (measured, this hardware)

| rung | params | tokens | wall | tok/s (end-to-end) | steady tok/s | peak VRAM |
|---|---|---|---|---|---|---|
| tiny | 2.41M | 100M | 161 s | 621,000 | — | <2 GiB |
| small | 14.26M | 100M | 507 s | 197,000 | — | 8.1 GiB |
| mid | 33.70M | 100M | 955 s | 105,000 | ~102,000 | 12 GiB |
| large | 113.47M | 100M | 2,551 s | 39,200 | 40,011 | 19.1 GiB |

Wall includes 1.36 GB checkpoint writes and the final exhaustive eval. Peak draw is 19.1
of 95 GiB, so every rung is **compute-bound**; batch size is the only throughput lever
left and the 90-minute idle ceiling, not memory, bounds segment length.

---

## 3. The binding constraint is data, not compute

The GPU can produce ~1.6B tokens in nine hours. Until this pass the corpus held **102M
unique bytes**, so every experiment at D/N = 10 above the tiny rung required multiple epochs
of the same bytes, and §2.1 shows what that does past D/N ≈ 2: the model memorises and
held-out turns over. No scheduling choice fixes that; data was the binding constraint.

**Resolved.** `scripts/build_corpus.py` builds a 1.63B-token stream from four public
sources (fineweb-edu 50%, wikipedia 22%, cosmopedia 16%, open-web-math 12%), exact-sha1
deduplicated:

| | |
|---|---|
| stream | 3.27 GB, 1,633,580,647 u16 tokens |
| documents | 339,841 unique (112 exact duplicates dropped) |
| held-out | 33,232,664 tokens (2.03%), carved **by document** |
| encoding | `id = byte + 4`, `SEP(3)` between documents |
| sha256 | `9267a2ea071a517803b3b47c01adf9464608a8178fbf12018ce5dcc9ca38365f` |

Two independent builds produced this sha256, so the corpus is a function of its inputs
rather than of the run. The held-out rule is `sha1(document) % 50 == 0`, and the boundary
is recorded in `<stream>.split.json`, which `pretrain.py` and `eval_full.py` now prefer
over the 2% rule — otherwise the tail cut would fall inside a document and train on part of
what it scores.

Consequence: at D/N = 10 the 113M rung needs 1.13B tokens, which is **0.71 epochs** of
1.60B. The iso-D/N law is now measurable. One planned source (code) is absent — it hangs
during dataset resolution — and the manifest records the omission rather than silently
redistributing it, because a corpus description that drops a planned source is a false
provenance claim.

---

## 4. Experiment programme

Each experiment states its falsifier. An experiment whose result cannot change the thesis
is not run.

### E1 — Fixed-data size sweep ✅ complete
Design: four sizes, D = 100M, everything else identical. Scored exhaustively.
Result: U-shaped, minimum at D/N ≈ 7, 39× generalisation gap at D/N = 0.88.
Falsifier of the memorisation explanation: a large rung whose train/held-out gap stays
near 1×. Not observed.
Caveat: one seed per rung. Seed variance of the *training* run is not yet characterised
(only eval variance is), so treat the tiny/small/mid ordering as provisional until E1b.

### E1b — Seed replication (cheap, do first)
Design: tiny, small, mid at seeds 0/1/2, D = 100M. 3 × (161+507+955) s ≈ 1.4 GPU-hours.
Purpose: separate the real small-vs-mid difference (0.106 bits/byte) from seed noise.
Falsifier: if seed spread exceeds 0.1, the small-beats-mid ordering is not a result.

### E2 — The epoch turnover curve (running)
Design: fix N = large, sweep D ∈ {25, 50, 100, 200, 300, 400}M, i.e. epochs
{0.25, 0.5, 1, 2, 3, 4}. Held-out scored at every 50M-token milestone.
Question: at what epoch does held-out stop improving, and does that point scale with N/D?
Prediction from §2.1: turnover at D/N ≈ 1–2.
Falsifier: held-out continuing to improve at 4 epochs, which would mean the corpus is not
the constraint and E3 becomes unnecessary.
Cost: 400M tokens at 39.2k tok/s ≈ 2.8 GPU-hours (200M already spent).

### E3 — Iso-token-per-param scaling law (blocked on corpus)
Design: D/N = 20 (Chinchilla) and D/N = 10, four sizes, unique data scaling with N.
Cost at D/N = 10: 1.64B tokens ≈ 9.1 GPU-hours. **Requires the 1.5B-byte corpus.**
Falsifier: bpb not monotone in N along an iso-D/N line.
This is the experiment that produces the thesis's headline scaling figure. It cannot run
until §3 is resolved, and running it on 102M bytes would be the same error §2.1 diagnoses.

### E4 — Accuracy vs candidate-set size (the TypeSafe claim, direct test)
Design: take a trained checkpoint, evaluate a decider task at N = 2, 4, 8, 16, 32, 64
candidates on held-out data. Plot accuracy vs log N at fixed context.
Question: does accuracy fall off a cliff as the candidate set widens?
This is the measurement that speaks directly to §1. If accuracy degrades sharply with N,
"near-perfect accuracy" is a property of narrow evaluation, and the valuation's mechanism
claim does not survive.
Cost: inference only, minutes. **Highest scientific value per GPU-second in the programme.**
Depends on: a task harness with deterministic candidate sampling (no eval noise).

### E5 — Context-length degradation
Design: accuracy vs context length 1k → 32k on a needle-style retrieval and a
classification task, at fixed N.
Question: does the advantage of long context survive past the training length?
Cost: inference, ~1 GPU-hour.
Falsifier: flat accuracy to 32k.

**Order of execution:** E4 and E5 before E3. They test the thesis's central claim against
TypeSafe directly and cost minutes; E3 is expensive and blocked. E2 runs in the background
now because it is already partly paid for and it bounds what the corpus can support.

---

## 5. Threats to validity

1. **Single seed per rung** (E1). Mitigated by E1b.
2. **One corpus, one domain mix.** Byte-level bpb is corpus-dependent; the bigram baseline
   is 3.767 on the mixed stream and 4.117 on the math-only stream. No cross-corpus claim
   is made without re-fitting the baseline.
3. **Eval length vs training length.** tiny/small trained at seq 256; evaluating them at
   1024 measures them outside their competence (they score 1.5935 vs 1.3600). All
   ladder comparisons in §2.1 are at matched seq 1024, trained the same way, so the
   comparison is internally valid but the absolute numbers are not the models' best.
4. **Tokenizer-free bpb is not accuracy.** bits/byte is a proxy. E4/E5 measure the thing
   the thesis actually claims.
5. **Contamination.** Held-out is the last 2% of the stream; provided datasets (banking77,
   clinc150, hellaswag) overlap it unless checked. `scripts/contam_check.py` and
   `scripts/heldout_overlap.py` exist for this and must gate E4.

---

## 6. Deliverables

1. **Thesis** (LaTeX, Overleaf): scaling behaviour of byte-level deciders; the
   fixed-data confound; the measurement-protocol result; the candidate-set and
   context-length degradation results; Azurite's architecture and its measured frontier.
2. **Papers**: (a) the measurement-protocol result — self-contained, submittable now;
   (b) the memorisation boundary as a function of N/D; (c) Azurite.
3. **Artifact**: the training/eval stack, the exhaustive scorer, the segmented resumable
   driver, and every number in the thesis regenerable from a command.
4. **Product**: Azurite as a deployable decider with a documented accuracy/cost frontier.

## 7. What would make me abandon the thesis claim

If E4 shows accuracy holding to N = 64 and E5 shows flat accuracy to 32k context, then
constrained selection *is* a transferable mechanism, the TypeSafe bet is well founded, and
the thesis becomes a constructive one about Azurite's efficiency rather than a critique.
That outcome is written down here in advance so the result cannot be reinterpreted after
the fact.

---

## 8. Build status (measured, not asserted)

Both documents compile in the sandbox with pdfLaTeX + BibTeX. Recipe and
toolchain: [thesis/BUILD.md](../thesis/BUILD.md).

| document | pages | errors | undefined refs | undefined cites | `??` in PDF |
|---|---:|---:|---:|---:|---:|
| `thesis/main.pdf` | 29 | 0 | 0 | 0 | 0 |
| `paper/azurite.pdf` | 4 | 0 | 0 | 0 | 0 |

Populated from measurement: corpus composition and split, the evaluation-protocol
result (0.3520 → 0.0017 bits/byte on one frozen checkpoint), the fixed-`D`
U-curve with the 38× train/held-out gap, the tiny-rung transfer matrix
(banking77 +0.1753, ag_news +0.0640 over three seeds).

Still `\emph{pending}` in the PDF, because the runs have not finished: the
matched-`D/N` ladder (4 rungs), the fitted exponent α, and the context axis
(seq 256/1024/4096). `gen_tables.py` writes a real value the moment `results/`
holds one, so no table can show a projected number.

Two honest limits on the paper: at 4 pages it is workshop length, and the
scaling exponent that would make it a conference submission does not exist yet.
The thesis says so in §5 rather than arguing around it.
