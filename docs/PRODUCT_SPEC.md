# Azurite product specification

**Status:** definition complete, implementation not started.
**Date:** 2026-10-30.
**Scope:** this document defines what Azurite must do and how it will be
measured. It is not a results report. Every number labelled *current* is a
measured baseline with a stored record; every number labelled *target* is a
commitment to be tested, not a claim that it has been achieved.

The product is a **local decision model**: it chooses among alternatives that
the caller supplies at inference time. It is not a fixed 77-way classifier, and
Banking77 is one regression test, not the product identity.

---

## 1. Product statement

Azurite answers one question:

> Given a context, a question or instruction, and a set of mutually exclusive
> alternatives, which alternative is supported by the evidence — or is the
> evidence insufficient to decide?

The caller supplies the alternatives. The model must not substitute a built-in
taxonomy, must not silently choose a label outside the supplied set, and must be
able to return an explicit abstention.

The first product surface is a Python API and a local CLI. A local HTTP service
is a later packaging step, not a prerequisite for the model contract.

### 1.1 Primary users

- A developer routing support, documents, or form submissions without sending
  data to a hosted model.
- A researcher running reproducible decision experiments on a single GPU.
- An auditor who needs per-decision evidence, a checkpoint hash, a calibration
  identifier, and a stated abstention reason.

### 1.2 Product principles

1. **Caller-owned alternatives.** The label set is an input, not a checkpoint
   property. A new label set must not require a new classification head.
2. **Evidence before answer.** A grounded decision names the spans it used. If
   it cannot name evidence, it abstains rather than guessing.
3. **Abstention is a first-class outcome.** "Insufficient evidence" is a valid
   decision, not an error and not a low-confidence label.
4. **Local and inspectable.** Weights, tokenizer, calibration, and thresholds
   are content-addressed and run without a network call.
5. **No hidden capability claim.** A family that has not been measured is
   reported as missing, even if the interface accepts it.

---

## 2. The decision contract

### 2.1 Request

```python
DecisionRequest(
    context: str | bytes | Document | Sequence[Document],
    question: str,
    alternatives: Sequence[Alternative],   # 1..K, unique ids
    decision_type: "auto" | "boolean" | "classify" | "pairwise"
                 | "ordinal" | "grounded",
    evidence_required: bool = False,
    abstain_allowed: bool = True,
    policy: str | None = None,             # reserved; not a v1 gate
    calibration: str | None = None,        # calibration artifact id
)
```

`Alternative` has a stable `id`, a display `label`, and an optional
`description`. The `id` is opaque and is what the result returns; the label is
what the model reads.

`Document` carries text plus an optional image or page image, a media type, and
a stable document id. A list of documents is the document-grounded input; the
context is not flattened into one string before the model sees it.

### 2.2 Result

```python
DecisionResult(
    choice_id: str | None,          # None only when abstained
    choice_label: str | None,
    abstained: bool,
    abstain_reason: "insufficient_evidence" | "ambiguous"
                 | "out_of_scope" | "invalid_input" | None,
    scores: Mapping[str, float],   # one entry per alternative
    calibrated: bool,
    evidence: Sequence[EvidenceSpan],
    ranking: Sequence[str],        # ordinal / full ranking
    model_id: str,
    checkpoint_sha256: str,
    calibration_id: str | None,
    latency_ms: float,
    diagnostics: Mapping[str, object],
)
```

`EvidenceSpan` carries a document id, UTF-8 byte start/end, the quoted text, and
an optional page/box for a visual document. Byte offsets are used because the
current encoder is byte-level; changing the tokenizer does not change the
contract.

### 2.3 Semantics by family

| Family | Request shape | Result |
|---|---|---|
| Boolean | K=2, ids `yes`/`no`, a question | `choice_id`, score for each; abstention allowed |
| Classification / routing | K>=2 arbitrary labels, one context | `choice_id`, full score map, ranking |
| Pairwise | K=2 alternatives A/B, optional `tie` | winner, tie, or abstain; `scores` are pairwise preferences |
| Ordinal | K>=2 ordered alternatives, ordered ids | full ranking, cut-point scores, predicted grade |
| Document-grounded | one or more documents, question, evidence required | choice plus spans; no span means abstain |
| Abstention | any family, `abstain_allowed=True` | explicit `abstained`, reason, and scores retained |

Boolean is not a special head. It is the K=2 case of classification with
natural-language `yes`/`no` alternatives. Pairwise and ordinal are distinct
because their training signal and metrics are distinct: a pairwise model may
care about a tie, and an ordinal model must respect order.

### 2.4 Contract invariants

These are testable without a trained model and must be enforced by the runtime:

1. `alternatives` must be non-empty; duplicate ids are rejected.
2. `choice_id` is always one of the supplied ids or `None` with `abstained=True`.
3. The fixed 77-way Banking77 path remains available and byte-identical; the
   general path never falls back to it.
4. Candidate order is not a feature: a seeded permutation must not change the
   ranking beyond a stated numerical tolerance.
5. `decision_type="auto"` infers boolean from K=2, ordinal only when the
   request declares ordered alternatives, and otherwise classification.
6. `evidence_required=True` on a grounded request without a returned span must
   abstain; it must not return an unsupported choice.
7. Scores are not probabilities unless `calibrated=True`; `confidence` is not
   used as a synonym for a softmax value.
8. An invalid request raises `InvalidRequest`; if a transport requires a result
   envelope, it sets `abstain_reason="invalid_input"` and returns no choice. It
   never returns a label.
9. An image-bearing document is rejected as `unsupported_input` (or explicitly
   degraded to its OCR transcript with a flag) until the visual path passes its
   gate. The text path must never silently pretend to see an image.

### 2.5 Deferred families, not dropped

Tool/action selection and policy-conditioned decisions are mission objectives
(O3) but are not v1 gates. The request carries `policy` as a reserved field, and
the result has a place for policy diagnostics, but no policy evaluator is
implemented and no capability claim is made. A tool/action registry and a
policy-compliance benchmark are added only after the five core families and the
registry exist; until then, advertising either would repeat the current gap in
a new vocabulary.

---

## 3. Capability targets

The machine-readable commitment is
[`research/CAPABILITY_TARGETS.json`](../research/CAPABILITY_TARGETS.json). The
table below is the readable summary. **Functional** means the feature works and
beats a trivial baseline; **competitive** means it is useful as a product;
**stretch** is a research goal that does not block release.

Targets are frozen before training. An amendment is allowed only in a dated
pre-registration written before the affected run; moving a target after seeing
a result is prohibited and must be recorded as a protocol violation.

### 3.1 Current measured floors

These are the 113.47M suite numbers on 2,000 test rows (1,000 for Financial
PhraseBank), from `results/suite_large_113m/COMPARISON.json`. They are the
regression floor, not the product target.

| Task | K | Current | Cited Jev | Gap |
|---|---:|---:|---:|---:|
| AG News | 4 | 85.75% | 91.0% | -5.25 |
| DAIR Emotion | 6 | 81.70% | 48.0% | +33.70 |
| Banking77 | 77 | 88.45% | 78.8% | +9.65 |
| CLINC-OOS | 151 | 88.50% | 89.5% | -1.00 |
| MASSIVE Intent | 60 | 82.00% | 78.3% | +3.70 |
| MASSIVE Scenario | 18 | 83.90% | — | — |
| XNLI | 3 | 33.90% | 86.0% | -52.10 |
| Language ID | 20 | 95.80% | 99.6% | -3.80 |
| Financial PhraseBank | 3 | 90.10% | 73.0% | +17.10 |

The full 3,080-row Banking77 product path is 89.06% (87.91–90.11), not 88.45%;
the 2,000-row suite draw is a different sample. The 3,080-row number is the
regression gate.

Other measured floors that matter to the targets:

- 3-level Financial PhraseBank ordinal: **QWK 0.759**, adjacent accuracy
  **97.4%**, MAE **0.125** on the 1,000-row test draw.
- Banking77 selective accuracy at 50% coverage: **98.8%** (suite), but ECE
  **0.0618**. The current confidence is not a calibrated probability.
- 5,120-byte Banking77 arm: **87.11%**, below the 1,024-byte arm. This is a
  negative result, not a long-context baseline.
- Fixed-checkpoint dev curve (14.4M, pretrained at 256 bytes, evaluated to 4,096):
  **no context-additive task**. Classification is flat; BoolQ is at majority;
  SQuAD 2.0 answerability gains +1.47 pp but stays at majority; ContractNLI falls
  **69.79% → 37.03%**. One seed, dev only, and ContractNLI is partial-context at
  every length. This is a failure signal, not a product gate result.

### 3.2 Target table

| Family | Evaluation | Metric | Functional | Competitive | Stretch |
|---|---|---|---|---|---|
| Regression | Banking77 full 3,080 | accuracy | >= 88.56% (89.06 − 0.50) | >= 89.06% | >= 92.40% (Jev) |
| Boolean | BoolEval-1, 2,000 balanced | balanced accuracy | >= 75% | >= 85% | >= 92% |
| Boolean, adversarial | 500 negation / scope cases | balanced accuracy | >= 65% | >= 75% | >= 85% |
| Classification | nine-cell suite | unweighted mean accuracy | >= 83.1% (+2.0) | >= 86.0% | >= 88.0% |
| Classification, worst cell | XNLI / AG News / CLINC | accuracy | no cell below current −1.0 pp | XNLI >= 50%, AG >= 88%, CLINC >= 90% | XNLI >= 60%, AG >= 91%, CLINC >= 92% |
| Pairwise | PairEval-1, 8,000 pairs | pair accuracy | >= 70% | >= 85% | >= 90% |
| Pairwise, ties | 1,000 near-tie pairs | tie ECE | <= 0.15 | <= 0.10 | <= 0.05 |
| Ordinal, 3-level | Financial PhraseBank | QWK | >= 0.70 | >= 0.80 | >= 0.85 |
| Ordinal, 5-level | OrdEval-1, 5,000 items | QWK | >= 0.55 | >= 0.70 | >= 0.80 |
| Document-grounded | SQuAD 2.0-style, 2,000 | answer F1 | >= 60 | >= 75 | >= 85 |
| Document-grounded | same | answerability accuracy | >= 70% | >= 80% | >= 90% |
| Document-grounded | same | citation span IoU | >= 0.50 | >= 0.70 | >= 0.85 |
| Abstention | mixed suite, 20% unanswerable | selective accuracy at stated coverage | >= 90% at 70% coverage | >= 95% at 80% coverage | >= 97% at 80% coverage |
| Abstention | same | unanswerable abstain recall | >= 70% | >= 85% | >= 92% |
| Long context | 1k / 4k / 8k / 16k / 32k bytes | accuracy vs 1k | 8k within 3 pp | 16k within 2 pp | 32k within 2 pp |
| Long context, positions | early / middle / late | worst position gap | <= 10 pp | <= 5 pp | <= 3 pp |
| Visual | VisualEvidence-1, 2,000 | visual-only accuracy | >= 70% | >= 80% | >= 90% |
| Visual, control | same | gain over OCR-only | >= +10 pp | >= +15 pp | >= +20 pp |
| Visual, evidence | same | citation IoU | >= 0.50 | >= 0.65 | >= 0.80 |
| Reliability | all measured families | ECE | <= 0.08 | <= 0.05 | <= 0.03 |
| Reliability | all measured families | Brier skill vs class prior | > 0 | >= 0.10 | >= 0.20 |
| Reliability | all measured families | AURC vs confidence-only | better by 10% relative | better by 20% relative | better by 30% relative |
| Deployment, GPU | reference RTX PRO 6000 | p50 at 1k / 16k | <= 2 s / <= 8 s | <= 250 ms / <= 2 s | <= 100 ms / <= 1 s |
| Deployment, CPU | 20 vCPU | p50 at 1k | <= 5 s | <= 2 s | <= 1 s |
| Artifact | int8 weights | size / accuracy drop | <= 700 MB / <= 2 pp | <= 400 MB / <= 1 pp | <= 250 MB / <= 0.5 pp |

Notes:

- The classification "competitive" row is not a demand to beat Jev on every
  task; it sets three named gaps that must close and an aggregate floor. The
  full per-cell target is in the JSON.
- The document and visual targets are provisional until the registry freezes
  the exact datasets and licences. The *thresholds* are committed now; the
  dataset substitution cannot be used to lower them.
- An absolute target without a paired baseline is not enough. Every family must
  also beat a lexical, majority, or OCR-only control by a pre-registered margin
  with a 95% interval that excludes zero.
- One seed is not a capability estimate. Any competitive claim needs either
  three seeds or a paired design with a stated noise floor.

### 3.3 What would count as failure

A target is reported as missed, not reframed:

- A family below its functional gate is **not supported**, even if the API
  accepts it.
- A long-context arm that accepts 16k bytes but loses more than 2 points is a
  **negative result**; the context claim is withdrawn.
- A visual model that matches OCR-only is an **OCR pipeline**, not multimodal
  evidence processing.
- An abstention threshold fitted on the locked suite is a **protocol
  violation**, not a result.
- A classification aggregate that improves because one easy cell improved while
  XNLI stays at chance is **not** general routing capability.

---

## 4. Long-context requirement, investigated explicitly

"Long context" means **useful context**, not a larger positional buffer. The
contract is:

> At 16,384 bytes of packed evidence, accuracy is within 2 points of the
> 1,024-byte arm on the same rows, with no evidence-position gap above 5 points
> and no collapse under three levels of distracting text.

### 4.1 Why this is a real gap

- The current checkpoint has a native **1,024-byte** window.
- The only longer arm, 5,120 bytes, scores **87.11%**, below the native arm's
  89.06% on the same 3,080 rows. Accepting more bytes did not preserve quality.
- The Jev comparator's reconstructed median prompt is **10,683 bytes**, so the
  current model is shorter than the comparator, not longer.
- 16,384 bytes is about **1.5x** the Jev median; 32,768 is about **3x**. Those
  are the two product-relevant lengths.
- The 2026-10-30 fixed-checkpoint curve evaluates 256 → 4,096 bytes on five dev
  tasks and finds **no context-additive task**. ContractNLI degrades monotonically
  (69.79% → 37.03%), and the document/abstention heads sit at majority. The
  lexical control learns ContractNLI (64.71–66.71%), so the neural fall is not a
  task artefact. This does not test 16k, but it removes the assumption that a
  longer evaluation window is harmless.

### 4.2 Two implementation paths, both hypotheses

| Path | What it buys | Cost / risk |
|---|---|---|
| Full-window extension (RoPE scaling + length adaptation at 4k, then 16k) | One attention pass over all evidence; simple evidence offsets | Quadratic attention; 16k full-information arm projected at **12.6–15.9 GPU-hours**, 32k at **39.8–56.2**; position generalisation may fail |
| Chunked hierarchical encoder + retrieval/reranking | Linear cost; 32k+ effective context; fits the 12-hour sandbox | Cross-chunk reasoning can be lost; retrieval errors become decision errors; must be measured against the full-window control |

The order is deliberate: **4k first, 16k only if 4k is non-inferior, 32k only
after 16k**. A negative 4k result stops the 16k spend. The fixed-checkpoint 4k
arm has now failed (no context-additive task; ContractNLI degraded), so the 16k
spend is stopped until a length-adapted 4k arm passes the gate.

### 4.3 Long-context test design

Every arm is evaluated on the same rows with:

1. **Clean evidence** at the start, middle, and end of the context.
2. **Distractor levels**: 0, 2, and 8 irrelevant documents.
3. **Distributed evidence**: the answer requires two spans that are far apart.
4. **No-evidence control**: the answer is absent; the model must abstain.
5. **Latency and memory** at each length, not just accuracy.

The primary metric is the drop from the 1,024-byte arm. A model that is 2 points
worse at 16k but 20 points better at 32k is not a clean long-context win unless
the 16k non-inferiority gate is also met.

### 4.4 What is not claimed

No long-context claim will be made from a positional-embedding change, a
truncation policy, or a prompt that fits in 16k but was not trained at that
length. The 4k fixed-checkpoint curve is now available and is **negative**, so
the statement is no longer "the context curve is unavailable"; it is "the 4k
curve is measured and the 16k non-inferiority gate is not met." The 16k claim
stays withdrawn until that gate is measured.

### 4.5 Document-grounded decisions, explicitly

A document-grounded request supplies one or more documents and a question; the
result must name the evidence spans that decide it. The first benchmark is
SQuAD 2.0-style: extractive answers with a known span and a substantial
unanswerable subset. The context can reach 16k bytes after packing, so this
family is also the first consumer of the long-context gate.

The failure semantics are strict: if `evidence_required=True` and no span is
returned, the result is `abstained` with `insufficient_evidence`. An answer with
no supporting span is an unsupported answer, even if it matches the gold text.
Citation IoU is computed against the gold span, and the unsupported-answer rate
is reported separately from answer F1.

Controls are a no-retrieval text-only path, BM25 best-sentence, and a majority
answer. The no-retrieval control separates "the model can read the document"
from "retrieval found the sentence". If BM25 alone matches the end-to-end
system, the document path is a retrieval demo, not a grounded decision model.
The public SQuAD licence and any image-derived document source are recorded in
the registry before admission; the locally rendered set is the licence-clean
fallback.

---

## 5. Visually grounded decisions, investigated explicitly

"Visually grounded" means the decision changes when the image changes and cannot
be reproduced from OCR text alone. A document screenshot with OCR plus a text
model is a **text pipeline**, not visual evidence.

### 5.1 Minimum visual task

The first suite is deliberately small and controlled:

- **Rendered documents** with checkboxes, tables, stamps, signatures, and
  layout-dependent fields. The answer depends on a mark or a spatial relation,
  not only on a text string.
- **Charts** where the answer depends on a bar/line position or a trend, not a
  caption.
- **Scanned forms** with a field value that OCR can read but a validity mark
  that OCR cannot.

Each item has a gold answer, a gold evidence region with a page and box, and an
OCR-only transcript. The OCR-only transcript is a required control, not an
optional ablation.

### 5.2 Model path

The visual path is a **frozen, open-licence image encoder** plus a trained
projection/adapter into the decision model. The encoder is not chosen here:
selection requires a recorded licence, provenance hash, and a check that its
pretraining data does not contain the evaluation images. Candidates are
admissible only after that check.

The adapter must be trained on the training split with the text path frozen or
jointly adapted under a pre-registration. A visual model that is merely an OCR
call followed by the text model is rejected by the control.

### 5.3 Visual targets and failure

- Functional: visual-only accuracy >= 70% and >= +10 points over OCR-only.
- Competitive: >= 80% and >= +15 points, with citation IoU >= 0.65.
- If OCR-only is within 5 points, the visual claim is withdrawn regardless of
  the absolute score.
- If the visual model is worse than OCR-only on the visual-only subset, the
  adapter is not shipped.

### 5.4 Data and licence gate

Public candidates (DocVQA, ChartQA, VisualMRC, SlideVQA) are **not admitted**
until the registry records the exact licence, split, and image provenance. The
first suite can be locally rendered and therefore licence-clean; it is a
development instrument, not a public benchmark claim. A public visual benchmark
number is reported only after the licence gate and contamination check pass.

---

## 6. Architecture hypotheses and admission gates

The encoder, tokenizer, pooling, retrieval, and training procedure are
replaceable. The contract and the targets are the stable parts.

### 6.1 Decision head

The reference design is a **candidate-conditioned scorer**, not a fixed
classifier:

- Encode the context once.
- Encode each alternative (or a shortlist) with a shared candidate encoder.
- Score with a learned compatibility function; support listwise softmax,
  pairwise margin, and ordinal cut-points as heads over the same representation.
- Train with candidate dropout, label paraphrases, and hard negatives so that
  unseen label sets are not out-of-distribution by construction.
- Fit calibration and the abstention threshold on a dev split, freeze them, and
  evaluate once on the locked split.

The existing 77-way head may be reused as an initializer, but it is not the
product head. A fixed label embedding table that cannot score an unseen label is
a baseline, not the contract.

### 6.2 Encoder choice

Three admissible starting points, to be compared on the dev registry, not
selected by intuition:

1. The current 113.47M byte encoder (no pretrained weights; known checkpoint,
   licence-clean, already measured).
2. A small pretrained text encoder with a documented licence and a
   contamination audit (for example a ModernBERT/MiniLM-class model). This is a
   hypothesis, not a dependency; adding it requires provenance and a size
   budget.
3. A hybrid: pretrained text encoder plus a byte fallback for unknown scripts
   and OCR noise.

The byte encoder is the **incumbent, not the default**. The full architecture
record, candidate licences, contamination risks, and admission gates are in
[`research/ARCHITECTURE.md`](../research/ARCHITECTURE.md). Pretrained components
are permitted; training from scratch is not a requirement merely because the
previous model was trained that way. A component stays only if it wins on the
dev registry at an acceptable licence, contamination, size, and latency cost.

The encoder decision is made on the dev suite with the same decision head and
the same training budget. A larger encoder is not admitted merely because it is
larger; the 113M vs 14.4M suite result already shows that scale alone did not
buy decision accuracy.

### 6.3 Vision encoder

Frozen, open-licence, hash-recorded. The adapter is small (projection plus
cross-attention or pooled patch tokens). The vision encoder's licence and
pretraining data are part of the model card, not a footnote.

### 6.4 Retrieval

Retrieval is part of the decision system when evidence exceeds the window. It
must be evaluated as a component: retrieval recall, evidence coverage, and the
end-to-end decision. A better retriever that does not improve decisions is not a
product improvement.

---

## 7. Evaluation protocol

### 7.1 Dev and locked suites

- A **dev suite** is used for architecture, hyper-parameter, and protocol
  choices. It is development evidence, not final confirmation.
- **Calibration is a separate partition**, carved before any fitting and disjoint
  from train, dev, and locked. A threshold fitted on dev test or locked is a
  protocol violation, not a result. If the suite has no calibration partition,
  nothing may be calibrated on it.
- The v1 dev suite has train/test rows only. No calibration was fitted for the
  fixed-checkpoint context experiment, so its argmax results do not consume a
  calibration split. Any future abstention or calibration run needs a new suite
  version.
- A **locked suite** is hashed before method selection and opened once, after
  the method is frozen.
- No target is tuned on the locked suite. No locked item is inspected during
  development.
- The locked suite must contain at least two datasets per family, and no single
  dataset may carry more than 25% of a family's weight.

The current selection and provenance record is
[`research/BENCHMARK_REGISTRY.md`](../research/BENCHMARK_REGISTRY.md), with the
machine-readable companion at
[`research/benchmark_registry.json`](../research/benchmark_registry.json). It
names six families and 44 datasets (26 core). The row-level suites are now
materialised:

- [`research/suites/dev_suite.json`](../research/suites/dev_suite.json) — five
  tasks, 20% each, built only from training splits, zero train/test document
  overlap, 2,499–2,500 train / 748–750 test rows.
- [`research/suites/locked_suite.json`](../research/suites/locked_suite.json) —
  twelve tasks, 8.33% each, 493–500 rows each, sealed 2026-10-30. It is broader
  but not complete: tool/action, ordinal, and visual families remain missing;
  five tasks are research-only because their licences are unresolved.

No dataset exceeds 25% of either suite. The lock is procedural, not an
untouched-data claim: nine locked tasks were inspected by earlier work and are
marked `previously_inspected`.

**Overlap audits are part of the gate.** Dev train/test context overlap is **0**
for all five tasks, and dev/locked overlap is **0** after excluding 45 BoolQ
passages shared by the official splits. The product retrieval bank was audited
separately: 25 of 9,993 support messages matched evaluation texts, affecting 255
rows and 283 instances; the shipped bank removes them and has 9,968 rows. The
Jev-frozen prompts deliberately retain the comparator's 24 examples, so they are
a comparator artifact, not a clean product retrieval bank.

Comparator points are now separated into `published_reference`,
`local_reconstruction`, and `reproduced_live`. There are thirteen published
references, one local reconstruction (Banking77 0.924), and **zero live
reproduced Jev runs**. A published number is not a paired comparison; the
reconstruction is not an API call.

### 7.2 Controls

Every family has at least one of:

- a majority/prior baseline;
- a lexical baseline (TF-IDF/BM25 or character n-gram);
- a text-only control for a visual task;
- an OCR-only control for a visual task;
- a no-retrieval control for document-grounded decisions;
- a forced-choice control for abstention (the model without a reject option).

### 7.3 Statistics

- Report 95% Wilson intervals for accuracy and paired bootstrap intervals for
  differences.
- Use exact McNemar for paired binary decisions.
- Report the run-to-run noise floor; the measured suite repeat floor is up to
  **1.25 points** and 274/2,000 flipped predictions. A delta below that is not
  a capability claim.
- Multiplicity: pre-register the primary metric per family; label the rest
  exploratory.
- One seed is labelled one seed. Competitive claims need three seeds or a
  paired design.

### 7.4 Provenance

Every reported number must have:

- a stored per-example record with gold and prediction;
- a checkpoint SHA-256 and a calibration id;
- a protocol/pre-registration hash;
- a generator script that recomputes the table from the record;
- a manifest that verifies the files.

No number is transcribed by hand into the thesis or paper before it exists in a
record.

### 7.5 Capability milestones and measurement axes

Long-context and visual-evidence work is gated by explicit milestones in
[`research/CAPABILITY_MILESTONES.md`](../research/CAPABILITY_MILESTONES.md):
entry condition, pass rule, stop rule, compute cap, and status for LC0–LC4 and
V0–V4. A longer accepted window is not a pass; visual accuracy is not a pass
without an OCR-only control.

Every substantial result reports quality, calibration (ECE, Brier, reliability),
abstention risk–coverage and unanswerable recall, robustness (positions,
distractors, distributed evidence, negation, paraphrase, blank/shuffled
images), latency p50/p95, throughput at batch 1/8, and peak RSS/VRAM plus
artifact size. “Not measured” is a reported limitation, not a blank cell.

Near-perfect accuracy is a distribution-specific ambition, never a guarantee. A
97% target is meaningful only with its named distribution, coverage, confidence
interval, and failure slices.

---

## 8. Deployment contract

- **Local-first.** The default path runs on CPU; GPU is an accelerator, not a
  requirement.
- **Weights are optional and content-addressed.** The wheel does not bundle the
  1.36 GB checkpoint. A missing checkpoint is an explicit error, not a silent
  download.
- **No network at inference.** No telemetry, no hosted fallback, no API key.
- **Deterministic.** Fixed seed, fixed candidate order, fixed calibration
  artifact; the same request returns the same result on the same device.
- **Reported cost.** Every release note states p50/p95 latency, peak memory, and
  artifact size for the reference machine.
- **Quantisation is a release option, not a claim.** An int8 artifact is shipped
  only if its accuracy drop is measured and within the target.

---

## 9. Non-goals

Azurite is **not**:

- a general-purpose chatbot or an instruction-following LLM;
- a replacement for a human in credit, fraud, eligibility, safety, or legal
  decisions;
- a fixed taxonomy service; Banking77 is one regression test;
- a multimodal model until a visual-only subset beats the OCR control;
- a long-context model until the 16k non-inferiority gate is measured;
- a calibrated probability service until calibration is fitted on dev and
  verified on the locked suite.

---

## 10. Release gates

| Gate | Requirement | Status |
|---|---|---|
| P0 | Product contract and targets frozen; ledger updated | **done 2026-10-30** |
| P1 | `decide` API implemented with the invariants in §2.4; arbitrary K tested | not started |
| P2 | Benchmark registry frozen; dev/locked split hashed; licences recorded | **partial 2026-10-30**: dev/locked suites frozen and hashed; five research-only tasks and several dataset licences unresolved; three families absent |
| P3 | Current model measured across the dev registry with all controls | **partial 2026-10-30**: 5 dev tasks, majority and lexical controls, one seed; locked not run |
| P4 | A candidate-conditioned head trained and compared on dev | not started |
| P5 | 4k long-context arm measured; 16k only if 4k passes | **partial 2026-10-30**: fixed-checkpoint 4k arm measured and negative; length-adapted 4k not run; 16k not authorised |
| P6 | Visual adapter measured against OCR-only on the visual-only subset | not started |
| P7 | Locked-suite evaluation, calibration freeze, model card, paper/thesis update | not started |

A release that skips P2/P3 is a demo, not a capability result. A release that
reports a family without its control is not admissible.

---

## 11. Open questions, with owners and decision points

These are hypotheses to test, not blockers:

1. **Encoder.** Byte vs pretrained tokenizer vs hybrid. Decision on dev
   classification + document F1 + latency. The current byte encoder has no
   licence or contamination debt; a pretrained encoder must earn its place.
2. **Candidate scoring.** Dual encoder vs cross encoder vs hybrid rerank.
   Decision on K=2/8/32/151 accuracy, latency, and permutation invariance.
3. **Abstention.** A separate sufficiency head vs calibrated max score vs
   conformal threshold. Decision on selective risk at 80% coverage and
   unanswerable recall.
4. **Long context.** Full-window 16k vs chunked 32k. Decision on the 4k/16k
   non-inferiority gates and the GPU-hour budget.
5. **Vision encoder.** Which open-licence encoder and adapter. Decision after
   licence/provenance review and OCR-only control.

## 12. Definition of done for this definition step

- The decision contract is explicit enough to implement without guessing.
- Every family has a metric, a functional gate, a competitive gate, and a
  failure condition.
- Long context and visual evidence have separate feasibility sections with
  costs, controls, and stop rules.
- Targets are frozen in a machine-readable file with a hash and an amendment
  policy.
- No training has been run against the targets; no target is reported as a
  result.
