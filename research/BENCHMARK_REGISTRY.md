# Azurite benchmark registry

**Version:** 0.3 (selection registry; row-level suites frozen separately).
**Date:** 2026-10-30.
**Machine-readable companion:** [`research/benchmark_registry.json`](benchmark_registry.json).
**Owner:** mission ledger, [`research/MISSION_LEDGER.md`](MISSION_LEDGER.md), objective O7 / experiment N2.

**Status in one sentence:** the registry names six decision families, their
candidate datasets, comparator evidence, licences, scoring, and blockers; the
compact dev suite and a twelve-task sealed locked suite are now materialised and
hash-verified, but the locked suite is **not complete across six families**, no
new family capability is implemented, and several licences remain unresolved.
O7 stays **partial**. Nothing in this file is a capability result.

---

## 1. What this registry is, and what it is not

This is a **selection and provenance registry**, not a leaderboard.

It exists to stop three failures that the audit found:

1. **Family collapse.** The old nine-task suite was classification-heavy. A
   model could look general while only ever choosing among fixed labels.
2. **Comparator mush.** Jev numbers from a vendor demo, a 300-row pilot, a
   37-dataset study, and a local prompt reconstruction were being placed in the
   same column. They are different protocols, splits, and versions.
3. **Licence debt discovered after the result.** Several datasets already on
   disk have no stated licence; others are research-only. That is recorded here
   before any new training run, not after.

What this file does **not** do:

- It does not claim any dataset is uncontaminated. Public benchmarks are public.
- It does not make this document the executable suite. The row-level dev and
  locked manifests are [`research/suites/`](suites/README.md); this registry is
  the selection and provenance layer above them.
- It does not add document, tool, policy, ordinal, or visual capability to the
  model. Those are missing functionality, not registry entries.
- It does not turn a published Jev number into a paired comparison. Only a
  same-row, same-version run is paired.

### 1.1 Evidence labels used below

| Label | Meaning |
|---|---|
| **official** | Read from the dataset/model owner's page, repository, or paper. |
| **local** | Reproduced from a file and hash in this checkout. |
| **third-party** | Independent or external run; not the vendor's own claim. |
| **reported** | A count or claim from a source that was not independently re-counted here. |
| **unresolved** | No licence or terms could be established from the sources reviewed. Treat as not shippable. |
| **conflict** | Two sources disagree, or the same benchmark name hides different splits/label spaces. |

### 1.2 Non-negotiable rules

1. **No dataset may carry more than 25% of its family's dev weight.** The
   machine-readable file enforces this with explicit weights.
2. **Dev and locked are separate.** Dev is for architecture, calibration, and
   thresholds. Locked is hashed before method selection and opened once.
3. **No prompt or threshold tuning on locked.** If it was inspected, it is not
   locked.
4. **Every comparator point names its source, version, split, and n.** No
   averaging across sources. No silent substitution of `jev-latest`.
5. **Every dataset names a licence or records `unresolved`.** Unresolved means
   research-only until resolved; it cannot enter a shipped product.
6. **Every family reports its own result.** The composite score is descriptive,
   never a substitute for the family table.
7. **Failures stay in the record.** A family with no licensable dataset is
   recorded as blocked, not replaced by a toy task.

---

## 2. Comparator registry: Jev

Jev is the primary external comparator because the mission names it. It is a
**hosted, closed-weight, text-only decision API**, not a local checkpoint. That
changes what reproducibility means: a run can be repeated at the API level if
the version, prompt, and request bytes are stored, but the weights and training
data cannot be inspected.

### 2.1 Official model facts

Sources: [models](https://docs.typesafe.ai/models),
[System One API](https://docs.typesafe.ai/api/system-one),
[primitives](https://docs.typesafe.ai/primitives),
[Choice](https://docs.typesafe.ai/primitives/choice),
[confidence](https://docs.typesafe.ai/confidence),
[Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13),
[launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

| Field | `jev-1.13.0` | `jev-1.14.0` | Evidence |
|---|---|---|---|
| Release | earlier release | 2026-10-14 | official |
| Price | $0.042 / Mtok input; output free | same | official |
| Context | 64k per request / 32k per question | same | official |
| Long-context mode | 262k both limits with `TypeSafe-Danger-Zone-Long-Context: true`; slower and less accurate | same | official |
| Input | text only; no audio/video | text only | official |
| Primitives | Noul (0–1), Choice (up to 255 options), Score (ordered levels), Box (image, danger-zone header) | same | official |
| Confidence | Choice: `(p_max − 1/n) / (1 − 1/n)`; Score: distance from most-likely level; Noul has none | same | official |
| Aliases | `jev-latest` points to 1.14.0; `jev-preview` also 1.14.0 at time of writing | alias moves with releases | official |
| Fine-tuning | none per account; same weights for all customers | none per account | official |
| Customer data | vendor states requests/responses are not used for training | same | official |

**Version pin rule.** `jev-1.13.0` is the version used by the public studies that
this registry cites. `jev-1.14.0` is the current product. They are **not
interchangeable**. Any comparison must state which one answered; the API response
returns the resolved model ID, and that ID must be logged. Never use
`jev-latest` in a pre-registered run.

**Scoring rules used here.** Choice: argmax option; report accuracy/macro-F1,
15-bin ECE, Brier, selective accuracy at 80/50% coverage, and AURC. Noul:
threshold selected on dev (start at 0.5; 0.7/0.3 is a conservative gate); report
accuracy/F1, AUPRC, ECE, and Brier; do not call 0.5 calibrated. Score:
probability-weighted expected level; report MAE, QWK, and Spearman, with any
threshold pre-registered; do not interpolate exact magnitudes between levels.

**No independent 1.14 result was found in the sources reviewed here.** Treat
1.14 as unmeasured by third parties in this registry until a pinned run exists.

### 2.2 What Jev's confidence means

Confidence is a **concentration statistic**, not a correctness probability. A
peaked distribution can still be wrong. The vendor's own guidance is to
calibrate thresholds on your data; for Noul, 0.5 is the neutral point and a
starting gate is 0.7 yes / 0.3 no. For Choice, the formula above gives 0 at an
even split and 1 at certainty, regardless of whether the selected option is
correct.

This matters for the abstention family: a registry entry that reports accuracy
without ECE, Brier, risk–coverage, and selective accuracy at fixed coverage is
incomplete.

### 2.3 Known Jev failure modes that become controls

The 1.13 jaggedness page is unusually candid. It lists failure modes that the
registry must test, not assume away:

| Failure mode | Consequence for this registry |
|---|---|
| Literal reading; negation and implied conditions | Boolean and policy tasks need explicit negation/exception subsets. |
| Math, counting, dates, numeric representations | Do not use Jev as a calculator; keep arithmetic in code. Numeric-answer visual tasks need a separate check. |
| Multi-hop indirection | Document-grounded and policy tasks need hop-count strata. |
| Large state with irrelevant detail / context rot | Measure accuracy vs context length and distractor count. |
| Adversarial content in state | Prompt-injection and contradictory-evidence subsets. |
| Contradictory instructions/criteria | Freeze instructions and audit them before the run. |
| Choice option order | Permute option order and require consistency. |
| No generation | Extraction must be a Choice over candidates, not free text. |

### 2.4 Comparator evidence register

| # | Source | Scope | Jev version | What is public | Reproducibility tier | Use in this registry |
|---|---|---|---|---|---|---|
| C1 | [TypeSafe launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and vendor workflow evals | vendor demos, four workflows | 1.13 era | claims and selected scores; no raw rows | **D** (vendor, undisclosed protocol) | context only; never a benchmark row |
| C2 | [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) | 300 held-out BTZSC examples, 100 per condition | `jev-1.13.0` | Apache-2.0 code, config, aggregate report; raw predictions withheld | **B/C** | classification comparator with n and split caveats |
| C3 | [Bonn study](https://arxiv.org/abs/2609.37647), [code](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking), [responses](https://doi.org/10.5281/zenodo.23039006) | 37 datasets, 346,009 requests, under $10 | `jev-1.13.0` | MIT code; raw responses under Jev Responses License 1.0 | **A/B** | strongest public comparator; per-task points where split matches |
| C4 | [Vals independent evaluation](https://www.vals.ai/blogs/independent-evaluation-of-jev) | 400 SEC-filing claim items + 396 LegalBench items | `jev-1.13.0` | independent report with per-item labels, probabilities, time, cost | **B** | document-grounded and legal/policy comparator |
| C5 | [AY Automate](https://www.ayautomate.com/blog/jev-vs-llm-benchmark) | 791 decisions across tasks | `jev-1.13.0` | aggregate post; harness not independently verified | **C** | cited points only; conflicts recorded |
| C6 | [Ravvich](https://github.com/ravvich/jev-benchmark) | 500 training + 500 test examples | `jev-1.13.0` | code public; third-party | **C** | secondary; not used for a claim |
| C7 | Local Banking77 reconstruction, [`results/jev_matched/`](../results/jev_matched/) | 3,080 Banking77 rows | reconstructed prompt, not a live API run | per-example correctness vector; no Jev probabilities | **B/C** | regression comparator only; do not call it a Jev 1.14 run |
| C8 | [TypeSafe WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals) | four workflows: invoice 150, customer service 204, agent trace 111, security 240; 705 cases | `jev-1.13.0` | Apache-2.0 code; datasets separate; consensus references, not independent ground truth | **D/B** (labels/code) | secondary diagnostic; not a benchmark row |

**Licence warning.** The Bonn responses are under the **Jev Responses License
1.0**: research and evaluation use; no model distillation; no training a model
to imitate Jev; no developing or facilitating similar or competing products.
The raw responses must not be bundled into Azurite, used as training data, or
redistributed. Aggregate facts may be cited with attribution. The TypeSafe site
terms are not a substitute for the API/customer agreement; read the applicable
service terms before redistributing outputs.

### 2.4a Reproduced vs published: the distinction this registry enforces

Every comparator point in the machine-readable registry now carries a
`comparison_class` and `reproduced` flag. The classes are deliberately narrow:

| class | count | meaning |
|---|---:|---|
| **published_reference** | 13 | A vendor or third-party number, cited with source, version, split, and n. It was not produced by this study. |
| **local_reconstruction** | 1 | The Banking77 0.924 cell. It is a stored per-example comparator vector reconstructed locally; it is **not** a live API call and **not** a reproduced run. |
| **reproduced_live** | **0** | Same rows, same pinned version, same frozen request, live call, request/response hashes. Nothing in this checkout meets that bar. |

So the correct sentence is: **this study has no reproduced live Jev
comparison.** It has thirteen published reference points and one local
reconstruction. A local reconstruction can be paired against our predictions,
but it is evidence about the public Jev vector, not a fresh measurement of Jev.
A published reference cannot be paired at all unless the rows and protocol are
reproduced.

The registry's `comparator_classes.live_reproduction_claim` records this as a
machine-readable zero; no future edit may turn it into a positive count without
a pinned run, frozen requests, and stored responses.

The counts above are for the 14 points attached to dataset entries in
`benchmark_registry.json`. The prose table in §2.5 also lists six
sequential/workflow/control points that are not benchmark rows; they are
published references too, but they are not counted in the dataset-attached
total.

### 2.5 Jev points by task, with conflicts left visible

These are **published reference points**, not paired runs, except the one row
explicitly marked `local reconstruction`. The same task name can hide a
different label space or split. Use the source and n, not the bare number.

| Task | Jev point | n | Source | Version | Note |
|---|---:|---:|---|---|---|
| AG News | 0.885 | 7,600 | Bonn (C3) | 1.13.0 | full test split, as reported |
| AG News | 0.910 | 100 | AbdelStark (C2) | 1.13.0 | pilot sample, not the full split |
| AG News | 0.910 | — | AY (C5) | 1.13.0 | **conflict** with Bonn; split not established |
| Banking77 | 0.797 | 3,076 | Bonn (C3) | 1.13.0 | 77-way; local mirror is 3,076 |
| Banking77 | 0.788 | 231 | AY (C5) | 1.13.0 | 77-way; small sample |
| Banking77 | 0.870 | 100 | AbdelStark (C2) | 1.13.0 | **72 labels**, not 77; not like-for-like |
| Banking77 | 0.924 | 3,080 | local reconstruction (C7) | reconstructed | 2,846/3,080; not a live API run |
| CLINC150 | 0.895 | 5,500 | Bonn (C3) | 1.13.0 | 151-way including OOS |
| DAIR Emotion | 0.480 | 100 | AbdelStark (C2) | 1.13.0 | pilot; 6-way |
| DAIR Emotion | 0.585 | — | `DATA.md` citation | 1.13.0 | **conflict**; split not established |
| Financial PhraseBank | 0.730 | 970 | Bonn (C3) | 1.13.0 | different split from local 1,000 |
| Language ID | 0.996 | 10,000 | Bonn (C3) | 1.13.0 | 20-way |
| XNLI (en) | 0.860 | — | AY (C5) via `suite_citations.json` | 1.13.0 | **no clean source/split**; do not use as a comparator until reconciled |
| SEC claim verification | 0.975 | 400 | Vals (C4) | 1.13.0 | supported / contradicted / insufficient |
| LegalBench 12-task slice | 0.730 class-balanced | 396 | Vals (C4) | 1.13.0 | last of twelve; raw accuracy also last |
| ALFWorld unseen | 35% success | 134 episodes | Bonn (C3) | 1.13.0 | sequential track; not a single-call decision |
| TextWorld | 42% success | 90 games | Bonn (C3) | 1.13.0 | sequential track |
| BabyAI-Text | 21% success | 250 missions | Bonn (C3) | 1.13.0 | sequential track |
| Blackjack | 90% optimal action | 5,000 | Bonn (C3) | 1.13.0 | sequential track |
| FrozenLake | near-goal avoidance failures | 500 | Bonn (C3) | 1.13.0 | sequential track |

**Do not average these.** The AG News 0.885 and 0.910 are different samples; the
Banking77 0.870 is a 72-label task; the XNLI point has no clean provenance. A
registry that reported a single “Jev mean” would be manufacturing a number.

### 2.6 Comparator runbook

When a live Jev comparison is authorised:

1. **Pin the version**: `jev-1.13.0` for alignment with C2–C5; `jev-1.14.0` for
   current-product comparison. Record the response's `model` field.
2. **Freeze the request**: state, instructions, every criterion, option order,
   and image header (if any). Hash the request JSON.
3. **Respect the primitives**: Choice ≤255 options; if K>255, decompose
   hierarchically and record the decomposition. Noul for yes/no; Score for
   ordered levels; do not ask for generation.
4. **Control order**: run at least two option permutations on a sample. 1.13 has
   a documented first-option bias.
5. **Calibrate on dev only**: Noul threshold and Choice/Score confidence gates
   are fitted on dev, frozen, then applied once to locked.
6. **Log**: resolved model, input/output tokens, latency, cost, raw
   probabilities, request hash, response hash, error/retry. The SDK retries 429s;
   do not hide them.
7. **Do not mix sources**: a C3 published point and a local run are separate
   rows. A paired claim requires identical rows and an identical request.
8. **Do not redistribute raw outputs** unless the applicable terms allow it.
   C3 responses are research/evaluation only and prohibit distillation or
   competing-product training.

**Cost estimate.** At $0.042/Mtok, 1,000 decisions with ~10k input tokens each is
~$0.42 before retries; 10,000 decisions is ~$4.20. The constraint is not price,
it is API access and the discipline of freezing requests. Budgets must still be
approved before spending.

---

## 3. The six families

Family weights below are **equal (16.67% each)** for a descriptive composite.
That is a transparency choice, not a product priority. Per-family results are
primary; the composite is not a capability claim. Weights may change only
before the registry is frozen, and the change must be recorded.

The materialised suites use a **different, per-task equal weighting**: 20% per
task in the five-task dev suite and 8.33% per task in the twelve-task locked
suite. That is a sampling control, not the six-family composite above. The two
weighting schemes are never averaged together, and neither is a capability
claim.

### 3.1 Multi-domain classification and routing

**Decision:** choose one label from a caller-supplied set, across domains and
label counts. Tests whether the model can score unseen labels rather than
memorise a fixed 77-way head.

**Core datasets (4 × 25% of family dev weight):**

| Dataset | Source | Licence | Splits (train / dev / locked) | Scoring | Jev public point? | Status |
|---|---|---|---|---|---|---|
| Banking77 | [PolyAI](https://huggingface.co/datasets/PolyAI/banking77), [paper](https://arxiv.org/abs/2003.04807) | **CC BY 4.0** | 10,003 / 1,500 carved / 3,080 | accuracy, macro-F1, ECE, Brier, selective accuracy | yes, conflicting (C2/C3/C5/C7) | **core**; local mirror is short |
| CLINC150 (`plus`) | [CLINC](https://huggingface.co/datasets/clinc/clinc_oos), [paper](https://arxiv.org/abs/1903.08097) | **CC BY 3.0** | 15,250 / 3,100 / 5,500 | accuracy, OOS recall/F1, ECE | yes, 0.895 (C3) | **core**; 151 labels incl. OOS |
| MASSIVE intent (en-US) | [AmazonScience](https://huggingface.co/datasets/AmazonScience/massive), [paper](https://arxiv.org/abs/2204.08582) | **CC BY 4.0** | 11,514 / 2,033 / 2,974 | accuracy, macro-F1 | no Jev point; Laya 0.783 | **core**; 60 labels |
| MASSIVE scenario (en-US) | same | **CC BY 4.0** | 11,514 / 2,033 / 2,974 | accuracy, macro-F1 | no | **core**; 18 labels, same corpus but different label space |

**Secondary (0% of weighted dev):** AG News (licence unresolved; 120,000 /
5,000 carved / 7,600; Jev 0.885 C3, 0.910 C2/C5 conflict), DAIR Emotion
(unresolved; 16,000 / 2,000 / 2,000; Jev 0.480 C2 vs 0.585 cited conflict), XNLI
en (unresolved; 392,702 / 2,490 / 5,010; pair classification; Jev 0.860
unreconciled), Language ID (unresolved; 70,000 / 10,000 / 10,000; Jev 0.996 C3).

**Controls:** majority/prior; lexical TF-IDF or character n-gram; random-init
same architecture; label permutation; second draw; candidate-order permutation;
K=2/8/32/full-K width curve. The old suite already shows the width curve is not
flat, so report width, not only full-K accuracy.

**Why this family is not enough:** it is the easiest family for a classifier and
the one most likely to be contaminated by public pretraining. It cannot carry a
general-capability claim.

### 3.2 Document-grounded decisions and evidence sufficiency

**Decision:** answer from supplied evidence, return the supporting span(s), and
abstain when the evidence does not decide. The target is not open-domain QA; it
is “given this document, is the claim supported, contradicted, or not decidable?”

**Core datasets (4 × 25%):**

| Dataset | Source | Licence | Splits (train / dev / locked) | Scoring | Jev public point? | Status |
|---|---|---|---|---|---|---|
| SQuAD 2.0 | [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [paper](https://arxiv.org/abs/1806.03822) | **CC BY-SA 4.0** | 130,319 / 2,000 carved / 11,873 official dev | EM, token F1, answerability F1, ECE | no direct Jev point | **core**; official test hidden; dev is the public eval |
| MuSiQue-Full v1.0 | [MuSiQue](https://github.com/StonyBrookNLP/musique), [paper](https://arxiv.org/abs/2108.00573) | **CC BY 4.0** | 19,938 / 2,417 / 2,459 | answer F1, support F1, group answer/support sufficiency F1 | no | **core**; Full adds unanswerable; use official evaluator |
| QASPER | [QASPER](https://huggingface.co/datasets/allenai/qasper), [paper](https://arxiv.org/abs/2105.03011) | **CC BY 4.0** | 2,593 / 500 carved / 1,005 official dev | answer F1, evidence F1, answerability | no | **core**; no official test labels; do not invent a 1,451 test split |
| FEVER v1.0 | [FEVER](https://fever.ai/dataset/fever.html), [paper](https://arxiv.org/abs/1803.05355) | **CC BY-SA 3.0** plus GPL-3.0 metadata; Wikipedia text | 311,431 / 10,000 carved / 18,999 `paper_dev` | label accuracy, FEVER score, evidence F1 | no | **core**; split names are a trap (`labelled_dev` 37,566, `unlabelled_dev` 19,998, `paper_dev` 18,999) |

**Secondary (0%):** SciFact (claims CC BY 4.0, abstracts ODC-By 1.0, code
Apache-2.0; 809 / 339 / 300 with test labels withheld), HotpotQA (CC BY-SA 4.0;
90,447 / 2,000 / 7,405; support + answer), BoolQ (CC BY-SA 3.0; 9,427 / 1,000 /
3,270; yes/no with passage), LLM-AggreFact (gated; do not use until terms and
contamination are checked).

**Controls:** no-retrieval (full document vs retrieved passages); lexical BM25
retrieval; text-only vs document-only; blank/shuffled evidence; answerable vs
unanswerable strata; evidence-position early/middle/late; hop count; distractor
count. The abstention metric is not optional: a model that always answers is
not a document-grounded decider.

**Why this family is not enough:** SQuAD-style evidence is short and often
lexically overlapping. QASPER and ContractNLI add long documents; MuSiQue adds
multi-hop composition. None of them proves tool use or policy compliance.

### 3.3 Tool/action selection and abstention

**Decision:** choose the next tool/action, fill arguments, and ask for missing
information or refuse when no available tool fits. This family is about the
interface between a decision and an executable environment.

**Core datasets (4 × 25%):**

| Dataset | Source | Licence | Splits (train / dev / locked) | Scoring | Jev public point? | Status |
|---|---|---|---|---|---|---|
| BFCL V3 single-turn live | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html), [V3 blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html), [code](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | **Apache-2.0** | 2,251 live / 500 carved from non-live / 2,251 | AST accuracy, executable accuracy, irrelevance detection | no | **core**; V3, not V4 |
| BFCL V3 multi-turn | same | **Apache-2.0** | 800 / 100 carved / 700 | state-based success, missing-parameter/function detection, long-context subset | no | **core**; custom split, not leaderboard-comparable |
| τ²-bench `text-v1` (English) | [τ²-bench](https://github.com/sierra-research/tau2-bench), [paper](https://arxiv.org/abs/2406.12045) | **MIT** | 150 tasks (50 × airline/retail/telecom) / 100 (airline+retail) / 50 (telecom) | pass^3, policy compliance, action accuracy | no | **core**; domain holdout; 2.0 scores are not 1.x-comparable |
| API-Bank | [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank), [paper](https://arxiv.org/abs/2304.08244) | **Apache-2.0** | 314 dialogues / 100 / 214 | API-call accuracy, result accuracy, ROUGE-L, planning | no | **core**; 73 tools, 753 calls; custom split |

**Secondary (0%):** BFCL V3 irrelevance (240 non-live + 882 live; subset of the
single-turn rows; do not double-count), BFCL V4 (agentic web/memory/format; not
comparable to V3), StableToolBench (Apache-2.0 data but RapidAPI terms and
unstable APIs), τ² `banking_knowledge` (retrieval domain; MIT; not the same task
as text-v1).

**Controls:** no-tool/forced-answer; available-tool removal; missing-parameter
subset; irrelevant-tool subset; one-turn vs multi-turn; API error/retry;
tool-name paraphrase; argument permutation. Record executable success, not only
AST match.

**Why this family is not enough:** BFCL is function-calling, not general
planning; τ² is customer-service policy; API-Bank is a smaller synthetic tool
set. A high score here does not imply document reasoning or visual grounding.

### 3.4 Policy and rule-conditioned decisions

**Decision:** apply an explicit policy or rule set to a case, with the policy in
the input, and distinguish entailed, contradicted, and not-mentioned. This is
where “the model chooses an option” becomes “the model follows a rule it was
given.”

**Core datasets (4 × 25%):**

| Dataset | Source | Licence | Splits (train / dev / locked) | Scoring | Jev public point? | Status |
|---|---|---|---|---|---|---|
| ContractNLI | [ContractNLI](https://stanfordnlp.github.io/contract-nli/), [paper](https://arxiv.org/abs/2110.01799) | **CC BY 4.0** | 7,191 / 1,037 / 2,091 (607 contracts × 17 hypotheses) | 3-class macro-F1, evidence micro/macro mAP, P@R80 | no direct; Vals legal slice overlaps in spirit | **core**; document-level NLI |
| CUAD v1 | [CUAD](https://www.atticusprojectai.org/cuad/), [HF](https://huggingface.co/datasets/theatticusproject/cuad-qa) | **CC BY 4.0** | 408 contracts / 102 test; official split | clause presence F1, evidence span F1, per-category recall | no | **core**; 510 contracts, 41 clause types; frame as yes/no + evidence |
| PolicyQA | [PolicyQA](https://github.com/wasiahmad/PolicyQA), [paper](https://aclanthology.org/2020.findings-emnlp.66/) | **MIT code**; underlying OPP-115 text has research terms | 25,017 examples / 115 policies / 714 questions; split to be read from release | EM, F1, answerability | no | **core, research-only**; do not ship without OPP-115 terms |
| LegalBench Vals-12 | [LegalBench](https://github.com/HazyResearch/legalbench), [Vals](https://www.vals.ai/blogs/independent-evaluation-of-jev) | **mixed per task**; select only clear licences | 396 questions / 12 tasks; freeze task IDs | class-balanced accuracy, raw accuracy, ECE | **yes: 0.730 class-balanced (C4)** | **core, conditional**; not the full 162-task benchmark |

**Secondary (0%):** τ² policy compliance (cross-family; already counted in 3.3;
do not treat as independent), LEDGAR (contract provision classification; exact
split/licence to verify), PrivacyQA/OPP-115 (research terms), MAUD (merger
agreements; terms to verify).

**Controls:** policy removed; policy contradicted by the case; policy with an
exception; “not mentioned” vs “contradicted” boundary; negation; long document
with irrelevant clauses; evidence span required. A model that scores high on
the label but cannot cite the clause has not demonstrated policy compliance.

**Why this family is not enough:** legal contracts are one policy domain.
ContractNLI’s 17 fixed hypotheses are a small hypothesis space; CUAD is clause
detection, not full legal reasoning; LegalBench is a mixed-licence slice.

### 3.5 Relevance, preference, and ordinal judgments

**Decision:** rank alternatives, choose the preferred one, or place a case on an
ordered scale. These are three distinct output contracts and must be reported
separately, even though they share a family.

Family split: relevance 33.3%, preference 33.3%, ordinal 33.3%. Within each
subfamily, two datasets at 50/50 → each dataset 16.7% of the family. No dataset
exceeds 25%.

#### 3.5.1 Relevance

| Dataset | Source | Licence | Splits (train / dev / locked) | Scoring | Jev public point? | Status |
|---|---|---|---|---|---|---|
| MS MARCO passage | [MS MARCO](https://microsoft.github.io/msmarco/) | **non-commercial research**; terms unresolved for shipping | ~8.8M passages; dev qrels 6,980; use 3,000 dev / 3,980 locked by query ID | MRR@10, Recall@100, nDCG@10 | no | **core, research-only** |
| TREC DL 2020 passage | [TREC](https://trec.nist.gov/data/deep2020.html) | NIST qrels + MS MARCO corpus terms; research-only | 200 queries / 11,386 qrels; 100 dev / 100 locked | nDCG@10, MAP | no | **core, research-only**; four-point graded qrels |

**Secondary (0%):** BEIR selected 6 datasets (per-dataset licences; no umbrella
licence), Natural Questions retrieval, FiQA. Do not use a BEIR “average” without
naming the components and their licences.

#### 3.5.2 Preference

| Dataset | Source | Licence | Splits (train / dev / locked) | Scoring | Jev public point? | Status |
|---|---|---|---|---|---|---|
| HH-RLHF | [Anthropic](https://huggingface.co/datasets/Anthropic/hh-rlhf), [paper](https://arxiv.org/abs/2204.05862) | **MIT** | ~160.8k preference pairs / 5,000 / 8,552 test | pairwise accuracy, tie calibration | no | **core**; helpfulness and harmlessness subsets separate |
| UltraFeedback (binarized) | [OpenBMB](https://huggingface.co/datasets/openbmb/UltraFeedback), [paper](https://arxiv.org/abs/2310.01377) | **MIT**; upstream model-output terms apply | 64k prompts / 256k responses; 5,000 / 5,000 | pairwise accuracy, ECE | no | **core**; GPT-4 annotations; contamination unknown |

**Secondary (0%):** RewardBench, HelpSteer2, MT-Bench human preferences. Do not
mix model-judged and human preferences in one score.

#### 3.5.3 Ordinal

| Dataset | Source | Licence | Splits (train / dev / locked) | Scoring | Jev public point? | Status |
|---|---|---|---|---|---|---|
| Financial PhraseBank (all-agree) | [Takala](https://huggingface.co/datasets/takala/financial_phrasebank) | **CC BY-NC-SA 3.0** | 1,264 / 500 carved / 1,000 | 3-level accuracy, MAE, QWK, Spearman | yes, 0.730 at n=970 (C3) | **core, research-only** |
| SST-5 | [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/), [paper](https://aclanthology.org/D13-1170/) | **unresolved** | 8,544 / 1,101 / 2,210 | 5-level accuracy, MAE, QWK, Spearman | no clean point | **core, conditional**; cannot freeze until licence resolved |

**Secondary (0%):** Amazon Reviews 2023 5-core (licence unresolved), Yelp 5-star
(licence unresolved). Do not substitute a toy ordinal set for a blocked real
one; record the blocker.

**Controls:** random/majority; tie handling; adjacent-level confusion; class
imbalance; threshold/calibration; pairwise vs listwise; position bias. For
relevance, compare BM25, a frozen dense retriever, and the decision model on the
same qrels.

### 3.6 Multimodal decisions requiring visual evidence

**Decision:** use the image, not just OCR text, to choose an answer or evidence
region. The minimum viable test is a **visual-only subset** where the answer
changes with the image and OCR cannot recover it: checkbox marks, chart trends,
layout-dependent fields, evidence-page selection.

**Core datasets (4 × 25%):**

| Dataset | Source | Licence | Splits (train / dev / locked) | Scoring | Jev public point? | Status |
|---|---|---|---|---|---|---|
| VisualEvidence-Docs (planned) | project-generated from synthetic templates | **MIT** (project) | 1,000 docs / 500 / 500 QA (2,000 total) | visual-only accuracy, span/citation IoU, ECE | no | **core, planned**; not built |
| VisualEvidence-Charts (planned) | project-generated from synthetic templates | **MIT** (project) | 1,000 charts / 500 / 500 QA (2,000 total) | visual-only accuracy, relaxed accuracy, ECE | no | **core, planned**; not built |
| DocVQA | [DocVQA](https://www.docvqa.org/datasets/docvqa), [paper](https://arxiv.org/abs/2007.00398) | **unresolved**; images from UCSF Industry Documents Library | 50k questions / 12k images; train 39,463 / val 5,349 / test 5,188 (reported) | ANLS, exact match, answer/evidence | no | **core, conditional**; use val as locked until terms clear |
| ChartQA | [ChartQA](https://github.com/vis-nlp/ChartQA), [paper](https://aclanthology.org/2022.findings-acl.177/) | **GPL-3.0 repo metadata**; image provenance mixed/unresolved | 32,719 QA (9,608 human / 23,111 augmented); test 2,500 | relaxed accuracy, exact match, ECE | no | **core, conditional**; licence/provenance gate |

**Secondary (0%):** VisualMRC (terms unresolved; 30,562 QA / 10,000+ images),
SlideVQA (NTT evaluation terms; 14,484 QA / 2,619 decks; evidence-page
retrieval), InfographicsVQA (terms unresolved; 30,035 QA / 5,485 images),
PlotQA/FigureQA (synthetic charts; terms to verify).

**Controls (mandatory):** OCR-only; text-only; blank image; shuffled image;
image + OCR; image-only; evidence-page retrieval; chart/table parsing. A visual
claim requires a pre-registered margin over OCR-only on the visual-only subset.
If OCR-only is within 5 points, the multimodal claim is withdrawn regardless of
absolute accuracy.

**Why this family is blocked:** the two local anchors are planned, not built;
the public datasets have unresolved image/licence terms; the current model has
no visual encoder. This family cannot be in a locked suite today.

---

## 4. Long-context stress axis

Long context is not a seventh family; it is a stratifier across document,
policy, tool, and visual decisions.

### 4.1 Context bands

| Band | Byte length (approx.) | Datasets |
|---|---:|---|
| short | <2,000 | classification, preference, MS MARCO passages, SQuAD paragraphs |
| medium | 2,000–8,000 | SQuAD 2.0, MuSiQue, FEVER pages, DocVQA text |
| long | 8,000–32,000 | QASPER papers, ContractNLI NDAs, PolicyQA policies |
| very long | >32,000 | CUAD contracts, SlideVQA decks, multi-document retrieval |

These are byte bands, not token bands. Every item records both byte length and
the model tokeniser's length; a comparison that mixes them is invalid.

### 4.2 Required measurements

- Accuracy at 2k, 8k, 32k, and (if claimed) 64k bytes.
- Evidence position early / middle / late; 0 / 4 / 8 distractors; distributed
  evidence across passages.
- Full-context vs retrieval-then-decide; retrieval recall and evidence coverage.
- Latency p50/p95, peak memory, and cost at each length.
- Truncation audit: how many items lost the answer or the hypothesis. The old
  XNLI/Emotion harness defects were truncation and scoring-path bugs; do not
  repeat them.
- No single long prompt is evidence. A 32k “needle” test is a smoke test, not a
  capability result.

### 4.3 Current facts that constrain the claim

- Azurite large-model native window: **1,024 bytes**. The 5,120-byte arm is
  **87.11%**, below the native 89.06% on the same rows.
- The 14.4M checkpoint was pretrained at **256 positions**. On the 2026-10-30
  fixed-checkpoint dev curve, no task is context-additive: classification is
  flat, BoolQ is at majority, SQuAD 2.0 answerability gains +1.47 pp but stays
  at majority, and ContractNLI falls **69.79% → 37.03%** at 4,096 bytes. A flat
  curve for a majority-collapsed head is not evidence about the encoder.
- Jev default: **32k per question / 64k per request**; 262k only in the
  danger-zone mode, which the vendor says is slower and less accurate.
- Jev's own docs warn about context rot: irrelevant detail in the state costs
  accuracy. A long-context comparison must therefore report distractor count,
  not only length.
- The 16k/32k Azurite projections are **projections**, not results.

---

## 5. Contamination and licence gates

### 5.1 Contamination status

| Scope | Status | Evidence |
|---|---|---|
| Existing 20 local tasks | **partially checked** | exact-window + 13-gram audit; 17 hits in 2,000 probes; WikiText-103 24% overlap; verbatim only |
| Banking77 product path | **checked** | 25/9,993 support messages removed; 255 rows affected; original 89.03 not clean |
| New registry datasets | **not checked** | no scan against corpus v2 yet |
| Jev training data | **unknown/undisclosed** | vendor says customer requests are not training data; benchmark contamination is not disclosed |
| Visual datasets | **not checked** | image-level near-duplicate and OCR/template leakage not audited |

### 5.2 Required contamination audit before a locked run

1. Exact 24–64 character window match against the pretraining corpus.
2. 13-gram overlap with document-level counts.
3. Near-duplicate detection (MinHash/SimHash) at the item and passage level.
4. Time-split check for news/contracts/policies where dates exist.
5. Canary/held-out private rows where the public benchmark is likely memorised.
6. Report **raw and decontaminated** results; never silently drop rows.
7. For visual data, check image near-duplicates, chart templates, and OCR text
   leakage separately.

A zero-hit result from a verbatim-only checker is not “uncontaminated”; it is
“no verbatim match found.”

### 5.3 Licence and shipping status

| Status | Datasets | Consequence |
|---|---|---|
| **clear for commercial use** | Banking77 (CC BY 4.0), MASSIVE (CC BY 4.0), SQuAD 2.0 (CC BY-SA 4.0), MuSiQue (CC BY 4.0), QASPER (CC BY 4.0), ContractNLI (CC BY 4.0), CUAD (CC BY 4.0), BFCL (Apache-2.0), API-Bank (Apache-2.0), τ² (MIT), HH-RLHF (MIT), UltraFeedback (MIT) | attribution/share-alike obligations still apply |
| **research-only** | MS MARCO, TREC DL, Financial PhraseBank (NC-SA), PolicyQA/OPP-115, XNLI (non-commercial per `DATA.md`), AG News/Emotion/Language ID (unresolved) | thesis/research use only; not shipped |
| **unresolved** | SST-5, DocVQA, ChartQA image provenance, VisualMRC, SlideVQA, InfographicsVQA, LegalBench task-level subset | do not freeze or ship until resolved |
| **restricted outputs** | Jev Responses License 1.0 (Bonn raw responses) | no distillation, no imitation training, no competing-product use, no redistribution |
| **gated** | ContractNLI HF, LLM-AggreFact | accept terms; record account/date; do not assume redistribution |

**Share-alike warning.** CC BY-SA datasets (SQuAD 2.0, FEVER, HotpotQA) can
require derivative datasets to be shared under compatible terms. A model
trained on them is not automatically a derivative, but the legal question is
not decided here. Record it; do not hand-wave it.

---

## 6. Dev and locked protocol

### 6.1 What is frozen now

- The six family definitions.
- The core/secondary roles and per-dataset weights.
- The comparator source/version table.
- The licence statuses.
- The rule that unresolved datasets cannot be in a shipped path.
- **The v1 dev and locked suites** (2026-10-30): exact item IDs, class-balanced
  samples, source-record hashes, and file hashes in `research/suites/`. The
  locked suite is sealed and has not been opened. See `MISSION_LEDGER.md` §14.
- **Calibration is not dev-test reuse.** v1 has no calibration partition, and no
  threshold or abstention policy was fitted. Calibration requires a separate
  partition carved before fitting and disjoint from train, dev, and locked; that
  requires a new suite version. Fitting on the inspected dev test or on locked is
  a protocol violation.

### 6.2 What is not frozen

- A **new** suite version, if one is created. v1 is frozen; it is not reopened.
- The per-family primary metric and multiplicity plan.
- The prompt templates and option order.
- The Jev run (no API key/budget in this checkout).

### 6.3 Freeze procedure (executed for v1; required for any later version)

1. Download each core dataset at a pinned revision; record the source URL,
   revision, file hashes, row counts, and licence.
2. Build dev from train/validation only; build locked from test or a held-out
   public split. Never build dev from locked.
3. Write `split_manifest.json` with item IDs, byte lengths, labels, and source
   hashes; hash the manifest.
4. Write a pre-registration per family: primary metric, controls, success/fail
   rule, multiplicity correction, and stopping rule.
5. Do **not** run the model on locked until the method is frozen.
6. Record any deviation in an amendment; never edit the original manifest.

### 6.4 Comparability rules

- A published Jev point is not a paired comparison.
- A local Jev run is paired only if the same rows and the same request bytes are
  used; log the resolved model ID.
- BFCL V3 and V4 are different benchmarks. τ² 2.0 and 1.x are not comparable.
- TREC DL and MS MARCO use different qrels; do not average them.
- FEVER `paper_dev`, `labelled_dev`, and `unlabelled_dev` are different splits.
- Banking77's 77-label task and AbdelStark's 72-label BTZSC task are not the
  same label space.
- QASPER has no official test labels; a “test” number must name its source.

---

## 7. Blockers

1. **No live Jev run.** No API key or spend authorization in this checkout. The
   registry can cite C2–C5 but cannot make a paired claim outside Banking77.
2. **Comparator coverage is classification-heavy.** Jev has public points for
   classification/ordinal tasks, but little or nothing for SQuAD 2.0, MuSiQue,
   QASPER, BFCL, τ², API-Bank, ContractNLI, CUAD, PolicyQA, MS MARCO, DocVQA,
   ChartQA, or VisualMRC. Absence of a published point is not a Jev failure.
3. **Locked suite is materialised but incomplete.** Twelve tasks are sealed and
   hash-verified, but tool/action, relevance/ordinal, and multimodal families
   have no licensed local rows. O7 stays partial until those families are
   represented or explicitly waived.
4. **No document/tool/policy/visual implementation.** The model is still a
   fixed-label text classifier. Registry entries do not create capability.
5. **Licences unresolved** for SST-5, DocVQA, ChartQA image provenance,
   VisualMRC, SlideVQA, InfographicsVQA, PolicyQA/OPP-115, and the LegalBench
   subset. XNLI, AG News, Emotion, and Language ID are research-only or
   unresolved.
6. **No positive long-context result.** The 5,120 arm is negative. The
   fixed-checkpoint 256→4,096 dev curve is now complete: **no task is
   context-additive**, ContractNLI falls 32.75 points, and the document heads
   are at majority. See [`results/context_scaling/RESULTS.md`](../results/context_scaling/RESULTS.md).
   16k/32k remain projections.
7. **No visual encoder or image path.** VisualEvidence anchors are planned, not
   built.
8. **No contamination audit for the new datasets.** The existing audit is
   verbatim-only and covers 20 tasks.
9. **Split ambiguity** in FEVER, QASPER, BFCL, τ², CUAD, PolicyQA, and LegalBench.
   Each needs a pinned split decision, not a remembered convention.
10. **Comparator conflicts** for AG News, DAIR Emotion, Banking77, and XNLI.
    They must be reconciled by source, or reported as separate rows.
11. **Jev 255-option limit.** A 500-way or 1,000-way label set requires
    hierarchical decomposition; that decomposition changes the task.
12. **Jev 1.13 vs 1.14.** Public evidence is 1.13; the product is 1.14. No
    independent 1.14 result was found.
13. **Licence and storage cost** for DocVQA/ChartQA/VisualMRC/SlideVQA images;
    these are not downloaded.
14. **Local Banking77 mirror is 14 rows short** (9,993/3,076 vs canonical
    10,003/3,080); the product path uses 3,080 rows from another source. Do not
    merge the two counts.

---

## 8. Next experiments

These are registry-level, not capability claims. None starts with training.

### B0 — Freeze the split manifest (done 2026-10-30)

**Action:** download core datasets at pinned revisions, build dev/locked, write
and hash `split_manifest.json`. **Success:** every core dataset has item IDs,
hashes, byte lengths, labels, and a licence status; locked is not opened.

**Status:** the five-task dev suite and twelve-task locked suite are frozen and
hash-verified. `scripts/verify_suites.py` checks weights, counts, digests, file
hashes, and disjointness. The locked suite is sealed. Three families and four
licences remain open, so this completes B0, not O7.
**Cost:** storage/download only.

### B1 — Licence resolution (no GPU)

**Action:** resolve or formally block SST-5, DocVQA, ChartQA, VisualMRC,
SlideVQA, InfographicsVQA, PolicyQA/OPP-115, and the LegalBench task subset.
**Success:** each is `clear`, `research-only`, or `blocked`; no “assume CC BY.”
**Cost:** reading terms and, where needed, email to the dataset owner.

### B2 — Six-family smoke harness (inference, 100 rows/family)

**Action:** run the current checkpoint through the packing/evaluation path on
100 rows per family. **Success:** every family produces a per-example record,
including abstention/failure; no family silently crashes or is dropped.
**Decision:** identifies interface gaps before any expensive run.

### B3 — Paired Jev comparison (API, budget approval required)

**Action:** 200 rows each on Banking77, SQuAD 2.0, and BFCL V3; run
`jev-1.13.0` and `jev-1.14.0`; freeze requests and option order.
**Success:** paired accuracy with confidence intervals, resolved model ID, cost,
and raw probabilities; a loss is reported as a loss. **Cost:** likely under $10
at current pricing, plus access setup.

### B4 — Contamination audit of the core registry

**Action:** exact, 13-gram, and near-duplicate scans against corpus v2; report
raw and decontaminated results. **Success:** every core dataset has a hit count
and a decision (keep, drop, or stratify). **Cost:** CPU/IO.

### B5 — Long-context ladder (inference; training only if 4k is non-inferior)

**Action:** evaluate QASPER, MuSiQue, and ContractNLI at 2k/8k/32k with evidence
positions and distractors. **Success:** a length/position curve with truncation
audit; no claim beyond the measured range. **Cost:** inference; 16k/32k training
is gated.

### B6 — Visual anchor (data + small adaptation)

**Action:** build VisualEvidence-Docs and -Charts; measure OCR-only, text-only,
blank-image, shuffled-image, and image-only controls. **Success:** a visual-only
subset where the answer changes with the image; no visual claim if OCR-only is
within 5 points.

### B7 — Calibration and abstention (inference)

**Action:** fit thresholds on dev, freeze, evaluate on locked; report ECE,
Brier, AURC, selective accuracy at 80/50% coverage. **Success:** the
pre-registered risk target is met on locked; otherwise the claim is withdrawn.

---

## 9. Registry completion criteria

The registry itself is complete only when:

- All six families have at least two datasets; core families have four where
  the task supports it.
- No dataset exceeds 25% of its family's dev weight.
- Every dataset has source, licence status, split counts, scoring, comparator
  status, contamination status, and local availability.
- The locked split is materialised and hashed; the manifest is linked here.
- The comparator runbook is pinned to versioned model IDs.
- Long-context and visual stress axes are represented in the split, not only in
  prose.
- Every unresolved licence is either resolved or marked `blocked` with the
  consequence.
- The machine-readable JSON validates against the same rules.

It is **not** complete merely because this Markdown file exists. The current
status is **v0.1 draft**: content complete enough to review, split not frozen.

---

## 10. Evidence index

### Comparator

- TypeSafe models: https://docs.typesafe.ai/models
- TypeSafe System One API: https://docs.typesafe.ai/api/system-one
- TypeSafe primitives: https://docs.typesafe.ai/primitives
- TypeSafe Choice: https://docs.typesafe.ai/primitives/choice
- TypeSafe confidence: https://docs.typesafe.ai/confidence
- Jev 1.13 jaggedness: https://docs.typesafe.ai/model-jaggedness/jev-1.13
- TypeSafe launch: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe WorkflowEvals: https://github.com/typesafe-ai/WorkflowEvals
- AbdelStark benchmark: https://github.com/AbdelStark/jev-benchmarks
- Bonn study: https://arxiv.org/abs/2609.37647
- Bonn code: https://github.com/AppliedMachineLearning-Lab/jev-benchmarking
- Bonn responses: https://doi.org/10.5281/zenodo.23039006
- Vals evaluation: https://www.vals.ai/blogs/independent-evaluation-of-jev
- AY Automate: https://www.ayautomate.com/blog/jev-vs-llm-benchmark
- Ravvich: https://github.com/ravvich/jev-benchmark
- Local comparator: [`results/jev_matched/`](../results/jev_matched/)
- Local comparator narrative: [`results/comparison/FULL_SUITE.md`](../results/comparison/FULL_SUITE.md)

### Datasets

- Banking77: https://huggingface.co/datasets/PolyAI/banking77
- CLINC150: https://huggingface.co/datasets/clinc/clinc_oos
- MASSIVE: https://huggingface.co/datasets/AmazonScience/massive
- SQuAD 2.0: https://rajpurkar.github.io/SQuAD-explorer/
- MuSiQue: https://github.com/StonyBrookNLP/musique
- QASPER: https://huggingface.co/datasets/allenai/qasper
- FEVER: https://fever.ai/dataset/fever.html
- SciFact: https://github.com/allenai/scifact
- HotpotQA: https://hotpotqa.github.io/
- BoolQ: https://huggingface.co/datasets/google/boolq
- BFCL: https://gorilla.cs.berkeley.edu/leaderboard.html
- τ²-bench: https://github.com/sierra-research/tau2-bench
- API-Bank: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank
- StableToolBench: https://github.com/zhichengg/StableToolBench
- ContractNLI: https://stanfordnlp.github.io/contract-nli/
- CUAD: https://www.atticusprojectai.org/cuad/
- PolicyQA: https://github.com/wasiahmad/PolicyQA
- LegalBench: https://github.com/HazyResearch/legalbench
- MS MARCO: https://microsoft.github.io/msmarco/
- TREC DL 2020: https://trec.nist.gov/data/deep2020.html
- BEIR: https://github.com/beir-cellar/beir
- HH-RLHF: https://huggingface.co/datasets/Anthropic/hh-rlhf
- UltraFeedback: https://huggingface.co/datasets/openbmb/UltraFeedback
- Financial PhraseBank: https://huggingface.co/datasets/takala/financial_phrasebank
- SST-5: https://nlp.stanford.edu/sentiment/
- DocVQA: https://www.docvqa.org/datasets/docvqa
- ChartQA: https://github.com/vis-nlp/ChartQA
- VisualMRC: https://huggingface.co/datasets/NTT-hil-insight/VisualMRC
- SlideVQA: https://github.com/nttmdlab-nlp/SlideVQA
- InfographicsVQA: https://www.docvqa.org/datasets/infographicvqa

### Local provenance

- Data record: [`DATA.md`](../DATA.md)
- Mission ledger: [`research/MISSION_LEDGER.md`](MISSION_LEDGER.md)
- Product spec: [`docs/PRODUCT_SPEC.md`](../docs/PRODUCT_SPEC.md)
- Architecture: [`research/ARCHITECTURE.md`](ARCHITECTURE.md)
- Old suite: [`results/suite_final/`](../results/suite_final/)
- Erratum: [`results/suite_final/ERRATUM.md`](../results/suite_final/ERRATUM.md)

---

## 11. Change log

- **2026-10-30 — v0.1 draft.** Created the six-family registry, comparator
  register, licence table, conflict table, long-context/visual axes, blockers,
  and B0–B7. No dataset was downloaded, no model was run, no locked split was
  hashed, and no capability claim was added. O7 remains **partial**.
- **2026-10-30 — v0.2 suites and comparator classes.** Materialised a five-task
  compact dev suite (20% each; train-only, zero document overlap) and a
  twelve-task locked suite (8.33% each, 493–500 rows), both hash-verified. The
  locked suite excludes the 45 BoolQ passages that overlap its train split.
  Added `published_reference` / `local_reconstruction` / `reproduced_live` to
  every comparator point; live reproduced Jev runs: **0**. Added the
  fixed-checkpoint context-scaling pre-registration. No capability claim was
  added and O7 remains **partial**.
- **2026-10-30 — v0.3 context-scaling result.** The pre-registered fixed-checkpoint
  curve completed on five dev tasks. Zero context-additive cells; four degraded
  (all ContractNLI); the five-task mean falls 70.60% → 64.34%. Added the result
  to §4.3 and marked B0 complete. The locked suite remains sealed and unopened.
