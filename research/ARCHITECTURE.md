# Azurite architecture decision record

**Status:** architecture open; incumbent measured; no candidate selected.
**Date:** 2026-10-30.
**Related:** [`docs/PRODUCT_SPEC.md`](../docs/PRODUCT_SPEC.md),
[`research/CAPABILITY_TARGETS.json`](CAPABILITY_TARGETS.json),
[`research/MISSION_LEDGER.md`](MISSION_LEDGER.md).

This record exists because the current byte encoder is an inherited choice, not
a requirement. The contract and the capability targets are fixed; the encoder,
tokenizer, pooling, retrieval, head, and training procedure are hypotheses. A
component stays only if it wins on the development registry at an acceptable
licence, contamination, size, and latency cost.

**Decision in one sentence:** keep the 113.47M byte encoder as the measured
incumbent and regression baseline, screen pretrained encoders and alternative
heads without training the encoder first, and select the architecture on the
development registry only. Do not start another from-scratch pretraining run
merely because the previous one exists.

---

## 1. What is fixed and what is open

| Layer | Status | Why |
|---|---|---|
| Product contract and family semantics | **fixed** | `docs/PRODUCT_SPEC.md`; changing it would invalidate the targets |
| Capability thresholds and evaluation gates | **fixed** | `research/CAPABILITY_TARGETS.json`; amendments only by dated pre-registration |
| Tokenizer | **open** | byte, corpus BPE, and pretrained subword are all admissible |
| Encoder | **open** | byte transformer, pretrained encoder, or hybrid |
| Pooling | **open** | mean, query attention, CLS, or learned pooling |
| Retrieval | **open** | BM25, dense, hybrid, or no retrieval |
| Decision head | **open** | dual cosine, cross-encoder span scorer, hybrid rerank, ordinal/pairwise heads |
| Training procedure | **open** | from scratch, continued pretraining, supervised fine-tune, frozen probe, adapters |

The current shipped path is one point in this space: byte tokenizer, 113.47M
bidirectional byte encoder, message-query attention pool, BM25 over 9,968
support messages, scaled-cosine 77-way head. It is a strong Banking77 result and
a weak general decision system. Those are both true and must not be conflated.

---

## 2. The incumbent, stated honestly

### 2.1 What it is

- 113,465,856 encoder parameters; 114,845,185 including the packaged query head.
- 12 layers, `d_model=768`, 24 heads, byte vocabulary 260, native 1,024 bytes.
- Trained from random initialisation on 1,134,673,920 byte tokens from the local
  v2 corpus. No pretrained weights, no downloaded vocabulary.
- Checkpoint SHA-256
  `aba65b60e30654a9895dc9c365d62b555f96dcce96ad8a8ad56cc718ee0865ec`,
  1,361,732,233 bytes.

### 2.2 What it has measured

| Measurement | Result | Record |
|---|---:|---|
| Banking77 full 3,080, clean product path | **89.06%** | `results/product_path_clean/summary.json` |
| Banking77, 5,120-byte arm | **87.11%** | `results/jev_query_5120/eval/summary.json` |
| Nine-cell suite, 113.47M | **81.12% mean** | `results/suite_large_113m/COMPARISON.json` |
| XNLI at 113.47M | **33.90%** (uniform predictor) | same |
| 14.4M vs 113.47M, same token budget | 113M wins 4, loses 1, 4 unmoved | `RESULTS.md` |
| Candidate-width sweep | width 2 is much better than full K | `results/e4_candidate_sweep/` |
| Fixed-checkpoint context curve, 256→4,096 | **no context-additive task**; ContractNLI 69.79→37.03 | `results/context_scaling/RESULTS.md` |

### 2.3 Why it is not automatically the answer

- It is trained from scratch because the earlier project chose that, not because
  a comparison showed it beats pretrained encoders.
- It has no positive long-context result: its 5,120-byte arm is worse, and the
  fixed-checkpoint curve has no context-additive task.
- It has no document-grounded, pairwise, ordinal, boolean, or abstention result.
- It is 1.36 GB fp32, which is a real local-deployment cost.
- More capacity did not buy decision accuracy on the suite; the 14.4M rung beat
  the 33.7M rung on six of nine cells at the matched recipe.

The byte encoder is therefore the **incumbent to beat**, not the default to
preserve. Its licence and provenance are clean, which is an advantage; that
advantage does not establish capability.

---

## 3. Candidate components

### 3.1 Encoders

All candidates below are **hypotheses**. None is admitted. Revisions are the
local cache refs observed on 2026-10-30; they must be pinned and re-checked at
admission. Weights were not present in the audited cache for the pretrained
candidates, only configs/tokenizers, so no weight hash is claimed here.

| Candidate | Size / context | Licence (model card) | Why consider | Main risk |
|---|---|---|---|---|
| Azurite byte 113M | 113M / 1,024 bytes | MIT (project) | incumbent; clean provenance; exact byte offsets; measured | 1k window; expensive per byte at long context; no general head |
| Azurite byte 14.4M | 14.4M / 256 bytes | MIT (project) | best measured Banking77/CLINC at matched recipe; small; fast | short window; weaker other cells |
| `answerdotai/ModernBERT-base` | ~149M / 8,192 tokens | Apache-2.0 | modern bidirectional encoder; 8k token window covers far more bytes than 1,024; strong long-context candidate; replaceable pooling | 2T-token web/code pretraining is opaque; benchmark contamination likely; CPU latency unknown |
| `answerdotai/ModernBERT-large` | ~395M / 8,192 tokens | Apache-2.0 | ceiling candidate; the Laya comparator uses a ModernBERT-large backbone | twice the size; likely too slow for the CPU-first product; same contamination risk |
| `sentence-transformers/all-MiniLM-L6-v2` | ~23M / 512 tokens | Apache-2.0 | cheap frozen-feature baseline; good CPU latency | short context; sentence-pair pretraining may overlap NLI tasks; weak long context |
| `BAAI/bge-small-en-v1.5` | ~33M / 512 tokens | MIT | small retrieval/embedding baseline; good frozen features | short context; weak-supervision pretraining overlaps public tasks |
| `BAAI/bge-base-en-v1.5` | ~109M / 512 tokens | MIT | larger retrieval baseline | short context; same contamination risk; no long-context advantage |

Pinned local refs observed:

```
ModernBERT-base       8949b909ec900327062f0ebf497f51aef5e6f0c8
ModernBERT-large      45bb4654a4d5aaff24dd11d4781fa46d39bf8c13
all-MiniLM-L6-v2      1110a243fdf4706b3f48f1d95db1a4f5529b4d41
bge-small-en-v1.5     5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
bge-base-en-v1.5      a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
```

Licence sources: the model cards at
<https://huggingface.co/answerdotai/ModernBERT-base>,
<https://huggingface.co/answerdotai/ModernBERT-large>,
<https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>,
<https://huggingface.co/BAAI/bge-small-en-v1.5>, and
<https://huggingface.co/BAAI/bge-base-en-v1.5>. Licence is necessary, not
sufficient: the contamination and redistribution gates below still apply.

### 3.2 Tokenizers

| Option | Evidence | Trade-off |
|---|---|---|
| Byte (incumbent) | exact round-trip; 1.000 byte/token; 1,024-byte window | no unknown tokens; exact UTF-8 offsets; sequence cost is one position per byte |
| Corpus BPE, 32k | measured **3.327 bytes/token**, 0 round-trip failures on 300 held-out docs | 3.3× more text per position; tokenizer must be shipped; no completed full-scale BPE pretrain in this checkout |
| Pretrained subword | ModernBERT 50,368 vocab; BERT 30,522 | no tokenizer training; but vocabulary and normalisation are foreign, and byte offsets require mapping |
| Hybrid (subword + byte fallback) | not measured | best of both in principle; extra complexity; must be tested, not assumed |

The BPE measurement is in `results/tokenizer/TOKENIZER.md`; it is a compression
result, not a decision result. The 3.327× sequence-length advantage is large
enough to matter for 16k bytes, but it does not establish better accuracy.

### 3.3 Pooling

| Option | Status | Risk |
|---|---|---|
| Mean pooling | baseline in the old head; cheap | dilutes the message when examples dominate the context |
| Message-query attention (incumbent) | +29.90 points over mean pooling on identical Banking77 contexts | query is task-specific; may overfit the message field |
| CLS token | standard for BERT-family encoders | needs a pretrained CLS that learned a useful summary; not guaranteed |
| Learned attention pool | open | more parameters; needs a control against query pooling |

The current mechanism result is real but narrow: it isolates pooling on one
task. It is not evidence that query pooling is the right general head.

### 3.4 Retrieval

| Option | Status | What it must prove |
|---|---|---|
| BM25 (incumbent) | measured product path | strong lexical baseline; no dense training |
| Dense bi-encoder | candidate | retrieval recall and end-to-end decision gain over BM25, not just embedding similarity |
| Hybrid BM25 + dense | candidate | gain over both, at acceptable latency |
| No retrieval | control | separates reading from retrieval; already the message-only control in Banking77 |

Retrieval is part of the system when evidence exceeds the window. A better
retriever that does not improve decisions is not a product improvement.

### 3.5 Decision heads

| Head | What it supports | Status |
|---|---|---|
| Dual scaled-cosine | arbitrary K, cheap candidate cache | incumbent; works at K=151; candidate order fixed |
| Cross-encoder span scorer | context and candidate seen jointly; stronger for hard pairs | used in suite for K<=16; expensive at large K |
| Hybrid dual + cross rerank | cheap shortlist, expensive top-k | hypothesis; must beat both components |
| Pairwise margin head | pairwise decisions and ties | not implemented |
| Ordinal cut-point head | ordered alternatives | not implemented |
| Sufficiency/abstention head | explicit abstain | not implemented |

The product head must be candidate-conditioned. A fixed 77-way table is a
baseline, not the contract.

### 3.6 Training procedure

| Procedure | When admissible | Why |
|---|---|---|
| From scratch | only with a licence-clean corpus at least ~10× model parameters and a funded multi-GPU budget | current 1.63B-token corpus and one 95 GiB card do not justify another from-scratch run |
| Continued pretraining | after a pretrained candidate is admitted and a local corpus is licence-cleared | adapts domain without paying full pretraining cost; risk of forgetting |
| Supervised fine-tune | default for a pretrained candidate | task adaptation is cheap relative to pretraining |
| Frozen features + linear probe | first screen; no encoder training | cheapest way to test whether the representation is useful |
| Parameter-efficient adapters | if full fine-tune exceeds memory or time | must be compared at equal inference cost |
| Distillation | after a teacher is selected; for local deployment | can produce a small byte/student model without retraining from scratch |

**Do not train from scratch merely because previous work did.** The existing
from-scratch run is a sunk asset and a baseline, not a mandate. A new
from-scratch run needs a written justification: corpus size, licence, compute,
and a reason a pretrained initialisation cannot answer the question.

---

## 4. Pretrained component admission gate

A pretrained component is admitted only when all of the following are recorded:

1. **Identity.** Exact model id, revision commit, config hash, tokenizer hash,
   weight SHA-256. No floating `main`.
2. **Licence.** SPDX identifier, commercial-use permission, redistribution
   permission, notice/attribution requirements, and a copy of the model card.
3. **Provenance.** Pretraining corpus description, languages, size, cutoff date,
   and known benchmark exclusions. If the corpus is undisclosed, record
   `undisclosed`; do not infer cleanliness from the licence.
4. **Contamination.** Exact-window and 13-gram overlap against the dev and
   locked splits; benchmark canary strings where available; a date-cutoff check.
   If overlap is found, either remove the affected rows from the clean report or
   report the result as an upper bound. Do not call it clean.
5. **Reproducibility.** Pin the revision; use `trust_remote_code=False`; load
   safetensors, not pickle; record the library versions; run offline at
   inference.
6. **Size and latency.** Parameter count, weight bytes, peak memory, p50/p95
   latency on the reference CPU and GPU; must fit the deployment target.
7. **Behaviour.** A licence is not a capability claim. The component must pass
   the dev screen before it is used in a product path.

### 4.1 Contamination risk, stated plainly

| Component | Risk | Reason |
|---|---|---|
| Azurite byte | **medium** | local corpus, but only verbatim/13-gram audited; 24% WikiText-103 overlap; 0.16% decision-suite hits; paraphrase/translation invisible |
| ModernBERT-base/large | **high** | ~2T tokens of primarily English web/code; exact mixture undisclosed; benchmark text may be present |
| MiniLM | **high** | 1B sentence pairs from heterogeneous sources; NLI/QQP-style tasks are plausible sources |
| BGE-small/base | **high** | large-scale weak supervision; retrieval/NLI data likely overlaps public benchmarks |

A pretrained encoder's score on a public benchmark is an **upper bound** until
the overlap screen is run. This is not a reason to reject pretrained models; it
is a reason not to call a benchmark win a capability win without the screen.

---

## 5. Screening protocol

Architecture selection happens on the **development registry**, never on the
locked suite, and never on Banking77 alone.

### Stage A — frozen-feature probe (cheap, no encoder training)

1. Build a dev split from each task's **train** split: 2,000 rows, stratified,
   seed 0. Do not use the inspected test split for architecture selection.
2. Extract one pooled vector per row from each candidate:
   - byte encoder: masked mean of the final hidden state; query-pooled vector as
     a second view;
   - BERT/MiniLM/BGE: CLS and mean, whichever the model card specifies;
   - ModernBERT: mean pooling with the attention mask.
3. Fit the same multinomial logistic regression on 4,000 rows for every
   candidate; tune only L2 on a 1,000-row dev slice; evaluate once on the 2,000
   dev rows.
4. Report accuracy, macro-F1, NLL, ECE after a single temperature fit, and
   extraction latency. Use paired bootstrap over rows.
5. Tasks: AG News (4-way), Banking77 (77-way), CLINC-OOS (151-way), XNLI
   (3-way), and one document-grounded task once the registry exists.

**Stop rule:** if a pretrained candidate does not beat the byte encoder's frozen
features by at least 2 points on the dev mean with a 95% interval excluding
zero, do not spend GPU on full fine-tuning. A tie is a reason to keep the
licence-clean incumbent.

### Stage B — supervised fine-tune of finalists

Top two candidates by Stage A, plus the byte incumbent, get the same head, the
same 4,000 rows, the same optimizer/LR/step budget, and three seeds. The
comparison is paired by row and by seed. Report the seed variance, not only the
best seed.

**Replacement rule:** a candidate replaces the incumbent only if it meets one
of:

- **+2.0 points** on the dev mean with a 95% paired interval excluding zero; or
- **+5.0 points** on a mission-critical family (document-grounded, long-context,
  visual) with no more than **2×** the incumbent's p50 latency and **2×** its
  artifact size.

Parity is not a reason to switch. A larger model that is 0.2 points better at
3× latency is a worse local product.

### Stage C — long context

ModernBERT-base's 8,192-token window is the strongest long-context prior among
the candidates. It is still a hypothesis: 8,192 subword tokens is not 16,384
bytes, and a pretrained position table is not a guarantee of quality at the
edge. The long-context test is the one in `docs/PRODUCT_SPEC.md` §4: 1k/4k/8k/
16k/32k bytes, early/middle/late positions, 0/2/8 distractors, no-evidence
control, latency and memory.

A chunked encoder remains admissible even if a full-window model passes; the
chunked path may be cheaper at 32k. The two are compared on the same rows.

### Stage D — visual

Visual evidence is a separate component decision. A frozen open-licence vision
encoder plus an adapter is the hypothesis; OCR-only is the mandatory control. A
text encoder choice does not settle the visual path, and an OCR pipeline does
not become multimodal because the text encoder is pretrained.

---

## 6. Decision record

### 6.1 Current decision

- **Architecture remains open.** No encoder, tokenizer, pooling, retrieval, or
  head is selected beyond the measured incumbent.
- **The byte encoder is the incumbent and regression baseline.** It is not the
  default.
- **Pretrained components are permitted**, subject to the admission gate.
- **The first architecture experiment is a frozen-feature probe**, not a new
  pretraining run.
- **No from-scratch pretraining** until a written justification passes the
  corpus/licence/compute gate.
- **No architecture selection on Banking77 alone**; the dev registry decides.

### 6.2 Rejected arguments

| Argument | Why it is rejected |
|---|---|
| "We already trained the byte model, so use it." | Sunk cost. The existing checkpoint is a baseline, not evidence of the best architecture. |
| "Pretrained is always better." | Not at fixed latency, size, licence, and contamination. The 113M vs 14.4M suite result already shows larger is not uniformly better. |
| "BPE is better because 3.327 bytes/token." | Compression is not decision quality. The BPE arm has no completed full-scale pretrain or suite result. |
| "ModernBERT is Apache-2.0, so it is clean." | A licence is not a contamination audit. The pretraining corpus is opaque. |
| "A 4B model would settle it." | At 1.63B tokens that is D/N=0.41 and would measure memorisation; the measured 113M run already shows the failure mode. |
| "Scale will close the gap." | 113M vs 14.4M won 4, lost 1, and left 4 unmoved; XNLI degenerated. Scale is not the missing variable on this suite. |

### 6.3 What would change the decision

- A pretrained candidate beats the byte incumbent by the Stage B rule on the dev
  registry, at acceptable latency and size.
- A document-grounded or long-context candidate clears its gate where the byte
  encoder cannot.
- A byte/BPE hybrid closes the long-context gap without a pretrained licence or
  contamination debt.
- A distillation result gives most of a pretrained teacher's quality at the
  byte model's deployment cost.

Until one of those is measured, the honest statement is: **the architecture is
open and the current model is a baseline, not a selection.**

---

## 7. Open questions

1. Does ModernBERT-base's 8k tokenizer window actually cover 16k bytes of
   decision evidence without a quality drop? (Stage C)
2. Does a frozen pretrained representation beat the byte encoder on 77-way and
   151-way routing, or only on short 4-way tasks? (Stage A)
3. Is the query-attention pool still useful with a pretrained encoder, or does
   CLS/mean pooling win? (Stage B)
4. Does dense retrieval improve end-to-end document decisions over BM25, or only
   retrieval recall? (Stage A/C)
5. Can a byte student be distilled from a pretrained teacher and keep the
   licence-clean deployment path? (later)
6. Which open-licence vision encoder passes the OCR-only control? (Stage D)

---

## 8. Provenance and reproduction

- This record changes no model, checkpoint, result, thesis, or paper.
- It uses only already-measured numbers; no new accuracy is claimed.
- Candidate licences were checked on 2026-10-30 from the linked model cards;
  they must be re-checked at admission.
- The local cache contains configs/tokenizers for the candidates but not their
  weights; no weight hash is claimed. A candidate screen must download and hash
  the weights before it can be admitted.
- The architecture decision is not final. It is a protocol for making the
  decision without inheriting the previous one.
