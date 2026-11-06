# Azurite — persistent mission ledger

**Status:** working audit and plan. **Last audited:** 2026-10-30.
**Checkout:** branch `clean-slate`, HEAD `7d26a080`. Most mission artifacts in
this checkout are untracked, so the commit hash does **not** identify them. The
evidence path and the hashes recorded below are the provenance; do not treat
`git rev-parse HEAD` as a complete artifact manifest.

This is the durable record for the mission. It is append-only with respect to
negative results: corrections may be added, but no inconvenient result, failed
run, stale claim, or missing artifact may be deleted to make the record cleaner.
`thesis/GAPS.md` remains the detailed list of thesis holes; this ledger is the
research and product control plane.

**Mission, in one sentence:** build Azurite as a general-purpose, locally
deployable decision model competitive with TypeSafe's Jev, with substantially
longer useful context, multimodal evidence processing, and reliable decisions,
and document the work honestly in an Overleaf-compatible thesis and paper.

**Current answer: the mission is not met.** The shipped artifact is a
Banking77-only 77-way classifier. It is a real, measured artifact, but it is a
legacy experiment and a narrow research preview, not a general decision model,
not competitive with Jev, not long-context, and not multimodal. A product
specification and pre-training capability targets now exist; no general
decision API or checkpoint exists yet.

---

## 1. Objectives and status

Status vocabulary: **verified** = reproduced in this audit or by a stored,
hash-verified record; **partial** = a component exists but the objective is not
met; **missing** = no implementation or measurement; **contradicted** = a
written claim disagrees with the artifact.

| # | Objective | Status | Evidence / reason |
|---|---|---|---|
| O1 | Audit the existing code, checkpoints, datasets, experiments, papers, and thesis; preserve provenance and negative results | **partial** | This ledger; `thesis/GAPS.md`; 95 tests pass; suite manifest verifies 92 files. Several docs remain stale (§4). |
| O2 | Define a product that decides over user-supplied alternatives | **partial** | Contract and targets now specified in `docs/PRODUCT_SPEC.md` and `research/CAPABILITY_TARGETS.json` (2026-10-30, §12). No public `decide(context, alternatives)` surface or checkpoint exists yet. |
| O3 | Support boolean, classification/routing, pairwise/ordinal, document-grounded, policy-conditioned, and tool/action decisions | **missing** | Only fixed-label classification is implemented and measured. `azurite/decisions.py` can pack a fixed dataset task, but there is no runtime API or trained checkpoint for arbitrary alternatives. |
| O4 | Abstain when evidence is insufficient | **missing** | Forced 77-way argmax. No `none`/`unknown` option, reject head, threshold policy, or calibrated abstention guarantee. |
| O5 | Demonstrate substantially longer useful context | **contradicted** | Native window is 1,024 bytes. A 5,120-byte arm is **worse** (87.11% vs 89.03%). The 2026-10-30 fixed-checkpoint curve (14.4M, trained at 256, evaluated to 4,096) finds **no context-additive task**: classification controls are flat, SQuAD 2.0 gains +1.47 pp but stays at majority, and ContractNLI falls **69.79% → 37.03%**. No 8k/16k/32k result and no positive long-context result. |
| O6 | Demonstrate multimodal evidence processing | **missing** | Text bytes only. The thesis explicitly says no multimodal data was trained on and no multimodal result exists. |
| O7 | Build a broad benchmark registry and a locked evaluation suite | **partial** | Registry v0.2 names six families and 44 datasets. The compact dev suite (5 tasks, 20% each) and a **sealed 12-task locked suite** (493–500 rows each, 8.33% each) are materialised and hash-verified in `research/suites/`. Tool/action, relevance/ordinal, and visual families are still missing; five locked tasks are research-only. Comparator classes are explicit: 13 published references, 1 local reconstruction, **0 live reproduced Jev runs**. |
| O8 | Protect scientific validity: frozen checkpoints, pre-registration, contamination audit, fresh final evaluation | **partial** | Strong pre-registration/hash discipline on several runs; contamination audit exists; 27 suite cells are one seed each; test sets have been repeatedly inspected; no untouched final suite. |
| O9 | Deliver a working product | **partial** | Wheel installs, CLI works, stored-record replay passes. Weights are excluded from the wheel; the default checkpoint is not installed in the audited cache; no server/API; no general decision surface. |
| O10 | Deliver an Overleaf-compatible thesis and justified paper | **verified (as documents)** | `thesis/main.pdf` 84 pages; `paper/azurite.pdf` 9 pages; 0 LaTeX errors; 0 undefined references/citations. The paper is a negative-result comparator paper, not a general-capability paper. |
| O11 | Choose the architecture on measured evidence, not inheritance; admit pretrained components only with provenance, licence, and contamination records | **partial** | Architecture is explicitly open in `research/ARCHITECTURE.md` (2026-10-30). The 113M byte encoder is the incumbent, not the default. Candidate licences are checked; no candidate weights are hashed, no screen has run, and no replacement is selected. |

The ordering is deliberate: O2–O4 and O7 are prerequisites for meaningful work on
O5/O6. O11 is a cross-cutting requirement: no architecture choice, including
keeping the byte encoder, is valid without the dev registry and a measured
comparison.

---

## 2. Verified capabilities

These are measured or mechanically checked, not inferred from prose.

### 2.1 Model and training

- Byte-level bidirectional encoder; 113,465,856 parameters at the large rung
  (`d_model=768`, 12 layers, 24 heads). The packaged query checkpoint has
  114,845,185 parameters including the head.
- Large pretraining run `l1_large_byte_iso10v3`: 1,134,673,920 tokens seen
  against a 1,134,658,560 target; step 69,255.
- Full checkpoint: 1,361,732,233 bytes, SHA-256
  `aba65b60e30654a9895dc9c365d62b555f96dcce96ad8a8ad56cc718ee0865ec`.
  It is present at
  `results/recovered_20261028/checkpoints/seg_9_step_69255_aba65b60e306.pt`.
- Corpus v2: 1,633,580,647 tokens, 3,267,161,294 bytes, SHA-256
  `9267a2ea071a517803b3b47c01adf9464608a8178fbf12018ce5dcc9ca38365f`.
- The small-rung checkpoint `results/pretrain/l1_small_byte_ctx256/ckpt.pt`
  hashes to
  `8fd8bedee794ee4aa464af7c11fb06eb1d8dcfbfe82f529f8106a11a798ff131`.
- The 2.41M-parameter transfer probe is the controlled instrument; the 4B
  ceiling is not a result and is correctly labelled a permission.

### 2.2 Banking77 product measurement

- Official Banking77 mirror used here: 3,080 test rows, 9,993 train rows. The
  canonical release is 3,080/10,003; the mirror is 14 rows short. This is a
  known data limitation, not rounded away.
- **Shipped decontaminated path:** 2,743/3,080 = **89.06%**, 95% Wilson
  **87.91–90.11**. The bundled support bank is **9,968** rows (25 evaluation
  texts removed from 9,993; 6 exact).
- **Original path:** 2,742/3,080 = **89.03%**, 95% Wilson **87.87–90.08**.
  It is **not clean**: 25 of 9,993 support messages are evaluation texts;
  255 rows (8.28%) retrieve at least one; 283 instances (25 self, 258 cross-row).
- **Unaffected subset:** 2,511/2,825 = **88.88%** (87.67–89.99).
- Decontamination changes 256 contexts and 3 predictions (2 wins, 1 loss;
  exact McNemar p = 1.00). The overlap did **not** inflate the original
  headline.
- **Jev comparator:** 2,846/3,080 = **92.40%**. The primary pre-registered
  query head is **−3.38 points** (exact McNemar p = 3.99e-9; paired 95% lower
  bound −4.32). The clean product path is **−3.34 points** against that
  comparator.
- **This is not a win.** The paper's title and abstract correctly describe a
  negative result.

### 2.3 Comparator controls and mechanism

- Old mean-pool head on the same frozen contexts: 1,821/3,080 = 59.12%.
- Query head on the same contexts: 2,742/3,080 = 89.03%.
- Mechanism gain: **+29.90 points** (McNemar p = 3.62e-212); this isolates
  mean-pool dilution from the retrieval change.
- Query head, message only: 1,880/3,080 = 61.04% (out-of-distribution control;
  trained with example contexts).
- The 5,120-byte joint fine-tune: 2,683/3,080 = **87.11%** (85.88–88.25),
  **−5.29 points** vs Jev, and **−1.92 points** vs the native query head.
  It reads the examples (+27.99 over old head on packed contexts) but does not
  buy parity. This is the closest thing to a long-context experiment and it is
  a negative result.

### 2.4 Suite, scale, and candidate width

- Nine-task suite, 14.4M / 33.7M / 113.47M, one pre-registered recipe:
  27 cells (2000 test rows each; Financial PhraseBank 1000). Three cells were
  repeated; repeat deltas up to **1.25 points**, flipping up to 274/2,000
  predictions. The suite manifest verifies **92/92 files**.
- At 113.47M vs 14.4M (7.95× parameters, same token budget): wins AG News
  +2.60, CLINC150 +1.70, MASSIVE intent +2.80, Financial PhraseBank +2.00;
  loses XNLI −9.95; four cells statistically unmoved. **Scale is not the path
  to parity on this suite.**
- XNLI at 113.47M is degenerate: one class on 94.7% of rows, NLL 1.0986 = ln 3,
  mean confidence 0.3337. This is a uniform output, not a weak attempt.
- Held-out bits/byte at D/N=10 falls from 1.8234 at 2.41M to 0.4515 at 113.47M;
  fitted α = 0.341 (3 exhaustive rungs) / 0.355 (4 sampled rungs). A clean
  proxy law and flat decision accuracy coexist.
- E4 candidate-set sweep: 12 fresh heads, 3 wide-bank tasks, widths
  {2,8,32,K}. The pre-registered 5-point non-inferiority rule holds on 3/3
  tasks at width 32 vs K. But the curve falls with width: width 2 reaches
  98.05/98.00/96.70; width 32 reaches 92.10/92.20/82.85; full K reaches
  89.15/87.70/81.90. This is not a flat-width result.
- Transfer at 2.41M: Banking77 **+42.1 points** over matched random init
  (three seeds); class-balanced AG News **+1.13 points**, crossed-bootstrap 95%
  interval **−2.60 to +4.80**, therefore inconclusive. The historical AG News
  gain is withdrawn.

### 2.5 Packaging and reproducibility

- Wheel: `dist/pkg/azurite_encoder-0.1.0-py3-none-any.whl`, **249,460 bytes**,
  SHA-256 `ebfb6e0e78d2dc3a9aeb7ad5c3b3ff2824a57778e26f33ad670607a444a50e43`.
- Bundled model card is byte-identical to `MODEL_CARD.md` (checked with `cmp`).
- `python -m azurite info` reports **9,968** bundled support rows, 25 of 9,993
  removed, 89.06% on 3,080 rows, and the 89.03% original arm with its overlap
  disclosure. It correctly says the default checkpoint is **not installed** in
  the audited cache.
- Stored-record replay passes **8/8 native** and **4/4 5,120-byte** prediction
  and context-digest checks (documented in `PRODUCT_VERIFICATION.md`).
- Tests: **99 passed** in 21.96 seconds on this checkout (2026-10-30)
  (`.venv/bin/python -m pytest -q`).
- Suite manifest: **92/92 files verified** with
  `python scripts/verify_manifest.py results/suite_large_113m`.
- Documents: `thesis/main.pdf` **84 pages**, `paper/azurite.pdf` **9 pages**,
  0 LaTeX errors, 0 undefined references, 0 undefined citations in the current
  build. The thesis has 93 bibliography entries, all cited; the paper has 21.

### 2.6 Data and audit assets

- 20 evaluation tasks, 2,659,054 rows on disk (9 primary, 5 multilingual,
  6 secondary). `DATA.md` and `data/MANIFEST.json` record every split hash.
- Pulled corpus: 5,142,706 documents, 20.60 GB text, 21 shards, from five
  sources. The v2 training stream is the 1.63B-token stream above.
- Contamination audit: exact-window and 13-gram arms across all corpus
  documents; 24% WikiText-103 overlap (no LM number is reported there); three
  decision-suite hits in ~1,900 (0.16%), reported rather than rounded to zero.
- Held-out overlap: 0/600 sampled 128-id windows found in training; one shared
  64-byte fragment (1/600). Positive and negative controls exist.
- `results/recovered_20261028/` preserves the lost large run, its hashes, and
  the recovery record. `_failed_wave1/` and `_failed_wave2/` preserve the
  CUDA-OOM waves as infrastructure records, not results.

---

## 3. Unsupported, stale, or contradicted claims

This section is the audit's most important output. A claim is listed here if it
is broader than the artifact, or if a document says something the artifact no
longer supports. **Do not delete these entries after fixing the documents; mark
them corrected and keep the record.**

### 3.1 Mission-level claims not supported by the product

| Claim | Where it appears | What the artifact shows | Status |
|---|---|---|---|
| “general-purpose … decision model” | Mission statement; implicit in project framing | `Predictor` is hard-wired to 77 Banking77 labels. `pyproject.toml` describes “A byte-level 77-way Banking77 intent classifier.” No arbitrary candidate API. | **Not supported** |
| “competitive with Jev” | Mission statement | No head-to-head win. Banking77 is **3.38 points below** pre-registered Jev (3.34 below on the clean product path). Other suite cells compare to published points, not paired Jev runs. | **Contradicted on the only paired task** |
| “substantially longer useful context” | Mission statement; README framing | Native 1,024 bytes. 5,120-byte arm is lower (87.11%). Jev median reconstructed prompt is 10,683 bytes, so Azurite is **shorter**, not longer. No 16k/32k result. | **Contradicted** |
| “multimodal evidence processing” | Mission statement | No image/audio/document pipeline, no multimodal training, no visual-evidence evaluation. Thesis says so explicitly. | **Missing** |
| “highly reliable decisions” | Mission statement | Forced 77-way choice; softmax confidence is uncalibrated; no reject option. The risk–coverage curve is descriptive on one test set, not a deployment guarantee. | **Not supported** |
| “decisions over user-supplied alternatives” | Mission statement | `TaskSpec`/`CandidateBank` exist for dataset experiments, but the installed API accepts only a text string and returns one of 77 fixed intents. | **Missing** |
| Boolean decisions | Mission statement | No boolean surface. A K=2 synthetic task is packable in research code, but no checkpoint, API, or evaluation exists. | **Missing** |
| Pairwise / ordinal judgments | Mission statement | No pairwise or ordinal head, data, metric, or run. | **Missing** |
| Document-grounded decisions / evidence sufficiency | Mission statement | Only message + retrieved examples; no arbitrary document evidence, no “insufficient evidence” state. | **Missing** |
| Tool/action selection, policy/rule-conditioned decisions | Mission statement | No task family, data, or implementation. | **Missing** |
| Near-perfect accuracy | Mission statement warns against it | The thesis does not claim it; the measured suite has clear deficits. This is a **correct** restraint, not a gap. | **Correctly not claimed** |

### 3.2 Stale status and generated documents

These are documentation defects, not scientific results. They must be corrected
or explicitly marked historical before the next external review.

| Artifact | Stale statement | Actual state | Action |
|---|---|---|---|
| `PLAN.md` | “Status: working tree reset … No code, no checkpoint, no claim.” | Code, checkpoints, suite, paper, and thesis exist. | Mark as historical restart plan; link to this ledger. |
| `README.md` | “Nothing is trained and nothing is spent until a corpus and a budget are approved.” | 113.47M checkpoint trained; 27 suite cells run; Jev/query/5,120 arms run; GPU-hours recorded. | Correct the status block; keep the budget warning for future spend. |
| `results/comparison/COMPARISON.md` | “largest local model is 86M”; “5.2B available locally”; Azurite cells “pending”. | Largest local model is 113.47M; corpus v2 is 1.63B tokens; nine-cell suite exists. | Regenerate or mark as the pre-suite snapshot. |
| `research/THESIS.md` | Build table says thesis 29 pages / paper 4 pages. | Thesis 84 pages; paper 9 pages. | Mark as historical research plan, not current build status. |
| `results/suite_large_113m/RESULTS.md` and `COMPARISON.json` | Checkpoint path is `results/pretrain/l1_large_byte_iso10v3/ckpt.pt`, 1.36 GB. | That path is absent. The matching 1.36 GB checkpoint is at `results/recovered_20261028/checkpoints/seg_9_step_69255_aba65b60e306.pt`; a 433 MB `weights_only.pt` derivative exists under the old directory with a different hash. | Record the recovered path and the derivative; do not claim the old path exists. |
| `thesis` abstract / `thesis/GAPS.md` | Fixed-data sweep: only tiny/small survive; mid/large reclaimed. Later abstract also reports a three-rung iso-D/N suite. | These are **different ladders**: fixed-`D` vs iso-`D/N`. The first is incomplete; the second has three rungs. | Keep the distinction explicit; do not let a reader collapse them. |
| `MODEL_CARD.md` | “3.38 points below” Jev. | 89.03% pre-registered arm is −3.38; the clean 89.06% arm is −3.34. | The card already labels the arm; keep it, or state both gaps explicitly. |

### 3.3 Claims that are correctly bounded

The thesis and paper are unusually explicit about limits. These are **not**
violations and should be preserved:

- “No U-shaped scaling law is claimed from the surviving two rows.”
- “The exhaustive protocol reduction is not available … not made here.”
- “The context curve is not available … no long-context claim is made.”
  **(Superseded 2026-10-30: a 4,096-byte curve now exists and is negative; §14.
  The thesis source has not yet been updated, which is a documented gap in
  `thesis/GAPS.md` §2b.)**
- “The coverage curve is not an abstention guarantee.”
- “Multimodality is unimplemented.”
- “Near-perfect accuracy across every benchmark is not demonstrated.”
- Paper: “we make no claim about multimodal models or about long-context
  models”; 1,020 bytes is far below long-context regimes.
- The 4B ceiling is a permission, not a result.

These statements must remain in the thesis/paper until replaced by measurements.

---

## 4. Failures and negative results to preserve

| Failure / negative result | Record | What it means |
|---|---|---|
| Fixed-data U-curve withdrawn | `thesis/sections/06-conclusion.tex`; `thesis/GAPS.md` §3 | Only tiny/small fixed-`D` cells survive; mid/large reclaimed. Do not restore the U-shape from memory. |
| Exhaustive-protocol reduction withdrawn | same; `thesis` abstract | The 0.352 → 0.0017 bits/byte claim belongs to the lost build; it is not made. |
| Long-context curve is negative | `results/context_scaling/`; `results/xnli_context`; `results/jev_query_5120` | 5,120 arm is lower; the 2026-10-30 fixed-checkpoint 256→4,096 curve has no context-additive task and ContractNLI falls 32.75 pp. No long-context capability. |
| Jev comparator missed three ways | `results/jev_matched/`, `results/jev_query/`, `results/jev_query_5120/` | 22.44% primary, 89.03% query head, 87.11% 5,120; all miss the 92.40% bar. |
| Scale does not close the suite | `results/suite_large_113m/RESULTS.md` | 7.95× parameters: 4 wins, 1 loss, 4 unmoved; XNLI degenerates. |
| XNLI collapse / uniform predictor | `results/suite_large_113m/` | 94.7% one class; NLL = ln 3. Not explained; no fix. |
| Candidate-width curve falls | `results/e4_candidate_sweep/` | 32-way non-inferiority holds, but width 2 is much better; not a flat curve. |
| Laya Banking77 0/20 invalid | `results/laya_baseline/laya_banking77_20rows.json` | 77-option interface returned 0/20; the harness had a label-string/index mismatch. It is a protocol failure, not a Laya capability number. |
| Candidate-only cross-scorer bug | `results/rootcause/RESULTS.md` | Emotion 16.45% was a scoring-path artifact; dual path fixed it. The bug and the fix remain recorded. |
| AG News historical gain withdrawn | `results/transfer/TRANSFER.md`; `thesis` abstract | Class-balanced comparison is inconclusive; do not quote the old gain. |
| Replication noise | `results/suite_large_113m/REPLICATION.md` | Up to 1.25 pp / 274 predictions; one-seed deltas below that are not capability. |
| Contamination is not zero | `results/product_path/summary.json`; `MODEL_CARD.md` | 25/9,993 support messages; 255 rows. Original 89.03 is not clean. |
| Verbatim-only contamination audit | `results/contam_check/CONTAM.md` | Paraphrase/translation/reformatting remains invisible. |
| Data/licence gaps | `DATA.md`; `data/MANIFEST.json` | Several tasks, including the XNLI variants, have unresolved licences (the exact per-task status is in `DATA.md`); Banking77 mirror 14 rows short; WikiText 24% overlap. |
| MoE not run | `results/moe_feasibility/FEASIBILITY.md` | Feasibility only; no arm trained, no result. |
| `verify_manifest.py` fails on `results/product_path_clean` | audit command | It reports unlisted `clean_support.jsonl.gz`, `per_example.jsonl`, `removed_support.json`, `summary.json`. The directory has a `MANIFEST.json`, not a complete `.sha256` file set. Do not call this clean directory “fully manifest-verified.” |

---

## 5. Missing functionality, by product family

| Family | Required behaviour | Current state | Concrete gap |
|---|---|---|---|
| Boolean | Decide yes/no over supplied evidence | Missing | No API, no checkpoint, no data/eval |
| Classification / routing | Choose among user-supplied labels | Only fixed 77-way Banking77 | No arbitrary label set at inference; no routing dataset |
| Pairwise / ordinal | Compare A vs B or rank ordered alternatives | Missing | No head, loss, labels, or metric |
| Document-grounded | Answer from a supplied document with evidence | Missing | No document encoder path beyond a message; no evidence spans or sufficiency |
| Evidence sufficiency | Abstain when the document does not decide | Missing | No “insufficient” label, no reject option |
| Tool/action selection | Pick a tool/action from context | Missing | No tool registry, no action data |
| Policy/rule-conditioned | Apply an explicit policy to a case | Missing | No policy input or compliance metric |
| Abstention | Return “unknown” or a calibrated reject | Missing | Forced argmax; confidence is uncalibrated |
| Long context | Preserve quality across lengths, positions, distractors | Not achieved | 1,024 native; 5,120 worse; 4,096 fixed-checkpoint curve measured and negative; 8k/16k/32k not run |
| Multimodal | Use visual evidence, not just OCR | Missing | No image/audio path, data, or evaluation |
| Calibration / reliability | Report ECE and risk–coverage; validate a threshold | Partial | ECE exists in research; product softmax is not calibrated; risk curve is descriptive |
| Deployment | Local, reproducible, latency/memory reported | Partial | Wheel excludes weights; no server; no p50/p99 or throughput SLO for the product |
| Benchmark registry | Multi-family, license-audited, locked suite | Partial | Six-family registry; 5-task dev and 12-task sealed locked suite hashed; tool/ordinal/visual families missing; no document/tool/policy/visual implementation |
| Fresh final evaluation | Untouched locked test after method freeze | Missing | Test sets have been repeatedly inspected; no final holdout |

---

## 6. Blockers

1. **No general decision interface.** The model and checkpoint are label-fixed.
   Adding an API is cheap; making it meaningful requires training/evaluation on
   arbitrary candidate sets, not just wrapping the 77-way head.
2. **No positive long-context evidence.** The 5,120-byte arm is negative, and
   the 4,096-byte fixed-checkpoint curve has no context-additive task. The
   16k–32k full-information arm is projected at **12.6–15.9 GPU-hours (16k)**
   and **39.8–56.2 GPU-hours (32k)** before further allowance; under the 4k
   stop rule it is not authorised until a length-adapted 4k arm passes.
3. **No multimodal data or model path.** Nothing has been trained on images,
   audio, or documents. A pipeline and visual-evidence evaluation are prerequisites.
4. **The registry is drafted and the suites exist, but three families do not.**
   The six-family registry names 44 datasets; the 5-task dev and 12-task locked
   suites are hashed and sealed. No dataset has been downloaded for tool/action,    relevance/ordinal, or multimodal visual, and five locked tasks are
    research-only. Architecture selection on the current suite would still be
   classification-heavy.
5. **Compute and budget.** Local RTX PRO 6000, 95 GiB, 90-minute idle drop,
   12-hour session ceiling, 16 GB free disk; spend authorization is $0. The
   target-scale run is not fundable from this checkout.
6. **Checkpoint path drift.** The suite records name
   `results/pretrain/l1_large_byte_iso10v3/ckpt.pt`, which does not exist. The
   hash-matching 1.36 GB checkpoint is under `results/recovered_20261028/`.
   A 433 MB `weights_only.pt` derivative is not the same file. Any replay must
   use the recovered path and verify `aba65b60…`.
7. **Licences and mirrors.** Several datasets have unresolved licences; the
   Banking77 mirror is 14 rows short; WikiText-103 overlaps the corpus. These
   constrain publication and shipping, not just footnotes.
8. **No untouched final test.** The current numbers are legacy evidence;
   repeated inspection means they cannot be presented as fresh confirmation.
9. **One seed per suite cell.** Replication exists for three cells only; the
   1.25-point repeat floor is larger than many reported deltas.
10. **Documentation drift.** `PLAN.md`, `README.md`, `COMPARISON.md`, and
    `research/THESIS.md` contain stale status claims. They mislead a new reader
    even though the thesis itself is careful.
11. **Architecture cannot be selected yet.** A compact dev suite now exists,
    but it covers only classification, document, and policy rows; candidate
    comparison would still omit three families and would reuse inspected test
    sets. Pretrained candidates also have opaque pretraining corpora; their benchmark scores are
    upper bounds until the overlap screen is run. The framework is in
    `research/ARCHITECTURE.md`; the selection is not.

---

## 7. Next experiments

Rules for this queue:

- No new Banking77 head or prompt tuning until the general decision surface and
  the benchmark registry exist. Banking77 is a regression test, not the agenda.
- Every experiment below needs a pre-registration, a frozen checkpoint, a
  paired control, a success/failure rule, and a cost. If it cannot change a
  mission-level decision, do not run it.
- “Verified” means a stored per-example record and a hash, not a prose summary.

### N0 — Repair the record (no GPU)

**Hypothesis:** the current documentation drift is the main reason a reader
would overestimate the product.
**Action:** mark `PLAN.md`, `COMPARISON.md`, `research/THESIS.md` as historical;
correct `README.md`'s “nothing is trained” block; record the recovered
checkpoint path in the suite record; add a single “current state” pointer to
this ledger.
**Success:** every stale claim in §3.2 is either corrected or visibly labelled
historical; no result file is edited.
**Cost:** <1 hour.
**Decision:** required before any new claim is written.

### N1 — Define and test the general decision contract (no GPU)

**Hypothesis:** the interface, not the encoder, is the first blocker.
**Action:** implement the contract already frozen in `docs/PRODUCT_SPEC.md`
(`decide(context, alternatives, policy) -> {choice, score, abstain, evidence}`);
provide boolean, multi-class, pairwise, ordinal, and document-grounded adapters
over the existing encoder/head where possible; add unit tests for arbitrary
label sets, empty/duplicate alternatives, candidate-order invariance, and
insufficient evidence.
**Controls:** fixed 77-way path must remain byte-identical; a K=2 and K=8
synthetic task must be scored by the same code path.
**Success:** all five families are expressible and testable through one public
API; no model retraining required for the interface test.
**Cost:** engineering only, ~1–2 days.
**Decision:** if the existing head cannot score arbitrary candidates without
leakage, record that as a model limitation and design the next head.

### N2 — Build the benchmark registry and freeze a dev/locked split (no training)

**Hypothesis:** the current nine-task suite over-represents classification and
cannot justify general-capability claims.
**Action:** create `research/BENCHMARK_REGISTRY.md` with, per family, source,
licence, split, scoring, contamination status, and comparator status. Minimum
families: multi-domain classification/routing; document-grounded decisions and
evidence sufficiency; tool/action selection and abstention; policy/rule
conditions; relevance/preference/ordinal; visual evidence. Use a compact dev
suite and a locked suite; never select architecture on the locked suite.
**Success:** every family has at least two datasets with resolved licences or a
recorded unresolved status; no dataset exceeds 25% of the dev weight; the locked
suite is hashed and not opened.
**Cost:** 1–2 days, no GPU.
**Decision:** no architecture selection without a registry. If a family has no
licensable dataset, record the blocker rather than substituting a toy task.

**Result (2026-10-30): registry drafted, not frozen.** Created
`research/BENCHMARK_REGISTRY.md` and `research/benchmark_registry.json`. The
registry covers six families — classification/routing, document-grounded and
evidence sufficiency, tool/action and abstention, policy/rule-conditioned,
relevance/preference/ordinal, and multimodal visual —with 44 datasets (26 core, 18 secondary). The four-dataset core families use
25% per dataset; the relevance/preference/ordinal family uses three
subfamilies at 33.3333%, with two datasets at 16.66665% each. No dataset exceeds 25% of its family weight. The
JSON parses and the weight/required-field checks pass. The comparator register
separates eight evidence sources (TypeSafe vendor material, TypeSafe
WorkflowEvals, AbdelStark, Bonn, Vals, AY, Ravvich, and the local Banking77
reconstruction), pins `jev-1.13.0`
for published-study alignment and `jev-1.14.0` for current-product comparison,
and records twenty cited points with n/source/version. It also records the
comparator conflicts for AG News, DAIR Emotion, Banking77, and XNLI rather than
choosing a convenient number. **Not done:** no dataset was downloaded, no split
manifest was hashed, no model or Jev API was run, no visual anchor was built, and
no licence was resolved. O7 stays **partial**. The next action is B0 (freeze the
split manifest), not a training run.

**B0 update (2026-10-30):** the split manifest is now materialised. The dev
suite is five tasks at 20% each, built only from training splits with zero
document overlap (2,499–2,500 train / 748–750 test). The locked suite is twelve
tasks at 8.33% each, 493–500 rows each, sealed and hash-verified. The lock is
procedural: nine locked tasks were inspected earlier. Three families remain
blocked. See §14 for artifacts, hashes, and the context-scaling result. O7 stays
**partial** because the suite is not complete across six families.

### N3 — Measure the current model across the dev registry (inference only)

**Hypothesis:** the 113M encoder has a measurable capability floor outside
Banking77, and the floor will not be uniform across families.
**Action:** evaluate the frozen 113M checkpoint on every dev family; report
accuracy, ECE, risk–coverage, abstention rate, latency p50/p95, throughput, and
memory. Use paired controls where a text-only or majority baseline exists.
**Success:** every family is measured with uncertainty; no family is silently
dropped; the result determines whether the next spend is data, context, or
multimodality.
**Cost:** inference; estimate from the nine-cell suite (8.343 GPU-hours for 27
cells) and record the actual.
**Decision:** if document-grounded/ordinal/tool families are near chance, the
bottleneck is data/interface, not scale.

### N3a — Architecture screen on the dev registry (inference only)

**Hypothesis:** the inherited byte encoder is not the best representation for
general decisions, and a pretrained encoder or a different head will beat it on
the dev registry at acceptable cost.
**Action:** run the Stage A frozen-feature probe in
`research/ARCHITECTURE.md`: byte encoder (mean and query views), ModernBERT-base,
all-MiniLM-L6-v2, and BGE-small, with the same logistic head, rows, and seed.
Only finalists proceed to Stage B fine-tuning. Do not use the inspected test
splits; build the dev split from each task's train split. No encoder training in
Stage A.
**Success:** a ranked candidate table with paired intervals; either a candidate
clears the replacement rule (+2.0 dev mean with a 95% interval excluding zero,
or +5.0 on a mission-critical family at no more than 2x latency/artifact), or the
byte incumbent is retained on evidence rather than inheritance.
**Cost:** inference plus logistic regression; no GPU training. Weight download
and hashing are required before a candidate is admitted.
**Decision:** if no candidate beats the incumbent by the rule, keep the byte
encoder and proceed; if one does, allocate Stage B fine-tuning. Do not start N4
before this decision.

### N4 — Long-context training, staged (GPU, only after N2/N3/N3a)

**Hypothesis:** length adaptation at 4k preserves decision quality, and a
16k arm can be attempted only if 4k is non-inferior.
**Action:** train/adapt the selected encoder (the 113M byte incumbent unless
N3a selects another) at 4,096 bytes with position, distractor, and
distributed-evidence tests; only then consider 16k. Include the 24 examples +
definitions condition at 16k under a new pre-registration.
**Success (4k):** accuracy within 2 points of the native 1,024-byte arm on the
same rows, with correct evidence at early/middle/late positions and no
distractor collapse; memory and latency reported.
**Success (16k):** the same non-inferiority rule holds; if not, the 16k claim is
withdrawn.
**Cost:** 4k adaptation and tests; 16k is projected 12.6–15.9 GPU-hours before
the full-information arm, and 32k 39.8–56.2. These are projections, not results.
**Decision:** a negative 4k result stops the 16k spend; do not buy a longer
window to rescue a negative native-window result.

### N5 — Multimodal evidence, with an OCR-only control (GPU/data)

**Hypothesis:** visual evidence changes the answer on a task where OCR-only and
text-only controls fail.
**Action:** build a small visual-evidence suite (charts/documents/diagrams);
establish text-only and OCR-only baselines; train or adapt a visual path only
after the controls are measured.
**Success:** the visual model beats the OCR-only control by a pre-registered
margin on the visual-only subset, and does not merely reproduce OCR text.
**Cost:** data pipeline + a small adaptation; do not quote a number until run.
**Decision:** if OCR-only matches the visual model, do not claim multimodal
capability.

### N6 — Reliability and abstention on the locked suite (no architecture change)

**Hypothesis:** a calibrated reject option can hit a stated risk at a stated
coverage without a new encoder.
**Action:** fit calibration on dev only; freeze the threshold; evaluate on the
locked suite. Report risk–coverage, AURC, abstention rate, and coverage at
fixed risk.
**Success:** the pre-registered risk target is met on the locked suite with a
95% interval; otherwise the abstention claim is withdrawn.
**Cost:** inference + analysis.
**Decision:** no product “reliable” claim without this.

### N7 — Fresh confirmation and replication

**Hypothesis:** the effects that survive are larger than the 1.25-point
run-to-run floor.
**Action:** two additional seeds for the primary suite cells; one untouched
final evaluation after all method choices are frozen.
**Success:** the headline direction is stable across seeds; the final suite is
reported once, with no subsequent tuning.
**Cost:** ~35 GPU-hours for two extra suite passes (from `thesis/GAPS.md`),
plus the locked-suite run.
**Decision:** if the direction flips inside the noise floor, withdraw it.

---

## 8. Completion criteria

The mission is complete only when all of the following are true. These are
gates, not aspirations; a missing gate is reported as missing.

### 8.1 Product

- A local `decide` API accepts a context, an arbitrary list of alternatives, and
  an optional policy; it returns a choice, a score, and an explicit abstention
  when evidence is insufficient.
- Boolean, multi-class/routing, pairwise/ordinal, document-grounded, policy-
  conditioned, and tool/action families each have at least one working adapter
  and one measured task.
- The model never silently substitutes a fixed taxonomy for the user's
  alternatives; invalid/empty/duplicate alternatives are rejected explicitly.
- The wheel installs outside the repository and the documented commands work
  from a clean environment; weights are content-addressed and optional.
- Latency, throughput, and memory are reported for each family; no “local” claim
  without a measured runtime.

### 8.2 Long context

- At least one checkpoint is trained/evaluated at **16,384 bytes** (or an
  explicitly justified equivalent) and compared with the native 1,024-byte arm
  on identical rows.
- Quality is preserved across lengths, evidence positions (early/middle/late),
  distractors, and distributed evidence; a single long prompt is not enough.
- The 16k result is non-inferior under a pre-registered margin; if it is not,
  the long-context claim is withdrawn.
- Memory and latency at 16k are reported; “accepts more bytes” is not a result.

### 8.3 Multimodal

- A visual-evidence task exists where the answer changes with the image and
  where text-only and OCR-only controls are measured.
- The visual path beats the OCR-only control by a pre-registered margin on the
  visual-only subset.
- No multimodal claim is made from an OCR pipeline, a text-only model, or an
  architectural argument.

### 8.4 Research validity

- A benchmark registry covers all six families, with licences, splits, scoring,
  and contamination status; no family is represented by a single dataset.
- A compact dev suite and a locked evaluation suite are hashed before method
  selection; the locked suite is untouched until the final run.
- Every headline number has a stored per-example record, a checkpoint hash, a
  protocol hash, and a generator that recomputes it.
- One seed is labelled one seed; run-to-run variability is reported where
  practical; multiplicity is controlled or the analysis is labelled exploratory.
- Reproduced Jev comparisons are separated from published Jev numbers. No
  comparator number is invented.
- Negative results and lost artifacts stay in the record; no result is deleted
  because it is inconvenient.

### 8.5 Documents

- The thesis and paper state the same measured numbers as the ledger and the
  stored records; no stale status block, no “pending” that is actually measured,
  no “measured” that is actually projected.
- The thesis is Overleaf-compatible and builds with 0 errors and 0 undefined
  references/citations; the paper builds at its venue length.
- Every claim broader than the artifact is marked as a target, a hypothesis, or
  a limitation. The current “no long-context / no multimodal / no general
  decision surface” statements stay until replaced by measurements.

### 8.6 Mission-level

- A same-split, paired comparison with Jev exists on at least one task, and the
  claim is stated with its direction and uncertainty. A loss is reported as a
  loss.
- The product is general enough that Banking77 is one regression test among
  many, not the identity of the system.
- The final report can be rebuilt from committed scripts and hashes on a clean
  machine, without relying on a lost sandbox or a remembered number.

---

## 9. Evidence index

Paths are relative to the repository root. “Verified here” means this audit ran
the command or read the hash; “stored” means the artifact is present and its
record is hash-checked but the experiment was not re-executed.

| Evidence | Path | Status | Key fact |
|---|---|---|---|
| Large pretraining checkpoint | `results/recovered_20261028/checkpoints/seg_9_step_69255_aba65b60e306.pt` | verified hash | 1,361,732,233 bytes; `aba65b60…`; 1,134,673,920 tokens |
| Large run record | `results/recovered_20261028/results/pretrain/l1_large_byte_iso10v3/` | stored | step 69,255; complete |
| Nine-task suite | `results/suite_large_113m/` | manifest verified 92/92 | 9 cells; 3-rung comparison in `COMPARISON.md` |
| Jev protocol-matched | `results/jev_matched/` | stored | A0 88.38, A2 58.21, A1 22.44; verdict gap survives |
| Query head | `results/jev_query/` | stored | 89.03 primary; 59.12 old-head control; P0 missed |
| 5,120 arm | `results/jev_query_5120/` | stored | 87.11; 1.69 GPU-h; P0 missed |
| Shipped product path | `results/product_path_clean/summary.json` | verified hash | 2,743/3,080 = 89.06%; 9,968 bank |
| Original product path | `results/product_path/summary.json` | verified hash | 2,742/3,080 = 89.03%; 25/9,993 overlap |
| E4 width sweep | `results/e4_candidate_sweep/SUMMARY.json` | stored | 12 arms; 3/3 holds at 32 vs K; curve falls |
| XNLI context test | `results/xnli_context/RESULTS.md` | stored | 42.15 → 42.35 inference repair; not significant |
| Scale state | `results/scale_state/REPORT.md`; `results/large_v3/` | stored | lost run and recovery record |
| Laya baseline | `results/laya_baseline/laya_banking77_20rows.json` | stored | 0/20, invalid harness; do not cite as capability |
| MoE feasibility | `results/moe_feasibility/FEASIBILITY.md` | stored | no arm trained |
| Data manifest | `data/MANIFEST.json`; `DATA.md` | stored | 20 tasks; 2,659,054 rows; 5,142,706 docs |
| Corpus stream | `data/tokens/byte_corpus_v2.u16` | stored | 1,633,580,647 tokens; `9267a2ea…` |
| Tests | `tests/` | verified | 99 passed in 21.96 s |
| Product specification | `docs/PRODUCT_SPEC.md` | created 2026-10-30; updated with registry pointer | Decision contract, families, long-context and visual feasibility, release gates; SHA-256 `685fb1da…` (recomputed after the 2026-10-30 milestone update) |
| Capability targets | `research/CAPABILITY_TARGETS.json` | created 2026-10-30 | Machine-readable pre-training gates; JSON valid; SHA-256 `aa910a3a…` |
| Architecture decision record | `research/ARCHITECTURE.md` | created 2026-10-30 | Open architecture, incumbent baseline, candidate licences, contamination risk, no-training screening protocol; SHA-256 `b10273b9…` (recomputed after the 2026-10-30 context update) |
| Benchmark registry (human) | `research/BENCHMARK_REGISTRY.md` | created 2026-10-30 | Six families, 44 datasets, comparator register, licence table, conflicts, long-context/visual axes, B0–B7; SHA-256 `641fd364…` (recomputed after the 2026-10-30 context update) |
| Benchmark registry (machine) | `research/benchmark_registry.json` | created 2026-10-30 | 44 datasets, weights, comparator points, licence gates, blockers; JSON parses and the weight/required-field check passes; SHA-256 `1dcf5f09…` (recomputed after the 2026-10-30 context update) |
| Wheel | `dist/pkg/azurite_encoder-0.1.0-py3-none-any.whl` | verified hash | 249,460 bytes; `ebfb6e0e…` |
| Overleaf zip | `dist/overleaf/azurite-thesis-overleaf.zip` | verified hash | `e123565c…`; extracted build 84 pages |
| Thesis | `thesis/main.pdf` | verified | 84 pages; 0 errors/undefined; `d045f69a…` |
| Paper | `paper/azurite.pdf` | verified | 9 pages; 0 errors/undefined; `fb2e761f…` |

### 9.1 Quoted record hashes from this audit

```
product_path/summary.json                 8e0326de99b0031114800149a3521b14bd13ef9e40a4843f00f6111bf197a6f7
product_path_clean/summary.json           5f117ea1da460b4769dc2f0493c0c706f4cc45a7ffeb480dfecc3ce233dde0b6
jev_query/eval/summary.json               4b3476b41d5b5dbe6ba1990a1ce9e06b7823ae1cd4c5183aa4c26a6f09cb08a7
jev_query_5120/eval/summary.json          b96620f1efd5c35e6ea4136fb57aa6d63821dc139652abee2a1f45742b919426
large checkpoint                          aba65b60e30654a9895dc9c365d62b555f96dcce96ad8a8ad56cc718ee0865ec
wheel                                     ebfb6e0e78d2dc3a9aeb7ad5c3b3ff2824a57778e26f33ad670607a444a50e43
Overleaf zip                              e123565c60e41e06ed7002d866ddace071cd7c6773bfb7bb161f1175d01a0ae3
thesis PDF                                d045f69accdbdb1a4f59a2460419c4ce3cf0dd947a60f68d5e1cd975e0a57fec
paper PDF                                 fb2e761fb65386d1de6f5c40b9e3f10dbbea68901f8ffb4a645b32efba031a4d
```

---

## 10. Update protocol

1. **Append, do not rewrite.** New evidence goes in a dated subsection; a
   correction cites the old entry and the artifact that supersedes it.
2. **One owner per claim.** A number lives in its stored record first, then in
   this ledger, then in the thesis/paper. If those disagree, the record wins and
   the document is wrong.
3. **No claim without a path.** Every row in §2 and §9 must name a file or a
   command. A remembered number is not evidence.
4. **Negative results stay.** `_failed_wave*`, `DISCARDED_*`, the lost large run,
   the 0/20 Laya baseline, and the withdrawn U-curve remain in the tree and in
   this ledger.
5. **Banking77 is a regression test.** It may be cited as a measured comparator,
   but it cannot be the sole basis for a general-capability claim, architecture
   choice, or product roadmap.
6. **Re-audit before writing.** Re-run the tests, the manifest checks, the CLI
   info, and the document builds; never quote a build from an earlier commit
   after changing a source that feeds it.

---

## 11. Audit log

- **2026-10-30:** initial ledger. Read code, tests, data manifests, 39 GB of
  results, current thesis and paper. Ran 95 tests (pass), `python -m azurite
  info` (9,968 support; 89.06/89.03; default checkpoint not installed),
  `verify_manifest.py` on the 113M suite (92/92), `pdfinfo` (84/9 pages), and
  hash checks for the wheel, Overleaf zip, PDFs, summary records, and the
  1.36 GB checkpoint. Recorded the stale status claims and the checkpoint path
  drift. No result file was edited.
- **2026-10-30 (product definition):** created `docs/PRODUCT_SPEC.md` and
  `research/CAPABILITY_TARGETS.json`. The contract covers caller-supplied
  alternatives, boolean, classification/routing, pairwise, ordinal,
  document-grounded decisions, explicit abstention, and stops on insufficient
  evidence. The targets fix a functional, competitive, and stretch gate per
  family, with the current measured floors as regression bounds. Long context
  is specified at 8k/16k/32k bytes with position and distractor tests; the 16k
  gate is non-inferiority within 2 points of the 1,024-byte arm. Visual
  evidence is specified with an OCR-only control and a visual-only subset; an
  OCR match withdraws the multimodal claim. No training run was started. The
  JSON parses; hashes are recorded in §9.
- **2026-10-30 (architecture open):** created `research/ARCHITECTURE.md` and
  updated the product spec §6.2. Recorded the byte encoder as the incumbent
  baseline, not the default; admitted pretrained encoders as candidates subject
  to identity, licence, provenance, contamination, size/latency, and behaviour
  gates; rejected a new from-scratch run without a written justification; and
  fixed a frozen-feature screen (N3a) before any architecture selection. No
  candidate weights were downloaded or hashed; no encoder was trained; no
  result was changed.
- **2026-10-30 (benchmark registry):** created `research/BENCHMARK_REGISTRY.md`
  and `research/benchmark_registry.json`. Read official TypeSafe model/API/
  primitive/confidence/jaggedness pages and the AbdelStark, Bonn, Vals, AY, and
  Ravvich comparator material; checked dataset cards/repositories for Banking77,
  CLINC150, MASSIVE, SQuAD 2.0, MuSiQue, QASPER, FEVER, ContractNLI, CUAD, BFCL,
  τ²-bench, API-Bank, MS MARCO, TREC DL, HH-RLHF, UltraFeedback, ChartQA, DocVQA,
  SlideVQA, and related sources. The registry names six families and 44
  datasets (26 core); the JSON parses and the weight/required-field/unique-id/
  25%-cap checks pass. Recorded the Jev 1.13/1.14 distinction, the Jev Responses
  License 1.0 restrictions, the AG News/Emotion/Banking77/XNLI comparator
  conflicts, and the unresolved licences. No dataset was downloaded, no split
  manifest was hashed, no model or Jev API was run, no visual anchor was built,
  and no capability claim was added. O7 changed from **missing** to **partial**.

---

## 12. Product definition and pre-training targets (2026-10-30)

**Status:** definition complete; implementation not started. This section is a
dated addition, not a rewrite of the audit.

### 12.1 Artifacts

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `docs/PRODUCT_SPEC.md` | Product statement, decision contract, family semantics, long-context and visual feasibility, release gates; updated with the context-scaling result | `685fb1dac705ba9b64c73fe0efb625066537728d6121fe0dab93678acf73b892` |
| `research/CAPABILITY_TARGETS.json` | Machine-readable functional/competitive/stretch gates, current baselines, amendment policy, global evaluation gates | `aa910a3a7bb7d7f54d2a2268945dc6dd66a2ed39df847c8864efab809e2b6140` |
| `research/ARCHITECTURE.md` | Architecture decision record: open choices, incumbent baseline, candidate admission gate, contamination risks, no-training screening protocol | `b10273b9df6a35318908d1dd0a8748a616e0a4660f89cd10b47559328b7e1c04` |

### 12.2 The contract in one paragraph

`decide(context, alternatives, policy) -> {choice, score, abstain, evidence}`.
The caller supplies 1..K unique alternatives; the result is a supplied id or
`None` with `abstained=true`. Boolean is the K=2 case. Pairwise and ordinal are
distinct because ties and order are part of the result. Document-grounded calls
return byte-offset evidence spans; `evidence_required=True` without a span must
abstain. Empty/duplicate alternatives are rejected, candidate order must not
matter, and a softmax score is not called a probability unless a calibration
artifact is named. The fixed 77-way Banking77 path remains a regression test,
never a fallback.

### 12.3 Targets that now bind future work

- **Regression:** full Banking77 3,080 rows must stay at or above 88.56%; the
  current 89.06% is the reference.
- **Boolean:** functional 75% balanced accuracy, competitive 85%, adversarial
  negation subset 65%/75%.
- **Classification:** functional no cell below the current 113M floor by more
  than 1.0 point and mean at least 83.1%; competitive mean 86%, with XNLI at
  50%, AG News at 88%, and CLINC-OOS at 90%.
- **Pairwise:** functional 70%, competitive 85%, tie ECE at most 0.10.
- **Ordinal:** 3-level QWK functional 0.70 / competitive 0.80, against the
  measured 0.759; 5-level QWK 0.55/0.70.
- **Document-grounded:** functional F1 60, answerability 70%, citation IoU
  0.50; competitive F1 75, answerability 80%, citation IoU 0.70.
- **Abstention:** functional 90% selective accuracy at 70% coverage and 70%
  unanswerable recall; competitive 95% at 80% coverage and 85% recall.
- **Long context:** 8k within 3 points (functional), 16k within 2 points
  (competitive), 32k within 2 points (stretch); worst position gap 10/5/3
  points.
- **Visual:** visual-only accuracy 70/80/90% with a +10/+15/+20 point gain over
  OCR-only at functional/competitive/stretch.
- **Reliability:** ECE at most 0.08/0.05/0.03, with positive Brier skill and an
  AURC improvement over raw confidence.
- **Deployment:** GPU p50 at 16k at most 8 s/2 s/1 s; int8 artifact at most
  700/400/250 MB with a 2/1/0.5-point accuracy drop.

These are commitments, not results. The current suite means 81.12%, the
3-level QWK is 0.759, and the 5,120-byte arm is 87.11%; none of the targets above
has been met by a general model.

### 12.4 Long context, explicitly

Current native window: **1,024 bytes**. The only longer arm, **5,120 bytes**, is
**87.11%**, below the native 89.06% on the same rows. Jev's reconstructed median
prompt is **10,683 bytes**. The product gate is therefore not "accept 16k"; it
is "16,384 bytes of packed evidence with accuracy within 2 points of 1,024, no
position gap above 5 points, and no collapse under 8 distractors." Two paths are
admissible: full-window RoPE/length adaptation (projected 12.6–15.9 GPU-hours at
16k, 39.8–56.2 at 32k) and a chunked hierarchical encoder (linear cost, but a
separate hypothesis). The stop rule is 4k -> 16k -> 32k; a negative 4k result
stops the 16k spend. The 4,096-byte fixed-checkpoint arm has now been measured
and is negative (§14.3), so the 16k spend is stopped until a length-adapted 4k
arm passes. Accepting more bytes is not a result.

### 12.5 Visual evidence, explicitly

The minimum task is a **visual-only subset** where the answer changes with the
image and OCR cannot recover it: checkbox marks, chart trends, layout-dependent
fields. The model path is a frozen, open-licence image encoder plus a trained
adapter. OCR-only, text-only, blank-image, and shuffled-image controls are
mandatory. Functional target: 70% visual-only accuracy and at least +10 points
over OCR-only; competitive 80% and +15. If OCR-only is within 5 points, the
multimodal claim is withdrawn regardless of absolute score. Public datasets
(DocVQA, ChartQA, VisualMRC) are not admitted until licence, image provenance,
and contamination gates pass; a locally rendered set is the first instrument.

### 12.6 What this does not establish

- It does not implement `decide`; that is N1.
- It does not freeze the benchmark registry. The registry is drafted in
  `research/BENCHMARK_REGISTRY.md` and `research/benchmark_registry.json`; the
  **suites** are now hashed and sealed (§14), but the full six-family registry
  is not frozen because three families and several dataset licences are unresolved.
- It does not select the encoder, tokenizer, pooling, retrieval, or head; that
  protocol is `research/ARCHITECTURE.md`, and the screen is N3a.
- It does not measure the current model on the new families; that is N3.
- It does not start long-context or visual training. Those remain N4/N5 and are
  gated on the earlier results.
- It does not change any thesis or paper claim. The thesis still correctly says
  no long-context and no multimodal result exists.

---

## 13. Benchmark registry (2026-10-30)

**Historical note (2026-10-30):** this section records registry v0.1. Its suite
state is superseded by §14: the dev and locked manifests now exist, BoolQ’s
train/validation overlap was found and excluded, and the context-scaling run is
complete. The registry content (families, licences, comparator conflicts) still
stands; the hashes in §13.1 are v0.1 and are not current.

**Status at time of writing:** draft v0.1; content complete enough to review;
**not frozen**; no dataset downloaded for the new families, no split manifest
hashed, no benchmark run. This section is a dated addition, not a rewrite of the
audit.

### 13.1 Artifacts

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `research/BENCHMARK_REGISTRY.md` | Human-readable six-family registry, comparator register, licence table, conflicts, long-context/visual axes, blockers, B0–B7 | `641fd3641b15092d84ca47d49018f77f6e0288d3a0e67436eee5f57ef4219b08` |
| `research/benchmark_registry.json` | Machine-readable companion; families, 44 datasets, weights, comparator points, licence gates, blockers, next experiments | `1dcf5f0908bf9a3b71675f883aaf5d13c524886bbffb44fb7c2f2b424cca8618` |

### 13.2 What the registry establishes

- **Six families, not one.** Classification/routing; document-grounded decisions
  and evidence sufficiency; tool/action selection and abstention; policy and
  rule-conditioned decisions; relevance/preference/ordinal; multimodal visual.
- **44 datasets, 26 core.** The four-dataset core families use 25% per dataset;
  the relevance/preference/ordinal family has three subfamilies at 33.3333%,
  with two datasets at 16.66665% each. No dataset exceeds 25% of its family.
- **Comparator provenance.** Eight evidence sources, two pinned Jev versions,
  twenty cited points, and a rule that published points are not paired runs.
  Conflicts for AG News, DAIR Emotion, Banking77, and XNLI are recorded, not
  averaged away.
- **Licence gates.** Clear-commercial, research-only, unresolved, restricted
  outputs, and gated datasets are separated. Unresolved means research-only until
  resolved; it cannot enter a shipped product.
- **Long-context and visual axes.** Context bands, position/distractor tests,
  truncation audits, and mandatory OCR-only/blank-image/shuffled-image controls
  are specified. The visual family is explicitly blocked until its anchors are
  built and public terms are resolved.
- **Structural validation.** The JSON parses; six families, 44 unique dataset
  ids, required fields, core weights, and the 25% cap pass a local check.

### 13.3 What is verified, and what is not

**Verified here:** the registry files exist and parse; the JSON structure,
unique ids, family coverage, core weights, and 25% cap pass; official source
URLs and quoted licence/row-count facts were read from the dataset/model owner
pages; the Jev model IDs, price, context limits, primitive shapes, confidence
formula, and 1.13 jaggedness list were read from TypeSafe docs; the Bonn scope
(37 datasets, 346,009 requests, `jev-1.13.0`) and the Jev Responses License 1.0
were read from the Bonn repository.

**Not verified / not done:** no locked split manifest; no new dataset download;
no contamination scan for the new datasets; no live Jev run; no 1.14 independent
result; no visual anchor; no document/tool/policy/ordinal implementation; no
licence resolution for SST-5, DocVQA, ChartQA, VisualMRC, SlideVQA,
InfographicsVQA, PolicyQA/OPP-115, or the LegalBench subset. Several split
counts remain `reported` rather than re-counted, and FEVER/QASPER/BFCL/τ²/CUAD/
PolicyQA/LegalBench split decisions still need pinning.

### 13.4 Comparator finding

Public Jev evidence is **1.13-only** in the sources reviewed. The Bonn study
pins 1.13.0 and releases raw responses under a research/evaluation licence that
prohibits distillation, imitation training, competing-product use, and
redistribution; Vals and AbdelStark also pin 1.13.0. No independent 1.14 result
was found. The current product is 1.14, so a 1.14 comparison requires a new
pinned run, not a relabelled 1.13 number.

The public points are not a leaderboard: AG News is 0.885 on 7,600 rows and
0.910 on a 100-row pilot; Banking77 is 0.797 on 3,076 rows, 0.788 on 231 rows,
and 0.870 on a 72-label task; XNLI's 0.860 has no clean split provenance. The
only same-row local comparator remains the Banking77 reconstruction at 92.40%
(2,846/3,080), which is **not** a live Jev API run.

### 13.5 Why O7 remains partial

**Updated 2026-10-30.** The completion gate is a hashed split manifest plus a
locked suite that is not opened during development. Both now exist (§14): the
dev suite is frozen and the 12-task locked suite is sealed. O7 remains partial
because tool/action, relevance/ordinal, and visual families are absent and five
locked tasks are research-only. B1 (resolve licences) remains; B3 (paired Jev
run) needs explicit budget/API approval. No architecture selection, training
run, or capability claim is authorised by the registry.

### 13.6 Registry-specific blockers

1. No live Jev run (no API key/spend authorization).
2. Public comparator coverage is classification-heavy; most mission-critical
   families have no published Jev point.
3. ~~No locked split manifest.~~ **Resolved 2026-10-30 (§14.1); the locked suite is sealed, not opened.**
4. No document/tool/policy/visual implementation.
5. Unresolved licences for SST-5, DocVQA, ChartQA, VisualMRC, SlideVQA,
   InfographicsVQA, PolicyQA/OPP-115, and the LegalBench subset.
6. No positive long-context result. The 4,096-byte curve is measured and negative (§14.3); 16k/32k are projections.
7. No visual encoder or image path.
8. No contamination audit for the new datasets.
9. Split ambiguity in FEVER, QASPER, BFCL, τ², CUAD, PolicyQA, and LegalBench.
10. Comparator conflicts for AG News, DAIR Emotion, Banking77, and XNLI.
11. Jev's 255-option limit requires hierarchical decomposition for larger label
    sets; that decomposition changes the task.
12. Jev 1.13 vs 1.14: public evidence is 1.13; 1.14 has no independent result.
13. Image licence/storage cost for DocVQA, ChartQA, VisualMRC, SlideVQA.
14. The local Banking77 mirror is 14 rows short; do not merge 3,076 and 3,080.

---

## 14. Suite freeze and context scaling (2026-10-30)

**Status:** the dev and locked suites are materialised; the context-scaling
experiment is complete. The result is **negative**: no task is context-additive,
and the one learned long-document head degrades severely as the window grows.
This section is a dated addition, not a rewrite of §13.

### 14.1 Suites

| suite | tasks | weights | rows | status |
|---|---:|---|---:|---|
| dev `azurite-context-dev-v1` | 5 | 20% each | 2,499–2,500 train / 748–750 test | frozen; built from training splits only |
| locked `azurite-locked-v1` | 12 | 8.33% each | 493–500 each | sealed 2026-10-30; not opened |

- Dev tasks: Banking77, CLINC150, BoolQ, SQuAD 2.0 answerability, ContractNLI.
- Locked tasks: the nine existing local tasks plus BoolQ, SQuAD 2.0
  answerability, and ContractNLI.
- No dataset exceeds 25% of either suite. Dev train/test document overlap is
  **0**. Dev/locked overlap is **0** after excluding 45 BoolQ passages that the
  official train and validation splits share.
- **Split roles.** Train rows (2,499–2,500) fit the model; dev rows (748–750)
  make architecture and protocol choices; the locked suite is the final test.
  There is **no calibration partition** in v1. No threshold or abstention policy
  was fitted in the context experiment, so its argmax results consume no
  calibration split. Fitting a threshold on the inspected dev test or on locked
  requires a new suite version; it is not permitted on v1.
- BoolQ’s overlap is the kind of leak a nominal split misses. The exclusion is
  recorded, not hidden.
- Locked limitations: tool/action, relevance/preference/ordinal, and multimodal
  visual families are absent. Five tasks are research-only because their
  licences are unresolved. Nine tasks were inspected by earlier work, so this is
  a procedural lock, not an untouched-data claim.
- Artifacts: `research/suites/dev_suite.json`, `locked_suite.json`,
  `dev_items.jsonl`, `locked_items.jsonl`, `README.md`; verifier
  `scripts/verify_suites.py` passes weights, counts, digests, file hashes, and
  disjointness.

### 14.2 Context-scaling design

- Checkpoint: 14.26M encoder from
  `results/pretrain/l1_small_byte_ctx256/ckpt.pt`, SHA-256
  `8fd8bedee794ee4aa464af7c11fb06eb1d8dcfbfe82f529f8106a11a798ff131`;
  pretrained at 256 positions.
- One head per dev task, trained at 256 bytes for 500 steps (pre-registration
  amendment 1), batch 16, LR 1e-4, seed 0. The 1,000-step schedule was stopped
  at 166/1,000 for CPU cost; the amendment was written before any result file
  existed.
- Longer lengths evaluate the **same weights**; only the packing window changes.
  Primary lengths 256/1024/4096, secondary 512/2048.
- Pre-registration: `results/context_scaling/PRE_REGISTRATION.md`, SHA-256
  `cd18c84ccc8800a82ac474220d93b3074204becd4114807351c83d8316f152c7`.
  Amendment 1: SHA-256
  `1f3af729fac4be4f9f69ebe243ff5a5edd60a0ddd25b356ba673e78de9432f1f`.
- One seed, dev only, no Jev row. This is not a general-capability result.

### 14.3 Result

| task | family | 256 | 1024 | 4096 | majority | reading |
|---|---|---:|---:|---:|---:|---|
| Banking77 | classification | 77.33 | 77.47 | 77.47 | 2.53 | flat control; all context fits at 256 |
| CLINC150 | classification | 79.33 | 79.33 | 79.33 | 1.60 | flat control |
| BoolQ | document-grounded | 62.40 | 62.27 | 62.27 | 62.13 | majority-collapse; uninformative |
| SQuAD 2.0 answerability | document + abstention | 64.13 | 65.60 | 65.60 | 66.13 | +1.47 pp, n.s.; still at majority |
| ContractNLI | policy/rule | **69.79** | **54.28** | **37.03** | 50.53 | **degraded**; −32.75 pp at 4,096 |

- **0 context-additive cells; 4 degraded cells.** All four degraded cells are
  ContractNLI (512/1024/2048/4096). Exact McNemar at 4,096: 336 old-correct /
  91 new-correct, p ≈ 4.4e-34; paired bootstrap 95% lower bound −37.70 pp.
- The five-task mean falls from 70.60% to 64.34%. The family macro is the
  primary table; the five-task mean is descriptive.
- ContractNLI is a **partial-context** row at every length: median dev document
  9,817 bytes, max 41,093; only 9.1% of test contexts fit at 4,096. The fall is
  therefore not a full-document result. It is a result about extending a
  256-trained model’s window.
- The lexical control (TF-IDF + class-balanced logistic regression) reaches
  64.71–66.71% on ContractNLI, 14–16 pp above majority, so the task is learnable
  and the neural degradation is not a task artefact. BoolQ and SQuAD 2.0 are at
  or below majority for the lexical model too; their neural curves are
  uninterpretable as context evidence.
- The result is consistent with the earlier 5,120-byte Banking77 arm: longer
  windows did not help and the joint fine-tune was 1.92 pp below the native
  query head.

### 14.4 Comparator distinction

`research/benchmark_registry.json` now annotates every dataset-attached
comparator point:

| class | count | meaning |
|---|---:|---|
| `published_reference` | 13 | vendor/third-party number, not rerun here |
| `local_reconstruction` | 1 | Banking77 0.924 from a stored per-example Jev vector; not a live call |
| `reproduced_live` | **0** | same rows, pinned version, frozen request, live response |

There is **no reproduced live Jev comparison** in this checkout. `RESULTS.md`
for the context experiment contains no Jev number at all; the published points
in the registry are references, not rows from this run.

### 14.5 Artifact hashes

| artifact | SHA-256 |
|---|---|
| `research/suites/dev_suite.json` | `43737b16b3ec85d3ffcdc264932c33499bc00417779a14eb8a526b4d49797630` |
| `research/suites/locked_suite.json` | `1f2115ffab5ee0ca1959d534b424d99db1e3f286dce6d11e534b92fb058c87b2` |
| `research/suites/dev_items.jsonl` | `9c29c0938b0f5c9797a7e6d2a3f7bcca43aca2e8d15b06e4481e3ad1c3511cd6` |
| `research/suites/locked_items.jsonl` | `59abbd685c4de345344363a4442fb2ad04dc9786f76d765deb8c0c106c1b7ddb` |
| `research/suites/README.md` | `cb80588118ae7cc79fd8ad209408d56cb3a79c6b2668edab5f30a19c16c1ce2c` |
| `results/context_scaling/RESULTS.md` | `113a14db6543939b29f5c6658d4313e895866e0d01b7a23922298b934d3e2704` |
| `results/context_scaling/CONTEXT_CURVE.json` | `4fcef82e61d536896e60a572578ac0c1d87ea0d6c839fb6ccc43e331d934f2c3` |

`results/context_scaling/MANIFEST.sha256` verifies every top-level file in that
directory (including both pre-registration files and all five raw run records).

### 14.6 What this establishes and what it does not

**Establishes:** the suite is materialised and document-disjoint; no single
dataset exceeds 25%; the 14.4M model does not use a longer window on the tested
tasks; ContractNLI degrades monotonically as the window grows; the locked suite
is sealed.

**Does not establish:** that a model trained at 4,096 would fail; that a
full-document ContractNLI model would fail; that visual, tool, or ordinal
families behave similarly; that the result generalises beyond one seed or the
14.4M rung. It is not a Jev comparison and not a product capability claim.

### 14.7 Next experiments

**Priority note (2026-10-30):** §15 superseded this list as the immediate plan.
The context experiments remain valid. H1 has now reported negative (§18), so
no training screen is currently authorised; length adaptation stays deferred
until a new falsifiable mechanism is approved.

1. **Length-adapted ContractNLI arm.** Train at 4,096 (same rows, same budget)
   and compare to the 256-trained curve. Only this separates “the window is
   unused” from “the weights were never trained for it.”
2. **Class-balanced heads for BoolQ/SQuAD.** The current heads are at majority;
   a balanced loss or balanced sampler is needed before their curves mean
   anything. Record it as a protocol amendment, not a silent rerun.
3. **B1 licences.** Resolve SST-5, DocVQA, ChartQA, VisualMRC, SlideVQA,
   InfographicsVQA, PolicyQA/OPP-115, and the LegalBench subset before any
   product-admissible claim.
4. **B3 live Jev.** Needs explicit API/spend authorization. Do not call a
   published point a reproduced run.

---

## 15. Pivot review after three failures (2026-10-30)

**Decision:** stop the scale/context direction on the legacy fixed-label byte
path. The pre-registration text is
[`research/PIVOT_REVIEW.md`](PIVOT_REVIEW.md), SHA-256
`4418ad345e53cf6b239fbaac97b40c9e4bfb9f99cd4d41cda60ffa93e53ea326`. The file
now also carries a clearly marked post-run outcome (§9); its current SHA-256 is
`773ce0312fbb03c57f93a9f48d68ac5ea7ac38ae480b2d3e23441d0599dfa557`, and the
original hash above remains the pin for the pre-registered text.

### 15.1 Why

Three substantial failures share one bottleneck:

1. **Capacity.** 113.47M is 7.95x the 14.4M rung but wins 4 of 9 cells, loses
   XNLI by 9.95 pp, and leaves 4 unmoved; XNLI is a uniform predictor.
2. **Banking77 head/context packing.** Native query head 89.03% misses the
   92.40% bar; the 5,120-byte joint fine-tune is 87.11%, 1.92 pp below native.
3. **Untrained window.** The 256→4,096 fixed-checkpoint curve has 0
   context-additive cells and 4 degraded cells; ContractNLI falls 69.79% →
   37.03%.

The fixed-label head cannot test the product contract (arbitrary alternatives,
abstention, evidence). Another Banking77-only head is not progress.

### 15.2 What was authorised

Only H1 in `PIVOT_REVIEW.md`: a candidate-conditioned cross-encoder screen with
majority, TF-IDF, byte-cross-encoder, MiniLM, and ModernBERT arms on the five
dev tasks. Success requires ≥3.0 pp macro over both TF-IDF and the byte
cross-encoder, a paired 95% lower bound above zero, ContractNLI ≥10 pp above
majority, SQuAD answerability ≥5 pp above majority, and no task >2 pp below the
byte arm. Banking77 is one control; a Banking77-only gain fails H1.

**Budget:** 4–8 projected GPU-hours for Stage 1 (one seed), 6–10 for Stage 2
(three seeds), **18 GPU-hour cap**. Stage 1 ran on 2026-10-30 in 5 min 36 s and
failed three gates (§18); Stage 2 was not run. Any result is a development
screen, not a capability claim; the locked suite stays sealed and calibration
requires a suite v2. No further training screen is authorised until a new
hypothesis is written and accepted.

### 15.3 Stop rules

- No more parameters on the legacy suite.
- No more Banking77-only heads or 5,120-byte packing variants.
- No 16k/32k training until a length-adapted 4k arm passes ContractNLI.
- No calibration threshold without a disjoint calibration partition.
- No locked-suite access for development or diagnosis.
- No best-of-seeds selection; report the pre-registered seed set.

---

## 16. Capability milestones and measurement axes (2026-10-30)

**Decision:** long context and visual evidence now have explicit gate ladders,
budgets, and stop rules instead of open-ended aspirations. The record is
[`research/CAPABILITY_MILESTONES.md`](CAPABILITY_MILESTONES.md), SHA-256
`100c389f09eb878e7f00a227ba7ed034007b9da2965d2ec4726d827b59e5e1e0`.

### 16.1 Long context

**Definition:** preserved decision quality across lengths, evidence positions,
distractors, and distributed evidence — not merely accepting more bytes.

- **LC0 (done, negative):** fixed-checkpoint 256→4,096 curve; 0 additive, 4
  degraded; ContractNLI 69.79→37.03.
- **LC1:** length-adapted 4k on rows whose full document fits; pass if within
  2.0 pp of the 256 baseline, position gap ≤5 pp, ≤2 pp drop with 8 distractors,
  memory and latency reported. 2–4 projected GPU-h; target 2026-11-15.
- **LC2:** 16k conditional on LC1; within 2.0 pp of 1,024 on the same rows, no
  position/distractor collapse, memory/latency reported. 12.6–15.9 projected
  GPU-h; target 2026-12-15.
- **LC3:** 32k conditional on LC2; 39.8–56.2 projected GPU-h for full-window.
- **LC4:** three-seed confirmatory claim on suite v2; locked opened once.

A negative LC1 stops LC2. A negative LC2 stops LC3. Partial-context rows cannot
establish a full-document claim.

### 16.2 Visual evidence

**Definition:** the decision changes when the image changes and cannot be
reproduced from OCR text alone. OCR plus a text model is a text pipeline.

- **V0:** at least 200 locally rendered development items with gold answer,
  evidence box, OCR transcript, and template/document-disjoint splits; target
  2026-11-15.
- **V1:** OCR-only and text-only baselines on the same rows; visual-only subset
  identified; no model claim.
- **V2:** frozen open-licence image encoder plus adapter; ≥70% visual-only
  accuracy and ≥+10 pp over OCR-only; citation IoU ≥0.50; memory and latency
  reported. 4–8 projected GPU-h; target 2026-12-15.
- **V3:** ≥80% and ≥+15 pp, citation IoU ≥0.65, three seeds; public benchmarks
  only after licence/contamination gates. 8–12 projected GPU-h; target
  2027-01-31.
- **V4:** confirmatory locked evaluation, opened once.

If OCR-only is within 5 points of the visual arm, the visual claim is withdrawn
regardless of absolute accuracy.

### 16.3 Measurement axes

Every substantial result reports quality, calibration (ECE/Brier/reliability),
abstention risk–coverage and unanswerable recall, robustness (positions,
distractors, distributed evidence, negation/paraphrase, blank/shuffled images),
latency p50/p95, throughput at batch 1/8, and peak RSS/VRAM plus artifact size.
“Not measured” is a reported limitation, not a blank cell.

**Near-perfect accuracy is a distribution-specific ambition, never a guarantee.**
Any 97% target must name its distribution, coverage, confidence interval, and
failure slices. No release note may say “near-perfect” without those qualifiers.

### 16.4 Sequencing

H1 in §15 is the only currently authorised training screen (18 GPU-hour cap).
LC1 and V0/V1 are next; LC2 and V2 are conditional and cannot be postponed
without a dated blocker entry. A milestone pass is permission for the next gate,
not a product claim.

---

## 17. Execution blocker — Molab session unreachable (2026-10-30)

**Status:** resolved by §18. The session below was replaced by a fresh
sandbox; the failure is kept because it explains the missing-run interval.

`.stage/molab_session.json` points at
`https://sb-ec84d577adb31af1.sb.molab.run` (session captured 2026-10-29
20:08). A remote preflight on 2026-10-30 returned:

```
Failed to connect to marimo server at https://sb-ec84d577adb31af1.sb.molab.run
```

The session is recycled or expired. The token is not reused, and no run is
fabricated in its place. A fresh Molab iframe URL/token with an open notebook is
required before any remote execution. If Molab cannot be restored, the
alternatives are an explicitly authorised local CPU run or another GPU provider;
both need a budget decision before use.

**What is ready:** H1 is pre-registered in `PIVOT_REVIEW.md`; the dev suite is
hashed; the candidate interface and task definitions already exist in
`azurite/decisions.py`; the 18 GPU-hour cap and stop rules are fixed. What is
missing is a live execution target, not a hypothesis.

---

## 18. H1 Stage 1 execution (fresh Molab session, 2026-10-30)

**Session:** a fresh Molab sandbox was supplied after §17. Preflight reported
Python 3.13.11, `torch 2.11.0+cu130`, CUDA available, NVIDIA RTX PRO 6000
Blackwell (97,887 MiB), `transformers 5.14.1`, `datasets 5.0.0`,
`sentence-transformers 5.6.0`, and 3 MiB of VRAM already used. The old
`/marimo/azurite_ws` directory survived but contained no dev data and no
checkpoint; the workspace was re-staged over it rather than treated as intact.

**Staged and verified in-sandbox:**

- `scripts/h1_screen.py` (29,553 B), all `azurite/` code, and the five dev
  tasks' `train`/`test` splits.
- Byte checkpoint `results/pretrain/l1_small_byte_ctx256/ckpt.pt`,
  SHA-256 `8fd8bedee794ee4aa464af7c11fb06eb1d8dcfbfe82f529f8106a11a798ff131`,
  matching the pre-registration.

**Protocol actually executed** (`results/h1_stage1/launch.json`):

- five tasks (`banking77_ctx`, `clinc_oos_ctx`, `boolq_ctx`,
  `squad2_answerability_ctx`, `contract_nli_ctx`), 2,500 train / 750 test
  rows (ContractNLI 2,499 / 748), K=8, seed 0, 3 epochs, batch 16, lr 2e-5;
- 256-byte cap for every arm; byte arm at 256 positions; pretrained arms at
  256 native tokens;
- prefix order **instruction, question, context**, identical for every arm.
  This order was chosen before launching because the pre-registered 256-byte
  cap would otherwise truncate ContractNLI's 112-byte policy behind its
  5–9 KB contract, making the policy row unanswerable by construction;
- arms: train-label prior, TF-IDF (1–2 gram) + logistic regression, frozen
  MiniLM pair embeddings + logistic regression, ModernBERT-base
  cross-encoder, 14.4M byte cross-encoder.

**Status:** Stage 1 ran to completion at 2026-10-30T20:14:10Z, 5 min 36 s
from launch, well inside the 18 GPU-hour cap. A 20-row smoke of all five arms
completed before launch; TF-IDF was the only arm that needed more than one
class in its 8-row smoke, and the full run has 77 Banking77 classes.

**Result: NEGATIVE.** Full table and records in
[`results/h1_stage1/RESULTS.md`](../results/h1_stage1/RESULTS.md).

| task | prior | TF-IDF | MiniLM | ModernBERT | byte |
|---|---:|---:|---:|---:|---:|
| Banking77 | 0.1653 | 0.9320 | 0.8733 | **0.9333** | 0.4960 |
| CLINC150 | 0.1160 | 0.9640 | 0.9173 | **0.9813** | 0.4160 |
| BoolQ | 0.6213 | 0.5827 | 0.6213 | **0.6533** | 0.6213 |
| SQuAD 2.0 answerability | **0.6613** | 0.6547 | **0.6613** | 0.6600 | **0.6613** |
| ContractNLI | 0.5053 | 0.6390 | 0.6524 | 0.6471 | **0.6751** |
| **macro** | 0.4139 | 0.7545 | 0.7451 | **0.7750** | 0.5740 |

ModernBERT clears the byte arm by +20.11 pp (paired bootstrap 95% CI 18.64 to
21.65), but fails three pre-registered gates:

1. it beats TF-IDF by **2.05 pp**, not 3.0 pp;
2. SQuAD answerability is at the majority class (**−0.13 pp**, not +5 pp);
3. ContractNLI is **2.81 pp below** the byte arm, outside the −2.0 pp floor.

ContractNLI is +14.17 pp over its majority, so gate 3 passes. Gate 6 (Stage 2)
was not run.

**Decision:** H1 fails. Per the stop rule, the encoder-replacement direction
is closed: no Stage 2, no second encoder sweep, no seed selection. The locked
suite stays sealed. The next proposal must be a different falsifiable
mechanism with its own baselines and gates. The 256-byte cap leaves both
document-grounded rows floor-limited; that is a limitation to diagnose in a
separate, clearly labelled experiment, not a reason to relax this gate after
the fact. This screen contains no Jev measurement and says nothing about
whether Azurite beats Jev.
