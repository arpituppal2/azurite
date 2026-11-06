# What this thesis still lacks, and in what order to close it

Written 2026-10-29, after the fourth rung (113.47M), the candidate-set sweep
(E4) and the protocol-matched Banking77 test were measured and integrated.
This is the deterministic target for the next passes. Every gap names the artifact
that would close it and, where the cost is measured, the GPU-hours it would take.
Nothing here is aspirational language: if a number is not in `results/`, it is
listed as missing.

**Update 2026-10-30.** A compact dev suite and a sealed locked suite now exist,
and a fixed-checkpoint context-scaling experiment has been run. Its result is
negative and is **not yet integrated into the thesis**: no task is
context-additive and ContractNLI falls 69.79% → 37.03% from 256 to 4,096 bytes.
See §2b and `results/context_scaling/RESULTS.md`.

## 0. State of the build this list is read against

- 84 pages, ~21,500 words of section source, 93 bibliography entries, all cited.
  0 LaTeX errors, 0 undefined references, 0 undefined citations, 0 overfull boxes
  in the last build. The front matter is built (title page, degree page, abstract,
  acknowledgments, contents, list of tables, list of figures) and the list of
  figures carries four generated figures.
- 67 arXiv identifiers and 11 Crossref DOIs were resolved against the arXiv and
  Crossref APIs; the twelfth DOI is the arXiv DOI for `bonn2026jev`, which
  Crossref does not index and the arXiv API does.
- The fourth rung is integrated: `tables/suite3.tex` (Table 4.5), `tables/suite3_ci.tex`
  (Table 4.6: every cell with its Wilson interval and paired test), the scale
  paragraph in §4.7, the abstract, the contributions list, the threats chapter, the
  conclusion, and Appendix A's checkpoint-discipline section. All of its numbers
  are macros emitted by `thesis/gen_suite_three.py` from
  `results/suite_large_113m/COMPARISON.json`; the generator re-hashes every per-cell
  `result.json`, recomputes each accuracy from its own predictions, and asserts the
  source-checkpoint digest and the pre-registration digest before writing a table
  cell.
- E4 is integrated as §4.10 and Table 4.9: twelve trained width arms at 113.47M,
  every per-example record stored and hash-verified, and every number in prose
  emitted by `thesis/gen_e4.py` from `results/e4_candidate_sweep/SUMMARY.json`.
  The same generator writes the pre-registration digest into the section; the
  manifest covers the records.
- The protocol-matched Banking77 test is integrated as §5.4.2 and Table 5.3:
  all 3,080 official rows, the exact 24 retrieved examples, the existing
  113.47M full-bank head, a 5,120-byte window and a $-2.0$-point
  non-inferiority rule, all fixed in a hashed pre-registration (`ee1b5bb5…`)
  before any forward pass. The primary arm scores 22.44% (691/3,080), the
  zero-example arm 88.38% (2,722/3,080), and the pre-registered verdict is
  "the gap survives". Every number in the section is emitted by
  `thesis/gen_jev_matched.py` from `results/jev_matched/summary.json`; the
  generator re-hashes every per-example file and recomputes every arm and
  Wilson interval before writing a cell. The evaluator-only amendment is
  `73f3fdc4…`; A0 and A1 are byte-identical under both evaluators.
- The query-anchored head is integrated as §5.4.3 and Table 5.4:
  `results/jev_query/` runs one pre-registered head (`bde97e74…`) on the same
  3,080 rows and the same 24 examples. It scores 89.03% (2,742/3,080; Wilson
  87.87--90.08) against Jev's 92.40%, exact McNemar p=3.99e-9, bootstrap lower
  -4.32 points; the pre-registered verdict is "P0 missed". The old head on the
  identical contexts scores 59.12%, so the query pool is worth +29.90 points
  (p=3.6e-212), confirming mean-pool dilution. Every number is emitted by
  `thesis/gen_query.py` from `results/jev_query/eval/summary.json`; the
  generator re-hashes every per-example file and recomputes every arm and
  Wilson interval. Two evaluator-only amendments (`3bd978b0…` dispatch,
  `a3cb2838…` manifest path) change no protocol.
- The 5,120-byte arm is integrated as §5.4.4: `results/jev_query_5120/` runs one
  pre-registered joint fine-tune (amendment 1, `7993177d…`) on the same 3,080
  rows, all 24 examples on every row, and scores 87.11% (2,683/3,080; Wilson
  85.88--88.25) against Jev's 92.40%, exact McNemar p=5.0e-18, bootstrap lower
  -6.30. The pre-registered verdict is "P0 missed"; it is also 1.92 points below
  the native-window head and 1.27 below the message-only control. Every number
  is emitted by `thesis/gen_query_5120.py` from `results/jev_query_5120/eval/
  summary.json`; the generator re-hashes every per-example file, the init
  checkpoint and the head record. The arm cost 1.69 GPU-hours.
- **It is not yet a submittable MIT PhD thesis.** The measured record is narrow and
  openly so. Chapter 2 is now a related-work chapter at committee length
  (3,020 words, 92 cited works) and the bibliography has been verified; the
  remaining dominant gap is a methods chapter detailed enough to rebuild the
  instrument. The central experiment (E4) has a measured answer at one rung; its
  remaining variants are costed below. The rest of this file is the list.

## 1. The candidate-set sweep (E4) is run; its variants remain

The thesis's own falsification criterion (§1.7) is accuracy as the candidate set
widens, at fixed context. E4 is now stored under `results/e4_candidate_sweep/`:
twelve arms (three wide-bank tasks, $|C| \in \{2,8,32,K\}$), one fresh head per
arm trained from the 113.47M checkpoint at the suite's frozen recipe, every
per-example record in `per_example.jsonl` with the gold list verified against its
recorded hash, `SUMMARY.json` recomputed by `scripts/e4_summary.py`, and
`MANIFEST.sha256` over every artifact. The pre-registration is
`762739463e8b…` and was hashed before any valid accuracy existed; both
amendments are inside the hashed text.

**Result.** Under the pre-registered five-point non-inferiority rule the claim
holds on 3/3 tasks: 32 candidates score +2.95, +4.50 and +0.95 points against the
full bank (paired lower bounds +1.95, +3.45, −0.20). But the curve falls with
width: 2 candidates reach 98.05/98.00/96.70 and the full bank 89.15/87.70/81.90,
so widening from 2 to 32 costs 5.95/5.80/13.85 points. The registered claim
survives because 32 is at or above the full bank (decisively on Banking77 and
CLINC150, within noise on MASSIVE intent), not because width is free; the
stronger reading that accuracy is flat as the set widens is not supported. Heads
trained at width 2 re-scored on the full bank fall to 70.25/64.80/69.25, so the
arms are width-specific and not interchangeable.

**Harness check.** The width-$K$ arms score 89.15/87.70/81.90 against published
88.45/88.50/82.00; |Δ| = 0.70/0.80/0.10 points. Two exceed the 0.2-point
tolerance set in the pre-registration and are reported as discrepancies; the
largest is inside the 1.25-point suite replication floor. No published value was
substituted.

**Remaining variants, at the measured 0.78 h per arm wall time:**

- second seed at 113.47M: 12 arms, ~9.3 h;
- the same twelve-arm grid at 33.7M and at 14.4M: ~9.3 h each, plus staging the
  checkpoint on the box (an infrastructure task, not compute);
- $|C|=64$ cannot be posed by these banks ($K=77$, $151$, $60$). If a 64-point is
  required, add widths 48 and 64 on Banking77 and CLINC150: 4 arms, ~3.1 h;
- the six narrow-bank tasks (XNLI 3, AG News 4, DAIR Emotion 6, Financial
  PhraseBank 3, MASSIVE scenario 18, Language ID 20) cannot pose $|C|=32$ and are
  excluded structurally, not by result.

One seed plus one extra rung is ~18.6 h; both rungs ~28.1 h. Neither changes the
registered direction on these three tasks until it is run.

## 2. The XNLI collapse is described but not explained

Measured: at 113.47M the cell is a uniform predictor (NLL $\ln 3$, one class on
94.7% of rows), and at 14.4M/33.7M it is a late-training collapse into confident
errors. The thesis reports the signatures and does not diagnose the mechanism.

- Missing analysis, no GPU: from the stored metric histories and evidence tensors,
  determine whether the collapse is (a) an optimisation failure visible in the loss
  trace, (b) a label-space geometry problem (three-way NLI with a 1,024-byte window
  and a byte-level encoder), or (c) a data effect (XNLI premise/hypothesis language
  mix versus the corpus). Each has a different next experiment.
- Closing experiment, if the diagnosis points at optimisation: a pre-registered
  learning-rate/length grid on XNLI only. The 14.4M and 33.7M cells suggest length,
  not rate; do not assume.
- Cost: diagnosis is hours of analysis; a three-arm XNLI grid at 113.47M is
  ~3 GPU-hours at the measured 0.93 GPU-hour per large suite cell.

## 2b. Context scaling is measured but not integrated (2026-10-30)

The thesis's long-context claim rests on one 5,120-byte arm. A second,
complementary measurement now exists and is negative: five dev tasks, one 14.4M
checkpoint pretrained at 256 positions, one head per task, the same weights
evaluated at 256/512/1024/2048/4096 bytes. No task is context-additive;
ContractNLI degrades monotonically (69.79% → 37.03%), and the three
document/abstention heads are at or near majority. The lexical control shows
ContractNLI is learnable (64.71--66.71%), so this is not a task artefact.

- Artifacts: `results/context_scaling/` (`PRE_REGISTRATION.md`, amendment 1,
  `CONTEXT_CURVE.json`, `RESULTS.md`, `LEXICAL_CONTROL.md`). One seed.
- Thesis gap: §5.4.4 currently carries the 5,120 arm as the only length result.
  It needs a short subsection that reports this curve and labels it a
  **partial-context, one-seed** result, not a general long-context claim.
- Missing follow-up: a length-adapted 4,096 arm on the same rows. Without it,
  the result cannot separate “the weights were never trained for the window”
  from “the model cannot use the window.”
- The sealed 12-task locked suite is documented in `research/suites/` and
  `MISSION_LEDGER.md` §14; it has not been opened. Do not run it to fill this gap.

## 3. Evidence holes still printed as `pending`

| table | missing rows | cause | cost to close |
|---|---|---|---|
| 4.1 | 2 exhaustive protocol replicates | run directory lost with sandbox | ~3 min (62--81 s per exhaustive pass) |
| 4.2 | mid and large fixed-budget cells (2) | lost with sandbox, no summary | ~1.0 GPU-h (100M tokens at 105k and 39.2k tok/s) |
| 4.3 | large-rung milestone trajectory | pre-loss build; no surviving intermediate checkpoints | a new run, 11.2 GPU-h, not a repair |
| 4.7 | large exhaustive row (1) | exhaustive pass never run at 113.47M | ~7 min forward at 39.2k tok/s |
| 4.8 | context rows at 1,024 and 4,096 | reclaimed | ~0.4 GPU-h (142.6M tokens each at 197k tok/s) |
| 5.6 | small-rung 8,000-row arm | not run | ~1 GPU-h |
| 5.8 | CDN probe at r=8 | vendor quota cell | an account/quota, not compute |
| prose | `ProtoNewSpread`, `ProtoRatio`, `LargeHeldout`, `LargeTrainBits`, `LargeGap`, `RowSmallWide*` | dependent on the rows above | follows from the rows above |

Total to close every table hole except the trajectory: **~1.5 GPU-hours**. The
trajectory is not worth 11.2 GPU-hours to repair: it is a milestone curve, not a
claim the thesis needs.

**The strongest public Jev comparator is protocol-matched in two pre-registered
tests, and both fall short.** `results/jev_matched/` fixes all 3,080 official
rows, the exact 24 examples, the 113.47M full-bank head, a 5,120-byte window and
a $-2.0$-point non-inferiority rule in a pre-registered, hashed file before any
forward pass. The primary arm scores 22.44% (691/3,080; Wilson 21.00--23.94),
the zero-example arm 88.38% (2,722/3,080; 87.20--89.46), and Jev 92.40%
(2,846/3,080). The pre-registered verdict is "the gap survives"; exact McNemar
$p<10^{-300}$ for the primary arm. The row-set confound is dead. A2, at the
head's native 1,024 bytes, falls to 58.21% at 12.2 examples, so examples alone
cost 30 points inside the trained length; A1 at 5,120 bytes additionally
extrapolates beyond the 1,024 positions the head was trained on, so its 22.44%
is a joint example-and-length failure. The pattern is consistent with
mean-pooling dilution, but A1 changes two things at once.

`results/jev_query/` separates them. A query-anchored head, pre-registered
before any training step (`bde97e74…`), replaces mean pooling with a masked
attention pool queried by the customer message. On the same 3,080 rows and the
same 24 examples it scores 89.03% (2,742/3,080; 87.87--90.08), 3.38 points below
Jev; exact McNemar p=3.99e-9, bootstrap lower -4.32, non-inferior at -2 points
false; the pre-registered verdict is "P0 missed". The control settles the
mechanism: the old head on the identical contexts scores 59.12%, so the query
pool is worth +29.90 points (p=3.6e-212) and the examples were averaged away,
not unreadable. But the primary arm is only +0.65 points above the old head's
no-example arm (88.38%), so at a 1,020-byte budget the retrieved examples buy
almost nothing beyond the message. The primary arm fits 11.9 complete examples
on average (range 2--19) and receives no category definitions; Jev supplies all
24 examples and all 77 definitions. The residual is therefore an information
budget, not a pooling defect.

`results/jev_query_5120/` tests that lever, after amendment 1 (`7993177d…`)
replaced the two-stage schedule with one joint fine-tune for infrastructure
reasons (the session that ran it stayed up for 1.7 h, longer than the three
30-minute sessions the amendment was written against; the amendment was already
hashed and was followed unchanged): 3,000 steps at 5,120 and batch 8, 122.88M
tokens, 1.69 GPU-hours, all
24 examples on every training and test row. It scores **87.11%** (2,683/3,080;
95% Wilson 85.88--88.25), 5.29 points below Jev (exact McNemar p=5.0e-18,
bootstrap lower -6.30, non-inferior at -2 points false), 1.92 points below the
native-window query head (89.03%) and 1.27 points below the old head's
message-only arm (88.38%, p=0.038). It does read the examples: +27.99 points
over the old head on the same packed contexts (p=6.9e-180) and +25.36 over its
own no-example arm, which is out of distribution. The training/test separation
is 96.62% vs 87.11% (-9.51 points), final training loss 0.727 and mean last-50
loss 0.221, so the arm is less interpolated than the native head and still
scores lower. At 5,120 bytes the examples are not merely insufficient; they are
negative transfer for this joint schedule, which also confounds the example
budget with the separate masked-LM adaptation that amendment 1 dropped.

The next lever is the full-information arm: the 24 examples plus the 77
category definitions at 16k--32k bytes, preceded by a separate masked-LM
adaptation. Cost is projected from the measured 2.02 s/step at 5,120 with
$t(L)=2.02\,[(L/5120)(1-a)+(L/5120)^2 a]$, attention share $a\in[0.3,0.5]$,
3,000 head steps plus 1,220 adaptation steps: **12.6--15.9 GPU-hours at 16k**
and **39.8--56.2 at 32k**, before any further allowance if 32k needs more than
8-way gradient accumulation. These are projections, not results.

Plausibility: the untested information is the 77 category definitions. The
largest context effect measured in this thesis is +0.65 points (the native
query head's examples over the message-only arm), and at 5,120 bytes the
examples were negative against the message-only control (-1.27). Recovering the
remaining 5.29 points from definitions alone would be eight times the largest
measured context gain, so the full-information arm is the last lever worth
paying for, not a likely win. If it misses, the honest statement is that the P0
bar is out of reach for this model class and budget, not that another head will
recover it.

## 4. Statistical coverage

- **No calibration partition.** The v1 dev suite has train/test only. No
  threshold or abstention policy was fitted in the context experiment, so its
  argmax results are unaffected, but a future calibration run needs a new suite
  version with a disjoint calibration split. Fitting on the inspected dev test
  or on the locked suite is a protocol violation.
- **One seed per suite cell.** 27 cells, one run each. Three cells were executed
  twice on the same checkpoint and test draw: repeats differ by up to 1.25 points
  and flip up to 274 of 2,000 predictions. Two more full suite passes would cost
  ~35 GPU-hours and give a per-cell run-to-run interval rather than a
  binomial-only one.
- **No multiplicity control.** 27 comparisons are reported without family-wise or
  FDR correction; the Wilson intervals are marginal. A defensible version either
  pre-registers a small family of primary contrasts or reports Holm-adjusted
  $p$-values. McNemar $p$-values are stored per cell and in `COMPARISON.json`, so
  this is a writing/analysis change, not a rerun.
- **Equivalence language.** "Tie" is currently interval overlap. A pre-registered
  TOST bound (e.g. $\pm 2$ points) would make the tie claim falsifiable. Not done.
- **$\alpha$ has no seed interval.** E3 (two more seeds at tiny, small, mid) is
  costed at 1.4 GPU-hours in §6.

## 5. Related work and citations: closed in this build

Chapter 2 now runs to 3,020 words and cites 92 distinct works. It covers, with a
stated position for each: scaling laws and their replication (Kaplan; Hoffmann;
Besiroglu; Muennighoff; Sardana; Schaeffer), byte-level and token-free modelling
(Sennrich; SentencePiece; ByT5; CANINE; Charformer; MegaByte; BLT), long context
(RoPE; PI; YaRN; FlashAttention; Longformer; streaming; Mamba; Hyena; Lost in the
Middle; RULER; LongBench), in-context learning and task induction (GPT-3;
chain-of-thought; induction heads; Wei et al. 2023; Akyürek; Xie; Min), selection
among candidates (RankNet; ListNet; Liu's review; Shen; Lee; and constrained
decoding: Hokamp; Post; PICARD; NeuroLogic; grammar decoding; contrastive
decoding), set-valued and conformal prediction (Sadinle; Mortier; Chzhen; Romano;
Angelopoulos; Bates), selective classification and calibration (Geifman;
SelectiveNet; Guo; Nixon; Kumar), set architectures (Deep Sets; Set Transformer;
DETR; Pointer Networks), retrieval (DPR; RAG; REALM; FiD), multimodal byte spaces
(wav2vec 2.0; Whisper; EnCodec; ImageBind; Chameleon; Transfusion; Gemini), the
decider evidence (TypeSafe; Laya; AY Automate; jev-benchmarks; Bonn; Vodilka), and
evaluation statistics (Wilson; McNemar; Dietterich; Dror; Card; Recht; CheckList;
Contrast Sets).

**Verification.** All 67 arXiv identifiers resolve through the arXiv API with
titles matching the entries. Eleven Crossref DOIs resolve, including the newly
added Sennrich DOI; the twelfth is the arXiv DOI for `bonn2026jev`, which
Crossref does not index but the arXiv API resolves. Two metadata errors were corrected in this pass (Charformer title;
Sennrich year and venue) and two preprint records were promoted to their
published venue (Sadinle, JASA 2019; Sennrich, ACL 2016). No entry was removed.
Every bibliography entry is cited; an uncited entry is an audit finding and must
not be left in a submitted bibliography.

What remains is not a related-work hole but a methods hole; see §6. The chapter
is long enough to position the thesis but not so long that it substitutes for the
instrument description a committee needs to rebuild it.

## 6. Method chapter detail

Chapter 3 is 1,009 words. It explains the encoding, objective, ladder geometry,
corpus, evaluation protocols and execution policy. Missing, in the order an
external rebuild would need it:

- full architecture table: layers, widths, heads, FFN multiplier, activations,
  parameter counts per rung, and the parameter-count discrepancy between
  `113.47M` (suite record) and `113{,}465{,}856` (ladder table);
- optimizer and schedule: AdamW betas/eps/weight decay, warmup shape, clipping,
  dropout, mask policy and its RNG;
- corpus construction: mixture, deduplication counts, split boundaries, the exact
  held-out tail, and the builder version that reproduced `\CorpusSha`;
- evaluation harness: candidate construction, packing, scoring, the dual path, and
  the definition of every metric used (accuracy, NLL, ECE, Wilson, McNemar);
- compute accounting: measured throughputs (present) and the derived GPU-hour
  ledger;
- failure policy: retries, OOM handling, and what makes a cell non-gradeable
  (partly present in Appendix A).

## 7. Reproducibility appendix

Present: a provenance table of generator commands, a hash manifest, the
checkpoint-discipline record, and the rule that a partial run cannot be graded.

Missing:
- pinned environment (Python, torch, CUDA/driver, GPU model) and the exact
  commands with their working directories;
- dataset download provenance with hashes and licences for all nine tasks,
  including a citation for each benchmark dataset rather than only a path;
- a single `make verify` that regenerates every table, re-hashes every per-cell
  record and fails on any mismatch (the suite generator does this for Table 4.5;
  the other generators do not);
- the corpus builder's own hash and a statement of what would change it.

## 8. External validity

One corpus, one domain mix, no code share (the dataset resolution hangs), one GPU
model, English-heavy tasks, and 4,000 training rows per suite cell. The thesis
states these. It does not measure a second corpus, a second accelerator, or a
non-English suite; no cross-corpus bits/byte claim is made and none should be
added without refitting the bigram baseline.

## 9. Claims deliberately not made

Multimodality (no data pipeline, no run), a 4B model (permission, not result),
an abstention guarantee from the coverage curve, and near-perfect accuracy.
These are listed in §6 and must stay listed until the artifacts exist.

Long context is no longer a single unmeasured point, but it is still not a
positive claim. The 5,120-byte arm is 87.11% (below native), and the new
fixed-checkpoint curve has **no context-additive task**; ContractNLI falls
32.75 points at 4,096. The dev curve is one seed and the ContractNLI rows are
partial-context at every length, so it is evidence against a simple
“longer window helps” claim, not proof that a length-adapted or full-document
model must fail.

A head-to-head win over Jev on shared rows is also not claimed. Three
pre-registered tests now exist on the same 3,080 rows and the same 24 examples:
the mean-pool head scores 22.44%, the query-anchored head 89.03%, and the
5,120-byte joint fine-tune 87.11%, all against Jev's 92.40% (`results/
jev_matched/`, §5.4.2; `results/jev_query/`, §5.4.3; `results/jev_query_5120/`,
§5.4.4). What remains unrun is the full-information arm at 16k--32k bytes (24
examples plus the 77 definitions, preceded by a separate masked-LM adaptation);
its projected cost is 12.6--15.9 GPU-hours at 16k and 39.8--56.2 at 32k. No win
is claimed from the retired 2,000-row draw or the weaker 78.8% comparator.

## 10. Author-supplied fields (blocking)

These are the only front-matter fields the repository cannot establish. They are
marked in the document by requirement label rather than filled with a guess, and
they must be supplied by the author before filing.

- **AUTH-1. Department and institution.** No record in this source tree names a
  department or institution, so the degree page asserts none. Supply the exact
  departmental and institutional wording required by the degree-granting body.
- **AUTH-2. Committee.** No committee member appears in any stored record, so the
  degree page names none and implies none. Supply the committee list, including
  any external member and the chair.
- **AUTH-3. Final submission date.** The title page carries `Version of
  30 October 2026`, the date this build was verified, not a claimed filing date.
  Replace it with the date the dissertation is submitted when that date exists.

## 11. Submission mechanics

- Front matter is present and built: title page, degree page, abstract,
  acknowledgments, table of contents, list of tables and a non-empty list of
  figures. The three fields above remain author-supplied by design; there is no
  provisional-status marker, invented department, invented committee or
  fabricated name anywhere in the source.
- `thesis/figures/` holds four figures, all emitted by `thesis/gen_figures.py`
  from the stored per-example records and summaries: pooling dilution against
  three context settings, accuracy against context length with the Jev reference,
  the 5,120-byte arm's train/test separation, and per-class Banking77 error
  concentration. `--check` re-verifies each record against its recorded SHA-256
  and every gold index against `results/jev_matched/inputs.json` before writing
  anything.
- The Overleaf bundle under `dist/overleaf/` must be regenerated and re-manifested
  after any content change; `scripts/pack_overleaf.py` ships `frontmatter/` and
  `figures/` with the sections and tables, then compiles the extracted bundle and
  checks errors, undefined references/citations, overfull boxes and the figure
  count. Current state is recorded in `thesis/BUILD.md` after each pack.
- `paper/` now holds a submission-format paper (EMNLP 2026, official ACL
  style, 9 pages, 0 errors, 0 undefined references/citations, 21 references,
  213 generated macros). Its scope is the pre-registered Jev comparator, the
  mean-pooling mechanism and the information-budget bound; it deliberately
  does not carry the nine-cell third-rung comparison, the negative scale
  verdict or the E4 width result. Those remain thesis material and are the
  extension path if the paper is widened. Build and reconciliation are in
  `paper/BUILD.md` and `paper/RECONCILIATION.md`.

## Ordered queue

1. Full-information query head on the same 113.47M encoder: 16k--32k bytes with
   all 24 examples and the 77 category definitions, preceded by a separate
   masked-LM adaptation and a new pre-registration. Cost from the measured
   2.02 s/step at 5,120: 12.6--15.9 GPU-hours at 16k, 39.8--56.2 at 32k. The
   native-window query head is done (89.03%, missed by 3.38) and the 5,120
   arm is done (87.11%, missed by 5.29); this is the only remaining path to the
   P0 bar, and if it misses the bar is out of reach for this class and budget.
2. Method chapter expansion from the artifacts listed in §6 (writing, no GPU).
   This is the dominant committee blocker.
3. XNLI collapse diagnosis from stored artifacts (hours, no GPU), then decide the
   grid.
4. Close the table holes in §3 (~1.5 GPU-hours).
5. E3 $\alpha$ seed interval (1.4 GPU-hours).
6. E4 variants: a second seed at 113.47M (~9.3 h) and/or the 33.7M grid (~9.3 h
   plus checkpoint staging), if a seed-aware or cross-rung width claim is needed.
7. Reproducibility harness (engineering). Front matter is built; only the three
   author-supplied fields AUTH-1--3 remain open.
8. Suite seed replication (~35 GPU-hours) only if the paper route requires
   seed-aware intervals; it changes error bars, not the verdict.
9. Paper extension to the full suite once 1--6 are in; the current paper is the
   comparator/mechanism paper.

GPU total to close every remaining evidence hole except the long-parity arm and
seed replication: **~5.9 GPU-hours** (table holes 1.5, XNLI grid 3, E3 1.4); the
short 5,120 arm is done and cost 1.69. The full-information 16k--32k arm adds
12.6--15.9 h at 16k or 39.8--56.2 h at 32k; two extra suite seeds add ~35 h.
Neither total includes a new pretraining run, and no further rung is justified
by the measured result: the 7.96$\times$ parameter increase did not close the
four cells.
