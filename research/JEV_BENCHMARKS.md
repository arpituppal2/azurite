# Jev benchmark inventory (published points)

**Source:** Deusser, Sparrenberg & Sifa, *Evaluating and Benchmarking the System One Model Jev*, [arXiv:2609.37647v1](https://arxiv.org/html/2609.37647v1), Table 2. Model `jev-1.13.0`, evaluated 2026-09-29 on full public evaluation splits. These are **published zero-shot points, not paired runs**.

This file answers the inventory question first: what did Jev actually publish, on how many rows, and which of those can Azurite be run on at all. The machine-readable copy is [`jev_benchmarks.json`](jev_benchmarks.json).

**Coverage:** 37 datasets in 7 families: text classification 9, intent topic routing 3, nli paraphrase grounding 4, reading comprehension knowledge 5, commonsense reasoning 6, safety moderation legal 6, rubric scoring 4.

## 1. The 37 published points

### Text Classification

| dataset | split | n | metric | Jev | 95% CI | ECE | primitive | Azurite status |
|---|---|---:|---|---:|---|---:|---|---|
| AG News | test | 7,600 | accuracy | 0.885 | 0.878-0.891 | 0.077 | Choice(4) | running_full_split |
| IMDB | test | 25,000 | accuracy | 0.965 | 0.963-0.967 | 0.019 | Choice(2) | prepared_local |
| Rotten Tomatoes | test | 1,066 | accuracy | 0.933 | 0.917-0.947 | 0.038 | Choice(2) | prepared_local |
| SST-2 | validation | 872 | accuracy | 0.964 | 0.951-0.975 | 0.015 | Choice(2) | prepared_local |
| Emotion | test | 2,000 | accuracy | 0.585 | 0.563-0.605 | 0.279 | Choice(6) | running_full_split |
| Financial PhraseBank | test | 970 | accuracy | 0.730 | 0.699-0.756 | 0.138 | Choice(3) | running_full_split |
| SMS Spam | all | 5,574 | F1 | 0.938 | 0.924-0.950 | 0.033 | Noul | prepared_local |
| Language ID | test | 10,000 | accuracy | 0.996 | 0.995-0.997 | 0.003 | Choice(20) | running_full_split |
| GoEmotions | test | 5,427 | macro-F1 | 0.243 | 0.236-0.249 | 0.176 | Noul x28 | needs_multilabel_harness |

### Intent Topic Routing

| dataset | split | n | metric | Jev | 95% CI | ECE | primitive | Azurite status |
|---|---|---:|---|---:|---|---:|---|---|
| Banking77 | test | 3,076 | accuracy | 0.797 | 0.782-0.812 | 0.087 | Choice(77) | running_full_split |
| CLINC150 | test | 5,500 | accuracy | 0.895 | 0.885-0.902 | 0.025 | Choice(151) | running_full_split |
| SIB-200 | test | 41,820 | accuracy | 0.815 | 0.812-0.819 | 0.045 | Choice(7) | not_started |

### Nli Paraphrase Grounding

| dataset | split | n | metric | Jev | 95% CI | ECE | primitive | Azurite status |
|---|---|---:|---|---:|---|---:|---|---|
| ANLI | test R1-R3 | 3,200 | accuracy | 0.739 | 0.725-0.754 | 0.102 | Choice(3) | prepared_local |
| AfriXNLI | test | 10,800 | accuracy | 0.640 | 0.631-0.650 | 0.165 | Choice(3) | not_started |
| PAWS | test | 8,000 | accuracy | 0.850 | 0.842-0.857 | 0.040 | Noul | prepared_local |
| LLM-AggreFact | test | 29,320 | balanced accuracy | 0.786 | 0.776-0.796 | 0.128 | Noul | not_started |

### Reading Comprehension Knowledge

| dataset | split | n | metric | Jev | 95% CI | ECE | primitive | Azurite status |
|---|---|---:|---|---:|---|---:|---|---|
| BoolQ | validation | 3,270 | accuracy | 0.913 | 0.904-0.922 | 0.043 | Noul | prepared_local |
| Belebele | test | 109,800 | accuracy | 0.867 | 0.865-0.869 | 0.015 | Choice(4) | needs_per_row_choice_harness |
| PubMedQA | all | 1,000 | accuracy | 0.787 | 0.762-0.812 | 0.129 | Choice(3) | not_started |
| MMLU | test | 14,042 | accuracy | 0.918 | 0.913-0.923 | 0.025 | Choice(4) | needs_per_row_choice_harness |
| C-Eval | test | 12,342 | accuracy | 0.839 | 0.832-0.846 | 0.014 | Choice(4) | needs_per_row_choice_harness |

### Commonsense Reasoning

| dataset | split | n | metric | Jev | 95% CI | ECE | primitive | Azurite status |
|---|---|---:|---|---:|---|---:|---|---|
| BIG-bench (MC) | validation | 13,227 | accuracy | 0.814 | 0.808-0.821 | 0.026 | Choice(2-118) | needs_per_row_choice_harness |
| HellaSwag | validation | 10,042 | accuracy | 0.955 | 0.951-0.959 | 0.019 | Choice(4) | needs_per_row_choice_harness |
| WinoGrande | validation | 1,267 | accuracy | 0.914 | 0.899-0.930 | 0.025 | Choice(2) | needs_per_row_choice_harness |
| ARC (E+C) | test | 3,548 | accuracy | 0.988 | 0.984-0.991 | 0.005 | Choice(3-5) | needs_per_row_choice_harness |
| CommonsenseQA | validation | 1,221 | accuracy | 0.882 | 0.864-0.900 | 0.031 | Choice(5) | needs_per_row_choice_harness |
| alphaNLI (ART) | validation | 1,532 | accuracy | 0.839 | 0.819-0.858 | 0.065 | Choice(2) | needs_per_row_choice_harness |

### Safety Moderation Legal

| dataset | split | n | metric | Jev | 95% CI | ECE | primitive | Azurite status |
|---|---|---:|---|---:|---|---:|---|---|
| ToxiGen | test | 940 | accuracy (toxic) | 0.878 | 0.857-0.898 | 0.046 | Noul + Score | prepared_local |
| OpenAI Moderation | all | 1,680 | mean AUPRC | 0.717 | 0.675-0.759 | 0.036 | Noul x8 | needs_multilabel_harness |
| ToxicChat | test | 5,083 | F1 (toxic) | 0.786 | 0.753-0.821 | 0.016 | Noul x2 | prepared_local |
| Prompt Injections | test | 116 | accuracy | 0.741 | 0.664-0.828 | 0.236 | Noul | prepared_local |
| AGB-DE | test | 755 | F1 | 0.204 | 0.122-0.282 | 0.264 | Noul | prepared_local |
| UNFAIR-ToS | test | 1,607 | micro-F1 | 0.499 | 0.449-0.544 | 0.088 | Noul x8 | needs_multilabel_harness |

### Rubric Scoring

| dataset | split | n | metric | Jev | 95% CI | ECE | primitive | Azurite status |
|---|---|---:|---|---:|---|---:|---|---|
| STS-B | test | 1,379 | Spearman | 0.890 | 0.879-0.901 | — | Score(6) | needs_ordinal_head |
| SST-5 | test | 2,210 | accuracy | 0.579 | 0.560-0.599 | — | Score(5) | needs_ordinal_head |
| SummEval | test | 1,600 | mean Spearman | 0.554 | 0.523-0.571 | — | Score(5) x4 | needs_ordinal_head |
| HelpSteer2 | validation | 1,038 | mean Spearman | 0.412 | 0.380-0.444 | — | Score(5) x5 | needs_ordinal_head |

## 2. Other public Jev points

These are outside the Bonn 37 but are cited in [`BENCHMARK_REGISTRY.md`](BENCHMARK_REGISTRY.md). They are weaker evidence: different splits, pilots, or a local reconstruction.

| task | Jev | n | source | caveat |
|---|---:|---:|---|---|
| SEC claim verification | 0.975 | 400 | Vals (C4) | supported / contradicted / insufficient |
| LegalBench 12-task slice | 0.730 class-balanced | 396 | Vals (C4) | subset of the full 162-task benchmark |
| AG News | 0.910 | 100 | AbdelStark (C2) | pilot sample |
| AG News | 0.910 | — | AY (C5) | conflicts with Bonn 0.885; split not established |
| Banking77 | 0.788 | 231 | AY (C5) | small sample |
| Banking77 | 0.870 | 100 | AbdelStark (C2) | 72 labels, not 77; not like-for-like |
| Banking77 | 0.924 | 3,080 | local reconstruction (C7) | not a live API run |
| DAIR Emotion | 0.480 | 100 | AbdelStark (C2) | pilot; conflicts with Bonn 0.585 |
| XNLI (en) | 0.860 | — | AY (C5) | no clean source/split; do not use as a comparator |
| ALFWorld unseen | 35% success | 134 episodes | Bonn (C3) | sequential; not a single-call decision |
| TextWorld | 42% success | 90 games | Bonn (C3) | sequential |
| BabyAI-Text | 21% success | 250 missions | Bonn (C3) | sequential |
| Blackjack | 90% optimal action | 5,000 | Bonn (C3) | sequential |
| FrozenLake | near-goal avoidance failures | 500 | Bonn (C3) | sequential |

## 3. What can Azurite run today

Azurite is a text encoder with a candidate decision head. It is not an agent and it has no generation path. That splits the 37 into:

- **Single-label classification:** a fixed candidate bank, one decision per row. Six are running on their full official splits in `results/jev_full_20261030/` (AG News, Banking77, CLINC150, Emotion, Financial PhraseBank, Language ID). Eleven more are normalized locally and queued next: IMDB, Rotten Tomatoes, SST-2, SMS Spam, ANLI, PAWS, BoolQ, ToxiGen, ToxicChat, Prompt Injections, AGB-DE. SMS Spam uses a fixed 80/20 split because the HF release ships one split; it is marked as a split mismatch.
- **Per-row multiple choice:** HellaSwag, ARC, WinoGrande, CommonsenseQA, alphaNLI, MMLU, C-Eval, Belebele, BIG-bench. The candidate set changes every row, which the current fixed `CandidateBank` does not represent. This needs a per-row candidate harness; it is a harness change, not a model change.
- **Multi-label:** GoEmotions (28), OpenAI Moderation (8), UNFAIR-ToS (8). One sigmoid per label, thresholded per label; the current softmax-over-candidates path is the wrong shape.
- **Rubric/ordinal:** STS-B, SST-5, SummEval, HelpSteer2. Needs an ordinal head and Spearman/MAE scoring, not argmax accuracy.
- **Sequential:** ALFWorld, TextWorld, BabyAI-Text, Blackjack, FrozenLake. Azurite has no policy loop. Running these would require an agent harness and a different claim; they are not single-call benchmarks.

## 4. Comparability rules

1. The Jev point is a **published reference**, not a paired run. A gap is a gap to a claim, not a head-to-head measurement.
2. Match the split and n where possible. Financial PhraseBank is the known mismatch: Jev n=970, local n=1,000. SMS Spam is a larger mismatch because HF ships one 5,574-row split; the local run uses an 80/20 train/test split. Both are reported, never silently averaged away.
3. Azurite fine-tunes on each task train split; Jev is zero-shot. A win or loss mixes protocol and capacity. It is still the only way to put the two systems on the same rows with the artifacts in this checkout.
4. Do not average across the 37. They have different metrics (accuracy, F1, AUPRC, Spearman) and different label spaces.

