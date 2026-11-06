# Azurite Banking77 intent classifier

**Version** 0.1.0 · **Status** research preview · **Date** 30 October 2026

## What this is

A byte-level encoder that classifies one English banking support message into
exactly one of 77 Banking77 intents. It is a 113.47M-parameter bidirectional
encoder (`d_model=768`, 12 layers, 24 heads) with a message-query attention
pool and a 77-way scaled-cosine head. The packaged state dict has 114,845,185
parameters including the head.

The model was trained from random initialisation. It uses no pretrained weights
and no downloaded vocabulary; the tokenizer is a 256-byte alphabet with four
special tokens.

This is a **research artifact**, not a production banking system.

## Intended use

- Research replication of the Banking77 comparison in `paper/azurite.pdf`.
- Triage experiments on English banking support messages that are known to be
  in-scope for one of the 77 intents.
- A worked example of a small, local, byte-level decision model.

**Out of scope.** Open-set or out-of-domain detection; non-English messages;
images, audio or documents; credit, fraud, account or eligibility decisions;
safety-critical routing; any use where a wrong intent causes financial or legal
harm. The model always returns one of its 77 labels, even for unrelated text.

## How a prediction is made

1. BM25 retrieves up to 24 labelled messages from the bundled 9,968-message
   decontaminated Banking77 support split, at most four per intent. The 25
   evaluation texts found in the 9,993-row source split are removed at build
   time, and the build refuses to emit a bundle containing a normalised
   evaluation text.
2. Complete `message => intent` lines are packed after the customer message
   until the checkpoint's 1,020-byte context budget is reached. A partial
   example is never included.
3. The encoder runs once over the packed context and once over the message. The
   message embedding supplies the query for a masked attention pool over the
   context.
4. The pooled context vector is scored against the 77 intent embeddings by
   scaled cosine similarity. The softmax over those scores is returned as
   "confidence".

**Both example sources are measured.** The pre-registered arm used the frozen
24-example set from the Jev comparator protocol
(`results/jev_matched/inputs.json`). The shipped `azurite predict` path uses
local BM25 over the bundled 9,968-message decontaminated support split; it was
measured end-to-end through the installed wheel on the same 3,080 rows and
produced **2,743 correct = 89.06%**. The same path over the original 9,993-row
bank produced 2,742 correct = 89.03%, with the same prediction on every row and
2,971 of the 3,080 contexts byte-identical to the frozen arm. That original run
is not an independent retrieval result: the packaged BM25 reproduces the frozen
example contexts for most rows, so it is a product-path check of the same
protocol rather than a second statistical arm. Decontaminating the bank changes
256 contexts and 3 predictions (2 wins, 1 loss; exact paired McNemar p = 1.00).
`azurite verify` replays the frozen path exactly.

## Measured result

The frozen-example rows below are on the 3,080 official Banking77 test rows,
one forward pass per row, with the pre-registered protocol. The shipped BM25
path is measured separately in the next table. Sources:
`results/jev_query/eval/summary.json`,
`results/jev_query_5120/eval/summary.json`,
`results/product_path/summary.json` and
`results/product_path_clean/summary.json`.

| Arm | Context | Correct / 3,080 | Accuracy | 95% Wilson |
|---|---|---:|---:|---|
| `query-native`, frozen 24-example set (pre-registered) | message + examples, 1,020 B | 2,742 | 89.03% | 87.87–90.08 |
| `query-native`, message only | no examples | 1,880 | 61.04% | 59.30–62.75 |
| Old mean-pool head, message only | no examples | 2,722 | 88.38% | 87.20–89.46 |
| Old mean-pool head, examples | 1,020 B | 1,821 | 59.12% | 57.38–60.85 |
| `query-joint-5120` | all 24 examples, 5,118 B | 2,683 | 87.11% | 85.88–88.25 |
| Jev comparator | reconstructed JSON prompt | 2,846 | 92.40% | — |

**Shipped product path (local BM25, decontaminated bank):**

| Path | Rows | Correct | Accuracy | 95% Wilson |
|---|---:|---:|---:|---|
| **shipped BM25, clean bank (default)** | 3,080 | **2,743** | **89.06%** | 87.91–90.11 |
| shipped BM25, original bank (historical) | 3,080 | 2,742 | 89.03% | 87.87–90.08 |
| shipped BM25, unaffected rows | 2,825 | 2,511 | 88.88% | 87.67–89.99 |
| frozen-example reference | 3,080 | 2,742 | 89.03% | 87.87–90.08 |

The shipped bundle is the clean bank. The original-bank path is not materially
different from the frozen arm: 0.00 points, zero discordant rows (exact paired
McNemar p = 1), and all 3,080 prediction indices identical. The **source**
support bank contained **25 of its 9,993 messages** as evaluation texts under
normalised text (6 exact; all 25 carry the same gold label). They occurred in
the frozen example sets of **255 rows (8.28%)**, as **283 instances**: 25
self-retrievals and 258 cross-row retrievals. 236 rows retrieved at least one
cross-row test message; 230 did so without also retrieving their own. 81
instances carried the row's gold label and 202 a different label, so this was a
content leak rather than a label leak. The contamination was not uniform: 2,825
rows retrieved no test text. The 25 source messages are cross-split duplicates
in Banking77; the wheel removes them, so 89.03% is **not a clean held-out
estimate and is not the shipped default**.

Rebuilding the support bank with every one of those 25 messages removed
(normalised match, zero residue) and re-running the shipped path scores
**2,743/3,080 = 89.06%** (95% Wilson 87.91–90.11). That is +0.03 points versus
89.03%, with 2 wins and 1 loss (exact paired McNemar p = 1.00). The genuinely
unaffected subset scores 2,511/2,825 = 88.88% (87.67–89.99). The decontaminated
rerun is therefore 0.03 points above the original, so the overlap did not inflate
the headline. Excluding only the 25 self-retrievals would not be a clean subset:
it would leave 230 cross-row contaminated rows in place. The decontaminated
rerun's 337 errors are spread across 70 of the 77 intents; the
largest concentrations are `pending_transfer` (12), `failed_transfer` (11),
`compromised_card` (10) and six intents with 9 each. Per-class errors and every
per-example record are in `results/product_path_clean/`; the original records
remain in `results/product_path/`.

The pre-registered primary arm is `query-native` with examples. Against Jev it
is **−3.38 points**, exact McNemar **p = 3.99 × 10⁻⁹**, paired 95% lower bound
**−4.32 points**. The registered verdict is **“P0 missed.”** The 5,120-byte
joint fine-tune is **−5.29 points** against Jev and **1.92 points below** the
native query head.

The mechanism is real: on identical contexts, replacing mean pooling with the
message-query attention pool raises accuracy from 59.12% to 89.03%
(+29.90 points). But the fix does not beat the comparator. The 89.03% arm is only
0.65 points above the old head's message-only arm, which is not significant.

## Limitations

- **One task, one language.** Banking77 English only. The 77 intents are not a
  general taxonomy.
- **Forced choice.** There is no “none of these” option and no calibrated
  reject. Out-of-domain input still receives a confident label.
- **Confidence is not calibrated.** The softmax of scaled cosine scores is
  frequently near 0 or 1. Do not threshold it as a probability without fitting
  calibration on held-out data.
- **Native window 1,024 bytes.** The `query-joint-5120` checkpoint accepts 5,120
  bytes but is *less* accurate (87.11%) and is not the default. Neither is a
  long-context model.
- **No multimodality.** Text bytes only.
- **Source support is 9,993 rows, not the published 10,003.** The pulled split
  had ten fewer rows; `DATA.md` records the discrepancy. The wheel removes 25
  evaluation texts and ships 9,968.
- **Retrieval is part of the system.** Changing the support index changes the
  predictions, even with identical weights. The shipped BM25 bank is
  decontaminated and scores 89.06%. The original 9,993-row bank scored 89.03%
  and is reported only with its disclosure: 25 evaluation messages as normalised
  duplicates (6 exact), 255 rows retrieving at least one test message (283
  instances: 25 self, 258 cross-row). The 2,825 genuinely unaffected rows score
  88.88%. Do not present the original 89.03% as a clean held-out estimate.
- **Device differences.** The measurement ran on an NVIDIA RTX PRO 6000 with
  PyTorch 2.11. CPU or MPS inference can differ in the last bits; `azurite
  verify` re-checks the stored rows on the device you name.

## Provenance

| Artifact | SHA-256 |
|---|---|
| `query-native` weights (`results/jev_query/final.pt`) | `7992c97362b3c6b19ad39cf94792218ebb9c5508a8d3f3969f3526f8d15cb723` |
| `query-joint-5120` weights (`results/jev_query_5120/final.pt`) | `e7455fc462e9371e5529a79d81d2f98714f154a9b136a9cd114c2a8affb2046d` |
| Product-path summary (`results/product_path/summary.json`) | `8e0326de99b0031114800149a3521b14bd13ef9e40a4843f00f6111bf197a6f7` |
| Decontaminated summary (`results/product_path_clean/summary.json`) | `5f117ea1da460b4769dc2f0493c0c706f4cc45a7ffeb480dfecc3ce233dde0b6` |
| Decontaminated records (`results/product_path_clean/per_example.jsonl`) | `6811e25bc07b798653dca210607d3910ed8a7f76080676ecf3725a71888cb6d9` |
| Product-path per-example records (`results/product_path/per_example_bm25.jsonl`) | `262ebd5ca50dce4c315c48542ff4d8ebe567dd5739c215fb182c4df3358cfd70` |
| Bundled Banking77 bundle (decontaminated) | `cb0b7d6389b45b437fb662943ad9e87332531990df8ccb918f79babd07a70a43` |
| Clean support split (`results/product_path_clean/clean_support.jsonl.gz`) | `707f979f75c5c2e8734684030940d62609ee67e944880e0e1b29fbb00c7175ab` |
| Removed support rows (`results/product_path_clean/removed_support.json`) | `2cfcfacd54214518da52d316623c2a176d7c99f2ef8a2b599c797da5760f6671` |
| Source support split (`results/jev_query/support.jsonl`) | `2e95269ca99cc6de51ef19c458c55f0a49c3aff30ca3c3d84871ccaa2df2f197` |
| Frozen evaluation inputs (`results/jev_matched/inputs.json`) | `8dccd8c898865be08bd272ab88c2b4fcc829472a9252f96d966c0aab78c3c418` |

The checkpoint registry inside the package carries the same digests and refuses
to load a file whose hash does not match. The bundle is built deterministically
by `scripts/build_package_resources.py`; its source hashes are in
`azurite/resources/provenance.json`.

## Licence and attribution

Code and trained weights: MIT (`LICENSE`). The bundled Banking77 support text is
CC BY 4.0; attribution and the exact source are in `THIRD_PARTY_NOTICES.md`.
No Jev weights, prompts or outputs are redistributed. The comparison uses only
the frozen 0/1 correctness vector recorded from a public run.

## Citation

> *More Context, Less Accuracy: A Pre-Registered Negative Result for a
> Byte-Level Decider.* Submission draft, EMNLP 2026 format, `paper/azurite.pdf`.

The full study, including the scaling ladder and the decision-suite results, is
in `thesis/main.pdf`.
