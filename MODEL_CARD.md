# Azurite v1 — Banking77 intent classifier (archived artifact)

**Status:** archived research artifact. The project moved to transfer
experiments (see `README.md`) and was retired. This card describes the v1
Banking77 system only. Not a production system; not for credit, fraud,
eligibility, or safety routing. It always returns one of 77 labels, even
for unrelated text.

## What it is

Byte-level encoder, 113.47M parameters (`d_model=768`, 12 layers, 24 heads),
message-query attention pool, 77-way scaled-cosine head. Trained from random
initialisation. No pretrained weights. Tokenizer: 256-byte alphabet plus 4
special tokens. English banking support messages only.

Prediction = BM25 retrieval (up to 24 example `message => intent` lines from
the bundled decontaminated 9,968-message support split) + one encoder pass
over the packed context + one over the message, scored against the 77 intents.

## Measured result (3,080 official Banking77 test rows)

| Arm | Correct | Accuracy | 95% Wilson |
|---|---:|---:|---|
| query-native, frozen 24-example set (pre-registered) | 2,742 | 89.03% | 87.87–89.08 |
| shipped path (local BM25, clean bank) | 2,743 | **89.06%** | 87.91–89.11 |
| Jev comparator | 2,846 | 92.40% | — |

Registered verdict: **P0 missed** — −3.38 points vs Jev, exact McNemar
p = 3.99 × 10⁻⁹, paired 95% lower bound −4.32. The mechanism is real
(+29.90 from attention pooling on identical contexts) but it does not beat
the comparator. Message-only: 61.04%. Confidence is not calibrated — do not
threshold it.

Contamination disclosure: the source split contained 25 evaluation texts as
normalised duplicates (255 rows retrieved one). The shipped bank removes all
25; the clean rerun scores 89.06% (+0.03, McNemar p = 1.00), so the overlap
did not inflate the headline. Full per-example records were in the original
workspace; this release keeps the summaries. Do not present 89.03% as a
clean held-out estimate.

## Download and use

- `pip install .` installs the `azurite` CLI (`predict`, `verify`).
  Requires `torch>=2.4`, `numpy>=1.26`.
- Runnable now: self-contained 2.5M-parameter byte-level checkpoints
  (`results/transfer/**/final.pt`, config inside each file), evaluated with
  `scripts/transfer_modernbert.py`. The 113M v1 checkpoints (437MB finals)
  are **not** in this release — they exceed hosted-file limits and the
  originals were lost to sandbox reclamation. Reproduce via
  `scripts/pretrain.py` + `scripts/query_head_train.py` with the corpus
  provenance in `data/MANIFEST.json`.
- The decontaminated support index (`azurite/resources/banking77.json.gz`)
  ships under CC BY 4.0 (see `THIRD_PARTY_NOTICES.md`). No third-party
  model weights, prompts, or raw outputs are redistributed.
- Reference writeup: `paper/azurite.pdf`.

## Licence

Code: Apache-2.0 (`LICENSE`). Bundled Banking77 text: CC BY 4.0.
