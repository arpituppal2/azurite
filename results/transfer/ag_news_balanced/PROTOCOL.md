# AG News balanced transfer: frozen protocol

Recorded before launching this comparison, 2026-10-28.

## Scope and acceptance

Only the missing AG News comparison is authorized: tiny byte encoder, random vs pretrained, seeds 0/1/2. No new rung, optimizer sweep, early stopping, GPU work, thesis editing, or GPU queue changes. Existing corrected Banking77 settings are held: 2,000 training examples, 500 test examples, six nominal epochs, batch 32, AdamW learning rate 0.0005 / weight decay 0.01, clipping 1.0, all parameters trainable. AG News uses its existing four-candidate cross-encoder path. The original batch scheduler and 378 updates are preserved (it drops incomplete batches; six nominal epochs are not six exact full passes).

Use the existing stratified sampler, split seed 0 for train and 1 for test, fixed across all training seeds and both initializations: 500 training / 125 test examples per canonical class. Candidate order is the existing per-example deterministic shuffle controlled by training seed; keep it identical between initialization arms within each seed. All six jobs are sequential, CPU-only, CUDA hidden, four intra-op threads, low OS priority. Do not pause or alter any existing job.

The pretrained source is the sandbox's `results/pretrain/l1_tiny_byte/ckpt.pt`, not the different local file at that path. Before training, copy it into this evidence directory, hash it, freeze the harness and imported Azurite sources, and retain dataset fingerprints, exact row indices, canonical labels, context hashes, candidate banks/orders, and training-schedule hashes. Preserve source and final checkpoint identities and retain all six fine-tuned checkpoints, raw per-example predictions, canonical and candidate-slot gold labels, and full candidate probabilities. Refuse overwriting completed evidence.

## Amendment 2026-10-28: reclamation and source-checkpoint identity

The first launch ran inside Molab sandbox `sb-b379f12a3548609a`, which was reclaimed
before any cell finished. The named source checkpoint,
`3cf42123682c0ae74ed9949259f24eed2d123c0756d72ae16d1b2e4c0cd4bed8`
(28,973,877 bytes; step 6,104; 100,007,936 tokens), was lost with that disk and was
never copied to the local checkout. It cannot be restored byte-for-byte.

The v2 corpus was rebuilt from the checked-in builder and reproduced the recorded
SHA-256 exactly: `9267a2ea071a517803b3b47c01adf9464608a8178fbf12018ce5dcc9ca38365f`
(1,633,580,647 tokens, 339,841 documents, 112 exact duplicates dropped). The tiny
checkpoint was then retrained with the same protocol named above: seq 1024, batch 16,
seed 0, 100,000,000 requested tokens, AdamW lr 6e-4, cosine schedule, 6,104 steps and
100,007,936 tokens seen. Its held-out score is 1.2937 bits/byte on 19,610 tokens.

The retrained checkpoint reproduces the protocol but **not the bitwise weights**:

- old (lost): `3cf42123682c0ae74ed9949259f24eed2d123c0756d72ae16d1b2e4c0cd4bed8`
- amended:    `817e7f0449fae7d228f3d570bdbfe4c0efe031383b3579a4528a077ba6140410`
- size: 28,973,877 bytes; step 6,104; 100,007,936 tokens; seq 1024; batch 16; seed 0

The cause is non-bitwise-deterministic GPU execution, not a changed corpus or schedule.
This amendment is material: the AG News pretrained arm is now initialised from a different
checkpoint than the corrected Banking77 runs. AG News remains a valid paired
pretrained-vs-random comparison, but it is **not** a continuation of the exact Banking77
initialisation. No Banking77 result is rewritten on the strength of this run. The amended
checkpoint hash is frozen in `frozen_manifest.json` before the six cells run.

## Analysis fixed before outcomes

- Per seed and arm: correct/500, accuracy, 95% Wilson interval, NLL and ECE. Per paired seed: accuracy difference, discordant counts and exact two-sided McNemar p-value; no pooled McNemar that treats repeated items as independent.
- Pooled accuracy is sum(correct)/1500, explicitly 1,500 decisions over the same 500 unique test items and three training seeds. Report the requested descriptive pooled Wilson interval but do not interpret it as accounting for seed/item dependence.
- Primary uncertainty: 10,000 paired crossed bootstrap replicates, RNG seed 20261028. Resample the three training seeds with replacement and the 125 test items within each of four canonical classes with replacement, sharing all resampling between arms and across seeds. Report percentile 95% intervals for pooled arm accuracies and their paired difference. This describes variation over these seeds/items conditional on the fixed train draw/checkpoint, not retraining/data-draw or deployment uncertainty. Three seeds give limited information about seed variation.
- Compare with the historical +6.4 percentage-point gain without silently reusing it. Positive transfer survives if the primary paired difference interval is wholly above zero; a wholly negative interval is adverse, otherwise inconclusive. Separately state whether 6.4 pp lies inside, below, or above the new interval. A lower bound above 6.4 supports a larger gain in this new protocol; it is not a paired test against the old imbalanced run.
- No model/threshold selection on test results. Use the final scheduled update for every arm; preserve intermediate evaluations only to match the prior protocol.

## Required checks before reporting

1. All six jobs exit zero with seeds exactly 0/1/2 and settings matching this protocol.
2. Both arms have identical draw fingerprints, canonical gold, candidate orders, packing, training schedule hashes, initial head hashes, and source identities per seed.
3. Exact canonical class counts are 500 each in training and 125 each in testing; 2,000/500 unique source row IDs; no discarded rows. Canonical predictions are reconstructed from retained candidate order, not compared across seeds as candidate-slot IDs.
4. Source checkpoint SHA-256 stays fixed; final checkpoints exist and their hashes match records. Re-load them on CPU and reproduce raw predictions/probabilities.
5. Copy evidence into the local checkout and verify hashes; keep existing thesis/PDF hashes unchanged. Observe large_iso10 progress without writing its state.

## Progress

- Protocol frozen; evidence instrumentation and CPU preflight are next.
- 2026-10-28: first sandbox reclaimed before any final cell; no result retrieved.
- 2026-10-28: corpus rebuilt (hash verified) and source checkpoint retrained; identity
  amended above. Six-cell rerun is the next step.
