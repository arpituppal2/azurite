# AG News balanced transfer — completed on the replacement sandbox

> **Update 2026-10-28:** all six cells completed on `sb-be0b890dd77be9db` using
> the amended checkpoint. The result is +1.13 pp pooled, 95% crossed-bootstrap
> interval -2.60 to +4.80 pp, verdict inconclusive; per-seed +0.20 / +2.80 /
> +0.40 pp. See `summary.json`, `BALANCED_RESULT.md`, and the six `*/result.json`
> files. The interrupted-run history below is retained as provenance.

## Earlier outcome (superseded)

**No final AG News comparison result is available. Whether the historical +6.4 percentage-point gain survives is unmeasured.** Do not use intermediate accuracy as a final result or reuse the historical number as evidence for this comparison.

Molab displayed: “Your sandbox stopped responding and was reclaimed.” Its execution endpoint became unreachable. The sandbox was not relaunched, no GPU job was stopped or restarted, and the GPU queue was not edited by this comparison. Its present state is unavailable, not verified running.

The first run, pretrained seed 0, was last observed at step 126 of 378 at 2026-10-28T08:12:05Z. The last intermediate evaluation was 41.2%; it is **not a completed run**. No final predictions or fine-tuned checkpoints were retrieved. The other five cells were not observed starting. The retained local monitor log records the successful observation and subsequent connection failures.

## Recovery attempt on the replacement sandbox

A fresh Molab sandbox (`sb-be0b890dd77be9db`) was opened after reclamation. The code
was re-staged and the v2 corpus rebuilt; it reproduced the recorded SHA-256 exactly
(`9267a2ea…8365f`). The named tiny checkpoint was retrained at the same protocol
(6,104 steps, 100,007,936 tokens, seq 1024, batch 16, seed 0), giving a new identity
`817e7f04…0410` rather than the lost `3cf421…bed8`. The amendment is recorded in
[PROTOCOL.md](PROTOCOL.md). The six-cell rerun subsequently completed; its
per-seed and pooled records are `summary.json` and the six `*/result.json` files.

## Protocol and preflight that were verified

See the protocol frozen before launch in [PROTOCOL.md](PROTOCOL.md).

- Tiny byte encoder, AG News four-candidate cross-encoder, seeds 0/1/2, pretrained versus random.
- Same corrected Banking77 budget: 2,000 training examples, 500 test examples, batch 32, six nominal epochs / 378 optimizer updates, AdamW learning rate 0.0005 and weight decay 0.01, clipping 1.0, all parameters trainable.
- Train/test split seeds 0/1, fixed across all six runs. Exactly 500 training and 125 test examples per canonical class, 2,000/500 unique source indices, no discarded examples.
- Seed-0 draw hashes including candidate permutations: train `c9f8411a7423ec8665c032141d6b42b3336a00ddddd3c8a6113945cbe6ce9dc4`; test `9dbf2e995913a86d29acee8914218148d1c6e97b4312db2e58d7c6be8cac7a14`.
- Both arms' initial decision heads matched for every seed in sandbox preflight. Core model construction and scheduling functions were AST-identical to the sandbox's original transfer harness.
- CPU-only, CUDA hidden, four threads, OS nice 15, sequential execution from an isolated source snapshot. The active shared sandbox harness and GPU training sources were not overwritten.
- Targeted local tests: 28 passed. Isolated sandbox transfer tests: 14 passed before launch. These verify harness behavior, not experiment completion.

At the final successful observation, `large_iso10` had reached 143,196,160 tokens, reported approximately 39,775.7 tokens/s, and GPU utilization was 100%. That observation predates reclamation; it does not establish its current state or why the sandbox was reclaimed.

## Exact initialization required to resume

The original sandbox pretrained source had:

- SHA-256: `3cf42123682c0ae74ed9949259f24eed2d123c0756d72ae16d1b2e4c0cd4bed8`
- Size: 28,973,877 bytes
- Pretraining step: 6,104
- Tokens: 100,007,936
- Original path: `/marimo/azurite_ws/results/pretrain/l1_tiny_byte/ckpt.pt`

It was copied and hashed within the sandbox before training, but was **not retrieved into the local checkout before reclamation**. A project-local search of checkpoint files and readable checkpoint archives found no matching file SHA-256. The local file at the same relative path differs materially: SHA-256 `499fd68c634ba9fe8c43e08d5af53691198552e1e680786e0d33affbc24eebdc`, step 24,415, tokens 100,003,840. It was not substituted.

The frozen sandbox config source also differed from the local config source. Restore the original frozen source snapshot along with the exact source checkpoint before restarting this protocol. An equivalent model-state backup, if recovered, needs explicit identity verification; a matching architecture or approximately matching token count is insufficient.

## Retained local work and recovery limits

Retained: frozen protocol, CPU-only six-run driver, evidence-verification/analysis script, local tests and instrumented harness, interruption record, local monitor log, the six final result files, six final checkpoints, and the recovered per-seed/pooled intervals.

Still missing: the original pretrained checkpoint, the frozen sandbox source snapshot, and a bitwise replay of the lost checkpoint. The amended checkpoint is identified in the top banner; it is not the lost one.

The analysis specified before final outcomes was executed: per-seed Wilson intervals, exact McNemar tests, descriptive pooled Wilson intervals, and 10,000 paired crossed seed/item bootstrap replicates. No summary statistics or transfer verdict were fabricated.

The thesis was subsequently revised and rebuilt against these records; see `thesis/main.pdf`.
