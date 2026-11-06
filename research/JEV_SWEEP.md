# Jev benchmark sweep: final state

Completed 2026-10-30 on the Molab RTX PRO 6000 sandbox
(`sb-4de1577da2e9c976.sb.molab.run`). The full report is
[`results/jev_sweep_20261030/RESULTS.md`](../results/jev_sweep_20261030/RESULTS.md).

## Result

Eleven Jev-overlap benchmarks completed at their full official test splits:
**5 above the published Jev point, 0 ties, 6 below**. The all-benchmark
"beats or equates Jev" thesis is false.

| result | tasks |
|---|---|
| above | Banking77 (+8.50 pp), DAIR Emotion (+25.15), Financial PhraseBank (+17.20), Prompt Injections (+24.18), AGB-DE (+21.54 F1) |
| below | CLINC150 (−16.14), Rotten Tomatoes (−20.41), SST-2 (−20.37), SMS Spam (−3.77 F1), ToxiGen (−12.69), ToxicChat (−12.68 F1) |

Four more cells were started and stopped at a point already far below Jev:
IMDB (83.22 at step 333 vs 96.5), PAWS (57.46 at 666 vs 85.0), ANLI (33.41 at
step 0 vs 73.9), BoolQ (62.17 at step 0 vs 91.3). They are not counted as final
scores.

## Coverage

`research/jev_benchmarks.json` records all 37 datasets in the Bonn study.
Eleven are complete here; four are partial. The remaining 22 need a per-row
multiple-choice harness (9), a multi-label harness (3), an ordinal/rubric
harness (4), multilingual data (2), reading/knowledge handling (2), or were not
completed in this sweep (AG News, Language ID). The five sequential-agent Jev
rows are outside the 37-dataset table and outside this single-call interface.

## Protocol

- Checkpoint: `results/pretrain/l1_large_byte_iso10v3/weights_only.pt`, 113M
  byte encoder, SHA-256 `aba65b60e30654a9895dc9c365d62b555f96dcce96ad8a8ad56cc718ee0865ec`.
- Supervised fine-tune on each task's own train split, full official test split,
  batch 32, LR 1e-4, seed 0, 1,024-byte native context.
- 3,000 steps unless the task's recorded step count differs.
- Jev points are published cross-harness references, not paired runs.
- 5.70 measured GPU-hours across the 11 completed cells.

## Infrastructure notes

- The first full-split wave used the pre-fix runner and therefore a 4,000-row
  draw; CLINC150, Rotten Tomatoes and SST-2 retain those numbers.
- A stale `queue_jev_extra.py` process launched a second grid on top of the
  first and caused CUDA OOM on IMDB, SMS Spam, PAWS and ANLI. The duplicate
  runner and its orphans were stopped.
- Small-file uploads through the Molab file endpoint did not overwrite an
  existing remote script; tar + `tar -xf` in the sandbox is the working path.
- The 13-task repair queue was stopped after the two F1 cells (ToxicChat,
  AGB-DE) completed, because the remaining cells could not change the verdict
  and were consuming GPU time.
