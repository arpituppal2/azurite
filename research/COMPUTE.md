# Compute Plan — how the GPU is kept busy and what it buys

Policy: **the GPU is never idle between experiments.** A finished rung must be followed
immediately by the next one, chosen in advance. Idle GPU-hours are pure loss, and the
expensive failure mode is not a slow run — it is a gap while a human notices a run ended.
Hence `scripts/iso_ladder.py` (a queue) rather than four shell commands.

---

## 1. Hardware

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell, 94.97 GiB, sm_120 |
| CPU / RAM | 20 vCPU / 160 GiB (sandbox, measured 2026-10-28 with `nproc` and `free`; earlier sessions saw 4 cores / 32 GiB) |
| Host | Molab sandbox `sb-b379f12a3548609a`, jupyter + marimo pair session |
| Constraints | 90-minute idle drop, 12-hour session ceiling, **disposable disk** |

The disk constraint is not theoretical: one restart reduced 60 files / 204.7 MB to 40
files / 0.41 MB. Everything needed to resume is either re-stageable by one command
(`scripts/molab_sync.sh --all`) or written outside the sandbox.

## 2. Measured throughput (not assumed)

| rung | params | tok/s end-to-end | steady tok/s | peak VRAM | GiB·s per 100M tokens |
|---|---|---|---|---|---|
| tiny | 2.41M | 621,000 | — | <2 | 161 |
| small | 14.26M | 197,000 | — | 8.1 | 507 |
| mid | 33.70M | 105,000 | ~102,000 | 12 | 955 |
| large | 113.47M | 39,200 | 40,011 | 19.1 | 2,551 |
| 403M (395M class) | 402.97M | 15,347 | — | 50.2 | 6,516 |
| 403M, bf16 autocast | 402.97M | 35,672 | — | 20.2 | 2,803 |

The 403M row is `results/scale_state/BENCH_395M.json` (2026-10-31): 24 layers,
d_model 1,024, 16 heads, d_ff 4,096, seq 1,024, synthetic ids. It is the first
rung that is **not** compute-bound in fp32: batch 16 peaks at 50.2 GiB, and
batch 32 allocates 92.7 GiB and then fails on the next 512 MiB. bf16 autocast
is 2.32x fp32 at batch 8 and uses 20.2 GiB, which is the configuration a 403M
run would use if the precision change were pre-registered.

Peak VRAM is 19.1 of 95 GiB through 113M, and every rung through that size is
compute-bound, so:
- batch size is the only remaining throughput lever (memory is not the limit);
- segment length is bounded by the 90-minute idle rule, not by OOM;
- no rung through 113M needs gradient checkpointing or sharding — those would cost
  throughput for memory that is not scarce. A 403M fp32 run at batch 16 does not
  need them either, but it has no headroom for a larger batch.

Consequence: the schedule is built from tok/s, and a token budget converts to wall clock
with one division. No GPU-hour estimate in this project is taken from literature.

## 3. Fault tolerance

Two independent failure modes, handled separately:

1. **Idle drop (90 min).** Every run is a sequence of bounded segments
   (`train_segments.py --segment-seconds 3000`). Each segment checkpoints, exits 0, and
   writes `segment.json`; the driver resumes from the newest checkpoint. A drop costs at
   most one segment plus one checkpoint interval (200 steps ≈ 80 s at large).
2. **Disk wipe.** The token stream is re-staged by one command and verified by sha256
   `6780940…f9ce776` before training touches it. Weights are the only unrecoverable
   artifact; checkpoints are copied out at milestones (`scripts/track_curve.py`) so a wipe
   loses at most the current segment.

A partial run never writes `summary.json`, so an interrupted rung cannot be graded as
finished — the failure is visible in the report rather than silently averaged in.

## 4. Cost model

At the measured rates, one GPU-hour buys:

- tiny: 2.24B tokens
- small: 709M tokens
- mid: 378M tokens
- large: 141M tokens

| experiment | tokens | GPU-hours | status |
|---|---|---|---|
| E1 fixed-data size sweep | 400M | 1.2 | ✅ done |
| E1b seed replication (3 seeds × tiny/small/mid) | 1.2B | 1.4 | queued |
| E2 epoch turnover at large, D → 400M | 200M more | 1.4 | **running** |
| E3 iso-D/N = 10 ladder | 1.64B | 9.1 | **unblocked; queued on corpus v2** |
| corpus build (v2, 1.63B tokens) | — | ~0.03 | ✅ done, 49 s |
| E4 candidate-set sweep | inference | <0.2 | next |
| E5 context-length sweep | inference | ~1.0 | next |

E3 is 9.1 GPU-hours *and* impossible on 102M unique bytes (§3 of THESIS.md). It is priced
here so the corpus work is understood as an unblocking dependency, not a detour.

## 5. Schedule discipline

Rules the queue enforces or the operator follows:

1. **Ascending rung order within a ladder**, so every completed rung is a usable data
   point and an interruption truncates the most expensive rung rather than the whole line.
2. **Queue, never wait.** `iso_ladder.py` starts the next rung the moment one exits.
3. **Never spend GPU on an experiment whose result cannot change the thesis.** The 500M
   epoch-repetition run was killed for this reason: past D/N ≈ 2 it measures memorisation,
   not scaling, and E2 answers the same question in 1.4 h instead of 6.
4. **Measure the protocol before measuring the model.** 60% of this project's GPU time was
   nearly spent comparing numbers whose noise floor (0.35 bits/byte) exceeded every effect.
   The exhaustive scorer costs 60–80 s per checkpoint; that is the cheapest insurance in
   the pipeline.
5. **Sampling is scheduled, not manual.** `track_curve.py` scores milestones while training
   runs, so the trajectory exists even if the session dies.

## 6. What is still unknown

- Seed-to-seed variance of *training* (E1b). Eval variance is characterised; training
  variance is not, and the tiny/small/mid ordering depends on it.
- Whether batch 16 × seq 1024 is near-optimal for throughput at each size. Answered at
  403M: batch 8 is 14,770 tok/s, batch 16 is 15,347 (+3.9%), batch 32 OOMs. The
  20–40% is not there in fp32; bf16 autocast is 2.32x, and that is a precision
  decision rather than a batch-size one. The 113M batch probe is still unmeasured.
- Real ceiling on the 12-hour session with E3-sized runs — only exercised up to 2.5 h so
  far.
