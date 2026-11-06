# The queue is now a measurement, not a promise

Written after the ladder's own status file was caught lying: `ISO_QUEUE.json`
reported `small` as *running* while the card was running `large x5` and nothing
named `small` existed as a process.

## Why it lied

Two drivers write queue state, and only one of them is careful:

| script | state file | what it does wrong |
|---|---|---|
| `scripts/iso_ladder.py` | `ISO_QUEUE.json` | sets `status="running"` **before** launching the child, never waits for the card, and never clears the label if it dies |
| `scripts/gpu_queue.py` | `GPU_QUEUE_STATE.json` | (old version) wrote nothing until a job *finished*, so a waiting queue was invisible |

`iso_ladder.py` died mid-ladder leaving `small: running` on disk with no process
behind it. Anything reading that file was reading a stale claim.

## What `gpu_queue.py` records now

One state file, one writer, every transition stamped with wall-clock epoch
seconds:

- `waiting` while another trainer holds the card (with `waiting_on` naming it),
- `starting` between claiming a job and `Popen` returning,
- `running` **only after** `Popen` gives a live PID — the old bug is structurally
  impossible now,
- `scoring` while `eval_full.py` runs,
- `complete` / `failed` with `exit`.

Per job: `started`, `started_ts`, `finished`, `finished_ts`, `exit`,
`train_seconds`, `score_seconds`, and `idle_s` = this job's start minus the
previous job's stop (or the moment the card was observed free, for job 1).
`gpu_wait_s` records how long the queue sat armed before it got the card.
`events` keeps the last 200 transitions.

A notebook cell (`queue_proof`) reads that file; nothing in the thesis trusts
`ISO_QUEUE.json` any more.

## Decision: the card changed hands, priced in GPU-hours

`l1_large_byte_x5` was stopped at **323,338,240 / 400,000,000 tokens (81%)**,
driver and child both terminated (both reached state `Z`), checkpoint kept
(1,299 MiB, written at the segment boundary).

| option | tokens | measured tok/s | cost | what it buys |
|---|---:|---:|---:|---|
| finish `l1_large_byte_x5` | 76,661,760 | 39,203 | 0.54 h | old 100 MB stream, 4 epochs — epoch repetition |
| `small_ctx256` | 142,609,920 | 197,394 | 0.20 h | 1.63 B corpus, D/N = 10 |
| `small_ctx1024` | 142,609,920 | 197,394 | 0.20 h | same, context middle |
| `tiny_iso10` | 24,112,000 | 622,665 | 0.01 h | ladder point 1 |
| `mid_iso10` | 336,967,680 | 104,712 | 0.89 h | ladder point 2 |
| `large_iso10` | 1,134,658,560 | 39,203 | 8.04 h | ladder point 3 |

The deciding fact is not the 0.54 h alone — it is that x5's corpus
(`byte_100000000.u16`) is superseded by `byte_corpus_v2.u16` (1.63 B bytes) and
that at 4 epochs its held-out improvement is *epoch repetition*, which the thesis
table for it already says out loud. The same card time buys the entire context
axis on the corpus the law is fitted on. x5 remains resumable from its
checkpoint; the exact command is in `results/pretrain/l1_large_byte_x5/stopped.json`.

Handoff idle: **10.0 s**, measured from the external SIGTERM at 05:08:13.5Z to
the queue's observation of a free card at 05:08:23Z. That is the 5 s poll plus
the scheduling of the check, not a human gap — no job was waiting on a person.

## Numbers that were wrong before this pass

- **"x5 has no recorded token target."** It has one: `driver.json` stores it as
  `tokens_target`, and a reader looked for `tokens`. The field exists; the
  earlier claim was a key error, now corrected here and in the cell.
- **Per-rung throughput.** Reading `tokens_per_s` from the last five lines of
  `train.jsonl` produced 3,430,337 tok/s for the mid rung, which is impossible.
  The honest figures come from end-to-end driver records
  (`tokens_target / elapsed_total_s`), listed in the table above.

## 2026-10-29 — the 113.47M rung measured

The large rung is no longer missing. `l1_large_byte_iso10v3` reached its full
1,134,673,920-token budget at step 69,255 and its 1.36 GB checkpoint is on local
disk with SHA-256 `aba65b60e30654a9895dc9c365d62b555f96dcce96ad8a8ad56cc718ee0865ec`,
matching the sandbox's own record. All nine suite cells were run at the
unchanged pre-registered recipe; `results/suite_large_113m/RESULTS.md` states
the verdict and `COMPARISON.md` holds the three-rung table with Wilson
intervals, exact paired McNemar tests and the Jev gaps.

The answer the ladder was built to give: **scale is not the path to parity.**
Against 14.4M the 7.95× larger encoder wins four single-text cells by 1.70–2.80
pp, loses XNLI by 9.95 pp, and leaves four cells statistically unmoved. The
113.47M XNLI cell is degenerate — 94.7% of test rows get one class, NLL 1.0986 =
ln 3, mean confidence 0.3337 — so the remaining 52-point gap is not a capacity
gap. AG News is still 5.25 pp low and Language ID 3.80 pp low; CLINC-OOS is a
statistical tie 1.00 pp under its cited point.

Three cells were executed twice on the same checkpoint: repeat runs differ by up
to 1.25 pp and flip up to 274 of 2,000 predictions (`REPLICATION.md`). That is
the noise floor every rung delta in that directory is read against.

The nine-way concurrency wave and the five-way wave died with CUDA OOM and are
kept as infrastructure records under `_failed_wave1/` and `_failed_wave2/`, not
as results. The evaluation cost 8.343 GPU-hours serial-equivalent.
