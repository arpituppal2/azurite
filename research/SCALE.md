# Why the transfer result is at 2.41M parameters, not the 4B ceiling

## The short answer

`2.41M` is the **tiny rung**, not the project's maximum. The measured ladder is
2.41M → 14.26M → 33.70M → 113.47M parameters. The 4B figure is a permitted
ceiling, and spending it on the current 1.63B-token corpus would not produce a
stronger result; it would produce a memorisation result. The thesis already
measures that failure mode at 113M.

## The arithmetic that decides it

The v2 corpus contains **1,633,580,647 tokens**. Tokens per parameter at each
rung:

| model | parameters | D/N on this corpus | tokens needed for D/N = 20 |
|---|---:|---:|---:|
| tiny (transfer probe) | 2,411,200 | 677.5 | 48.2M |
| small | 14,260,992 | 114.5 | 285.2M |
| mid | 33,696,768 | 48.5 | 673.9M |
| large | 113,465,856 | 14.4 | 2.27B |
| 4B ceiling | 4,000,000,000 | **0.41** | **80.0B** |

A 4B model on 1.63B tokens sees **0.41 tokens per parameter**. On the same v2
corpus the 113M rung sees 14.4, so 4B is 35× further below the reference; even
the old 113M run at D/N = 0.88 — the rung where the measured train/held-out gap
exploded to 39× (train 0.074 bits/byte, held-out 2.895) — saw twice as many
tokens per parameter as 4B would. The model would be storing the corpus, not
learning a function that transfers. Adding parameters at fixed data moves
further into that regime, not toward a better decider.

## The measured scale result is the point

The fixed-data sweep is not an accident of budget. It is the result: held-out
loss is U-shaped in parameters when data is pinned, with the upturn at D/N < 2.
The matched-D/N ladder exists precisely to avoid that confound. The correct
scale-up is therefore **more tokens first**, then more parameters on the
iso-D/N line — not a 4B model on a 1.6B-token corpus.

## What a real 4B experiment would cost

At D/N = 20 it needs **80B tokens**, 49× the current corpus. The measured large
rung runs at about 39,000 tokens/s on the 95 GiB card; throughput falls roughly
inversely with parameter count, so a 4B model is on the order of 1,000 tokens/s
before any sharding or activation-checkpointing overhead. 80B / 1,000 ≈ 80M
seconds ≈ **22,000 GPU-hours on one card**, i.e. years of wall clock, before
data collection. Even a 10× efficiency gain leaves thousands of GPU-hours. That
is a cluster-scale experiment, not a 12-hour disposable-sandbox experiment.

## What is actually claimed

The transfer claim is deliberately at the tiny rung because that is where a
**matched random-initialisation control** is affordable and where the effect is
not confounded by the model's ability to store the fine-tuning set. The claim is
that byte-level pretraining carries decision-relevant structure at 2.41M
parameters, not that the effect is constant to 4B.

The 4B ceiling belongs in a future-work protocol with three preconditions:
1. a corpus of at least 80B unique tokens (not repeated passes);
2. multi-GPU sharding or a 4B-capable serving/training budget;
3. the same matched-random control and contamination audit at that scale.

Until those exist, a 4B number would be a memorisation measurement wearing a
scaling claim's clothes. The thesis says that plainly rather than spending the
ceiling for appearance.
