# Azurite V1 — from-scratch parity plan

**Status:** implementing.
**Date:** 2026-11-01.
**Supersedes:** the 400M byte-model assumption in
`2026-11-01-evidence-selection.md`. H2 remains the selector experiment; this is
the product model around it.

## Decision

One model: **Azurite Select+Read 400M**, trained from scratch. BPE 32,004 trained
on the local corpus; no external encoder and no pretrained weights.

```
32k-token document
 -> 64 chunks x 512 tokens
 -> BM25 or 30M contrastive selector
 -> top 8 chunks = 4,096 tokens
 -> 400M bidirectional reader at 4k
 -> candidate head + conformal abstention
```

Context claim: 32,768 tokens per question and 65,536 per request, accessible
through the selector. The reader sees 4k; this is retrieval, not dense 32k
attention. The 32k claim is valid only if selector recall@8 is at least 90% on
the locked battery. No 262k danger-zone claim.

## Fixed constraints

1. No Jev API call until the internal parity gate passes.
2. Deployment floor: M1/8 GB. Development: M4/16 GB. Training: RTX PRO 6000.
3. One model, one price, no capability ladder.
4. Every number comes from a stored per-row record. Losses remain.
5. No external encoders. The BPE vocabulary is trained on this corpus.

## Arithmetic

- Reader: 402,972,672 parameters, 24 layers, d_model 1024, 16 heads, d_ff 4096,
  RoPE, RMSNorm, SwiGLU, MLM 30%.
- Selector: 33.7M parameters (mid rung; the plan's 30M target), 8 layers,
  d_model 512, 16 heads; BM25 is the zero-parameter baseline.
- Corpus: 20.82 GB text, ~5.14M documents, ~6.2B BPE tokens. One pass, D/N 15.4.
- Pretrain: 6.2B tokens at 4k, 25–32k tok/s, 54–69 GPU-hours; total training
  budget ~77 GPU-hours, 7–10 calendar days with 12-hour segments.
- M1 inference: reader 3.2e12 FLOPs + selector 1.9e12 at 1.3 TFLOPS effective
  ~= 4.2 s. Gate: p95 <= 5.0 s, RSS <= 5.0 GiB, zero swap.

## Data

Local corpus: FineWeb-Edu 12.09 GB (ODC-BY), Wikipedia EN 3.05 GB (CC BY-SA/
GFDL), FineWeb 3.03 GB (ODC-BY), Wikipedia de/fr/es/zh 2.03 GB, FineMath 0.61 GB
(ODC-BY). Exact-sha1 dedup already applied. Hold out 2% by document. Run the
contamination audit before training.

Long-context packs: 100,000 packs of 32k tokens from ContractNLI, SQuAD 2.0,
BoolQ, and packed multi-document sets, with gold spans, 0/2/8 distractors,
early/middle/late positions, 10% distributed evidence.

Decision tuning: train splits only, candidate dropout, label paraphrases, hard
negatives, 1B tokens.

## Tasks

1. Freeze protocol; hash it.
2. Build the 6.2B token stream; record stream SHA and held-out boundary.
3. GPU-only 4k probe; require >=20k tok/s.
4. Reader pretrain; gate held-out bpb beats bigram by >=0.05.
5. Long-context packs + BM25; gate recall@8 >=85% dev.
6. Neural selector; gate +3 pp recall@8 over BM25, 95% interval excluding zero.
7. Decision tuning; gate every family beats majority and BM25 controls; evidence
   precision >=80%.
8. 10k battery; 2k classification, 1.5k boolean, 1k pairwise, 1k ordinal, 2k
   document, 1k abstention, 1k robustness, 500 multilingual; freeze and hash.
9. Internal parity gate; no Jev. Aggregate 95% lower bound > Jev published +3 pp;
   >=6/8 slices at/above; no slice below -5 pp; 32k NIAH >=95%; multi-hop >=80%;
   selector recall >=90%; evidence precision >=80%; abstention 90% +/-2 pp; M1
   p95 <=5 s/RSS <=5 GiB; Banking77 regression >=89.06 -0.50.
10. M1/M4 profile; fall back to 113M or 2k selected tokens if needed.
11. Product surface: `decide`, `serve`, `adapt`, `bench`, `doctor`; contract
    invariants; no network.
12. Live Jev once + external evaluation, only after task 9.

## No-Jev rule

No `jev-1.14.0` call in tasks 1–9. The gate is a 95% lower bound above Jev's
published point plus a 3 pp protocol margin. A point estimate above Jev is not
enough. A Jev call is confirmation, not diagnosis.

## XNLI

A 400M from-scratch model on 6.2B tokens, with ~0.5B multilingual tokens, will
not reach Jev's 86.0 XNLI. The plan targets aggregate parity and reports XNLI as
a loss. Closing XNLI requires 10–20B multilingual tokens and a 600M–1B model;
that is a separate plan. External encoders are ruled out.

## Timeline

Nov 2–8 protocol/stream/probe; Nov 9–22 reader pretrain; Nov 23–29 packs/BM25/
selector; Nov 30–Dec 6 decision tuning; Dec 7–13 battery and gate; Dec 14–20 M1
and product; Dec 21–Jan 3 buffer; Jan 4–17 external/live Jev/launch candidate;
Jan 19 public beta; Feb 16 launch if gate passed, otherwise March 30 for parity.

## Stop rules

1. No Jev until the gate.
2. No 32k claim if selector recall@8 <90% locked.
3. No XNLI parity claim unless >=86.0 same rows.
4. No M1 claim if p95 >5 s or RSS >5 GiB.
5. No from-scratch run without a frozen stream hash and held-out split.
6. No mid-run architecture change.
7. No deleting failed runs.
