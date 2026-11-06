# Azurite — a from-scratch decision system, measured honestly, retired with numbers

Azurite was an experiment: can one person with free compute train a decision
system from scratch that challenges the frontier (TypeSafe Jev) — faster,
cheaper, stronger? The honest answer is in `results/transfer/envelope_v17/`.
This repo is published as a postmortem with evidence, not a launch.

## What was built

- A ModernBERT-scale encoder trained **from scratch** (no pretrained weights;
  baselines like Laya were comparison-only), plus Choice/Noul/Score decision
  heads, retrieval front-end, and a hash-verified evaluation envelope.
- A measurement apparatus this project is prouder of than any checkpoint:
  fixed-control re-scoring, paired McNemar + bootstrap significance tests,
  mirrored evidence (base64 + sha256), and withdrawn claims. See
  `results/transfer/envelope_v17/CONTROL_DEFECT.md` — a control bug silently
  invalidated weeks of pooling comparisons; it was caught, fixed, and the
  prior "attn wins" claim was withdrawn in writing.

## Headline numbers

**Flagship benchmark — contract_nli, split 123, 2048 ctx, two seeds**
(`S123_2SEED_MEAN_contract_nli.json`; both seeds sha-verified mirrors):

| seed | recall | conditional | end-to-end | shuffled control |
|------|--------|-------------|------------|------------------|
| s0   | 0.886  | 0.7088      | 0.628      | 0.390            |
| s1   | 0.900  | 0.6889      | 0.620      | 0.354            |
| **mean** | **0.893** | **0.6989** | **0.624** | 0.372        |

Pooled end-to-end: 0.624, 95% binomial CI [0.594, 0.654] (n=1000).
Seed delta on e2e is 0.008 — the result is stable, not a lucky seed.
Signal over shuffled contexts is +0.252: the context carries the decision.
What isn't mirrored doesn't count — s1 (`...037bf488...`) was mirrored
from the live host with a sha check on 2026-11-06.
- **Truncation is the binding constraint:** 1024 → 2048 context gave **+0.100**
  end-to-end on the same split (`TRUNCATION_AB.md`).
- **Pooling (mean/attn/last):** no difference once the control defect was fixed
  (`POOLING_EXHAUSTED.md`).
- **Margin loss:** rejected on dev (0.500 vs 0.496, +45% cost).
- **External baseline (paired, settled):** ours 0.8707/0.8613 vs pplx
  0.8787 — deltas −0.0080/−0.0173, McNemar n.s. (`EXTERNAL_BASELINE_PAIRED.md`).
  A tie, reported as a tie.

## Why it was retired

Three facts, all verified, all in this repo or its references:

1. **The ~400M from-scratch ceiling is structural.** Von's public JevBench board
   (archived Oct 2026, 921 stars) puts every ~400M open model at Intelligence
   34–37 vs Jev's 53.1, while calibration is already at parity (75.7 vs 76.3).
   Three independent teams, same ceiling. The points aren't in this band.
2. **Generalization indicts everyone.** On sealed (out-of-domain) tiers even the
   incumbent halves (0.741 → 0.367). The prize is robustness, not benchmarks —
   a research program, not a training run.
3. **The efficient frontier moved.** A 4B open finetune (hopper) reaches 59.4
   composite vs Jev's 63.3; Liquid's d1 family ships open weights with vision
   and audio. From-scratch purity costs ~30 composite points no buyer pays for.

The wedge that survived verification — TypeSafe's own FAQ: *"Can we run Jev
in our own VPC or on-premises? **No. Jev is currently available only through
our hosted API**"* — belongs to a wire-compatible local server, not to another
400M checkpoint. That server was never built here. It should be.

## What's in this repo

- `azurite/` — model, heads, retrieval code.
- `scripts/` — training (`transfer_modernbert.py`, `pretrain.py`), evaluation,
  paired testing (`paired_test.py`), control re-scoring (`rescore_control.py`).
- `tests/` — CPU smoke tests.
- `results/transfer/` — the full evidence envelope: per-arm JSONs, paired
  tests, predictions, small checkpoints, and the four postmortem notes.
- `data/MANIFEST.json`, `data/tokenizers/` — provenance for public corpora
  (FineWeb, Wikipedia, etc.). Bulk corpus/token shards are **not** included;
  regenerate with `scripts/build_corpus.py` + `scripts/pull_datasets.py`.
- `MODEL_CARD.md` — model details and limits. `THIRD_PARTY_NOTICES.md`.

## What was lost

Sandbox reclamation destroyed the seed-1 run before mirroring, and full
fine-tuned encoder weights that lived only on the sandbox. Small heads and all
eval evidence survived (mirrored + hashed). Everything here reproduces from
public data + these scripts, given a GPU.

## License

Apache-2.0. Use it, fork it, beat Jev with it.
