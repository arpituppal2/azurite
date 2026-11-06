# Azurite

Can a model trained from scratch challenge the frontier on decision tasks? Answer: not in this weight class.


## Result

contract_nli, split 123, 2048 ctx, two seeds (sha-verified mirrors):

| seed | recall | cond | e2e | shuffled |
|------|--------|------|-----|----------|
| s0   | 0.886 | 0.7088 | 0.628 | 0.390 |
| s1   | 0.900 | 0.6889 | 0.620 | 0.354 |
| mean | 0.893 | 0.6989 | **0.624** | 0.372 |

Pooled e2e 0.624, 95% CI [0.594, 0.654] (n=1000). Seed delta 0.008.
Context carries +0.252 over shuffled.

## Why it stopped

- 400M from-scratch caps at Intelligence ~35 vs Jev's 53. Three independent
  teams, same ceiling (von's public board). Calibration already tied.
- A 4B open finetune reaches 59.4 vs 63.3. The points aren't in this band.
- Everyone halves out-of-domain (Jev 0.741 to 0.367 sealed). Robustness is
  the prize, not benchmarks.
- TypeSafe's own FAQ: no VPC, no on-prem — hosted API only. The open gap is
  a local wire-compatible server, not another checkpoint.

Also measured: truncation 1024→2048 is +0.100 e2e; pooling mean≡attn≡last
once the control was fixed (a prior "attn wins" claim was withdrawn);
margin loss rejected on dev; external baseline a paired tie.

## Use it

- `pip install .` gives the `azurite` CLI. Runnable weights: self-contained
  2.5M byte-level checkpoints under `results/transfer/` (config inside each
  `.pt`), evaluated via `scripts/transfer_modernbert.py`.
- The v1 Banking77 path is documented in `MODEL_CARD.md`; its 437MB
  checkpoints are not in this release. Bulk corpus is not shipped either —
  regenerate from `data/MANIFEST.json`.
