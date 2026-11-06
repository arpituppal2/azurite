# Truncation A/B — contract_nli, dev split 0, pool=mean, fixed control

## Question
Contract-NLI contexts average 11,414 chars (max 35,310, all unique), but `--max-len 1024`
shows the model only ~the first third of each document. Is truncation the binding
constraint on the absolute-level deficit, given the model demonstrably reads passages
(+0.12 gap under the fixed control)?

## Result (dev split 0, both seeds; fixed control)

| max-len | seed | recall | cond acc | e2e | shuffled | gap | secs |
|---|---|---|---|---|---|---|---|
| 1024 (`CTRLFIX_MEAN`) | 0 | 0.838 | 0.6232 | 0.526 | 0.402 | +0.124 | ~640 |
| **2048 (`MAXLEN2048`)** | 0 | **0.838** | **0.7255** | **0.608** | 0.438 | **+0.170** | 908.6 |
| **2048 (`MAXLEN2048`)** | 1 | **0.840** | **0.7024** | **0.590** | 0.422 | **+0.168** | 908.7 |

`MAXLEN2048_contract_nli_s0.json` sha256
`115ba584eec71a5ae413f1262b061d199bf0c389939a9239ca9dce182817e17c`;
`MAXLEN2048_contract_nli_s1.json` sha256
`4de9f6bb21928608e2e606b7ded1ef33a1c1e9cea008c627cbccf679c67ea9b9`.

Two-seed mean at 2048: cond **0.714**, e2e **0.599**. The gain replicates (seed spread
±0.012 cond, ±0.009 e2e) — not seed luck.

## Reading
- Recall is unchanged (0.838 → 0.838): the retrieval stage never saw the truncation.
- The entire gain is conditional (+0.102) and flows through to end-to-end (+0.082).
  The passage contains the answer; at 1024 the model could not see enough of it.
- The gap also widens (+0.124 → +0.170): more visible passage → more context use,
  exactly what a working control should show when the bottleneck is input, not head.
- Shuffled control sits at 0.438 vs prior baseline 0.418 — slightly elevated but far
  below real, consistent with a longer window carrying more same-label leakage under
  permutation. Not a concern at this magnitude; flag for the write-up.

## Split-123 promotion (shared-recipe comparison)

| arm (split 123) | pool | max-len | recall | cond acc | e2e |
|---|---|---|---|---|---|
| `ATTN_contract_nli_s0` (v12) | attn | 1024 | 0.838 | 0.6301 | 0.528 |
| `LAST_contract_nli_s0` (v17) | last | 1024 | — | 0.6014 | 0.504 |
| **`S123_MAXLEN2048_contract_nli_s0`** | mean | **2048** | **0.886** | **0.7088** | **0.628** |

`S123_MAXLEN2048_contract_nli_s0.json` sha256
`fb6f402d7d627b831d0bb5a26a3b7091ae7e982f27976101bbd4965bac8114f9`
(shuffled control 0.390, fixed True, 1701.4 s).

Recipe is identical to the v12 split-123 arms (full train, test 500, batch 16,
756/756 steps) except max-len and pool=mean (mean ≡ attn on dev, so the pool change
is not the driver; the 2048-vs-1024 dev pair isolates max-len cleanly at +0.073 e2e
mean over 2 seeds). Full-train recall also rises (0.838 → 0.886): more data helps
the funnel, while max-len helps the reader — the two stages scale independently.

## Status
`S123_MAXLEN2048_contract_nli_s1` (same config, seed 1) running at hand-off.
On completion: mirror, then the shared-recipe 2048 number is a 2-seed mean.
4096 probe stays optional and memory-gated.
