# Execution 1 — FROZEN protocol (dual-vs-cross structural micro-screen)

Frozen before execution. Any change requires a new dated amendment; this file is never edited.

## Scope of approval
H0 + Execution 1 ONLY, CPU-only, 0 GPU. Cap: 30 min wall per cell, 4 CPU-h total, sequential.
No reader resume, no objective screen, no expansion without separate approval.
An inconclusive outcome is valid and authorizes nothing further.

## Frozen configuration (Execution 1)
- task=ag_news, arm=byte, rung=tiny, path=dual
- cells: seeds {0,1,2} x init {pretrained,random} = 6 sequential jobs
- examples=2000, test-examples=500, epochs=6 (=378 updates, incomplete batches dropped),
  batch=32, lr=0.0005, AdamW weight_decay=0.01, clip=1.0, all params trainable
- stratified draw, split-seed 0 train / 1 test (500+125 per canonical class)
- checkpoint (pretrained arms only): results/pretrain/l1_tiny_byte/ckpt.pt,
  sha256 817e7f0449fae7d228f3d570bdbfe4c0efe031383b3579a4528a077ba6140410,
  28,973,877 B, step 6104, 100,007,936 tokens
- tokenizer: byte arm built-in (no external vocab file)
- env: --device cpu --threads 4, CUDA_VISIBLE_DEVICES='' (hidden), cwd = workspace root,
  new out dir only (script refuses overwrite)
- per-cell command:
  .venv/python -u scripts/transfer_eval.py --task ag_news --arm byte --rung tiny --path dual
  --init {pretrained|random} --seed {s} --examples 2000 --test-examples 500 --epochs 6
  --batch 32 --lr 0.0005 --stratify --split-seed 0 --device cpu --threads 4
  --evidence-dir results/transfer/ag_news_balanced_dual/{init}_s{s}
  --out results/transfer/ag_news_balanced_dual/{init}_s{s}/result.json
  [+ --ckpt results/pretrain/l1_tiny_byte/ckpt.pt for pretrained]
- expected artifacts per cell: result.json, provenance.json, final.pt, training log

## Frozen comparator (cross path, must match or stop)
- source: results/transfer/ag_news_balanced/ (BALANCED_RESULT.md, 6 result.json + provenance.json)
- per-seed accuracy — pretrained: 0.512 / 0.518 / 0.490; random: 0.510 / 0.490 / 0.486
- pooled: pretrained 760/1500 = 0.5067, random 743/1500 = 0.4953, paired diff +1.13 pp,
  crossed-bootstrap 95% CI -2.60..+4.80 pp, verdict INCONCLUSIVE
- matching conditions: same task/arm/rung/init/seeds/optimizer/budget, same canonical
  train/test row IDs + gold labels (verify via provenance dataset fingerprints + draw hashes).
  KNOWN non-identical: candidate packing/order by path (dual canonical vs cross per-example shuffle).
- tolerance: all 6 exit 0; unpackable=0; steps=378; schedule/provenance recorded;
  final.pt retained with recorded sha. Kill: non-finite loss, unpackable rows,
  fingerprint mismatch, overwrite refusal, any cell failure (stop sequence).

## Frozen decision rule
Paired dual-vs-cross difference per seed/init on the same 500 canonical test rows,
crossed seed/item bootstrap (10k replicates, RNG seed 20261028, same method as frozen protocol);
report accuracy, NLL, ECE, acc@50% coverage.
- cross-dual 95% CI wholly above zero AND no ECE/coverage regression
  -> retain cross-encoder hypothesis, proceed to confirmation (separate approval).
- wholly below -> dual becomes the structural reference (separate approval for next step).
- otherwise -> INCONCLUSIVE, no architecture claim, no further runs.

## H0 outcome (recorded before any execution)
H0 run on sb-58ad8e167881cecb: COMPARATOR NOT MATCHED on this host.
Missing: results/transfer/* (cross cells), results/pretrain/l1_tiny_byte/ckpt.pt,
data/raw/ag_news (extracted), results/parity/eval_battery_10k.
Present: code (azurite, scripts incl. transfer_eval.py), torch 2.11.0, datasets 5.0.0,
20 CPUs, 160 GiB RAM, cuda True (unused: Execution 1 is CPU-only by spec).
Per approval condition: STOP before comparative claims. Execution 1 BLOCKED pending staging.

## Proposed staging (NOT approved, NOT executed)
Copy from verified local checkout to this host, then re-hash before use:
results/pretrain/l1_tiny_byte/ckpt.pt (29 MB) + data/raw/ag_news (30 MB) +
results/transfer/ag_news_balanced cross cells (21 MB) [+ eval_battery_10k (73 MB) for H0 battery check].
Then re-run H0; only a passing H0 unblocks Execution 1.
