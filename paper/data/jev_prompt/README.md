# Comparator prompt inputs

These two JSON files are reproduced from the public Banking77 Jev harness so
that the paper's information-budget measurement is reproducible without a
network call.

- Repository: <https://github.com/simonmesmith/jev-banking77-experiment>
- Commit: `5cac4ff7783a4cfc0badba4124a309dd2a2b9862`
- Files: `configs/descriptions.json`, `data/categories.json`
- Dataset: Banking77, CC-BY-4.0 (Casanueva et al., 2020)
- Purpose: reconstruct the comparator's JSON prompt size from the published
  descriptions, the 77 labels and the stored per-row examples. No model
  weights, predictions or vendor outputs are copied.

`SOURCE.json` records the SHA-256 and byte size of each file. The paper's
generator verifies those hashes before computing the budget; a changed file is
an error, not a silent update.
