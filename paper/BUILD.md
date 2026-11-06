# Build record

## Venue and style

Target: **EMNLP 2026**. Engine: `tectonic 0.17.0`, XeTeX. Class:
`article` 11pt with `acl.sty` in `review` mode. The official style files are
from <https://github.com/acl-org/acl-style-files> and are copied to the paper
root so a local build does not depend on a TeX Live path.

## Commands

```bash
.venv/bin/python paper/gen_paper.py
cd paper
tectonic --keep-logs azurite.tex
cd ..
.venv/bin/python paper/check_refs.py --online
```

`gen_paper.py` exits non-zero on a per-example hash mismatch, an accuracy or
Wilson mismatch, a paired-test mismatch, a decision-rule mismatch, a missing
paper macro, or a disagreement with the overlapping thesis macros.

## Last measured

- Output: `paper/azurite.pdf`, **9 pages**.
- TeX errors: **0**.
- Undefined references: **0**.
- Undefined citations: **0**.
- BibTeX errors: **0**; 21 citations, 21 entries, 0 uncited.
- Reference check: 21/21 DOI or URL locators resolve (Crossref 200 or HTTP 200).
- Generated macros: **250**, all letters-only and all defined.
- Accuracy arms: 7, over 3,080 rows, with the Jev vector 2,846/3,080.
- Primary: 89.03% vs 92.40%, exact McNemar \(p=3.99\times10^{-9}\), paired 95%
  lower bound -4.32 points, verdict `P0 missed`.
- Contamination audit: 25 of 9,993 support messages are test rows (6 exact);
  255 of 3,080 rows retrieve at least one (283 instances). The decontaminated
  product rerun is 2,743/3,080 = **89.06%** (87.91--90.11); the 2,825
  unaffected rows are 2,511/2,825 = **88.88%** (87.67--89.99).
- Prompt-budget reconstruction: 7,801 definition bytes; 10,683-byte median
  prompt (10,009--13,524); 5,118 bytes is 47.9% of the median.
- Thesis cross-check: 25 shared macros, all agree.
- Unavoidable cosmetic warnings: 35 underfull hboxes, concentrated in the
  reference list and long `\texttt` paths. No overfull boxes are reported by
  the engine.

## Artifacts

`RECONCILIATION.md` maps the headline values to summary fields and lists all
250 generated macros. `REFERENCE_CHECK.md` records the citation and locator
checks. `generated/prompt_budget.json` records the public prompt inputs, their
hashes and the distribution.

## Archived draft

`archive/noise-floor-draft.tex` is the earlier 5-page archival note. It retains
its original warning and is not part of this submission.
