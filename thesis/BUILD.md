# Building the written record

The thesis compiles from `thesis/` against the results in this checkout. That is
deliberate: every number in the built PDF comes from `results/` next to it, not
from a value typed into prose. The same tree also builds inside the Molab
sandbox, where the GPU results are produced.

The laptop has a TeX engine --- `tectonic 0.17.0` at
`/opt/homebrew/bin/tectonic` --- and it builds the same 84-page PDF from the
same `thesis/tables/*.tex` once those tables have been generated where the
results live and copied back. Tectonic's bundle CDN is unreachable from the
sandbox, which is why the sandbox path uses TeX Live; from the laptop, tectonic
is the faster route and needs no install.

## Toolchain

TeX Live from the sandbox's package manager (no `sudo` needed; the sandbox runs
as root):

```bash
apt-get update && apt-get install -y --no-install-recommends \
  texlive-latex-base texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-bibtex-extra texlive-science
```

`texlive-science` is not optional: `siunitx` lives there, and without it
`main.tex` aborts on `\qty`/`\num`.

Tectonic was tried first in the sandbox and abandoned --- its bundle CDN times
out from that network, so a `tectonic` build is not reproducible there.

## Commands

```bash
# tables first: they are generated, never transcribed
python thesis/gen_tables.py          # run at the workspace root, where results/ exists
python thesis/gen_comparator.py      # comparator tables + macros_cmp.tex (decode-vs-score, CDN)
python thesis/gen_scaling.py         # row-scaling and coverage tables
python thesis/gen_suite.py           # the two-rung suite, prior metadata
python thesis/gen_suite_three.py     # Table 4.5 and every 113.47M number in prose
python thesis/gen_jev_matched.py     # Table 5.3 and every protocol-matched Banking77 number
python thesis/gen_query.py           # Table 5.4 and every query-anchored head number
python thesis/gen_query_5120.py      # the 5,120-byte joint-fine-tune table and every number in it
python thesis/gen_contamination.py   # support-bank audit and decontaminated rerun macros
python thesis/gen_figures.py         # the four figures and their macros, from stored records

# protocol-matched Banking77 verdict and manifest (run where the records live)
python scripts/jev_matched_verdict.py
python scripts/jev_matched_manifest.py
python scripts/query_verdict.py      # regenerate results/jev_query/eval/VERDICT.md
python scripts/query_verdict_5120.py --dir results/jev_query_5120/eval
                                     # regenerate the 5,120 verdict from its summary

# E4 (candidate-set sweep): export and verify per-example records, aggregate,
# then format.  The summary refuses to run if any gold hash or correctness bit
# disagrees with its record; gen_e4.py only formats the verified summary.
python scripts/e4_per_example.py --root results/e4_candidate_sweep
python scripts/e4_summary.py --root results/e4_candidate_sweep
python thesis/gen_e4.py             # Table 4.9 and every E4 number in prose

# artifact manifest (run where the weights live, i.e. the Molab box)
python scripts/e4_manifest.py --root results/e4_candidate_sweep

# thesis: tectonic runs BibTeX automatically
cd thesis
/opt/homebrew/bin/tectonic main.tex

# or, with TeX Live: LaTeX, BibTeX, then two passes to settle references
# pdflatex -interaction=nonstopmode main.tex
# bibtex main
# pdflatex -interaction=nonstopmode main.tex
# pdflatex -interaction=nonstopmode main.tex

# paper
cd ../paper
pdflatex -interaction=nonstopmode azurite.tex
bibtex azurite
pdflatex -interaction=nonstopmode azurite.tex
pdflatex -interaction=nonstopmode azurite.tex
```

The thesis has its own bibliography (`thesis/refs.bib`, included with
`\bibliography{refs}`), so BibTeX is required; tectonic runs it automatically.
The paper needs four passes: one to write `.aux`, BibTeX, then two to settle
cross-references.

## Overleaf bundle

```bash
python3 scripts/pack_overleaf.py --verify
```

Writes `dist/overleaf/azurite-thesis-overleaf.zip` and its `MANIFEST.sha256`
(one line per file inside the zip, sorted, fixed timestamps, macOS `._*`
resource forks excluded), then extracts the bundle to a temporary directory and
compiles it, failing on any LaTeX error, undefined reference, undefined
citation, overfull box or missing figure. The bundle contains 52 files:
`main.tex`, `refs.bib`, three frontmatter files, eleven section files,
thirty-two generated table fragments and four figures. `--verify` on
2026-10-30 produced 84 pages, 4 figures, 0 errors, 0 undefined references,
0 undefined citations, 0 overfull boxes and 0 missing glyphs. Packing the same
tree twice yields the same zip hash; the current zip SHA-256 is
`e123565c60e4…`.

## Last measured build

Thesis only, parsed from the `.log` the build emitted and from the extracted
PDF text:

| document | pages | errors | undefined refs | undefined cites | overfull | `??` in PDF |
|---|---:|---:|---:|---:|---|---:|
| `thesis/main.pdf` | 84 | 0 | 0 | 0 | 0 | 0 |

21 underfull boxes remain in the tectonic log (cosmetic; justified
`p`-columns in the result tables, the bibliography and the appendix provenance
table). There are no overfull boxes. The xdvipdfmx warning `Object @page.i already defined` is a
duplicate page destination in the PDF backend, not a LaTeX error; links and page
numbers render correctly. The bibliography has 93 entries, all cited.

Citation audit, 2026-10-29: all 67 arXiv identifiers resolve through the arXiv
API with titles matching the entries; 11 Crossref DOIs resolve, including the
newly added Sennrich DOI; the twelfth is the arXiv DOI for `bonn2026jev`, which
Crossref does not index and the arXiv API resolves. Chapter 2 grew from 705 words and 11 citations to 3,020
words and 92 distinct cited works. Two metadata errors were corrected
(Charformer title; Sennrich year and venue) and two preprint records were
promoted to their published venue (Sadinle, JASA 2019; Sennrich, ACL 2016).

`??` is counted on the *extracted PDF text*, not on the log: a cross-reference
that failed to resolve would render as `??` even if LaTeX reported no error.

## What "pending" means in the built PDF

32 `pending` strings appear in the built text (three of them from the
`pending_transfer` class named in the error-concentration figure and its
caption). The cell-level data holes are:
two exhaustive-replicate rows (Table 4.1), the mid and large fixed-budget cells
(Table 4.2), the empty large-rung trajectory (Table 4.3), the large exhaustive
row (Table 4.7), two context-length rows (Table 4.8), the small-rung 8k-row arm
(Table 5.6), and the r=8 CDN quota cell (Table 5.8). The rest are prose that
names the hole and explains why it is open. `gen_tables.py` writes
`\emph{pending}` for a missing measurement and a real value the moment
`results/` contains it, so no table can silently show a stale or projected
number.

## Why the PDF is not in this directory

The agent channel truncates payloads above roughly 20 KB, and the sandbox
exposes no static file route, so a large PDF cannot be pulled back intact
through either path. The documents are therefore rendered **inside the
notebook** (the `documents` cell shows the PDFs from disk, with their page
counts and sha256), and this file is the recipe that reproduces them. Nothing
about the build depends on the agent channel. The extracted Overleaf bundle has
been compiled locally, so the uploaded package is known to build.
