# More Context, Less Accuracy

Submission draft for **EMNLP 2026**, formatted with the official ACL style files
in review mode. The paper is a measurement and mechanism paper, not a victory
lap: the best arm is 3.38 points below the frozen Jev comparator, and the paper
says so in the abstract.

## What is here

| Path | Role |
|---|---|
| `azurite.tex` | The submission. All quantitative claims are generated macros. |
| `refs.bib` | 21 references; every entry is cited and every DOI/URL resolves. |
| `acl.sty`, `acl_natbib.bst` | Official ACL style files, from `acl-org/acl-style-files`. |
| `gen_paper.py` | The only producer of numbers, tables and figures. |
| `generated/` | Macros, tables, figures and prompt-budget record. |
| `BUILD.md` | Build procedure and last measured compile/verification results. |
| `RECONCILIATION.md` | Macro-to-record map and all 213 generated values. |
| `REFERENCE_CHECK.md` | Citation coverage and live DOI/URL checks. |
| `data/jev_prompt/` | Public comparator prompt inputs used only for the budget bound. |
| `archive/noise-floor-draft.tex` | The superseded archival draft, kept for provenance. |

## Build

```bash
python paper/gen_paper.py
cd paper
tectonic azurite.tex
python check_refs.py --online
```

The generator verifies every per-example SHA-256, recomputes every accuracy,
Wilson interval, exact McNemar test and paired bootstrap, and refuses to write
a macro if a stored value differs. It also cross-checks 20 overlapping values
against the thesis macros.

## Claim boundary

The paper tests a text-only, byte-level decider at a 5,120-byte context on
Banking77. It makes no claim about multimodal models or about long-context
models in the modern sense, and it does not claim architectural superiority
over Jev: the comparator prompt is 2.09x larger and contains 77 category
definitions that no Azurite arm receives. The negative result is bounded to the
information budget that was actually tested.

The paper's contribution is the measurement, the located mean-pooling
mechanism, and the bound on the example-budget explanation. It is not a claim
that deciders fail in general.
