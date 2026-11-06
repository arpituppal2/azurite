# Third-party notices

## Banking77

The bundled support messages, the 77 intent names and the evaluation rows used
to measure this checkpoint come from **Banking77**:

> Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan
> Vulić. *Efficient Intent Detection with Dual Sentence Encoders.* NLP4ConvAI
> at ACL 2020. https://arxiv.org/abs/2003.04807

Banking77 is published by PolyAI under the **Creative Commons Attribution 4.0
International (CC BY 4.0)** licence. The copy used here was pulled from the
`mteb/banking77` mirror at commit `18072d2685ea`; `DATA.md` records the split
hashes and the two row-count differences from the canonical release (9,993
training rows and 3,080 test rows were available, versus the published 10,003
and 3,080).

The wheel contains a 9,968-message retrieval index derived from that split. It
does **not** contain the evaluation questions or gold labels. The intent names
are reproduced because the classifier's output space is those names.

Twenty-five messages in the source split duplicated evaluation texts under the
decontamination normalisation; they were removed from the retrieval index, not
edited. The support messages are otherwise unmodified.
`azurite/resources/provenance.json` records the SHA-256 of the exact clean
support file and of the removed-row record used to build the wheel.

## Weights and training data

The checkpoint is trained from random initialisation by this project. It does
not contain or fine-tune any third-party model weights. The pretraining corpus
provenance, including per-source licences and unresolved licences, is recorded
in `DATA.md` and `data/MANIFEST.json`.

## Python dependencies

`torch` (BSD-3-Clause) and `numpy` (BSD-3-Clause) are declared dependencies and
are not vendored into the wheel. Their licences travel with their own
distributions.
