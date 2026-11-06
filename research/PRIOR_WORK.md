# Prior Work — what is established, what is contested, what this thesis borrows

Citations marked ⚠ need the PDF read before submission; the claim is not in doubt but the
exact figure is quoted from memory and must be checked against the source.

---

## 1. Scaling laws

**Kaplan et al., 2020, "Scaling Laws for Neural Language Models."** Loss falls as a power
law in parameters, data, and compute, when all three move together. The result is a family
of curves, not a single curve: it says nothing about what happens when parameters move and
data does not. This is the distinction the first ladder collapsed, and §2.1 of THESIS.md is
a direct demonstration of why it matters.

**Hoffmann et al., 2022, "Training Compute-Optimal Large Language Models" (Chinchilla).**
For a fixed compute budget, parameters and tokens should scale together at roughly
**20 tokens per parameter** ⚠. Two consequences the thesis uses:

1. A model at D/N = 0.88 is ~23× below the compute-optimal token count for its size. Its
   poor held-out score is the expected outcome, not an anomaly.
2. Published architecture comparisons at a fixed token budget are, in general, comparisons
   at different distances from this line. The thesis's methodological claim (§2.2) applies
   to that literature, not only to this project.

## 2. Memorisation and generalisation

**Zhang et al., 2017, "Understanding deep learning requires rethinking generalization."**
Networks fit random labels perfectly and still generalise on real labels; capacity to fit
is not capacity to generalise. The 39× train/held-out gap at D/N = 0.88 in §2.1 is the
same phenomenon in the data-limited direction: the model stores the corpus because storing
is the cheapest descent direction available.

**Double descent** (Belkin et al., 2019; Nakkiran et al., 2020) predicts a non-monotone
risk curve as capacity grows past the interpolation threshold. The U-shape in §2.1 is
consistent with it, but the thesis should **not** claim to have observed double descent:
the sweep is at fixed D, one seed per point, and the upturn coincides with D/N crossing 1.
The memorisation explanation is the one that makes a prediction (E2) and it is testable;
"double descent" here would be a name for the shape rather than an explanation of it.

## 3. Byte-level modelling

**Xue et al., 2022, "ByT5: Towards a token-free future with pre-trained byte-to-byte
models."** Byte-level models are competitive with subword models, more robust to noise and
spelling, and slower because sequences are longer. Azurite takes the byte-level route for
two reasons the thesis must defend rather than assume:

1. **No tokeniser boundary.** A tokeniser is a lossy, corpus-specific interface; it is also
   where multimodal input breaks, since audio and images have no natural subword
   segmentation. Bytes give one input space for text, code, and quantised audio.
2. **The decider framing makes sequence length cheaper.** If the model selects among
   candidates rather than generating freely, the expensive part is encoding context once,
   not decoding many bytes.

The cost is length. At 4× the sequence length of a 4-byte tokeniser, attention is 16× more
expensive per unit text. This is the honest weakness of the approach and the thesis must
report the accuracy-per-FLOP frontier, not just accuracy.

**CANINE** (Clark et al., 2022) and **Charformer** (Tay et al., 2021) are the relevant
alternatives: both avoid subword tokenisation but reintroduce a learned downsampling step,
which is a tokeniser in all but name. Azurite's position is that the downsampling should be
in the model's own attention, not a fixed preprocessing stage.

## 4. Masked training

**Devlin et al., 2019 (BERT)** uses 15% masking, tuned for a small token budget.
**MosaicBERT** (Portes et al., 2023) shows 30% masking trains ~2× faster at equal compute
because more positions carry gradient per forward pass ⚠. Azurite uses 30%, following
MosaicBERT. The mask rate is a hyperparameter of the *objective*, not of the architecture,
and it interacts with D/N: at 30% and D/N < 1 the model sees each byte roughly once per
epoch as a target, which is part of why the memorisation boundary sits near D/N ≈ 1–2.

## 5. Deciders and the TypeSafe claim

Jev is a decider: given a context and a set of candidates, select one. The reliability
argument is that constraining the output space to |C| choices removes the failure mode of
free generation. That argument is sound as far as it goes, and the thesis should say so
plainly — a decider cannot hallucinate a candidate that is not in the set.

The part that is **not** established, and which this thesis tests, is the inference from
"cannot emit an invalid answer" to "selects the correct answer". Accuracy on |C| = 2 is not
evidence about |C| = 64, and neither is accuracy at 2k context evidence at 32k. The
valuation's mechanism claim requires the second and third, not the first. E4 and E5 in
THESIS.md are the direct measurements.

This is also where the thesis must be careful about what it is attacking. The claim under
test is the *mechanism* claim, not the product: a decider that is 97% accurate at |C| = 2
on in-distribution data may be an excellent product and still not support a $10B valuation
premised on near-perfect judgement in general.

## 6. Measurement practice

There is no canonical citation for "small-scale scaling comparisons are inside their own
noise floor", because the field reports point estimates. The contribution in §2.2 is
therefore methodological and self-contained: a protocol whose sampling error (0.352
bits/byte across six draws of one checkpoint) exceeds the effect being measured (0.106
between small and mid) cannot support an ordering, and an exhaustive scorer removes the
sampling error at a cost of 60–80 s per checkpoint. **Paper (a) in §6 of THESIS.md is this
result alone**, and it is submittable without any of the architecture work.
