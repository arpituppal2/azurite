# COMPETITIVE UPDATE — perplexity-ai/pplx-decider-flash-v1.2

Status: **investigated independently, evidence separated from claims. Now also an
executed external baseline** (its own decision API, our rows, Molab GPU).
Date: 2026-11-04. Author: Azurite mission.

This document does **not** change Azurite's foundational requirement. Azurite
remains scratch-trained from raw data. Everything below about pplx-decider is
about a **pretrained, distilled derivative used as a comparison baseline** — see
§2 and §11 for why that is not a substitute.

Sources actually read (not inherited from any summary):

| # | Source | What it establishes |
|---|---|---|
| S1 | `https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2` (model card, retrieved 2026-11-04) | architecture summary, training method, Decision Index numbers, primitives, context/latency claims |
| S2 | `decision_config.json` from the repo (downloaded, read) | `base_model: Qwen/Qwen3.5-4B`, `revision 851bf6e8…`, `attention_mode: noncausal_full_attention`, `pooling: last`, `temperature: 1.5539092221462472`, `max_pixels: 1638400`, 255 codes |
| S3 | `NOTICE` from the repo (downloaded, read) | "Derived from Qwen/Qwen3.5-4B, revision 851bf6e8…, under Apache License 2.0" |
| S4 | `LICENSE` from the repo (downloaded, read) | full Apache License 2.0 text, 11,343 chars |
| S5 | `joint_config.json` from the repo (downloaded, read) | `joint-question-inline-readout-v1`, marker `<|fim_suffix|>`, `max_questions: null`, `max_length: null` |
| S6 | `requirements.txt` / `requirements-fast.txt` | torch==2.14.0, transformers==5.17.0, safetensors==0.8.0, pillow==12.3.0, hub==1.31.0; fast variant adds `flash-linear-attention==0.5.2` |
| S7 | repo file listing | `model-0000{1,2}-of-00002.safetensors`, `readout.safetensors`, `source/src/pplx_decider/{model,protocol,fast,types}.py`, `chat_template.jinja`, `release-manifest.json` |
| S8 | `https://anyrouter.dev` (landing page, retrieved 2026-11-04) | a generic multi-provider LLM gateway (323 models / 18 providers, `provider/model` naming, BYOK at $0 markup, $4 free credit) |
| S9 | our own execution on Molab, 2026-11-04 | load + decision API works; accuracy and latency measured on our held-out split (below) |

---

## 1. What is VERIFIED

- **It exists, it is public, and it downloads without credentials.** This
  contradicts the model card's own sentence "The repository is private:
  authenticate with a Hugging Face token that has access to it." Verified by
  `snapshot_download` succeeding unauthenticated on Molab (S9), and by repository
  metadata reporting `private: false`, `gated: false`. **The card text is stale
  or wrong; the artifact is open.** No access gate was bypassed — nothing was
  authenticated.
- **It is a fine-tune of a pretrained LLM.** `base_model: Qwen/Qwen3.5-4B` in the
  card, in `decision_config.json` (S2), and in `NOTICE` (S3) with a pinned
  revision. Initialization is therefore pretrained, not scratch — this is not an
  inference, it is printed in three separate files shipped with the weights.
- **It is distilled from teacher models.** The card states distillation from 27B
  decision models over teacher probability distributions, plus model souping.
  This is exactly the "teacher-generated signal" case our mission requires to be
  disclosed rather than treated as equivalent to learning from raw data (see §11).
- **License: Apache-2.0**, shipped in full (S4), with the upstream Qwen3.5-4B
  chain also Apache-2.0 per the NOTICE. Commercial use is permitted by the text
  as published.
- **Decision primitives** (card + `protocol.py` behaviour): `choice` over 1–255
  options, `noul` = P(not understood) binary, `score` over 2–255 ordinal levels;
  returns per-choice probabilities and a `confidence`. A BF16 `[255, 2560]`
  readout head decodes at `<|fim_suffix|>` markers, `readout_offset: 0` (S5).
- **The readout is non-causal by construction.** `attention_mode:
  noncausal_full_attention` with causal linear-attention layers elsewhere. The
  card states default causal text-generation inference does **not** reproduce its
  decision behaviour — so any evaluation through an ordinary generate() call
  measures a different object. We therefore used its own `Decider` class, not a
  text shim.
- **Multiple questions can share one request** (`joint_config.json`,
  `max_questions: null`), with the card's own measured warning that joint
  batching *hurts* (BFCL −15.4, ACOS −20.1) and *helps* some suites (SATA-Bench
  +6.6).
- **It is multimodal**: `max_pixels: 1638400`, images occupy ~1600 positions,
  card reports vision Decision Index 72.53 for Flash versus 77.43 for the 27B.
- **Reference deployment needs a CUDA GPU, not a Mac.** ~9 GiB BF16 weights;
  card's own reference figures are ~39 GiB / ~1 min and ~54 GiB / 20 s on a
  GB200. Our load succeeded only on the RTX PRO 6000 Blackwell.
- **It runs on our harness today** (S9). Loaded with the host's existing torch
  2.11.0 / transformers 5.14.1 (the repo pins 2.14.0 / 5.17.0 — version skew was
  tolerated, not validated). Its fast path reported: `flash-linear-attention`
  absent, "Falling back to torch implementation". So the numbers below are the
  **reference path, not the fast path**.

## 2. What is CLAIMED by the vendor, not independently established

- Distillation recipe, teacher identities ("27B decision models"), soup
  composition, and the data mixture. The card cites tasksource data but does not
  publish the training mixture or per-source licenses.
- All Decision Index 0.3 numbers: Flash 58.88 joint / 60.05 single-question,
  27B 68.09, vision 72.53 vs 77.43. This is the vendor's own benchmark, with a
  methodology we have not reproduced. It is **not** comparable to our envelope.
- Dev-set calibration: 1,497 questions, accuracy 88.8%, ECE 0.021, Brier 0.163.
  Plausible and specific, but unverified by us.
- Context: `max_length` 262,144 default, trained to ~32k, validated to ~243k
  tokens with P(correct) > 0.999 "spot-checked". "Trained context" and "accepted
  context" are therefore ~32k and ~243k respectively — an 8× gap the card itself
  implies but does not measure end-to-end. Our own context work already showed
  that nominal window ≫ usable evidence (§10).
- Throughput/latency on consumer hardware, quantization behaviour, and any
  memory figure below 9 GiB. None published that we found.

## 3. The listing, verified AS-LISTED — and what that still does not establish

The AnyRouter catalog API (queried from Molab, 2026-11-04; the per-model detail
endpoint returns the same record) **does** carry `perplexity/pplx-decider-flash-v1.2`,
alongside `pplx-decider-v1-27b`, `pplx-decider-v1.1-27b` (the likely 27B teachers),
and two `pplx-embed` models. The listing states, as its description: the model is
"Perplexity's on-device decision model, run by the Decider for Mac app on your
own paired Mac and relayed to AnyRouter over the outbound relay connection";
"Only your own Mac serves your own requests; it is never shared with other
users"; it "speaks the Decisions API contract: send application state and
typed questions (noul / choice / score) and get back a probability distribution";
route is `POST /api/v1/decisions` (alias `/api/v1/systemone`). Pricing fields are
all `"0"`, `"billing": "free"`; the single upstream route declares
`"data_policy": {"zdr": true}`. Request parameters are `["state", "model",
"questions"]` — which corroborates the HF `protocol.py` request shape
cross-source. Capability tag is `"systemone"`, not chat/completions.

That upgrades the listing claims from "not found" to **listing-verified** (S8
now = catalog API + detail endpoint, not just the landing page). What it does
**not** do is establish any of the underlying facts: the existence or behaviour
of the Mac app, that inference actually executes on-device rather than being
fronted, what quantization the on-device path uses and what it costs in accuracy,
or that only the owner's Mac ever sees the traffic. We have no Mac, no paired
device, no download page, and no traffic to inspect — and we will not sign up,
install, or pair anything to get them (no new agreements). So the relay
architecture is *their stated design*, and the "never shared" / ZDR properties
are *their stated policy*, not our measurements. The upstream repo, the only
artifact we can execute, remains CUDA-oriented code with no on-device path.

Minor discrepancy for the record: the catalog says `created: 1792713600`
(~2026-06-23); the HF card page showed an October 2026 creation. One of the two
is stale; it does not affect any measurement.
- **Whether a 4.54B BF16 model can meet any interactive on-device budget on
  Apple silicon**, and at what quantization, and with what decision-accuracy
  loss. The upstream repo gives no path for this. Note the tension: the card's
  own reference path is CUDA-only; a Mac deployment would be a *re-implementation*
  whose fidelity to the released readout is unverified.
- Commercial terms beyond the Apache-2.0 text (support, indemnity, redistribution
  of the readout head) — unknown.
- Any independent third-party evaluation on standard decision benchmarks.

## 4. Matched evidence against Jev

**There is none yet.** We have no pplx-decider measurement on Jev's evaluation
set, at Jev's model version, with Jev's adaptation access. What we do have is a
matched-to-our-own-envelope measurement: same held-out rows, same candidate
labels, same scoring as our funnel cells — but **zero-shot**, while our models
were fine-tuned on 2,000 rows. Contamination is uncontrolled in both directions:
Qwen3.5-4B's pretraining almost certainly saw Banking77-style intent text, and
the card's distillation teachers may have seen the same suites. Any pplx number
on a public classification task is an **upper-bounded, contamination-unknown**
figure. Do not report it as a clean generalization estimate.

---

## 5. Does "local decision model" remain sufficient differentiation?

**No — and it was already the weakest part of our pitch.** A competitor that
claims on-device inference removes "runs locally" as a differentiator even before
anyone verifies the claim, because the market comparison will be made against the
claim. Differentiation has to move to properties a listing cannot fake:

| Claimed differentiator | Status after this update |
|---|---|
| "Local / on-device" | **No longer distinctive.** Claimed by the competitor (unverified), and the upstream artifact is a 9 GiB CUDA model. |
| "Decision model, not a chatbot" | **No longer distinctive.** `choice`/`noul`/`score` with probabilities is a more complete decision API than ours. |
| "Calibrated uncertainty" | **No longer distinctive by assertion.** They publish ECE 0.021. We have never measured our own ECE. This is now a **gap against us**, not a feature of ours. |
| "Long context" | **Not a differentiator we can win.** They claim ~243k validated. Our evidence use is the binding constraint, not window size. |
| **Scratch-trained from raw data** | **Still distinctive, and now the primary one.** Verifiable: no pretrained init, no teacher distribution, no soup. Their NOTICE file is the exact opposite claim. |
| **Footprint** | **Still distinctive, now dramatically so.** 4.54B vs our ~14M is ~324× fewer parameters; ours runs on CPU at ~50 ms with no CUDA and no relay. |
| **No relay, no account, no network** | **Still distinctive** *if* the relay claim is true and unavoidable in their product. Unverified — do not lean on it as fact. |
| **Independence of the Qwen chain** | **Still distinctive.** Our license exposure is our own code and data; theirs inherits an upstream model lineage. |

The honest new framing: *"a scratch-trained decision model that is ~324× smaller,
runs offline on commodity CPUs at tens of milliseconds, and has no pretrained
lineage or teacher distillation anywhere in it"* — with the load-bearing word
being **scratch**, because local/small/calibrated are now contested.

## 6. What changes in the comparison envelope

Additive changes only; no success criterion is relaxed and no bar moves.

1. **New row: `pplx-decider-flash-v1.2` (zero-shot, its own `Decider` API).**
   Columns: accuracy, Wilson LB, mean confidence, s/request, params, license,
   pretrained?, distilled?, relay required?
2. **Calibration becomes a first-class metric** (ECE + Brier) for every future
   cell, not just accuracy. Without it we cannot answer the competitor's
   ECE 0.021 claim at all.
3. **Primitive coverage matrix**: choice (1–255), noul, score (2–255). We
   currently cover choice only; `noul` partially; `score` not at all.
4. **Multi-question batching cost** recorded separately from single-question
   accuracy, because the competitor's own data shows batching is not free
   (BFCL −15.4).
5. **Deployment-truth columns**: device, quantized?, network required?,
   parameter count. This is where we actually win and it is currently absent.
6. **Adaptation disclosure is mandatory on every row** (fine-tuned on N rows vs
   zero-shot vs few-shot). The pplx row is zero-shot and must be labelled so in
   every table, or the comparison is dishonest.
7. **Image decisions**: deferred. It is a real capability gap, but adding a
   vision path is a new research programme, not an envelope column, and it does
   not serve the binding constraint.

## 7. What changes in the next scratch-training experiments

Ordered by expected information value, and each one is an experiment, not a
document:

1. **Evidence pooling: mean → max (RUNNING NOW).** The competitor's config uses
   `pooling: last` on a non-causal readout — it does not average evidence, it
   reads a position that can attend to everything. Our passage encoder averages
   over all span tokens, which is why a single decisive sentence is diluted by
   hundreds of irrelevant ones (this is the mechanism behind the context deficit:
   0.91 intent vs 0.62–0.69 document-grounded). Max-pooling is the cheap,
   architecture-level test of that hypothesis, and it **transfers directly to a
   scratch-trained encoder** because it is a pooling choice, not a pretrained
   benefit. Acceptance is not "did boolq go up" but "does the real-minus-shuffled
   gap become consistently positive", the same control that already caught an
   unstable result.
2. **Calibration on our own models (next).** We cannot claim uncertainty
   advantage and have never measured ECE. Temperature scaling fit on validation
   only, reported alongside raw. Cheap, no new data, and it closes a
   competitor-published gap.
3. **`score` primitive on the scratch head (queued).** An ordinal head over K
   levels is a small architectural addition and is genuine capability parity. It
   is *not* urgent: no user has asked for ordinal decisions yet, and it must not
   displace the context work.
4. **Do NOT chase 243k context.** Their long-context claim is a nominal window;
   our shuffled control shows our failure is evidence *use*, not window size.
   Chasing tokens would burn GPU on the wrong constraint.
5. **Do NOT adopt teacher distillation.** Explicitly considered and declined as a
   default: it would forfeit the one differentiator that survives this update.
   If ever revisited, it must be disclosed as a separate methodological track with
   licensing and access terms, never presented as scratch learning.

## 8. What remains unchanged, and why

- **Scratch training requirement.** The competitor's own NOTICE file
  ("Derived from Qwen/Qwen3.5-4B") is precisely what we refuse to become. Its
  success, where it exists, is evidence about pretraining and distillation, not
  permission to abandon the objective.
- **Bars** (banking77 0.797, prompt_injections 0.741, emotion 0.585,
  financial_phrasebank 0.73), **split hygiene** (split 0 dev-only; 99 and 123
  spent as confirmatory), and **no post-hoc selection** — none of these move
  because a competitor appeared.
- **Tracks stay separate.** Track A (end-product under realistic deployment)
  gains a row. Track B (attribution: scratch vs pretrained/distilled) gains a
  documented contrast: 4.54B pretrained+distilled+vision+long-context versus
  ~14M scratch, with initialization, data, compute and adaptation disclosed for
  both. Their product performance is not an excuse to ignore our scratch run;
  their pretrained origin is not an excuse to ignore their product performance.
- **Molab-only compute.** Nothing here ran locally. The competitor baseline ran
  on the Molab GPU; the laptop carried only transfers.
- **The speed result.** Jev 283 ms vs our 49.4 ms (K=77, CPU) stands on its own
  measurement and is untouched by a listing claiming zero token charge — those
  are different quantities (charge ≠ latency) and conflating them would be
  exactly the error this document exists to prevent.

---

## 9. Measured baseline (our rows, our harness, competitor's own API)

Full detail, environment table and caveats in `PPLX_BASELINE.md`. Headline facts
established by execution rather than by reading:

- The released artifact loads and decides **unauthenticated**, through its own
  `Decider` path. The decision path used was the **reference torch path** —
  `flash-linear-attention` was absent, so its fast path never ran.
- banking77, untouched split 123, n=1000, K=77: **0.906 accuracy, Wilson LB
  0.8863, 0.151 s/request**. Our v11 confirmatory cell on the same rows: 0.910 /
  0.919, LB 0.8907 / 0.9004, at ~14 M params and 49.4 ms CPU. **A tie on accuracy
  at ~324× the parameters and 3–7× the latency, achieved zero-shot.**
- That zero-shot property is the competitor's genuinely uncomfortable advantage:
  our deployment story needs 2,000 labelled in-domain rows, theirs needs none.
- On `prompt_injections` (n=116, **our own synthesised data**, so a contamination
  probe by construction): **0.9397, LB 0.8807, 0.022 s/request** — zero-shot.
  Our fine-tuned seeds are 0.974 / 0.948; the bar is 0.741.
- **That result partially refutes my own contamination hypothesis, and it is
  recorded that way.** Pretraining recall cannot explain 0.94 on rows we
  generated. On this suite the competitor is genuinely deciding, not remembering.
  Banking77's 0.906 stays contamination-unknown, but the model is no longer
  dismissible as a memoriser, and I am not going to present it as one to make our
  position look better.
- **The measurement exposed a gap on our side**: our own ECE on boolq context
  cells is 0.33–0.37 against their published 0.021 (different task, not
  head-to-head, but not a rounding difference). Confidence-gated routing is
  unsupported until we fix calibration, which is now the top queued experiment.
- Evidence pooling was tested as a consequence of this update and **rejected**:
  max pooling at matched batch 16 is *worse* than mean on both suites (boolq
  0.608/0.576 vs 0.638; squad2 0.652 vs 0.694/0.686) and does not reliably widen the
  real-minus-shuffled gap. Rejected with a replication. The competitor's mechanism is a
  non-causal learned readout position (`pooling: last`), not a fixed reduction,
  and that is the next architecture test.
