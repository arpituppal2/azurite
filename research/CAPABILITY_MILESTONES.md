# Capability milestones — long context and visual evidence

**Date:** 2026-10-30.
**Status:** planning gates, not results. Dates are planning targets; a missed
target is recorded with its blocker, not silently moved.

These milestones exist because “long context” and “multimodal” cannot remain
open-ended aspirations. Each gate has an entry condition, a measurable pass
rule, a stop rule, and a compute cap. A gate is passed only by stored,
hash-verifiable records; prose is not evidence.

## 1. Long context

Long context means **preserved decision quality**, not a larger positional
buffer. Every arm reports accuracy at the tested lengths, evidence-position
gaps, distractor sensitivity, distributed-evidence accuracy, peak memory, and
p50/p95 latency. A longer accepted window with worse decisions is a negative
result.

| gate | target | entry condition | pass rule | budget / status |
|---|---|---|---|---|
| **LC0** | Fixed-checkpoint curve, 256→4,096 | 14.4M checkpoint, dev rows | Report the curve and apply the pre-registered labels | **Done 2026-10-30; negative.** 0 additive, 4 degraded; ContractNLI 69.79→37.03. |
| **LC1** | Length-adapted 4k | LC0 recorded; rows whose full document fits 4k identified | On the full-context subset, accuracy within **2.0 pp** of the 256 baseline; position gap ≤5 pp; ≤2 pp drop with 8 distractors; memory and latency reported. Report subset coverage. | 2–4 projected GPU-h; target 2026-11-15. |
| **LC2** | 16k full-window | LC1 passes on its full-context subset | Accuracy within **2.0 pp** of the 1,024-byte arm on the same rows; position gap ≤5 pp; no collapse at 0/2/8 distractors; distributed-evidence and no-evidence controls; memory/latency at 16k. | 12.6–15.9 projected GPU-h; target 2026-12-15, conditional on LC1. |
| **LC3** | 32k | LC2 passes | Accuracy within **2.0 pp** of the 16k arm; same position/distractor controls; memory/latency at 32k. A chunked path is a separate hypothesis and needs its own quote. | 39.8–56.2 projected GPU-h for full-window; target 2027-01-31, conditional. |
| **LC4** | Confirmatory context claim | LC2 or LC3 passes; suite v2 with training/dev/calibration/final separation | Three seeds, pre-registered primary contrast, locked evaluation opened once. No claim before this. | Projected after LC2/LC3; no silent extension. |

**Stop rules.** A negative LC1 stops LC2. A negative LC2 stops LC3. No 16k/32k
spend is authorised by a flat or degraded fixed-checkpoint curve. Partial-context
rows are labelled as such; they cannot establish a full-document claim.

## 2. Visual evidence

A visual task is admissible only when the answer changes with the image and
cannot be recovered from OCR alone. A document screenshot plus OCR plus a text
model is a text pipeline. Every visual number is paired with an OCR-only
control, and a blank-image and shuffled-image control.

| gate | target | entry condition | pass rule | budget / status |
|---|---|---|---|---|
| **V0** | VisualEvidence development instrument | Locally rendered documents and charts; gold answer, evidence box, OCR transcript | At least 200 development items across checkboxes, tables, stamps/signatures, and chart trends; template/document-disjoint train/dev split; image hashes and provenance recorded. | CPU/IO; target 2026-11-15. |
| **V1** | Controls | V0 frozen | OCR-only and text-only baselines measured on the same rows; visual-only subset identified where OCR is insufficient; no model claim yet. | CPU/IO; target 2026-11-15. |
| **V2** | Adapter | V1 recorded; frozen open-licence image encoder with provenance hash | Visual-only accuracy ≥**70%** and ≥**+10 pp** over OCR-only; citation IoU ≥0.50; peak VRAM/RSS and p50/p95 latency reported. If OCR-only is within 5 pp, withdraw the visual claim. | 4–8 projected GPU-h; target 2026-12-15. |
| **V3** | Competitive visual arm | V2 passes | Visual-only accuracy ≥**80%** and ≥**+15 pp** over OCR-only; citation IoU ≥0.65; three seeds; latency/throughput/memory reported. Public benchmarks only after licence and contamination gates. | 8–12 projected GPU-h; target 2027-01-31, conditional. |
| **V4** | Confirmatory visual claim | V3 passes; suite v2 final split | Locked evaluation opened once; OCR-only, text-only, blank, and shuffled controls repeated; claim limited to the measured distribution. | Projected after V3; no silent extension. |

**Stop rules.** If V1 shows OCR-only is within 5 points of the visual arm, the
visual claim is withdrawn regardless of absolute accuracy. If V2 fails, do not
add a larger image encoder; diagnose the task or the adapter first. A visual
model that only reproduces OCR is not multimodal evidence processing.

## 3. Measurement axes

Every substantial result reports all applicable rows. “Not measured” is a
reported limitation, not a blank cell.

| axis | required measurement | where it applies |
|---|---|---|
| **Quality** | Accuracy, macro-F1, answer F1, answerability, citation IoU; per-task and macro | all families |
| **Calibration** | ECE, Brier score, reliability curve, calibration artifact id | classification, boolean, ordinal, abstention |
| **Abstention risk/coverage** | Risk–coverage curve, selective accuracy at 70/80/90% coverage, unanswerable abstain recall, AURC | document-grounded, abstention, visual |
| **Robustness** | Evidence position early/middle/late; 0/2/8 distractors; distributed evidence; negation/scope; paraphrase; blank/shuffled images; OCR-only control | long-context, document, visual |
| **Latency** | p50 and p95 at 1k/4k/16k/32k bytes; CPU and GPU separately | all deployable arms |
| **Throughput** | Decisions/s at batch 1 and batch 8; rows/s for evaluation | all deployable arms |
| **Memory** | Peak RSS and peak VRAM; artifact size; int8 size/accuracy drop if quantised | all deployable arms |

**Near-perfect accuracy** is a distribution-specific ambition, never a
guarantee. A 97% target is meaningful only with the named distribution,
coverage, confidence interval, and failure slices attached. No release note may
describe a model as “near-perfect” without those qualifiers.

## 4. Sequencing and anti-postponement rule

1. H1 in `research/PIVOT_REVIEW.md` was the only authorised training screen.
   It ran on 2026-10-30 (5 min 36 s, 18 GPU-hour cap) and **failed** three
   pre-registered gates; see `results/h1_stage1/RESULTS.md`. No training screen
   is currently authorised. A new proposal needs its own falsifiable mechanism,
   baselines, gates, and budget.
2. LC1 and V0/V1 are the next capability milestones. They may run after a new
   proposal is accepted, or in parallel only if they do not consume an
   authorised training budget or alter H1's rows.
3. LC2 and V2 are conditional on LC1 and V1 respectively. They are not
   “postponed indefinitely”; a missed target date is recorded in the ledger with
   the blocker and a new dated target.
4. No capability claim is made from a milestone that has not passed. A pass is
   permission for the next gate, not a product claim.

## 5. Provenance

- Product contract: `docs/PRODUCT_SPEC.md`.
- Suite hashes: `research/suites/dev_suite.json` and
  `research/suites/locked_suite.json`.
- Pivot review: `research/PIVOT_REVIEW.md`. Pre-registration text SHA-256
  `4418ad345e53cf6b239fbaac97b40c9e4bfb9f99cd4d41cda60ffa93e53ea326`; the file
  now includes a post-run outcome section, current SHA-256
  `773ce0312fbb03c57f93a9f48d68ac5ea7ac38ae480b2d3e23441d0599dfa557`.
- H1 result: `results/h1_stage1/summary.json`, SHA-256
  `70f52a4f81166cfedbaef96079fb9bcac266409af967d4441512cabcfb76dd95`.
- Context result: `results/context_scaling/RESULTS.md`, SHA-256
  `113a14db6543939b29f5c6658d4313e895866e0d01b7a23922298b934d3e2704`.
