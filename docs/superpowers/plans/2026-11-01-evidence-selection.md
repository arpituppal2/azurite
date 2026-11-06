# Evidence-Selection (H2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run H2: a document-level evidence-selection harness that compares 1 KB truncation, all-chunk pooling, and contrastive span selection on ContractNLI at 16 KB and 32 KB, entirely on an Apple M1 with 8 GB unified memory.

**Architecture:** A byte chunker splits a document into 512-byte overlapping chunks. BM25 retrieves the chunks most relevant to the instruction, question and candidate labels. A frozen 403M byte encoder embeds the retrieved chunks and the candidates. A 2.1M-parameter bilinear head scores every (span, candidate) pair. The spans with the widest top-1/top-2 spread decide, and those spans are the cited evidence. The same head is trained once on all chunks, so the three arms differ only in what they read.

**Tech Stack:** Python 3.11, PyTorch 2.11 (MPS/CPU), NumPy, pytest, local `datasets` Arrow files. No network, no pretrained weights, no tokenizer downloads.

## Global Constraints

- **Minimum hardware:** Apple M1, 8 GB unified memory, macOS 14+, `.venv/bin/python` (Python 3.11, torch 2.11). Every command below uses `.venv/bin/python`; the system Python is 3.0.1 and cannot run MPS.
- **Encoder:** `azurite.config.rung("xlarge", "byte")` — 402,972,672 parameters, `max_seq=1024`, byte vocabulary 260. Checkpoint: `results/pretrain/l1_xlarge_byte_dn1/ckpt.pt`, SHA-256 `207a8afc3732114096fa5fc5a4671d3f686231196796dc35d3da4afe89f03299`.
- **Precision:** fp16 on MPS, bf16 on CUDA, fp32 on CPU. The encoder runs in the device dtype; the 2,097,153-parameter head always runs in fp32.
- **Memory envelope:** peak process RSS ≤ **6.5 GiB**; swap growth ≤ **16 MB**; page-outs ≤ **128**. A run that pages is a failed run.
- **Windows:** 512-byte chunks, 512-byte stride, top-k 4 retrieved chunks, keep 3 decision spans, 514 token positions per span. The native 1,024-byte window is an implementation detail, not the product limit.
- **Data:** `data/raw/contract_nli_ctx/{train,test}` only. The locked suite is never read. The dev test split may be inspected; no threshold is fitted on it.
- **Protocol:** one seed (0); gates are pre-registered in Task 1 and are not edited after the first run. No post-hoc threshold, no "best of" reruns.
- **Do not modify** `azurite/retrieval.py`, `azurite/predictor.py`, `azurite/cli.py`, or the shipped wheel. The new code is additive.
- **Every task ends** with the listed test command passing and one commit. Do not batch commits across tasks.

## File Structure

| Path | Responsibility |
|---|---|
| `azurite/evidence.py` | `Chunk`, `chunk_document`, `ChunkIndex`, `SpanCandidateScorer`, `select_contrastive`, `fit_head` |
| `scripts/m1_profile.py` | Device/dtype resolution, RSS and latency measurement, 403M fit check |
| `scripts/build_evidence_docs.py` | ContractNLI 16 KB / 32 KB document views with target spans |
| `scripts/train_evidence_head.py` | Freeze encoder, train the bilinear head, save head-only checkpoint |
| `scripts/h2_evidence.py` | Three-arm harness with per-row records and summary JSON |
| `scripts/h2_report.py` | Paired deltas, Wilson intervals, McNemar, RESULTS.md |
| `scripts/h2_smoke.py` | 20-row M1 acceptance run with envelope assertions |
| `tests/test_evidence.py` | Chunking, retrieval, scorer, selection, head training |
| `tests/test_evidence_docs.py` | Document-view builder invariants |
| `tests/test_h2_evidence.py` | Harness arm behaviour and metrics |
| `tests/test_h2_report.py` | Paired statistics and report rendering |
| `tests/test_m1_profile.py` | Percentiles, RSS parsing, envelope checks |
| `results/h2_evidence/PRE_REGISTRATION.md` | Frozen gates and hashes |
| `results/h2_evidence/docs_*.json.gz` | Materialised 16 KB / 32 KB views |
| `results/h2_evidence/head.npz` | Head-only checkpoint + provenance |
| `results/h2_evidence/records_*.json` | Per-row arm records |
| `results/h2_evidence/RESULTS.md` | Final table and verdict |

`azurite/evidence.py` is one responsibility: everything between "a document" and "a cited decision." The scripts are thin: load, loop, write. No model code in scripts.

---

## Pre-registration (frozen before the first run)

| Gate | H2 passes only if |
|---|---|
| Primary accuracy | Contrastive ≥ truncate + **10.0 pp** at 16 KB, paired McNemar p < 0.05, paired bootstrap 95% lower bound > 0 |
| Selection efficiency | Contrastive ≥ chunk-pool − **2.0 pp** at 16 KB while encoding ≤ **25%** of document bytes |
| Retrieval | Target chunk appears in the top-4 for ≥ **95%** of rows at 16 KB |
| Evidence precision | ≥ **80%** of rows have at least one selected span with ≥ 50% of its bytes inside the target span |
| 32 KB | Same direction as 16 KB; reported separately, not pooled |
| Envelope | Peak RSS ≤ 6.5 GiB, zero swap, p95 ≤ 5,000 ms on the M1 |

**Falsification:** if the 16 KB accuracy gate fails, retrieval-first evidence selection is not the mechanism; the next proposal is a hierarchical 4 KB encoder, not a rerun with a different k. If retrieval hit rate is below 95%, fix retrieval before scoring. If the envelope fails, reduce spans before reducing precision.

**Not tested here:** calibration, abstention, install-time adaptation, arbitrary label sets, and Jev comparison. Those are H3/H4 and separate plans.

---

### Task 1: Runtime envelope and pre-registration

**Files:**
- Create: `results/h2_evidence/PRE_REGISTRATION.md`
- Create: `scripts/__init__.py`
- Create: `scripts/m1_profile.py`
- Create: `tests/test_m1_profile.py`
- Modify: `pyproject.toml` (add the `slow` marker)

**Interfaces:**
- Consumes: `azurite.config.EncoderConfig`, `azurite.config.rung`, `azurite.model.Azurite`, `azurite.env.swap_used_mb`, `azurite.env.pageouts`.
- Produces: `resolve_dtype(device: str, requested: str) -> torch.dtype`; `percentile(values: list[float], q: float) -> float`; `peak_rss_gib() -> float`; `profile_checkpoint(ckpt: Path, *, device: str, dtype: str, seq: int, batch: int, iters: int) -> dict`; `check_envelope(report: dict, *, max_rss_gib: float = 6.5, max_p95_ms: float = 5000.0) -> list[str]`.

- [ ] **Step 1: Create the package marker and write the pre-registration file**

Create an empty `scripts/__init__.py`, then create `results/h2_evidence/PRE_REGISTRATION.md` with exactly this content and never edit it:

```markdown
# H2 pre-registration — contrastive evidence selection

Frozen: 2026-11-01, before any H2 run.

## Question

At 16 KB and 32 KB of document context, does selecting the spans that separate
the candidate answers beat (a) truncating to 1 KB and (b) pooling every chunk,
using the same frozen encoder and the same trained head?

## Arms

- `truncate`: question + first 1,022 bytes of the document, one span.
- `chunk_pool`: every 512-byte chunk (max 64), mean-pooled, scored.
- `contrastive`: BM25 top-4 chunks by instruction + question + candidates,
  scored, keep the 3 spans with the widest top-1/top-2 margin, max-aggregate.

## Gates

- Contrastive ≥ truncate + 10.0 pp at 16 KB; McNemar p < 0.05; bootstrap
  lower 95% > 0.
- Contrastive ≥ chunk_pool − 2.0 pp at 16 KB while encoding ≤ 25% of bytes.
- Retrieval hit (target in top-4) ≥ 95% at 16 KB.
- Selected-evidence precision ≥ 80% of rows at 16 KB.
- 32 KB reported separately with the same direction; no pooling.
- M1 envelope: peak RSS ≤ 6.5 GiB, swap delta ≤ 16 MB, p95 ≤ 5,000 ms.

## Data and seed

- `data/raw/contract_nli_ctx/train` for head training.
- `data/raw/contract_nli_ctx/test` for evaluation, 748 rows, seed 0.
- No locked-suite rows. No threshold fitted on test.

## Falsification

Failure of the 16 KB accuracy gate stops the encoder-replacement direction.
The next proposal is a hierarchical 4 KB encoder, not a rerun at another k.
Retrieval hit rate below 95% is a retrieval bug, fixed before scoring.
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_m1_profile.py`:

```python
from __future__ import annotations

import pytest

from scripts.m1_profile import (
    check_envelope,
    peak_rss_gib,
    percentile,
    resolve_dtype,
)


def test_percentile_interpolates_between_ranks() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 95.0) == pytest.approx(3.85)
    assert percentile([7.0], 95.0) == 7.0
    assert percentile([], 50.0) != percentile([], 50.0)  # NaN for empty


def test_peak_rss_is_positive_gib() -> None:
    assert 0.0 < peak_rss_gib() < 1024.0


def test_dtype_resolution_is_explicit() -> None:
    import torch

    assert resolve_dtype("cpu", "auto") is torch.float32
    assert resolve_dtype("mps", "auto") is torch.float16
    assert resolve_dtype("cuda", "auto") is torch.bfloat16
    assert resolve_dtype("cpu", "fp16") is torch.float16


def test_envelope_reports_every_violation() -> None:
    bad = {
        "peak_rss_gib": 7.1,
        "swap_delta_mb": 40.0,
        "pageout_delta": 300,
        "p95_ms": 6200.0,
    }
    problems = check_envelope(bad)
    assert len(problems) == 4
    assert any("RSS" in p for p in problems)
    assert any("swap" in p for p in problems)
    assert any("page" in p for p in problems)
    assert any("p95" in p for p in problems)

    good = {
        "peak_rss_gib": 5.4,
        "swap_delta_mb": 0.0,
        "pageout_delta": 0,
        "p95_ms": 2400.0,
    }
    assert check_envelope(good) == []
```

- [ ] **Step 3: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_m1_profile.py -q
```

Expected: `ModuleNotFoundError: No module named 'scripts.m1_profile'` (or an import error for the missing functions).

- [ ] **Step 4: Implement `scripts/m1_profile.py`**

```python
#!/usr/bin/env python3
"""Measure the M1 envelope for the 403M encoder and the H2 harness.

The numbers this writes are the acceptance evidence for "runs on an 8 GB M1".
It measures live process RSS, not the allocator's reservation: on MPS the
reservation includes freed blocks and would overstate the requirement.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.config import EncoderConfig
from azurite.env import pageouts, swap_used_mb
from azurite.model import Azurite


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def peak_rss_gib() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    bytes_used = raw if sys.platform == "darwin" else raw * 1024
    return bytes_used / 2**30


def resolve_dtype(device: str, requested: str) -> torch.dtype:
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp32":
        return torch.float32
    if requested != "auto":
        raise ValueError(f"unknown dtype {requested!r}")
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def profile_checkpoint(
    ckpt: Path,
    *,
    device: str,
    dtype: str,
    seq: int = 514,
    batch: int = 4,
    iters: int = 10,
) -> dict:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["config"])
    model = Azurite(cfg).eval()
    model.load_state_dict(payload["model"], strict=True)
    chosen = resolve_dtype(device, dtype)
    model.to(device=device, dtype=chosen)
    start_swap = swap_used_mb()
    start_pages = pageouts()

    ids = torch.randint(4, cfg.vocab_size, (batch, seq), device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    timings: list[float] = []
    with torch.no_grad():
        for i in range(iters):
            _sync(device)
            started = time.perf_counter()
            hidden = model(ids, mask)
            _sync(device)
            timings.append((time.perf_counter() - started) * 1000.0)
            if not torch.isfinite(hidden).all():
                raise RuntimeError(
                    f"non-finite activations in {chosen} on {device}; use --dtype fp32"
                )
    return {
        "device": device,
        "dtype": str(chosen).replace("torch.", ""),
        "parameters": sum(p.numel() for p in model.parameters()),
        "seq": seq,
        "batch": batch,
        "p50_ms": round(percentile(timings, 50.0), 2),
        "p95_ms": round(percentile(timings, 95.0), 2),
        "peak_rss_gib": round(peak_rss_gib(), 3),
        "swap_delta_mb": round(swap_used_mb() - start_swap, 1),
        "pageout_delta": pageouts() - start_pages,
        "checkpoint": str(ckpt),
    }


def check_envelope(
    report: dict,
    *,
    max_rss_gib: float = 6.5,
    max_p95_ms: float = 5000.0,
) -> list[str]:
    problems: list[str] = []
    if report.get("peak_rss_gib", 0.0) > max_rss_gib:
        problems.append(
            f"RSS {report['peak_rss_gib']:.2f} GiB > {max_rss_gib:.2f} GiB"
        )
    if report.get("swap_delta_mb", 0.0) > 16.0:
        problems.append(f"swap grew {report['swap_delta_mb']:.1f} MB")
    if report.get("pageout_delta", 0) > 128:
        problems.append(f"{report['pageout_delta']} page-outs")
    if report.get("p95_ms", 0.0) > max_p95_ms:
        problems.append(f"p95 {report['p95_ms']:.0f} ms > {max_p95_ms:.0f} ms")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/pretrain/l1_xlarge_byte_dn1/ckpt.pt")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--seq", type=int, default=514)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="results/m1_profile/profile.json")
    args = ap.parse_args()

    report = profile_checkpoint(
        Path(args.ckpt),
        device=args.device,
        dtype=args.dtype,
        seq=args.seq,
        batch=args.batch,
        iters=args.iters,
    )
    problems = check_envelope(report)
    report["problems"] = problems
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: runs on real hardware or a real checkpoint"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_m1_profile.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Measure the real 403M checkpoint**

Run:

```bash
.venv/bin/python scripts/m1_profile.py --device mps --dtype auto \
  --out results/m1_profile/profile.json
```

Expected: exit 0, `peak_rss_gib` ≤ 6.5, `p95_ms` recorded, no `problems`. If fp16 produces non-finite activations, rerun with `--dtype fp32` and record that in `results/m1_profile/profile.json`; do not silently switch.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml scripts/__init__.py scripts/m1_profile.py tests/test_m1_profile.py \
  results/h2_evidence/PRE_REGISTRATION.md results/m1_profile/profile.json
git commit -m "feat(evidence): freeze H2 protocol and measure the M1 envelope"
```

---

### Task 2: Byte chunking and BM25 chunk retrieval

**Files:**
- Create: `azurite/evidence.py`
- Create: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `azurite.retrieval.BM25`.
- Produces: `Chunk(index: int, start: int, end: int, text: str)`; `chunk_document(text: str, *, size: int = 512, stride: int = 384) -> list[Chunk]`; `retrieval_query(instruction: str, question: str, candidates: list[str]) -> str`; `ChunkIndex(chunks: list[Chunk])` with `.search(query: str, k: int = 4) -> list[Chunk]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evidence.py`:

```python
from __future__ import annotations

import pytest

from azurite.evidence import ChunkIndex, chunk_document, retrieval_query


def test_chunks_cover_every_byte_with_overlap() -> None:
    text = ("aé日本語 " * 200).strip()
    raw = text.encode("utf-8")
    chunks = chunk_document(text, size=64, stride=48)
    assert chunks
    covered = set()
    for chunk in chunks:
        assert chunk.start < chunk.end
        assert chunk.end - chunk.start <= 64
        covered.update(range(chunk.start, chunk.end))
    assert covered == set(range(len(raw)))
    assert "".join(chunk.text for chunk in chunks).count("日本語") > 0


def test_chunks_decode_strictly_and_respect_utf8_boundaries() -> None:
    text = "é" * 500
    for chunk in chunk_document(text, size=17, stride=11):
        chunk.text.encode("utf-8")  # must not contain replacement characters
        assert "\ufffd" not in chunk.text


def test_chunking_is_deterministic_and_rejects_bad_windows() -> None:
    text = "one two three " * 100
    first = chunk_document(text, size=64, stride=32)
    second = chunk_document(text, size=64, stride=32)
    assert first == second
    with pytest.raises(ValueError):
        chunk_document(text, size=0)
    with pytest.raises(ValueError):
        chunk_document(text, size=64, stride=65)


def test_empty_and_whitespace_documents_have_no_chunks() -> None:
    assert chunk_document("") == []
    assert chunk_document("   \n\t") == []


def test_bm25_finds_the_needle_chunk() -> None:
    document = "\n".join(
        [f"filler paragraph number {i} about nothing in particular" for i in range(30)]
        + ["the termination fee is payable within thirty days of notice"]
        + [f"more filler paragraph {i}" for i in range(30)]
    )
    chunks = chunk_document(document, size=96, stride=64)
    index = ChunkIndex(chunks)
    hits = index.search("termination fee notice", k=3)
    assert hits
    assert any("termination fee" in hit.text for hit in hits)


def test_retrieval_query_contains_every_candidate_and_the_question() -> None:
    query = retrieval_query(
        "Choose the relation.",
        "Does the contract allow assignment?",
        ["entailment", "contradiction", "not_mentioned"],
    )
    for token in ["Choose", "assignment", "entailment", "contradiction", "not_mentioned"]:
        assert token in query


def test_search_k_is_bounded_by_the_number_of_chunks() -> None:
    chunks = chunk_document("alpha beta gamma", size=8, stride=6)
    index = ChunkIndex(chunks)
    assert len(index.search("alpha", k=99)) == len(chunks)
    assert index.search("alpha", k=0) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_evidence.py -q
```

Expected: `ModuleNotFoundError: No module named 'azurite.evidence'`.

- [ ] **Step 3: Implement chunking and retrieval in `azurite/evidence.py`**

```python
"""Document evidence selection: chunk, retrieve, score, cite.

The product question is not "what is this document about"; it is "which
passage separates these candidate answers". This module owns the first half:
turning a document into byte-exact chunks and retrieving the chunks that could
matter for a given candidate set. Scoring and selection follow in the same
module so the harness cannot accidentally use a different chunker.
"""
from __future__ import annotations

from dataclasses import dataclass

from .retrieval import BM25


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    text: str


def _continuation(byte: int) -> bool:
    return (byte & 0xC0) == 0x80


def _ceil_boundary(raw: bytes, pos: int) -> int:
    pos = min(max(pos, 0), len(raw))
    while pos < len(raw) and _continuation(raw[pos]):
        pos += 1
    return pos


def _floor_boundary(raw: bytes, pos: int) -> int:
    pos = min(max(pos, 0), len(raw))
    while pos > 0 and _continuation(raw[pos]):
        pos -= 1
    return pos


def chunk_document(
    text: str, *, size: int = 512, stride: int = 512
) -> list[Chunk]:
    """Split `text` into overlapping byte windows on UTF-8 boundaries.

    `start` and `end` are byte offsets into the UTF-8 encoding, so a selected
    chunk can be cited in the original document without re-tokenising.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if stride <= 0 or stride > size:
        raise ValueError("stride must be positive and no larger than size")
    raw = text.encode("utf-8")
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(raw):
        end = _ceil_boundary(raw, start + size)
        piece = raw[start:end].decode("utf-8", errors="strict")
        if piece.strip():
            chunks.append(Chunk(index=index, start=start, end=end, text=piece))
            index += 1
        if end >= len(raw):
            break
        nxt = _floor_boundary(raw, start + stride)
        start = nxt if nxt > start else end
    return chunks


def retrieval_query(instruction: str, question: str, candidates: list[str]) -> str:
    return " ".join([instruction, question, *candidates]).strip()


class ChunkIndex:
    """BM25 over chunks, using the shipped retriever's exact formula."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._rows = [{"id": c.index, "text": c.text} for c in self.chunks]
        self._bm25 = BM25(self._rows)

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        if k <= 0 or not self.chunks:
            return []
        order = self._bm25.rank(query)
        return [self.chunks[i] for i in order[:k]]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_evidence.py -q
```

Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add azurite/evidence.py tests/test_evidence.py
git commit -m "feat(evidence): add byte-exact chunking and BM25 chunk retrieval"
```

---

### Task 3: Span-candidate scorer and contrastive selection

**Files:**
- Modify: `azurite/evidence.py`
- Modify: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `azurite.model.Azurite`.
- Produces: `SpanCandidateScorer(encoder: Azurite)` with `.encode(ids, mask)`, `.encode_candidates(ids, cand_token_mask)`, `.score_embeddings(span_emb, cand_emb, cand_valid=None)`, `.score_pooled(span_emb, cand_emb, cand_valid=None)`, `.forward(span_ids, span_mask, cand_ids, cand_token_mask, cand_valid=None)`; `select_contrastive(logits, keep=3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]` returning `(decision_logits, selected_indices, spread)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evidence.py`:

```python
import math

import torch

from azurite.config import rung
from azurite.evidence import SpanCandidateScorer, select_contrastive
from azurite.model import Azurite
from azurite.tokenizers import CLS_ID, PAD_ID, SEP_ID, ByteTokenizer

TOK = ByteTokenizer()


def _ids(text: str, width: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    body = TOK.encode(text)[: width - 2]
    ids = [CLS_ID, *body, SEP_ID]
    mask = torch.ones(len(ids), dtype=torch.bool)
    pad = width - len(ids)
    return (
        torch.tensor([ids + [PAD_ID] * pad]),
        torch.tensor([mask.tolist() + [False] * pad]),
    )


def test_scorer_shapes_and_invalid_candidates() -> None:
    torch.manual_seed(0)
    model = SpanCandidateScorer(Azurite(rung("tiny", "byte"))).eval()
    span_ids = torch.cat([_ids("the fee is thirty days")[0] for _ in range(3)], dim=0)
    span_mask = torch.ones_like(span_ids, dtype=torch.bool)
    cand_ids = torch.cat([_ids("alpha")[0], _ids("beta")[0], _ids("gamma")[0]], dim=0)
    cand_token_mask = torch.ones_like(cand_ids, dtype=torch.bool)
    cand_valid = torch.tensor([True, True, False])
    logits = model(span_ids, span_mask, cand_ids, cand_token_mask, cand_valid)
    assert logits.shape == (3, 3)
    assert torch.isneginf(logits[:, 2]).all()
    assert torch.isfinite(logits[:, :2]).all()


def test_candidate_permutation_permutes_logits() -> None:
    torch.manual_seed(1)
    model = SpanCandidateScorer(Azurite(rung("tiny", "byte"))).eval()
    span_ids = torch.cat([_ids("alpha evidence")[0], _ids("beta evidence")[0]], dim=0)
    span_mask = torch.ones_like(span_ids, dtype=torch.bool)
    a = _ids("alpha")[0]
    b = _ids("beta")[0]
    c = _ids("gamma")[0]
    cand_ids = torch.cat([a, b, c], dim=0)
    cand_token_mask = torch.ones_like(cand_ids, dtype=torch.bool)
    cand_valid = torch.ones(3, dtype=torch.bool)
    first = model(span_ids, span_mask, cand_ids, cand_token_mask, cand_valid)
    order = torch.tensor([2, 0, 1])
    second = model(span_ids, span_mask, cand_ids[order], cand_token_mask, cand_valid)
    assert torch.allclose(first[:, order], second, atol=1e-5)


def test_contrastive_selection_keeps_the_separating_spans() -> None:
    logits = torch.tensor([[[2.0, 0.0, 0.0], [0.4, 0.3, 0.3], [1.9, 0.0, 0.0]]])
    decision, selected, spread = select_contrastive(logits, keep=2)
    assert selected.tolist() == [[0, 2]]
    assert torch.allclose(spread, torch.tensor([[2.0, 0.1, 1.9]]))
    assert torch.allclose(decision, torch.tensor([[2.0, 0.0, 0.0]]))


def test_contrastive_selection_handles_k_one() -> None:
    logits = torch.tensor([[[1.0], [0.5]]])
    decision, selected, spread = select_contrastive(logits, keep=1)
    assert selected.shape == (1, 1)
    assert spread.shape == (1, 2)
    assert torch.isfinite(decision).all()


def test_pooled_score_averages_spans() -> None:
    torch.manual_seed(2)
    model = SpanCandidateScorer(Azurite(rung("tiny", "byte"))).eval()
    span_emb = torch.randn(2, 4, 8)
    cand_emb = torch.randn(3, 8)
    pooled = model.score_pooled(span_emb, cand_emb)
    assert pooled.shape == (2, 3)


def test_head_has_exactly_two_projection_matrices() -> None:
    cfg = rung("tiny", "byte")
    model = SpanCandidateScorer(Azurite(cfg))
    expected = 2 * cfg.d_model * cfg.d_model + 1
    actual = sum(
        p.numel()
        for name, p in model.named_parameters()
        if not name.startswith("encoder.")
    )
    assert actual == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_evidence.py -q
```

Expected: `ImportError: cannot import name 'SpanCandidateScorer'`.

- [ ] **Step 3: Implement the scorer**

Append to `azurite/evidence.py`:

```python
import math

import torch
import torch.nn.functional as F
from torch import nn

from .model import Azurite


def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)


class SpanCandidateScorer(nn.Module):
    """A frozen encoder plus a bilinear (span, candidate) scoring head.

    The encoder runs in the device dtype (fp16 on MPS); the head stays fp32 so
    AdamW updates are stable. `encode` casts the pooled vector to fp32 before
    projecting, which is the only place the two dtypes meet.
    """

    def __init__(self, encoder: Azurite) -> None:
        super().__init__()
        self.encoder = encoder
        d_model = encoder.cfg.d_model
        self.span_proj = nn.Linear(d_model, d_model, bias=False)
        self.cand_proj = nn.Linear(d_model, d_model, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(ids, mask).float()
        pooled = _pool(hidden, mask)
        return F.normalize(torch.tanh(self.span_proj(pooled)), dim=-1)

    def encode_candidates(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(ids, mask).float()
        pooled = _pool(hidden, mask)
        return F.normalize(torch.tanh(self.cand_proj(pooled)), dim=-1)

    def score_embeddings(
        self,
        span_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        cand_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.log_scale.exp() * torch.einsum("bsd,kd->bsk", span_emb, cand_emb)
        if cand_valid is not None:
            logits = logits.masked_fill(~cand_valid.view(1, 1, -1), float("-inf"))
        return logits

    def score_pooled(
        self,
        span_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        cand_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = span_emb.mean(dim=1)
        logits = self.log_scale.exp() * torch.einsum("bd,kd->bk", pooled, cand_emb)
        if cand_valid is not None:
            logits = logits.masked_fill(~cand_valid.view(1, -1), float("-inf"))
        return logits

    def forward(
        self,
        span_ids: torch.Tensor,
        span_mask: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_token_mask: torch.Tensor,
        cand_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        span_emb = self.encode(span_ids, span_mask)
        cand_emb = self.encode_candidates(cand_ids, cand_token_mask)
        return self.score_embeddings(span_emb, cand_emb, cand_valid)


def select_contrastive(
    logits: torch.Tensor, keep: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the decision, the selected span indices, and the per-span margin.

    The margin is top-1 minus top-2 over candidates: a span on which the two
    leading answers are nearly tied is not evidence for either. Selection by
    margin is the core claim of H2; it is deliberately not relevance.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be [batch, spans, candidates], got {logits.shape}")
    batch, spans, k = logits.shape
    if k < 2:
        spread = torch.zeros((batch, spans), device=logits.device, dtype=logits.dtype)
    else:
        top2 = logits.topk(2, dim=-1).values
        spread = (top2[..., 0] - top2[..., 1]).nan_to_num(0.0)
    count = min(max(keep, 1), spans)
    selected = spread.topk(count, dim=1).indices
    gathered = torch.gather(
        logits, 1, selected.unsqueeze(-1).expand(-1, -1, k)
    )
    return gathered.max(dim=1).values, selected, spread
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_evidence.py -q
```

Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add azurite/evidence.py tests/test_evidence.py
git commit -m "feat(evidence): score spans against candidates and select by margin"
```

---

### Task 4: Head training on frozen encoder features

**Files:**
- Modify: `azurite/evidence.py`
- Modify: `tests/test_evidence.py`

**Interfaces:**
- Produces: `fit_head(model: SpanCandidateScorer, batches, *, steps: int, lr: float = 3e-3, device: str = "cpu") -> list[float]` where each batch is `(span_ids, span_mask, cand_ids, cand_token_mask, cand_valid, gold)`.
- Consumes: `SpanCandidateScorer.encode`, `.encode_candidates`, `.score_embeddings`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evidence.py`:

```python
import itertools

from azurite.evidence import fit_head


def _synthetic_batches(n: int = 32, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    folds = []
    for _ in range(n):
        gold = int(torch.randint(0, 2, (1,), generator=gen))
        cue = "alpha" if gold == 0 else "beta"
        filler = "the rest of this clause is unrelated boilerplate"
        spans = [_ids(cue)[0], _ids(filler)[0], _ids(filler)[0], _ids(filler)[0]]
        span_ids = torch.cat(spans, dim=0)
        span_mask = torch.ones_like(span_ids, dtype=torch.bool)
        cand_ids = torch.cat([_ids("alpha")[0], _ids("beta")[0]], dim=0)
        cand_token_mask = torch.ones_like(cand_ids, dtype=torch.bool)
        cand_valid = torch.ones(2, dtype=torch.bool)
        folds.append(
            (span_ids, span_mask, cand_ids, cand_token_mask, cand_valid, torch.tensor(gold))
        )
    return folds


def test_fit_head_reduces_loss_on_synthetic_cues() -> None:
    torch.manual_seed(0)
    model = SpanCandidateScorer(Azurite(rung("tiny", "byte")))
    losses = fit_head(model, _synthetic_batches(), steps=120, lr=3e-3, device="cpu")
    assert len(losses) == 120
    assert losses[-1] < losses[0] * 0.8, (losses[0], losses[-1])


def test_fit_head_leaves_encoder_untouched() -> None:
    torch.manual_seed(1)
    model = SpanCandidateScorer(Azurite(rung("tiny", "byte")))
    before = [p.detach().clone() for p in model.encoder.parameters()]
    fit_head(model, _synthetic_batches(n=8), steps=5, lr=3e-3, device="cpu")
    for old, new in zip(before, model.encoder.parameters()):
        assert torch.equal(old, new.detach())
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_evidence.py -q
```

Expected: `ImportError: cannot import name 'fit_head'`.

- [ ] **Step 3: Implement `fit_head`**

Append to `azurite/evidence.py`:

```python
def fit_head(
    model: SpanCandidateScorer,
    batches,
    *,
    steps: int,
    lr: float = 3e-3,
    device: str = "cpu",
) -> list[float]:
    """Train only the bilinear head; the encoder is frozen.

    The encoder forward is under `no_grad` and its pooled output is detached,
    so the 403M backbone never allocates a backward graph. This is what makes
    head training fit in 8 GB next to the model.
    """
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.train()
    model.encoder.eval()
    parameters = [model.span_proj.weight, model.cand_proj.weight, model.log_scale]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    losses: list[float] = []
    stream = itertools.cycle(batches)
    for _ in range(steps):
        span_ids, span_mask, cand_ids, cand_token_mask, cand_valid, gold = next(stream)
        span_ids = span_ids.to(device)
        span_mask = span_mask.to(device)
        cand_ids = cand_ids.to(device)
        cand_token_mask = cand_token_mask.to(device)
        cand_valid = cand_valid.to(device)
        gold = gold.to(device)
        with torch.no_grad():
            span_emb = model.encode(span_ids, span_mask)
            cand_emb = model.encode_candidates(cand_ids, cand_token_mask)
        logits = model.score_embeddings(span_emb, cand_emb, cand_valid).max(dim=1).values
        loss = F.cross_entropy(logits, gold)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses
```

Add `import itertools` at the top of `azurite/evidence.py`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_evidence.py -q
```

Expected: `15 passed`.

- [ ] **Step 5: Commit**

```bash
git add azurite/evidence.py tests/test_evidence.py
git commit -m "feat(evidence): train the bilinear head on frozen encoder features"
```

---

### Task 5: ContractNLI document views

**Files:**
- Create: `scripts/build_evidence_docs.py`
- Create: `tests/test_evidence_docs.py`

**Interfaces:**
- Produces: `build_view(target: str, distractors: list[str], *, size: int, seed: int) -> dict` with keys `document`, `target_start`, `target_end`; `build_rows(rows: list[dict], *, size: int, seed: int) -> list[dict]`.
- Consumes: `data/raw/contract_nli_ctx/{train,test}` fields `id`, `context`, `question`, `label`, `label_text`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evidence_docs.py`:

```python
from __future__ import annotations

import pytest

from scripts.build_evidence_docs import build_rows, build_view

CANDIDATES = ["entailment", "contradiction", "not_mentioned"]


def make_rows(n: int = 12) -> list[dict]:
    return [
        {
            "id": f"row-{i}",
            "context": f"contract {i}: " + (f"clause {i} " * 40),
            "question": f"does clause {i} allow assignment?",
            "label": i % 3,
            "label_text": CANDIDATES[i % 3],
        }
        for i in range(n)
    ]


def test_target_span_round_trips_exactly() -> None:
    target = "the termination fee is thirty days"
    view = build_view(target, ["unrelated clause " * 20], size=512, seed=0)
    raw = view["document"].encode("utf-8")
    assert raw[view["target_start"] : view["target_end"]].decode("utf-8") == target


def test_document_respects_the_byte_budget() -> None:
    view = build_view("target clause " * 20, ["distractor " * 50] * 5, size=512, seed=0)
    assert len(view["document"].encode("utf-8")) <= 512


def test_builder_is_deterministic_and_moves_the_target() -> None:
    target = "target clause"
    distractors = ["filler one " * 10, "filler two " * 10]
    first = build_view(target, distractors, size=256, seed=3)
    second = build_view(target, distractors, size=256, seed=3)
    assert first == second
    starts = {
        build_view(target, distractors, size=256, seed=s)["target_start"]
        for s in range(12)
    }
    assert len(starts) > 1, "target was always in the same position"


def test_build_rows_preserves_gold_and_question() -> None:
    rows = make_rows()
    built = build_rows(rows[:1], size=1024, seed=0)
    assert len(built) == 1
    record = built[0]
    assert record["question"] == rows[0]["question"]
    assert record["gold"] == rows[0]["label"]
    assert record["candidates"] == CANDIDATES
    raw = record["document"].encode("utf-8")
    assert raw[record["target_start"] : record["target_end"]].decode("utf-8") == rows[0]["context"]


def test_build_rows_rejects_a_target_larger_than_the_budget_with_truncation() -> None:
    row = {
        "id": "long",
        "context": "x" * 10_000,
        "question": "q",
        "label": 0,
        "label_text": "entailment",
    }
    record = build_rows([row], size=1024, seed=0)[0]
    assert len(record["document"].encode("utf-8")) <= 1024
    assert record["target_end"] > record["target_start"]


def test_build_rows_does_not_mix_test_rows_into_train() -> None:
    train = make_rows(3)
    test = [dict(row, id=f"test-{i}") for i, row in enumerate(make_rows(3))]
    built = build_rows(train, size=1024, seed=0)
    assert all(not record["id"].startswith("test-") for record in built)
    assert {record["id"] for record in built} == {row["id"] for row in train}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_evidence_docs.py -q
```

Expected: `ModuleNotFoundError: No module named 'scripts.build_evidence_docs'`.

- [ ] **Step 3: Implement the builder**

```python
#!/usr/bin/env python3
"""Build 16 KB and 32 KB ContractNLI document views with exact target spans.

The target contract is always present and its byte span is recorded. Distractor
contracts fill the rest of the budget, so a retriever must find the right
contract among several plausible ones. Labels are unchanged: the question is
still about the target, and distractors are not labelled.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw" / "contract_nli_ctx"
CANDIDATES = ["entailment", "contradiction", "not_mentioned"]


def _truncate_utf8(text: str, budget: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= budget:
        return text
    return raw[:budget].decode("utf-8", errors="ignore")


def build_view(
    target: str, distractors: list[str], *, size: int, seed: int | str
) -> dict:
    if size <= 0:
        raise ValueError("size must be positive")
    target = _truncate_utf8(target, size)
    pieces: list[tuple[str, bool]] = [(target, True)]
    budget = size - len(target.encode("utf-8"))
    for text in distractors:
        if budget <= 0:
            break
        piece = _truncate_utf8(text, budget)
        if not piece:
            continue
        pieces.append((piece, False))
        budget -= len(piece.encode("utf-8"))
    rng = random.Random(seed)
    rng.shuffle(pieces)
    document_parts: list[str] = []
    target_start = target_end = 0
    offset = 0
    for text, is_target in pieces:
        if document_parts:
            offset += 1  # the joining newline
        if is_target:
            target_start = offset
            target_end = offset + len(text.encode("utf-8"))
        document_parts.append(text)
        offset += len(text.encode("utf-8"))
    document = "\n".join(document_parts)
    return {
        "document": document,
        "target_start": target_start,
        "target_end": target_end,
    }


def build_rows(rows: list[dict], *, size: int, seed: int) -> list[dict]:
    out: list[dict] = []
    for i, row in enumerate(rows):
        rng = random.Random(f"{seed}:{row['id']}")
        pool = [r["context"] for j, r in enumerate(rows) if j != i]
        rng.shuffle(pool)
        view = build_view(row["context"], pool, size=size, seed=f"{seed}:{row['id']}")
        out.append(
            {
                "id": row["id"],
                "question": row["question"],
                "candidates": list(CANDIDATES),
                "gold": int(row["label"]),
                "document": view["document"],
                "target_start": view["target_start"],
                "target_end": view["target_end"],
                "size": size,
                "seed": seed,
            }
        )
    return out


def main() -> int:
    import datasets

    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[16384, 32768])
    ap.add_argument("--out-dir", default="results/h2_evidence")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import gzip

    for split in ["train", "test"]:
        ds = datasets.load_from_disk(str(RAW / split))
        rows = [dict(row) for row in ds]
        for size in args.sizes:
            built = build_rows(rows, size=size, seed=args.seed)
            path = out_dir / f"docs_{split}_{size}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                for record in built:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{path}: {len(built)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_evidence_docs.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Build the real views**

```bash
.venv/bin/python scripts/build_evidence_docs.py --sizes 16384 32768
```

Expected: `docs_train_16384.json.gz`, `docs_test_16384.json.gz`, and the 32 KB pair, with 2,499 / 748 rows.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_evidence_docs.py tests/test_evidence_docs.py \
  results/h2_evidence/docs_train_16384.json.gz results/h2_evidence/docs_test_16384.json.gz \
  results/h2_evidence/docs_train_32768.json.gz results/h2_evidence/docs_test_32768.json.gz
git commit -m "feat(evidence): materialise 16 KB and 32 KB ContractNLI document views"
```

---

### Task 6: Head-training script

**Files:**
- Create: `scripts/train_evidence_head.py`
- Modify: `tests/test_evidence.py` (only if a CLI helper is extracted)

**Interfaces:**
- Consumes: `build_evidence_docs` JSONL, `SpanCandidateScorer`, `fit_head`, `ByteTokenizer`.
- Produces: `results/h2_evidence/head.npz` with arrays `span_proj`, `cand_proj`, `log_scale`, and a JSON `meta` string carrying `encoder_sha256`, `steps`, `lr`, `train_rows`, `loss_first`, `loss_last`, `config`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evidence.py`:

```python
from scripts.train_evidence_head import checkpoint_payload, load_head_into, save_head


def test_head_checkpoint_round_trips_without_encoder_weights(tmp_path) -> None:
    torch.manual_seed(3)
    model = SpanCandidateScorer(Azurite(rung("tiny", "byte")))
    payload = checkpoint_payload(model, encoder_sha256="abc", steps=7, lr=0.001)
    path = tmp_path / "head.npz"
    save_head(payload, path)
    loaded = SpanCandidateScorer(Azurite(rung("tiny", "byte")))
    meta = load_head_into(loaded, path)
    assert meta["encoder_sha256"] == "abc"
    assert torch.allclose(model.span_proj.weight, loaded.span_proj.weight)
    assert torch.allclose(model.cand_proj.weight, loaded.cand_proj.weight)
    assert torch.allclose(model.log_scale, loaded.log_scale)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_evidence.py::test_head_checkpoint_round_trips_without_encoder_weights -q
```

Expected: `ModuleNotFoundError: No module named 'scripts.train_evidence_head'`.

- [ ] **Step 3: Implement the script**

```python
#!/usr/bin/env python3
"""Train the H2 bilinear head on 16 KB ContractNLI views.

The encoder is frozen. Only 2.1M head parameters are trained, so this fits in
8 GB and finishes on the M1 in under an hour for the planned 400 steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.config import EncoderConfig
from azurite.evidence import SpanCandidateScorer, fit_head
from azurite.model import Azurite
from azurite.tokenizers import CLS_ID, PAD_ID, SEP_ID, ByteTokenizer

TOK = ByteTokenizer()


def pack(texts: list[str], *, max_seq: int = 514) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for text in texts:
        body = TOK.encode(text)[: max_seq - 2]
        rows.append([CLS_ID, *body, SEP_ID])
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, : len(row)] = True
    return ids, mask


def encoder_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_docs(path: Path, *, limit: int | None, seed: int) -> list[dict]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def batches_from(rows: list[dict], *, span_limit: int = 16, seed: int = 0):
    for row in rows:
        chunks = row["chunks"]
        if len(chunks) > span_limit:
            rng = random.Random(f"{seed}:{row['id']}")
            chunks = rng.sample(chunks, span_limit)
        span_ids, span_mask = pack([c["text"] for c in chunks])
        cand_ids, cand_token_mask = pack(row["candidates"])
        cand_valid = torch.ones(len(row["candidates"]), dtype=torch.bool)
        gold = int(row["gold"])
        yield span_ids, span_mask, cand_ids, cand_token_mask, cand_valid, torch.tensor(gold)


def checkpoint_payload(
    model: SpanCandidateScorer, *, encoder_sha256: str, steps: int, lr: float
) -> dict:
    meta = {
        "encoder_sha256": encoder_sha256,
        "steps": steps,
        "lr": lr,
        "config": model.encoder.cfg.to_dict(),
    }
    return {
        "span_proj": model.span_proj.weight.detach().cpu().numpy(),
        "cand_proj": model.cand_proj.weight.detach().cpu().numpy(),
        "log_scale": np.asarray([float(model.log_scale.detach())], dtype=np.float32),
        "meta": np.asarray(json.dumps(meta)),
    }


def save_head(payload: dict, path: Path) -> None:
    np.savez(path, **payload)


def load_head_into(model: SpanCandidateScorer, path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        model.span_proj.weight.data.copy_(torch.from_numpy(data["span_proj"]))
        model.cand_proj.weight.data.copy_(torch.from_numpy(data["cand_proj"]))
        model.log_scale.data.copy_(torch.tensor(float(data["log_scale"][0])))
        return json.loads(str(data["meta"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="results/h2_evidence/docs_train_16384.json.gz")
    ap.add_argument("--ckpt", default="results/pretrain/l1_xlarge_byte_dn1/ckpt.pt")
    ap.add_argument("--out", default="results/h2_evidence/head.npz")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["config"])
    encoder = Azurite(cfg).eval()
    encoder.load_state_dict(payload["model"], strict=True)
    dtype = torch.float16 if args.device == "mps" else (torch.bfloat16 if args.device == "cuda" else torch.float32)
    encoder.to(device=args.device, dtype=dtype)
    model = SpanCandidateScorer(encoder)
    model.span_proj.to(args.device)
    model.cand_proj.to(args.device)
    model.log_scale.data = model.log_scale.data.to(args.device)

    rows = load_docs(Path(args.docs), limit=args.limit, seed=args.seed)
    # Attach chunks once; this is the same 512/384 chunking the harness uses.
    from azurite.evidence import chunk_document

    for row in rows:
        row["chunks"] = [
            {"index": c.index, "start": c.start, "end": c.end, "text": c.text}
            for c in chunk_document(row["document"], size=512, stride=512)
        ]
    batches = list(batches_from(rows, seed=args.seed))
    losses = fit_head(model, batches, steps=args.steps, lr=args.lr, device=args.device)

    payload = checkpoint_payload(
        model,
        encoder_sha256=encoder_sha256(Path(args.ckpt)),
        steps=args.steps,
        lr=args.lr,
    )
    meta = json.loads(str(payload["meta"]))
    meta.update({"train_rows": len(rows), "loss_first": losses[0], "loss_last": losses[-1]})
    payload["meta"] = np.asarray(json.dumps(meta))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_head(payload, Path(args.out))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_evidence.py -q
```

Expected: `16 passed`.

- [ ] **Step 5: Train the head**

```bash
.venv/bin/python scripts/train_evidence_head.py --device mps --steps 400
```

Expected: exit 0, `loss_last < loss_first`, `results/h2_evidence/head.npz` written, peak RSS under 6.5 GiB. If MPS fp16 is non-finite, rerun with `--device cpu`; record the device in the checkpoint metadata.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_evidence_head.py tests/test_evidence.py results/h2_evidence/head.npz
git commit -m "feat(evidence): train the H2 head on frozen 403M features"
```

---

### Task 7: Three-arm H2 harness

**Files:**
- Create: `scripts/h2_evidence.py`
- Create: `tests/test_h2_evidence.py`

**Interfaces:**
- Produces: `truncate_view(question: str, document: str, budget: int = 1020) -> str`; `pack(texts, max_seq=514) -> (ids, mask)`; `score_row(model, row, *, device="cpu", top_k=4, keep=3) -> dict[str, dict]`; CLI writing `records_{size}.json` and `summary_{size}.json`.
- Consumes: `chunk_document`, `ChunkIndex`, `retrieval_query`, `select_contrastive`, `load_head_into`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_h2_evidence.py`:

```python
from __future__ import annotations

import pytest
import torch

from azurite.config import rung
from azurite.evidence import SpanCandidateScorer
from azurite.model import Azurite
from scripts.h2_evidence import score_row, truncate_view


def row() -> dict:
    return {
        "id": "r1",
        "question": "does the contract allow assignment?",
        "candidates": ["entailment", "contradiction", "not_mentioned"],
        "gold": 0,
        "document": (
            "filler clause about notices. " * 40
            + "the contract allows assignment with written consent. "
            + "filler clause about governing law. " * 40
        ),
        "target_start": 1080,
        "target_end": 1160,
    }


def test_truncate_view_keeps_the_question() -> None:
    view = truncate_view("does it allow assignment?", "x" * 5000)
    assert view.startswith("does it allow assignment?")
    assert len(view.encode("utf-8")) <= 1020


def test_truncate_view_short_document_is_unchanged() -> None:
    assert truncate_view("q", "short document") == "q\nshort document"


def test_score_row_returns_all_three_arms() -> None:
    torch.manual_seed(0)
    model = SpanCandidateScorer(Azurite(rung("tiny", "byte"))).eval()
    result = score_row(model, row(), top_k=2, keep=1)
    assert set(result) == {"truncate", "chunk_pool", "contrastive"}
    for arm in result.values():
        assert 0 <= arm["pred"] < 3
        assert arm["correct"] in (0, 1)
        assert arm["encoded_bytes"] > 0
        assert arm["doc_bytes"] > 0


def test_contrastive_encodes_less_than_the_document() -> None:
    torch.manual_seed(1)
    model = SpanCandidateScorer(Azurite(rung("tiny", "byte"))).eval()
    result = score_row(model, row(), top_k=2, keep=1)
    assert result["contrastive"]["encoded_bytes"] < result["chunk_pool"]["encoded_bytes"]
    assert result["contrastive"]["encoded_fraction"] < 1.0


def test_chunk_pool_reads_every_chunk() -> None:
    torch.manual_seed(2)
    model = SpanCandidateScorer(Azurite(rung("tiny", "byte"))).eval()
    result = score_row(model, row(), top_k=2, keep=1)
    assert result["chunk_pool"]["encoded_bytes"] >= result["contrastive"]["encoded_bytes"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_h2_evidence.py -q
```

Expected: `ModuleNotFoundError: No module named 'scripts.h2_evidence'`.

- [ ] **Step 3: Implement the harness**

```python
#!/usr/bin/env python3
"""H2: compare 1 KB truncation, all-chunk pooling, and contrastive selection.

All arms share one frozen encoder and one trained head. The only difference is
which spans each arm reads. Per-row records carry the selected byte spans so a
reviewer can inspect the evidence, not just the accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.config import EncoderConfig
from azurite.evidence import (
    ChunkIndex,
    SpanCandidateScorer,
    chunk_document,
    retrieval_query,
    select_contrastive,
)
from azurite.model import Azurite
from azurite.stats import wilson
from azurite.tokenizers import CLS_ID, PAD_ID, SEP_ID, ByteTokenizer

TOK = ByteTokenizer()
INSTRUCTION = (
    "Decide whether the contract entails, contradicts, or does not mention the "
    "stated policy."
)


def truncate_view(question: str, document: str, budget: int = 1020) -> str:
    head = f"{question}\n"
    if len(head.encode("utf-8")) >= budget:
        return head.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
    room = budget - len(head.encode("utf-8"))
    body = document.encode("utf-8")[:room].decode("utf-8", errors="ignore")
    return head + body


def pack(texts: list[str], *, max_seq: int = 514) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for text in texts:
        body = TOK.encode(text)[: max_seq - 2]
        rows.append([CLS_ID, *body, SEP_ID])
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, : len(r)] = True
    return ids, mask


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _encode_spans(
    model: SpanCandidateScorer,
    texts: list[str],
    *,
    device: str,
    max_seq: int,
    span_batch: int = 8,
) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(texts), span_batch):
        ids, mask = pack(texts[start : start + span_batch], max_seq=max_seq)
        with torch.no_grad():
            embeddings.append(model.encode(ids.to(device), mask.to(device)))
    return torch.cat(embeddings, dim=0)


def _score(
    model: SpanCandidateScorer,
    span_texts: list[str],
    candidates: list[str],
    *,
    device: str,
    mode: str,
    max_seq: int,
) -> torch.Tensor:
    cand_ids, cand_token_mask = pack(candidates)
    cand_valid = torch.ones(len(candidates), dtype=torch.bool, device=device)
    span_emb = _encode_spans(model, span_texts, device=device, max_seq=max_seq)
    with torch.no_grad():
        cand_emb = model.encode_candidates(
            cand_ids.to(device), cand_token_mask.to(device)
        )
        if mode == "pool":
            logits = model.score_pooled(span_emb, cand_emb, cand_valid)
        else:
            logits = model.score_embeddings(span_emb, cand_emb, cand_valid)
    return logits


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def score_row(
    model: SpanCandidateScorer,
    row: dict,
    *,
    device: str = "cpu",
    top_k: int = 4,
    keep: int = 3,
) -> dict[str, dict]:
    document = row["document"]
    candidates = row["candidates"]
    gold = int(row["gold"])
    doc_bytes = len(document.encode("utf-8"))
    chunks = chunk_document(document, size=512, stride=512)
    target = (int(row["target_start"]), int(row["target_end"]))
    output: dict[str, dict] = {}

    def record(
        arm: str, logits: torch.Tensor, encoded: int, selected: list, started: float
    ) -> None:
        _sync(device)
        probs = F.softmax(logits.float(), dim=-1)[0]
        pred = int(probs.argmax())
        output[arm] = {
            "pred": pred,
            "correct": int(pred == gold),
            "confidence": round(float(probs[pred]), 6),
            "encoded_bytes": int(encoded),
            "encoded_fraction": round(encoded / max(1, doc_bytes), 6),
            "doc_bytes": doc_bytes,
            "selected": selected,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    _sync(device)
    started = time.perf_counter()
    view = truncate_view(row["question"], document)
    logits = _score(model, [view], candidates, device=device, mode="single", max_seq=1024)
    record("truncate", logits, len(view.encode("utf-8")), [], started)

    _sync(device)
    started = time.perf_counter()
    pool_texts = [c.text for c in chunks]
    logits = _score(model, pool_texts, candidates, device=device, mode="pool", max_seq=514)
    record("chunk_pool", logits, sum(len(t.encode("utf-8")) for t in pool_texts), [], started)

    _sync(device)
    started = time.perf_counter()
    query = retrieval_query(INSTRUCTION, row["question"], candidates)
    retrieved = ChunkIndex(chunks).search(query, k=top_k)
    retrieved = retrieved or chunks[:top_k]
    span_emb = _encode_spans(
        model, [c.text for c in retrieved], device=device, max_seq=514
    )
    cand_ids, cand_token_mask = pack(candidates)
    cand_valid = torch.ones(len(candidates), dtype=torch.bool, device=device)
    with torch.no_grad():
        cand_emb = model.encode_candidates(
            cand_ids.to(device), cand_token_mask.to(device)
        )
        per_span = model.score_embeddings(span_emb, cand_emb, cand_valid)
    decision, selected_idx, _ = select_contrastive(per_span, keep=keep)
    selected = [
        {
            "start": retrieved[int(i)].start,
            "end": retrieved[int(i)].end,
            "text_sha256": hashlib.sha256(retrieved[int(i)].text.encode("utf-8")).hexdigest(),
        }
        for i in selected_idx[0].tolist()
    ]
    encoded = sum(len(c.text.encode("utf-8")) for c in retrieved)
    retrieval_hit = any(
        _overlap(c.start, c.end, *target) >= 0.5 * max(1, c.end - c.start)
        for c in retrieved
    )
    evidence_hit = any(
        _overlap(s["start"], s["end"], *target) >= 0.5 * max(1, s["end"] - s["start"])
        for s in selected
    )
    record("contrastive", decision, encoded, selected, started)
    output["contrastive"]["retrieval_hit"] = int(retrieval_hit)
    output["contrastive"]["evidence_hit"] = int(evidence_hit)
    return output


def summarise(records: list[dict], arm: str, size: int) -> dict:
    rows = [r["arms"][arm] for r in records]
    correct = np.asarray([r["correct"] for r in rows], dtype=np.int8)
    latencies = [r.get("latency_ms", 0.0) for r in rows if "latency_ms" in r]
    lo, hi = wilson(int(correct.sum()), len(correct))
    summary = {
        "size": size,
        "arm": arm,
        "n": len(rows),
        "accuracy": float(correct.mean()),
        "wilson95": [lo, hi],
        "encoded_fraction_mean": float(np.mean([r["encoded_fraction"] for r in rows])),
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }
    if arm == "contrastive":
        summary["retrieval_hit_rate"] = float(np.mean([r["retrieval_hit"] for r in rows]))
        summary["evidence_hit_rate"] = float(np.mean([r["evidence_hit"] for r in rows]))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="results/h2_evidence/docs_test_16384.json.gz")
    ap.add_argument("--size", type=int, default=16384)
    ap.add_argument("--ckpt", default="results/pretrain/l1_xlarge_byte_dn1/ckpt.pt")
    ap.add_argument("--head", default="results/h2_evidence/head.npz")
    ap.add_argument("--out-dir", default="results/h2_evidence")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--keep", type=int, default=3)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["config"])
    encoder = Azurite(cfg).eval()
    encoder.load_state_dict(payload["model"], strict=True)
    dtype = torch.float16 if args.device == "mps" else (torch.bfloat16 if args.device == "cuda" else torch.float32)
    encoder.to(device=args.device, dtype=dtype)
    model = SpanCandidateScorer(encoder)
    from scripts.train_evidence_head import encoder_sha256, load_head_into

    head_payload = load_head_into(model, Path(args.head))
    if head_payload["encoder_sha256"] != encoder_sha256(Path(args.ckpt)):
        raise SystemExit("head checkpoint was trained on a different encoder")
    model.span_proj.to(args.device).float()
    model.cand_proj.to(args.device).float()
    model.log_scale.data = model.log_scale.data.to(args.device).float()

    import gzip

    opener = gzip.open if str(args.docs).endswith(".gz") else open
    with opener(args.docs, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows):
        arms = score_row(model, row, device=args.device, top_k=args.top_k, keep=args.keep)
        records.append({"id": row["id"], "gold": row["gold"], "arms": arms})
        if (i + 1) % 50 == 0:
            print(f"[h2] {i + 1}/{len(rows)}", flush=True)
    record_path = out_dir / f"records_{args.size}.json"
    record_path.write_text(json.dumps(records, indent=2) + "\n")
    summary = {
        "size": args.size,
        "n": len(records),
        "device": args.device,
        "checkpoint_sha256": encoder_sha256(Path(args.ckpt)),
        "arms": {arm: summarise(records, arm, args.size) for arm in ["truncate", "chunk_pool", "contrastive"]},
    }
    (out_dir / f"summary_{args.size}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_h2_evidence.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/h2_evidence.py tests/test_h2_evidence.py
git commit -m "feat(evidence): add the three-arm H2 harness with cited spans"
```

---

### Task 8: Paired report and RESULTS.md

**Files:**
- Create: `scripts/h2_report.py`
- Create: `tests/test_h2_report.py`

**Interfaces:**
- Produces: `paired(records: list[dict], left: str, right: str) -> dict` with `n`, `delta`, `mcnemar`, `bootstrap_lower`; `render(results: dict) -> str`.
- Consumes: `azurite.stats.wilson`, `mcnemar_exact`, `bootstrap_lower`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_h2_report.py`:

```python
from __future__ import annotations

from scripts.h2_report import paired, render


def records(good: int, bad: int) -> list[dict]:
    out = []
    for i in range(good + bad):
        left = 1 if i < good else 0
        right = 0 if i < good else 1
        out.append({"id": str(i), "arms": {"a": {"correct": left}, "b": {"correct": right}}})
    return out


def test_paired_counts_wins_and_losses() -> None:
    result = paired(records(8, 0), "a", "b")
    assert result["n"] == 8
    assert result["delta"] == 1.0
    assert result["mcnemar"]["wins"] == 8
    assert result["mcnemar"]["losses"] == 0
    assert result["mcnemar"]["p"] < 0.01
    assert result["bootstrap_lower"] > 0.0


def test_paired_handles_a_tie() -> None:
    records_tied = [
        {"id": "1", "arms": {"a": {"correct": 1}, "b": {"correct": 1}}},
        {"id": "2", "arms": {"a": {"correct": 0}, "b": {"correct": 0}}},
    ]
    result = paired(records_tied, "a", "b")
    assert result["delta"] == 0.0
    assert result["mcnemar"]["p"] == 1.0


def test_render_contains_the_verdict_table() -> None:
    payload = {
        "sizes": {
            16384: {
                "n": 748,
                "arms": {
                    "truncate": {"accuracy": 0.50, "p95_ms": 100.0, "encoded_fraction_mean": 0.06},
                    "chunk_pool": {"accuracy": 0.60, "p95_ms": 400.0, "encoded_fraction_mean": 1.0},
                    "contrastive": {"accuracy": 0.65, "p95_ms": 300.0, "encoded_fraction_mean": 0.12},
                },
                "contrastive_vs_truncate": {"delta": 0.15, "mcnemar": {"p": 0.001}, "bootstrap_lower": 0.08},
                "contrastive_vs_chunk_pool": {"delta": 0.05, "mcnemar": {"p": 0.2}, "bootstrap_lower": 0.0},
            }
        }
    }
    text = render(payload)
    assert "16 KB" in text
    assert "contrastive" in text
    assert "+15.0" in text or "15.0" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_h2_report.py -q
```

Expected: `ModuleNotFoundError: No module named 'scripts.h2_report'`.

- [ ] **Step 3: Implement the report**

```python
#!/usr/bin/env python3
"""Render the pre-registered H2 verdict from stored per-row records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azurite.stats import bootstrap_lower, mcnemar_exact


def paired(records: list[dict], left: str, right: str) -> dict:
    a = [r["arms"][left]["correct"] for r in records]
    b = [r["arms"][right]["correct"] for r in records]
    delta = sum(a) / len(a) - sum(b) / len(b)
    return {
        "n": len(a),
        "delta": delta,
        "mcnemar": mcnemar_exact(a, b),
        "bootstrap_lower": bootstrap_lower(a, b, seed=0),
    }


def render(payload: dict) -> str:
    lines = [
        "# H2 — contrastive evidence selection",
        "",
        "| size | arm | n | accuracy | encoded % | p95 ms |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for size, result in sorted(payload["sizes"].items()):
        for arm in ["truncate", "chunk_pool", "contrastive"]:
            row = result["arms"][arm]
            lines.append(
                f"| {int(size) // 1024} KB | {arm} | {result['n']} | "
                f"{100 * row['accuracy']:.2f} | {100 * row['encoded_fraction_mean']:.1f} | "
                f"{row['p95_ms']:.0f} |"
            )
        vs_t = result["contrastive_vs_truncate"]
        vs_p = result["contrastive_vs_chunk_pool"]
        lines.extend(
            [
                "",
                f"Contrastive − truncate: **{100 * vs_t['delta']:+.2f} pp** "
                f"(McNemar p = {vs_t['mcnemar']['p']:.4g}, "
                f"bootstrap lower {100 * vs_t['bootstrap_lower']:+.2f} pp).",
                f"Contrastive − chunk_pool: **{100 * vs_p['delta']:+.2f} pp** "
                f"(McNemar p = {vs_p['mcnemar']['p']:.4g}).",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/h2_evidence")
    args = ap.parse_args()
    out = Path(args.out_dir)
    payload = {"sizes": {}}
    for path in sorted(out.glob("records_*.json")):
        size = int(path.stem.split("_")[1])
        records = json.loads(path.read_text())
        summary = json.loads((out / f"summary_{size}.json").read_text())
        summary["contrastive_vs_truncate"] = paired(records, "contrastive", "truncate")
        summary["contrastive_vs_chunk_pool"] = paired(records, "contrastive", "chunk_pool")
        payload["sizes"][size] = summary
    text = render(payload)
    (out / "RESULTS.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_h2_report.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/h2_report.py tests/test_h2_report.py
git commit -m "feat(evidence): render paired H2 deltas and verdict"
```

---

### Task 9: Run H2 and the M1 acceptance check

**Files:**
- Create: `scripts/h2_smoke.py`
- Create: `tests/test_m1_smoke.py`
- Create (generated): `results/h2_evidence/records_{16384,32768}.jsonl`, `summary_{16384,32768}.json`, `RESULTS.md`, `m1_smoke.json`

**Interfaces:**
- Consumes: `scripts.h2_evidence.score_row`, `scripts.train_evidence_head.load_head_into`, `scripts.m1_profile.check_envelope`.
- Produces: `smoke(records: list[dict], *, limit: int) -> dict` and an exit code that is non-zero on any envelope violation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_m1_smoke.py`:

```python
from __future__ import annotations

from scripts.h2_smoke import smoke_report


def test_smoke_report_flags_an_over_budget_run() -> None:
    report = smoke_report(
        n=20,
        peak_rss_gib=7.0,
        swap_delta_mb=0.0,
        pageout_delta=0,
        p95_ms=1200.0,
    )
    assert report["ok"] is False
    assert any("RSS" in problem for problem in report["problems"])


def test_smoke_report_accepts_a_fitting_run() -> None:
    report = smoke_report(
        n=20,
        peak_rss_gib=5.2,
        swap_delta_mb=0.0,
        pageout_delta=0,
        p95_ms=2500.0,
    )
    assert report["ok"] is True
    assert report["problems"] == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_m1_smoke.py -q
```

Expected: `ModuleNotFoundError: No module named 'scripts.h2_smoke'`.

- [ ] **Step 3: Implement the smoke runner**

```python
#!/usr/bin/env python3
"""Run 20 rows through the contrastive arm and assert the M1 envelope."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h2_evidence import score_row
from scripts.m1_profile import check_envelope, peak_rss_gib
from azurite.env import pageouts, swap_used_mb


def smoke_report(
    *, n: int, peak_rss_gib: float, swap_delta_mb: float, pageout_delta: int, p95_ms: float
) -> dict:
    problems = check_envelope(
        {
            "peak_rss_gib": peak_rss_gib,
            "swap_delta_mb": swap_delta_mb,
            "pageout_delta": pageout_delta,
            "p95_ms": p95_ms,
        }
    )
    return {
        "n": n,
        "peak_rss_gib": peak_rss_gib,
        "swap_delta_mb": swap_delta_mb,
        "pageout_delta": pageout_delta,
        "p95_ms": p95_ms,
        "problems": problems,
        "ok": not problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="results/h2_evidence/docs_test_16384.json.gz")
    ap.add_argument("--ckpt", default="results/pretrain/l1_xlarge_byte_dn1/ckpt.pt")
    ap.add_argument("--head", default="results/h2_evidence/head.npz")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results/h2_evidence/m1_smoke.json")
    args = ap.parse_args()

    from azurite.config import EncoderConfig
    from azurite.evidence import SpanCandidateScorer
    from azurite.model import Azurite
    from scripts.train_evidence_head import load_head_into

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = EncoderConfig(**payload["config"])
    encoder = Azurite(cfg).eval()
    encoder.load_state_dict(payload["model"], strict=True)
    dtype = torch.float16 if args.device == "mps" else torch.float32
    encoder.to(device=args.device, dtype=dtype)
    model = SpanCandidateScorer(encoder)
    load_head_into(model, Path(args.head))
    model.span_proj.to(args.device).float()
    model.cand_proj.to(args.device).float()
    model.log_scale.data = model.log_scale.data.to(args.device).float()

    import gzip

    with gzip.open(args.docs, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()][: args.limit]
    before_swap = swap_used_mb()
    before_pages = pageouts()
    latencies = []
    for row in rows:
        started = time.perf_counter()
        score_row(model, row, device=args.device, top_k=4, keep=3)
        latencies.append((time.perf_counter() - started) * 1000.0)
    report = smoke_report(
        n=len(rows),
        peak_rss_gib=round(peak_rss_gib(), 3),
        swap_delta_mb=round(swap_used_mb() - before_swap, 1),
        pageout_delta=pageouts() - before_pages,
        p95_ms=round(float(np.percentile(latencies, 95)), 2),
    )
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_m1_smoke.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Run the two H2 evaluations**

```bash
.venv/bin/python scripts/h2_evidence.py --docs results/h2_evidence/docs_test_16384.json.gz --size 16384 --device mps
.venv/bin/python scripts/h2_evidence.py --docs results/h2_evidence/docs_test_32768.json.gz --size 32768 --device mps
.venv/bin/python scripts/h2_report.py
```

Expected: `RESULTS.md` with the 16 KB and 32 KB tables and the paired deltas.

- [ ] **Step 6: Run the M1 acceptance check**

```bash
.venv/bin/python scripts/h2_smoke.py --device mps --limit 20
```

Expected: exit 0, `m1_smoke.json` with `"ok": true`, peak RSS ≤ 6.5 GiB, zero swap, p95 ≤ 5,000 ms.

- [ ] **Step 7: Commit the result**

```bash
git add scripts/h2_smoke.py tests/test_m1_smoke.py \
  results/h2_evidence/records_16384.json results/h2_evidence/records_32768.json \
  results/h2_evidence/summary_16384.json results/h2_evidence/summary_32768.json \
  results/h2_evidence/RESULTS.md results/h2_evidence/m1_smoke.json
git commit -m "feat(evidence): run H2 and record the M1 acceptance envelope"
```

---

## Self-review

**Spec coverage.** M1 8 GB → Task 1 and Task 9. 1 KB useless → Task 2/7 (16/32 KB views). New method → Task 3/7 (margin-based span selection). Frozen encoder, 403M → Task 4/6 (head-only training). No benchmark theater → pre-registration and paired report. Licence/product not in scope; this plan produces the H2 evidence only.

**Placeholder scan.** No TBD, no "add error handling," no "similar to Task N." Every code step has runnable code.

**Type consistency.** `Chunk` fields `index/start/end/text` are used by `ChunkIndex`, `score_row`, and the builder. `select_contrastive` returns `(decision, selected_indices, spread)` everywhere. Head payload keys `span_proj.weight`, `cand_proj.weight`, `log_scale` match `checkpoint_payload` and `load_head_into`. `pack` returns `(ids, mask)` and is used for both spans and candidates.

**Known limitations, stated up front.** The head is trained on 16 KB all-chunk views, so the contrastive arm is a zero-shot selection test, not a jointly trained retriever. Evidence precision is document-level (ContractNLI annotates the contract, not the sentence). One seed. These are reported, not hidden.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-11-01-evidence-selection.md`.

Two execution options:

1. **Subagent-Driven (recommended):** dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution:** execute tasks in this session with checkpoints for review.

Which approach?
