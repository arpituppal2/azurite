"""Chunking and evidence selection for the Select+Read model.

The reader has a 4,096-token window. The selector is what makes a 32,768-token
document accessible without paying 32k dense attention. Two selectors are
provided:

* ``BM25ChunkSelector``: zero parameters, deterministic, the baseline.
* ``ChunkEncoder`` + ``NeuralChunkSelector``: a small bidirectional encoder
  trained with InfoNCE. It is only shipped if it beats BM25 on the same rows.

Chunks are byte slices with exact UTF-8 byte offsets, so an evidence span points
back into the caller's document even though the reader consumes BPE tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .model import Azurite
from .retrieval import BM25
from .tokenizers import PAD_ID


@dataclass(frozen=True)
class Chunk:
    """One evidence chunk with a byte range in the original document."""

    index: int
    text: str
    byte_start: int
    byte_end: int
    tokens: tuple[int, ...] = field(default=(), compare=False)


def chunk_text(
    text: str,
    *,
    chunk_bytes: int = 2_048,
    stride: int = 1_024,
) -> list[Chunk]:
    """Split text into overlapping UTF-8 byte chunks.

    2,048 bytes is roughly 615 BPE tokens at the measured 3.327 bytes/token.
    The exact token count is capped by the selector/reader packing code. The
    stride is positive and smaller than the chunk, so a boundary term survives
    in at least one chunk.
    """
    if chunk_bytes <= 0 or stride <= 0 or stride > chunk_bytes:
        raise ValueError("chunk_bytes and stride must be positive; stride <= chunk_bytes")
    raw = text.encode("utf-8")
    if not raw:
        return []
    chunks: list[Chunk] = []
    start = 0
    while start < len(raw):
        end = min(start + chunk_bytes, len(raw))
        # Do not cut a multi-byte UTF-8 sequence. The byte range keeps the
        # original boundary; decoding ignores a partial sequence only when the
        # source itself is invalid.
        while end < len(raw) and (raw[end] & 0xC0) == 0x80:
            end += 1
        piece = raw[start:end].decode("utf-8", errors="ignore")
        if piece.strip():
            chunks.append(Chunk(index=len(chunks), text=piece,
                                byte_start=start, byte_end=end))
        if end >= len(raw):
            break
        start += stride
    return chunks


def tokenize_chunks(
    tokenizer, chunks: Sequence[Chunk], *, max_tokens: int = 512
) -> list[list[int]]:
    return [tokenizer.encode(c.text)[:max_tokens] for c in chunks]


def _pad(rows: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(r) for r in rows), default=1)
    ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, row in enumerate(rows):
        if row:
            ids[i, : len(row)] = torch.tensor(list(row), dtype=torch.long)
            mask[i, : len(row)] = True
    return ids, mask


class BM25ChunkSelector:
    """Deterministic lexical selector over chunks."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.index = BM25(
            [{"id": c.index, "text": c.text} for c in self.chunks]
        )

    def rank(self, query: str) -> list[Chunk]:
        return [self.chunks[i] for i in self.index.rank(query)]

    def select(self, query: str, *, top_k: int = 8) -> list[Chunk]:
        if top_k <= 0:
            return []
        return self.rank(query)[:top_k]


class ChunkEncoder(torch.nn.Module):
    """Mean-pooled chunk encoder with a shared projection."""

    def __init__(self, encoder: Azurite) -> None:
        super().__init__()
        self.encoder = encoder
        self.proj = torch.nn.Linear(encoder.cfg.d_model, encoder.cfg.d_model, bias=False)
        self.log_scale = torch.nn.Parameter(torch.tensor(2.659260036932778))  # log(1/0.07)

    @staticmethod
    def _mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    def embed(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self._mean(self.encoder(ids, mask), mask)
        return F.normalize(torch.tanh(self.proj(pooled)), dim=-1)

    def forward(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        chunk_ids: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.embed(query_ids, query_mask)
        chunks = self.embed(chunk_ids, chunk_mask)
        return self.log_scale.exp() * (query @ chunks.T)


class NeuralChunkSelector:
    """Select chunks by cosine similarity in the trained chunk space."""

    def __init__(
        self,
        tokenizer,
        model: ChunkEncoder,
        *,
        max_tokens: int = 512,
        device: str = "cpu",
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.max_tokens = max_tokens
        self.device = device

    @torch.no_grad()
    def select(
        self, chunks: Sequence[Chunk], query: str, *, top_k: int = 8
    ) -> list[Chunk]:
        if not chunks or top_k <= 0:
            return []
        q_ids, q_mask = _pad([self.tokenizer.encode(query)[: self.max_tokens]])
        c_ids, c_mask = _pad(tokenize_chunks(self.tokenizer, chunks, max_tokens=self.max_tokens))
        scores = self.model(
            q_ids.to(self.device), q_mask.to(self.device),
            c_ids.to(self.device), c_mask.to(self.device),
        )[0]
        order = torch.argsort(scores, descending=True).tolist()
        return [chunks[i] for i in order[:top_k]]


def recall_at_k(
    selected: Sequence[Chunk], gold_indices: Iterable[int], *, k: int | None = None
) -> float:
    gold = set(int(i) for i in gold_indices)
    if not gold:
        return 0.0
    picked = {c.index for c in (selected[:k] if k is not None else selected)}
    return len(gold & picked) / len(gold)
