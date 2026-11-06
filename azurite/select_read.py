"""Select+Read inference: chunk selection, 4k reader, candidate scoring.

The model is one reader and one selector. The reader never sees more than
``max_reader_tokens``; the selector sees every chunk of the 32k document. A
32k context claim is therefore a selector-recall claim, not a dense-attention
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .config import EncoderConfig
from .model import Azurite, AzuriteDual
from .selector import (
    BM25ChunkSelector,
    Chunk,
    NeuralChunkSelector,
    chunk_text,
    tokenize_chunks,
)
from .tokenizers import CLS_ID, PAD_ID, SEP_ID, load as load_tokenizer


@dataclass
class SelectReadOutput:
    choice: int
    scores: list[float]
    abstained: bool
    abstain_reason: str | None
    selected: list[Chunk]
    reader_tokens: int


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_reader_checkpoint(path: str | Path, *, device: str = "cpu") -> Azurite:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg_data = payload.get("config") if isinstance(payload, dict) else None
    if not cfg_data:
        raise ValueError(f"{path}: checkpoint has no encoder config")
    cfg = EncoderConfig(**cfg_data)
    model = Azurite(cfg)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    return model.to(device).eval()


def _pad(rows: Sequence[Sequence[int]], *, width: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    w = width or max((len(r) for r in rows), default=1)
    ids = torch.full((len(rows), w), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(rows), w), dtype=torch.bool)
    for i, row in enumerate(rows):
        take = list(row)[:w]
        if take:
            ids[i, : len(take)] = torch.tensor(take, dtype=torch.long)
            mask[i, : len(take)] = True
    return ids, mask


class SelectRead:
    def __init__(
        self,
        reader: Azurite,
        tokenizer,
        *,
        selector=None,
        top_k: int = 8,
        max_reader_tokens: int = 4_096,
        abstain_threshold: float = 0.0,
        device: str = "cpu",
    ) -> None:
        device = resolve_device(device)
        self.reader = reader.to(device).eval()
        self.dual = AzuriteDual(self.reader).to(device).eval()
        self.tokenizer = tokenizer
        self.selector = selector
        self.top_k = int(top_k)
        self.max_reader_tokens = int(max_reader_tokens)
        self.abstain_threshold = float(abstain_threshold)
        self.device = device

    def select(self, context: str, question: str) -> tuple[list[Chunk], list[Chunk]]:
        chunks = chunk_text(context)
        if not chunks:
            return [], []
        if self.selector is None:
            selected = chunks[: self.top_k]
        elif isinstance(self.selector, NeuralChunkSelector):
            selected = self.selector.select(chunks, question, top_k=self.top_k)
        else:
            selected = self.selector.select(question, top_k=self.top_k)
        # Preserve document order for the reader even when the selector ranks
        # differently. The selector score is still available on the chunks.
        selected = sorted({c.index: c for c in selected}.values(), key=lambda c: c.index)
        return chunks, selected

    def _selected_ids(self, selected: Sequence[Chunk], question: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        parts = [question.strip()] if question.strip() else []
        parts.extend(c.text for c in selected)
        text = "\n\n".join(parts)
        ids = self.tokenizer.encode(text)[: self.max_reader_tokens - 2]
        row = [CLS_ID, *ids, SEP_ID]
        ids_t, mask_t = _pad([row], width=max(1, len(row)))
        return ids_t, mask_t, len(row)

    def _candidate_ids(self, labels: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = [self.tokenizer.encode(str(label))[:256] for label in labels]
        ids, mask = _pad(rows)
        valid = torch.ones(len(rows), dtype=torch.bool)
        return ids, mask, valid

    @torch.no_grad()
    def decide(
        self,
        context: str,
        question: str,
        alternatives: Sequence[str],
        *,
        evidence_required: bool = False,
        abstain_allowed: bool = True,
    ) -> SelectReadOutput:
        if not alternatives:
            raise ValueError("at least one alternative is required")
        chunks, selected = self.select(context, question)
        if evidence_required and not selected:
            return SelectReadOutput(
                choice=-1, scores=[0.0] * len(alternatives), abstained=True,
                abstain_reason="insufficient_evidence", selected=[], reader_tokens=0,
            )
        ids, mask, reader_tokens = self._selected_ids(selected, question)
        cand_ids, cand_mask, cand_valid = self._candidate_ids(alternatives)
        logits = self.dual(
            ids.to(self.device), mask.to(self.device),
            cand_ids.to(self.device), cand_mask.to(self.device),
            cand_valid.to(self.device),
        )[0]
        scores = F.softmax(logits.float(), dim=-1).tolist()
        choice = int(torch.argmax(logits).item())
        best = float(scores[choice])
        abstained = bool(abstain_allowed and self.abstain_threshold > 0.0 and best < self.abstain_threshold)
        return SelectReadOutput(
            choice=-1 if abstained else choice,
            scores=scores,
            abstained=abstained,
            abstain_reason="insufficient_evidence" if abstained else None,
            selected=selected,
            reader_tokens=reader_tokens,
        )

    @classmethod
    def from_checkpoint(
        cls,
        reader_path: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        selector_path: str | Path | None = None,
        head_path: str | Path | None = None,
        device: str = "cpu",
        top_k: int = 8,
        max_reader_tokens: int = 4_096,
        abstain_threshold: float = 0.0,
    ) -> "SelectRead":
        reader = load_reader_checkpoint(reader_path, device=device)
        tokenizer = load_tokenizer("bpe", Path(tokenizer_path) if tokenizer_path else None)
        selector = None
        if selector_path is not None:
            # A neural selector checkpoint is optional; BM25 is the default.
            from .selector import ChunkEncoder

            payload = torch.load(selector_path, map_location=device, weights_only=False)
            scfg = EncoderConfig(**payload["config"])
            enc = Azurite(scfg)
            model = ChunkEncoder(enc)
            model.load_state_dict(payload["model"])
            selector = NeuralChunkSelector(tokenizer, model, device=device)
        engine = cls(
            reader, tokenizer, selector=selector, top_k=top_k,
            max_reader_tokens=max_reader_tokens,
            abstain_threshold=abstain_threshold, device=device,
        )
        if head_path is not None:
            import numpy as np

            data = np.load(head_path, allow_pickle=False)
            with torch.no_grad():
                engine.dual.proj.weight.copy_(torch.from_numpy(data["proj"]).to(device))
                engine.dual.log_scale.data.copy_(
                    torch.tensor(float(data["log_scale"][0]), device=device)
                )
        return engine
