"""A dual head whose context vector is pooled around the customer message.

`AzuriteDual` mean-pools every context token. On a retrieved-example context the
message is a few percent of the tokens, so the pooled vector is dominated by
the examples. This head keeps the message as a separate encoder input and uses
its embedding as the query for a masked attention pool over the full context:

    q = W_q(mean_pool(encode(message)))
    a = softmax((H q) / sqrt(d)) over real context tokens
    c = sum(a * H)

Candidates are embedded exactly as in `AzuriteDual`, so the only differences are
the query pass and the attention pool. The encoder is shared.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from azurite.model import Azurite


class QueryDual(nn.Module):
    def __init__(self, encoder: Azurite) -> None:
        super().__init__()
        self.encoder = encoder
        d = encoder.cfg.d_model
        self.proj = nn.Linear(d, d, bias=False)
        self.query = nn.Linear(d, d, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @staticmethod
    def _mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    def embed_candidates(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self._mean(self.encoder(ids, mask), mask)
        return F.normalize(torch.tanh(self.proj(pooled)), dim=-1)

    def embed_context(
        self,
        ctx_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        msg_ids: torch.Tensor,
        msg_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(ctx_ids, ctx_mask)
        message = self._mean(self.encoder(msg_ids, msg_mask), msg_mask)
        query = self.query(message)
        scores = (hidden * query.unsqueeze(1)).sum(-1) / math.sqrt(hidden.shape[-1])
        scores = scores.masked_fill(~ctx_mask, float("-inf"))
        attention = torch.softmax(scores, dim=1)
        pooled = (hidden * attention.unsqueeze(-1)).sum(1)
        return F.normalize(torch.tanh(self.proj(pooled)), dim=-1)

    def forward(
        self,
        ctx_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        msg_ids: torch.Tensor,
        msg_mask: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self.embed_context(ctx_ids, ctx_mask, msg_ids, msg_mask)
        cand = self.embed_candidates(cand_ids, cand_mask)
        return self.log_scale.exp() * (ctx @ cand.T)
