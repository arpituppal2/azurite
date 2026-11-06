"""Azurite: a bidirectional encoder plus a candidate-span decision head.

Two objects, deliberately separate:

    Azurite           the encoder. Pretrained with masked language modelling.
    AzuriteDecision    the encoder plus a head that scores K candidate spans in
                      one packed sequence, softmax over candidates.

Why a bidirectional encoder and not a decoder: for small supervised decisions
the published evidence favours bidirectional attention and task-specific
fine-tuning over causal decoding (see the text-classification review in the
prior-art note), and ModernBERT — the architecture this project is priced
against — is an encoder. Matching the family removes one confound from the
comparison; it does not remove the data confound, which is the one that matters.

Why candidate spans in one sequence rather than K forward passes: it makes the
per-decision cost one pass, it keeps every rung's latency comparable to the
baseline's single `Router.predict` call, and it lets one packing routine serve
the 3-option and 150-option tasks identically.

`parameter_counts` is analytic and `Azurite.num_parameters` is measured from the
instantiated modules. A test asserts they agree, because a parameter count that
is asserted rather than derived is how a "matched-scale" comparison quietly
stops being matched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import EncoderConfig
from .tokenizers import CLS_ID, MASK_ID, PAD_ID, SEP_ID


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class RoPE(nn.Module):
    """Rotary position embedding as a precomputed cos/sin table.

    No parameters, and its presence is a deliberate cost saving: a learned
    positional table would be `max_seq * d_model` parameters, which at the byte
    arm's 1,024 positions is 15% of the tiny rung — a confound between the two
    tokenizer arms, which have different sequence lengths.
    """

    def __init__(self, d_head: int, max_seq: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if d_head % 2:
            raise ValueError(f"d_head must be even for RoPE, got {d_head}")
        self.d_head = d_head
        self.theta = theta
        inv = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
        pos = torch.arange(max_seq).float()
        freqs = torch.outer(pos, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, T, Dh) -> rotated by absolute position.

        Named `rotate` and not `apply`: `nn.Module.apply` is how weights get
        initialised, and a method of the same name here shadows it, which fails
        silently at construction time and only shows up as an untrained model.
        """
        t = x.shape[-2]
        if t > self.cos.shape[0]:
            # Grow the table rather than slicing a shorter one: the context
            # axis runs this rung at 4096 positions, and slicing produced a
            # broadcast error deep in the forward pass instead of a fix here.
            inv = 1.0 / (self.theta ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
            freqs = torch.outer(torch.arange(t).float(), inv)
            self.cos = freqs.cos().to(self.cos.device)
            self.sin = freqs.sin().to(self.sin.device)
        cos = self.cos[:t].view(1, 1, t, -1)
        sin = self.sin[:t].view(1, 1, t, -1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.qk_norm = (
            nn.ModuleList([RMSNorm(self.d_head), RMSNorm(self.d_head)])
            if cfg.qk_norm
            else None
        )

    def forward(
        self, x: torch.Tensor, rope: RoPE, pad_mask: torch.Tensor | None
    ) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, Dh)
        if self.qk_norm is not None:
            q = self.qk_norm[0](q)
            k = self.qk_norm[1](k)
        q, k = rope.rotate(q), rope.rotate(k)

        attn_mask = None
        if pad_mask is not None:
            # SDPA wants True = keep. pad_mask is True for real tokens.
            attn_mask = pad_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).reshape(b, t, self.n_head * self.d_head)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        d, inner = cfg.d_model, cfg.d_ff
        self.gate = nn.Linear(d, inner, bias=False)
        self.up = nn.Linear(d, inner, bias=False)
        self.down = nn.Linear(inner, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(
        self, x: torch.Tensor, rope: RoPE, pad_mask: torch.Tensor | None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, pad_mask)
        return x + self.ffn(self.norm2(x))


# --------------------------------------------------------------------------
# The encoder
# --------------------------------------------------------------------------


class Azurite(nn.Module):
    """The pretrainable encoder."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.rope = RoPE(cfg.d_head, cfg.max_seq, cfg.rope_theta)
        # Masked-LM head. Tied by default (see config docstring).
        self.mlm = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_mlm_head:
            self.mlm.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            with torch.no_grad():
                m.weight[PAD_ID].zero_()

    def forward(
        self, ids: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if pad_mask is None:
            pad_mask = ids.ne(PAD_ID)
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, self.rope, pad_mask)
        return self.norm(x)

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def mlm_loss(
        self, ids: torch.Tensor, *, mask_prob: float = 0.30, generator=None
    ) -> tuple[torch.Tensor, dict]:
        """Masked-LM loss with 80/10/10 corruption.

        Masking rate is 30%, following MosaicBERT's encoder recipe (BERT's 15%
        was tuned for a much smaller token budget); at 30% a 0.1B-token run sees
        roughly twice as many gradient-bearing positions for the same compute.
        """
        if generator is not None and generator.device.type != ids.device.type:
            # Silently substituting the global RNG here would make runs
            # unreproducible while looking seeded; fail instead.
            raise ValueError(
                f"generator on {generator.device.type!r} but ids on "
                f"{ids.device.type!r}; construct torch.Generator(device=...)"
            )
        pad_mask = ids.ne(PAD_ID)
        eligible = pad_mask
        rand = torch.rand(ids.shape, device=ids.device, generator=generator)
        selected = (rand < mask_prob) & eligible
        # Guarantee at least one masked position per row when possible.
        empty = ~selected.any(dim=1) & eligible.any(dim=1)
        if empty.any():
            first = eligible.float().argmax(dim=1)
            selected[empty, first[empty]] = True
        labels = ids.clone()
        labels[~selected] = -100

        corrupt = torch.rand(ids.shape, device=ids.device, generator=generator)
        inputs = ids.clone()
        m_mask = selected & (corrupt < 0.8)
        m_rand = selected & (corrupt >= 0.8) & (corrupt < 0.9)
        inputs[m_mask] = MASK_ID
        if m_rand.any():
            inputs[m_rand] = torch.randint(
                0, self.cfg.vocab_size, (int(m_rand.sum()),), device=ids.device
            )
        hidden = self.forward(inputs, pad_mask)
        n_masked = int(selected.sum())
        if n_masked == 0:
            # Cross-entropy over an all-ignore_index target is NaN, and a NaN
            # loss poisons every parameter on the backward pass. A batch with
            # nothing to predict contributes zero gradient instead.
            zero = hidden.sum() * 0.0
            return zero, {"masked_positions": 0, "loss": 0.0}

        # Project ONLY the masked positions. Materialising logits over every
        # position costs B*T*V floats, which at a 32k vocabulary is the real
        # memory bottleneck: 32x256x32004 is 1.05 GB of fp32 for one batch at
        # the smallest rung, and it made the BPE arm OOM where the byte arm fit
        # at 2.1 GiB. Gathering first is the same loss, arithmetically — the
        # ignored positions contribute no term either way.
        rows, cols = selected.nonzero(as_tuple=True)
        logits = self.mlm(hidden[rows, cols])
        loss = F.cross_entropy(logits, ids[rows, cols])
        stats = {
            "masked_positions": n_masked,
            "loss": float(loss.detach()),
        }
        return loss, stats


# --------------------------------------------------------------------------
# The decision head
# --------------------------------------------------------------------------


@dataclass
class DecisionBatch:
    """A packed batch of decisions.

    ids          (B, T)      token ids: ctx + <sep> + cand_1 + <sep> + ...
    pad_mask     (B, T)      True for real tokens
    spans        (B, K, 2)   (start, end) token indices per candidate, end-excl
    cand_mask    (B, K)      True for a real candidate
    gold         (B,)        index of the correct candidate
    """

    ids: torch.Tensor
    pad_mask: torch.Tensor
    spans: torch.Tensor
    cand_mask: torch.Tensor
    gold: torch.Tensor

    def to(self, device: torch.device) -> "DecisionBatch":
        return DecisionBatch(
            ids=self.ids.to(device),
            pad_mask=self.pad_mask.to(device),
            spans=self.spans.to(device),
            cand_mask=self.cand_mask.to(device),
            gold=self.gold.to(device),
        )


class AzuriteDecision(nn.Module):
    """Encoder + span-pooling scorer over K candidates."""

    def __init__(self, encoder: Azurite) -> None:
        super().__init__()
        self.encoder = encoder
        self.span_proj = nn.Linear(encoder.cfg.d_model, encoder.cfg.d_model, bias=False)
        self.score = nn.Linear(encoder.cfg.d_model, 1, bias=True)

    def forward(self, batch: DecisionBatch) -> torch.Tensor:
        hidden = self.encoder(batch.ids, batch.pad_mask)
        pooled = self._pool_spans(hidden, batch.spans, batch.cand_mask)
        logits = self.score(torch.tanh(self.span_proj(pooled))).squeeze(-1)
        # Invalid candidates must never win. -inf, not a large negative number:
        # a finite value leaks probability mass into the softmax and would show
        # up as a small but systematic loss floor.
        return logits.masked_fill(~batch.cand_mask, float("-inf"))

    def _pool_spans(
        self, hidden: torch.Tensor, spans: torch.Tensor, cand_mask: torch.Tensor
    ) -> torch.Tensor:
        b, t, d = hidden.shape
        k = spans.shape[1]
        idx = torch.arange(t, device=hidden.device).view(1, 1, t)
        start = spans[..., 0].unsqueeze(-1)
        end = spans[..., 1].unsqueeze(-1)
        inside = (idx >= start) & (idx < end)  # (B, K, T)
        inside = inside & cand_mask.unsqueeze(-1)
        weights = inside.to(hidden.dtype)
        denom = weights.sum(-1, keepdim=True).clamp(min=1.0)
        pooled = torch.einsum("bkt,btd->bkd", weights, hidden) / denom
        return pooled

    @staticmethod
    def loss(logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, gold)

    @torch.no_grad()
    def probabilities(self, batch: DecisionBatch) -> torch.Tensor:
        logits = self.forward(batch)
        return F.softmax(logits, dim=-1)


class AzuriteDual(nn.Module):
    """Dual encoder: score a context against a fixed candidate set.

    This is the path that works at any K. Candidate embeddings are computed once
    per checkpoint and reused across every example, so a 150-option decision
    costs one context pass plus a 150-way dot product — the same shape of cost a
    serving product would have, and the only honest way to compare latency with
    a vendor that answers 77 options in 283 ms.

    Scoring is a scaled cosine similarity, which bounds logits and keeps the
    softmax from saturating before the encoder has learned anything.
    """

    def __init__(self, encoder: Azurite) -> None:
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(encoder.cfg.d_model, encoder.cfg.d_model, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    def embed(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self._pool(self.encoder(ids, mask), mask)
        return F.normalize(torch.tanh(self.proj(pooled)), dim=-1)

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_token_mask: torch.Tensor,
        cand_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score every valid candidate.

        context_ids (B,T), context_mask (B,T) — per token
        cand_ids (K,L), cand_token_mask (K,L) — per token, for padding
        cand_valid (K,) — per candidate, which entries exist

        The two masks are separate parameters on purpose: they were once the
        same argument here, and passing the token mask where the candidate mask
        belonged produced an `expand` error whose message pointed at the wrong
        tensor.
        """
        if cand_token_mask.shape != cand_ids.shape:
            raise ValueError(
                f"cand_token_mask {tuple(cand_token_mask.shape)} must match "
                f"cand_ids {tuple(cand_ids.shape)}; the per-candidate mask is "
                "`cand_valid`"
            )
        if cand_valid is None:
            cand_valid = torch.ones(cand_ids.shape[0], dtype=torch.bool, device=cand_ids.device)
        ctx = self.embed(context_ids, context_mask)
        cand = self.embed(cand_ids, cand_token_mask)
        logits = self.log_scale.exp() * (ctx @ cand.T)
        return logits.masked_fill(~cand_valid.unsqueeze(0), float("-inf"))

    @torch.no_grad()
    def candidate_embeddings(
        self, cand_ids: torch.Tensor, cand_token_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.embed(cand_ids, cand_token_mask)


# --------------------------------------------------------------------------
# Exact accounting
# --------------------------------------------------------------------------


def parameter_counts(cfg: EncoderConfig) -> dict[str, int]:
    """Analytic parameter counts. A test asserts these against the modules."""
    d, L, inner = cfg.d_model, cfg.n_layer, cfg.d_ff
    embed = cfg.vocab_size * d
    per_block = 4 * d * d  # qkv + out projections
    per_block += 3 * d * inner  # gated FFN
    per_block += 2 * d  # two RMSNorms
    if cfg.qk_norm:
        # QK-norm operates per head, so it is 2 * d_head per block, not 2 * d.
        # This was wrong until the accounting test caught it: the error is
        # 1,536 parameters per block at the large rung, which is small but
        # systematic across every rung and every token budget.
        per_block += 2 * cfg.d_head
    encoder = embed + L * per_block + d  # + final norm
    mlm_extra = 0 if cfg.tie_mlm_head else cfg.vocab_size * d
    decision = d * d + d + 1  # span projection + score weight + score bias
    dual = d * d + 1  # projection + log scale
    return {
        "encoder": encoder,
        "mlm_head_extra": mlm_extra,
        "pretrain_total": encoder + mlm_extra,
        "decision_total": encoder + decision,
        "dual_total": encoder + dual,
        "blocks": per_block * L,
        "embeddings": embed,
    }
