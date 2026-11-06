"""Size presets for the scaling ladder.

The four rungs use the **same dimensions that were measured** in
`results/local_bench/bench_20261027T030406.json`, so a throughput measured on
this implementation is directly comparable to that measurement rather than
needing a correction factor.

    rung    d_model  n_layer  n_head  d_ff
    tiny        192        4       6   768
    small       384        6      12  1536
    mid         512        8      16  2048
    large       768       12      24  3072

Caveat, stated because it matters for the wall-clock estimates: the bench
binary that produced the tokens/s figures has been deleted (working tree reset),
so its module inventory cannot be inspected. The parameter counts printed by
`scripts/model_smoke.py` are therefore *this* implementation's counts at those
dimensions, and they are NOT expected to match the bench's counts to the digit.
Throughput is re-measured on the real implementation before any ladder run is
scheduled; `scripts/budget.py` will be re-run against that measurement rather
than the old one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class EncoderConfig:
    """A pre-norm bidirectional encoder.

    Design choices, each with a reason:

    - **Pre-norm RMSNorm**: standard for stability at small batch, and the
      small-scale-instability literature recommends normalising the query/key
      path. `qk_norm` does that explicitly.
    - **SwiGLU FFN with `d_ff` as the inner dimension**: the gated FFN is what
      the modern encoders use; `d_ff` is the *inner* width, so FFN parameters
      are `3 * d_model * d_ff`.
    - **RoPE, no learned positional embedding**: zero parameters, and it makes
      the context-length extension question a config change instead of a
      retrain.
    - **Tied MLM head**: the masked-LM projection shares the input embedding
      matrix. At a 32k BPE vocabulary an untied head is 24.6M extra parameters
      at the largest rung, which would corrupt the parameter-matched comparison
      the whole study rests on.
    - **No biases in the linear layers**: BERT-era legacy, and removing them
      makes the parameter accounting exact and auditable.
    """

    d_model: int
    n_layer: int
    n_head: int
    d_ff: int
    vocab_size: int
    max_seq: int
    rope_theta: float = 10_000.0
    qk_norm: bool = True
    tie_mlm_head: bool = True

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_head:
            raise ValueError(
                f"d_model {self.d_model} not divisible by n_head {self.n_head}"
            )
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


# The ladder. Dims are the measured ones; vocab_size and max_seq are per-arm.
RUNG_DIMS: dict[str, dict[str, int]] = {
    "tiny": {"d_model": 192, "n_layer": 4, "n_head": 6, "d_ff": 768},
    "small": {"d_model": 384, "n_layer": 6, "n_head": 12, "d_ff": 1536},
    "mid": {"d_model": 512, "n_layer": 8, "n_head": 16, "d_ff": 2048},
    "large": {"d_model": 768, "n_layer": 12, "n_head": 24, "d_ff": 3072},
}

# Context windows per arm, matched on TEXT, not on positions.
#
# Measured on this corpus (`scripts/train_tokenizers.py`, 300 held-out
# documents): the BPE arm averages **3.327 bytes per token**, so 1,024 byte
# positions and 308 BPE tokens both hold ~1,024 bytes of text. A 256-token BPE
# window would hold 852 bytes — 17% less text per example than the byte arm,
# which would silently hand the byte arm an advantage in any comparison that
# did not account for it.
#
# ByT5 reported 4.1 bytes/token on mC4 (range 2.5-9.0 by language); the ratio is
# corpus-specific, which is why it is measured here rather than cited.
MAX_SEQ = {"byte": 1024, "bpe": 308}

# 4 specials (pad, mask, cls, sep) plus the alphabet: 256 bytes, or a 32k BPE
# trained on our own corpus. Neither vocabulary is pretrained or downloaded.
VOCAB_SIZE = {"byte": 260, "bpe": 32_004}
N_SPECIALS = 4


def rung(name: str, arm: str, max_seq: int | None = None) -> EncoderConfig:
    """Build the config for one ladder rung in one tokenizer arm.

    `max_seq` overrides the arm default, and exists because the context-length
    axis trains the same rung at 256, 1024 and 4096 positions. Without it,
    `pretrain.py --seq 4096` built a 1024-position RoPE table and died on the
    first batch with a tensor-shape error that read like a model bug.
    """
    if name not in RUNG_DIMS:
        raise KeyError(f"unknown rung {name!r}; have {sorted(RUNG_DIMS)}")
    if arm not in MAX_SEQ:
        raise KeyError(f"unknown arm {arm!r}; have {sorted(MAX_SEQ)}")
    dims = RUNG_DIMS[name]
    return EncoderConfig(
        vocab_size=VOCAB_SIZE[arm], max_seq=max_seq or MAX_SEQ[arm], **dims
    )


# Number of candidate options per task, used to dimension the decision head's
# packing and to refuse accidental averaging across very different K.
K_BY_TASK: dict[str, int] = {
    "banking77": 77,
    "clinc_oos": 150,
    "ag_news": 4,
    "emotion": 6,
    "massive_intent": 60,
    "massive_scenario": 18,
    "xnli": 3,
    "language_id": 12,
    "financial_phrasebank": 3,
}

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "cls": "<cls>",
    "sep": "<sep>",
    "mask": "<mask>",
}
