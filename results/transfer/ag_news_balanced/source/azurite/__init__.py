"""Azurite — a from-scratch decision encoder.

Built and tested:

    config      the four ladder rungs and both tokenizer arms, one source of truth
    tokenizers  byte-level (exact, 256 symbols) and BPE trained on our corpus
    model       bidirectional encoder, masked-LM objective, candidate-span
                decision head, and exact parameter accounting
    env         live-memory cap, allocator reclaim, zero-swap guard

Not yet built (the next step, and the thing the L0 gate needs):

    decisions   packing (context, instruction, K candidates) from the task
                splits into `DecisionBatch` tensors, one row per decision, with
                candidate spans and a gold index

No pretrained weights and no pretrained vocabulary enter any part of this
package. Both tokenizer arms are learned from the local corpus:
`data/tokenizers/azurite_bpe_32004.json` carries a record of the sha256 of the
text it was trained on.
"""

from .config import EncoderConfig, K_BY_TASK, RUNG_DIMS
from .model import Azurite, AzuriteDecision, DecisionBatch, parameter_counts

__all__ = [
    "EncoderConfig",
    "RUNG_DIMS",
    "K_BY_TASK",
    "Azurite",
    "AzuriteDecision",
    "DecisionBatch",
    "parameter_counts",
]
