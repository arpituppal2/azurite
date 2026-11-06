"""Azurite — a from-scratch byte-level decision encoder.

The packaged product answers one question: which of the 77 Banking77 intents
does this English customer message express? The shipped retrieval bank is
decontaminated: 9,968 of the 9,993 source rows, after removing 25 evaluation
texts (6 exact). It scores 2,743/3,080 = 89.06% (95% Wilson 87.91-90.11). The
original 9,993-row bank scores 2,742/3,080 = 89.03%, but 255 of the 3,080 rows
retrieve at least one test text, so that number is reported only with that
disclosure. The 2,825 rows that never retrieve test text score 88.88%. It
remains 3.38 points below the Jev comparator; see `MODEL_CARD.md` for the
mechanism, the negative result and the limits.

    from azurite import Predictor
    predictor = Predictor.load()
    print(predictor.predict("My card was swallowed by the ATM").intent)

Core modules:

    config       four ladder rungs and both tokenizer arms, one source of truth
    tokenizers   byte-level (exact) and BPE trained on this project's corpus
    model        bidirectional encoder, masked-LM objective, candidate-span
                 decision head, dual encoder, exact parameter accounting
    decisions    packing (context, instruction, K candidates) into batches
    bundle       packaged labels, support index and checkpoint registry
    checkpoints  content-addressed cache; weights resolve by SHA-256
    predictor    end-to-end message -> retrieved examples -> intent
    cli          the `azurite` console entry point

No pretrained weights and no pretrained vocabulary enter any part of this
package. Both tokenizer arms are learned from the local corpus, and the
classifier checkpoint is trained from random initialisation.
"""

from .bundle import Bundle, BundleError, Checkpoint, load_bundle, load_registry
from .config import EncoderConfig, K_BY_TASK, RUNG_DIMS
from .model import Azurite, AzuriteDecision, DecisionBatch, parameter_counts
from .predictor import Prediction, PredictionError, Predictor

__all__ = [
    "Azurite",
    "AzuriteDecision",
    "Bundle",
    "BundleError",
    "Checkpoint",
    "DecisionBatch",
    "EncoderConfig",
    "K_BY_TASK",
    "Prediction",
    "PredictionError",
    "Predictor",
    "RUNG_DIMS",
    "load_bundle",
    "load_registry",
    "parameter_counts",
]
