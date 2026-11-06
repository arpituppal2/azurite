"""End-to-end prediction: message -> retrieved examples -> query head -> intent.

The predictor is deliberately narrow. It answers one question, *which of the 77
Banking77 intents does this customer message express*, with the decontaminated
recipe that scores 2,743/3,080 = 89.06% on all official rows (95% Wilson
87.91-90.11). The shipped bank removes 25 evaluation texts from the 9,993-row
source split (6 exact); the original bank scores 89.03%, but 255 rows retrieve
at least one test text, so that number is reported only with that disclosure.
The 2,825 rows that never retrieve test text score 88.88%. The predictor does
not accept images, does not use a 5,120-byte window by default, and does not
pretend its softmax scores are calibrated probabilities.

The inference path mirrors `azurite.jev` exactly: the same byte tokenizer, the
same 77-label candidate order, `build_context` with complete examples only, and
the same scaled-cosine query head. The one deliberate difference is the source
of examples. Evaluation uses the frozen Jev example set stored in
`results/jev_matched/inputs.json`; a product call has no such file, so examples
come from local BM25 over the bundled 9,968-message decontaminated support split.

`verify_frozen` keeps both worlds honest: it feeds the frozen examples and
compares the resulting context digest and prediction against a stored evaluation
record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from azurite import checkpoints, jev
from azurite.artifacts import sha256_text
from azurite.bundle import Bundle, BundleError, Checkpoint, load_bundle
from azurite.config import EncoderConfig
from azurite.decisions import _pad
from azurite.retrieval import BM25, build_context, select_examples
from azurite.tokenizers import load as load_tokenizer


class PredictionError(RuntimeError):
    """The model or its inputs cannot produce a valid prediction."""


@dataclass(frozen=True)
class Prediction:
    text: str
    intent: str
    intent_index: int
    confidence: float
    margin: float
    top: tuple[dict, ...]
    context: str
    examples_used: int
    examples_retrieved: bool
    checkpoint: str
    checkpoint_sha256: str
    device: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "intent": self.intent,
            "intent_index": self.intent_index,
            "confidence": self.confidence,
            "margin": self.margin,
            "top": list(self.top),
            "context_bytes": len(self.context.encode("utf-8")),
            "context_sha256": sha256_text(self.context),
            "examples_used": self.examples_used,
            "examples_retrieved": self.examples_retrieved,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
        }


def choose_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _truncate_utf8(text: str, budget: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= budget:
        return text
    return raw[:budget].decode("utf-8", errors="ignore")


class Predictor:
    """A loaded checkpoint plus the bundle needed to use it."""

    def __init__(
        self,
        model,
        tokenizer,
        bundle: Bundle,
        checkpoint: Checkpoint,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.bundle = bundle
        self.checkpoint = checkpoint
        self.device = device
        self.scale = float(model.log_scale.exp().item())
        support = tuple(dict(row, text=row["message"]) for row in bundle.support)
        self._retriever: BM25 | None = None
        self._support = support
        with torch.no_grad():
            _, candidate_ids, candidate_mask = jev.candidate_bank(
                bundle.labels, tokenizer
            )
            self.candidate_embeddings = model.embed_candidates(
                candidate_ids.to(device), candidate_mask.to(device)
            )

    @classmethod
    def load(
        cls,
        checkpoint_id: str | None = None,
        *,
        device: str = "auto",
        home: Path | None = None,
    ) -> "Predictor":
        try:
            resolved = checkpoints.resolve(checkpoint_id, home=home)
        except checkpoints.CheckpointError as exc:
            raise PredictionError(str(exc)) from exc
        target = choose_device(device)
        try:
            model, state = jev.load_query_head(resolved.path, target)
        except SystemExit as exc:
            raise PredictionError(str(exc)) from exc
        config = EncoderConfig(**state["config"])
        if config.max_seq != resolved.checkpoint.max_seq:
            raise PredictionError(
                f"checkpoint {resolved.checkpoint.id!r} declares max_seq "
                f"{resolved.checkpoint.max_seq}, weights use {config.max_seq}"
            )
        if resolved.checkpoint.context_budget + 2 > config.max_seq:
            raise PredictionError(
                f"context budget {resolved.checkpoint.context_budget} does not fit "
                f"a {config.max_seq}-position window"
            )
        bundle = load_bundle()
        try:
            resolved.checkpoint.require_task(bundle.task)
        except BundleError as exc:
            raise PredictionError(str(exc)) from exc
        tokenizer = load_tokenizer("byte")
        if tokenizer.vocab_size != config.vocab_size:
            raise PredictionError(
                f"byte tokenizer has {tokenizer.vocab_size} symbols, "
                f"weights expect {config.vocab_size}"
            )
        return cls(model, tokenizer, bundle, resolved.checkpoint, target)

    def retrieve(
        self, text: str, *, limit: int | None = None, per_label: int = 4
    ) -> list[Mapping[str, object]]:
        if self._retriever is None:
            self._retriever = BM25(list(self._support))
        chosen = select_examples(
            self._retriever,
            text,
            exclude_id=None,
            limit=limit or self.checkpoint.examples,
            per_label=per_label,
        )
        return chosen

    def make_context(
        self,
        text: str,
        *,
        examples: Sequence[Mapping[str, object]] | None = None,
        truncate: bool = True,
    ) -> tuple[str, str, int]:
        """Return (packed context, message actually encoded, examples used)."""
        chosen = list(self.retrieve(text)) if examples is None else list(examples)
        used = len(chosen)
        message = text
        if truncate:
            # `build_context` can exceed the budget when the message alone fills
            # it. Truncate the message, never a half example: the measured arm
            # only ever contained complete `message => intent` lines.
            head_bytes = len("Customer message: \nLabeled examples:\n".encode("utf-8"))
            room = max(1, self.checkpoint.context_budget - head_bytes)
            message = _truncate_utf8(text, room)
        if chosen:
            context, used = build_context(
                message, chosen, budget=self.checkpoint.context_budget
            )
        else:
            context, used = message, 0
        if used == 0 and chosen:
            # The head itself did not leave room for one complete example.
            context = message
        return context, message, used

    @torch.no_grad()
    def score(
        self,
        contexts: Sequence[str],
        messages: Sequence[str],
        *,
        batch: int = 8,
    ) -> list[list[float]]:
        if len(contexts) != len(messages):
            raise PredictionError("contexts and messages must have equal length")
        out: list[list[float]] = []
        for start in range(0, len(contexts), batch):
            stop = start + batch
            ctx_text = contexts[start:stop]
            msg_text = messages[start:stop]
            ctx_ids, ctx_mask = _pad(
                [self.tokenizer.encode(text, add_specials=True) for text in ctx_text]
            )
            msg_ids, msg_mask = _pad(
                [self.tokenizer.encode(text, add_specials=True) for text in msg_text]
            )
            if ctx_ids.shape[1] > self.checkpoint.max_seq:
                raise PredictionError(
                    f"context is {ctx_ids.shape[1]} tokens, checkpoint window is "
                    f"{self.checkpoint.max_seq}"
                )
            embedded = self.model.embed_context(
                ctx_ids.to(self.device),
                ctx_mask.to(self.device),
                msg_ids.to(self.device),
                msg_mask.to(self.device),
            )
            logits = self.scale * (embedded @ self.candidate_embeddings.T)
            out.extend(logits.float().cpu().tolist())
        return out

    def predict(
        self,
        text: str,
        *,
        top_k: int = 5,
        examples: Sequence[Mapping[str, object]] | None = None,
        truncate: bool = True,
    ) -> Prediction:
        if not text.strip():
            raise PredictionError("text is empty")
        if not 1 <= top_k <= len(self.bundle.labels):
            raise PredictionError(f"top_k must be between 1 and {len(self.bundle.labels)}")
        context, message, used = self.make_context(
            text, examples=examples, truncate=truncate
        )
        logits = self.score([context], [message], batch=1)[0]
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).tolist()
        ranked = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
        top = tuple(
            {
                "rank": rank,
                "intent": self.bundle.labels[index],
                "intent_index": index,
                "score": probs[index],
            }
            for rank, index in enumerate(ranked[:top_k], 1)
        )
        winner = ranked[0]
        runner_up = ranked[1]
        margin = probs[winner] - probs[runner_up]
        return Prediction(
            text=text,
            intent=self.bundle.labels[winner],
            intent_index=winner,
            confidence=probs[winner],
            margin=margin,
            top=top,
            context=context,
            examples_used=used,
            examples_retrieved=examples is None,
            checkpoint=self.checkpoint.id,
            checkpoint_sha256=self.checkpoint.sha256,
            device=str(self.device),
        )

    def verify_frozen(
        self,
        rows: Sequence[Mapping[str, object]],
        records: Sequence[Mapping[str, object]],
        *,
        batch: int = 8,
    ) -> list[dict]:
        """Reproduce stored evaluation rows from their frozen examples.

        The context digest is checked before inference: if `build_context`
        drifted, the comparison fails on identity rather than quietly scoring a
        different prompt.
        """
        if len(rows) != len(records):
            raise PredictionError("rows and records must have equal length")
        contexts, messages, used, digests = [], [], [], []
        for row, record in zip(rows, records):
            context, message, count = self.make_context(
                str(row["text"]), examples=row["examples"], truncate=False
            )
            digest = sha256_text(context)
            if digest != record["context_sha256"]:
                raise PredictionError(
                    f"{row['id']}: rebuilt context digest {digest} does not match "
                    f"stored {record['context_sha256']}"
                )
            contexts.append(context)
            messages.append(message)
            used.append(count)
            digests.append(digest)
        logits = self.score(contexts, messages, batch=batch)
        results = []
        for row, record, digest, count, scores in zip(
            rows, records, digests, used, logits
        ):
            pred = int(max(range(len(scores)), key=lambda i: scores[i]))
            expected = int(record["pred_index"])
            results.append(
                {
                    "id": row["id"],
                    "context_sha256": digest,
                    "stored_pred": expected,
                    "cli_pred": pred,
                    "match": int(pred == expected),
                    "stored_correct": int(record["correct"]),
                    "cli_correct": int(pred == int(row["gold_index"])),
                    "gold_index": int(row["gold_index"]),
                    "gold": row["gold"],
                    "examples_used": count,
                }
            )
        return results
