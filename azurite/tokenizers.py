"""The two tokenizer arms, both learned from our own corpus.

    byte:  4 specials + the 256 byte values. Parameter-free, no training, and
           exact round-trip by construction.
    bpe:   4 specials + 32,000 merges trained on `data/corpus/` with the
           `tokenizers` library. No downloaded vocabulary, no pretrained model
           file: `train()` writes the vocabulary it learned and records the
           corpus digest it learned it from.

The point of having two arms is that the literature has a concrete, falsifiable
claim about which one wins *by scale*: ByT5 beats subword on GLUE at Small
(80.5 vs 75.6) and Base (85.3 vs 83.0), and loses from Large upward (87.0 vs
87.6). Its Large is ~300M parameters. Our whole ladder is 2M-140M, below the
smallest published comparison point, so "where does the advantage invert?" is a
question nobody has answered in this regime.

Measured on this corpus, 300 held-out documents: **3.327 bytes per BPE token**,
0 round-trip failures in either arm. That is why `config.MAX_SEQ` gives the BPE
arm 308 positions against the byte arm's 1,024: the two windows must hold the
same amount of text, or the comparison is between text volumes rather than
between tokenizers.

Special token layout is identical in both arms so the model, the packing code
and the decision head are arm-agnostic:

    0 <pad>   1 <mask>   2 <cls>   3 <sep>   [4..] alphabet
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Protocol

from .config import N_SPECIALS, SPECIAL_TOKENS, VOCAB_SIZE

PAD_ID, MASK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3

TOKENIZER_DIR = Path("data/tokenizers")


class Tokenizer(Protocol):
    arm: str
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Iterable[int]) -> str: ...


def _specials() -> list[str]:
    return [
        SPECIAL_TOKENS["pad"],
        SPECIAL_TOKENS["mask"],
        SPECIAL_TOKENS["cls"],
        SPECIAL_TOKENS["sep"],
    ]


class ByteTokenizer:
    """256-byte alphabet with four specials in front. Exact and reversible.

    `decode(encode(s)) == s` for every valid Python string, which is a property
    a test can and does assert on real corpus lines rather than on a toy sample.
    """

    arm = "byte"
    vocab_size = VOCAB_SIZE["byte"]

    def encode(self, text: str, *, add_specials: bool = False) -> list[int]:
        ids = [b + N_SPECIALS for b in text.encode("utf-8")]
        if add_specials:
            return [CLS_ID, *ids, SEP_ID]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        raw = bytes(i - N_SPECIALS for i in ids if i >= N_SPECIALS)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: Path | None = None) -> Path:
        """Write a descriptor. There is nothing to learn, and saying so is the
        point: this file records that fact rather than implying a vocabulary."""
        path = path or (TOKENIZER_DIR / "azurite_byte.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "arm": self.arm,
                    "vocab_size": self.vocab_size,
                    "alphabet": "raw UTF-8 bytes",
                    "specials": _specials(),
                    "learned_from": None,
                    "note": "no vocabulary is learned or downloaded in this arm",
                },
                indent=1,
            )
            + "\n"
        )
        return path


class BpeTokenizer:
    """A BPE vocabulary trained on our corpus.

    Wraps a `tokenizers.Tokenizer`. The wrapped object is the *only* thing in
    the repo that needs that library, which keeps the dependency surface small.
    """

    arm = "bpe"

    def __init__(self, inner, vocab_size: int) -> None:
        self._tok = inner
        self.vocab_size = vocab_size

    def encode(self, text: str, *, add_specials: bool = False) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_specials:
            return [CLS_ID, *ids, SEP_ID]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))
        return path

    @classmethod
    def load(cls, path: Path) -> "BpeTokenizer":
        from tokenizers import Tokenizer as HFTokenizer

        inner = HFTokenizer.from_file(str(path))
        return cls(inner, inner.get_vocab_size())

    @classmethod
    def train(
        cls,
        corpus_files: Iterable[Path],
        *,
        vocab_size: int = VOCAB_SIZE["bpe"],
        out_path: Path | None = None,
        sample_bytes: int | None = None,
        seed: int = 0,
    ) -> tuple["BpeTokenizer", dict]:
        """Train BPE on the corpus, and return the tokenizer with its record.

        The record carries a sha256 of the exact bytes fed to the trainer and the
        byte count, so the vocabulary's provenance is auditable from the file
        alone. `sample_bytes` bounds the training input; when it is set the
        digest covers the sample, and the record says so.
        """
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers import decoders, models, pre_tokenizers, trainers

        files = [Path(f) for f in corpus_files]
        if not files:
            raise ValueError("no corpus files given")

        h = hashlib.sha256()
        n_bytes = 0
        texts: list[str] = []
        for f in files:
            with open(f, "rb") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = rec.get("text")
                    if not isinstance(text, str):
                        continue
                    texts.append(text)
                    h.update(text.encode("utf-8"))
                    h.update(b"\n")
                    n_bytes += len(text.encode("utf-8"))
                    if sample_bytes is not None and n_bytes >= sample_bytes:
                        break
            if sample_bytes is not None and n_bytes >= sample_bytes:
                break
        if not texts:
            raise ValueError(f"no text found in {[str(f) for f in files]}")

        inner = HFTokenizer(models.BPE(unk_token=None))
        inner.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        inner.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=_specials(),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        inner.train_from_iterator(texts, trainer=trainer)

        tok = cls(inner, inner.get_vocab_size())
        out_path = out_path or (TOKENIZER_DIR / f"azurite_bpe_{vocab_size}.json")
        tok.save(out_path)
        record = {
            "arm": "bpe",
            "out_path": str(out_path),
            "vocab_size": tok.vocab_size,
            "corpus_files": [str(f) for f in files],
            "documents": len(texts),
            "bytes": n_bytes,
            "sha256_of_training_text": h.hexdigest(),
            "sample_bytes_cap": sample_bytes,
            "seed": seed,
            "learned_from": "local corpus, trained here",
        }
        rec_path = out_path.with_suffix(out_path.suffix + ".record.json")
        rec_path.write_text(json.dumps(record, indent=1) + "\n")
        return tok, record


def load(arm: str, path: Path | None = None) -> ByteTokenizer | BpeTokenizer:
    if arm == "byte":
        return ByteTokenizer()
    if arm == "bpe":
        if path is None:
            matches = sorted(TOKENIZER_DIR.glob("azurite_bpe_*.json"))
            matches = [m for m in matches if not m.name.endswith(".record.json")]
            if not matches:
                raise FileNotFoundError(
                    f"no trained BPE in {TOKENIZER_DIR}; run "
                    "`scripts/train_tokenizers.py` first"
                )
            path = matches[-1]
        return BpeTokenizer.load(path)
    raise KeyError(f"unknown arm {arm!r}")
