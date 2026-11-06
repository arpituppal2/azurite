"""Tests for the pretraining data path.

The invariant that matters most: **training must never read the held-out tail.**
If it does, every bits/byte number is optimistic and the L1 gate is meaningless.
That is checked directly against the sampler, not inferred from a comment.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from pretrain import HELDOUT_FRACTION, batches, prepare  # noqa: E402
from azurite.tokenizers import CLS_ID, MASK_ID, PAD_ID, SEP_ID  # noqa: E402


def write_corpus(directory: Path, docs: int = 12, size: int = 400) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "shard.000.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for i in range(docs):
            fh.write(
                json.dumps(
                    {"text": f"document {i} " + ("alpha beta gamma delta " * (size // 23))}
                )
                + "\n"
            )
    return path


def test_prepare_writes_the_requested_length_plus_heldout(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus, docs=12, size=400)
    out = tmp_path / "tokens.u16"
    record = prepare("byte", 5_000, out, corpus_dir=corpus)
    assert out.exists()
    assert record["tokens"] == out.stat().st_size // 2
    assert record["tokens"] > 0
    # the stream is built 2% long so that `--tokens` counts training tokens only
    assert record["heldout_tokens"] == int(record["tokens"] * HELDOUT_FRACTION)
    assert record["documents"] == 12
    assert record["arm"] == "byte"
    assert record["sources"], "provenance must record which shards were read"


def test_prepared_ids_are_inside_the_byte_alphabet(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus, docs=4)
    out = tmp_path / "tokens.u16"
    prepare("byte", 3_000, out, corpus_dir=corpus)
    stream = np.asarray(np.memmap(out, dtype=np.uint16, mode="r"))
    assert int(stream.max()) < 260
    # <sep> is legitimate: it is the document boundary. Every other special is
    # not, and <pad> in particular would mean a short corpus padded with zeros.
    present = set(np.unique(stream).tolist())
    assert PAD_ID not in present, "padding leaked into the stream"
    assert MASK_ID not in present and CLS_ID not in present
    assert min(present - {SEP_ID}) >= 4, "an id below 4 that is not <sep>"


def test_every_shard_contributes_to_the_stream(tmp_path: Path) -> None:
    """The defect this guards: reading shards in sorted order until the budget
    is full. With three shards, one of them supplied 100% of the stream, and the
    resulting model was trained on a single source while every downstream number
    claimed otherwise."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    for shard in ("a.000", "b.000", "c.000"):
        with gzip.open(corpus / f"{shard}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for i in range(40):
                fh.write(json.dumps({"text": f"{shard} doc {i} " + "lorem ipsum " * 30}) + "\n")

    out = tmp_path / "tokens.u16"
    record = prepare("byte", 20_000, out, corpus_dir=corpus)
    per_source = record["documents_per_source"]
    assert set(per_source) == {"a.000.jsonl.gz", "b.000.jsonl.gz", "c.000.jsonl.gz"}
    # no shard may take more than ~2x its proportional share
    share = max(per_source.values()) / min(per_source.values())
    assert share < 2.0, f"stratification failed: {per_source}"


def test_heldout_tail_is_a_mixture_not_one_source(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    for shard in ("a.000", "b.000", "c.000"):
        with gzip.open(corpus / f"{shard}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for i in range(40):
                fh.write(json.dumps({"text": f"{shard} doc {i} " + "lorem ipsum " * 30}) + "\n")

    out = tmp_path / "tokens.u16"
    record = prepare("byte", 20_000, out, corpus_dir=corpus)
    stream = np.asarray(np.memmap(out, dtype=np.uint16, mode="r"))
    tail = stream[int(len(stream) * (1 - HELDOUT_FRACTION)):]
    text = bytes(int(i) - 4 for i in tail if i >= 4).decode("utf-8", "replace")
    for shard in ("a.000", "b.000", "c.000"):
        assert shard in text, f"held-out tail contains no {shard} text: {text[:200]!r}"


def test_training_batches_never_read_the_heldout_region(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus, docs=12, size=400)
    out = tmp_path / "tokens.u16"
    prepare("byte", 8_000, out, corpus_dir=corpus)

    # Poison the held-out tail so that reading it would be obvious.
    stream = np.memmap(out, dtype=np.uint16, mode="r+")
    n_total = len(stream)
    n_train = int(n_total * (1 - HELDOUT_FRACTION))
    stream[n_train:] = 259
    stream.flush()

    gen = batches(
        np.asarray(stream), n_train, seq=32, batch=8, seed=0, device="cpu"
    )
    for _ in range(5):
        block = next(gen)
        assert block.shape == (8, 32)
        assert int(block.max()) < 259, "a training batch touched the poisoned tail"
        assert int(block.min()) >= SEP_ID, "a non-boundary special reached a batch"
        assert MASK_ID not in set(block.flatten().tolist()), "<mask> in the data"


def test_prepare_is_deterministic_for_the_same_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus, docs=6)
    first, second = tmp_path / "a.u16", tmp_path / "b.u16"
    prepare("byte", 2_000, first, corpus_dir=corpus)
    prepare("byte", 2_000, second, corpus_dir=corpus)
    assert first.read_bytes() == second.read_bytes()


def test_prepare_records_a_short_corpus_honestly(tmp_path: Path) -> None:
    """Asking for more tokens than the corpus holds must truncate and say so,
    not pad with zeros that would train as a run of <pad>."""
    corpus = tmp_path / "corpus"
    write_corpus(corpus, docs=2, size=200)
    out = tmp_path / "tokens.u16"
    record = prepare("byte", 10_000_000, out, corpus_dir=corpus)
    assert record["tokens"] < 10_000_000
    stream = np.asarray(np.memmap(out, dtype=np.uint16, mode="r"))
    assert PAD_ID not in set(np.unique(stream).tolist()), "padding leaked into the stream"
    assert out.stat().st_size == record["tokens"] * 2
