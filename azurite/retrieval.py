"""BM25 retrieval and the retrieved-example context used by the query arm.

The retriever is a direct port of the public Jev Banking77 experiment's
`Retriever` (word unigrams plus adjacent word pairs, k1=1.5, b=0.75) so the
example set is the same kind of evidence Jev received. It retrieves context; it
never chooses the label or removes candidates.
"""

from __future__ import annotations

import collections
import math
import re

TOKEN_RE = re.compile(r"[a-z0-9]+")


def terms(text: str) -> list[str]:
    words = TOKEN_RE.findall(text.lower())
    return words + [a + "_" + b for a, b in zip(words, words[1:])]


class BM25:
    """Small, deterministic BM25 index over labelled training messages."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        self.lengths: list[int] = []
        for i, row in enumerate(rows):
            ts = terms(row["text"])
            self.lengths.append(len(ts))
            for term, n in collections.Counter(ts).items():
                self.postings[term].append((i, n))
        self.avg = sum(self.lengths) / max(1, len(rows))
        self.idf = {
            term: math.log(1 + (len(rows) - len(v) + 0.5) / (len(v) + 0.5))
            for term, v in self.postings.items()
        }

    def rank(self, text: str) -> list[int]:
        scores: dict[int, float] = collections.defaultdict(float)
        for term in set(terms(text)):
            for i, tf in self.postings.get(term, []):
                scores[i] += (
                    self.idf[term]
                    * tf
                    * 2.5
                    / (tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.avg))
                )
        return sorted(range(len(self.rows)), key=lambda i: (-scores[i], self.rows[i]["id"]))


def select_examples(
    retriever: BM25,
    query: str,
    *,
    exclude_id: int | None,
    limit: int = 24,
    per_label: int = 4,
) -> list[dict]:
    """Top `limit` hits, at most `per_label` from any one label, self excluded."""
    counts: collections.Counter[str] = collections.Counter()
    out: list[dict] = []
    for i in retriever.rank(query):
        row = retriever.rows[i]
        if exclude_id is not None and row["id"] == exclude_id:
            continue
        if counts[row["label"]] >= per_label:
            continue
        out.append(row)
        counts[row["label"]] += 1
        if len(out) == limit:
            break
    return out


def build_context(message: str, examples: list[dict], *, budget: int = 1_020) -> tuple[str, int]:
    """Serialize complete examples until the byte budget is reached.

    Only complete `message => intent` lines are added; a line that would cross
    the budget stops the loop. That keeps the native-window arm free of a
    half-truncated example, which the earlier byte-sliced diagnostic arm had.
    """
    head = f"Customer message: {message}\nLabeled examples:\n"
    out = head
    used = 0
    for j, example in enumerate(examples, 1):
        label = example.get("label", example.get("intent"))
        line = f"{j}. {example['message']} => {label}\n"
        if len((out + line).encode("utf-8")) > budget:
            break
        out += line
        used += 1
    return out, used
