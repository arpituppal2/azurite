"""Pack-to-pack evidence contamination audit.

The stream audit answers whether a pack's evidence was in pretraining. This
module answers the complementary question: does a target pack contain the same
evidence windows as a dev or locked pack? That is the leak that matters when a
training pack set and an evaluation pack set are built from the same source
documents.

The unit of evidence is the exact token-aligned window used by
``scripts/audit_pack_contamination.py``: up to five 48-token windows per gold
span, with short spans kept whole. A pack is rejected if any window overlaps a
reference pack's evidence. An unauditable pack (no gold span, or a span too short
to form a window) is rejected too; silence is not a clean bill of health.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Protocol, Sequence

from .tokenizers import SEP_ID

WINDOW_TOKENS = 48
MIN_WINDOW_TOKENS = 8
DEFAULT_WINDOWS = 5


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


def token_windows(
    ids: Iterable[int],
    *,
    window_tokens: int = WINDOW_TOKENS,
    min_tokens: int = MIN_WINDOW_TOKENS,
) -> list[tuple[int, ...]]:
    """Return the audit windows for one token sequence.

    Separators and control ids are removed first, matching the stream audit's
    definition of text. Long sequences get five deterministic windows: start,
    quarter, midpoint, three-quarter, and the final full window.
    """
    clean = [int(i) for i in ids if int(i) > SEP_ID]
    if len(clean) < min_tokens:
        return []
    if len(clean) <= window_tokens:
        return [tuple(clean)]
    offsets = {
        0,
        len(clean) // 4,
        len(clean) // 2,
        (3 * len(clean)) // 4,
        len(clean) - window_tokens,
    }
    return [tuple(clean[o : o + window_tokens]) for o in sorted(offsets)]


def evidence_texts(pack: dict) -> list[str]:
    """Slice the pack's recorded gold byte spans from its own context."""
    raw = str(pack.get("context", "")).encode("utf-8")
    texts: list[str] = []
    for span in pack.get("gold_spans") or []:
        start, end = int(span["start"]), int(span["end"])
        if 0 <= start < end <= len(raw):
            text = raw[start:end].decode("utf-8", "replace").strip()
            if text:
                texts.append(text)
    return texts


def pack_evidence_windows(
    pack: dict,
    tokenizer: Tokenizer,
    *,
    window_tokens: int = WINDOW_TOKENS,
    min_tokens: int = MIN_WINDOW_TOKENS,
) -> list[tuple[int, ...]]:
    """Tokenize every gold-evidence span and return all audit windows."""
    windows: list[tuple[int, ...]] = []
    for text in evidence_texts(pack):
        windows.extend(
            token_windows(
                tokenizer.encode(text),
                window_tokens=window_tokens,
                min_tokens=min_tokens,
            )
        )
    return windows


def build_reference_windows(
    reference_packs: Iterable[dict],
    tokenizer: Tokenizer,
    *,
    window_tokens: int = WINDOW_TOKENS,
    min_tokens: int = MIN_WINDOW_TOKENS,
) -> set[tuple[int, ...]]:
    """Collect every evidence window from the dev/locked reference packs."""
    out: set[tuple[int, ...]] = set()
    for pack in reference_packs:
        out.update(
            pack_evidence_windows(
                pack,
                tokenizer,
                window_tokens=window_tokens,
                min_tokens=min_tokens,
            )
        )
    return out


@dataclass(frozen=True)
class PackAuditResult:
    pack_id: str
    accepted: bool
    auditable: bool
    evidence_windows: int
    matched_windows: int
    reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def audit_pack(
    pack: dict,
    reference: set[tuple[int, ...]],
    tokenizer: Tokenizer,
    *,
    window_tokens: int = WINDOW_TOKENS,
    min_tokens: int = MIN_WINDOW_TOKENS,
) -> PackAuditResult:
    """Audit one pack against a prepared reference window set.

    A pack with no auditable window is rejected rather than called clean: the
    old audit recorded null as "not measured", and a downstream boolean coercion
    could turn that into a pass.
    """
    pack_id = str(pack.get("id", "<unnamed>"))
    windows = pack_evidence_windows(
        pack,
        tokenizer,
        window_tokens=window_tokens,
        min_tokens=min_tokens,
    )
    if not windows:
        return PackAuditResult(
            pack_id=pack_id,
            accepted=False,
            auditable=False,
            evidence_windows=0,
            matched_windows=0,
            reason="unauditable: no gold evidence window",
        )
    if not reference:
        return PackAuditResult(
            pack_id=pack_id,
            accepted=False,
            auditable=True,
            evidence_windows=len(windows),
            matched_windows=0,
            reason="no dev/locked reference windows to audit against",
        )
    matched = sum(1 for window in windows if window in reference)
    if matched:
        return PackAuditResult(
            pack_id=pack_id,
            accepted=False,
            auditable=True,
            evidence_windows=len(windows),
            matched_windows=matched,
            reason="gold evidence overlaps the dev/locked reference",
        )
    return PackAuditResult(
        pack_id=pack_id,
        accepted=True,
        auditable=True,
        evidence_windows=len(windows),
        matched_windows=0,
        reason=None,
    )


def audit_packs(
    packs: Iterable[dict],
    reference: set[tuple[int, ...]],
    tokenizer: Tokenizer,
    *,
    window_tokens: int = WINDOW_TOKENS,
    min_tokens: int = MIN_WINDOW_TOKENS,
) -> list[PackAuditResult]:
    return [
        audit_pack(
            pack,
            reference,
            tokenizer,
            window_tokens=window_tokens,
            min_tokens=min_tokens,
        )
        for pack in packs
    ]


def filter_packs(
    packs: Iterable[dict],
    reference_packs: Iterable[dict],
    tokenizer: Tokenizer,
    *,
    window_tokens: int = WINDOW_TOKENS,
    min_tokens: int = MIN_WINDOW_TOKENS,
) -> tuple[list[dict], list[dict], dict]:
    """Split target packs into accepted and rejected sets with a stored report.

    The rejected list is the actionable output: callers must not train on it.
    The gate fails when the reference is empty, because an empty reference
    cannot demonstrate that the audit was exercised.
    """
    pack_list = list(packs)
    reference = build_reference_windows(
        reference_packs,
        tokenizer,
        window_tokens=window_tokens,
        min_tokens=min_tokens,
    )
    results = audit_packs(
        pack_list,
        reference,
        tokenizer,
        window_tokens=window_tokens,
        min_tokens=min_tokens,
    )
    accepted: list[dict] = []
    rejected: list[dict] = []
    for pack, result in zip(pack_list, results):
        (accepted if result.accepted else rejected).append(pack)
    passed = bool(reference) and not rejected
    report = {
        "reference_windows": len(reference),
        "target_packs": len(pack_list),
        "accepted_packs": len(accepted),
        "rejected_packs": len(rejected),
        "unauditable_packs": sum(1 for r in results if not r.auditable),
        "matched_packs": sum(1 for r in results if r.matched_windows),
        "results": [r.to_dict() for r in results],
        "gate": {
            "criterion": "no target pack evidence window overlaps the dev/locked reference",
            "state": "pass" if passed else "fail",
        },
    }
    return accepted, rejected, report
