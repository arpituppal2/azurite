"""The Jev-comparator protocol: inputs, packing, inference and decision.

The protocol is one object with two context settings. Everything that decides
what a row means -- the frozen Jev correctness vector, the exact 24 retrieved
examples, the context budget, the candidate bank, the paired tests and the P0
rule -- lives here. The scripts under ``scripts/query_head_eval*`` are
interfaces: they parse arguments, choose a protocol, and write the record.

Data flows one way:

    results/jev_matched/inputs.json
        -> validated rows + labels
        -> packed contexts (one per arm)
        -> per-arm inference records
        -> summaries + paired tests + decision
        -> summary.json / per-example JSONL / verdict rendering

No script re-derives an accuracy, an interval or a p-value from prose. The
stored records are the only source, and ``verify_arm_records`` is the gate the
verdict renderers use before quoting one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from azurite.artifacts import sha256_text
from azurite.config import EncoderConfig
from azurite.decisions import CandidateBank, _pad
from azurite.model import Azurite, AzuriteDual
from azurite.query import QueryDual
from azurite.retrieval import build_context
from azurite.stats import bootstrap_lower, mcnemar_exact, wilson

N_ROWS = 3_080
JEV_CORRECT = 2_846
P0_BAR = JEV_CORRECT / N_ROWS
MARGIN = -0.02
BOOT_SEED = 0
EXAMPLES = 24
NATIVE_BUDGET = 1_020
LONG_BUDGET = 5_118
NATIVE_PROTOCOL = "jev-query-head-eval-v1"
LONG_PROTOCOL = "jev-query-head-5120-eval-v1"


@dataclass(frozen=True)
class Arm:
    """One inference arm: head family, example count and byte budget."""

    name: str
    kind: str  # "dual" | "query"
    examples: int = 0
    budget: int = NATIVE_BUDGET
    require_full: bool = False
    reuse_from: str | None = None


@dataclass
class Packed:
    """Tokenised arms sharing one message pass."""

    messages: list[str]
    msg_ids: torch.Tensor
    msg_mask: torch.Tensor
    contexts: dict[str, list[str]]
    effective: dict[str, list[int]]
    ids: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]


def load_inputs(path: Path, *, require_examples: int | None = None) -> dict:
    """Load the frozen row set and refuse any drift in it."""
    data = json.loads(path.read_text())
    rows = data["rows"]
    if len(rows) != N_ROWS:
        raise SystemExit(f"expected {N_ROWS:,} rows, got {len(rows):,}")
    correct = sum(int(row["jev_correct"]) for row in rows)
    if correct != JEV_CORRECT:
        raise SystemExit(f"Jev correctness vector sums to {correct}, not {JEV_CORRECT}")
    labels = data["labels"]
    if len(labels) != 77:
        raise SystemExit(f"expected 77 labels, got {len(labels)}")
    if require_examples is not None:
        for row in rows:
            if len(row["examples"]) != require_examples:
                raise SystemExit(
                    f"row {row['id']}: {len(row['examples'])} examples, "
                    f"expected {require_examples}"
                )
    return data


def candidate_bank(labels: Sequence[str], tok) -> tuple[CandidateBank, torch.Tensor, torch.Tensor]:
    bank = CandidateBank(task="banking77", labels=list(labels), raw_ids=list(range(len(labels))))
    bank.tokenize(tok)
    ids, mask = _pad(bank.token_ids)
    return bank, ids, mask


def pack_contexts(rows: Sequence[dict], arms: Sequence[Arm], tok) -> Packed:
    """Build each arm's context once, with complete examples only."""
    messages = [row["text"] for row in rows]
    contexts: dict[str, list[str]] = {}
    effective: dict[str, list[int]] = {}
    ids: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for arm in arms:
        if arm.reuse_from is not None:
            continue
        if arm.examples == 0:
            contexts[arm.name] = list(messages)
            effective[arm.name] = [0] * len(rows)
        else:
            built: list[str] = []
            used: list[int] = []
            for row in rows:
                context, count = build_context(row["text"], row["examples"], budget=arm.budget)
                if arm.require_full and count != arm.examples:
                    raise SystemExit(
                        f"row {row['id']}: only {count} examples fit in "
                        f"{arm.budget} bytes, expected {arm.examples}"
                    )
                built.append(context)
                used.append(count)
            contexts[arm.name] = built
            effective[arm.name] = used
        ids[arm.name], masks[arm.name] = _pad(
            [tok.encode(text, add_specials=True) for text in contexts[arm.name]]
        )
    msg_ids, msg_mask = _pad([tok.encode(text, add_specials=True) for text in messages])
    return Packed(
        messages=messages,
        msg_ids=msg_ids,
        msg_mask=msg_mask,
        contexts=contexts,
        effective=effective,
        ids=ids,
        masks=masks,
    )


def load_dual_head(path: Path, device: torch.device) -> tuple[AzuriteDual, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("task") != "banking77" or state.get("path") != "dual":
        raise SystemExit(f"{path}: not a Banking77 dual head")
    model = AzuriteDual(Azurite(EncoderConfig(**state["config"])))
    model.load_state_dict(state["model"], strict=True)
    return model.eval().to(device), state


def load_query_head(
    path: Path, device: torch.device, *, require_max_seq: int | None = None
) -> tuple[QueryDual, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("task") != "banking77" or state.get("path") != "query":
        raise SystemExit(f"{path}: not a Banking77 query head")
    configured = state.get("config", {}).get("max_seq")
    if require_max_seq is not None and configured != require_max_seq:
        raise SystemExit(f"{path}: max_seq {configured} is not {require_max_seq}")
    model = QueryDual(Azurite(EncoderConfig(**state["config"])))
    model.load_state_dict(state["model"], strict=True)
    return model.eval().to(device), state


def run_arm(
    model,
    *,
    kind: str,
    arm: Arm,
    packed: Packed,
    rows: Sequence[dict],
    cand_ids: torch.Tensor,
    cand_mask: torch.Tensor,
    batch: int,
    device: torch.device,
    head_sha: str,
    head_path: str,
) -> tuple[dict, list[dict]]:
    """Run one head over one packed arm and return its summary and records."""
    if kind == "dual":
        cand_emb = model.embed(cand_ids.to(device), cand_mask.to(device))
    else:
        cand_emb = model.embed_candidates(cand_ids.to(device), cand_mask.to(device))
    scale = float(model.log_scale.exp().item())
    context_ids, context_mask = packed.ids[arm.name], packed.masks[arm.name]
    records: list[dict] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            sl = slice(start, start + batch)
            ids = context_ids[sl].to(device)
            mask = context_mask[sl].to(device)
            if kind == "dual":
                context = model.embed(ids, mask)
            else:
                context = model.embed_context(
                    ids,
                    mask,
                    packed.msg_ids[sl].to(device),
                    packed.msg_mask[sl].to(device),
                )
            preds = (scale * (context @ cand_emb.T)).argmax(-1).tolist()
            for offset, pred in enumerate(preds):
                row = rows[start + offset]
                text = packed.contexts[arm.name][start + offset]
                pred = int(pred)
                gold = int(row["gold_index"])
                records.append(
                    {
                        "arm": arm.name,
                        "id": row["id"],
                        "i": int(row["i"]),
                        "text_sha256": sha256_text(row["text"]),
                        "context_sha256": sha256_text(text),
                        "context_bytes": len(text.encode("utf-8")),
                        "effective_examples": packed.effective[arm.name][start + offset],
                        "gold": row["gold"],
                        "gold_index": gold,
                        "pred": pred,
                        "pred_index": pred,
                        "correct": int(pred == gold),
                        "head_path": head_path,
                        "head_sha256": head_sha,
                    }
                )
            if start == 0 or (start // batch) % 100 == 0:
                print(
                    f"  {arm.name}: {len(records)}/{len(rows)} rows "
                    f"({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    seconds = time.perf_counter() - started
    return summarize_records(arm.name, records, seconds=seconds), records


def summarize_records(name: str, records: Sequence[dict], *, seconds: float,
                      reused_from: str | None = None) -> dict:
    correct = np.asarray([row["correct"] for row in records], dtype=np.int8)
    k = int(correct.sum())
    lo, hi = wilson(k, len(records))
    summary = {
        "arm": name,
        "n": len(records),
        "correct": k,
        "accuracy": k / len(records),
        "wilson95": [lo, hi],
        "seconds": seconds,
        "mean_context_bytes": float(np.mean([row["context_bytes"] for row in records])),
        "mean_effective_examples": float(
            np.mean([row["effective_examples"] for row in records])
        ),
        "min_effective_examples": int(min(row["effective_examples"] for row in records)),
        "max_effective_examples": int(max(row["effective_examples"] for row in records)),
    }
    if reused_from is not None:
        summary["reused_from"] = reused_from
    return summary


def paired_stats(
    correctness: dict[str, np.ndarray],
    jev: np.ndarray,
    extra_pairs: Sequence[tuple[str, str, str]] = (),
) -> dict:
    """Every arm against Jev, plus explicitly named arm-to-arm comparisons."""
    paired: dict[str, dict] = {}
    for name, values in correctness.items():
        paired[name] = {
            "diff_vs_jev": float(values.mean() - jev.mean()),
            "mcnemar": mcnemar_exact(values, jev),
            "bootstrap_lower_95": bootstrap_lower(values, jev, seed=BOOT_SEED),
        }
    for label, left, right in extra_pairs:
        a, b = correctness[left], correctness[right]
        paired[label] = {
            "diff": float(a.mean() - b.mean()),
            "mcnemar": mcnemar_exact(a, b),
            "bootstrap_lower_95": bootstrap_lower(a, b, seed=BOOT_SEED),
        }
    return paired


def decide(
    *,
    primary_arm: str,
    summaries: dict[str, dict],
    paired: dict[str, dict],
    rule: str,
    p0_bar: float = P0_BAR,
    margin: float = MARGIN,
) -> dict:
    """Apply the pre-registered P0 rule to the stored summaries."""
    primary = summaries[primary_arm]
    diff = primary["accuracy"] - p0_bar
    p = paired[primary_arm]["mcnemar"]["p"]
    noninferior = bool(paired[primary_arm]["bootstrap_lower_95"] > margin)
    if diff > 0 and p < 0.05:
        verdict = "P0 exceeded"
    elif primary["accuracy"] + 1e-12 >= p0_bar:
        verdict = "P0 met (equal or above)"
    else:
        verdict = "P0 missed"
    return {
        "primary_arm": primary_arm,
        "p0_bar": p0_bar,
        "point_estimate_meets_bar": bool(primary["accuracy"] + 1e-12 >= p0_bar),
        "diff_vs_jev": diff,
        "mcnemar_p": p,
        "bootstrap_lower_95": paired[primary_arm]["bootstrap_lower_95"],
        "noninferior_at_minus_2pp": noninferior,
        "verdict": verdict,
        "rule": rule,
    }
