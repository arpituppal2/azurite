#!/usr/bin/env python3
"""L0 gate: can the plumbing memorise 50 *real* decisions?

PLAN.md §6 defines this gate: "a 2M model reaches 100% train accuracy on 50
examples it is allowed to memorise. If it cannot overfit 50 rows, the plumbing is
broken and nothing downstream is worth reading."

That is the cheapest test that can fail for the right reason. It exercises the
whole path — task loading, label resolution, packing, spans, attention mask,
decision head, loss, optimiser — on real text from the frozen splits, and it has
an unambiguous pass condition: 1.00.

Both scoring paths are gated, because both will be used:

    cross-encoder   ag_news (K=4) and emotion (K=6)
    dual-encoder    banking77 (K=77), where the cross-encoder cannot be honest

A pass here means "the plumbing can learn". It says nothing about
generalisation, which is the entire point of keeping it separate from L2/L3.

Usage:
    .venv/bin/python3 scripts/l0_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import rung  # noqa: E402
from azurite.decisions import (  # noqa: E402
    TASKS,
    candidate_bank,
    load_examples,
    pack_cross,
    pack_dual,
)
from azurite.env import MemoryGuard, configure_mps  # noqa: E402
from azurite.model import Azurite, AzuriteDecision, AzuriteDual  # noqa: E402
from azurite.tokenizers import PAD_ID, load as load_tokenizer  # noqa: E402

RESULTS = PROJECT / "results" / "l0"


def candidate_tensor(bank, tok) -> tuple[torch.Tensor, torch.Tensor]:
    if not bank.token_ids:
        bank.tokenize(tok)
    width = max(len(c) for c in bank.token_ids)
    ids = torch.full((len(bank), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(bank), width), dtype=torch.bool)
    for i, c in enumerate(bank.token_ids):
        ids[i, : len(c)] = torch.tensor(c, dtype=torch.long)
        mask[i, : len(c)] = True
    return ids, mask


def run_cross(task: str, tok, arm: str, steps: int, examples: int, seed: int) -> dict:
    spec = TASKS[task]
    bank = candidate_bank(spec)
    rows = load_examples(spec, bank, split="train", limit=examples)
    batch, stats = pack_cross(
        tok, spec, bank, rows, max_seq=rung("tiny", arm).max_seq, seed=seed
    )
    batch = batch.to("mps")

    torch.manual_seed(seed)
    model = AzuriteDecision(Azurite(rung("tiny", arm))).to("mps")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    guard = MemoryGuard(cap_gib=6.0)
    loss_fn = AzuriteDecision.loss
    history = []
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, batch.gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            with torch.no_grad():
                acc = (model(batch).argmax(-1) == batch.gold).float().mean().item()
            history.append(
                {"step": step, "loss": round(float(loss.detach()), 4), "acc": round(acc, 4)}
            )
        guard.check()
        if step % 50 == 49:
            guard.reclaim_if_needed()
    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        final = (model(batch).argmax(-1) == batch.gold).float().mean().item()
    guard.assert_zero_swap()
    return {
        "task": task,
        "path": "cross",
        "arm": arm,
        "k": len(bank),
        "examples": stats["packed"],
        "unpackable": stats["unpackable"],
        "mean_seq_len": stats["mean_seq_len"],
        "steps": steps,
        "seconds": round(elapsed, 1),
        "final_accuracy": round(final, 4),
        "passed": final >= 0.999,
        "history": history,
        "memory": guard.report(),
    }


def run_dual(task: str, tok, arm: str, steps: int, examples: int, seed: int) -> dict:
    spec = TASKS[task]
    bank = candidate_bank(spec)
    rows = load_examples(spec, bank, split="train", limit=examples)
    batch, stats = pack_dual(tok, spec, bank, rows, max_seq=rung("tiny", arm).max_seq)
    cand_ids, cand_token_mask = candidate_tensor(bank, tok)
    cand_valid = torch.ones(len(bank), dtype=torch.bool)
    batch = batch.to("mps")
    cand_ids = cand_ids.to("mps")
    cand_token_mask = cand_token_mask.to("mps")
    cand_valid = cand_valid.to("mps")

    torch.manual_seed(seed)
    model = AzuriteDual(Azurite(rung("tiny", arm))).to("mps")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    guard = MemoryGuard(cap_gib=6.0)
    history = []
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(
            batch.context_ids, batch.context_mask, cand_ids, cand_token_mask, cand_valid
        )
        loss = torch.nn.functional.cross_entropy(logits, batch.gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            with torch.no_grad():
                acc = (
                    model(
                        batch.context_ids,
                        batch.context_mask,
                        cand_ids,
                        cand_token_mask,
                        cand_valid,
                    ).argmax(-1)
                    == batch.gold
                ).float().mean().item()
            history.append(
                {"step": step, "loss": round(float(loss.detach()), 4), "acc": round(acc, 4)}
            )
        guard.check()
        if step % 50 == 49:
            guard.reclaim_if_needed()
    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        final = (
            model(
                batch.context_ids,
                batch.context_mask,
                cand_ids,
                cand_token_mask,
                cand_valid,
            ).argmax(-1)
            == batch.gold
        ).float().mean().item()
    guard.assert_zero_swap()
    return {
        "task": task,
        "path": "dual",
        "arm": arm,
        "k": len(bank),
        "examples": stats["packed"],
        "unpackable": stats["unpackable"],
        "mean_seq_len": stats["mean_seq_len"],
        "steps": steps,
        "seconds": round(elapsed, 1),
        "final_accuracy": round(final, 4),
        "passed": final >= 0.999,
        "history": history,
        "memory": guard.report(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--examples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--mem-cap-gib", type=float, default=6.0)
    args = ap.parse_args()

    applied = configure_mps(args.mem_cap_gib)
    tok = load_tokenizer(args.arm)
    print(
        f"tokenizer arm={args.arm} vocab={tok.vocab_size} mps={applied['mps']}",
        flush=True,
    )
    print(
        f"gate: 2.4M model, {args.examples} real examples, {args.steps} steps, "
        "pass at 1.00\n",
        flush=True,
    )

    plan = [("cross", "ag_news"), ("cross", "emotion"), ("dual", "banking77")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    json_path = RESULTS / f"l0_{stamp}.json"
    out = {
        "gate": "L0",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "arm": args.arm,
            "steps": args.steps,
            "examples": args.examples,
            "seed": args.seed,
            "mem_cap_gib": args.mem_cap_gib,
        },
        "runs": [],
    }
    failures = 0
    for path, task in plan:
        fn = run_cross if path == "cross" else run_dual
        try:
            r = fn(task, tok, args.arm, args.steps, args.examples, args.seed)
        except Exception as exc:  # a broken task is a gate result, not a crash
            r = {"task": task, "path": path, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        out["runs"].append(r)
        # Write after every run: a killed process should still leave the results
        # it earned, not nothing.
        json_path.write_text(json.dumps(out, indent=1) + "\n")
        if r.get("passed"):
            print(
                f"  PASS  {path:5s} {task:14s} K={r['k']:<3d} "
                f"{r['examples']:>2d} rows  acc={r['final_accuracy']:.4f}  "
                f"{r['seconds']:.1f}s  seq~{r['mean_seq_len']:.0f}",
                flush=True,
            )
        else:
            failures += 1
            print(
                f"  FAIL  {path:5s} {task:14s} acc={r.get('final_accuracy')} "
                f"{r.get('error', '')}",
                flush=True,
            )

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["passed"] = failures == 0
    json_path.write_text(json.dumps(out, indent=1) + "\n")

    lines = [
        "# L0 gate — can the plumbing memorise 50 real decisions?\n",
        f"Generated by `scripts/l0_gate.py` at {stamp}. A 2.4M-parameter model, "
        f"trained on {args.examples} examples from the **train** split of each "
        "task, scored on those same examples. The pass condition is 1.00: this is "
        "a plumbing test, not a generalisation test.\n",
        "| path | task | K | rows | seq | steps | final acc | s | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in out["runs"]:
        if "error" in r:
            lines.append(f"| {r['path']} | {r['task']} | — | — | — | — | — | — | FAIL |")
            continue
        lines.append(
            f"| {r['path']} | {r['task']} | {r['k']} | {r['examples']} | "
            f"{r['mean_seq_len']:.0f} | {r['steps']} | **{r['final_accuracy']:.4f}** | "
            f"{r['seconds']:.1f} | {'PASS' if r['passed'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append(
        "What a pass proves: task loading, label resolution, packing, span "
        "construction, attention masking, the decision head, the loss and the "
        "optimiser are all wired correctly, and the model has the capacity to fit "
        "real text. What it does not prove: anything at all about generalisation.\n"
    )
    if failures:
        lines.append(
            f"**{failures} gate(s) failed.** Nothing downstream is worth running "
            "until this passes.\n"
        )
    (RESULTS / "L0.md").write_text("\n".join(lines))
    print(f"\n{'PASS' if out['passed'] else 'FAIL'} — wrote {RESULTS.relative_to(PROJECT)}/L0.md")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
