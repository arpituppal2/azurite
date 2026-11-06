#!/usr/bin/env python3
"""Length-adapt the 113.47M byte encoder from 1,024 to 5,120 positions.

This is one arm, fixed by `results/jev_query_5120/PRE_REGISTRATION.md`:

* source `results/recovered_20261028/results/pretrain/l1_large_byte_iso10v3/ckpt.pt`
  (SHA-256 aba65b60e30654a9895dc9c365d62b555f96dcce96ad8a8ad56cc718ee0865ec);
* the same `Azurite` dimensions with `max_seq=5,120`; RoPE table grown, no learned
  positional parameters, so the state loads strictly;
* 1,220 optimizer steps at sequence 5,120 and batch 8 = 49,971,200 tokens;
* AdamW, fresh state, LR 1e-4, betas (0.9, 0.95), weight decay 0.1, clip 1.0,
  cosine to 10% of peak with 100 warmup steps, seed 0;
* the pretraining masked-LM objective unchanged (30% masking, 80/10/10).

The run is deliberately resumable: the Molab sandbox can drop mid-run, and the
pre-registered budget is 1,220 steps, not "as many as fit in one session". A
rolling checkpoint carries model, optimizer and RNG state; `--resume` continues
the same stream. A non-finite loss aborts the run and is reported as a failure.

    python scripts/length_adapt.py \
        --source results/pretrain/l1_large_byte_iso10v3/ckpt.pt \
        --out-dir results/pretrain/l1_large_byte_iso10v3_len5120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from azurite.config import EncoderConfig, rung  # noqa: E402
from azurite.model import Azurite  # noqa: E402
import pretrain  # noqa: E402

SOURCE_SHA = "aba65b60e30654a9895dc9c365d62b555f96dcce96ad8a8ad56cc718ee0865ec"
STREAM_SHA = "9267a2ea071a517803b3b47c01adf9464608a8178fbf12018ce5dcc9ca38365f"
SEQ = 5_120
BATCH = 8
STEPS = 1_220
LR = 1e-4
WARMUP = 100
SEED = 0
SAVE_EVERY = 100


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    with tmp.open("rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = min(1.0, (step - WARMUP) / max(1, total - WARMUP))
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=ROOT / "results/pretrain/l1_large_byte_iso10v3/ckpt.pt", type=Path)
    ap.add_argument("--stream", default=ROOT / "data/tokens/byte_corpus_v2.u16", type=Path)
    ap.add_argument("--split", default=ROOT / "data/tokens/byte_corpus_v2.split.json", type=Path)
    ap.add_argument("--out-dir", default=ROOT / "results/pretrain/l1_large_byte_iso10v3_len5120", type=Path)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 steps, /tmp output, never a reported result")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 2
        args.out_dir = Path("/tmp/length_adapt_smoke")
    if (args.seq, args.batch, args.lr, args.seed) != (SEQ, BATCH, LR, SEED):
        raise SystemExit("geometry/LR/seed are fixed by the pre-registration; do not change them")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "ckpt.pt"
    run_path = args.out_dir / "run.json"
    if not args.resume and any(p.exists() for p in (ckpt_path, run_path)):
        raise SystemExit(f"refusing to overwrite run state in {args.out_dir}; pass --resume")
    if not args.source.exists():
        raise SystemExit(f"missing source checkpoint {args.source}")
    if sha256_file(args.source) != SOURCE_SHA:
        raise SystemExit("source checkpoint SHA-256 does not match the pre-registration")
    if sha256_file(args.stream) != STREAM_SHA:
        raise SystemExit("corpus SHA-256 does not match the pre-registration")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    split = json.loads(args.split.read_text())
    n_train = int(split["n_train"])
    stream = np.memmap(args.stream, dtype=np.uint16, mode="r")
    print(f"stream={len(stream):,} n_train={n_train:,} seq={args.seq} batch={args.batch}", flush=True)

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    if source.get("config", {}).get("max_seq") != 1024:
        raise SystemExit("source checkpoint is not the 1,024-position model")
    cfg = rung("large", "byte", max_seq=args.seq)
    model = Azurite(cfg)
    missing, unexpected = model.load_state_dict(source["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"state mismatch: missing={missing} unexpected={unexpected}")
    del source
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    data_rng = np.random.default_rng(args.seed)
    mask_gen = torch.Generator(device=device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    step = 0
    tokens_seen = 0
    losses: list[float] = []
    resumed_from = None
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = int(state["step"])
        tokens_seen = int(state["tokens_seen"])
        data_rng.bit_generator.state = state["data_rng"]
        mask_gen.set_state(state["mask_gen"].cpu())
        torch.set_rng_state(state["torch_cpu"].cpu())
        if device.type == "cuda" and state.get("torch_cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
        losses = list(state.get("losses", []))
        resumed_from = {"step": step, "tokens_seen": tokens_seen,
                        "checkpoint_sha256": sha256_file(ckpt_path)}
        print(f"resumed at step {step:,} ({tokens_seen:,} tokens)", flush=True)

    if not args.resume:
        run_path.write_text(json.dumps({
            "protocol": "length-adapt-5120-v1",
            "source": {"path": str(args.source), "sha256": SOURCE_SHA,
                       "step": 69_255, "tokens_seen": 1_134_673_920, "max_seq": 1024},
            "target": {"seq": args.seq, "batch": args.batch, "steps": args.steps,
                       "tokens": args.steps * args.seq * args.batch,
                       "max_seq": args.seq},
            "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.95],
                          "weight_decay": 0.1, "clip": 1.0, "warmup": WARMUP,
                          "schedule": "cosine to 10% of peak"},
            "objective": {"name": "masked-LM", "mask_prob": 0.30, "corruption": "80/10/10"},
            "seed": args.seed,
            "stream": {"path": str(args.stream), "sha256": STREAM_SHA,
                       "n_train": n_train, "split": str(args.split)},
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "resumed_from": None,
        }, indent=1) + "\n")

    data = pretrain.batches(stream, n_train, args.seq, args.batch, seed=args.seed,
                            device=str(device), rng=data_rng)
    total_tokens = args.steps * args.seq * args.batch
    log_path = args.out_dir / "train.jsonl"
    log = log_path.open("a", encoding="utf-8")
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    while step < args.steps:
        lr = lr_at(step, args.steps)
        for group in opt.param_groups:
            group["lr"] = lr
        ids = next(data)
        opt.zero_grad(set_to_none=True)
        loss, stats = model.mlm_loss(ids, generator=mask_gen)
        value = float(loss.detach())
        if not math.isfinite(value):
            print(f"FATAL non-finite loss at step {step}", flush=True)
            return 1
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tokens_seen += args.seq * args.batch
        losses.append(value)

        if step % 25 == 0 or step == args.steps:
            elapsed = time.perf_counter() - t0
            entry = {
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": value,
                "mean_loss_25": round(sum(losses[-25:]) / len(losses[-25:]), 6),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "masked_positions": int(stats["masked_positions"]),
                "ms_per_step": round(1000 * elapsed / max(1, step - (resumed_from["step"] if resumed_from else 0)), 1),
                "peak_gib": round(torch.cuda.max_memory_allocated() / (1 << 30), 3) if device.type == "cuda" else None,
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(f"step {step:>4}/{args.steps} loss {value:.4f} mean25 {entry['mean_loss_25']:.4f} "
                  f"lr {lr:.2e} peak {entry['peak_gib']} GiB", flush=True)

        if step % SAVE_EVERY == 0 and step < args.steps:
            atomic_save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "config": cfg.to_dict(),
                "step": step,
                "tokens_seen": tokens_seen,
                "target_tokens": total_tokens,
                "seq": args.seq,
                "batch": args.batch,
                "lr": args.lr,
                "seed": args.seed,
                "losses": losses,
                "data_rng": data_rng.bit_generator.state,
                "mask_gen": mask_gen.get_state().cpu(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()] if device.type == "cuda" else None,
                "source_sha256": SOURCE_SHA,
                "schema": 1,
            }, ckpt_path)
            print(f"  checkpoint step {step}", flush=True)

    train_seconds = time.perf_counter() - t0
    log.close()
    atomic_save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "config": cfg.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "target_tokens": total_tokens,
        "seq": args.seq,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "losses": losses,
        "data_rng": data_rng.bit_generator.state,
        "mask_gen": mask_gen.get_state().cpu(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()] if device.type == "cuda" else None,
        "source_sha256": SOURCE_SHA,
        "schema": 1,
    }, ckpt_path)
    ckpt_sha = sha256_file(ckpt_path)

    heldout = pretrain.evaluate(model, stream, n_train, args.seq, 64, str(device), seed=1234)
    checkpoint = {
        "schema": 1,
        "status": "complete" if step == args.steps else "partial",
        "step": step,
        "tokens_seen": tokens_seen,
        "target_tokens": total_tokens,
        "fraction": round(tokens_seen / total_tokens, 6),
        "checkpoint": {"path": ckpt_path.name, "bytes": ckpt_path.stat().st_size, "sha256": ckpt_sha},
        "config": cfg.to_dict(),
        "seq": args.seq,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "source": {"path": str(args.source), "sha256": SOURCE_SHA},
        "resumed_from": resumed_from,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.out_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=1) + "\n")
    summary = {
        "protocol": "length-adapt-5120-v1",
        "source": {"path": str(args.source), "sha256": SOURCE_SHA},
        "checkpoint": {"path": str(ckpt_path), "sha256": ckpt_sha},
        "steps": step,
        "tokens_seen": tokens_seen,
        "train_seconds": train_seconds,
        "ms_per_step": 1000 * train_seconds / max(1, step - (resumed_from["step"] if resumed_from else 0)),
        "peak_gib": round(torch.cuda.max_memory_allocated() / (1 << 30), 3) if device.type == "cuda" else None,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "mean_loss_last_50": sum(losses[-50:]) / len(losses[-50:]) if losses else None,
        "heldout": {"seq": args.seq, "windows": 64, "mask_prob": 0.30, "seed": 1234, **heldout},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform(),
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {ckpt_path} sha256={ckpt_sha}")
    print(json.dumps(summary["heldout"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
