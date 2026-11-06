#!/usr/bin/env python3
"""Pretrain one ladder rung on the local corpus.

Two phases, deliberately separate:

    --prepare   tokenise a slice of the corpus once into a uint16 memmap. Doing
                this ahead of training makes every run reproducible from a file
                and removes the tokeniser from the training loop's critical
                path — which matters because the byte arm's encoder is a Python
                loop and the BPE arm's is Rust, so leaving tokenisation inline
                would mean the two arms' measured throughput differed for
                reasons that have nothing to do with the model.

    (default)   train masked-LM on random windows of that memmap, with cosine
                LR + warmup, gradient clipping, periodic checkpoints, resume,
                a JSONL loss log, and a held-out tail slice never trained on.

The last 2% of the token stream is reserved before training starts and is only
ever read by `--eval-only`. `--tokens` counts *training* tokens, so the memmap is
built 2% longer than asked.

Bits per byte is the L1 gate: a byte-level model that cannot beat a bigram model
on held-out text has learned nothing, and scaling it up would be waste. Use
`scripts/bigram_baseline.py` for the comparison number.

Usage:
    .venv/bin/python3 scripts/pretrain.py --rung tiny --arm byte \
        --tokens 100e6 --prepare
    .venv/bin/python3 scripts/pretrain.py --rung tiny --arm byte --tokens 100e6 \
        --out results/pretrain/l1_tiny_byte
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.config import rung  # noqa: E402
from azurite.env import (  # noqa: E402
    MemoryGuard, amp_policy, configure_mps, device_report, select_device,
)
from azurite.model import Azurite, parameter_counts  # noqa: E402
from azurite.tokenizers import SEP_ID, load as load_tokenizer  # noqa: E402

CORPUS = PROJECT / "data" / "corpus"
TOKENS = PROJECT / "data" / "tokens"
RESULTS = PROJECT / "results" / "pretrain"
HELDOUT_FRACTION = 0.02


def _rel(path: Path) -> str:
    """Project-relative when possible, absolute otherwise.

    Records are read by scripts that may see the same file from a different
    working directory, and a test's tmp_path is legitimately outside the repo;
    `relative_to` raises on that instead of falling back.
    """
    try:
        return str(path.resolve().relative_to(PROJECT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Replace a small state file so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_torch_save(obj: dict, path: Path) -> None:
    """Write a checkpoint through a temp file and rename it into place.

    A checkpoint is the only artifact that makes a killed run resumable. A
    direct `torch.save` to `ckpt.pt` can leave a truncated file if the process
    dies mid-write, which is indistinguishable from "no checkpoint" to the
    resume path. The old checkpoint survives until the new one is complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # A failed save must not leave a half-written `.tmp` behind; the old
        # checkpoint is still intact and the next attempt can reuse the name.
        tmp.unlink(missing_ok=True)
        raise


def _checkpoint_payload(
    model: Azurite,
    opt: torch.optim.Optimizer,
    cfg,
    args,
    step: int,
    tokens_seen: int,
    *,
    data_rng: np.random.Generator,
    mask_gen: torch.Generator,
) -> dict:
    """Everything needed to continue the optimizer and the data/mask streams.

    The token offset alone is not enough to resume: the next batch and the next
    MLM mask are drawn from RNGs. Saving their states makes a resumed segment
    continue the same stream instead of silently starting a different one.
    """
    rng: dict = {
        "numpy_data": data_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "mask": mask_gen.get_state().cpu(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "target_tokens": int(args.tokens),
        "seq": args.seq,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "config": cfg.to_dict(),
        "rng": rng,
        "schema": 2,
    }


def _archive_segment_checkpoint(out_dir: Path, record: dict, index: int) -> None:
    """Keep the weights at a segment boundary under a stable, non-rolling name.

    `ckpt.pt` is deliberately one rolling file so a resume has one obvious
    source. That means a transfer process watching `ckpt.pt` can miss the
    boundary weights when the next segment starts. This copy is the durable
    object the puller can fetch at leisure; it carries its own record and hash.
    """
    sha = record["checkpoint"]["sha256"]
    name = f"seg_{index:03d}_step_{int(record['step'])}"
    dest_dir = out_dir / "checkpoints" / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "ckpt.pt"
    try:
        if not dest.exists() or _sha256(dest) != sha:
            tmp = dest.with_suffix(".pt.tmp")
            shutil.copy2(out_dir / "ckpt.pt", tmp)
            os.replace(tmp, dest)
        _atomic_write_json(dest_dir / "checkpoint.json", {
            **record,
            "archived_from": "ckpt.pt",
            "archive_path": str(dest.relative_to(out_dir)),
        })
    except OSError as exc:
        # The rolling checkpoint is already safe; failing the run because the
        # optional archive copy failed would be worse than losing the copy.
        print(f"WARNING could not archive segment checkpoint {name}: {exc}", flush=True)


def _save_checkpoint(
    out_dir: Path,
    model: Azurite,
    opt: torch.optim.Optimizer,
    cfg,
    args,
    step: int,
    tokens_seen: int,
    *,
    data_rng: np.random.Generator,
    mask_gen: torch.Generator,
    status: str,
) -> dict:
    """Write the weights atomically, then publish a hash-bearing resume record."""
    ckpt_path = out_dir / "ckpt.pt"
    payload = _checkpoint_payload(
        model, opt, cfg, args, step, tokens_seen,
        data_rng=data_rng, mask_gen=mask_gen,
    )
    _atomic_torch_save(payload, ckpt_path)
    record = {
        "schema": 2,
        "status": status,
        "step": step,
        "tokens_seen": tokens_seen,
        "target_tokens": int(args.tokens),
        "fraction": round(tokens_seen / max(1, int(args.tokens)), 6),
        "segment_index": int(args.segment_index),
        "checkpoint": {
            "path": ckpt_path.name,
            "bytes": ckpt_path.stat().st_size,
            "sha256": _sha256(ckpt_path),
        },
        "config": cfg.to_dict(),
        "seq": args.seq,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_write_json(out_dir / "checkpoint.json", record)
    return record


def shard_paths() -> list[Path]:
    paths = sorted(glob_gz())
    if not paths:
        raise SystemExit(f"no corpus shards in {CORPUS}")
    return paths


def glob_gz() -> list[Path]:
    return sorted(CORPUS.glob("*.jsonl.gz")) or sorted(CORPUS.glob("*.jsonl"))


def encode_byte(text: str, offset: int) -> np.ndarray:
    """Vectorised byte encoding: ids = byte + 4."""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16) + offset


def shard_quotas(paths: list[Path], want: int) -> dict[Path, int]:
    """Give each shard a share of the budget proportional to its file size.

    Proportional-to-size, not equal-per-shard: the shards are unequal slices of
    the corpus, and an equal split would over-represent the small ones.
    """
    sizes = {p: max(1, p.stat().st_size) for p in paths}
    total = sum(sizes.values())
    return {p: int(want * sizes[p] / total) for p in paths}


def prepare(
    arm: str,
    target_tokens: int,
    out: Path,
    *,
    max_docs: int | None = None,
    corpus_dir: Path | None = None,
    sep: bool = True,
) -> dict:
    """Tokenise corpus text into a uint16 memmap and record its provenance.

    Two properties this function exists to guarantee, both of which were
    violated by the first version of it:

    - **Stratified.** Shards are read round-robin, one document at a time, each
      capped at a size-proportional quota. The naive version consumed shards in
      sorted order, so `finemath.000` filled the entire 100M budget and the
      "corpus" the model actually saw was 100% math text — invisible in the
      loss curve, fatal to every downstream transfer number.
    - **Interleaved, therefore held-out is a mixture too.** Because the stream is
      a document-level interleave rather than a concatenation of shards, the
      reserved 2% tail is drawn from every source. Concatenation would have made
      the held-out set 100% `wikipedia_multilingual`, silently measuring a
      different distribution than the one being trained on.

    A `<sep>` id is written between documents (not after the last one). Without
    a boundary token the model cannot distinguish "end of document" from "two
    unrelated sentences adjacent", and MLM's random windows will happily straddle
    boundaries regardless.

    `corpus_dir` is a parameter rather than the module constant so the
    train/held-out boundary can be tested on a synthetic corpus instead of the
    real 20 GB one.
    """
    tok = load_tokenizer(arm)
    offset = 4  # specials first, as in azurite.tokenizers
    want = int(target_tokens * (1 + HELDOUT_FRACTION))
    out.parent.mkdir(parents=True, exist_ok=True)

    paths = (
        sorted(corpus_dir.glob("*.jsonl.gz")) or sorted(corpus_dir.glob("*.jsonl"))
        if corpus_dir is not None
        else shard_paths()
    )
    quota = shard_quotas(paths, want)

    stream = np.memmap(out, dtype=np.uint16, mode="w+", shape=(want,))
    filled = 0
    docs = 0
    started = time.time()
    sources: dict[str, int] = {}
    docs_per_source: dict[str, int] = {}
    written: dict[Path, int] = {p: 0 for p in paths}

    handles = {p: (gzip.open if p.suffix == ".gz" else open)(
        p, "rt", encoding="utf-8") for p in paths}
    exhausted = {p: False for p in paths}
    try:
        while filled < want and not all(exhausted.values()):
            progressed = False
            for path in paths:
                if filled >= want:
                    break
                if exhausted[path] or written[path] >= quota[path]:
                    continue
                if max_docs is not None and docs >= max_docs:
                    break

                text = None
                for line in handles[path]:  # continues the same iterator
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    candidate = rec.get("text")
                    if isinstance(candidate, str) and candidate:
                        text = candidate
                        break
                if text is None:
                    exhausted[path] = True
                    continue

                if arm == "byte":
                    ids = encode_byte(text, offset)
                else:
                    ids = np.asarray(tok.encode(text), dtype=np.uint16)

                room = min(quota[path] - written[path], want - filled)
                lead = 1 if (sep and filled > 0) else 0
                if room <= lead:
                    exhausted[path] = True
                    continue
                if lead:
                    stream[filled] = SEP_ID
                    filled += 1
                take = min(len(ids), room - lead)
                if take <= 0:
                    exhausted[path] = True
                    continue
                stream[filled : filled + take] = ids[:take]
                filled += take
                written[path] += take + lead
                docs += 1
                sources[path.name] = sources.get(path.name, 0) + take
                docs_per_source[path.name] = docs_per_source.get(path.name, 0) + 1
                progressed = True
            if not progressed:
                break
    finally:
        for fh in handles.values():
            try:
                fh.close()
            except Exception:
                pass

    stream.flush()
    del stream
    # Truncate the file to what was actually written, so a short corpus does not
    # leave a tail of zeros that would train as a run of <pad>.
    with open(out, "r+b") as fh:
        fh.truncate(filled * 2)

    record = {
        "arm": arm,
        "tokens": int(filled),
        "training_tokens": int(filled * (1 - HELDOUT_FRACTION)),
        "heldout_tokens": int(filled * HELDOUT_FRACTION),
        "documents": docs,
        "path": _rel(out),
        "seconds": round(time.time() - started, 1),
        "sources": sources,
        "documents_per_source": docs_per_source,
        "quota": {p.name: int(quota[p]) for p in paths},
        "separator": SEP_ID if sep else None,
    }
    (out.with_suffix(".record.json")).write_text(json.dumps(record, indent=1) + "\n")
    return record


def batches(
    stream: np.ndarray,
    n_train: int,
    seq: int,
    batch: int,
    *,
    seed: int,
    device: str,
    rng: np.random.Generator | None = None,
):
    """Infinite random-window batches from the training region only.

    `rng` is injectable so a resumed run can restore the exact data stream
    state; when it is omitted the historical seeded generator is used.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    while True:
        starts = rng.integers(0, n_train - seq - 1, size=batch)
        block = np.stack([stream[s : s + seq] for s in starts])
        yield torch.from_numpy(block.astype(np.int64)).to(device)


_BYTE_LEN_CACHE: dict[str, np.ndarray] = {}


def token_byte_lengths(arm: str) -> np.ndarray:
    """UTF-8 byte length of each vocabulary id, indexed by id.

    The gate is stated in bits per *byte*. One token is one byte only in the
    byte arm; the BPE arm averages ~3.6 bytes/token on this corpus, so without
    this table a BPE run's "bpb" is really bits per token and cannot be compared
    with a byte bigram. <sep> is charged one byte, as in the byte arm.
    """
    if arm in _BYTE_LEN_CACHE:
        return _BYTE_LEN_CACHE[arm]
    tok = load_tokenizer(arm)
    table = np.zeros(tok.vocab_size, dtype=np.int64)
    table[SEP_ID] = 1
    if arm == "byte":
        # One id is one byte by construction. Decoding a lone non-UTF-8 byte
        # would report a 3-byte replacement character instead of the byte.
        table[4:] = 1
    else:
        # BPE tokens are byte-level strings, so a token's byte length is its
        # length in alphabet characters. Going through decode() is wrong: a
        # token holding half a UTF-8 sequence decodes to a replacement char.
        from tokenizers import pre_tokenizers

        char_to_byte = {ch: i for i, ch in enumerate(pre_tokenizers.ByteLevel.alphabet())}
        for token, idx in tok._tok.get_vocab().items():
            if idx < 4:
                continue
            if all(ch in char_to_byte for ch in token):
                table[idx] = len(token)
            else:
                table[idx] = len(tok.decode([idx]).encode("utf-8"))
    _BYTE_LEN_CACHE[arm] = table
    return table


def evaluate(model: Azurite, stream: np.ndarray, start: int, seq: int, windows: int,
             device: str, *, seed: int = 1234, arm: str = "byte") -> dict:
    """Mean cross-entropy in bits per byte over held-out windows.

    Masking is drawn from a generator seeded with `seed`, not from the global
    RNG. Without that, the same checkpoint measures differently on every call:
    the tiny rung's 64-window estimate moved between 1.2242 and 1.2988 on
    identical weights, because *which* 30% of positions are masked is a fresh
    draw each time. That spread is a third of the tiny→small gap, so an
    unreproducible number here would be measuring the draw, not the model.
    """
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    avail = len(stream) - start - seq - 1
    byte_table = token_byte_lengths(arm)
    total_bytes = 0
    total_window_tokens = 0
    with torch.no_grad():
        for _ in range(windows):
            s = start + int(rng.integers(0, max(1, avail)))
            window = stream[s : s + seq].astype(np.int64)
            ids = torch.from_numpy(window)[None].to(device)
            loss, stats = model.mlm_loss(ids, generator=gen)
            n = stats["masked_positions"]
            if n:
                total_nats += float(loss) * n
                total_tokens += n
                total_bytes += int(byte_table[window].sum())
                total_window_tokens += int(window.size)
    model.train()
    per_token = total_nats / max(1, total_tokens)
    bytes_per_token = total_bytes / max(1, total_window_tokens)
    return {
        # Masking is iid uniform over positions, so the window's bytes/token is
        # an unbiased estimate of the masked positions' bytes/token. Without
        # this division the BPE arm's number is bits per token, ~3.6x too high.
        "bits_per_byte": per_token / max(bytes_per_token, 1e-12) / math.log(2),
        "bits_per_token": per_token / math.log(2),
        "nats_per_token": per_token,
        "bytes_per_token": bytes_per_token,
        "scored_tokens": total_tokens,
        "scored_bytes": total_bytes,
        "metric_schema": 2,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="tiny")
    ap.add_argument("--arm", default="byte")
    ap.add_argument("--tokens", type=float, default=100e6)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-cap-gib", type=float, default=6.0)
    # Autocast. `auto` is CUDA-only: measured 2.4x there, unmeasured on MPS and
    # CPU, so those keep the previous fp32 path unless asked explicitly.
    ap.add_argument("--amp", choices=("off", "auto", "bf16", "fp16"), default="off")
    # Bounded segment: stop cleanly after this many seconds, checkpoint, and
    # exit 0 without writing summary.json. A 90-minute idle drop is treated as
    # routine, so a run is a sequence of segments that each fit inside one.
    ap.add_argument("--max-seconds", type=float, default=0.0)
    # Decouple the token BUDGET from the length of the memmap. A multi-epoch
    # run (500M tokens over a 100M-token stream) is a legitimate experiment,
    # but the filename convention `{arm}_{tokens}.u16` would send it looking for
    # a stream that does not and should not exist.
    ap.add_argument("--stream", default="",
                    help="explicit uint16 memmap; overrides the arm_tokens filename")
    ap.add_argument("--segment-index", type=int, default=-1,
                    help="selector index written into segment.json; -1 for a standalone run")
    args = ap.parse_args()

    if args.stream:
        tok_path = Path(args.stream)
        if not tok_path.is_absolute():
            tok_path = PROJECT / tok_path
        if not tok_path.exists():
            raise SystemExit(f"--stream: no such memmap {tok_path}")
    else:
        tok_path = TOKENS / f"{args.arm}_{int(args.tokens)}.u16"
    if (args.prepare or not tok_path.exists()) and not args.stream:
        print(f"preparing {tok_path.name} …", flush=True)
        rec = prepare(args.arm, args.tokens, tok_path)
        print(
            f"  {rec['tokens']:,} tokens from {rec['documents']:,} documents "
            f"in {rec['seconds']:.0f}s -> {rec['path']}",
            flush=True,
        )
        if args.prepare:
            return 0

    applied = configure_mps(args.mem_cap_gib)
    device = select_device()
    print(f"device {device} {device_report()}", flush=True)
    amp = amp_policy(args.amp, device)
    amp_name = "off" if amp is None else str(amp[1]).replace("torch.", "")
    print(
        f"amp {args.amp} -> {amp_name}"
        + (f" autocast on {amp[0]}" if amp else ""),
        flush=True,
    )
    stream = np.memmap(tok_path, dtype=np.uint16, mode="r")
    n_total = len(stream)
    n_train = int(n_total * (1 - HELDOUT_FRACTION))
    # A corpus built by `scripts/build_corpus.py` records the exact
    # document-level boundary. Recomputing 98% of the total here would cut into
    # a held-out document, so the recorded boundary wins when it exists. The
    # fallback keeps every stream built the old way bit-identical.
    _split = tok_path.with_suffix(".split.json")
    if _split.exists():
        _rec = json.loads(_split.read_text(encoding="utf-8"))
        if 0 < int(_rec["n_train"]) < n_total:
            n_train = int(_rec["n_train"])
            print(f"  document-level split from {_split.name}: "
                  f"n_train={n_train:,}", flush=True)
    print(
        f"token stream {n_total:,} ({n_train:,} train / {n_total - n_train:,} held out)"
        f"  device={device}",
        flush=True,
    )

    # max_seq follows --seq, not the arm default: the context axis trains this
    # rung at 256/1024/4096 and the RoPE table has to cover the positions used.
    cfg = rung(args.rung, args.arm, max_seq=args.seq)
    counts = parameter_counts(cfg)
    print(
        f"rung={args.rung} arm={args.arm} params={counts['pretrain_total']:,} "
        f"seq={args.seq} batch={args.batch}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    model = Azurite(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    out_dir = Path(args.out) if args.out else RESULTS / f"{args.rung}_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    checkpoint_path = out_dir / "checkpoint.json"
    log_path = out_dir / "train.jsonl"
    meta_path = out_dir / "run.json"

    step = 0
    tokens_seen = 0
    data_rng = np.random.default_rng(args.seed)
    mask_gen = torch.Generator(device=device).manual_seed(args.seed)
    resumed_from = None
    if args.resume:
        if not ckpt_path.exists():
            # The 113M run's failure mode: a partial run's log survived but its
            # weights did not, and `--resume` silently started again at step 0.
            # Never overwrite evidence of an interrupted rung without a human
            # choosing a different --out.
            if checkpoint_path.exists() or log_path.exists() or meta_path.exists():
                raise SystemExit(
                    f"--resume: no checkpoint at {ckpt_path}, but run state exists "
                    f"({', '.join(p.name for p in (checkpoint_path, log_path, meta_path) if p.exists())}); "
                    "refusing to restart from zero. Restore ckpt.pt or use a new --out."
                )
            print(f"no checkpoint yet; starting a fresh run in {out_dir}", flush=True)
        else:
            record = None
            if checkpoint_path.exists():
                try:
                    record = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    record = None
            actual_sha = _sha256(ckpt_path)
            recorded_sha = ((record or {}).get("checkpoint") or {}).get("sha256")
            if recorded_sha and recorded_sha != actual_sha:
                # This is the crash window between the atomic weights rename
                # and the record rename. The weights are self-contained (step
                # and token offset live in the payload), so use them and make
                # the mismatch visible rather than silently restarting.
                print(
                    f"WARNING {checkpoint_path.name} sha256 {recorded_sha} does not match "
                    f"{ckpt_path.name} {actual_sha}; using the newer weights and "
                    "rewriting the record at the next checkpoint",
                    flush=True,
                )
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
            except Exception as exc:  # corrupted/truncated weights are not resumable
                raise SystemExit(
                    f"--resume: {ckpt_path} is not a loadable checkpoint "
                    f"({type(exc).__name__}: {exc}); refusing to restart from zero"
                ) from exc
            saved_target = int(state.get("target_tokens", args.tokens))
            if saved_target != int(args.tokens):
                raise SystemExit(
                    f"--resume: checkpoint target {saved_target:,} does not match "
                    f"--tokens {int(args.tokens):,}; refusing to change the budget mid-run"
                )
            if int(state.get("seq", args.seq)) != args.seq or int(
                    state.get("batch", args.batch)) != args.batch:
                raise SystemExit(
                    "--resume: checkpoint seq/batch do not match this invocation; "
                    "refusing to change the training geometry mid-run"
                )
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            step = int(state["step"])
            tokens_seen = int(state["tokens_seen"])
            saved = state.get("rng")
            if saved:
                data_rng.bit_generator.state = saved["numpy_data"]
                # set_state always takes a CPU ByteTensor, including for an
                # MPS generator; moving it to the device raises TypeError.
                mask_gen.set_state(saved["mask"].cpu())
                torch.set_rng_state(saved["torch_cpu"].cpu())
                if torch.cuda.is_available() and saved.get("torch_cuda"):
                    torch.cuda.set_rng_state_all([s.cpu() for s in saved["torch_cuda"]])
            resumed_from = {"step": step, "tokens_seen": tokens_seen,
                            "checkpoint_sha256": _sha256(ckpt_path)}
            print(f"resumed at step {step:,} ({tokens_seen:,} tokens)", flush=True)

    # Throughput is a property of THIS process, not of the run. `tokens_seen`
    # carries everything restored from the checkpoint, so dividing it by
    # `perf_counter() - t0` reports the checkpoint's history at this segment's
    # speed: a resumed mid-rung segment logged 656,952 tok/s against a real
    # ~102,000 — a 6.4x overstatement sitting in a field the write-up reads.
    tokens_at_start = tokens_seen

    meta = {
        "rung": args.rung,
        "arm": args.arm,
        "params": counts["pretrain_total"],
        "seq": args.seq,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "target_tokens": int(args.tokens),
        "token_stream": _rel(tok_path),
        "device": device,
        "amp": amp_name,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_schema": 2,
        "resumed_from": resumed_from,
    }
    # A fresh run must not quietly overwrite the only record of an earlier one.
    # `--resume` has its own refusal path for missing weights; this guards the
    # other mistake, invoking the same --out without --resume.
    if (not args.eval_only and not args.resume
            and any(p.exists() for p in
                    (checkpoint_path, log_path, meta_path, out_dir / "segment.json"))):
        raise SystemExit(
            f"refusing to start a fresh run in {out_dir}: run state exists "
            "(checkpoint.json/train.jsonl/run.json/segment.json); "
            "pass --resume or use a new --out"
        )

    # A re-measurement (`--eval-only`) is not a new run, so it must not overwrite
    # the run's own provenance — `started` would move and the record would lie.
    if not args.eval_only:
        # Publish a usable checkpoint before writing any run/segment state. If
        # the sandbox dies between these lines, the checkpoint still licenses a
        # resume; the reverse order leaves state without weights.
        if not ckpt_path.exists():
            initial = _save_checkpoint(
                out_dir, model, opt, cfg, args, step, tokens_seen,
                data_rng=data_rng, mask_gen=mask_gen, status="initial",
            )
            print(f"initial checkpoint {initial['checkpoint']['sha256'][:12]} at step 0",
                  flush=True)
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")
        prior = None
        if checkpoint_path.exists():
            try:
                prior = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prior = None
        _atomic_write_json(out_dir / "segment.json", {
            "schema": 2,
            "status": "running",
            "index": args.segment_index,
            "step": step,
            "tokens_seen": tokens_seen,
            "target_tokens": int(args.tokens),
            "fraction": round(tokens_seen / max(1, int(args.tokens)), 6),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "checkpoint": prior.get("checkpoint") if prior else None,
            "resumed_from": resumed_from,
        })

    if args.eval_only:
        # `--eval-only` means "measure the checkpoint", so the weights have to come
        # off disk. Without this block the branch below evaluated a *randomly
        # initialised* model: its bits/byte is worse than the bigram's, which reads
        # as a catastrophic failed rung rather than as a missing load.
        if not ckpt_path.exists():
            raise SystemExit(f"--eval-only: no checkpoint at {ckpt_path}")
        if not args.resume:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model"])
            step = int(state["step"])
            print(f"loaded step {step:,} from {ckpt_path.name}", flush=True)
        heldout = evaluate(model, stream, n_train, args.seq, args.eval_windows, device,
                           seed=args.eval_seed, arm=args.arm)
        # Recorded separately from summary.json, and with its protocol attached:
        # a re-measurement is a different draw, and the protocol (window count and
        # masking seed) is what makes two numbers comparable.
        rec = {
            "step": int(step),
            "windows": args.eval_windows,
            "seq": args.seq,
            "mask_prob": 0.30,
            "seed": args.eval_seed,
            "measured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **heldout,
        }
        (out_dir / "heldout.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec, indent=1), flush=True)
        return 0

    guard = MemoryGuard(cap_gib=args.mem_cap_gib)
    data = batches(stream, n_train, args.seq, args.batch, seed=args.seed,
                   device=device, rng=data_rng)
    tokens_per_step = args.seq * args.batch
    total_steps = max(1, int(args.tokens // tokens_per_step))

    # cosine schedule down to 10% of peak
    def lr_at(s: int) -> float:
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        progress = min(1.0, (s - args.warmup) / max(1, total_steps - args.warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    log = open(log_path, "a", encoding="utf-8")
    t0 = time.perf_counter()
    deadline = t0 + args.max_seconds if args.max_seconds > 0 else None
    truncated = False
    losses: list[float] = []

    while tokens_seen < args.tokens:
        if deadline is not None and time.perf_counter() > deadline:
            truncated = True
            print(
                f"segment budget {args.max_seconds:.0f}s reached at step {step:,}; "
                "checkpointing and exiting so the next segment can resume",
                flush=True,
            )
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        ids = next(data)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=amp[0] if amp else device,
                            dtype=amp[1] if amp else torch.float32,
                            enabled=amp is not None):
            loss, stats = model.mlm_loss(ids, generator=mask_gen)
        if not math.isfinite(float(loss.detach())):
            print(f"FATAL non-finite loss at step {step}", flush=True)
            return 1
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step += 1
        tokens_seen += tokens_per_step
        losses.append(float(loss.detach()))

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            session_tokens = tokens_seen - tokens_at_start
            tps = session_tokens / elapsed if elapsed > 0 else 0.0
            entry = {
                "step": step,
                "tokens_seen": tokens_seen,
                "session_tokens": session_tokens,
                "loss": round(sum(losses[-args.log_every :]) / len(losses[-args.log_every :]), 4),
                "lr": round(lr_at(step), 6),
                "grad_norm": round(float(grad_norm), 3),
                "amp": amp_name,
                "tokens_per_s": round(tps, 1),
                "peak_live_gib": round(guard.peak_live_gib, 3),
                "swap_mb": guard.swap_delta_mb(),
                "elapsed_s": round(elapsed, 1),
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(
                f"step {step:>6,}/{total_steps:,}  loss {entry['loss']:.4f}  "
                f"{tps:,.0f} tok/s  lr {entry['lr']:.2e}  "
                f"live {entry['peak_live_gib']:.2f} GiB  swap {entry['swap_mb']:+.0f} MB",
                flush=True,
            )

        guard.check()
        if step % 50 == 0:
            guard.reclaim_if_needed()

        if step % args.ckpt_every == 0:
            _save_checkpoint(
                out_dir, model, opt, cfg, args, step, tokens_seen,
                data_rng=data_rng, mask_gen=mask_gen, status="running",
            )

    checkpoint_record = _save_checkpoint(
        out_dir, model, opt, cfg, args, step, tokens_seen,
        data_rng=data_rng, mask_gen=mask_gen,
        status="truncated" if truncated else "complete",
    )
    log.close()
    guard.assert_zero_swap()

    if checkpoint_record["status"] in ("truncated", "complete"):
        _archive_segment_checkpoint(out_dir, checkpoint_record, args.segment_index)

    if truncated:
        # A segment is not a run: no summary.json, because `l1_report.py` grades
        # any summary.json it finds and a half-finished rung would be reported as
        # a catastrophic failure rather than as an unfinished job. The durable
        # marker is segment.json plus checkpoint.json; train_segments.py appends
        # this record to segments.jsonl and never deletes it.
        segment = {
            **checkpoint_record,
            "status": "truncated",
            "index": args.segment_index,
            "max_seconds": args.max_seconds,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "device": device,
            "complete": False,
        }
        _atomic_write_json(out_dir / "segment.json", segment)
        print(
            f"checkpointed {tokens_seen:,}/{int(args.tokens):,} tokens "
            f"({segment['fraction']:.1%}); resume with --resume",
            flush=True,
        )
        return 0

    heldout = evaluate(model, stream, n_train, args.seq, args.eval_windows, device,
                       seed=args.eval_seed, arm=args.arm)
    # Record the protocol WITH the number. Without it, summary.json's
    # `heldout.bits_per_byte` cannot be re-derived: the mid rung's 1.2319 was
    # recorded without windows/seq/seed, and re-scoring the same file later gave
    # 1.2161 with no way to tell which field differed.
    heldout = {**heldout, "windows": args.eval_windows, "seq": args.seq,
               "seed": args.eval_seed, "mask_prob": 0.30}
    if losses:
        final_train_loss = round(sum(losses[-50:]) / min(50, len(losses)), 4)
    else:
        # A segment resumed at the target token count does no training, so the
        # in-memory loss list is empty. Read the last logged loss instead of
        # dividing by zero.
        final_train_loss = None
        if log_path.exists():
            for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
                try:
                    final_train_loss = json.loads(line).get("loss")
                    break
                except (json.JSONDecodeError, AttributeError):
                    continue
    summary = {
        **meta,
        "steps": step,
        "tokens_seen": tokens_seen,
        "final_train_loss": final_train_loss,
        "checkpoint": checkpoint_record["checkpoint"],
        "heldout": heldout,
        "seconds": round(time.perf_counter() - t0, 1),
        "memory": guard.report(),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    _atomic_write_json(out_dir / "segment.json", {
        **checkpoint_record,
        "status": "complete",
        "index": args.segment_index,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "device": device,
        "complete": True,
    })
    print(
        f"\nheld-out bits/byte {heldout['bits_per_byte']:.4f} on "
        f"{heldout['scored_tokens']:,} tokens",
        flush=True,
    )
    print(f"wrote {_rel(out_dir)}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
