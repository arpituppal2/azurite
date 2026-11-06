"""The installed command line interface.

    azurite info
    azurite checkpoint add ~/weights/final.pt --id query-native
    azurite predict "My card was swallowed by the ATM"
    azurite verify --records per_example_query_a2c.jsonl \
                   --inputs inputs.json --n 8

Presentation lives here; checkpoint identity lives in `azurite.checkpoints`,
and the inference protocol lives in `azurite.predictor`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from azurite import checkpoints
from azurite.bundle import BundleError, load_bundle, load_provenance, load_registry
from azurite.checkpoints import CheckpointError
from azurite.predictor import PredictionError, Predictor


def _version() -> str:
    try:
        return version("azurite-encoder")
    except PackageNotFoundError:
        return "0.1.0.dev0"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sample(records: list[dict], n: int, ids: str | None) -> list[dict]:
    if ids:
        wanted = [part.strip() for part in ids.split(",") if part.strip()]
        by_id = {str(record["id"]): record for record in records}
        missing = [ident for ident in wanted if ident not in by_id]
        if missing:
            raise PredictionError(f"records do not contain ids: {', '.join(missing)}")
        return [by_id[ident] for ident in wanted]
    if n < 1:
        raise PredictionError("--n must be positive")
    if n > len(records):
        raise PredictionError(f"--n {n} exceeds {len(records)} stored records")
    if n == 1:
        return [records[0]]
    picks = sorted({round(i * (len(records) - 1) / (n - 1)) for i in range(n)})
    return [records[index] for index in picks]


def command_info(_: argparse.Namespace) -> int:
    bundle = load_bundle()
    registry = load_registry()
    provenance = load_provenance()
    source_support = provenance.get("sources", {}).get("support", {})
    removed = int(source_support.get("removed_rows", 0))
    source_rows = int(source_support.get("original_rows", 0))
    print(f"azurite {_version()}")
    print(f"task: {bundle.task} ({len(bundle.labels)} intents, English banking messages)")
    support_note = ""
    if removed and source_rows:
        support_note = (
            f" ({removed} of {source_rows:,} source rows removed as evaluation texts)"
        )
    print(f"support: {len(bundle.support):,} labelled messages bundled{support_note}")
    print("limits: byte-level, 1,024-byte native window, no image or audio input")
    print("checkpoints:")
    for checkpoint in registry.checkpoints:
        path = checkpoints.find(checkpoint)
        state = f"installed at {path}" if path else "not installed"
        measured = checkpoint.measured
        print(
            f"  {checkpoint.id:<18} {checkpoint.sha256[:12]}  "
            f"{measured['accuracy'] * 100:6.2f}% on {measured['n']:,} rows  {state}"
        )
        bank = measured.get("bank")
        wilson = measured.get("wilson95")
        if bank and wilson:
            print(
                f"      bank: {bank}; 95% Wilson "
                f"{wilson[0] * 100:.2f}-{wilson[1] * 100:.2f}"
            )
        frozen = checkpoint.frozen
        if frozen:
            contamination = frozen.get("contamination", {})
            fwilson = frozen.get("wilson95")
            print(
                f"      original 9,993-bank arm: {frozen['correct']:,}/{frozen['n']:,} "
                f"= {frozen['accuracy'] * 100:.2f}%"
                + (
                    f" (95% Wilson {fwilson[0] * 100:.2f}-{fwilson[1] * 100:.2f})"
                    if fwilson
                    else ""
                )
            )
            if contamination:
                print(
                    "      overlap: "
                    f"{contamination['support_messages']} of "
                    f"{contamination['support_rows']:,} support messages are "
                    f"test rows ({contamination['exact']} exact); "
                    f"{contamination['affected_rows']:,} rows "
                    f"({contamination['affected_share']:.2f}%) retrieve at least one"
                )
    return 0


def command_predict(args: argparse.Namespace) -> int:
    predictor = Predictor.load(args.checkpoint, device=args.device)
    prediction = predictor.predict(args.text, top_k=args.top)
    payload = prediction.to_dict()
    if args.json:
        print(json.dumps(payload, indent=1, sort_keys=True))
        return 0
    print(f"intent:     {prediction.intent}")
    print(f"confidence: {prediction.confidence:.4f} (softmax over 77 candidates)")
    print(f"margin:     {prediction.margin:.4f}")
    print(
        f"context:    {len(prediction.context.encode('utf-8'))} bytes, "
        f"{prediction.examples_used} examples"
    )
    print(f"checkpoint: {prediction.checkpoint} ({prediction.checkpoint_sha256[:12]})")
    print("top:")
    for item in prediction.top:
        print(f"  {item['rank']}. {item['intent']:<32} {item['score']:.4f}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    records = _read_jsonl(args.records)
    data = json.loads(args.inputs.read_text())
    rows = {str(row["id"]): row for row in data["rows"]}
    selected = _sample(records, args.n, args.ids)
    missing = [record["id"] for record in selected if str(record["id"]) not in rows]
    if missing:
        raise PredictionError(f"inputs do not contain rows: {', '.join(map(str, missing))}")
    predictor = Predictor.load(args.checkpoint, device=args.device)
    results = predictor.verify_frozen(
        [rows[str(record["id"])] for record in selected], selected, batch=args.batch
    )
    matches = sum(item["match"] for item in results)
    correct = sum(item["cli_correct"] for item in results)
    if args.json:
        print(
            json.dumps(
                {
                    "checkpoint": predictor.checkpoint.id,
                    "checkpoint_sha256": predictor.checkpoint.sha256,
                    "records": str(args.records),
                    "inputs": str(args.inputs),
                    "n": len(results),
                    "matches": matches,
                    "cli_correct": correct,
                    "results": results,
                },
                indent=1,
                sort_keys=True,
            )
        )
    else:
        print(
            f"{'id':<14} {'stored':>6} {'cli':>5} {'match':>6} {'gold':<30} correct"
        )
        for item in results:
            print(
                f"{str(item['id']):<14} {item['stored_pred']:>6} "
                f"{item['cli_pred']:>5} {'yes' if item['match'] else 'NO':>6} "
                f"{item['gold']:<30} {item['cli_correct']}"
            )
        print(
            f"{matches}/{len(results)} predictions identical; "
            f"{len(results)}/{len(results)} context digests identical; "
            f"CLI correct {correct}/{len(results)}"
        )
    return 0 if matches == len(results) else 1


def command_checkpoint_add(args: argparse.Namespace) -> int:
    resolved = checkpoints.add(
        args.source, ident=args.id, home=args.home, link=args.link
    )
    print(f"{resolved.checkpoint.id}: {resolved.sha256}")
    print(f"installed at {resolved.path}")
    return 0


def command_checkpoint_list(_: argparse.Namespace) -> int:
    for checkpoint, path in checkpoints.installed():
        state = str(path) if path else "not installed"
        print(f"{checkpoint.id:<18} {checkpoint.sha256}  {state}")
    return 0


def command_decide(args: argparse.Namespace) -> int:
    from azurite.api import Alternative, DecisionRequest, decide
    from azurite.select_read import SelectRead

    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")
    else:
        context = args.context or ""
    engine = SelectRead.from_checkpoint(
        args.reader, tokenizer_path=args.tokenizer,
        selector_path=args.selector, head_path=args.head, device=args.device,
    )
    request = DecisionRequest(
        context=context, question=args.question,
        alternatives=[Alternative(id=str(i), label=label) for i, label in enumerate(args.alt)],
        evidence_required=args.evidence,
    )
    result = decide(request, engine)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=1, sort_keys=True))
        return 0 if result.choice_id is not None or result.abstained else 1
    if result.abstained:
        print(f"abstained: {result.abstain_reason}")
    else:
        print(f"choice: {result.choice_label} ({result.choice_id})")
    print("scores:")
    for key, score in payload["scores"].items():
        print(f"  {key}: {score:.4f}")
    print(f"evidence: {len(payload['evidence'])} span(s)")
    for span in payload["evidence"][:5]:
        print(f"  [{span['byte_start']}:{span['byte_end']}] {span['text'][:120]!r}")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    from azurite.select_read import SelectRead
    from azurite.server import serve

    engine = SelectRead.from_checkpoint(
        args.reader, tokenizer_path=args.tokenizer,
        selector_path=args.selector, head_path=args.head, device=args.device,
    )
    serve(engine, host=args.host, port=args.port)
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    from azurite.select_read import SelectRead

    path = Path(args.reader)
    print(f"reader: {path} ({path.stat().st_size:,} bytes)" if path.exists() else f"reader: {path} MISSING")
    print(f"tokenizer: {args.tokenizer}")
    print(f"selector: {args.selector or 'BM25 (no neural selector)'}")
    print(f"head: {args.head or 'default dual head'}")
    if path.exists():
        engine = SelectRead.from_checkpoint(
            args.reader, tokenizer_path=args.tokenizer,
            selector_path=args.selector, head_path=args.head, device=args.device,
        )
        print(f"reader parameters: {sum(p.numel() for p in engine.reader.parameters()):,}")
        print(f"device: {args.device}")
    return 0


def command_adapt(args: argparse.Namespace) -> int:
    from azurite.abstain import ConformalCalibrator

    scores, correct = [], []
    with Path(args.rows).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            scores.append(float(row["score"]))
            correct.append(bool(row["correct"]))
    cal = ConformalCalibrator.fit(scores, correct, coverage=args.coverage)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal.to_dict(), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(cal.to_dict(), indent=1))
    return 0


def command_bench(args: argparse.Namespace) -> int:
    from azurite.api import Alternative, DecisionRequest, decide
    from azurite.battery import read_eval_battery, summarize
    from azurite.parity import write_json
    from azurite.select_read import SelectRead

    rows, _ = read_eval_battery(args.battery)
    if args.limit:
        rows = rows[: args.limit]
    engine = SelectRead.from_checkpoint(
        args.reader, tokenizer_path=args.tokenizer,
        selector_path=args.selector, head_path=args.head, device=args.device,
    )
    records = []
    for row in rows:
        request = DecisionRequest(
            context=row.context, question=row.question,
            alternatives=[Alternative(id=str(i), label=a) for i, a in enumerate(row.alternatives)],
        )
        t0 = time.perf_counter()
        result = decide(request, engine)
        records.append({
            "id": row.id, "slice": row.slice, "gold": row.gold,
            "choice": int(result.choice_id) if result.choice_id is not None else -1,
            "abstained": result.abstained,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        })
    summary = summarize(records)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records_10k.json").write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="azurite",
        description="77-way Banking77 intent classification from a byte-level encoder.",
    )
    parser.add_argument("--version", action="version", version=f"azurite {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show the task, bundle and checkpoint state")
    info.set_defaults(func=command_info)

    predict = sub.add_parser("predict", help="classify one customer message")
    predict.add_argument("text", help="the customer message")
    predict.add_argument("--top", type=int, default=5, help="candidates to print")
    predict.add_argument("--checkpoint", default=None, help="checkpoint id or hash prefix")
    predict.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    predict.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    predict.set_defaults(func=command_predict)

    verify = sub.add_parser(
        "verify", help="replay stored evaluation rows and compare predictions"
    )
    verify.add_argument("--records", required=True, type=Path, help="per-example JSONL")
    verify.add_argument("--inputs", required=True, type=Path, help="frozen inputs JSON")
    verify.add_argument("--n", type=int, default=8, help="rows to sample")
    verify.add_argument("--ids", default=None, help="comma-separated row ids instead")
    verify.add_argument("--checkpoint", default=None)
    verify.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    verify.add_argument("--batch", type=int, default=8)
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=command_verify)

    checkpoint = sub.add_parser("checkpoint", help="manage content-addressed weights")
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    add = checkpoint_sub.add_parser("add", help="verify and cache a weights file")
    add.add_argument("source", type=Path)
    add.add_argument("--id", default=None, help="published checkpoint id")
    add.add_argument("--home", type=Path, default=None, help="override the cache root")
    add.add_argument("--link", action="store_true", help="hard-link instead of copying")
    add.set_defaults(func=command_checkpoint_add)
    listing = checkpoint_sub.add_parser("list", help="show published checkpoints")
    listing.set_defaults(func=command_checkpoint_list)

    decide = sub.add_parser("decide", help="choose among caller-supplied alternatives")
    context_group = decide.add_mutually_exclusive_group(required=True)
    context_group.add_argument("--context", default=None, help="inline context text")
    context_group.add_argument("--context-file", default=None, type=Path)
    decide.add_argument("--question", default="")
    decide.add_argument("--alt", action="append", required=True, help="alternative label; repeat for K")
    decide.add_argument("--reader", required=True, type=Path)
    decide.add_argument("--tokenizer", default="data/tokenizers/azurite_bpe_32004.json")
    decide.add_argument("--selector", default=None, type=Path)
    decide.add_argument("--head", default=None, type=Path)
    decide.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    decide.add_argument("--evidence", action="store_true")
    decide.add_argument("--json", action="store_true")
    decide.set_defaults(func=command_decide)

    serve = sub.add_parser("serve", help="start the local-only HTTP server")
    serve.add_argument("--reader", required=True, type=Path)
    serve.add_argument("--tokenizer", default="data/tokenizers/azurite_bpe_32004.json")
    serve.add_argument("--selector", default=None, type=Path)
    serve.add_argument("--head", default=None, type=Path)
    serve.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.set_defaults(func=command_serve)

    doctor = sub.add_parser("doctor", help="check the local model artifacts")
    doctor.add_argument("--reader", required=True, type=Path)
    doctor.add_argument("--tokenizer", default="data/tokenizers/azurite_bpe_32004.json")
    doctor.add_argument("--selector", default=None, type=Path)
    doctor.add_argument("--head", default=None, type=Path)
    doctor.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    doctor.set_defaults(func=command_doctor)

    adapt = sub.add_parser("adapt", help="fit a conformal abstention threshold from JSONL rows")
    adapt.add_argument("--rows", required=True, type=Path)
    adapt.add_argument("--out", default=Path("results/parity/calibration.json"), type=Path)
    adapt.add_argument("--coverage", type=float, default=0.90)
    adapt.set_defaults(func=command_adapt)

    bench = sub.add_parser("bench", help="score a frozen battery and write per-row records")
    bench.add_argument("--battery", default="results/parity/eval_battery_10k")
    bench.add_argument("--reader", required=True, type=Path)
    bench.add_argument("--tokenizer", default="data/tokenizers/azurite_bpe_32004.json")
    bench.add_argument("--selector", default=None, type=Path)
    bench.add_argument("--head", default=None, type=Path)
    bench.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    bench.add_argument("--limit", type=int, default=0)
    bench.add_argument("--out-dir", default=Path("results/parity"), type=Path)
    bench.set_defaults(func=command_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (PredictionError, CheckpointError, BundleError) as exc:
        print(f"azurite: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"azurite: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
