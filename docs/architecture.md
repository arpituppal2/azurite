# Architecture — the Jev/query path

This note covers the code the Jev comparator added. The rest of the experiment
tree is unchanged and is not restructured here.

## Ownership

| module | owns | does not own |
|---|---|---|
| `azurite/stats.py` | Wilson interval, exact McNemar, paired bootstrap | protocol choices, formatting |
| `azurite/artifacts.py` | SHA-256 of files/text, JSONL reading, per-arm record verification | model or protocol |
| `azurite/jev.py` | frozen inputs, candidate bank, complete-example packing, dual/query head loading, per-arm inference, summaries, paired tests, P0 decision | argument parsing, file layout, markdown/LaTeX |
| `scripts/query_head_eval.py` | native-window arm selection and record layout | statistics, packing, inference |
| `scripts/query_head_eval_5120.py` | 5,120-byte arm selection, old-record reuse, train/test separation | statistics, packing, inference |
| `scripts/query_verdict.py` | native verdict rendering | counts, hashes, decision |
| `scripts/query_verdict_5120.py` | 5,120 verdict rendering | counts, hashes, decision |
| `thesis/gen_query*.py`, `gen_jev_matched.py` | table/macro presentation and record checks | statistics |
| `thesis/gen_{comparator,scaling,suite,suite_three}.py` | table/macro presentation | statistics |

The training scripts remain two procedures on purpose: `query_head_train.py`
trains only the new head at the native window; `query_head_train_5120.py`
jointly fine-tunes the encoder and head at 5,120 with resumable segments. They
share content hashing through `azurite.artifacts` but keep their distinct
schedules.

## Data flow

```
results/jev_matched/inputs.json
  -> jev.load_inputs        validates 3,080 rows, 77 labels, 2,846 Jev-correct
  -> jev.pack_contexts      message-only + complete-example contexts per arm
  -> jev.run_arm            one head, one arm, one JSONL record per row
  -> jev.summarize_records  count + Wilson interval
  -> jev.paired_stats       each arm vs Jev, plus named arm-to-arm pairs
  -> jev.decide             P0 rule -> summary.json
  -> verdict renderer       verifies each JSONL hash, then writes VERDICT.md
  -> thesis generator       verifies the same records, writes table + macros
```

No renderer computes an accuracy or a p-value. `jev.decide` is the only place
the P0 rule is applied; `stats.py` is the only place the interval/test/bootstrap
formulas exist for this path.

## Entry points

```
python scripts/query_head_eval.py --prereg-sha <sha> ...
python scripts/query_head_eval_5120.py --reuse-old --prereg-sha <sha> --amendment-sha <sha> ...
python scripts/query_verdict.py --out /tmp/verdict.md
python scripts/query_verdict_5120.py --out /tmp/verdict.md
python thesis/gen_query.py --check
python thesis/gen_query_5120.py --check
python thesis/gen_jev_matched.py --check
```

The stored verdicts are byte-identical under the refactor, and the generated
tables are hash-identical.

## Known remaining duplication

Four experiment scripts outside this path still carry local `wilson` copies:
`scripts/suite_replication.py`, `scripts/e4_width_train.py`,
`scripts/suite_three_rung.py`, `scripts/analyze_ag_news_local.py`. They are not
part of the Jev/query renderer and were left alone so this pass did not change
unexercised training/analysis code. The next migration should move them onto
`azurite.stats.wilson` when their owning runs are next touched.

`scripts/query_head_train*.py` still import `batch_schedule` from
`scripts/transfer_eval.py`. That schedule is core training behavior rather than
transfer-eval presentation; moving it to `azurite/training.py` is the next
boundary fix, but it would touch the transfer harness and is deferred.
