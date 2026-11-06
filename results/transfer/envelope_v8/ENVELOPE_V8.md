# Envelope v8 — verdict: BELOW, both configs (executed 2026-11-04, RTX 6000)

## Pre-registration compliance

4 cells ran the locked v8 recipe (`PREREG_V8.md` `a74c6d28…`): mid anchor
(`b78146e6…`, re-hashed on-host), explicit 1024 window via RoPE extension,
A (M8/378) + B (M16/756), seeds 0+1, no other flags. No deviation.

## Results (misses as errors, Wilson 95% LB vs recorded 0.797, n=500)

| cell | recall test | misses /500 | retrieved-only | end-to-end | Wilson LB |
|---|---|---|---|---|---|
| A_s0 | 0.942 | 29 | 0.7604 | 0.7300 | 0.6894 |
| A_s1 | 0.940 | 29 | 0.7686 | 0.7240 | 0.6832 |
| B_s0 | 0.970 | 15 | 0.7835 | 0.7600 | 0.7207 |
| B_s1 | 0.982 | 9 | 0.7841 | 0.7700 | 0.7311 |

## Verdict: BELOW — both configs fail under the unchanged rule

Config B dominates A at the same rung (+0.04, as at small), but the mid
anchor UNDERPERFORMS the small anchor everywhere (small B: 0.804 vs mid B:
0.76-0.77; small A: 0.75 vs mid A: 0.724-0.73) despite 2.3× params and
better held-out (0.9167 vs 1.0098 bpb).

## Diagnosis (mechanism, not excuse)

The mid anchor is 256-native; the funnel ran it at 1024 via RoPE
extrapolation. Retrieval works (recall 0.97-0.98), but the cross head
under an extrapolated window caps at ~0.78 retrieved-only vs 0.82 for the
small anchor in its (also extended? — small ran 1024 from a 256 ckpt
without the flag and WITHOUT error, so small's extension was implicit) —
either way, the head is the bottleneck and the extrapolated positions are
the prime suspect. The honest fix is a NATIVE-1024 mid pretrain, not more
budget on the extrapolated anchor.

## Artifacts (mirrored byte-identical)

- `A_s0.json` `2c686293…`, `A_s1.json` `c5be73a9…`,
  `B_s0.json` `f4c8080d…`, `B_s1.json` `574effec…`
- `ev_*/provenance.json` (`24ae41e1…`, `6ef22bc7…`, `4d0e8fa2…`,
  `e4be4f12…`), cell logs. Host weights stay on host.

## Next

Native-1024 mid pretrain (same 100M tokens) → v9 (same v8 configs, native
window). If native-1024 mid still caps ≤0.78, the head needs depth/width,
not context — rung scale-up continues (large needs its own pretrain).
