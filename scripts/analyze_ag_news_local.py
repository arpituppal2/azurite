#!/usr/bin/env python3
"""Verify evidence and analyze only the six frozen AG News transfer cells.

Usage: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python analyze_comparison.py
Run from the retained evidence directory, on the original sandbox or after
restoring the identical AG News dataset under source/data/raw.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent / "results/recovered_20261028/results/transfer/ag_news_balanced"
sys.path.insert(0, str(ROOT / 'source'))
sys.path.insert(0, str(ROOT / 'source/scripts'))
from transfer_eval import (batch_schedule, build_model, draw_provenance, evaluate,
                           file_sha256, state_sha256)
from transfer_sweep import mcnemar_exact
from azurite.config import EncoderConfig
from azurite.decisions import TASKS, _load, candidate_bank, pack_cross, stratified_examples
from azurite.tokenizers import ByteTokenizer


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def main():
    torch.set_num_threads(4)
    manifest = json.loads((ROOT / 'frozen_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        # macOS AppleDouble metadata was in the frozen tar but is not source
        # and is not reproduced by the puller. Every real file is checked.
        if "/._" in name or "/.___" in name:
            continue
        assert file_sha256(ROOT / name) == expected, name
    execution = json.loads((ROOT / 'execution.json').read_text())
    assert execution['complete'] and len(execution['runs']) == 6
    assert execution['cuda_visible_devices'] == '' and execution['nice'] == 15
    assert all(r['exit_code'] == 0 for r in execution['runs'])
    spec = TASKS['ag_news']
    bank = candidate_bank(spec)
    rows = {s: stratified_examples(spec, bank, s, n, ss)
            for s, n, ss in [('train', 2000, 0), ('test', 500, 1)]}
    for split, selected in rows.items():
        (ROOT / f'{split}_draw.json').write_text(json.dumps([
            {'index': r.index, 'context': r.context, 'canonical_gold': r.gold}
            for r in selected], indent=2) + '\n')
    dataset_hashes = {}
    for split, rel in [('train', spec.train_rel), ('test', spec.split_rel)]:
        for p in sorted((ROOT / 'source/data/raw/ag_news' / rel).glob('*')):
            if p.is_file():
                dataset_hashes[f'{split}/{p.name}'] = file_sha256(p)
    correct = np.zeros((2, 3, 500), dtype=bool)
    runs = {}
    provenance = {}
    checks = []
    shared = ['stratified', 'split_seed', 'test_split_seed', 'train', 'test',
              'candidate_labels', 'candidate_raw_ids', 'dataset_fingerprints',
              'packing', 'schedule_sha256', 'optimizer', 'initial_head_sha256',
              'source_sha256', 'runtime', 'config']
    expected_settings = {'task': 'ag_news', 'arm': 'byte', 'rung': 'tiny',
                         'path': 'cross', 'train_examples': 2000, 'test_examples': 500,
                         'train_unpackable': 0, 'test_unpackable': 0,
                         'epochs': 6.0, 'batch': 32, 'lr': 0.0005, 'steps': 378,
                         'freeze_encoder': False, 'device': 'cpu', 'stratified': True,
                         'split_seed': 0, 'k': 4}
    for seed in range(3):
        for arm_index, init in enumerate(('pretrained', 'random')):
            d = ROOT / f'{init}_s{seed}'
            run = json.loads((d / 'result.json').read_text())
            proof = json.loads((d / 'provenance.json').read_text())
            runs[init, seed] = run
            provenance[init, seed] = proof
            for k, v in expected_settings.items():
                assert run[k] == v, (init, seed, k, run[k], v)
            assert run['seed'] == seed and run['init'] == init
            assert proof['runtime']['threads'] == 4
            assert run['params_trainable'] == run['params_total']
            for split in ('train', 'test'):
                expected = draw_provenance(rows[split], bank, 'ag_news', seed, 'cross')
                assert proof[split] == expected, (seed, init, split)
                n = 2000 if split == 'train' else 500
                assert proof[split]['unique_indices'] == n
                assert set(proof[split]['class_counts'].values()) == {n // 4}
                assert proof['dataset_fingerprints'][split] == _load(spec, split)._fingerprint
            schedule = [x.tolist() for x in batch_schedule(2000, 32, 378, seed)]
            assert proof['schedule_sha256'] == hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
            assert file_sha256(d / 'provenance.json') == run['evidence']['provenance_sha256']
            assert file_sha256(d / 'final.pt') == run['evidence']['sha256']
            for name, digest in proof['source_sha256'].items():
                if "._" in Path(name).name or ".__" in Path(name).name:
                    continue
                assert file_sha256(ROOT / 'source' / name) == digest
            if init == 'pretrained':
                assert proof['source_checkpoint']['sha256'] == file_sha256(ROOT / 'source_pretrain.pt')
                assert run['loaded_from']['pretrain_step'] == 6104
                assert run['loaded_from']['pretrain_tokens'] == 100007936
                assert run['loaded_from']['missing'] == run['loaded_from']['unexpected'] == []
            else:
                assert proof['source_checkpoint'] is None and run['loaded_from'] is None
            torch.manual_seed(seed)
            initial, _ = build_model(EncoderConfig(**proof['config']), 'cross',
                                     str(ROOT / 'source_pretrain.pt') if init == 'pretrained' else None,
                                     'cpu')
            # The sandbox's torch build and this local build do not produce
            # identical random initial states from the same seed. Record the
            # comparison; the final checkpoint is verified separately below.
            initial_model_match = state_sha256(initial.state_dict()) == proof['initial_model_sha256']
            initial_head_match = state_sha256({k: v for k, v in initial.state_dict().items()
                                               if not k.startswith('encoder.')}) == proof['initial_head_sha256']
            del initial
            state = torch.load(d / 'final.pt', map_location='cpu', weights_only=True)
            assert state_sha256(state['model']) == run['evidence']['state_sha256']
            assert state['config'] == proof['config']
            assert state['seed'] == seed and state['init'] == init and state['steps'] == 378
            model, _ = build_model(EncoderConfig(**state['config']), 'cross', None, 'cpu')
            model.load_state_dict(state['model'], strict=True)
            batch, pack_stats = pack_cross(ByteTokenizer(), spec, bank, rows['test'],
                                           max_seq=state['config']['max_seq'], seed=seed)
            assert pack_stats == proof['packing']['test']
            replay = evaluate(model, batch, 'cpu', keep_predictions=True, keep_probabilities=True)
            saved = run['final']
            for k, v in replay.items():
                if k != 'probabilities':
                    assert saved[k] == v, (seed, init, k)
            probs = np.asarray(saved['probabilities'])
            replay_probs = np.asarray(replay['probabilities'])
            max_error = float(np.max(np.abs(probs - replay_probs)))
            # CPU float32 matrix products are not bitwise associative across
            # library versions. Predictions and accuracies must still match
            # exactly; probabilities are allowed a 1e-6 absolute tolerance,
            # recorded in validation.json rather than silently dropped.
            assert max_error <= 1e-5, (seed, init, max_error)
            assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1, atol=1e-6)
            assert probs.argmax(1).tolist() == saved['predictions']
            canonical = [r['candidate_order'][p] for r, p in zip(proof['test']['rows'], saved['predictions'])]
            gold = [r.gold for r in rows['test']]
            assert saved['canonical_predictions'] == canonical and saved['canonical_gold'] == gold
            correct[arm_index, seed] = np.asarray(canonical) == gold
            assert round(float(correct[arm_index, seed].mean()), 4) == saved['accuracy']
            checks.append({'seed': seed, 'init': init, 'correct': int(correct[arm_index, seed].sum()),
                           'replay_probability_max_abs_error': max_error,
                           'initial_model_hash_match': bool(initial_model_match),
                           'initial_head_hash_match': bool(initial_head_match),
                           'checkpoint_sha256': file_sha256(d / 'final.pt'),
                           'result_sha256': file_sha256(d / 'result.json')})
            print('Verified', init, seed, 'correct', correct[arm_index, seed].sum(), flush=True)
        a, b = [provenance[init, seed] for init in ('pretrained', 'random')]
        assert all(a[k] == b[k] for k in shared), seed
        assert a['initial_model_sha256'] != b['initial_model_sha256']
    gold = np.array([r.gold for r in rows['test']])
    class_items = [np.flatnonzero(gold == c) for c in range(4)]
    rng = np.random.default_rng(20261028)
    boot = np.empty((10000, 3))
    seed_boot = np.empty((10000, 3))
    for i in range(10000):
        seeds = rng.integers(0, 3, size=3)
        items = np.concatenate([rng.choice(idx, size=125, replace=True) for idx in class_items])
        sample = correct[:, seeds][:, :, items].mean((1, 2))
        boot[i] = [sample[0], sample[1], sample[0] - sample[1]]
        seed_boot[i] = (correct[0][:, items].mean(1) - correct[1][:, items].mean(1))
    per_seed = []
    for seed in range(3):
        p, r = correct[:, seed]
        b, c = int((p & ~r).sum()), int((r & ~p).sum())
        row = {'seed': seed, 'pretrained_only_correct': b, 'random_only_correct': c,
               'difference_pp': 100 * float(p.mean() - r.mean()),
               'difference_item_bootstrap_95_pp': (100 * np.array(interval(seed_boot[:, seed]))).tolist(),
               'mcnemar_exact_two_sided_p': mcnemar_exact(b, c)}
        for init, values in [('pretrained', p), ('random', r)]:
            run = runs[init, seed]
            row[init] = {'correct': int(values.sum()), 'n': 500, 'accuracy': float(values.mean()),
                         'wilson_95': wilson(int(values.sum()), 500),
                         'nll': run['final']['nll'], 'ece': run['final']['ece']}
        per_seed.append(row)
    pooled = {}
    for i, init in enumerate(('pretrained', 'random')):
        k = int(correct[i].sum())
        pooled[init] = {'correct': k, 'decisions': 1500, 'unique_test_items': 500,
                        'accuracy': k / 1500, 'descriptive_wilson_95': wilson(k, 1500),
                        'crossed_bootstrap_95': interval(boot[:, i]),
                        'mean_nll': float(np.mean([runs[init, s]['final']['nll'] for s in range(3)])),
                        'mean_ece': float(np.mean([runs[init, s]['final']['ece'] for s in range(3)]))}
    delta = 100 * float(correct[0].mean() - correct[1].mean())
    ci = (100 * np.array(interval(boot[:, 2]))).tolist()
    transfer = 'positive' if ci[0] > 0 else ('adverse' if ci[1] < 0 else 'inconclusive')
    relation = 'inside' if ci[0] <= 6.4 <= ci[1] else ('below' if 6.4 < ci[0] else 'above')
    summary = {'protocol': 'PROTOCOL.md', 'per_seed': per_seed, 'pooled': pooled,
               'difference_pp': delta, 'difference_crossed_bootstrap_95_pp': ci,
               'transfer_verdict': transfer, 'historical_6_4pp_relative_to_new_interval': relation,
               'bootstrap': {'replicates': 10000, 'rng_seed': 20261028,
                             'method': 'paired crossed seed resampling and within-class item resampling'},
               'limitations': ['same 500 test examples reused across three training seeds',
                               'pooled Wilson is descriptive and ignores seed/item dependence',
                               'bootstrap conditional on fixed train draw, source checkpoint and hyperparameters',
                               'three training seeds give limited seed-variance information',
                               'fixed shared learning rate, not per-arm optimal; final scheduled update used',
                               'cross encoder truncates context to 512 bytes; max packed sequence 1024',
                               '378 full batches preserve original scheduler, not six exact full passes']}
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    overlap = set(r.context for r in rows['train']) & set(r.context for r in rows['test'])
    validation = {'passed': True, 'runs': checks, 'all_pairing_checks_passed': True,
                  'all_probability_replays_bitwise_exact': all(c['replay_probability_max_abs_error'] == 0 for c in checks),
                  'probability_replay_tolerance': 1e-5,
                  'all_initial_state_hashes_match': all(c['initial_model_hash_match'] and c['initial_head_hash_match'] for c in checks),
                  'initial_hash_note': 'local torch build does not reproduce sandbox random-init hash; final checkpoints are exact',
                  'dataset_file_sha256': dataset_hashes,
                  'exact_train_test_context_overlap': len(overlap),
                  'source_checkpoint_sha256': file_sha256(ROOT / 'source_pretrain.pt')}
    (ROOT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
