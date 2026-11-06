#!/usr/bin/env python3
"""Run the six frozen AG News cells; never submit to or modify the GPU queue."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def observe():
    state = PROJECT / 'results/pretrain/GPU_QUEUE_STATE.json'
    queue = json.loads(state.read_text()) if state.exists() else {}
    return {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'queue_current': queue.get('current'), 'queue_status': queue.get('status'),
            'gpu': subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                   '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden'
    manifest = json.loads((ROOT / 'frozen_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(ROOT / name) == digest, name
    assert not (ROOT / 'execution.json').exists(), 'refusing to overwrite an execution'
    execution = {'before': observe(), 'python': sys.executable, 'cuda_visible_devices': '',
                 'threads': 4, 'nice': os.getpriority(os.PRIO_PROCESS, 0), 'runs': []}
    (ROOT / 'execution.json').write_text(json.dumps(execution, indent=2) + '\n')
    for seed in (0, 1, 2):
        for init in ('pretrained', 'random'):
            run = ROOT / f'{init}_s{seed}'
            run.mkdir(exist_ok=True)
            cmd = [sys.executable, '-u', str(ROOT / 'source/scripts/transfer_eval.py'),
                   '--task', 'ag_news', '--arm', 'byte', '--rung', 'tiny', '--path', 'cross',
                   '--init', init, '--seed', str(seed), '--examples', '2000',
                   '--test-examples', '500', '--epochs', '6', '--batch', '32', '--lr', '0.0005',
                   '--stratify', '--split-seed', '0', '--device', 'cpu', '--threads', '4',
                   '--evidence-dir', str(run), '--out', str(run / 'result.json')]
            if init == 'pretrained':
                cmd += ['--ckpt', str(ROOT / 'source_pretrain.pt')]
            record = {'seed': seed, 'init': init, 'command': cmd, 'started': time.time()}
            execution['runs'].append(record)
            (ROOT / 'execution.json').write_text(json.dumps(execution, indent=2) + '\n')
            print(f'START {init} seed={seed}', flush=True)
            with (run / 'training.log').open('w') as log:
                proc = subprocess.run(cmd, cwd=ROOT / 'source', stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            record.update(exit_code=proc.returncode, finished=time.time())
            (ROOT / 'execution.json').write_text(json.dumps(execution, indent=2) + '\n')
            print(f'END {init} seed={seed} exit={proc.returncode}', flush=True)
            if proc.returncode:
                raise SystemExit(proc.returncode)
    execution['after'] = observe()
    execution['complete'] = True
    (ROOT / 'execution.json').write_text(json.dumps(execution, indent=2) + '\n')
    print('All six CPU jobs completed.', flush=True)


if __name__ == '__main__':
    main()
