#!/usr/bin/env python3
"""기하 분기 감쇠 대조의 학습·최종 검증·독립 검산을 실행한다."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from _pdv2_no_context_ablation import write_json

OUT = REPO / 'results/pdv2_geometry_decay_20261023'
LOG = REPO / 'logs/geometry_decay_20261023'


def call(label: str, script: str, arguments: list) -> None:
    """작업별 로그를 보존하고 실패하면 중단한다."""
    with (LOG / (label + '.log')).open('w') as handle:
        result = subprocess.run([sys.executable, '-u', str(REPO / 'scripts' / script), *arguments],
            cwd=REPO, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                              OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
            stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(label + '\n' + (LOG / (label + '.log')).read_text()[-5000:])


def main() -> None:
    """새 학습 한 조건과 같은 고정 에포크의 두 최종 판정을 완료한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    start = time.time()
    script = '_pdv2_geometry_decay.py'
    call('prepare', script, ['prepare'])
    write_json(OUT / 'status.json', {'status': 'running', 'stage': 'train', 'test_used': False})
    call('train_nodecay', script, ['train'])
    print('TRAIN_DONE', round(time.time() - start, 1), flush=True)
    spec = json.loads((OUT / 'protocol.json').read_text())
    jobs = [(a, s) for a in spec['config']['arms'] for s in spec['val']]
    def evaluate(job):
        arm, sequence = job
        call('val_' + arm + '_' + sequence.replace('/', '_'), script,
             ['val', '--arm', arm, '--sequence', sequence])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate, job) for job in jobs]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            write_json(OUT / 'status.json', {'status': 'running', 'stage': 'val',
                'done': count, 'total': len(jobs), 'seconds': time.time() - start, 'test_used': False})
            print('VAL_DONE', count, len(jobs), flush=True)
    call('report', script, ['report'])
    call('finish', '_pdv2_geometry_decay_finish.py', [])
    write_json(OUT / 'status.json', {'status': 'complete', 'audit_passed': True,
        'seconds': time.time() - start, 'test_used': False})
    print('DECAY_COMPLETE', round(time.time() - start, 1), flush=True)


if __name__ == '__main__':
    main()
