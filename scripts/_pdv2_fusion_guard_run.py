#!/usr/bin/env python3
"""2x2 손실 학습 뒤 전체 최종 검증과 선택 모델의 학습 재검색을 실행한다."""
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

OUT = REPO / 'results/pdv2_fusion_guard_20261023'
LOG = REPO / 'logs/fusion_guard_20261023'


def call(label: str, script: str, arguments: list) -> None:
    """단계별 로그를 남기고 실패 시 중단한다."""
    path = LOG / (label + '.log')
    with path.open('w') as log:
        done = subprocess.run([sys.executable, '-u', str(REPO / 'scripts' / script), *arguments],
            cwd=REPO, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                              OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), stdout=log, stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError(label + '\n' + path.read_text()[-5000:])


def evaluate_jobs(jobs: list, started: float) -> None:
    """독립 시퀀스 두 개씩 실행하고 완료 수를 기록한다."""
    def worker(job):
        arm, epoch, role, sequence = job
        label = f'{role}_{arm}_e{epoch}_' + sequence.replace('/', '_')
        call(label, '_pdv2_fusion_guard.py', ['evaluate', '--arm', arm, '--epoch', str(epoch),
            '--role', role, '--sequence', sequence])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            write_json(OUT / 'status.json', {'status': 'running', 'stage': jobs[0][2], 'done': i,
                'total': len(jobs), 'seconds': time.time() - started, 'test_used': False})
            print('EVAL_DONE', jobs[0][2], i, len(jobs), flush=True)


def main() -> None:
    """고정된 네 조건을 끝까지 실행하고 최종 원장을 감사한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(OUT / 'status.json', {'status': 'running', 'stage': 'prepare', 'test_used': False})
    call('prepare', '_pdv2_fusion_guard.py', ['prepare'])
    spec = json.loads((OUT / 'protocol.json').read_text())
    for arm in spec['config']['arms']:
        write_json(OUT / 'status.json', {'status': 'running', 'stage': 'train', 'arm': arm, 'test_used': False})
        call('train_' + arm, '_pdv2_fusion_guard.py', ['train', '--arm', arm])
        print('TRAIN_DONE', arm, round(time.time() - started, 1), flush=True)
    jobs = [(arm, epoch, 'val', seq) for arm in spec['config']['arms']
            for epoch in spec['config']['checkpoint_epochs'] for seq in spec['val']]
    evaluate_jobs(jobs, started)
    call('select', '_pdv2_fusion_guard.py', ['select'])
    selection = json.loads((OUT / 'selection.json').read_text())['selected_epoch']
    jobs = [(arm, epoch, 'train', seq) for arm, epoch in selection.items() for seq in spec['train']]
    evaluate_jobs(jobs, started)
    call('finish', '_pdv2_fusion_guard_finish.py', [])
    write_json(OUT / 'status.json', {'status': 'complete', 'audit_passed': True,
        'seconds': time.time() - started, 'test_used': False})
    print('FUSION_GUARD_COMPLETE', round(time.time() - started, 1), flush=True)


if __name__ == '__main__':
    main()
