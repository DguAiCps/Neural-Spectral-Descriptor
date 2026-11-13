#!/usr/bin/env python3
"""고정 두 모델의 학습·검증 최종 검색 교차 진단을 실행한다."""
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

OUT = REPO / 'results/pdv2_fusion_factor_20261023'
LOG = REPO / 'logs/fusion_factor_20261023'


def call(label: str, script: str, arguments: list) -> None:
    """작업별 로그를 남기고 실패하면 중단한다."""
    with (LOG / (label + '.log')).open('w') as handle:
        result = subprocess.run([sys.executable, '-u', str(REPO / 'scripts' / script), *arguments],
            cwd=REPO, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                              OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(label + '\n' + (LOG / (label + '.log')).read_text()[-6000:])


def main() -> None:
    """키 추출 뒤 독립 시퀀스 두 개씩 평가한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    start = time.time()
    script = '_pdv2_fusion_factor.py'
    call('prepare', script, ['prepare'])
    write_json(OUT / 'status.json', {'stage': 'keys', 'status': 'running', 'test_used': False})
    call('training_keys', script, ['keys'])
    spec = json.loads((OUT / 'protocol.json').read_text())
    jobs = [(role, seq) for role in ('train', 'val') for seq in spec[role]]
    def worker(job):
        role, seq = job
        call(role + '_' + seq.replace('/', '_'), script, ['replay', '--role', role, '--sequence', seq])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            write_json(OUT / 'status.json', {'status': 'running', 'stage': 'replay', 'done': count,
                'total': len(jobs), 'seconds': time.time() - start, 'test_used': False})
            print('REPLAY_DONE', count, len(jobs), flush=True)
    call('finish', '_pdv2_fusion_factor_finish.py', [])
    write_json(OUT / 'status.json', {'status': 'complete', 'audit_passed': True,
        'seconds': time.time() - start, 'test_used': False, 'training_updates': 0})
    print('FUSION_FACTOR_COMPLETE', round(time.time() - start, 1), flush=True)


if __name__ == '__main__':
    main()
