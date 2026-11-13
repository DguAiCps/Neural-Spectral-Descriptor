#!/usr/bin/env python3
"""attention 관측을 끝낸 뒤 네 고정 추론 개입의 검증을 실행한다."""
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

OUT = REPO / 'results/pdv2_attention_review_20261023'
LOG = REPO / 'logs/attention_review_20261023'


def call(label: str, script: str, arguments: list) -> None:
    """독립 프로세스의 출력을 파일로 보존한다."""
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    path = LOG / (label + '.log')
    with path.open('a') as handle:
        result = subprocess.run([sys.executable, '-u', str(REPO / 'scripts' / script), *arguments],
                                cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(label + '\n' + path.read_text()[-4000:])


def main() -> None:
    """관측·최종 검증·집계를 실행한다. 가중치는 학습하지 않는다."""
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    start = time.time()
    script = '_pdv2_attention_review.py'
    try:
        call('prepare', script, ['prepare'])
        write_json(OUT / 'status.json', {'status': 'running', 'stage': 'attention_statistics', 'test_used': False})
        call('statistics', '_pdv2_attention_review_stats.py', [])
        spec = json.loads((OUT / 'protocol.json').read_text())
        jobs = [(a, s) for a in spec['config']['arms'] for s in spec['val']]
        def evaluate(job):
            a, s = job
            call('val_' + a + '_' + s.replace('/', '_'), script, ['val', '--arm', a, '--sequence', s])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(evaluate, job) for job in jobs]
            for count, f in enumerate(concurrent.futures.as_completed(futures), 1):
                f.result()
                write_json(OUT / 'status.json', {'status': 'running', 'stage': 'val',
                    'done': count, 'total': len(jobs), 'elapsed_s': time.time() - start, 'test_used': False})
                print('VAL_DONE', count, len(jobs), flush=True)
        call('report', script, ['report'])
        write_json(OUT / 'status.json', {'status': 'evaluation_complete',
            'elapsed_s': time.time() - start, 'test_used': False})
    except Exception as error:
        write_json(OUT / 'status.json', {'status': 'failed', 'reason': str(error), 'test_used': False})
        raise

if __name__ == '__main__':
    main()
