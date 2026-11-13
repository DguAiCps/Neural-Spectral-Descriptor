#!/usr/bin/env python3
"""동결된 비교 방법의 추가 지도 크기를 직렬 측정하고 완료마다 게시한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/pdv2_table3_20261029'
PAUSE = OUT / 'runner_pause'


def write(path: Path, value: dict) -> None:
    """상태를 원자적으로 저장한다."""
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def docker(script: str, *args: str) -> list:
    """공통 컨테이너와 수치 스레드 조건으로 명령을 만든다."""
    return ['docker', 'exec', '-w', '/workspace/Neural-Spectral-Codec',
            'ring_baseline', 'env', 'OPENBLAS_NUM_THREADS=1',
            'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'LOKY_MAX_CPU_COUNT=1',
            'python3', '-u', 'scripts/' + script, *args]


def commands(method: str, n_map: int) -> list:
    """기존 어댑터의 준비·계측만 호출하며 구현을 바꾸지 않는다."""
    size = ['--n-map', str(n_map)]
    if method == 'scpp':
        name = '_pdv2_table3_scpp.py'
        actions = [('prepare', docker(name, 'prepare', *size))]
        if not (OUT / f'scpp_n{n_map}/cache_complete.json').exists():
            actions.append(('cache', [sys.executable, 'scripts/' + name, 'cache', *size]))
        actions.append(('measure', [sys.executable, 'scripts/' + name, 'measure', *size]))
        return actions
    if method in ('disco', 'imlpr', 'helios'):
        name = '_pdv2_table3_global.py' if method == 'disco' else '_pdv2_table3_global_v2.py'
        actions = [('cache', docker(name, 'cache', '--method', method, *size))]
        if method in ('imlpr', 'helios'):
            actions = [('cache', docker('_pdv2_table3_prefix_cache.py', '--method', method, *size))]
        if method == 'helios':
            actions.append(('measure', docker('_pdv2_table3_helios_v3.py', *size)))
        else:
            actions.append(('measure', docker(name, 'measure', '--method', method, *size)))
        return actions
    if method == 'ringsharp':
        return [('cache', docker('_pdv2_table3_ringsharp.py', 'cache', *size)),
                ('measure', docker('_pdv2_table3_ringsharp_v2.py', *size))]
    name = {'seqot': '_pdv2_table3_seqot.py',
            'ringsharp': '_pdv2_table3_ringsharp.py',
            'deepring': '_pdv2_table3_deepring.py',
            'ring': '_pdv2_table3_ring.py',
            'ringpp': '_pdv2_table3_ringpp_v3.py'}[method]
    extra = ['--method', method] if method == 'ring' else []
    return [(action, docker(name, action, *extra, *size)) for action in ('cache', 'measure')]


def main() -> None:
    """개별 실패는 보존하고 다른 방법을 계속하며 게시 실패는 즉시 멈춘다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', default=[
        'seqot', 'ringsharp', 'scpp', 'disco', 'deepring', 'ring', 'ringpp', 'imlpr', 'helios'])
    parser.add_argument('--n-maps', nargs='+', type=int, default=[5000, 10000])
    args = parser.parse_args()
    logs = OUT / 'remaining_logs'
    logs.mkdir(exist_ok=True)
    status = {'status': 'running', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'started_unix': time.time(), 'completed': [], 'skipped': [], 'failed': [],
              'plan': [{'method': method, 'n_map': n_map} for method in args.methods for n_map in args.n_maps]}
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '1'
    for item in status['plan']:
        method, n_map = item['method'], item['n_map']
        result_path = OUT / 'methods' / f'{method}_n{n_map}.json'
        if result_path.exists():
            row = json.loads(result_path.read_text())
            assert row['status'] == row['verification']['status'] == 'passed' and row.get('publish_ready', True)
            status['skipped'].append(item)
            continue
        failed = False
        for action, command in commands(method, n_map):
            while PAUSE.exists():
                status.update(status='paused_between_commands', current={**item, 'next_action': action})
                write(OUT/'remaining_status.json', status)
                time.sleep(2)
            log = logs / f'{method}_n{n_map}_{action}_{time.time_ns()}.log'
            status.update(status='running', current={**item, 'action': action, 'command': command,
                                                    'log': str(log.relative_to(ROOT)), 'started_unix': time.time()})
            write(OUT/'remaining_status.json', status)
            print('START', method, n_map, action, flush=True)
            with log.open('w') as handle:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode:
                status['failed'].append({**status['current'], 'returncode': completed.returncode})
                print('FAILED', method, n_map, action, completed.returncode, flush=True)
                failed = True
                break
        if failed:
            continue
        row = json.loads(result_path.read_text())
        assert row['status'] == row['verification']['status'] == 'passed'
        status.update(current={**item, 'action': 'publish', 'started_unix': time.time()})
        write(OUT/'remaining_status.json', status)
        log = logs / f'{method}_n{n_map}_publish_{time.time_ns()}.log'
        command = [sys.executable, 'scripts/_pdv2_table3_publish.py', '--reason',
                   f'{method} N{n_map} scale measurement validated']
        with log.open('w') as handle:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            status.update(status='publication_failed', publication_log=str(log.relative_to(ROOT)))
            write(OUT/'remaining_status.json', status)
            raise RuntimeError(f'Publication failed: {log}')
        status['completed'].append(item)
        write(OUT/'remaining_status.json', status)
        print('PUBLISHED', method, n_map, flush=True)
    status.update(status='completed_with_failures' if status['failed'] else 'complete', completed_unix=time.time())
    write(OUT/'remaining_status.json', status)
    print(status['status'], len(status['completed']), 'new rows', flush=True)


if __name__ == '__main__':
    main()
