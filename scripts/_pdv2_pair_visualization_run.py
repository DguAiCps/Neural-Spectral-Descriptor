#!/usr/bin/env python3
"""그림 진단의 GPU 작업을 지정 컨테이너에서 차례로 실행한다."""
import json
import os
import subprocess
from pathlib import Path
import _pdv2_pair_visualization as analysis


def main() -> None:
    """기존 완료 기록을 재사용하며 단계별 로그와 상태를 남긴다."""
    logs = analysis.REPO/'logs/pdv2_pair_visualization_20261030'
    logs.mkdir(parents=True, exist_ok=True)
    for sequence in analysis.SEQUENCES:
        for action in ('raw', 'nsd', 'bevplace', 'helios', 'seqot', 'ringsharp'):
            if action in os.environ.get('PAIR_SKIP_METHODS', '').split(','):
                continue
            path = logs/(sequence.replace('/', '_')+'_'+action+'.log')
            analysis.write(analysis.OUT/'status.json', {'status': 'running', 'sequence': sequence, 'action': action})
            cmd = ['docker', 'exec', '-w', '/workspace/Neural-Spectral-Codec', 'ring_baseline',
                   'python3', '-u', 'scripts/_pdv2_pair_visualization.py', action, '--sequence', sequence]
            print('START', sequence, action, flush=True)
            with path.open('a') as output:
                result = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                analysis.write(analysis.OUT/'status.json', {'status': 'failed', 'sequence': sequence,
                               'action': action, 'log': str(path.relative_to(analysis.REPO))})
                print(path.read_text()[-4500:], flush=True)
                raise SystemExit(result.returncode)
            print('DONE', sequence, action, flush=True)
    analysis.write(analysis.OUT/'status.json', {'status': 'gpu_complete', 'sequences': list(analysis.SEQUENCES)})


if __name__ == '__main__':
    main()
