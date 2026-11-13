#!/usr/bin/env python3
"""진행 중인 두 실행이 끝나면 GPU 재사용 가능 표시와 CPU 검산을 남긴다."""
import os
from pathlib import Path
import subprocess
import sys
import time
import _review_direction_metrics as base

OUT = base.OUT / 'expanded46'


def active_gpu_workers() -> list:
    """이 작업이 실행한 GPU 실행기만 확인한다."""
    found = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            args = (folder/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(arg.decode(errors='replace').endswith(('_review_direction_expand.py', '_review_direction_cause_queue.py')) for arg in args):
            found.append(int(folder.name))
    return found


def main() -> None:
    """원시 실행을 재시작하지 않고 완료된 파일만 후처리한다."""
    spec = base.read(OUT/'yaw_protocol.json')
    while True:
        rotations = sum((OUT/'yaw'/s.replace('/', '_')/'rotate.json').exists() for s in spec['queries'])
        causes = sum((OUT/'causes'/(s.replace('/', '_')+'.json')).exists() for s in spec['queries'])
        if rotations == causes == 46 and not active_gpu_workers():
            break
        base.write(OUT/'finish_status.json', dict(status='waiting', rotations=rotations, causes=causes, checked_unix=time.time()))
        time.sleep(10)
    base.write(OUT/'gpu_work_complete.json', dict(status='complete', completed_unix=time.time(), active_owned_gpu_workers=[],
        scope='원시회전46시퀀스와원인대조46시퀀스의기존실행기종료. 이후명령은CPU집계·문서·검사이며이작업은GPU를다시사용하지않는다.'))
    print('GPU WORK COMPLETE', time.time(), flush=True)
    commands = [[sys.executable, 'scripts/_review_direction_expanded_report.py', 'all'],
                [sys.executable, 'scripts/_review_direction_write_report.py'],
                [sys.executable, 'scripts/_review_direction_complete.py'],
                [sys.executable, 'docs/도구/check_wiki.py', 'docs']]
    results = []
    for command in commands:
        print('RUN', command, flush=True)
        result = subprocess.run(command, cwd=base.ROOT, capture_output=True, text=True)
        print(result.stdout, result.stderr, flush=True)
        results.append(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        base.write(OUT/'finish_status.json', dict(status='running' if result.returncode == 0 else 'failed',
            checked_unix=time.time(), commands=results))
        if result.returncode:
            return
    base.write(OUT/'finish_status.json', dict(status='passed', completed_unix=time.time(), commands=results))
    print('FINISH PASSED', flush=True)


if __name__ == '__main__':
    main()
