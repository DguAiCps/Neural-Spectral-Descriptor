#!/usr/bin/env python3
"""장소 제외 예비의 학습과 키 생성을 순차 실행하고 상태를 보존한다."""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src'),str(REPO)]
import _pdv2_encoder_place as exp
from _pdv2_no_context_ablation import write_json


def main() -> None:
    """각 모델의 로그와 실패 위치를 남긴다."""
    logs=REPO/'logs/pdv2_encoder_place_20261019'
    logs.mkdir(parents=True,exist_ok=True)
    started=time.time()
    spec=exp.protocol()
    for index,arm in enumerate(spec['plans']):
        for action in ('train','keys'):
            write_json(exp.OUT/'status.json',{'status':'running','stage':action,'arm':arm,
                       'model_index':index+1,'total_models':len(spec['plans']),'started_unix':started})
            with (logs/f'{action}_{arm}.log').open('a') as handle:
                result=subprocess.run([sys.executable,'-u','scripts/_pdv2_encoder_place.py',action,'--arm',arm],
                    cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{action} {arm}: {logs}')
            print('DONE',action,arm,round(time.time()-started,1),flush=True)
    hashes={exp.read(exp.folder(a)/'training_complete.json')['initial_sha256'] for a in spec['plans']}
    assert len(hashes)==1
    for pair in spec['pairs'].values():
        a,b=[exp.read(exp.folder(pair[k])/'training_complete.json') for k in ('leave','matched')]
        assert a['updates']==b['updates']
    write_json(exp.OUT/'status.json',{'status':'training_complete','runtime_s':time.time()-started,
               'models':len(spec['plans']),'initial_states_identical':True,'matched_updates_equal':True,
               'test_used':False})
    print('TRAINING_COMPLETE',round(time.time()-started,1),flush=True)


if __name__=='__main__':
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:main()
    except Exception as error:
        write_json(exp.OUT/'status.json',{'status':'failed','error':repr(error)})
        raise
