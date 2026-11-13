#!/usr/bin/env python3
"""완료 입력 다운로드와 겹쳐 남은 검증 RIV를 서버 CPU에서 미리 계산한다."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _review_selection_bev import read,write
OUT=ROOT/'results/pdv2_review_selection_full_20261105'

lock=(OUT/'val_riv_prefetch.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
path=OUT/'val_riv_prefetch_state.json'
state=read(path) if path.exists() else dict(completed=[])
state.update(pid=os.getpid(),started_unix=time.time())
for seq in ('Roundabout02','Roundabout03'):
    if seq in state['completed']:continue
    state.update(status='running',sequence=seq,processes=24,updated_unix=time.time());write(path,state)
    # server_riv의 시퀀스 잠금이 기존 controller와 같은 입력의 중복 생성을 막는다.
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/_review_selection_full_server_riv.py'),
                    '--role','val','--sequence','helipr/'+seq,'--processes','24'],check=True)
    state['completed'].append(seq);write(path,state)
state.update(status='complete',updated_unix=time.time());write(path,state)
