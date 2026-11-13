#!/usr/bin/env python3
"""서버에서 계산한 검증736원장의 서술자를 로컬에서도 전수 재현한다."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from _review_selection_bev import read,write,sha
from _review_selection_full_eval import OUT,candidate,folder,positions,descriptors,query_base
from _review_selection_full_inputs import cache
from _review_selection_full_l2 import ledger_exact

torch.set_num_threads(2)
os.environ['NVIDIA_TF32_OVERRIDE']='0'
path=OUT/'seqot_validation_device_audit.json'
state=read(path) if path.exists() else dict(checks=[],failures=[])
cutover=read(OUT/'local_gpu_plan.json')['start_positions']['seqot']
assert cutover==736
jobs=read(OUT/'full_queue.json')['streams']['seqot'][:cutover]
state.update(pid=os.getpid(),status='running',total=len(jobs),selection_sha256=sha(OUT/'retained/selection_seqot.json'),
             scope='frozen validation only; descriptors from server segment are reproduced on local GPU; test not used')
for index,job in enumerate(jobs):
    if index<len(state['checks']):continue
    tick=time.time();sensor,seq=job['sequence'].split('/')
    args=argparse.Namespace(method='seqot',seed=job['seed'],candidate=job['candidate'],sequence=job['sequence'],role='val',scope='retained')
    row=candidate('seqot',args.seed,args.candidate,'retained');home=folder('seqot',args.seed,args.candidate,'retained','val')
    reference=read(home/f'{sensor}_{seq}.json');ids,pos=positions(sensor,seq,'val')
    desc,_=descriptors(args,row,ids,home)
    digest=hashlib.sha256();digest.update(json.dumps([list(desc.shape),str(desc.dtype),sha(cache(sensor,seq,'val')),
        sha(ROOT/'scripts/_disco_official_dump.py')]).encode());digest.update(np.ascontiguousarray(desc).tobytes())
    expected=(reference.get('audit') or {}).get('descriptor_sha256')
    if expected is not None:
        exact=digest.hexdigest()==expected;kind='full descriptor bytes SHA256'
    else:
        actual,_=ledger_exact(desc,pos,query_base(sensor,seq,'val'))
        with np.load(ROOT/reference['ledger']) as z:exact=all(np.array_equal(v,z[k]) for k,v in actual.items())
        kind='all ledger arrays exact; legacy record has no descriptor SHA'
    record=dict(job_index=index,seed=args.seed,candidate=args.candidate,sequence=args.sequence,exact=bool(exact),
                check=kind,original_descriptor_sha256=expected,local_descriptor_sha256=digest.hexdigest(),seconds=time.time()-tick)
    state['checks'].append(record)
    if not exact:state['failures'].append(record)
    state.update(completed=len(state['checks']),updated_unix=time.time());write(path,state)
    if index%25==0:print('DEVICE_AUDIT',index+1,len(jobs),'failures',len(state['failures']),flush=True)
state.update(status='complete',all_exact=not state['failures'],completed_unix=time.time());write(path,state)
