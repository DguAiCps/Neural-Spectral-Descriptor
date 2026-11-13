#!/usr/bin/env python3
"""선택 동결 후 시험 입력 작업량·CPU 처리율만 읽기 전용으로 조사한다."""
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
from _review_selection_bev import read,write,sha,TEST
from _review_selection_full_inputs import OUT,cache
from _pd_build_test_caches import _make_loader
import _seqot_official_prep as prep

assert not os.environ.get('CUDA_VISIBLE_DEVICES')
freeze=OUT/'retained/selection_seqot.json'
assert freeze.exists() and not read(freeze)['test_used_for_selection']
rows=[];profiled=set()
for sensor,seq in TEST:
    tick=time.time();ids=np.load(cache(sensor,seq,'test'))['scan_ids'];loader=_make_loader(sensor,seq)
    needed=prep.centered_raw_ids(ids,len(loader))
    row=dict(sequence=sensor+'/'+seq,keyframes=len(ids),raw_frames=len(loader),needed_range_rows=len(needed),inventory_seconds=time.time()-tick)
    if sensor not in profiled:
        samples=[]
        for at in (0,len(needed)//2,len(needed)-1):
            raw=int(needed[at]);begin=time.time();points=loader[raw]['points'];loaded=time.time()
            image=prep.range_image(sensor,points);end=time.time();assert image.shape==(32,900) and np.isfinite(image).all()
            samples.append(dict(raw=raw,n_points=len(points),load_s=loaded-begin,projection_s=end-loaded))
        row['read_only_CPU_samples']=samples;profiled.add(sensor)
    rows.append(row)
    write(OUT/'seqot_test_input_inventory.json',dict(status='running',rows=rows,selection_sha256=sha(freeze),test_metrics_not_read=True,updated_unix=time.time()))
    print(row,flush=True)
write(OUT/'seqot_test_input_inventory.json',dict(status='complete',rows=rows,selection_sha256=sha(freeze),test_metrics_not_read=True,updated_unix=time.time()))
