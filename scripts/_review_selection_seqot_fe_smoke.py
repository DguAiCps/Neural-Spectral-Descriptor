#!/usr/bin/env python3
"""검증 입력의 공식 batch64 FE를 로컬에서 실행해 시험 재생성 자원을 확인한다."""
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from _review_selection_bev import read,write,sha
from _review_selection_full_inputs import OUT,SeqSource
import _seqot_official_train as t

record=dict(started_unix=time.time(),sequence='kitti/09',role='val',batch=64,precision='original float32',test_used=False)
try:
    torch.set_num_threads(4)
    try:
        src=SeqSource('kitti','09')
        need=t.centered_need(src,src.kf)[:64]
        record['input_source']='actual validation range'
    except AssertionError as exc:
        # 원래 검증 FE만 보존돼 range가 없는 환경에서는 자원 확인만 한다.
        from types import SimpleNamespace
        values=np.random.default_rng(50309).uniform(0,80,size=(100,32,900)).astype(np.float32)
        src=SimpleNamespace(exists=lambda r:0<=r<100,rows=lambda rows:values[rows])
        need=np.arange(16,80)
        record.update(input_source='synthetic range; memory/shape only, not numerical equivalence',
                      original_input_unavailable=str(exc))
    row=read(OUT/'retained_protocol.json')['methods']['seqot']['1']['fixed_stage1']
    assert sha(ROOT/row['checkpoint'])==row['sha256']
    model=t._load_fe(ROOT/row['checkpoint'],'cuda')
    torch.cuda.reset_peak_memory_stats()
    tick=time.time();values=t.sub_descriptors(model,src,need,'cuda',64,'centered')
    assert values.shape==(64,256) and np.isfinite(values).all()
    record.update(status='complete',seconds=time.time()-tick,cuda_peak_GB=torch.cuda.max_memory_allocated()/1e9,
                  checkpoint_sha256=row['sha256'],output_shape=list(values.shape),output_dtype=str(values.dtype))
except Exception as exc:
    record.update(status='failed',error=repr(exc))
    raise
finally:
    write(OUT/'local_gpu_smoke/seqot_fe_batch64.json',record)
    print(record,flush=True)
