#!/usr/bin/env python3
"""학습·선택과 별개인 검색 동일성 검산에 완성 서술자를 복사한다."""
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _review_selection_bev import read,write,sha
from _review_selection_full_eval import OUT,positions

home=OUT/'cosine_acceleration_audit'
home.mkdir(parents=True,exist_ok=True)
source=OUT/'retained/val/helios/s2/e008_d7a45d80d7c9/descriptors_mulran_Sejong03.npy'
progress=source.with_name('descriptor_progress_mulran_Sejong03.json')
target=home/'helios_Sejong03.npy'
if not target.exists():
    while True:
        info=read(progress)
        if info['completed']==info['total']:
            values=np.load(source).copy()
            assert len(values)==info['total'] and np.isfinite(values).all()
            np.save(target,values)
            break
        write(home/'snapshot_status.json',dict(status='waiting_complete_helios_descriptors',completed=info['completed'],total=info['total'],updated_unix=time.time()))
        time.sleep(2)
cases=[]
for label,desc,sensor,seq in [('helios_s2_early_model_Sejong03',target,'mulran','Sejong03'),
    ('imlpr_s1_KITTI3600004',home/'imlpr_KITTI3600004.npy','kitti360','0004')]:
    _,pos=positions(sensor,seq,'val');dest=home/(label+'_positions.npy');np.save(dest,pos)
    assert len(np.load(desc,mmap_mode='r'))==len(pos)
    cases.append(dict(label=label,descriptors=str(desc.relative_to(ROOT)),sha256=sha(desc),positions=str(dest.relative_to(ROOT))))
write(home/'manifest.json',dict(created_unix=time.time(),scope='search equivalence only; frozen candidate set and objectives unchanged',cases=cases))
write(home/'snapshot_status.json',dict(status='complete',updated_unix=time.time()))
