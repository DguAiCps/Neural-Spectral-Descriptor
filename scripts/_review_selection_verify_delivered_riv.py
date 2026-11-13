#!/usr/bin/env python3
"""서버에서 받은 누적 검증 RIV의 전체 SHA가 맞아야 GPU 입력을 연다."""
import fcntl
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _review_selection_bev import read,write,sha
OUT=ROOT/'results/pdv2_review_selection_full_20261105'

lock=(OUT/'verify_delivered_riv.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for seq in ('Roundabout01','Roundabout02','Roundabout03'):
    target=OUT/'delivered_riv_verified'/f'helipr_{seq}.json'
    if target.exists() and read(target).get('all_files_exact'):continue
    while seq not in read(OUT/'restart_server_inputs_state.json').get('completed',[]):time.sleep(10)
    home=OUT/'inputs/imlpr/data/test_odom'/f'helipr_{seq}'
    done=read(home/'done.json');generated=read(home/'server_generated_hashes.json')
    assert sha(home/'server_generated_hashes.json')==done['generated_hash_manifest_sha256']
    preserved=read(OUT/'val_riv_handoff.json')['completed_files']
    tick=time.time()
    for i in range(done['n_frames']):
        f=home/f'{i:06d}.npy';expected=generated.get(f.name,preserved.get(str(f.relative_to(ROOT))))
        assert expected is not None and sha(f)==expected,f
        if i%128==0:write(OUT/'delivered_riv_verify_state.json',dict(pid=os.getpid(),sequence=seq,completed=i,total=done['n_frames'],seconds=time.time()-tick))
    write(target,dict(all_files_exact=True,n_frames=done['n_frames'],done_sha256=sha(home/'done.json'),
        generated_hash_manifest_sha256=sha(home/'server_generated_hashes.json'),seconds=time.time()-tick,completed_unix=time.time()))
write(OUT/'delivered_riv_verify_state.json',dict(status='complete',pid=os.getpid(),completed_unix=time.time()))
