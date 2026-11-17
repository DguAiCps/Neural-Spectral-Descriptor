#!/usr/bin/env python3
"""누락된 DiSCO 보존 가중치와 검증 입력을 D:에 해시로 복구한다."""
import os
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _review_selection_bev import read,write,sha
OUT=ROOT/'results/pdv2_review_selection_full_20261105'
DISK=Path('/mnt/d/NSD_datasets/review_selection_cache_20261105')
REMOTE='joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/'
spec=read(OUT/'retained_protocol.json')
rows=[r for e in spec['methods']['disco'].values() for r in e['candidates']]
status=OUT/'disco_local_restore_state.json'
missing=[r for r in rows if not (ROOT/r['path']).exists()]
listing=OUT/'disco_missing_checkpoint_files.txt';listing.write_text('\n'.join(r['path'] for r in missing)+'\n')
home=DISK/'checkpoints/disco';home.mkdir(parents=True,exist_ok=True)
write(status,dict(status='copying_checkpoints',pid=os.getpid(),missing=len(missing),GPU_started=False,updated_unix=time.time()))
if missing:
    subprocess.run(['rsync','-aO','--no-owner','--no-group','--no-perms','--files-from='+str(listing),REMOTE,str(home)+'/'],check=True)
    for row in missing:
        source=home/row['path'];assert sha(source)==row['sha256']
    # 기존 외부 저장소의 소유권은 바꾸지 않고 새 경로만 컨테이너에서 만든다.
    pairs=[(row['path'],str(home/row['path'])) for row in missing]
    code="import json,sys; from pathlib import Path; root=Path('/workspace/Neural-Spectral-Codec'); pairs=json.loads(sys.argv[1]);\nfor relative,source in pairs:\n target=root/relative\n if not target.exists():\n  assert not target.is_symlink(),target\n  target.parent.mkdir(parents=True,exist_ok=True)\n  target.symlink_to(source)\n"
    subprocess.run(['docker','exec','ring_baseline','python3','-c',code,json.dumps(pairs)],check=True)
for row in rows:assert sha(ROOT/row['path'])==row['sha256']
cache=DISK/'inputs/disco';cache.mkdir(parents=True,exist_ok=True)
logical=OUT/'inputs/disco'
if not logical.exists():logical.symlink_to(cache)
assert logical.resolve()==cache.resolve()
write(status,dict(status='copying_validation_polar_cache',pid=os.getpid(),weights_all_exact=True,GPU_started=False,updated_unix=time.time()))
subprocess.run(['rsync','-aO','--no-owner','--no-group','--no-perms','--exclude=.*',
                REMOTE+str(logical.relative_to(ROOT))+'/',str(cache)+'/'],check=True)
stores=[]
for path in sorted((cache/'polar/test_odom').glob('*/meta.json')):
    meta=read(path);stores.append(dict(sequence=path.parent.name,complete=meta.get('complete',False),metadata=meta))
write(status,dict(status='restored_pending_CPU_cache_audit_and_GPU_smoke',pid=os.getpid(),weights_all_exact=True,
    weights=rows,stores=stores,validation_cache_root=str(cache),GPU_started=False,updated_unix=time.time()))
