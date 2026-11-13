#!/usr/bin/env python3
"""재부팅 때 역전송됐을 수 있는 비누적 입력의 크기를 보존 원본과 점검한다."""
from pathlib import Path
import os
import time
import numpy as np

from _review_selection_bev import write,sha,VAL
from _review_selection_full_inputs import OUT,INPUT,cache

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    rows=[];bad=[];repaired=[];same_inode=0;same_hash=0;new_valid=0
    for sensor,seq in VAL:
        if sensor=="helipr":continue
        ids=np.load(cache(sensor,seq))["scan_ids"]
        old=ROOT/"_external/imlpr_official_pdv2/data/validation"/f"{sensor}_{seq}"
        lookup={int(s):i for i,s in enumerate(np.load(old/"index.npz")["scan_ids"])}
        home=INPUT/"imlpr/data/test_odom"/f"{sensor}_{seq}"
        for i,sid in enumerate(ids):
            target=home/f"{i:06d}.npy"
            source=old/f"{lookup[int(sid)]:06d}.npy" if int(sid) in lookup else None
            if source is not None and source.exists() and target.exists():
                if os.path.samefile(source,target):same_inode+=1;continue
                if sha(source)==sha(target):same_hash+=1;continue
            if source is None and target.exists():
                try:
                    arr=np.load(target,mmap_mode="r")
                    assert arr.dtype==np.float32 and arr.shape==(64,1022,3)
                    new_valid+=1;continue
                except Exception:pass
            if source is None or not source.exists() or source.stat().st_size!=64*1022*3*4+128:
                bad.append(dict(sequence=sensor+"/"+seq,position=i,raw_id=int(sid),target=str(target.relative_to(ROOT))))
                continue
            if target.exists():target.rename(target.with_name(".invalid_reboot_"+target.name))
            os.link(source,target)
            repaired.append(dict(target=str(target.relative_to(ROOT)),source=str(source.relative_to(ROOT)),sha256=sha(target)))
        rows.append(dict(sequence=sensor+"/"+seq,frames=len(ids)))
    write(OUT/"restart_server_nonheli_audit.json",dict(status="complete" if not bad else "requires_regeneration",
          updated_unix=time.time(),rows=rows,repaired=repaired,unrestored=bad,immutable_external_sources_unchanged=True,
          preserved_same_inode=same_inode,preserved_same_SHA=same_hash,new_input_valid_headers=new_valid))
    print('same_inode',same_inode,'same_SHA',same_hash,'new_valid',new_valid,'repaired',len(repaired),'unrestored',len(bad),flush=True)


if __name__=="__main__":main()
