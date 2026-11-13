#!/usr/bin/env python3
"""재부팅 뒤 생성 캐시를 검산해 D:로 연결하고 C: 사본은 보존한다."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import numpy as np

from _review_selection_bev import read,write,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/pdv2_review_selection_full_20261105"
DEST=Path("/mnt/d/NSD_datasets/review_selection_cache_20261105/inputs/imlpr")


def main() -> None:
    """복사 완료·전체파일 해시를 확인한 뒤 경로만 바꾼다."""
    source=OUT/"inputs/imlpr"
    status=OUT/"restart_imlpr_storage.json"
    if source.is_symlink():
        assert source.resolve()==DEST.resolve()
        return
    DEST.mkdir(parents=True,exist_ok=True)
    write(status,dict(status="copying",started_unix=time.time(),source=str(source),destination=str(DEST)))
    subprocess.run(["rsync","-aO","--no-owner","--no-group","--no-perms","--exclude=*.tmp.npy",str(source)+"/",str(DEST)+"/"],check=True)
    files=[f for f in source.rglob("*") if f.is_file() and not f.name.endswith(".tmp.npy")]
    verified=[];invalid=[]
    for i,path in enumerate(files):
        target=DEST/path.relative_to(source)
        a,b=sha(path),sha(target)
        assert a==b,path
        if path.suffix==".npy" and path.stem.isdigit():
            try:
                arr=np.load(target,mmap_mode="r")
                assert arr.dtype==np.float32 and arr.ndim==3 and arr.shape[1:]==(1022,3)
            except Exception as exc:
                invalid.append(dict(path=str(target),error=str(exc),sha256=a))
                target.rename(target.with_name(".invalid_reboot_"+target.name))
        verified.append(dict(relative=str(path.relative_to(source)),sha256=a))
        if (i+1)%500==0:write(status,dict(status="verifying",verified=i+1,total=len(files),invalid=invalid,updated_unix=time.time()))
    write(OUT/"restart_imlpr_storage_manifest.json",dict(files=verified,invalid=invalid))
    code="from pathlib import Path; import sys; p=Path(sys.argv[1]); backup=p.with_name('imlpr_pre_restart_2249'); assert not backup.exists(); p.rename(backup); p.symlink_to(sys.argv[2],target_is_directory=True)"
    subprocess.run(["docker","exec","-e","CUDA_VISIBLE_DEVICES=","ring_baseline","python3","-c",code,
                    "/workspace/Neural-Spectral-Codec/"+str(source.relative_to(ROOT)),str(DEST)],check=True)
    write(status,dict(status="complete",verified=len(files),invalid=invalid,destination=str(DEST),
                      C_source_preserved_as="inputs/imlpr_pre_restart_2249",completed_unix=time.time()))


if __name__=="__main__":
    main()
