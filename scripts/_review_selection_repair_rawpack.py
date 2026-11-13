#!/usr/bin/env python3
"""임시 rawpack을 원래 모든 스캔과 대조한 뒤 잘못된 offset 기록만 복구한다."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]
from _review_selection_bev import read,write,sha
from _review_selection_full_inputs import OUT
from _review_selection_full_rawpack import home_for


def repair(sequence: str) -> None:
    """원시 모든 점의 연결 SHA와 모든 offset이 맞을 때만 기록을 고친다."""
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    from _pd_build_test_caches import _make_loader
    sensor,seq=sequence.split("/")
    home=home_for("test",sensor,seq)
    info=read(home/"done.json")
    assert sha(home/"raw_ids.npy")==info["raw_ids_sha256"]
    raw=np.load(home/"raw_ids.npy")
    existing=np.load(home/"offsets.npy")
    expected=np.zeros(len(raw)+1,dtype=np.int64)
    loader=_make_loader(sensor,seq)
    digest=hashlib.sha256()
    start=time.time()
    for i,sid in enumerate(raw):
        points=np.asarray(loader[int(sid)]["points"])
        assert points.dtype==np.float32 and points.ndim==2 and points.shape[1]==4
        expected[i+1]=expected[i]+len(points)
        digest.update(np.ascontiguousarray(points).tobytes())
        if (i+1)%128==0 or i+1==len(raw):
            write(OUT/"rawpack_repair_progress"/f"{sensor}_{seq}.json",dict(
                completed=i+1,total=len(raw),seconds=time.time()-start))
            print("REPAIR_SOURCE_AUDIT",sequence,i+1,len(raw),flush=True)
    # 점 데이터는 바꾸지 않는다. 원래 로더 전체와 파일이 같은지 확인한다.
    assert digest.hexdigest()==info["points_sha256"]
    assert sha(home/"points.bin")==info["points_sha256"]
    np.testing.assert_array_equal(existing,expected)
    assert int(expected[-1])*16==info["points_bytes"]
    snapshot=OUT/"rawpack_repairs"/f"{sensor}_{seq}";snapshot.mkdir(parents=True,exist_ok=True)
    before_sha=sha(home/"offsets.npy")
    for name in ("offsets.npy","done.json","verified.json"):
        target=snapshot/("before_"+name)
        assert not target.exists(),target
        shutil.copyfile(home/name,target)
    temporary=home/"offsets.repaired.tmp.npy"
    np.save(temporary,expected)
    with temporary.open("rb") as stream:os.fsync(stream.fileno())
    temporary.replace(home/"offsets.npy")
    np.testing.assert_array_equal(np.load(home/"offsets.npy"),expected)
    previous_recorded=info["offsets_sha256"]
    info["offsets_sha256"]=sha(home/"offsets.npy")
    write(home/"done.json",info)
    write(home/"verified.json",dict(all_arrays_exact=True,all_original_raw_rows_verified=len(raw),
        original_concatenated_points_sha256=digest.hexdigest(),all_offsets_exact=True,
        done_sha256=sha(home/"done.json")))
    for name in ("offsets.npy","done.json","verified.json"):
        with (home/name).open("rb") as stream:os.fsync(stream.fileno())
    write(snapshot/"repair.json",dict(completed_unix=time.time(),seconds=time.time()-start,
        sequence=sequence,original_raw_rows_verified=len(raw),all_offsets_exact=True,
        all_point_bytes_SHA_exact=True,points_file_unchanged=True,
        previous_recorded_offsets_sha256=previous_recorded,observed_offsets_sha256=before_sha,
        repaired_offsets_sha256=info["offsets_sha256"],test_used_for_selection=False,
        cause="기록 SHA와 읽기 SHA 불일치; 원래 점열 전체로 복원 검산. 원인은 단정하지 않는다"))
    print("REPAIR_COMPLETE",sequence,info["offsets_sha256"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence",required=True)
    repair(parser.parse_args().sequence)
