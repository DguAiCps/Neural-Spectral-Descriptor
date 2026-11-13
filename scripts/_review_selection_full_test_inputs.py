#!/usr/bin/env python3
"""선택 후 시험에 필요한 입력 존재만 읽기 전용으로 점검한다."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]


def main() -> None:
    """시험 점수나 지표를 읽지 않고 준비된 입력의 ID·누락 여부만 기록한다."""
    from _review_selection_bev import TEST,write,sha
    from _review_selection_full_inputs import OUT,cache
    import _seqot_official_train as seq
    import _helios_official_prep as h
    import _imlpr_official_riv as im
    import _disco_official_prep as d
    import _deepring_reimpl as deep
    rows=[]
    for sensor,name in TEST:
        row=dict(sequence=sensor+"/"+name)
        expected=np.load(cache(sensor,name,"test"))["scan_ids"]
        try:
            src=seq.StoreSource("test_odom",sensor,name,ext=True)
            np.testing.assert_array_equal(src.kf,expected)
            need=seq.centered_need(src,src.kf)
            raw_need=np.unique(np.concatenate((need-1,need,need+1)))
            raw_need=raw_need[(raw_need>=0)&(raw_need<src.n_raw)]
            missing=np.setdiff1d(raw_need,np.union1d(src.raw_ids,src.ext_ids))
            row["seqot"]=dict(available=not len(missing),n_frames=len(src.kf),needed_rows=len(raw_need),n_missing=len(missing))
        except Exception as exc: row["seqot"]=dict(available=False,error=str(exc))
        home=h.seq_dir("test_odom",sensor,name)
        try:
            z=np.load(home/"index.npz");ids=z["scan_ids"][z["generate"]]
            np.testing.assert_array_equal(ids,expected)
            missing=[int(s) for s in ids if not (home/"LiDAR"/f"{s:06d}.bin").exists()]
            row["helios"]=dict(available=not missing,n_frames=len(ids),n_missing=len(missing),missing=missing[:10])
        except Exception as exc: row["helios"]=dict(available=False,error=str(exc))
        home=im.role_dir("test_odom",sensor,name)
        try:
            ids=np.load(home/"index.npz")["scan_ids"]
            np.testing.assert_array_equal(ids,expected)
            missing=[k for k in range(len(ids)) if not (home/f"{k:06d}.npy").exists()]
            row["imlpr"]=dict(available=not missing,n_frames=len(ids),n_missing=len(missing),missing=missing[:10])
        except Exception as exc: row["imlpr"]=dict(available=False,error=str(exc))
        home=d.store_dir("test_odom",sensor,name)
        try:
            bits,index,meta=d.load_store("test_odom",sensor,name)
            ids=index["scan_ids"]
            np.testing.assert_array_equal(ids,expected)
            assert sha(home/"bits.npy")==meta["bits_sha256"]
            row["disco"]=dict(available=True,n_frames=len(bits),bits_sha256=meta["bits_sha256"])
        except Exception as exc: row["disco"]=dict(available=False,error=str(exc))
        try:
            packed,meta=deep.load_bev("test_odom",sensor,name)
            row["deepring"]=dict(available=True,n_frames=len(packed))
        except Exception as exc: row["deepring"]=dict(available=False,error=str(exc))
        rows.append(row)
    write(OUT/"test_input_availability_audit.json",dict(created_unix=time.time(),read_only=True,test_metrics_not_read=True,rows=rows,
        corrected_initial_disco_predicate="원래cmd_gen은완료후done.npy를삭제; 공식load_store와bits SHA로검사"))
    methods=("seqot","helios","imlpr","disco","deepring")
    print({m:sum(r[m]["available"] for r in rows) for m in methods},flush=True)
    print([r for r in rows if not all(r[m]["available"] for m in methods)],flush=True)


if __name__=="__main__":
    main()
