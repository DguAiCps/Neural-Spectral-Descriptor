#!/usr/bin/env python3
"""서버에 없는 검증 원시 입력을 로컬 CPU로 복구해 전송한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]
from _review_selection_bev import read, write, sha, VAL
from _review_selection_full_inputs import OUT, INPUT, cache, configure


def plan() -> None:
    """서버의 실제 보존 입력에서 누락된 raw ID만 고정한다."""
    target = OUT / "input_relay_plan.json"
    if target.exists():
        return
    import _seqot_official_prep as s
    import _ringsharp_official as r
    rows = []
    for sensor, seq in VAL:
        ids = np.load(cache(sensor,seq))["scan_ids"]
        old = np.load(ROOT/f"_external/disco_official_pdv2/polar/val/{sensor}_{seq}/index.npz")["scan_ids"]
        missing = np.flatnonzero(~np.isin(ids,old)).tolist()
        mm, raw, _, _, nr, ext, eraw = s.open_store("val",sensor,seq,ext=True)
        need = s.centered_raw_ids(ids,nr)
        smissing = np.setdiff1d(need,np.union1d(raw,eraw))
        oldr = np.load(r.VAL_BEV/f"{sensor}_{seq}.npz")["scan_ids"]
        row = dict(sensor=sensor,seq=seq,cache_sha256=sha(cache(sensor,seq)),scan_ids=ids.tolist(),
            shared_missing_positions=missing,
            helios_positions=list(range(len(ids))) if sensor=="helipr" else missing,
            imlpr_positions=list(range(len(ids))) if sensor=="helipr" else missing,
            ringsharp_positions=np.flatnonzero(~np.isin(ids,oldr)).tolist(),
            seqot_missing_raw=smissing.tolist(),seqot_shape=list(mm.shape[1:]),seqot_dtype=str(mm.dtype),
            seqot_nraw=nr,seqot_base=len(raw),seqot_ext=len(eraw),seqot_needed=len(need))
        rows.append(row)
    write(target,dict(created_unix=time.time(),reason="서버 원시데이터 mount가 비어 있음; 로컬 원시를 CPU로 처리/무손실 전달",
        methods=["seqot","raw","ringsharp","helios","imlpr"],rows=rows))


def cpu_worker(method: str, key: str) -> None:
    """한 검증 시퀀스의 누락 입력만 공식 경로로 계산한다."""
    assert not os.environ.get("CUDA_VISIBLE_DEVICES"), "로컬 GPU 사용 금지"
    from _pd_build_test_caches import _make_loader
    sensor, seq = key.split("/")
    if method=="test_ringsharp":
        import _ringsharp_official as p
        home=INPUT/method/f"{sensor}_{seq}"; home.mkdir(parents=True,exist_ok=True)
        if (home/"done.json").exists(): return
        ids=np.load(cache(sensor,seq,"test"))["scan_ids"]
        path=home/"packed.npy"
        packed=np.lib.format.open_memmap(path,mode="r+" if path.exists() else "w+",dtype=np.uint8,
                                          shape=(len(ids),64000))
        progress=home/"progress.json"
        start=read(progress)["completed"] if progress.exists() else 0
        base,loader=p._base(),p.rol.make_loader(sensor,seq)
        for batch in p.bev_batches_from_loader(base,loader,ids[start:],p.rol.ringsharp_bounds(sensor)):
            bits=np.packbits(batch.numpy().reshape(len(batch),-1)>0,axis=1)
            packed[start:start+len(batch)]=bits; start+=len(batch)
            if start%128==0 or start==len(ids):
                packed.flush(); write(progress,dict(completed=start,total=len(ids)))
        write(home/"done.json",dict(total=len(ids),cache_sha256=sha(cache(sensor,seq,"test"))))
        return
    row = next(r for r in read(OUT/"input_relay_plan.json")["rows"] if r["sensor"]==sensor and r["seq"]==seq)
    assert sha(cache(sensor,seq)) == row["cache_sha256"]
    ids = np.asarray(row["scan_ids"],np.int64)
    marker = INPUT / "relay_status" / f"{method}_{sensor}_{seq}.json"
    if marker.exists() and read(marker).get("status")=="complete":
        return
    loader = _make_loader(sensor,seq)
    tick = time.time()
    if method=="seqot":
        import _seqot_official_prep as p
        home = INPUT/"seqot"/f"{sensor}_{seq}"
        home.mkdir(parents=True,exist_ok=True)
        raws=np.asarray(row["seqot_missing_raw"],np.int64)
        np.save(home/"missing_ids.npy",raws)
        arr=np.lib.format.open_memmap(home/"missing.npy",mode="r+" if (home/"missing.npy").exists() else "w+",
            dtype=row["seqot_dtype"],shape=(len(raws),*row["seqot_shape"]))
        start=read(marker).get("completed",0) if marker.exists() else 0
        for i in range(start,len(raws)):
            arr[i]=p.range_image(sensor,loader[int(raws[i])]["points"])
            if (i+1)%100==0 or i+1==len(raws):
                arr.flush()
                write(marker,dict(status="running",completed=i+1,total=len(raws),seconds=time.time()-tick))
        write(home/"done.json",dict(base_rows=row["seqot_base"],extension_rows=row["seqot_ext"],
            new_rows=len(raws),total_needed=row["seqot_needed"],cache_sha256=row["cache_sha256"]))
    elif method=="raw":
        home=INPUT/"raw"/f"{sensor}_{seq}"
        home.mkdir(parents=True,exist_ok=True)
        for pos in row["shared_missing_positions"]:
            sid=int(ids[pos]); dest=home/f"{sid:06d}.npz"
            if dest.exists():
                continue
            item=loader[sid]
            np.savez_compressed(dest,points=item["points"],pose=item["pose"])
    elif method=="ringsharp":
        import _ringsharp_official as p
        home=INPUT/"ringsharp"/f"{sensor}_{seq}"
        home.mkdir(parents=True,exist_ok=True)
        positions=np.asarray(row["ringsharp_positions"],np.int64)
        base=p._base()
        result=[]
        for b in p.bev_batches_from_loader(base,loader,ids[positions],p.rol.ringsharp_bounds(sensor)):
            result.append(np.packbits(b.numpy().reshape(len(b),-1)>0,axis=1))
        np.savez_compressed(home/"relay_missing.npz",positions=positions,
            packed=np.concatenate(result) if result else np.empty((0,64000),np.uint8))
    elif method=="helios":
        p=configure(method)
        # 한 시퀀스만 색인해 병렬 파일 쓰기 충돌을 피한다.
        p.ROLES["test_odom"] = ([(sensor,seq)],p.ODOM,"val_","poses_gt")
        p.cmd_index(["test_odom"],10.0,"helipr")
        orig_index=p._index
        allowed=np.zeros(len(ids),bool); allowed[row["helios_positions"]]=True
        def index_view(*args):
            z=orig_index(*args)
            return {k: (z[k]&allowed if k=="generate" else z[k]) for k in z.files}
        p._index=index_view
        p.cmd_gen("test_odom",key,0,1)
    elif method=="imlpr":
        import cv2
        cv2.setNumThreads(1)
        p=configure(method)
        p.ROLES["test_odom"] = ([(sensor,seq)],p.ODOM,"val_","test_odom")
        p.cmd_index("test_odom")
        home=p.role_dir("test_odom",sensor,seq)
        track=p.Track("test_odom",sensor,seq,loader) if p.SENSORS[sensor]["accumulate"] else None
        for k,pos in enumerate(row["imlpr_positions"]):
            dest=home/f"{pos:06d}.npy"
            if dest.exists():
                continue
            image,_,_=p.riv_for("test_odom",sensor,seq,pos,int(ids[pos]),loader,track)
            tmp=dest.with_suffix(".tmp.npy"); np.save(tmp,image); tmp.replace(dest)
            if k%100==0:
                write(marker,dict(status="running",completed=k+1,total=len(row["imlpr_positions"]),seconds=time.time()-tick))
                print("INPUT",method,key,k+1,len(row["imlpr_positions"]),time.time()-tick,flush=True)
    else:
        raise ValueError(method)
    write(marker,dict(status="complete",seconds=time.time()-tick,finished_unix=time.time()))


def run(workers: int) -> None:
    """호스트에서 CPU 컨테이너 작업과 증분 전송을 이어 실행한다."""
    spec=read(OUT/"input_relay_plan.json")
    jobs=[(m,r["sensor"]+"/"+r["seq"]) for m in spec["methods"] for r in spec["rows"]]
    logdir=OUT/"logs"; logdir.mkdir(parents=True,exist_ok=True)
    status=OUT/"input_relay_status.json"
    write(status,dict(status="running",pid=os.getpid(),jobs=len(jobs),workers=workers,started_unix=time.time()))
    def one(job):
        m,key=job
        done=INPUT/"relay_status"/f"{m}_{key.replace('/','_')}.json"
        with (logdir/f"local_input_{m}_{key.replace('/','_')}.log").open("a") as out:
            code=subprocess.call(["docker","exec","-e","CUDA_VISIBLE_DEVICES=","-e","OMP_NUM_THREADS=1",
                "-e","OPENBLAS_NUM_THREADS=1","-e","MKL_NUM_THREADS=1","-w","/workspace/Neural-Spectral-Codec",
                "ring_baseline","python3","-u","scripts/_review_selection_full_relay.py","worker","--method",m,
                "--sequence",key],stdout=out,stderr=subprocess.STDOUT)
        assert code==0,(job,code)
        # 동시 rsync의 파일 탐색 범위를 해당 입력으로 제한한다.
        sensor,seq=key.split("/")
        folder=INPUT/m/("data/test_odom/"+sensor+"_"+seq if m in ("helios","imlpr") else sensor+"_"+seq)
        rel=folder.relative_to(ROOT)
        remote="joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/"+str(rel.parent)+"/"
        subprocess.run(["ssh","joms-gpu","mkdir -p /home/qkrrlans00/Neural-Spectral-Codec/"+str(rel.parent)],check=True)
        subprocess.run(["rsync","-az",str(folder),remote],check=True)
        return dict(method=m,sequence=key,seconds=read(done)["seconds"],synced_unix=time.time())
    completed=[]; failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,j):j for j in jobs}
        for f in concurrent.futures.as_completed(futures):
            try:
                completed.append(f.result())
            except Exception as exc:
                failed.append(dict(job=futures[f],error=str(exc)))
                write(status,dict(status="running_with_failures",failed=failed,completed=completed))
                continue
            write(status,dict(status="running",pid=os.getpid(),jobs=len(jobs),completed=completed,updated_unix=time.time()))
    write(status,dict(status="worker_failed" if failed else "complete",jobs=len(jobs),completed=completed,
                     failed=failed,finished_unix=time.time()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("plan","run","worker"))
    parser.add_argument("--method")
    parser.add_argument("--sequence")
    parser.add_argument("--workers",type=int,default=2)
    a=parser.parse_args()
    if a.action=="plan": plan()
    elif a.action=="run": run(a.workers)
    else: cpu_worker(a.method,a.sequence)
