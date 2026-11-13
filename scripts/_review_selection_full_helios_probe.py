#!/usr/bin/env python3
"""HeLiOS 공식 단일 스캔 추론의 동시 실행 동일성과 처리량을 확인한다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]
from _review_selection_bev import read,write,sha
from _review_selection_full_inputs import OUT,configure


def worker(tag: str, sequence: str) -> None:
    import torch
    torch.set_num_threads(4)
    sensor,seq=sequence.split("/"); p=configure("helios")
    import _helios_official_dump as d
    home=OUT/"helios_concurrency_probe"; home.mkdir(parents=True,exist_ok=True)
    cfg=home/f"config_{tag}.txt"
    E,params=d._official(cfg)
    row=read(OUT/"retained_protocol.json")["methods"]["helios"]["1"]["candidates"][0]
    assert sha(ROOT/row["path"])==row["sha256"]
    model=d.official_model(E,params,ROOT/row["path"],"cuda")
    ids=np.load(p.seq_dir("test_odom",sensor,seq)/"index.npz")["scan_ids"]
    chosen=np.unique(np.linspace(0,len(ids)-1,128).round().astype(int))
    dataset={i:{"query":p.rel_scan_path("test_odom",sensor,seq,int(ids[k]))} for i,k in enumerate(chosen)}
    start=time.time(); desc=E.get_latent_vectors(model,dataset,"cuda",params)
    np.save(home/f"{tag}.npy",desc)
    write(home/f"{tag}.json",dict(seconds=time.time()-start,n=len(desc),sequence=sequence,
        cuda_peak_gb=torch.cuda.max_memory_allocated()/2**30,ids=ids[chosen].tolist(),checkpoint_sha256=row["sha256"]))


def run() -> None:
    home=OUT/"helios_concurrency_probe"; home.mkdir(parents=True,exist_ok=True)
    def launch(tag):
        log=(home/f"{tag}.log").open("w")
        return subprocess.Popen([sys.executable,"-u",__file__,"worker","--tag",tag,"--sequence","kitti/09"],
                                stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1"))
    t=time.time(); p=launch("single"); assert p.wait()==0
    single_wall=time.time()-t
    t=time.time(); jobs=[launch(f"parallel{i}") for i in range(4)]
    assert all(p.wait()==0 for p in jobs)
    parallel_wall=time.time()-t
    ref=np.load(home/"single.npy")
    checks=[]
    for i in range(4):
        arr=np.load(home/f"parallel{i}.npy")
        checks.append(dict(tag=f"parallel{i}",exact=bool(np.array_equal(arr,ref)),
                           max_abs_error=float(np.max(np.abs(arr-ref)))))
    write(home/"report.json",dict(single_wall_s=single_wall,four_parallel_wall_s=parallel_wall,
        effective_speedup=4*single_wall/parallel_wall,checks=checks,
        all_exact=all(c["exact"] for c in checks),training_shared_gpu=True,
        scope="동일 모델·공식 batch1 경로의 동시성 검산; 논문의 효율benchmark 아님"))
    print(read(home/"report.json"),flush=True)


def pair() -> None:
    """초기화를 분산한 두 프로세스의 공식 추론을 반복 검산한다."""
    home=OUT/"helios_concurrency_probe"
    jobs=[]; tick=time.time()
    for i in range(2):
        log=(home/f"pair{i}.log").open("w")
        jobs.append(subprocess.Popen([sys.executable,"-u",__file__,"worker","--tag",f"pair{i}","--sequence","kitti/09"],
            stdout=log,stderr=subprocess.STDOUT))
        if i==0: time.sleep(5)
    codes=[p.wait() for p in jobs]
    write(home/"pair_status.json",dict(codes=codes,wall_s=time.time()-tick))
    assert codes==[0,0],codes


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("action",choices=("run","worker","pair"))
    p.add_argument("--tag"); p.add_argument("--sequence")
    a=p.parse_args()
    if a.action=="run": run()
    elif a.action=="pair": pair()
    else: worker(a.tag,a.sequence)
