#!/usr/bin/env python3
"""선택 후 누락 시험 입력을 시퀀스 단위로 복구·전송·정리한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]
from _review_selection_bev import TEST,read,write,sha
from _review_selection_full_inputs import OUT,INPUT,cache


def input_home(method: str, sensor: str, seq: str) -> Path:
    """정리 가능한 이번 실험의 입력 디렉터리만 돌려준다."""
    assert method in ("seqot","imlpr","ringsharp"),method
    root=INPUT/("test_"+method)
    return root/"data/test_odom"/f"{sensor}_{seq}" if method=="imlpr" else root/f"{sensor}_{seq}"


def imlpr_module():
    import _imlpr_official_riv as p
    p.WORK=INPUT/"test_imlpr";p.DATA=p.WORK/"data"
    return p


def request(method: str, sensor: str, seq: str, selection_path: Path) -> bool:
    """선택 동결을 확인한 뒤 누락 입력 복구를 요청한다."""
    assert selection_path.exists()
    home=input_home(method,sensor,seq)
    marker=home/"transfer_complete.json"
    if marker.exists():
        assert read(home/"done.json")["cache_sha256"]==sha(cache(sensor,seq,"test"))
        assert read(marker)["done_sha256"]==sha(home/"done.json")
        return True
    gate_path=OUT/"test_request_gate.json"
    if gate_path.exists():
        gate=read(gate_path)
        if method==gate.get("method") and sensor+"/"+seq not in gate["allowed_sequences"]:
            return False
    path=OUT/"test_input_requests"/f"{method}_{sensor}_{seq}.json"
    if not path.exists():
        write(path,dict(method="test_"+method,sequence=sensor+"/"+seq,status="requested",
            selection_sha256=sha(selection_path),cache_sha256=sha(cache(sensor,seq,"test")),requested_unix=time.time()))
    return False


def remove_consumed_input(home: Path) -> None:
    """동시에 끝난 소비기의 중복 삭제만 허용하고 다른 파일 오류는 전파한다."""
    def on_error(function, path, error):
        if not isinstance(error[1], FileNotFoundError):
            raise error[1]
    try:
        shutil.rmtree(home, onerror=on_error)
    except FileNotFoundError:
        pass


def release(method: str, sensor: str, seq: str) -> None:
    """모든 선택 모델의 해당 시퀀스 원장이 확정된 뒤 임시 입력만 정리한다."""
    if method not in ("seqot","imlpr","ringsharp"): return
    path=OUT/"test_input_requests"/f"{method}_{sensor}_{seq}.json"
    if not path.exists(): return
    record=read(path)
    if record.get("status")=="consumed": return
    home=input_home(method,sensor,seq)
    record.update(status="consumed",consumed_unix=time.time(),done=read(home/"done.json") if (home/"done.json").exists() else None)
    write(path,record)
    remove_consumed_input(home)


class TestSeqSource:
    """원래 range_image로 복구한 공식 가운데 창 입력을 읽는다."""
    def __init__(self,sensor,seq):
        home=input_home("seqot",sensor,seq)
        info=read(home/"done.json")
        assert info["cache_sha256"]==sha(cache(sensor,seq,"test"))
        self.n_raw=info["n_raw"]
        self.raw=np.load(home/"raw_ids.npy")
        self.mm=np.load(home/"range.npy",mmap_mode="r")
        self.lookup={int(r):i for i,r in enumerate(self.raw)}
        z=np.load(cache(sensor,seq,"test"));self.kf=z["scan_ids"];self.kf_poses=z["poses_gt"]

    def exists(self,raw): return 0<=raw<self.n_raw

    def rows(self,raws):
        return self.mm[[self.lookup[int(r)] for r in raws]]


def worker(method: str, key: str) -> None:
    """로컬 원시를 CPU로 전처리한다. 모델이나 시험 지표는 계산하지 않는다."""
    assert not os.environ.get("CUDA_VISIBLE_DEVICES"),"로컬 GPU 금지"
    sensor,seq=key.split("/");home=input_home(method,sensor,seq);home.mkdir(parents=True,exist_ok=True)
    if (home/"done.json").exists(): return
    if method=="ringsharp":
        from _review_selection_full_relay import cpu_worker
        cpu_worker("test_ringsharp",key);return
    from _pd_build_test_caches import _make_loader
    ids=np.load(cache(sensor,seq,"test"))["scan_ids"]
    loader=_make_loader(sensor,seq);tick=time.time()
    if method=="seqot":
        import _seqot_official_prep as p
        raw=p.centered_raw_ids(ids,len(loader));np.save(home/"raw_ids.npy",raw)
        path=home/"range.npy"
        array=np.lib.format.open_memmap(path,mode="r+" if path.exists() else "w+",dtype=np.float32,shape=(len(raw),32,900))
        progress=home/"progress.json";start=read(progress)["completed"] if progress.exists() else 0
        for i in range(start,len(raw)):
            array[i]=p.range_image(sensor,loader[int(raw[i])]["points"])
            if (i+1)%128==0 or i+1==len(raw):
                array.flush();write(progress,dict(completed=i+1,total=len(raw),seconds=time.time()-tick))
        write(home/"done.json",dict(cache_sha256=sha(cache(sensor,seq,"test")),n_raw=len(loader),n_frames=len(ids),
            range_rows=len(raw),range_sha256=sha(path),source="원래 _seqot_official_prep.range_image + centered_raw_ids",seconds=time.time()-tick))
    else:
        import cv2
        cv2.setNumThreads(1)
        p=imlpr_module();p.ROLES["test_odom"]=([(sensor,seq)],p.ODOM,"test_","test_odom")
        p.cmd_index("test_odom")
        from _review_selection_full_riv_parallel import generate
        generate(sensor,seq,home,ids,processes=3)
        write(home/"done.json",dict(cache_sha256=sha(cache(sensor,seq,"test")),n_frames=len(ids),seconds=time.time()-tick,
            source="원래 _imlpr_official_riv.riv_for·KISS과거누적"))


def run(workers: int) -> None:
    """최대3시퀀스만 앞서 준비하고 소비된 임시 입력은 재개 기록과 함께 정리한다."""
    remote_root="/home/qkrrlans00/Neural-Spectral-Codec"
    remote="joms-gpu:"+remote_root+"/"
    requests=OUT/"test_input_requests";requests.mkdir(exist_ok=True)
    status_dir=OUT/"test_relay_status";status_dir.mkdir(exist_ok=True)
    logs=OUT/"logs";logs.mkdir(exist_ok=True)
    order={s+"/"+q:i for i,(s,q) in enumerate(TEST)}
    def execute(method,key):
        sensor,seq=key.split("/");home=input_home(method,sensor,seq);home.mkdir(parents=True,exist_ok=True)
        name=method+"_"+sensor+"_"+seq
        with (logs/f"test_input_{name}.log").open("a") as log:
            if method=="imlpr":
                from _review_selection_full_test_server_riv import execute as server_execute
                server_execute(key,log)
                code=0
            else:
                code=subprocess.call(["docker","exec","-e","CUDA_VISIBLE_DEVICES=","-e","OMP_NUM_THREADS=1",
                "-e","OPENBLAS_NUM_THREADS=1","-e","MKL_NUM_THREADS=1","-w","/workspace/Neural-Spectral-Codec",
                "ring_baseline","python3","-u","scripts/_review_selection_full_test_relay.py","worker","--method",method,
                "--sequence",key],stdout=log,stderr=subprocess.STDOUT)
        assert code==0,(method,key,code)
        hosts=read(OUT/"execution_hosts.json") if (OUT/"execution_hosts.json").exists() else {}
        if hosts.get(method)=="local":
            write(home/"transfer_complete.json",dict(done_sha256=sha(home/"done.json"),transferred_unix=time.time(),host="local"))
            write(status_dir/f"{name}.json",dict(status="synced",host="local",method=method,sequence=key,updated_unix=time.time()))
            return
        rel=home.relative_to(ROOT)
        subprocess.run(["ssh","joms-gpu","mkdir -p "+shlex.quote(remote_root+"/"+str(rel.parent))],check=True)
        subprocess.run(["rsync","-az",str(home),remote+str(rel.parent)+"/"],check=True)
        marker=dict(done_sha256=sha(home/"done.json"),transferred_unix=time.time())
        local_marker=status_dir/f"{name}_transfer.json";write(local_marker,marker)
        subprocess.run(["rsync","-a",str(local_marker),remote+str(rel/"transfer_complete.json")],check=True)
        write(status_dir/f"{name}.json",dict(status="synced",method=method,sequence=key,updated_unix=time.time()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        active={}
        while True:
            subprocess.run(["rsync","-a","--ignore-missing-args",remote+str(requests.relative_to(ROOT))+"/",str(requests)+"/"],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
            for future,name in list(active.items()):
                if not future.done():continue
                try:future.result()
                except Exception as exc:write(status_dir/f"{name}.json",dict(status="failed",error=str(exc),updated_unix=time.time()))
                del active[future]
            entries=sorted([read(f) for f in requests.glob("*.json")],key=lambda r:(order[r["sequence"]],r["method"]))
            owners_path=OUT/"test_external_input_owners.json"
            owners=read(owners_path) if owners_path.exists() else {}
            # 앞선 입력이 실패했을 때 뒤 입력 세 개로 슬롯을 채워 재시도를
            # 영구적으로 막지 않는다. 원래 시험 순서의 다음 세 입력만 준비한다.
            unconsumed=[r for r in entries if r.get("status")!="consumed"]
            allowed_names={r["method"].removeprefix("test_")+"_"+r["sequence"].replace("/","_")
                           for r in unconsumed[:3]}
            control_path=OUT/"test_relay_control.json"
            hold_new=read(control_path).get("hold_new",False) if control_path.exists() else False
            pending=[];outstanding=0
            for info in entries:
                method=info["method"].removeprefix("test_");key=info["sequence"];sensor,seq=key.split("/")
                name=method+"_"+sensor+"_"+seq;path=status_dir/f"{name}.json"
                state=read(path) if path.exists() else {}
                if info.get("status")=="consumed":
                    if state.get("status")!="cleaned" and name not in active.values():
                        if method=="imlpr":
                            from _review_selection_full_test_server_riv import clean as server_clean
                            server_clean(key)
                        subprocess.run(["docker","exec","-e","CUDA_VISIBLE_DEVICES=","-w","/workspace/Neural-Spectral-Codec",
                            "ring_baseline","python3","scripts/_review_selection_full_test_relay.py","clean","--method",method,"--sequence",key],check=True)
                        write(path,dict(status="cleaned",updated_unix=time.time()))
                    continue
                if state.get("status")=="synced" or name in active.values():outstanding+=1;continue
                if name in owners and owners[name].get("status")!="complete":
                    outstanding+=1;continue
                if state.get("status")=="failed" and time.time()-state.get("updated_unix",0)<120:continue
                pending.append((method,key,name))
            for method,key,name in pending:
                if hold_new:break
                if name not in allowed_names:continue
                if len(active)>=workers or outstanding>=3:break
                active[pool.submit(execute,method,key)]=name;outstanding+=1
            write(OUT/"test_relay_status.json",dict(status="running",pid=os.getpid(),active=list(active.values()),
                outstanding=outstanding,pending=len(pending),workers=workers,
                hold_new=hold_new,allowed_next_names=sorted(allowed_names),updated_unix=time.time()))
            time.sleep(10)


if __name__=="__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("worker","run","clean"));parser.add_argument("--method")
    parser.add_argument("--sequence");parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.action=="run":run(args.workers)
    elif args.action=="worker":worker(args.method,args.sequence)
    else:
        sensor,seq=args.sequence.split("/");home=input_home(args.method,sensor,seq)
        remove_consumed_input(home)
