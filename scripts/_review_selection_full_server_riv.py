#!/usr/bin/env python3
"""무손실 원시 묶음에서 원래 RIV를 서버 CPU로 독립 계산한다."""
from __future__ import annotations

import argparse
import fcntl
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]
from _review_selection_bev import read, write, sha
from _review_selection_full_inputs import OUT, INPUT, cache, configure
from _review_selection_full_rawpack import PackedLoader, home_for

_CTX = None


def initialize(role: str, sensor: str, seq: str, target: str) -> None:
    """GPU 없이 원래 KISS 누적 규칙과 같은 raw 배열을 연다."""
    global _CTX
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    import cv2
    cv2.setNumThreads(1)
    import _imlpr_official_riv as p
    loader = PackedLoader(home_for(role, sensor, seq))
    track = p.Track("test_odom", sensor, seq, loader) if p.SENSORS[sensor]["accumulate"] else None
    _CTX = p, sensor, seq, Path(target), loader, track


def calculate(rows: list) -> dict:
    """키프레임별 원래 계산을 수행하고 완성 파일만 원자적으로 공개한다."""
    p, sensor, seq, home, loader, track = _CTX
    tick = time.time()
    for pos, sid in rows:
        path = home / f"{pos:06d}.npy"
        if path.exists():
            continue
        image, _, _ = p.riv_for("test_odom",sensor,seq,pos,sid,loader,track)
        temp = home / f".{pos:06d}.{os.getpid()}.tmp.npy"
        np.save(temp,image);temp.replace(path)
    return dict(positions=[p for p,_ in rows],seconds=time.time()-tick,pid=os.getpid())


def run(role: str, sensor: str, seq: str, processes: int, smoke: bool) -> None:
    """입력 SHA를 검증한 뒤 smoke 또는 인계된 전체 입력을 처리한다."""
    locks=OUT/"server_riv_locks";locks.mkdir(exist_ok=True)
    lock=(locks/f"{role}_{sensor}_{seq}.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX)
    packed = home_for(role,sensor,seq)
    info = read(packed/"done.json")
    proof = read(packed/"verified.json")
    assert proof["all_arrays_exact"] and proof["done_sha256"]==sha(packed/"done.json")
    assert info["cache_sha256"]==sha(cache(sensor,seq,role))
    for name,key in (("points.bin","points_sha256"),("raw_ids.npy","raw_ids_sha256"),("offsets.npy","offsets_sha256")):
        assert sha(packed/name)==info[key],name
    ids = np.load(cache(sensor,seq,role))["scan_ids"]
    if smoke:
        home = OUT/"server_riv_smoke"/f"{sensor}_{seq}";home.mkdir(parents=True,exist_ok=True)
        positions = [99,100,101]
    else:
        assert read(OUT/"server_riv_smoke/validated.json")["all_arrays_exact"]
        if role=="val":
            handoff=read(OUT/"val_riv_server_handoff.json")
            assert handoff["local_workers_stopped"]
            p=configure("imlpr");p.ROLES["test_odom"]=([(sensor,seq)],p.ODOM,"val_","test_odom")
        else:
            from _review_selection_full_test_relay import imlpr_module
            p=imlpr_module();p.ROLES["test_odom"]=([(sensor,seq)],p.ODOM,"test_","test_odom")
        p.cmd_index("test_odom")
        home=p.role_dir("test_odom",sensor,seq)
        generated_path=home/"server_generated_hashes.json"
        generated=read(generated_path) if generated_path.exists() else {}
        preserved=(read(OUT/"val_riv_handoff.json")["completed_files"] if role=="val" else {})
        for path in home.glob("[0-9]*.npy"):
            if not path.stem.isdigit():continue
            expected=generated.get(path.name,preserved.get(str(path.relative_to(ROOT))))
            if expected is None or sha(path)!=expected:
                path.rename(path.with_name(".unverified_reboot_"+path.name))
        positions=[i for i in range(len(ids)) if not (home/f"{i:06d}.npy").exists()]
    rows=[(i,int(ids[i])) for i in positions]
    width=1 if smoke else 16
    blocks=[rows[i:i+width] for i in range(0,len(rows),width)]
    tick=time.time();completed=len(ids)-len(rows) if not smoke else 0
    with multiprocessing.get_context("spawn").Pool(processes,initializer=initialize,initargs=(role,sensor,seq,str(home))) as pool:
        for result in pool.imap_unordered(calculate,blocks):
            completed+=len(result["positions"])
            if not smoke:
                generated.update({f"{i:06d}.npy":sha(home/f"{i:06d}.npy") for i in result["positions"]})
                write(generated_path,generated)
            write(home/"server_progress.json",dict(completed=completed,total=len(ids),seconds=time.time()-tick,
                                                   processes=processes,last_block=result))
            print("SERVER_RIV",role,sensor,seq,completed,len(ids),round(time.time()-tick,1),flush=True)
    if smoke:
        cases=[]
        for i in positions:
            reference=OUT/"server_riv_smoke/reference"/f"{i:06d}.npy"
            np.testing.assert_array_equal(np.load(home/f"{i:06d}.npy"),np.load(reference))
            cases.append(dict(position=i,actual_sha256=sha(home/f"{i:06d}.npy"),reference_sha256=sha(reference)))
        write(OUT/"server_riv_smoke/validated.json",dict(all_arrays_exact=True,cases=cases,
              implementation_sha256=sha(Path(__file__)),rawpack_sha256=sha(ROOT/"scripts/_review_selection_full_rawpack.py")))
    else:
        write(home/"done.json",dict(cache_sha256=sha(cache(sensor,seq,role)),n_frames=len(ids),seconds=time.time()-tick,
                                    source="original riv_for from bit-exact raw loader pack on CPU",
                                    execution_host=__import__('socket').gethostname(),
                                    generated_hash_manifest_sha256=sha(generated_path)))
        if role=="val":
            write(INPUT/"relay_status"/f"imlpr_{sensor}_{seq}.json",dict(status="complete",seconds=time.time()-tick,finished_unix=time.time()))


if __name__=="__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role",choices=("val","test"),required=True)
    parser.add_argument("--sequence",required=True)
    parser.add_argument("--processes",type=int,default=24)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();sensor,seq=args.sequence.split("/")
    run(args.role,sensor,seq,args.processes,args.smoke)
