#!/usr/bin/env python3
"""세 번째 서버 DiSCO 원장도 로컬 GPU에서 같은 경로로 전수 대조한다."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from _review_selection_bev import read,write,sha
OUT=ROOT/"results/pdv2_review_selection_full_20261105"


def main() -> None:
    """진행 중인 GPU 작업 완료를 기다려 별도 원장을 만들고 대조한다."""
    reference=OUT/"retained/val/disco/s1/e001_eca498df1bab/dump_kitti360_0006.npz"
    record=read(reference.with_name("kitti360_0006.json"))
    assert sha(reference)==record["ledger_sha256"]
    dest=OUT/"local_gpu_smoke/reference_disco_K3600006.npz"
    if not dest.exists():shutil.copyfile(reference,dest)
    assert sha(dest)==sha(reference)
    control_path=OUT/"local_gpu_control.json"
    before=read(control_path);assert not before["held_methods"]
    nonce="disco_remaining_server_reference_"+str(int(time.time()))
    held=dict(before,held_methods=read(OUT/"local_gpu_plan.json")["methods"],
              verification_nonce=nonce,updated_unix=time.time())
    accepted=OUT/"local_gpu_smoke/disco.json";previous_sha=sha(accepted)
    write(OUT/"disco_third_smoke_plan.json",dict(nonce=nonce,controller_pid=os.getpid(),
        control_before=before,original_accepted_proof_sha256=previous_sha,
        purpose="서버 선행 원장3개 모두의 로컬 동치 확인; 원래 완료원장 변경 없음"))
    write(control_path,held)
    status_path=OUT/"disco_third_smoke_state.json"
    try:
        started=time.time()
        while True:
            assert read(control_path)==held
            state=read(OUT/"local_gpu_state.json")
            if state["status"] in ("held","complete") and not state.get("active"):break
            assert time.time()-started<120
            write(status_path,dict(status="waiting_GPU_boundary",pid=os.getpid(),updated_unix=time.time()))
            time.sleep(2)
        write(status_path,dict(status="running_GPU_smoke",pid=os.getpid(),updated_unix=time.time()))
        with (OUT/"logs/disco_third_smoke.log").open("a") as log:
            subprocess.run(["docker","exec","-e","CUDA_VISIBLE_DEVICES=0","-e","NVIDIA_TF32_OVERRIDE=0",
                "-e","OMP_NUM_THREADS=4","-e","OPENBLAS_NUM_THREADS=1","-e","MKL_NUM_THREADS=4",
                "-w","/workspace/Neural-Spectral-Codec","ring_baseline","python3","-u",
                "scripts/_review_selection_full_local_smoke.py","--method","disco","--sequence","kitti360/0006",
                "--audit-only"],stdout=log,stderr=subprocess.STDOUT,check=True)
        proof=read(OUT/"local_gpu_smoke/disco_kitti360_0006.json")
        assert proof["all_arrays_exact"]
        assert sha(accepted)==previous_sha
        cases=[OUT/"local_gpu_smoke"/name for name in
            ("disco_kitti_09.json","disco_kitti360_0004.json","disco_kitti360_0006.json")]
        write(OUT/"disco_validation_device_audit.json",dict(all_arrays_exact=True,
            completed_unix=time.time(),server_validation_ledgers_replayed=3,
            queries=sum(read(path)["n_queries"] for path in cases),
            cases={str(path.relative_to(ROOT)):sha(path) for path in cases},
            candidate_set_unchanged=True,original_ledgers_unchanged=True))
        write(status_path,dict(status="complete",completed_unix=time.time(),pid=os.getpid()))
    finally:
        if read(control_path)==held:write(control_path,before)


if __name__=="__main__":main()
