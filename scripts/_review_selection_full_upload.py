#!/usr/bin/env python3
"""완성 원장의 서버 백업을 GPU 작업열과 분리해 주기적으로 이어간다."""
import fcntl
import os
from pathlib import Path
import subprocess
import time

from _review_selection_bev import read,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/pdv2_review_selection_full_20261105"
REMOTE="joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/"


def main():
    lock=(OUT/"local_upload.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=OUT/"local_upload_status.json"
    while True:
        tick=time.time()
        hosts=read(OUT/"execution_hosts.json")
        methods=[m for m in ("seqot","imlpr","disco") if hosts.get(m)=="local"]
        metadata=[OUT/"local_gpu_state.json"]
        for method in methods:
            metadata.extend(p for n in ("selection","validation","report") if (p:=OUT/"retained"/f"{n}_{method}.json").exists())
        # 원장 목록보다 먼저 상태의 바이트를 고정한다. 전송 도중 진전된 cursor를
        # 옛 원장 목록과 함께 백업하면 재부팅 때 원장을 건너뛸 수 있다.
        snapshot=OUT/"upload_metadata_snapshot"
        for path in metadata:
            copied=snapshot/path.relative_to(ROOT)
            copied.parent.mkdir(parents=True,exist_ok=True)
            copied.write_bytes(path.read_bytes())
        files=[]
        for role in ("val","test"):
            for method in methods:
                home=OUT/"retained"/role/method
                files.extend(p for p in home.rglob("*") if p.is_file() and (p.suffix==".json" or p.name.startswith("dump_") and p.suffix==".npz"))
        # 디렉터리 순회 중 새 완료 JSON이 생겨도 연결된 원장을 반드시 같이 보낸다.
        for path in tuple(files):
            if path.suffix!=".json":continue
            record=read(path)
            if "ledger" in record and "ledger_sha256" in record:
                ledger=ROOT/record["ledger"]
                assert ledger.is_file(), ledger
                files.append(ledger)
        files=list(dict.fromkeys(files))
        # 임시 입력을 정리해도 원본과의 동일성 근거는 양쪽에 보존한다.
        for directory in ("test_input_provenance","rawpack_repairs"):
            files.extend(p for p in (OUT/directory).rglob("*") if p.is_file() and p.suffix in (".json",".npy"))
        files.extend(p for pattern in ("first_test_independent_verification_*.json","validation_independent_verification_*.json")
                     for p in OUT.glob(pattern))
        write(status,dict(status="syncing",pid=os.getpid(),methods=methods,files=len(files),updated_unix=time.time()))
        failures=[]
        for label,paths in (("ledgers",files),("metadata",metadata)):
            listing=OUT/f"async_upload_{label}.txt"
            listing.write_text("\n".join(str(p.relative_to(ROOT)) for p in paths)+"\n")
            source=ROOT if label=="ledgers" else snapshot
            code=subprocess.call(["rsync","-aO","--no-owner","--no-group","--no-perms","--files-from="+str(listing),str(source)+"/",REMOTE])
            if code:
                failures.append(dict(phase=label,returncode=code));break
        write(status,dict(status="retry_pending" if failures else "synced",pid=os.getpid(),methods=methods,files=len(files),
                          failures=failures,elapsed_s=time.time()-tick,updated_unix=time.time()))
        time.sleep(20 if failures else 30)


if __name__=="__main__":
    from _review_selection_local_reservation import require_background_available
    require_background_available()
    main()
