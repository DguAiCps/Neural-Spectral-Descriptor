#!/usr/bin/env python3
"""동결 후 ImLPR 시험 입력을 서버 CPU로 복구하고 소비 뒤 임시본을 정리한다."""
from pathlib import Path
import subprocess
import time

from _review_selection_bev import read,write,sha,TEST
from _review_selection_full_inputs import OUT,INPUT
from _review_selection_full_rawpack import home_for
from _review_selection_restart_server_inputs import copy_to_server,REMOTE,REMOTE_ROOT,REL,CONTAINER_ROOT


def execute(key: str, log) -> None:
    """두 활성시퀀스 각각12CPU를 사용해 같은 원래 RIV를 계산한다."""
    sensor,seq=key.split("/")
    assert (sensor,seq) in TEST
    selection=OUT/"retained/selection_imlpr.json"
    assert selection.exists()
    assert read(OUT/"server_riv_smoke/validated.json")["all_arrays_exact"]
    subprocess.run(["docker","exec","-e","CUDA_VISIBLE_DEVICES=","-e","OMP_NUM_THREADS=1","-e","OPENBLAS_NUM_THREADS=1",
                    "-e","MKL_NUM_THREADS=1","-w",CONTAINER_ROOT,"ring_baseline","python3","-u",
                    "scripts/_review_selection_full_rawpack.py","--role","test","--sequence",key],stdout=log,stderr=subprocess.STDOUT,check=True)
    raw=home_for("test",sensor,seq)
    info=read(raw/"done.json")
    for filename, field in (("offsets.npy", "offsets_sha256"), ("raw_ids.npy", "raw_ids_sha256")):
        assert sha(raw/filename)==info[field], (key, "local rawpack metadata mismatch before transfer")
    # 동시에 최대 두 시퀀스를 준비하므로 가장 큰 현재 작업 두 개의 여유를 둔다.
    # Hf<=126, W=1024, float32 3채널 + NPY 헤더의 파일 상한이다.
    required=int(info["points_bytes"]*1.01)+int(info["n_keyframes"])*1_550_000
    free=int(subprocess.check_output(["ssh","joms-gpu","df","--output=avail","-B1",REMOTE_ROOT],text=True).splitlines()[-1])
    write(OUT/"test_disk_preflight"/f"imlpr_{sensor}_{seq}.json",dict(checked_unix=time.time(),
          rawpack_bytes=info["points_bytes"],required_one_sequence_bytes=required,server_available_bytes=free,
          two_active_upper_bytes=2*required+15_000_000_000))
    assert free>2*required+15_000_000_000,(key,"server disk reserve",free,required)
    compression_proof=OUT/"transfer_compression_verified.json"
    assert read(compression_proof)["all_roundtrip_file_SHA_exact"]
    destination=REL/"inputs/rawpack/test"
    subprocess.run(["ssh","joms-gpu","mkdir","-p",REMOTE_ROOT+"/"+str(destination)],check=True)
    subprocess.run(["rsync","-aO","--no-owner","--no-group","--no-perms","--compress",
                    "--compress-choice=zstd","--compress-level=1",str(raw),REMOTE+str(destination)+"/"],check=True)
    subprocess.run(["ssh","joms-gpu","docker","exec","-e","CUDA_VISIBLE_DEVICES=","-e","OMP_NUM_THREADS=1",
                    "-e","OPENBLAS_NUM_THREADS=1","-e","MKL_NUM_THREADS=1","-w",CONTAINER_ROOT,"ring_baseline",
                    "python3","-u","scripts/_review_selection_full_server_riv.py","--role","test","--sequence",key,
                    "--processes","12"],stdout=log,stderr=subprocess.STDOUT,check=True)
    home=INPUT/"test_imlpr/data/test_odom"/f"{sensor}_{seq}"
    home.mkdir(parents=True,exist_ok=True)
    relative=REL/"inputs/test_imlpr/data/test_odom"/f"{sensor}_{seq}"
    subprocess.run(["rsync","-aO","--no-owner","--no-group","--no-perms","--exclude=.*","--exclude=*.tmp","--compress","--compress-choice=zstd","--compress-level=1",
                    REMOTE+str(relative)+"/",str(home)+"/"],check=True)
    hashes=read(home/"server_generated_hashes.json")
    assert len(hashes)==read(home/"done.json")["n_frames"]
    assert sha(home/"server_generated_hashes.json")==read(home/"done.json")["generated_hash_manifest_sha256"]
    for name,digest in hashes.items():
        assert sha(home/name)==digest,(key,name,"download SHA mismatch")
    write(OUT/"test_input_provenance"/f"imlpr_{sensor}_{seq}.json",dict(completed_unix=time.time(),
          selection_sha256=sha(selection),rawpack=read(raw/"done.json"),rawpack_verification=read(raw/"verified.json"),
          original_RIV_equivalence_proof_sha256=sha(OUT/"server_riv_smoke/validated.json"),done=read(home/"done.json"),
          downloaded_RIV_all_files_SHA_exact=True,generated_RIV_file_sha256=hashes,
          transfer_compression_proof_sha256=sha(compression_proof)))


def clean(key: str) -> None:
    """완성 원장 뒤에만 원래 데이터와 분리된 이번 시험 임시 입력을 정리한다."""
    sensor,seq=key.split("/");assert (sensor,seq) in TEST
    request=read(OUT/"test_input_requests"/f"imlpr_{sensor}_{seq}.json")
    assert request["status"]=="consumed"
    paths=[str(REL/"inputs/test_imlpr/data/test_odom"/f"{sensor}_{seq}"),str(REL/"inputs/rawpack/test"/f"{sensor}_{seq}")]
    code="from pathlib import Path; import shutil,sys; root=Path('/workspace/Neural-Spectral-Codec/results/pdv2_review_selection_full_20261105/inputs'); paths=[Path('/workspace/Neural-Spectral-Codec')/s for s in sys.argv[1:]]; assert all(p.is_relative_to(root) for p in paths); [shutil.rmtree(p) for p in paths if p.exists()]"
    subprocess.run(["ssh","joms-gpu","docker","exec","-e","CUDA_VISIBLE_DEVICES=","ring_baseline","python3","-c",
                    __import__('shlex').quote(code),*paths],check=True)
    raw=home_for("test",sensor,seq)
    subprocess.run(["docker","exec","-e","CUDA_VISIBLE_DEVICES=","ring_baseline","python3","-c",
                    "import shutil,sys; from pathlib import Path; p=Path(sys.argv[1]); shutil.rmtree(p) if p.exists() else None",
                    CONTAINER_ROOT+"/"+str(raw.relative_to(OUT.parents[1]))],check=True)
