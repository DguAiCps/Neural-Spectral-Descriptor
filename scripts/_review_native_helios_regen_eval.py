#!/usr/bin/env python3
"""재생성 Velodyne 세션의 HeLiOS 추론·전달·독립 검산을 별도로 수행한다."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _review_native_helios as native
from _review_native_learned_verify import independent_pair

RELEASED = native.OUT
OUT = RELEASED.parent / "helios_regenerated_velodyne"
MODELS = ["author", "v2_s1", "v2_s2", "v2_s3"]
SEQUENCES = [f"Roundabout{i:02d}-Velodyne" for i in (1, 2, 3)]


def link(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_symlink():
        assert not target.exists(), target
        target.symlink_to(os.path.relpath(source, target.parent))


def pack() -> None:
    """완료된 재생성 파일과 고정된 공개 지도만 별도 입력에 연결한다."""
    for name in ["protocol.json"]:
        shutil.copy2(RELEASED / name, OUT / name)
    (OUT / "data").mkdir(exist_ok=True)
    for name in ["database.pickle", "query.pickle"]:
        shutil.copy2(RELEASED / "data" / name, OUT / "data" / name)
    files = {}
    for seq in SEQUENCES:
        root = OUT / "prepared" / seq
        done = json.loads((root / "done.json").read_text())
        for row in done["files"]:
            filename = row["timestamp"] + ".bin"
            paths = list(root.glob("chunk_*/out/" + filename))
            assert len(paths) == 1 and native.sha(paths[0]) == row["sha256"]
            target = OUT / "data/validation" / seq / "LiDAR" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(paths[0], target)
            assert native.sha(target) == row["sha256"]
            files[str(target.relative_to(OUT))] = row["sha256"]
    for db in native.DATABASES:
        link(OUT / "data/validation" / db, RELEASED / "data/validation" / db)
        for model in MODELS:
            link(OUT / model / "desc" / db, RELEASED / model / "desc" / db)
    owner = OUT.stat()
    for model in MODELS:
        for directory in [OUT / model, OUT / model / "desc"]:
            os.chown(directory, owner.st_uid, owner.st_gid)
    native.write(OUT / "packed_manifest.json", {"files": files, "query_source": "공식 toolbox로 원시 재생성", "database_source": "저자 공개 가공파일·기존 추론 그대로", "regeneration_protocol_sha256": native.sha(OUT / "regeneration_protocol.json")})


def transfer() -> None:
    """입력 완료 후 전송·전수 해시 검증을 끝내고 서버 평가기를 시작한다."""
    while not all((OUT / "prepared" / seq / "done.json").exists() for seq in SEQUENCES):
        native.write(OUT / "relay_status.json", {"waiting_sequences": [seq for seq in SEQUENCES if not (OUT / "prepared" / seq / "done.json").exists()], "pid": os.getpid(), "unix_time": time.time()})
        time.sleep(30)
    subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "scripts/_review_native_helios_regen_eval.py", "pack"], check=True)
    remote = "/home/qkrrlans00/Neural-Spectral-Codec/"
    relative = str(OUT.relative_to(REPO))
    subprocess.run(["ssh", "joms-gpu", "mkdir", "-p", remote + relative], check=True)
    subprocess.run(["rsync", "-az", "--omit-dir-times", "--exclude=prepared/", "--exclude=smoke/", str(OUT) + "/", "joms-gpu:" + remote + relative + "/"], check=True)
    subprocess.run(["rsync", "-az", str(REPO / "scripts/_review_native_helios_regen_eval.py"), str(REPO / "scripts/_review_native_learned_verify.py"), "joms-gpu:" + remote + "scripts/"], check=True)
    command = "cd " + remote + " && nohup python3 -u scripts/_review_native_helios_regen_eval.py runner > " + relative + "/runner.log 2>&1 < /dev/null &"
    subprocess.run(["ssh", "joms-gpu", command], check=True)
    native.write(OUT / "relay_status.json", {"transferred": True, "unix_time": time.time(), "next": "서버가전수해시확인후각도대조완료를기다려추론"})


def runner() -> None:
    """앞선 native 진단이 끝난 뒤 GPU1에서 재개 가능한 추가 추론을 한다."""
    pidfile = OUT / "runner.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("이미 살아 있는 재생성 평가기가 있다")
    pidfile.write_text(str(os.getpid()))
    for name, expected in json.loads((OUT / "packed_manifest.json").read_text())["files"].items():
        assert native.sha(OUT / name) == expected, name
    while True:
        path = OUT.parent / "ring_angles_followup_status.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        entropy_path = OUT.parent / "ring_entropy_formula/runner_status.json"
        entropy = json.loads(entropy_path.read_text()) if entropy_path.exists() else {}
        entropy_ready = not entropy or entropy.get("complete") or entropy.get("stage") == "failed"
        if state.get("complete") and entropy_ready:
            break
        native.write(OUT / "runner_status.json", {"waiting": "기존 학습형 native와 RING 단일요인 대조 완료", "pid": os.getpid(), "unix_time": time.time()})
        time.sleep(30)
    for model in MODELS:
        for seq in SEQUENCES:
            if (OUT / model / "desc" / seq / "done.json").exists():
                continue
            native.write(OUT / "runner_status.json", {"active": [model, seq], "pid": os.getpid(), "unix_time": time.time()})
            cmd = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=1", "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2", "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "nice", "-n", "15", "python3", "-u", "scripts/_review_native_helios_regen_eval.py", "infer", "--model", model, "--sequence", seq]
            with (OUT / f"infer_{model}_{seq}.log").open("a") as stream:
                subprocess.run(cmd, stdout=stream, stderr=stream, check=True)
        with (OUT / f"score_{model}.log").open("a") as stream:
            subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2", "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "-u", "scripts/_review_native_helios_regen_eval.py", "score", "--model", model], stdout=stream, stderr=stream, check=True)
    native.write(OUT / "runner_status.json", {"complete": True, "pid": os.getpid(), "unix_time": time.time()})


def verify() -> None:
    """공식 평가함수 없이 저장된 예측과 정답에서 정수 hit를 다시 센다."""
    queries = pickle.loads((OUT / "data/query.pickle").read_bytes())
    databases = pickle.loads((OUT / "data/database.pickle").read_bytes())
    rows = []
    for model in MODELS:
        path = OUT / model / "native_partial_report.json"
        if not path.exists():
            continue
        for pair in json.loads(path.read_text())["pairs"]:
            db, name = pair["database"], pair["query"]
            p = OUT / model / "native_pairs" / f"{db}__{name}.npz"
            assert native.sha(p) == pair["prediction_sha256"]
            z = np.load(p)
            key = native.DATABASES.index(db)
            valid, good, hits, cells, fixed = independent_pair(z["indices"], z["distances"], queries[native.NAMES.index(name)], databases[key], key, db == name, 30, False)
            np.testing.assert_array_equal(valid, z["evaluated"])
            np.testing.assert_array_equal(good, z["first_correct"])
            np.testing.assert_array_equal(hits, z["recall_hits"])
            np.testing.assert_array_equal(cells, z["tp_fp_tn_fn"])
            assert hits[0] / valid.sum() == pair["r1"]
            rows.append({"model": model, "database": db, "query": name, "passed": True, "r1": pair["r1"], "fixed_positive": fixed})
    native.write(OUT / "independent_verification.json", {"rows": rows, "scope": "별도 재생성 원장만; 원공개입력/전체TableI로대체안함"})
    print(len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pack", "transfer", "runner", "infer", "score", "verify"])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--sequence", choices=SEQUENCES)
    args = parser.parse_args()
    if args.stage in ["infer", "score"]:
        native.OUT = OUT
        if args.stage == "infer":
            native.infer(args.model, args.sequence, False)
        else:
            native.score(args.model)
    else:
        {"pack": pack, "transfer": transfer, "runner": runner, "verify": verify}[args.stage]()
