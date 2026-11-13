#!/usr/bin/env python3
"""C5: 공식 실행 파일(무수정) 체인과 드라이버의 단층 BEV 비트 동치 (서버 호스트, 2026-11-07).

  python3 scripts/fresco_official/verify_prep.py
공식 체인: {kitti,mulran}_point_cloud_select <fixture> 2  ->  batch_multi_bev_gen
(KITTI는 드라이버와 같은 고정 두 줄 회귀 수정 사본으로 만든 kitti_point_cloud_select_groundfix, cpp/kitti_groundfix.py)
<fixture>/selected_keyframes_2.00m <HDL_64E|OS1_64>  ->  output_single_bev/csv/*.csv (공식 CSV).
fixture는 NAS tmp의 새 폴더에만 만든다(공식 바이너리가 출력 폴더를 rm -rf 하기 때문, audit C10).
가짜 자세는 프레임마다 10 m 떨어뜨려 모든 스캔이 키프레임이 되게 한다(기술자는 자세를 쓰지 않음).
공식 체인에는 원래 원시 파일(실제 intensity)을, 드라이버에는 codec.encode의 official bytes
(intensity 0)를 넣어 intensity 대체가 무손실임도 함께 확인한다.
입력: kitti = 서버 data/local_raw/kitti/.../05(학습) 원시, mulran = NAS tmp/c5_input/mulran(검증
Sejong01 원시), mcd = NAS tmp/c5_input/mcd(시험 kth_day_06 원시; 점 순서 전치 확인용, 설정 선택 없음).
nclt·helipr는 공식 실행 파일이 없어 드라이버 실행·통계만 기록한다.
-> NAS build/verify_prep.json, results/.../pipeline/verification_prep.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NAS_ROOT, REPO, RESULTS, write_json  # noqa: E402
import codec  # noqa: E402

B = NAS_ROOT / "build"
FIX = NAS_ROOT / "tmp/c5_fixture"
INP = NAS_ROOT / "tmp/c5_input"
N = 12


def run_driver(drv: str, blobs: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    work = Path(f"/dev/shm/fresco_c5_{drv}")
    work.mkdir(parents=True, exist_ok=True)
    data = b"".join(len(b).to_bytes(4, "little") + b for b in blobs) + (0).to_bytes(4, "little")
    r = subprocess.run([str(B / f"fresco_prep_{drv}"), str(work)], input=data, capture_output=True)
    assert r.returncode == 0, r.stderr[-2000:]
    shutil.rmtree(work)
    rec = np.frombuffer(r.stdout, dtype=np.uint8).reshape(len(blobs), 224 * 224 + 16)
    return rec[:, :224 * 224].reshape(-1, 224, 224), rec[:, 224 * 224:].copy().view("<i4")


def official_chain(kind: str, raws: list[bytes]) -> np.ndarray:
    d = FIX / kind
    if d.exists():
        shutil.rmtree(d)
    if kind == "kitti":
        (d / "velodyne").mkdir(parents=True)
        for i, b in enumerate(raws):
            (d / "velodyne" / f"{i:06d}.bin").write_bytes(b)
        (d / "times.txt").write_text("".join(f"{i}\n" for i in range(len(raws))))
        (d / "global_pose.txt").write_text("".join(
            f"1 0 0 0 0 1 0 0 0 0 1 {10.0 * (i + 1):.1f}\n" for i in range(len(raws))))
        sel, sensor = "kitti_point_cloud_select_groundfix", "HDL_64E"
    else:
        (d / "sensor_data/Ouster").mkdir(parents=True)
        stamps = [1000000000 + 100 * i for i in range(len(raws))]
        for s, b in zip(stamps, raws):
            (d / "sensor_data/Ouster" / f"{s:010d}.bin").write_bytes(b)
        (d / "sensor_data/ouster_front_stamp.csv").write_text("".join(f"{s}\n" for s in stamps))
        poses = stamps + [stamps[-1] + 100]
        (d / "global_pose.csv").write_text("".join(
            f"{s},1,0,0,{10.0 * (i + 1):.1f},0,1,0,0,0,0,1,0\n" for i, s in enumerate(poses)))
        sel, sensor = "mulran_point_cloud_select", "OS1_64"
    log = open(d.parent / f"{kind}_official.log", "w")
    subprocess.run([str(B / "official" / sel), str(d), "2"], check=True, stdout=log, stderr=log)
    kf = d / "selected_keyframes_2.00m"
    subprocess.run([str(B / "official/batch_multi_bev_gen"), str(kf), sensor], check=True,
                   stdout=log, stderr=log)
    csvs = sorted((kf / "output_single_bev/csv").glob("*.csv"))
    assert len(csvs) == len(raws), (kind, len(csvs))
    return np.stack([np.loadtxt(c, delimiter=",").astype(np.int64) for c in csvs])


def main():
    out = {"n_scans_each": N, "cases": {}}
    kitti = sorted((REPO / "data/local_raw/kitti/dataset/sequences/05/velodyne").glob("*.bin"))
    cases = {"kitti": ("kitti_hdl64", [p.read_bytes() for p in kitti[100:100 + 50 * N:50]]),
             "mulran": ("mulran_os1", [p.read_bytes() for p in sorted((INP / "mulran").glob("*.bin"))[:N]]),
             "mcd": ("mcd_os1_transposed", [p.read_bytes() for p in sorted((INP / "mcd").glob("*.bin"))[:N]])}
    for kind, (fam, raws) in cases.items():
        assert len(raws) == N, (kind, len(raws))
        official_in = []
        for r in raws:   # 공식 체인 입력: 원래 intensity를 둔 원시(mcd는 점 순서만 전치)
            if fam == "mcd_os1_transposed":
                a = np.frombuffer(r, "<f4").reshape(64, 1024, 4).transpose(1, 0, 2)
                official_in.append(np.ascontiguousarray(a).tobytes())
            else:
                official_in.append(r)
        ref = official_chain("mulran" if fam != "kitti_hdl64" else "kitti", official_in)
        bev, st = run_driver(codec.DRIVER[fam], [codec.encode(fam, r)[1] for r in raws])
        eq = bool(np.array_equal(ref, bev.astype(np.int64)))
        out["cases"][kind] = {"family": fam, "bit_equal": eq,
                              "n_mismatch_cells": int((ref != bev).sum()),
                              "bev_nonzero_mean": float((bev > 0).sum((1, 2)).mean()),
                              "label0_mean": float(st[:, 2].mean()), "points_mean": float(st[:, 0].mean())}
        print(kind, out["cases"][kind], flush=True)
    for kind, fam in (("nclt", "nclt_oxford_hdl32"), ("helipr", "helipr_vlp16")):
        raws = [p.read_bytes() for p in sorted((INP / kind).glob("*.bin"))[:N]]
        bev, st = run_driver(codec.DRIVER[fam], [codec.encode(fam, r)[1] for r in raws])
        out["cases"][kind] = {"family": fam, "official_executable": None,
                              "bev_nonzero_mean": float((bev > 0).sum((1, 2)).mean()),
                              "label0_mean": float(st[:, 2].mean()), "points_mean": float(st[:, 0].mean()),
                              "height_value_median": float(np.median(bev[bev > 0]))}
        print(kind, out["cases"][kind], flush=True)
    out["passed"] = all(c.get("bit_equal", True) for c in out["cases"].values())
    write_json(B / "verify_prep.json", out)
    write_json(RESULTS / "verification_prep.json", out)
    print("C5_PASSED" if out["passed"] else "C5_FAILED")


if __name__ == "__main__":
    main()
