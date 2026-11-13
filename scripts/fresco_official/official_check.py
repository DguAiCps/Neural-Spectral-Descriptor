#!/usr/bin/env python3
"""C1~C3: 공식 createDescsAndRetrieve.m 무수정 실행과 어댑터의 기술자·승자·점수 대조 (서버 호스트).

  python3 scripts/fresco_official/official_check.py --seq kitti/09 [--n 0]
검증 시퀀스의 드라이버 BEV를 공식 입력 배치(output_single_bev/csv/*.csv, keyframe_pose.csv)로
NAS tmp에 쓰고, 공식 함수를 원래 인자 규약 그대로 MATLAB -batch로 부른다(test4KITTI08.m과 같은
best_threshold [70; 7e-4], 캐시 사용 안 함). 공식 함수가 저장하는 cell_keys.mat·cell_frescos.mat와
pr_result/*.mat(50개 임계의 TP/FP/TN/FN)을 이진으로 내보낸다.
대조: (C1) keys 40D·fresco 20x120이 어댑터 값과 비트 동일, (C2) 어댑터를 원 조건 간격 100으로
돌린 승자·L1·cos_row로 공식 PR 산식(10 m, 2D, 이력 1..max(1,q-100))을 다시 세어 공식 카운트와
50개 임계 모두 일치, (C3) 공식 산식의 수락 규칙과 스칼라 점수 -max(L1, 1e5 cos_row)의 순서 동치.
원래 waitbar는 -nodisplay -batch에서 유사도 단계에 멈춰(2026-11-07 시도 1, 공식 rc -15로 기록) 진행 표시줄·궤적
그림 함수만 matlab_gui_shim/(보이지 않는 figure, 그리지 않음)으로 대체한다. 계산 줄은 그대로다.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NAS_ROOT, RESULTS, cache_path, name, write_json  # noqa: E402
from evaluate import MATLAB, MDIR, cmd_matlab, load_match, seq_dir  # noqa: E402

OFFICIAL_M = Path(__file__).resolve().parents[2] / "_external/fresco_official_20261107/FreSCo/fresco-matlab"
GUI_SHIM = Path(__file__).resolve().parent / "matlab_gui_shim"   # waitbar·visualizeMatchPairs만 대체(계산 무관)


def official_thresholds():
    f = np.concatenate([np.linspace(0, 200, 41), np.linspace(200, 1000, 9)])
    c = np.concatenate([np.linspace(0, 0.002, 41), np.linspace(0.002, 0.010, 9)])
    return f, c


def native_counts(m, xy, gap=100, radius=10.0):
    """createDescsAndRetrieve.m:186-246을 그대로 옮긴 카운트(1부터 번호)."""
    n = len(xy)
    f, c = official_thresholds()
    tp, fp, tn, fn = (np.zeros(len(f), np.int64) for _ in range(4))
    for qi in range(1, n + 1):
        hist = xy[0:max(1, qi - gap)]
        is_rev = np.linalg.norm(hist - xy[qi - 1], axis=1).min() <= radius
        if qi - gap <= 0:
            best, l1, cr = -1, 1e10, 1e10
        else:
            best, l1, cr = int(m[qi - 1, 0]), m[qi - 1, 1], m[qi - 1, 2]
        acc = (l1 <= f) & (cr <= c)
        if best > 0:
            good = np.linalg.norm(xy[best - 1] - xy[qi - 1]) <= radius
        else:
            good = False
        tp += acc & good; fp += acc & ~good
        fn += ~acc & is_rev; tn += ~acc & ~is_rev
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="kitti/09")
    ap.add_argument("--n", type=int, default=0, help="앞 n 키프레임만(0 = 전체)")
    a = ap.parse_args()
    sensor, seq = a.seq.split("/")
    base = seq_dir("val", sensor, seq)
    n_all = json.loads((base / "stream_done.json").read_text())["n"]
    n = a.n or n_all
    tag = f"c2_{name(sensor, seq)}_{n}"
    work = NAS_ROOT / "tmp" / tag
    if work.exists():
        shutil.rmtree(work)
    csvd = work / "kf/output_single_bev/csv"
    csvd.mkdir(parents=True)
    bev = np.fromfile(base / "bev.u8", dtype=np.uint8, count=n * 224 * 224).reshape(n, 224, 224)
    for i in range(n):   # cv::format FMT_CSV와 같은 정수 텍스트
        np.savetxt(csvd / f"{i:06d}.csv", bev[i], fmt="%d", delimiter=", ")
    with np.load(cache_path("val", sensor, seq)) as z:
        poses = z["poses"].astype(np.float64)[:n]
    rows = np.zeros((n, 16))
    rows[:, 0] = np.arange(n)
    rows[:, 1:4] = poses[:, :3, 3]
    rows[:, 7:16] = poses[:, :3, :3].reshape(n, 9)
    np.savetxt(work / "kf/keyframe_pose.csv", rows, delimiter=",", fmt="%.6f")
    out = work / "export"
    out.mkdir()
    script = (f"addpath('{OFFICIAL_M}'); addpath('{GUI_SHIM}'); cd('{work}'); "
              f"createDescsAndRetrieve('{csvd}/', '{work}/kf/keyframe_pose.csv', 'VAL', [70;0.000700000000000000], false); "
              f"k = load('./cached_descriptors/VAL/cell_keys.mat'); f = load('./cached_descriptors/VAL/cell_frescos.mat'); "
              f"K = cell2mat(cellfun(@(x) x', k.cell_keys, 'UniformOutput', false)); "
              f"F = zeros(120, 20, numel(f.cell_fresco)); for ii = 1:numel(f.cell_fresco), F(:,:,ii) = f.cell_fresco{{ii}}'; end; "
              f"fid = fopen('{out}/keys.f64', 'w'); fwrite(fid, K', 'double'); fclose(fid); "
              f"fid = fopen('{out}/fresco.f64', 'w'); fwrite(fid, F, 'double'); fclose(fid); "
              f"C = []; for nm = {{'num_true_pos','num_false_pos','num_true_neg','num_false_neg'}}, s = load(['./pr_result/VAL/' nm{{1}} '.mat']); C = [C; s.(nm{{1}})]; end; "
              f"fid = fopen('{out}/counts.f64', 'w'); fwrite(fid, C', 'double'); fclose(fid); disp('OFFICIAL_EXPORT_OK');")
    env = {k: v for k, v in os.environ.items() if k != "TMPDIR"}  # MATLAB 서비스 IPC는 로컬 /tmp 필요
    t0 = time.time()
    with open(work / "official_matlab.log", "w") as lf:
        r = subprocess.run(["nice", "-n", str(max(0, 10 - os.nice(0))), MATLAB, "-nodisplay", "-singleCompThread", "-batch", script],
                           stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env,
                           cwd=str(work), timeout=6 * 3600)
    t_off = time.time() - t0
    rep = {"seq": a.seq, "n": n, "official_rc": r.returncode, "official_sec": round(t_off, 1),
           "official_log": str(work / "official_matlab.log")}
    if r.returncode != 0:
        rep["passed"] = False
        write_json(RESULTS / f"verification_official_{tag}.json", rep)
        print(json.dumps(rep)); sys.exit(1)
    # 어댑터: 같은 BEV 앞 n개로 기술자, 간격 100 매칭
    ad = NAS_ROOT / "tmp" / f"{tag}_adapter"
    if ad.exists():
        shutil.rmtree(ad)
    ad.mkdir()
    (ad / "bev.u8").write_bytes(bev.tobytes())
    from evaluate import run_matlab
    t1 = time.time()
    run_matlab("all", ad, n, 100, ad / "matlab.log", timeout=6 * 3600)
    rep["adapter_sec"] = round(time.time() - t1, 1)
    ko = np.fromfile(out / "keys.f64", "<f8").reshape(n, 40)
    ka = np.fromfile(ad / "keys.f64", "<f8").reshape(n, 40)
    fo = np.fromfile(out / "fresco.f64", "<f8")
    fa = np.fromfile(ad / "fresco.f64", "<f8")
    rep["C1_keys_bit_equal"] = bool(np.array_equal(ko, ka, equal_nan=True))
    rep["C1_fresco_bit_equal"] = bool(np.array_equal(fo, fa, equal_nan=True))
    rep["nonfinite_keys"] = int((~np.isfinite(ka)).any(1).sum())
    m, c = load_match(ad, n)
    co = np.fromfile(out / "counts.f64", "<f8").reshape(4, 50).astype(np.int64)
    xy = np.loadtxt(work / "kf/keyframe_pose.csv", delimiter=",")[:, 1:3]   # 공식이 읽는 값 그대로
    mine = native_counts(m, xy)
    rep["C2_counts_equal"] = bool(all(np.array_equal(co[i], mine[k]) for i, k in enumerate(("tp", "fp", "tn", "fn"))))
    rep["official_counts"] = {k: co[i].tolist() for i, k in enumerate(("tp", "fp", "tn", "fn"))}
    rep["adapter_counts"] = {k: v.tolist() for k, v in mine.items()}
    f, cth = official_thresholds()
    q = np.arange(100, n)
    s = -np.maximum(m[q, 1], 1e5 * m[q, 2])
    acc2 = (m[q, 1][:, None] <= f) & (m[q, 2][:, None] <= cth)
    acc1 = s[:, None] >= -f   # 스칼라 점수 수락 (t1 = f)
    rep["C3_scalar_equiv_mismatch_cells"] = int((acc1 != acc2).sum())
    rep["C3_ratio_max_rel_err"] = float(np.max(np.abs(f[1:] / cth[1:] - 1e5) / 1e5))
    rep["passed"] = rep["C1_keys_bit_equal"] and rep["C1_fresco_bit_equal"] and rep["C2_counts_equal"] \
        and rep["C3_scalar_equiv_mismatch_cells"] == 0
    write_json(RESULTS / f"verification_official_{tag}.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if "counts" not in k}))


if __name__ == "__main__":
    main()
