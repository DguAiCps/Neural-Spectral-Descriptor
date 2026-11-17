#!/usr/bin/env python3
"""공식 FreSCo MATLAB 단계와 과거 전용 원장 (서버 호스트, 2026-11-07).

  matlab --role R --seq S [--gap 30]   NAS/<role>/<name>/ bev.u8 -> keys/fresco/match/cand
                                       (MATLAB -nodisplay -singleCompThread -batch, nice 10)
  ledger --role R --seq S              match -> results/.../pipeline/ledgers/<role>/dump_fresco_<name>.npz
기술자·후보·승자는 공식 createDescsAndRetrieve.m의 줄을 그대로 옮긴 fresco_adapter.m이 계산한다
(matlab_gen.py). 이 파일은 실행과 원장 형식만 맡는다.

원장(pr-dump-v1, BEVPlace++ 원장과 같은 열): 질의 t = 30..N-1(0부터), 후보 j <= t-30
(MATLAB LOOP_FRAME_INTERVAL = 30: DB 1..query_idx-30 = 0부터 j <= t-30), 승자 j_star = 공식
L1(log-log) 최소 후보. 주 점수 score_fresco = -max(L1, 1e5 * cos_row): 공식 2-threshold 수락
(L1 <= t1 AND cos_row <= t2, 공식 50쌍 모두 t1/t2 = 1e5)과 같은 순서의 스칼라(audit A2).
보조 진단 score_negl1 = -L1. 자세 poses_gt(시험)/poses(검증, GT)는 top1_geo_dist·min_past_gt_dist
라벨에만 쓴다. 비유한 공식 점수·승자 없음 규칙(사전 고정): score = -1e12(어떤 유한 임계에서도
최저), 승자 없으면 j_star = knnsearch 첫 후보, 개수는 meta에 기록.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (NAS_ROOT, REPO, RESULTS, ROLES, cache_path, name, sha256,  # noqa: E402
                    write_json, FRESCO_COMMIT, PREP_COMMIT, FAMILY)

MATLAB = "/mnt/aicps-nas/qkrrlans00/matlab/R2026a/bin/matlab"
MDIR = NAS_ROOT / "build/matlab"
GAP = 30
SCORE_RATIO = 1e5
NONFINITE_SCORE = -1e12


def seq_dir(role, sensor, seq, gap=GAP) -> Path:
    d = NAS_ROOT / role / name(sensor, seq)
    return d if gap == GAP else d / f"gap{gap}"


def run_matlab(mode: str, d: Path, n: int, gap: int, log: Path, timeout: int) -> float:
    cmd = (f"addpath('{MDIR}'); fresco_adapter('{mode}', '{d}', {n}, {gap});")
    env = {k: v for k, v in os.environ.items() if k != "TMPDIR"}
    env.update(OMP_NUM_THREADS="1",  # TMPDIR 기본값: MathWorks 서비스 IPC 소켓이 NFS에서 실패(5201)
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    t0 = time.time()
    with open(log, "a") as lf:
        lf.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {mode} gap={gap} n={n}\n")
        lf.flush()
        r = subprocess.run(["nice", "-n", str(max(0, 10 - os.nice(0))), MATLAB, "-nodisplay", "-singleCompThread", "-batch", cmd],
                           stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env,
                           cwd=str(NAS_ROOT / "tmp"), timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"MATLAB {mode} rc={r.returncode} ({log})")
    return time.time() - t0


def cmd_matlab(role, k, gap=GAP, mode="all"):
    sensor, seq = k.split("/")
    base = seq_dir(role, sensor, seq)
    sd = json.loads((base / "stream_done.json").read_text())
    n = sd["n"]
    d = seq_dir(role, sensor, seq, gap)
    d.mkdir(parents=True, exist_ok=True)
    if gap != GAP:   # 기술자는 공통, 매칭만 다른 간격
        assert (base / "desc_done.json").exists()
        for f in ("keys.f64", "fresco.f64"):
            if not (d / f).exists():
                os.symlink(base / f, d / f)
        mode = "match"
    done = d / "match_done.json"
    if done.exists():
        return json.loads(done.read_text())
    assert os.path.getsize(base / "bev.u8") == n * 224 * 224
    sec = run_matlab(mode, d, n, gap, d / "matlab.log", timeout=12 * 3600)
    for f, size in (("keys.f64", n * 40 * 8), ("fresco.f64", n * 2400 * 8),
                    ("match.f64", n * 6 * 8), ("cand.i32", n * 20 * 4)):
        assert os.path.getsize(d / f) == size, (f, os.path.getsize(d / f), size)
    logtxt = (d / "matlab.log").read_text()
    def tag(t):
        ln = [x for x in logtxt.splitlines() if x.startswith(t)]
        return ln[-1] if ln else None
    if mode == "all":
        write_json(base / "desc_done.json", {"n": n, "keys_sha256": sha256(d / "keys.f64"),
                                             "fresco_sha256": sha256(d / "fresco.f64"),
                                             "log": tag("FRESCO_DESC_DONE")})
    rec = {"role": role, "seq": k, "n": n, "gap": gap, "wall_sec": round(sec, 1),
           "desc_log": tag("FRESCO_DESC_DONE"), "match_log": tag("FRESCO_MATCH_DONE"),
           "match_sha256": sha256(d / "match.f64"), "cand_sha256": sha256(d / "cand.i32"),
           "adapter_sha256": sha256(MDIR / "fresco_adapter.m"),
           "bev_sha256": sd["bev_sha256"],
           "hardware_label": "diagnostic: cps2 server CPU, MATLAB R2026a -singleCompThread, shared load",
           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(done, rec)
    return rec


def load_match(d: Path, n: int):
    m = np.fromfile(d / "match.f64", dtype="<f8").reshape(n, 6)
    c = np.fromfile(d / "cand.i32", dtype="<i4").reshape(n, 20)
    return m, c


def build_ledger(role, k, gap=GAP) -> dict:
    sensor, seq = k.split("/")
    d = seq_dir(role, sensor, seq, gap)
    n = json.loads((seq_dir(role, sensor, seq) / "stream_done.json").read_text())["n"]
    cp = cache_path(role, sensor, seq)
    with np.load(cp) as z:
        poses = z["poses_gt" if role == "test" else "poses"].astype(np.float64)
    assert len(poses) == n
    pos = poses[:, :3, 3]
    m, c = load_match(d, n)
    q = np.arange(gap, n, dtype=np.int64)
    best, l1, cr, cc, ang, edb = (m[q, i] for i in range(6))
    assert np.all(edb == q - gap + 1), "end_db_idx != t - gap + 1"
    n_kd = (c[q] > 0).sum(1)
    assert np.all(n_kd == np.minimum(20, q - gap + 1)), "kd candidate count"
    assert np.all(c[q][c[q] > 0] <= np.repeat(q - gap + 1, n_kd)), "candidate beyond prefix"
    no_winner = best < 1
    j = np.where(no_winner, c[q, 0], best).astype(np.int64) - 1
    assert np.all((j >= 0) & (j <= q - gap)), "winner beyond the causal prefix"
    in_cand = (c[q] == (j + 1)[:, None]).any(1)
    assert in_cand.all(), "winner not among kd candidates"
    raw = -np.maximum(l1, SCORE_RATIO * cr)
    nonfinite = ~np.isfinite(raw) | no_winner
    score = np.where(nonfinite, NONFINITE_SCORE, raw)
    negl1 = np.where(np.isfinite(l1) & ~no_winner, -l1, NONFINITE_SCORE)
    min_past = np.array([np.linalg.norm(pos[:t - gap + 1] - pos[t], axis=1).min() for t in q])
    top1 = np.linalg.norm(pos[j] - pos[q], axis=1)
    arrays = dict(query_idx=q, j_star=j, score_fresco=score, score_negl1=negl1,
                  l1_dist=l1, cos_row_dist=cr, cos_col_dist=cc, angle_deg=ang,
                  top1_geo_dist=top1, min_past_gt_dist=min_past, n_candidates=q - gap + 1,
                  n_kd_candidates=n_kd)
    mrec = json.loads((d / "match_done.json").read_text())
    meta = {"schema": "pr-dump-v1", "method": "fresco_official", "role": role, "sensor": sensor,
            "sequence": seq, "min_index_gap": gap, "n_frames": int(n), "n_queries": int(len(q)),
            "primary_score": "score_fresco",
            "scores": {"score_fresco": "-max(L1_loglog, 1e5*cos_row_dist) of the official winner; "
                                       "official acceptance L1<=t1 AND cos_row<=t2 with t1/t2=1e5 "
                                       "(createDescsAndRetrieve.m:18-26,219) as one scalar; higher=accept",
                       "score_negl1": "-L1 of the official winner (diagnostic only)"},
            "retrieval": "official MATLAB: createns kdtree on 40D keys of DB 1..q-gap, knnsearch K=20, "
                         "computeFrescoDist offsets 0:59, winner = min L1 (strict <)",
            "n_nonfinite_or_no_winner": int(nonfinite.sum()), "n_no_winner": int(no_winner.sum()),
            "official": {"fresco": FRESCO_COMMIT, "preprocessing": PREP_COMMIT, "family": FAMILY[sensor]},
            "caches": {"keyframes": cp.name, "keyframes_sha256": sha256(cp),
                       "positions": "poses_gt" if role == "test" else "poses (GT, val cache)"},
            "matlab": mrec}
    out = RESULTS / "ledgers" / (role if gap == GAP else f"{role}_gap{gap}") / f"dump_fresco_{name(sensor, seq)}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
    os.replace(tmp, out)
    return {"ledger": str(out.relative_to(REPO)), "sha256": sha256(out), **{
        k2: meta[k2] for k2 in ("n_queries", "n_nonfinite_or_no_winner", "n_no_winner")}}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("matlab", "ledger"):
        p = sub.add_parser(c)
        p.add_argument("--role", required=True, choices=list(ROLES))
        p.add_argument("--seq", required=True)
        p.add_argument("--gap", type=int, default=GAP)
    a = ap.parse_args()
    if a.cmd == "matlab":
        print(json.dumps(cmd_matlab(a.role, a.seq, a.gap)))
    else:
        print(json.dumps(build_ledger(a.role, a.seq, a.gap)))


if __name__ == "__main__":
    main()
