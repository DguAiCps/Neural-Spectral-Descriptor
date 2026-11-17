#!/usr/bin/env python3
"""FreSCo KITTI 08 원 조건 재현의 로컬 controller와 분석 (2026-11-08).

  python3 controller.py run        로컬 D에서 detach 실행(nohup setsid). flock으로 하나만 돈다.
                                   설치 -> 스트리밍(이어받기) -> 서버 server_run(서버에서 detach)
                                   -> 감시 -> 작은 산출 미러 -> analyze -> done.json -> root 알림
  python3 controller.py analyze    미러된 산출만으로 지표·검산을 다시 계산(summary.json)

서버 계산은 서버에서 setsid로 떠 있어 WSL이 멈춰도 이어진다. controller를 다시 띄우면
완료 표시(stream_done, *_done.json, SERVER_DONE)를 보고 남은 단계만 이어간다.
알림: 계산 완료(+로컬 검산 통과) 또는 실패 때 completion_bridge.py arm(--root 절대 D 경로) 성공 뒤
sentinel(notified_<name>)을 쓰고 watch를 띄운다. 같은 이름 상태가 있으면 다시 arm하지 않는다.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
import native as N  # noqa: E402

OUT = N.OUT
ROOT = OUT.parent
BRIDGE = ROOT / "completion_bridge.py"
MIRROR = OUT / "server_mirror"
SRV_CODE = f"{N.SERVER_REPO}/scripts/fresco_native_20261108"
PAPER_F1, PAPER_P, PAPER_R = 0.8594, 0.8462, 0.8730
REF_COMMON = Path("/home/pkm/Neural-Spectral-Codec/results/pdv2_fresco_official_20261107/pipeline")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    with open(OUT / "controller.log", "a") as f:
        f.write(f"{now()} {msg}\n")


def status(**kw):
    p = OUT / "worker_status.json"
    d = N.rjson(p)
    d.update(role="fresco_native", updated_utc=now(), **kw)
    N.write_json(p, d)


def ssh(cmd, timeout=600, **kw):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                           "-o", "ServerAliveInterval=30", N.SERVER, cmd], timeout=timeout, **kw)


def srv_status():
    try:
        r = ssh(f"cd {N.SERVER_REPO} && python3 {SRV_CODE}/native.py server_status", timeout=120,
                capture_output=True, text=True, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None
    return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None


# ------------------------------------------------------------------ stages
def setup():
    w = str(N.WORK)
    ssh(f"mkdir -p {SRV_CODE} {w}/inputs {w}/logs", check=True)
    subprocess.run(["scp", "-q", str(HERE / "native.py"), f"{N.SERVER}:{SRV_CODE}/native.py"], check=True)
    st = srv_status() or {}
    kd = OUT / "keyframes"
    if st.get("committed", 0) == 0 and not st.get("stream_done"):
        files = [(kd / "keyframes.json", "keyframes.json"), (kd / "keyframe_pose.csv", "keyframe_pose.csv"),
                 (N.LOCAL_KITTI / "poses" / f"{N.SEQ}.txt", "inputs/poses.txt"),
                 (N.LOCAL_KITTI / "sequences" / N.SEQ / "times.txt", "inputs/times.txt")]
        for src, dst in files:
            subprocess.run(["scp", "-q", str(src), f"{N.SERVER}:{w}/{dst}"], check=True)
    log("setup ok")


def stream():
    for attempt in range(6):
        st = srv_status()
        if st is None:
            log("server status failed; retry in 60 s"); time.sleep(60); continue
        if st["stream_done"]:
            return True
        c = st["committed"]
        status(stage=f"streaming from keyframe {c}", running_processes=["local controller", "server recv"])
        log(f"stream start={c} attempt={attempt}")
        prod = subprocess.Popen(["nice", "-n", "10", sys.executable, str(HERE / "native.py"), "produce",
                                 "--start", str(c)], stdout=subprocess.PIPE, stderr=open(OUT / "produce.log", "a"))
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", N.SERVER,
                            f"cd {N.SERVER_REPO} && nice -n 10 ionice -c3 python3 {SRV_CODE}/native.py recv --start {c}"],
                           stdin=prod.stdout, stderr=open(OUT / "recv.log", "a"))
        prod.stdout.close(); prod.wait()
        log(f"stream rc={r.returncode} producer rc={prod.returncode}")
        time.sleep(5)
    st = srv_status()
    return bool(st and st["stream_done"])


def server_run():
    relaunched = 0
    while True:
        st = srv_status()
        if st is None:
            time.sleep(60); continue
        if st["server_done"]:
            return True
        if st["server_failed"]:
            return False
        try:
            alive = ssh(f"test -f {N.WORK}/server_run.pid && kill -0 $(cat {N.WORK}/server_run.pid) 2>/dev/null",
                        timeout=60, stdin=subprocess.DEVNULL).returncode == 0
        except subprocess.TimeoutExpired:
            time.sleep(60); continue
        if not alive:
            if relaunched >= 3:
                return False
            try:   # 실행 호출의 timeout은 실패가 아니다(다음 주기에 서버 PID로 판정)
                subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", N.SERVER,
                                f"cd {N.SERVER_REPO} && nohup setsid nice -n 10 ionice -c3 python3 {SRV_CODE}/native.py "
                                f"server_run >> {N.WORK}/logs/server_run.out 2>&1 < /dev/null &"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=60)
            except subprocess.TimeoutExpired:
                log("server_run launch call timed out; checking PID next cycle")
            relaunched += 1
            log(f"server_run launched ({relaunched})")
            time.sleep(20)
            continue
        status(stage=f"server {st['status'].get('stage')}", running_processes=["local controller"],
               server_pids={"running": [st["status"].get("pid")]})
        time.sleep(30)


def mirror():
    MIRROR.mkdir(exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude=bev.u8", "--exclude=*.csv", "--exclude=prefix_fixture/",
                    "--exclude=cached_descriptors/", "--include=*/", "--exclude=*.pcd",
                    f"{N.SERVER}:{N.WORK}/", f"{MIRROR}/"], check=True, timeout=1800)
    # 키프레임 CSV 원본 대조용(작다): 서버 사본과 로컬 사본이 같은지
    return True


# ------------------------------------------------------------------ analysis (local, mirrored files only)
def official_pr(tp, fp, tn, fn):
    """drawPresicionRecallCurve.m:42-59, 77 (MATLAB max는 NaN을 무시하고 첫 최대 위치)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
    i = int(np.nanargmax(f1))
    return {"max_F1": float(f1[i]), "idx_1based": i + 1, "precision": float(p[i]), "recall": float(r[i]),
            "F1_all": [None if not np.isfinite(x) else float(x) for x in f1]}


def thresholds50():
    f = np.concatenate([np.linspace(0, 200, 41), np.linspace(200, 1000, 9)])
    c = np.concatenate([np.linspace(0, 0.002, 41), np.linspace(0.002, 0.010, 9)])
    return f, c


def native_counts(m, xy, gap=100, radius=10.0):
    """createDescsAndRetrieve.m:186-246 (1부터 번호), 50쌍 임계."""
    n = len(xy)
    f, c = thresholds50()
    tp, fp, tn, fn = (np.zeros(len(f), np.int64) for _ in range(4))
    for qi in range(1, n + 1):
        hist = xy[0:max(1, qi - gap)]
        is_rev = np.linalg.norm(hist - xy[qi - 1], axis=1).min() <= radius
        if qi - gap <= 0:
            best, l1, cr = -1, 1e10, 1e10
        else:
            best, l1, cr = int(m[qi - 1, 0]), m[qi - 1, 1], m[qi - 1, 2]
        acc = (l1 <= f) & (cr <= c)
        good = best > 0 and np.linalg.norm(xy[best - 1] - xy[qi - 1]) <= radius
        tp += acc & good; fp += acc & ~good
        fn += ~acc & is_rev; tn += ~acc & ~is_rev
    return tp, fp, tn, fn


def native_continuous(m, xy, gap=100, radius=10.0):
    """N1: 같은 산식, 임계만 스칼라 점수 s=-max(L1,1e5 cos_row)의 모든 고유값."""
    n = len(xy)
    q = np.arange(1, n + 1)
    has = q - gap > 0
    rev = np.array([np.linalg.norm(xy[0:max(1, k - gap)] - xy[k - 1], axis=1).min() <= radius for k in q])
    best = np.where(has, m[:, 0], -1).astype(int)
    s = np.where(has, -np.maximum(m[:, 1], 1e5 * m[:, 2]), -np.inf)
    good = np.zeros(n, bool)
    good[has] = np.linalg.norm(xy[best[has] - 1] - xy[has], axis=1) <= radius
    taus = np.unique(s[np.isfinite(s)])[::-1]
    best_f1 = (-1.0, None)
    for t in taus:
        acc = s >= t
        tp = int((acc & good).sum()); fp = int((acc & ~good).sum()); fn = int((~acc & rev).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        p, r = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if f1 > best_f1[0]:
            best_f1 = (f1, {"tau": float(t), "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn})
    return {"max_F1": best_f1[0], **(best_f1[1] or {}), "n_queries": n, "n_revisit": int(rev.sum()),
            "n_with_candidates": int(has.sum())}


def common_metric(score, top1, min_past):
    from _pr_eval_lib import classify_queries, pr_sweep
    qc = classify_queries(min_past)
    sw = pr_sweep(score, top1, qc)
    r1 = float((top1[qc["positive"]] <= 5.0).mean()) if qc["n_pos"] else None
    return {k: sw[k] for k in ("AP", "max_F1", "R@100P", "R@99P", "R@95P")} | {"R@1": r1, **sw["counts"]}


def common_from_match(m, xyz, gap, q_lo, q_hi):
    """N2·N3: 0부터 질의 t, DB j <= t-gap, 3D GT, 공통 지표."""
    t = np.arange(q_lo, q_hi)
    j = m[t, 0].astype(int) - 1
    assert np.all((j >= 0) & (j <= t - gap))
    score = -np.maximum(m[t, 1], 1e5 * m[t, 2])
    assert np.isfinite(score).all()
    top1 = np.linalg.norm(xyz[j] - xyz[t], axis=1)
    mp = np.array([np.linalg.norm(xyz[:k - gap + 1] - xyz[k], axis=1).min() for k in t])
    return common_metric(score, top1, mp) | {"query_range_0based": [int(q_lo), int(q_hi)], "gap": gap}


def analyze():
    M = MIRROR
    kfcsv = np.loadtxt(OUT / "keyframes" / "keyframe_pose.csv", delimiter=",")
    n = len(kfcsv)
    xy, xyz = kfcsv[:, 1:3], kfcsv[:, 1:4]
    ver = {}
    pre = N.rjson(M / "prefix_check.json")
    ver["E1_keyframe_pose_csv_text_equal"] = pre.get("E1_keyframe_pose_csv_text_equal")
    ver["E1_idx_xyz_fields_equal"] = pre.get("E1_idx_xyz_fields_equal")
    ver["E1_n_keyframes_official_vs_port"] = [pre.get("n_keyframes_official"), pre.get("n_keyframes_port")]
    ver["E2_bev_bit_equal"] = pre.get("E2_bev_bit_equal")
    ver["E5_near_threshold"] = N.rjson(OUT / "keyframes" / "keyframes.json").get("near_threshold_comparisons_1e-5m")
    ex = M / "official" / "export"
    C = np.fromfile(ex / "counts.f64", "<f8").reshape(4, 50).astype(np.int64)
    tp, fp, tn, fn = C
    n0 = official_pr(tp.astype(float), fp.astype(float), tn.astype(float), fn.astype(float))
    f1o = ex / "f1_official.f64"
    if f1o.exists():
        fo = np.fromfile(f1o, "<f8")
        ver["official_F1scores_mat_equal_port"] = bool(np.allclose(
            np.nan_to_num(fo, nan=-1), np.nan_to_num(np.array([np.nan if v is None else v for v in n0["F1_all"]]),
                                                       nan=-1), rtol=0, atol=1e-15))
    ad = M / "adapter_gap100"
    ka = np.fromfile(ad / "keys.f64", "<f8").reshape(n, 40)
    ko = np.fromfile(ex / "keys.f64", "<f8").reshape(n, 40)
    ver["E3_keys_bit_equal"] = bool(np.array_equal(ka, ko, equal_nan=True))
    ver["E3_fresco_bit_equal"] = bool(np.array_equal(np.fromfile(ad / "fresco.f64", "<f8"),
                                                     np.fromfile(ex / "fresco.f64", "<f8"), equal_nan=True))
    ver["nonfinite_keys"] = int((~np.isfinite(ka)).any(1).sum())
    m100 = np.fromfile(ad / "match.f64", "<f8").reshape(n, 6)
    mine = native_counts(m100, xy)
    ver["E4_counts_equal"] = bool(all(np.array_equal(a, b) for a, b in zip(mine, (tp, fp, tn, fn))))
    n1 = native_continuous(m100, xy)
    n2 = common_from_match(m100, xyz, 100, 100, n)
    m30 = np.fromfile(M / "adapter_gap30" / "match.f64", "<f8").reshape(n, 6)
    n3 = common_from_match(m30, xyz, 30, 30, n - 15)
    # sanity: 같은 공통 지표 코드로 기존 공통 원장 kitti/08 재집계
    z = np.load(REF_COMMON / "ledgers/test/dump_fresco_kitti_08.npz")
    meta = json.loads(str(z["meta"]))
    q = z["query_idx"]; sel = (q >= 30) & (q < meta["n_frames"] - 15)
    ref_re = common_metric(z["score_fresco"][sel], z["top1_geo_dist"][sel], z["min_past_gt_dist"][sel])
    ref_sum = N.rjson(REF_COMMON / "summary.json")["scores"]["score_fresco"]["per_sequence"]["kitti/08"]
    ver["sanity_common_ledger_recomputed_equal"] = bool(
        abs(ref_re["max_F1"] - ref_sum["max_F1"]) < 1e-12 and abs(ref_re["AP"] - ref_sum["AP"]) < 1e-12
        and ref_re["n_pos"] == ref_sum["n_positive"])
    d = n0["max_F1"] - PAPER_F1
    verdict = ("reproduced_exact" if abs(d) < 0.0005 else
               "reproduced_within_tolerance" if abs(d) <= 0.01 else "not_reproduced")
    checks_ok = all(v is True for k, v in ver.items() if k.startswith(("E1_idx", "E2", "E3", "E4", "sanity"))) \
        and ver["E5_near_threshold"] == 0
    summ = {"created_utc": now(), "seq": "kitti/08", "n_keyframes": n,
            "paper": {"max_F1": PAPER_F1, "precision": PAPER_P, "recall": PAPER_R},
            "N0_native_official": {k: v for k, v in n0.items() if k != "F1_all"} | {
                "counts_at_best": {"tp": int(tp[n0["idx_1based"] - 1]), "fp": int(fp[n0["idx_1based"] - 1]),
                                   "tn": int(tn[n0["idx_1based"] - 1]), "fn": int(fn[n0["idx_1based"] - 1])},
                "thresholds_at_best": [float(thresholds50()[0][n0["idx_1based"] - 1]),
                                       float(thresholds50()[1][n0["idx_1based"] - 1])],
                "F1_all_50": n0["F1_all"]},
            "delta_vs_paper": {"max_F1": d, "precision": n0["precision"] - PAPER_P, "recall": n0["recall"] - PAPER_R},
            "verdict": verdict,
            "condition_gap": {"N1_native_all_thresholds": n1, "N2_native_kf_gap100_common_metric": n2,
                              "N3_native_kf_gap30_common_metric": n3,
                              "REF_common_table_kitti08": ref_sum, "REF_recomputed": ref_re},
            "verification": ver, "verification_all_pass": bool(checks_ok)}
    N.write_json(OUT / "summary.json", summ)
    return summ


# ------------------------------------------------------------------ notify / main
def notify(name):
    if (OUT / f"notified_{name}").exists():
        return True
    state = ROOT / "completion_monitor" / f"watch_{name}.json"
    if not state.exists():
        with open(OUT / "controller.log", "a") as f:
            r = subprocess.run([sys.executable, str(BRIDGE), "arm", "--root", str(ROOT), "--name", name,
                                "--workers", OUT.name, "--live", "--max-hours", "24"], stdout=f, stderr=f)
        if r.returncode != 0 or not state.exists():
            log(f"arm {name} failed rc={r.returncode}; retry")
            return False
    (OUT / f"notified_{name}").write_text(now())
    log(f"armed {name}; watch")
    with open(OUT / "controller.log", "a") as f:
        subprocess.Popen([sys.executable, str(BRIDGE), "watch", "--root", str(ROOT), "--name", name,
                          "--interval", "60"], stdout=f, stderr=f, start_new_session=True)
    return True


def finish(ok, why, summ=None):
    # fail_20261108 이름은 16:05Z 잘못된 arm(false_fail_alarm_note.json)에 쓰여 sentinel이 남아 있다
    name = "ovn_fresco_native_compute_20261108" if ok else "ovn_fresco_native_fail2_20261108"
    N.write_json(OUT / "done.json", {"compute_complete": ok, "reason": why, "utc": now(),
                                     "summary": "summary.json" if summ else None,
                                     "verdict": summ.get("verdict") if summ else None,
                                     "verification_all_pass": summ.get("verification_all_pass") if summ else None})
    status(stage=("complete: server compute + local analysis done" if ok else f"FAILED: {why}"),
           running_processes=[], server_pids={"running": []},
           next_action="report_ko.txt 작성(작업자) / root 검산")
    for _ in range(48):
        if notify(name):
            return
        time.sleep(300)


def run():
    lockf = open(OUT / "controller.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("controller already running"); return 0
    (OUT / "controller.pid").write_text(str(os.getpid()))
    log(f"controller start pid={os.getpid()}")
    try:
        setup()
        if not stream():
            finish(False, "stream failed"); return 1
        status(stage="stream done; server_run")
        if not server_run():
            mirror(); finish(False, "server_run failed (see server_mirror/SERVER_FAILED.json)"); return 1
        mirror()
        summ = analyze()
        log(f"analyze verdict={summ['verdict']} F1={summ['N0_native_official']['max_F1']:.4f} "
            f"checks={summ['verification_all_pass']}")
        finish(bool(summ["verification_all_pass"]),
               "complete + local checks pass" if summ["verification_all_pass"] else "complete but checks failed", summ)
    except Exception as e:  # noqa: BLE001
        log(f"controller error {type(e).__name__}: {e}")
        finish(False, f"controller error {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        sys.exit(run())
    elif cmd == "analyze":
        print(json.dumps({k: v for k, v in analyze().items() if k != "condition_gap"}, indent=1, default=str))
