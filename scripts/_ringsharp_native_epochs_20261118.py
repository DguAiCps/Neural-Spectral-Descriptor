#!/usr/bin/env python3
"""RING#-L native NCLT 규약1 실행의 에폭별 공식 평가(원 논문 재현 진단, 2026-11-18).

두 실행(공개 설정 lr 1e-4·wd 1e-5, 논문 본문 값 lr 1e-3·wd 1e-4)의 보존 가중치 e1–e30을 같은 공식 평가기
(_external/RINGSharp/tools/evaluate_ours_gl.py, README 명령 --n_rerank 1000 --refine)로 하나씩 평가한다.
에폭을 고르지 않는다. 범위만 보고한다. 표2와 무관하다.

모드
  run      cps2 호스트에서 실행. 컨테이너 ring_baseline 안에서 에폭마다 기존 래퍼 rs_wrap.py(장치 확인·numpy 2 호환,
           값 불변)로 공식 평가기를 돌리고, 기존 rs_metrics.py로 지표·독립 재계산을 남긴다. 이미 지표가 있으면 건너뛴다.
  collect  로컬에서 실행. 서버에서 받아 온 에폭별 final_metrics.json을 모아 per_epoch.json·table.md를 쓴다.

평가 입력은 기존 실행과 같다: 시험 피클 sha ed43777b…(바이트 사본)과 검증된 시험 스캔 329장(symlink).
공식 평가기는 dataset_root/<seq>/bev/에 BEV 캐시를 쓴다. BEV는 점군에서만 만들어 가중치와 무관하므로 모든 에폭이
/dev/shm의 같은 캐시를 쓴다(첫 평가가 캐시를 만든다). 캐시 재사용이 값을 바꾸지 않는지는 기존 e30 두 값의 재현으로 확인한다.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- 서버 경로(cps2)
REPO = Path("/home/qkrrlans00/Neural-Spectral-Codec")
REPO_C = "/workspace/Neural-Spectral-Codec"
OVN = "_external/overnight_repro_20261108"
SREL = "_external/ringsharp_native_epochs_20261118"
SW, CW = REPO / SREL, f"{REPO_C}/{SREL}"
RSC = f"{REPO_C}/_external/RINGSharp"
TOOLS_C = f"{REPO_C}/{OVN}/native_ringsharp_recovery/code"      # 기존 rs_wrap.py·rs_metrics.py(읽기만)
EVAL_SRC = REPO / OVN / "native_ringsharp_training/evaldata/nclt"
EVAL_SRC_C = f"{REPO_C}/{OVN}/native_ringsharp_training/evaldata/nclt"
CONTAINER = "ring_baseline"
GPU_UUID = "GPU-71d075cf-715e-8334-aa18-896f0d1c633c"            # 기존 두 실행의 평가 장치
GPU_BUS = "0x4c"
TEST_PK = "test_2012-02-04_2012-03-17_20.0_5.0.pickle"
TEST_PK_SHA = "ed43777b7e3964870246b95d8f6868f8caef71425cb9ac2c1f9a5760c03eeaea"
SHM = Path("/dev/shm/nsdrse_20261118")
RUNS = {
    "public": {"dir": "native_ringsharp_recovery", "exp": "nsdrsr_p1s123e30_r20261108",
               "stem": "model_ring_sharp_l_nclt_20261108_0824", "lr": 1e-4, "wd": 1e-5},
    "paper": {"dir": "ringsharp_paper_recipe_20261109", "exp": "nsdrsp_p1s123e30_paper_20261109",
              "stem": "model_ring_sharp_l_nclt_20261109_0325", "lr": 1e-3, "wd": 1e-4},
}
PAPER = {"Recall@1": 0.85, "F1": 0.87, "AUC": 0.91}
RADII = [2, 5, 10, 20, 25]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shm_root():
    root = SHM / "evaldata/nclt"
    root.mkdir(parents=True, exist_ok=True)
    if not (root / TEST_PK).exists():
        shutil.copyfile(EVAL_SRC / TEST_PK, root / TEST_PK)
    if sha256(root / TEST_PK) != TEST_PK_SHA:
        sys.exit("shm test pickle hash mismatch")
    for seq in ("2012-02-04", "2012-03-17"):
        (root / seq).mkdir(exist_ok=True)
        link = root / seq / "velodyne_sync"
        if not link.is_symlink():
            os.symlink(f"{EVAL_SRC_C}/{seq}/velodyne_sync", link)
    return root


def dexec(args, workdir):
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPU_UUID, "NSC_ACT_CKPT": "0", "NSC_MICRO_BATCH": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{RSC}:{RSC}/tools",
           "MPLBACKEND": "Agg", "MPLCONFIGDIR": f"{CW}/tmp/mpl"}
    cmd = ["docker", "exec", "-w", workdir]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    return cmd + [CONTAINER, "nice", "-n", "10"] + args


def eval_one(run, ep, root):
    R = RUNS[run]
    tag = f"nsdrse_{run}_e{ep:02d}"
    wd = SW / run / f"e{ep:02d}"
    wdc = f"{CW}/{run}/e{ep:02d}"
    if (wd / "metrics/final_metrics.json").exists():
        return "skip"
    if wd.exists():
        shutil.rmtree(wd)                       # 중단된 이전 시도만(지표 없음) 지운다
    wd.mkdir(parents=True)
    ck = f"{REPO_C}/{OVN}/{R['dir']}/run/results/weights/{R['exp']}/{R['stem']}_{ep}.pth"
    cfg = f"{REPO_C}/{OVN}/{R['dir']}/frozen/ring_sharp_l_nclt.txt"
    args = ["timeout", "3600", "python3", f"{TOOLS_C}/rs_wrap.py", "--tag", tag, "--env-out", f"{wdc}/env.json",
            "--expect-uuid", GPU_UUID, "--expect-pci-bus", GPU_BUS, "--script", f"{RSC}/tools/evaluate_ours_gl.py",
            "--np-scalar-compat", "--", "--dataset_type", "nclt", "--dataset_root", str(root), "--exp_name", tag,
            "--eval_set", TEST_PK, "--model_config", cfg, "--weight", ck, "--n_rerank", "1000", "--refine"]
    t0 = time.time()
    with open(wd / "eval.log", "w") as log:
        rc = subprocess.call(dexec(args, wdc), stdout=log, stderr=subprocess.STDOUT)
    dur = round(time.time() - t0, 1)
    if rc != 0:
        (wd / "FAILED.json").write_text(json.dumps({"rc": rc, "seconds": dur}))
        return f"fail rc={rc}"
    rc2 = subprocess.call(dexec(["python3", f"{TOOLS_C}/rs_metrics.py", "--results_dir",
                                 f"{wdc}/results/{tag}/test_2012-02-04_2012-03-17_20.0_5.0",
                                 "--out_dir", f"{wdc}/metrics"], wdc),
                          stdout=open(wd / "metrics.log", "w"), stderr=subprocess.STDOUT)
    (wd / "timing.json").write_text(json.dumps({"eval_seconds": dur, "metrics_rc": rc2, "weight": ck,
                                                "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return f"ok {dur}s rc_metrics={rc2}"


def cmd_run(a):
    root = shm_root()
    eps = [int(x) for x in a.epochs.split(",")]
    for run in a.runs.split(","):
        for ep in eps:
            st = eval_one(run, ep, root)
            print(time.strftime("%H:%M:%S"), run, ep, st, flush=True)


def cmd_collect(a):
    src, out = Path(a.src), Path(a.out)
    rows = []
    for run, R in RUNS.items():
        for ep in range(1, 31):
            d = src / run / f"e{ep:02d}"
            fm = d / "metrics/final_metrics.json"
            if not fm.exists():
                continue
            m = json.load(open(fm))
            t = json.load(open(d / "timing.json"))
            o = m["official_metrics"]
            rows.append({"run": run, "lr": R["lr"], "wd": R["wd"], "epoch": ep,
                         "Recall@1": o["Recall@1"], "F1": o["F1"], "AUC": o["AUC"], "AP_reference": o["AP_reference"],
                         "Recall@1_all_radii": o["Recall@1_all_radii"],
                         "n_correct_r10": round(o["Recall@1"] * m["independent_recompute"]["n_queries_with_positive_10m"]),
                         "n_pos_r10": m["independent_recompute"]["n_queries_with_positive_10m"],
                         "all_checks_pass": m["all_checks_pass"], "eval_seconds": t["eval_seconds"]})
            n_ok = rows[-1]["n_correct_r10"]
            # 최대 F1이 "모든 질의 수락" 점(2·TP/(TP+265))과 같은지: 기존 e30 분석의 구조가 에폭마다 유지되는지 확인
            rows[-1]["F1_equals_accept_all_point"] = abs(o["F1"] - 2 * n_ok / (n_ok + 265)) < 1e-6
    summ = {}
    for run in RUNS:
        rr = [r for r in rows if r["run"] == run]
        if not rr:
            continue
        s = {"n_epochs": len(rr)}
        for k in ("Recall@1", "F1", "AUC"):
            v = [r[k] for r in rr]
            best = max(v)
            s[k] = {"min": min(v), "max": best, "median": sorted(v)[len(v) // 2],
                    "epochs_at_max": [r["epoch"] for r in rr if r[k] == best],
                    "paper": PAPER[k], "epochs_reaching_paper_2dp": [r["epoch"] for r in rr if round(r[k], 2) >= PAPER[k]],
                    "epochs_reaching_paper_exact": [r["epoch"] for r in rr if r[k] >= PAPER[k]]}
        s["epochs_all_three_reach_paper_2dp"] = [r["epoch"] for r in rr
                                                 if all(round(r[k], 2) >= PAPER[k] for k in PAPER)]
        s["all_checks_pass_all_epochs"] = all(r["all_checks_pass"] for r in rr)
        s["epochs_F1_not_accept_all_point"] = [r["epoch"] for r in rr if not r["F1_equals_accept_all_point"]]
        s["eval_seconds_total"] = round(sum(r["eval_seconds"] for r in rr), 1)
        summ[run] = s
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "epoch-wise official evaluation of preserved native NCLT protocol-1 checkpoints; range report only, "
                       "no epoch is selected; not related to the main table",
               "paper": PAPER, "summary": summ, "rows": rows}, open(out / "per_epoch.json", "w"), indent=1)
    L = []
    for run in RUNS:
        rr = [r for r in rows if r["run"] == run]
        if not rr:
            continue
        R = RUNS[run]
        L.append(f"## {run} (lr {R['lr']:g}, wd {R['wd']:g})\n")
        L.append("| 에폭 | R@1 (10 m) | 정답/양성 | 최대 F1 | PR-AUC | AP(참고) | R@1 2 m | 5 m | 20 m | 25 m |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rr:
            ar = r["Recall@1_all_radii"]
            L.append(f"| {r['epoch']} | {r['Recall@1']:.4f} | {r['n_correct_r10']}/{r['n_pos_r10']} | {r['F1']:.4f} | "
                     f"{r['AUC']:.4f} | {r['AP_reference']:.4f} | {ar['2m']:.4f} | {ar['5m']:.4f} | {ar['20m']:.4f} | {ar['25m']:.4f} |")
        s = summ[run]
        L.append("")
        for k in ("Recall@1", "F1", "AUC"):
            x = s[k]
            L.append(f"- {k}: 범위 {x['min']:.4f}–{x['max']:.4f}(중앙값 {x['median']:.4f}), 최고 에폭 {x['epochs_at_max']}, "
                     f"논문 {x['paper']:.2f} 이상(소수 둘째 자리 반올림) 에폭 {x['epochs_reaching_paper_2dp'] or '없음'}")
        L.append(f"- 세 지표가 한 에폭에서 모두 논문값 이상(반올림): {s['epochs_all_three_reach_paper_2dp'] or '없음'}")
        L.append(f"- 독립 재계산 일치: {'모든 에폭' if s['all_checks_pass_all_epochs'] else '불일치 있음'}. "
                 f"최대 F1이 전부 수락점이 아닌 에폭: {s['epochs_F1_not_accept_all_point'] or '없음'}. "
                 f"평가 시간 합 {s['eval_seconds_total']:.0f} s")
        L.append("")
    (out / "table.md").write_text("# RING#-L native NCLT 규약1 에폭별 공식 평가\n\n논문(arXiv 2409.00206v2 표, NCLT): "
                                  "R@1 0.85, 최대 F1 0.87, PR-AUC 0.91. 에폭을 고르지 않고 범위만 본다.\n\n" + "\n".join(L))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", default="public")
    r.add_argument("--epochs", default=",".join(str(e) for e in range(1, 31)))
    c = sub.add_parser("collect")
    c.add_argument("--src", required=True, help="서버 작업 폴더의 로컬 사본")
    c.add_argument("--out", default="results/pdv2_ringsharp_native_epochs_20261118")
    a = ap.parse_args()
    cmd_run(a) if a.mode == "run" else cmd_collect(a)


if __name__ == "__main__":
    main()
