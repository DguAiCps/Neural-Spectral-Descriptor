#!/usr/bin/env python3
"""invkey 진단 연구: 학습 범위를 MCD·KITTI·KITTI-360으로 줄여 K=16 선형 키를 다시 학습한다 (2026-11-15).

모델·손실·배치·양성·검색 회계는 scripts/_invkey_train_linear_20261114.py를 그대로 import해 쓴다
(TrainData, train_one, encode, eval_keys, dev_summary; TF32 끔). 바뀐 것은 아래뿐이다.

분할 (사용자 선택, 2026-11-15; HeLiPR·MulRan·NCLT는 센서 리그의 방위 사각 때문에 쓰지 않는다)
  학습  kitti/01 02 05 06, kitti360/0000, mcd/kth 6개
  개발  kitti/00 07 08 (모델 선택 전용, 모두 HDL-64E)
  평가  kitti/09, kitti360/0004 0006, mcd/tuhh 6개 (이 스크립트는 평가용을 읽지 않는다)
  장소 묶음(data_survey.json overlap_units)이 분할 사이에 겹치지 않는지, 지리 감사 표에서 역할이 다른
  같은 도시 시퀀스 쌍의 최소 거리가 50 m 이상인지 단언한다.
s·PCA: 새 학습 11시퀀스만으로 다시 구한다(행별 RMS, 척도 맞춘 2차 모멘트 상위 16 고유벡터).
음성: 기존의 "다른 도시" 대신 "다른 장소 묶음"의 키프레임. KITTI와 KITTI-360은 둘 다 Karlsruhe라
  기존 규칙이면 서로 음성이 되지 않는다. 묶음이 다르면 지리 감사상 50 m 안에 점이 없다.
  같은 묶음 다른 시퀀스(kitti/05-06, kth 세션끼리)는 그대로 가린다. 같은 시퀀스 규칙(> 25 m, 간격 > 150)은
  그대로. 구현: train_one이 쓰는 td.city를 장소 묶음 번호로 바꾼다.
장소 몫 상한 (조사 스크립트의 capped_weights를 상한만 바꿔 쓴다)
  arm a  상한 없음 (모든 재방문 질의를 에폭마다 앵커로)
  arm b  한 장소 40% (기존 규칙과 같은 방식: 모든 장소에 같은 예산 B, 장소 안 시퀀스에 B를 같게 나눔)

실행 (리포 루트; prep·train은 컨테이너 ring_baseline, select는 호스트)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_train_scope3_20261115.py prep
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_train_scope3_20261115.py train --arm a --tau 0.02 0.05 0.1 --lr 1e-3 3e-3 \
      --seeds 0 --epochs 40 --tag sweep
  python3 scripts/_invkey_train_scope3_20261115.py select --arm a
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_train_scope3_20261115.py train --arm a --tau <선택> --lr <선택> \
      --seeds 0 1 2 --epochs <선택> --tag final --save
  (arm b도 같다)
출력: results/invkey_study_20261114/scope3/ (stats.npz, pca_K16.npz, dev_controls.json, composition.json,
  selection.json, run_*.json, run_final_*.npz)
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import _invkey_data_survey_20261114 as SV  # noqa: E402
from _invkey_data_survey_20261114 import DIRS, capped_weights, site_of  # noqa: E402
from _invkey_train_linear_20261114 import (  # noqa: E402
    CH, DEV, EXTRA, NCH, R, SURVEY, TrainData, chmod_write, dev_groups, dev_summary, encode,
    eval_keys, load_seq, train_one)

OUT3 = REPO / "results/invkey_study_20261114/scope3"
GEO = REPO / "results/pdv2_review_closure_20261107/geography"
SCOPE = ("kitti", "kitti360", "mcd")
TRAIN = ("kitti/01", "kitti/02", "kitti/05", "kitti/06", "kitti360/0000",
         "mcd/kth_day_06", "mcd/kth_day_09", "mcd/kth_day_10",
         "mcd/kth_night_01", "mcd/kth_night_04", "mcd/kth_night_05")
DEVSET = ("kitti/00", "kitti/07", "kitti/08")
EVAL = ("kitti/09", "kitti360/0004", "kitti360/0006",
        "mcd/tuhh_day_02", "mcd/tuhh_day_03", "mcd/tuhh_day_04",
        "mcd/tuhh_night_07", "mcd/tuhh_night_08", "mcd/tuhh_night_09")
ARMS = {"a": None, "b": 0.40}


# ---------------------------------------------------------------- 분할
def split3():
    """(recs, unit_of, 점검 dict). 분할 사이 장소 묶음·지리 거리를 단언한다."""
    d = json.load(open(SURVEY))
    recs = {s["key"]: s for s in d["sequences"]}
    units = d["overlap_units"]
    unit_of = {k: u for u, ms in units.items() for k in ms}
    role = {**{k: "train" for k in TRAIN}, **{k: "dev" for k in DEVSET}, **{k: "eval" for k in EVAL}}
    assert len(role) == len(TRAIN) + len(DEVSET) + len(EVAL), "sequence in two splits"
    in_scope = sorted(k for k in recs if recs[k]["dataset"] in SCOPE)
    assert sorted(role) == in_scope, set(role) ^ set(in_scope)
    unit_role = {}
    for k, r in role.items():
        unit_role.setdefault(unit_of[k], set()).add(r)
    for u, rs in unit_role.items():
        assert len(rs) == 1, (u, rs)
        assert set(units[u]) <= set(role), u
    # 지리 감사 표: 역할이 다른 같은 도시 쌍은 모두 측정돼 있고 최소 거리 >= 50 m
    meas = {}
    for f in ("geo_same_split_pairs.csv", "geo_pairs_cross_split.csv"):
        for p in csv.DictReader(open(GEO / f)):
            if p["min_dist_m"]:
                pr = tuple(sorted((p["a"], p["b"])))
                meas[pr] = min(meas.get(pr, np.inf), float(p["min_dist_m"]))
    ks = sorted(role)
    cross = {"n_pairs_cross_role": 0, "n_same_city": 0, "n_other_city": 0, "min_dist_m": {}}
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if role[a] == role[b]:
                continue
            cross["n_pairs_cross_role"] += 1
            if recs[a]["city"] != recs[b]["city"]:
                cross["n_other_city"] += 1
                continue
            cross["n_same_city"] += 1
            pr = (a, b)
            assert pr in meas, pr
            g = "|".join(sorted((role[a], role[b])))
            cur = cross["min_dist_m"].get(g)
            if cur is None or meas[pr] < cur[0]:
                cross["min_dist_m"][g] = (meas[pr], a, b)
    for g, (dm, a, b) in cross["min_dist_m"].items():
        assert dm >= 50.0, (g, dm, a, b)
    checks = {"units_by_split": {r: sorted({unit_of[k] for k in role if role[k] == r})
                                 for r in ("train", "dev", "eval")},
              "unit_single_split": True, "geo_cross_split": cross}
    return recs, unit_of, checks


def site_weights(recs, keys, cap):
    """시퀀스별 표본 비율 w와 장소 예산 B. cap None이면 모두 1."""
    rl = [recs[k] for k in keys]
    if cap is None:
        return {k: 1.0 for k in keys}, None
    old = SV.SITE_SHARE_CAP
    SV.SITE_SHARE_CAP = cap
    try:
        return capped_weights(rl)
    finally:
        SV.SITE_SHARE_CAP = old


def planned_composition(recs, keys, w):
    """조사 표의 재방문 질의(가장 가까운 과거 짝 방향)에 표본 비율을 곱한 에폭당 기대 앵커 수."""
    out = {"site": {}, "sensor": {}, "direction_nearest": {d: 0.0 for d in DIRS}, "per_seq": {}}
    for k in keys:
        r = recs[k]
        n = w[k] * r["revisit_queries"]
        out["per_seq"][k] = {"revisit_queries": r["revisit_queries"], "fraction": w[k], "anchors": n}
        out["site"][site_of(r)] = out["site"].get(site_of(r), 0.0) + n
        out["sensor"][r["sensor"]] = out["sensor"].get(r["sensor"], 0.0) + n
        for d in DIRS:
            out["direction_nearest"][d] += w[k] * r["queries_by_dir_nearest"][d]
    tot = sum(out["site"].values())
    out["total"] = tot
    out["site_share"] = {s: v / tot for s, v in out["site"].items()}
    out["sensor_share"] = {s: v / tot for s, v in out["sensor"].items()}
    out["direction_nearest_share"] = {d: v / tot for d, v in out["direction_nearest"].items()}
    return out


def neg_rule_counts(td, seq_city, seq_unit, n_epochs=5, seed=12345):
    """에폭 배치를 따로 뽑아(학습 난수와 무관) 앵커당 후보 풀의 다른 시퀀스 키프레임 수를 규칙별로 센다."""
    rng = np.random.default_rng(seed)
    acc = {"other_city": 0.0, "other_unit": 0.0, "same_unit_other_seq_masked": 0.0,
           "same_city_other_unit_new": 0.0, "anchors": 0}
    for _ in range(n_epochs):
        batches, _ = td.epoch(rng)
        for b in batches:
            seqs = np.array([i for i, _, _ in b])
            size = np.array([2 * len(a) + EXTRA for _, a, _ in b])
            na = np.array([len(a) for _, a, _ in b])
            for c, i in enumerate(seqs):
                o = np.arange(len(seqs)) != c
                oc = seq_city[seqs[o]] != seq_city[i]
                ou = seq_unit[seqs[o]] != seq_unit[i]
                su = (seq_unit[seqs[o]] == seq_unit[i]) & (seqs[o] != i)
                acc["other_city"] += na[c] * size[o][oc].sum()
                acc["other_unit"] += na[c] * size[o][ou].sum()
                acc["same_unit_other_seq_masked"] += na[c] * size[o][su].sum()
                acc["same_city_other_unit_new"] += na[c] * size[o][ou & ~oc].sum()
                acc["anchors"] += int(na[c])
    n = acc.pop("anchors")
    return {k: v / n for k, v in acc.items()} | {"anchors_counted": n, "epochs_drawn": n_epochs}


# ---------------------------------------------------------------- prep
def prep(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    recs, unit_of, checks = split3()
    M0 = torch.zeros(R, R, dtype=torch.float64, device=DEV)
    cnt = 0
    t0 = time.time()
    for k in TRAIN:
        spec, _, _ = load_seq(k)
        for c in range(0, len(spec), 1024):
            x = torch.as_tensor(spec[c:c + 1024]).to(DEV).double()
            xr = x.permute(0, 2, 1).reshape(-1, R)
            M0 += xr.T @ xr
            cnt += xr.shape[0]
        print(f"  moments {k} n {len(spec)}  {time.time() - t0:.0f}s", flush=True)
    M0 = M0.cpu().numpy()
    s = np.sqrt(np.diag(M0) / cnt)
    M = M0 / cnt / np.outer(s, s)
    ev, V = np.linalg.eigh(M)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    chmod_write(OUT3 / "stats.npz", arrays=dict(
        s=s.astype(np.float32), s64=s, eigvals=ev, eigvecs_top64=V[:, :64], n_vectors=cnt,
        train_keys=np.array(TRAIN)))
    chmod_write(OUT3 / "pca_K16.npz", arrays=dict(W=V[:, :16].T.astype(np.float32),
                                                  s=s.astype(np.float32)))
    print(f"stats: {len(TRAIN)} seqs, {cnt} row vectors; top eigvals {ev[:5]}; "
          f"energy K16 {ev[:16].sum() / ev.sum():.4f}", flush=True)
    # 기존 stats와 비교용 수치
    old = np.load(REPO / "results/invkey_study_20261114/linear/stats.npz")
    Vo = old["eigvecs_top64"][:, :16]
    sv = np.linalg.svd(V[:, :16].T @ Vo, compute_uv=False)   # 두 16차원 부분공간의 주각 코사인
    st = torch.as_tensor(s.astype(np.float32), device=DEV)
    reps = {"raw": (None, None), "scaled": (st, None),
            "pca16": (st, torch.as_tensor(V[:, :16].T.astype(np.float32), device=DEV))}
    groups = dev_groups(recs, DEVSET)
    out = {"dev": list(DEVSET), "dev_groups": groups, "reps": {},
           "energy_K16": float(ev[:16].sum() / ev.sum()),
           "vs_old_stats": {"s_ratio_new_over_old": {"median": float(np.median(s / old["s64"])),
                                                     "min": float(np.min(s / old["s64"])),
                                                     "max": float(np.max(s / old["s64"]))},
                            "pca16_subspace_cos_principal_angles": sv.tolist(),
                            "old_energy_K16": float(old["eigvals"][:16].sum() / old["eigvals"].sum())},
           "split_checks": checks}
    per = {rep: {} for rep in reps}
    for k in DEVSET:
        spec, xyz, _ = load_seq(k)
        for rep, (ss, W) in reps.items():
            with torch.no_grad():
                D = encode(spec, ss, W)
            per[rep][k] = eval_keys(D, xyz, chunk=16 if W is None else 64)
            del D
            torch.cuda.empty_cache()
        print(f"  dev {k} done {time.time() - t0:.0f}s", flush=True)
    for rep in reps:
        out["reps"][rep] = {"per_seq": per[rep], "summary": dev_summary(per[rep], groups)}
        print(f" dev {rep:7s} {json.dumps(out['reps'][rep]['summary'])}", flush=True)
    chmod_write(OUT3 / "dev_controls.json", text=json.dumps(out, indent=1))

    # 학습 구성: 자연 구성(상한 없음)과 상한 두 가지
    comp = {"split_checks": checks, "unit_of": {k: unit_of[k] for k in TRAIN},
            "sites": {k: site_of(recs[k]) for k in TRAIN}, "arms": {}}
    for arm, cap in ARMS.items():
        w, B = site_weights(recs, TRAIN, cap)
        comp["arms"][arm] = {"cap": cap, "site_budget": B, "planned": planned_composition(recs, TRAIN, w)}
    # 참고: 넘친 장소만 줄이는 다른 방식(쓰지 않음)
    tot = {}
    for k in TRAIN:
        tot[site_of(recs[k])] = tot.get(site_of(recs[k]), 0) + recs[k]["revisit_queries"]
    big = max(tot, key=tot.get)
    rest = sum(v for s_, v in tot.items() if s_ != big)
    comp["alt_cap_only_largest_site_40pct"] = {"site": big, "anchors_kept": 0.4 / 0.6 * rest,
                                               "total": rest / 0.6}
    chmod_write(OUT3 / "composition.json", text=json.dumps(comp, indent=1))
    print(json.dumps({arm: v["planned"]["site_share"] | {"total": v["planned"]["total"]}
                      for arm, v in comp["arms"].items()}, indent=1))
    return 0


# ---------------------------------------------------------------- train
def train(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    recs, unit_of, _ = split3()
    forbidden = list(DEVSET) + list(EVAL)
    st = np.load(OUT3 / "stats.npz")
    assert list(st["train_keys"]) == list(TRAIN)
    s = torch.as_tensor(st["s"], device=DEV)
    td = TrainData(recs, list(TRAIN), forbidden)
    cap = ARMS[a.arm]
    w, B = site_weights(recs, TRAIN, cap)
    td.w = np.array([w[k] for k in TRAIN])
    td.site_budget = B
    # 음성 규칙: train_one의 "다른 도시" 판정에 장소 묶음 번호를 넣는다
    unames = sorted({unit_of[k] for k in TRAIN})
    seq_unit = np.array([unames.index(unit_of[k]) for k in TRAIN])
    cnames = sorted({recs[k]["city"] for k in TRAIN})
    seq_city = np.array([cnames.index(recs[k]["city"]) for k in TRAIN])
    td.city = torch.as_tensor(seq_unit, device=DEV)[td.seq]
    negc = neg_rule_counts(td, seq_city, seq_unit)
    dev_data = {}
    for k in DEVSET:
        spec, xyz, _ = load_seq(k)
        dev_data[k] = (torch.as_tensor(spec).to(DEV), xyz)
    groups = dev_groups(recs, DEVSET)
    print(f"arm {a.arm} cap {cap} budget {B}; train {len(TRAIN)} seqs {len(td.X)} keyframes; "
          f"w {dict(zip(TRAIN, np.round(td.w, 3)))}; neg pool per anchor {negc}", flush=True)
    W0 = torch.as_tensor(np.load(OUT3 / "pca_K16.npz")["W"], device=DEV)
    for tau in a.tau:
        for lr in a.lr:
            for seed in a.seeds:
                name = f"{a.tag}_{a.arm}_K16_tau{tau}_lr{lr}_s{seed}_e{a.epochs}"
                print(f"== {name}", flush=True)
                W, info = train_one(td, dev_data, groups, s, W0, tau, lr, a.epochs, seed, 1, forbidden)
                info["neg_per_anchor"] = {"same_seq": info["neg_per_anchor"]["same_seq"],
                                          "other_unit": info["neg_per_anchor"]["other_city"]}
                info.update({"name": name, "arm": a.arm, "cap": cap, "site_budget": B, "K": 16,
                             "tau": tau, "lr": lr, "seed": seed, "epochs": a.epochs,
                             "batch": {"chunk": CH, "chunks": NCH, "extra_per_chunk": EXTRA},
                             "optimizer": "Adam, constant lr, no weight decay",
                             "init": "scope3/pca_K16.npz", "dev": list(DEVSET), "train": list(TRAIN),
                             "eval_not_loaded": list(EVAL),
                             "negative_rule": "same seq (> 25 m and gap > 150) or other overlap unit "
                                              "(old rule: other city); same unit other seq masked",
                             "neg_pool_count_by_rule": negc, "unit_of": {k: unit_of[k] for k in TRAIN},
                             "sites": {k: site_of(recs[k]) for k in TRAIN},
                             "sensors": {k: recs[k]["sensor"] for k in TRAIN},
                             "sample_fraction": dict(zip(TRAIN, map(float, td.w)))})
                chmod_write(OUT3 / f"run_{name}.json", text=json.dumps(info, indent=1))
                if a.save:
                    chmod_write(OUT3 / f"run_{name}.npz", arrays=dict(W=W.cpu().numpy(), s=st["s"]))
    return 0


def select(a):
    """dev 곡선에서 (tau, lr, 에폭) 선택. 기준은 기존과 같다: sel_R@1_sensor_mean 최대, 같으면 AP_seq_mean."""
    path = OUT3 / "selection.json"
    sel = json.load(open(path)) if path.is_file() else {}
    cands = []
    for p in sorted(OUT3.glob(f"run_sweep_{a.arm}_K16_*.json")):
        d = json.load(open(p))
        for c in d["curve"]:
            if c["epoch"] == 0:
                continue
            sm = c["summary"]
            cands.append((sm["sel_R@1_sensor_mean"], sm["AP_seq_mean"], d["tau"], d["lr"],
                          c["epoch"], d["name"]))
    cands.sort(reverse=True)
    best = cands[0]
    sel[a.arm] = {"criterion": "max dev sel_R@1_sensor_mean (= pooled R@1 over kitti 00/07/08, one "
                               "sensor group) (tie: AP_seq_mean)", "n_runs": len({c[5] for c in cands}),
                  "tau": best[2], "lr": best[3], "epochs": best[4], "dev_sel": best[0], "dev_AP": best[1],
                  "from_run": best[5],
                  "top5": [dict(zip(("sel", "AP", "tau", "lr", "epoch", "run"), c)) for c in cands[:5]]}
    chmod_write(path, text=json.dumps(sel, indent=1))
    print(json.dumps(sel[a.arm], indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    p = sub.add_parser("select")
    p.add_argument("--arm", choices=list(ARMS), required=True)
    p = sub.add_parser("train")
    p.add_argument("--arm", choices=list(ARMS), required=True)
    p.add_argument("--tau", type=float, nargs="+", default=[0.05])
    p.add_argument("--lr", type=float, nargs="+", default=[1e-3])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--tag", default="run")
    p.add_argument("--save", action="store_true")
    a = ap.parse_args()
    return {"prep": prep, "select": select, "train": train}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
