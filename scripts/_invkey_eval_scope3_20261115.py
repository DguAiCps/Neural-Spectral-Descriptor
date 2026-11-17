#!/usr/bin/env python3
"""invkey 진단 연구: 세 데이터셋(MCD·KITTI·KITTI-360)으로 다시 학습한 K=16 선형 키를 기존 선형 16과 비교한다
(2026-11-15). 학습은 scripts/_invkey_train_scope3_20261115.py.

키 (모두 t축 원형 상관 정렬 + 코사인, nfft 128; 정의는 scripts/_invkey_train_linear_20261114.py)
  raw             spec 805x16 그대로
  pca16_new       scope3/pca_K16.npz (새 학습 11시퀀스로 구한 s·PCA, 새 학습의 초기값)
  lin16_old_s0..2 linear/run_final_K16_tau0.05_lr0.003_s*_e40.npz (기존 33시퀀스 학습; 기존 s·W 그대로)
  new_a_s0..2     scope3 arm a (장소 몫 상한 없음), selection.json의 tau·lr·에폭
  new_b_s0..2     scope3 arm b (한 장소 40% 상한)

search     평가용 9시퀀스 + 학습 쪽 mcd/kth 6개 + 개발용 3개의 검색 지표(eval_keys, 기존과 같은 chunk).
retention  kitti360/0004, mcd/tuhh_day_02: 기존 retention과 같은 질의 300개 고르기(query_pairs).
           occl_90·crop_50은 deform_check 난수 흐름(시퀀스마다 default_rng(0), 질의마다 CONDS 9개 순서대로
           deform), near_bus·all은 augment 시드 seed_for(시퀀스, scan_id, "ret_<조건>"). real은 저장 키로
           실제 재방문 j가 20 m 밖 모든 키프레임보다 높은지.
           (tuhh_day_02의 occl_90·crop_50은 기존에 재지 않았다. 같은 난수 흐름 방식을 적용한 새 값이다.)
equiv      같은 두 시퀀스에서 기존 equiv와 같은 방식(default_rng(0)으로 키프레임 100개, 스캔을 30·90도 돌려
           다시 계산한 키와의 정렬 유사도, 그 키를 지도에 넣었을 때 원래 키프레임이 20 m 밖 모두보다 높은지).
report     (호스트) scope3/metrics.json에 모든 수치·점검·선택·학습 구성을 모으고 표를 찍는다.

실행 (리포 루트)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_eval_scope3_20261115.py search
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_eval_scope3_20261115.py retention --n 300
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_eval_scope3_20261115.py equiv --n 100
  python3 scripts/_invkey_eval_scope3_20261115.py report
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _invkey_eval_linear_aug_20261115 import (  # noqa: E402
    METRICS, group_means, max_abs_diff, query_pairs, spread_of)
from _invkey_train_linear_20261114 import (  # noqa: E402
    DEV, NFFT, chmod_write, encode, eval_keys, load_seq, to_freq_t)
from _invkey_train_scope3_20261115 import DEVSET, EVAL, OUT3, TRAIN, split3  # noqa: E402

STUDY = REPO / "results/invkey_study_20261114"
LIN = STUDY / "linear"
OUTA = STUDY / "linear_aug"
OLD_RUN = "run_final_K16_tau0.05_lr0.003_s{s}_e40"
KTH = tuple(k for k in TRAIN if k.startswith("mcd/kth"))
GROUPS = {"kitti_k360": ["kitti/09", "kitti360/0004", "kitti360/0006"],
          "mcd_tuhh": [k for k in EVAL if k.startswith("mcd/tuhh")]}
SIDE = {"mcd_kth_trainside": list(KTH), "dev_kitti": list(DEVSET)}
RET_SEQS = ("kitti360/0004", "mcd/tuhh_day_02")
DEF_CONDS = ("occl_90", "crop_50")
AUG_CONDS = ("near_bus", "all")
FAM = {"lin16_old": [f"lin16_old_s{i}" for i in (0, 1, 2)],
       "new_a": [f"new_a_s{i}" for i in (0, 1, 2)],
       "new_b": [f"new_b_s{i}" for i in (0, 1, 2)]}


def new_run(arm, seed):
    sel = json.load(open(OUT3 / "selection.json"))[arm]
    return f"run_final_{arm}_K16_tau{sel['tau']}_lr{sel['lr']}_s{seed}_e{sel['epochs']}"


def load_reps(pca=True):
    """이름 -> (s, W) GPU 텐서, 파일 목록."""
    import torch
    so = np.load(LIN / "stats.npz")["s"]
    sn = np.load(OUT3 / "stats.npz")["s"]
    t = lambda x: torch.as_tensor(x, device=DEV)  # noqa: E731
    reps, files = {"raw": (None, None)}, {"raw": None}
    if pca:
        z = np.load(OUT3 / "pca_K16.npz")
        assert np.array_equal(z["s"], sn)
        reps["pca16_new"] = (t(sn), t(z["W"]))
        files["pca16_new"] = "results/invkey_study_20261114/scope3/pca_K16.npz"
    for i in (0, 1, 2):
        for name, f, s in ((f"lin16_old_s{i}", LIN / (OLD_RUN.format(s=i) + ".npz"), so),
                           (f"new_a_s{i}", OUT3 / (new_run("a", i) + ".npz"), sn),
                           (f"new_b_s{i}", OUT3 / (new_run("b", i) + ".npz"), sn)):
            z = np.load(f)
            assert np.array_equal(z["s"], s), f
            assert z["W"].shape == (16, 805), f
            reps[name] = (t(s), t(z["W"]))
            files[name] = str(f.relative_to(REPO))
    order = ["raw"] + (["pca16_new"] if pca else []) + [r for f in FAM.values() for r in f]
    return {r: reps[r] for r in order}, files


# ---------------------------------------------------------------- 검색 지표
def search(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    split3()
    reps, files = load_reps()
    seqs = list(EVAL) + list(KTH) + list(DEVSET)
    out = {"seqs": seqs, "reps": list(reps), "files": files, "per_seq": {}, "tf32": False}
    t0 = time.time()
    for k in seqs:
        spec, xyz, _ = load_seq(k)
        out["per_seq"][k] = {}
        for rep, (s, W) in reps.items():
            with torch.no_grad():
                D = encode(spec, s, W)
                out["per_seq"][k][rep] = eval_keys(D, xyz, chunk=16 if W is None else 64)
            del D
            torch.cuda.empty_cache()
        print(f"{k:22s} n {len(spec):6d}  " + "  ".join(
            f"{r} {100 * out['per_seq'][k][r]['R@1']:.1f}" for r in reps)
            + f"  {time.time() - t0:.0f}s", flush=True)
    chmod_write(OUT3 / "search_eval.json", text=json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- 1위 유지율
def retention(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from _fullmap_probe_20261114 import CONDS, _make_loader, deform
    from _invkey_augment_20261115 import augment, seed_for
    from _poolalt_probe_20261114 import aligned_sim, describe
    from _scatter_probe_20261114 import bank, occupancy
    reps, files = load_reps(pca=False)
    F = torch.from_numpy(bank()).to(DEV)
    conds = list(DEF_CONDS) + list(AUG_CONDS)
    res = {"n_requested": a.n, "files": files, "seqs": list(RET_SEQS), "conds": conds + ["real"],
           "per_seq": {}}
    t0 = time.time()
    for key in RET_SEQS:
        sensor = key.split("/")[0]
        spec, xyz64, ids = load_seq(key)
        n = len(spec)
        loader = _make_loader(*key.split("/"))
        with torch.no_grad():
            db = {r: to_freq_t(encode(spec, s, W)) for r, (s, W) in reps.items()}
        xyz = torch.as_tensor(xyz64, dtype=torch.float64, device=DEV)
        pairs = query_pairs(xyz, n, a.n)
        rng = np.random.default_rng(0)
        keep = {r: {c: [] for c in conds + ["real"]} for r in reps}
        frac = {c: [] for c in AUG_CONDS}
        for kq, (q, j) in enumerate(pairs):
            if kq % 50 == 0:
                print(f"  {key} retention {kq}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
            sid = int(ids[q])
            A = loader[sid]["points"]
            dd = {c: deform(A, c, rng) for c in CONDS}          # 9개 모두: 난수 흐름 맞춤
            X = [describe(occupancy(dd[c]), F)[1] for c in DEF_CONDS]
            for c in AUG_CONDS:
                aug, info = augment(A, np.random.default_rng(seed_for(key, sid, f"ret_{c}")), sensor, c)
                frac[c].append(info["frac_affected"])
                X.append(describe(occupancy(aug), F)[1])
            X = np.stack(X)
            neg = (xyz - xyz[q]).norm(dim=1) > 20.0
            for r, (s, W) in reps.items():
                Fdb, ndb = db[r]
                with torch.no_grad():
                    Fq, nq = to_freq_t(encode(X, s, W))
                    S = aligned_sim(Fq, nq, Fdb, ndb, NFFT)
                    S0 = aligned_sim(Fdb[:, q:q + 1], ndb[q:q + 1], Fdb, ndb, NFFT)[0]
                    negmax = S[:, neg].amax(1)
                for ci, c in enumerate(conds):
                    keep[r][c].append(bool(S[ci, q] > negmax[ci]))
                keep[r]["real"].append(bool(S0[j] > S0[neg].max()))
        res["per_seq"][key] = {
            "n_queries": len(pairs),
            "retention": {r: {c: float(np.mean(v)) for c, v in d.items()} for r, d in keep.items()},
            "aug_frac_affected_median": {c: float(np.median(v)) for c, v in frac.items()}}
        print(f"\n{key} top-1 retention over {len(pairs)} queries")
        print(" " * 13 + "".join(f"{c:>10s}" for c in conds + ["real"]))
        for r in reps:
            print(f" {r:12s}" + "".join(f"{100 * res['per_seq'][key]['retention'][r][c]:9.1f}%"
                                        for c in conds + ["real"]), flush=True)
        del db
        torch.cuda.empty_cache()
    res["elapsed_s"] = time.time() - t0
    chmod_write(OUT3 / "retention_eval.json", text=json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- 회전
def equiv(a):
    """scripts/_invkey_eval_linear_20261114.py equiv와 같은 방식(같은 난수 흐름)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from _magspec_invariance_probe_20261114 import apply, yaw_mat
    from _pd_build_test_caches import _make_loader
    from _poolalt_probe_20261114 import aligned_sim, describe
    from _scatter_probe_20261114 import bank, occupancy
    reps, files = load_reps(pca=False)
    F = torch.from_numpy(bank()).to(DEV)
    res = {"files": files}
    for key in RET_SEQS:
        spec, xyz, ids = load_seq(key)
        n = len(spec)
        loader = _make_loader(*key.split("/"))
        rng = np.random.default_rng(0)
        pick = rng.choice(n, a.n, replace=False)
        conds = ("yaw_30", "yaw_90", "roll_2", "other", "far_max")
        ranks = ("fresh", "yaw_30", "yaw_90")
        rows = {r: {c: [] for c in conds} for r in reps}
        keep = {r: {c: [] for c in ranks} for r in reps}
        with torch.no_grad():
            db = {r: to_freq_t(encode(spec, s, W)) for r, (s, W) in reps.items()}
        xyz_t = torch.as_tensor(xyz, dtype=torch.float64, device=DEV)
        for i in pick:
            pts = loader[int(ids[i])]["points"]
            d0 = describe(occupancy(pts), F)[1]
            d30 = describe(occupancy(apply(yaw_mat(30.0), pts)), F)[1]
            d90 = describe(occupancy(apply(yaw_mat(90.0), pts)), F)[1]
            o = int(rng.integers(n))
            far = (xyz_t - xyz_t[i]).norm(dim=1) > 20.0
            X = np.stack([d0, d30, d90, np.roll(d0, 2, axis=-1), spec[o].astype(np.float32)])
            for r, (s, W) in reps.items():
                with torch.no_grad():
                    Fk, nk = to_freq_t(encode(X, s, W))
                    S = aligned_sim(Fk[:, :1], nk[:1], Fk, nk, NFFT)[0].cpu().numpy()
                    Sd = aligned_sim(Fk[:, :3], nk[:3], *db[r], NFFT)
                    fm = Sd[:, far].amax(1)
                for ci, c in enumerate(conds[:4]):
                    rows[r][c].append(float(S[ci + 1]))
                rows[r]["far_max"].append(float(fm[0]))
                for ci, c in enumerate(ranks):
                    keep[r][c].append(bool(Sd[ci, i] > fm[ci]))
        res[key] = {r: {c: {"median": float(np.median(v)), "min": float(np.min(v)),
                            "max": float(np.max(v))} for c, v in d.items()} for r, d in rows.items()}
        for r in reps:
            res[key][r]["top1_retention"] = {c: float(np.mean(v)) for c, v in keep[r].items()}
        res[key]["_n"] = int(a.n)
        for r in reps:
            v = res[key][r]
            print(f"{key:18s} {r:12s} yaw_30 med {v['yaw_30']['median']:.4f} min {v['yaw_30']['min']:.4f}"
                  f"  yaw_90 med {v['yaw_90']['median']:.4f}  roll_2 min {v['roll_2']['min']:.6f}"
                  f"  far_max med {v['far_max']['median']:.4f}  keep "
                  + " ".join(f"{c} {100 * x:.0f}%" for c, x in v["top1_retention"].items()), flush=True)
        del db
        torch.cuda.empty_cache()
    chmod_write(OUT3 / "equiv_check.json", text=json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- 호스트: 모으기
def run_summary(d):
    comp = {"site": {}, "sensor": {}, "direction": {}, "per_seq": {}}
    for k, c in d["composition_total"].items():
        comp["per_seq"][k] = sum(c.values()) / d["epochs"]
        for grp, lab in (("site", d["sites"][k]), ("sensor", d["sensors"][k])):
            comp[grp][lab] = comp[grp].get(lab, 0) + sum(c.values())
        for x, v in c.items():
            comp["direction"][x] = comp["direction"].get(x, 0) + v
    tot = sum(comp["direction"].values())
    curve = [{"epoch": c["epoch"], "loss": c["loss"], **c["summary"],
              "per_seq_R@1": {k: v["R@1"] for k, v in c["per_seq"].items()}} for c in d["curve"]]
    return {k: d[k] for k in ("arm", "cap", "site_budget", "tau", "lr", "seed", "epochs", "neg_per_anchor",
                              "neg_pool_count_by_rule", "leak_check", "train_s", "sample_fraction")} | {
        "composition": {"anchors_per_epoch": tot / d["epochs"],
                        "site_share": {s: v / tot for s, v in comp["site"].items()},
                        "sensor_share": {s: v / tot for s, v in comp["sensor"].items()},
                        "direction_share_sampled_pairs": {x: v / tot for x, v in comp["direction"].items()},
                        "anchors_per_epoch_by_seq": comp["per_seq"]},
        "dev_curve": curve, "step0_dev": d["step0_dev"]}


def report(a):
    recs, unit_of, split_checks = split3()
    S = json.load(open(OUT3 / "search_eval.json"))
    RT = json.load(open(OUT3 / "retention_eval.json"))
    EQ = json.load(open(OUT3 / "equiv_check.json"))
    dc = json.load(open(OUT3 / "dev_controls.json"))
    comp = json.load(open(OUT3 / "composition.json"))
    sel = json.load(open(OUT3 / "selection.json"))
    OLDM = json.load(open(LIN / "metrics.json"))
    old_dc = json.load(open(LIN / "dev_controls.json"))
    old_rt = json.load(open(OUTA / "retention_eval.json"))
    old_eq = json.load(open(LIN / "equiv_check.json"))
    per = S["per_seq"]
    reps = S["reps"]
    G = {**GROUPS, **SIDE}
    gm = group_means(per, G, reps)
    keys_m = METRICS + ("R@1_same_pooled", "R@1_reverse_pooled", "R@1_crossing_pooled", "R@1_pooled")
    spread = {f: {g: {m: spread_of([gm[g][r][m] for r in rs if gm[g][r][m] is not None])
                      if any(gm[g][r][m] is not None for r in rs) else None for m in keys_m}
                  for g in G} for f, rs in FAM.items()}
    per_seq_spread = {f: {k: {m: spread_of([per[k][r][m] for r in rs]) for m in METRICS}
                          for k in per} for f, rs in FAM.items()}
    # 학습 실행
    runs = {}
    raw_runs = {}
    for p in sorted(glob.glob(str(OUT3 / "run_*.json"))):
        d = json.load(open(p))
        raw_runs[d["name"]] = d
        runs[d["name"]] = run_summary(d)
    # 점검
    pk = dc["reps"]["pca16"]["per_seq"]
    c_step0 = {n: all(d["step0_dev"][k][m] == pk[k][m] for k in pk for m in ("R@1", "AP", "R@100P"))
               for n, d in raw_runs.items()}
    c_leak = {n: {"forbidden_touched": d["leak_check"]["forbidden_touched"],
                  "touched_subset_of_train": set(d["leak_check"]["sequences_touched"]) <= set(TRAIN),
                  "not_touched": sorted(set(TRAIN) - set(d["leak_check"]["sequences_touched"]))}
              for n, d in raw_runs.items()}
    # 최종 s0와 같은 설정 sweep s0의 곡선이 같은지(결정성)
    c_det = {}
    for arm in ("a", "b"):
        sv = sel[arm]
        fin = raw_runs[new_run(arm, 0)[4:]]
        sw = raw_runs[f"sweep_{arm}_K16_tau{sv['tau']}_lr{sv['lr']}_s0_e40"]
        ne = sv["epochs"] + 1
        dd, nu = max_abs_diff([c["per_seq"] for c in fin["curve"]], [c["per_seq"] for c in sw["curve"][:ne]])
        dl = max(abs(u["loss"] - v["loss"]) for u, v in zip(fin["curve"][1:], sw["curve"][1:ne]))
        c_det[arm] = {"per_seq_max_abs_diff": dd, "n_unequal": nu, "loss_max_abs_diff": dl}
    # 기존 수치 재현: raw·기존 선형 16
    c_old = {}
    for r_new, r_old in [("raw", "raw")] + [(f"lin16_old_s{i}", f"lin16_s{i}") for i in (0, 1, 2)]:
        ks = list(EVAL) + list(KTH)
        dmax, nu = max_abs_diff({k: OLDM["heldout_per_seq"][k][r_old] for k in ks},
                                {k: per[k][r_new] for k in ks})
        c_old[f"{r_new}|eval+kth_vs_linear_metrics"] = {"max_abs_diff": dmax, "n_unequal": nu}
    dmax, nu = max_abs_diff({k: old_dc["reps"]["raw"]["per_seq"][k] for k in DEVSET},
                            {k: per[k]["raw"] for k in DEVSET})
    c_old["raw|dev_vs_linear_dev_controls"] = {"max_abs_diff": dmax, "n_unequal": nu}
    for i in (0, 1, 2):
        od = json.load(open(LIN / (OLD_RUN.format(s=i) + ".json")))["curve"][-1]["per_seq"]
        dmax, nu = max_abs_diff({k: od[k] for k in DEVSET},
                                {k: {m: per[k][f"lin16_old_s{i}"][m] for m in od[k]} for k in DEVSET})
        c_old[f"lin16_old_s{i}|dev_vs_old_run_epoch40"] = {"max_abs_diff": dmax, "n_unequal": nu}
    c_ret = {}
    for key in RET_SEQS:
        o = old_rt["per_seq"][key]
        assert o["n_queries"] == RT["per_seq"][key]["n_queries"], key
        for r_new, r_old in [("raw", "raw")] + [(f"lin16_old_s{i}", f"lin16_s{i}") for i in (0, 1, 2)]:
            cs = [c for c in RT["conds"] if c in o["conds"] + ["real"]]
            c_ret[f"{key}|{r_new}"] = {c: [o["retention"][r_old][c], RT["per_seq"][key]["retention"][r_new][c]]
                                       for c in cs}
    c_ret_ok = all(v[0] == v[1] for d in c_ret.values() for v in d.values())
    c_eq = {}
    for r_new, r_old in [("raw", "raw")] + [(f"lin16_old_s{i}", f"lin16_s{i}") for i in (0, 1, 2)]:
        dmax, nu = max_abs_diff(old_eq["kitti360/0004"][r_old], EQ["kitti360/0004"][r_new])
        c_eq[r_new] = {"max_abs_diff": dmax, "n_unequal": nu}
    # 개발용 곡선 요약(과적합 보기)
    curves = {}
    for n, d in runs.items():
        cv = d["dev_curve"]
        best = max(cv[1:], key=lambda c: (c["sel_R@1_sensor_mean"], c["AP_seq_mean"]))
        curves[n] = {"ep0_sel": cv[0]["sel_R@1_sensor_mean"], "best_epoch": best["epoch"],
                     "best_sel": best["sel_R@1_sensor_mean"], "last_sel": cv[-1]["sel_R@1_sensor_mean"],
                     "last_AP": cv[-1]["AP_seq_mean"], "loss_first": cv[1]["loss"], "loss_last": cv[-1]["loss"],
                     "anchors_per_epoch": d["composition"]["anchors_per_epoch"]}
    # 이득: 학습 쪽(kth)·개발용 대 평가용
    gain = {}
    for f, rs in FAM.items():
        gain[f] = {g: {"R@1_pooled": spread[f][g]["R@1_pooled"],
                       "minus_raw": spread[f][g]["R@1_pooled"]["mean"] - gm[g]["raw"]["R@1_pooled"],
                       "minus_pca16_new": spread[f][g]["R@1_pooled"]["mean"] - gm[g]["pca16_new"]["R@1_pooled"]}
                   for g in G}
    ret = {}
    for key in RET_SEQS:
        rr = RT["per_seq"][key]["retention"]
        ret[key] = {"n_queries": RT["per_seq"][key]["n_queries"], "raw": rr["raw"]}
        for f, rs in FAM.items():
            ret[key][f] = {c: spread_of([rr[r][c] for r in rs]) for c in RT["conds"]}
    eqs = {}
    for key in RET_SEQS:
        e = EQ[key]
        eqs[key] = {"raw": {"yaw_30_median": e["raw"]["yaw_30"]["median"], "yaw_30_min": e["raw"]["yaw_30"]["min"],
                            "far_max_median": e["raw"]["far_max"]["median"], **e["raw"]["top1_retention"]}}
        for f, rs in FAM.items():
            eqs[key][f] = {"yaw_30_median": spread_of([e[r]["yaw_30"]["median"] for r in rs]),
                           "yaw_30_min": spread_of([e[r]["yaw_30"]["min"] for r in rs]),
                           "far_max_median": spread_of([e[r]["far_max"]["median"] for r in rs]),
                           **{c: spread_of([e[r]["top1_retention"][c] for r in rs])
                              for c in ("fresh", "yaw_30", "yaw_90")}}
    out = {"date": "2026-11-15",
           "scope": "MCD, KITTI, KITTI-360 only (HeLiPR, MulRan, NCLT excluded: azimuth blind sectors of rigs)",
           "split": {"train": list(TRAIN), "dev": list(DEVSET), "eval": list(EVAL)},
           "config": {"K": 16, "grid": {"tau": [0.02, 0.05, 0.1], "lr": [1e-3, 3e-3]}, "epochs_sweep": 40,
                      "seed_sweep": 0, "seeds_final": [0, 1, 2], "arms": {"a": "no site cap", "b": "site cap 0.40"},
                      "s_pca": "recomputed on the 11 new train sequences", "tf32": False,
                      "negatives": "same seq (> 25 m, gap > 150) or other overlap unit; same unit other seq masked "
                                   "(old: other city)",
                      "train_script": "scripts/_invkey_train_scope3_20261115.py",
                      "eval_script": "scripts/_invkey_eval_scope3_20261115.py"},
           "composition": comp, "selection": sel, "dev_controls": dc, "dev_curve_summary": curves,
           "groups": G, "keys": S["files"], "per_seq": per, "group_means": gm, "seed_spread": spread,
           "per_seq_seed_spread": per_seq_spread, "gain_trainside_dev_vs_eval": gain,
           "retention": RT, "retention_summary": ret, "equiv": EQ, "equiv_summary": eqs,
           "checks": {"split": split_checks, "step0_equals_new_pca16_on_dev": c_step0, "leak": c_leak,
                      "final_s0_equals_sweep_s0_prefix": c_det, "reproduce_old_numbers": c_old,
                      "retention_reproduces_linear_aug": {"all_equal": c_ret_ok, "pairs": c_ret},
                      "equiv_kitti360_0004_reproduces_linear": c_eq},
           "runs": runs}
    chmod_write(OUT3 / "metrics.json", text=json.dumps(out, indent=1))

    # ---- 표
    f = lambda x: "–" if x is None else f"{100 * x:.1f}"  # noqa: E731
    print("checks step0", all(c_step0.values()), "leak", {n: (v["forbidden_touched"], v["not_touched"])
                                                           for n, v in list(c_leak.items())[:1]})
    print("det", c_det)
    print("old", c_old)
    print("ret ok", c_ret_ok, "equiv", c_eq)
    print("\n## dev curves")
    for n, c in curves.items():
        print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    print("\n## group means (seed mean [min–max])")
    cols = ("R@1_pooled", "R@1", "R@100", "AP", "R@100P", "R@100P_10m", "R@1_same_pooled",
            "R@1_reverse_pooled", "R@1_crossing_pooled")
    for g in G:
        print(f"\n### {g} ({len(G[g])}) n_pos {sum(per[k]['raw']['n_pos'] for k in G[g])} dirs "
              f"{ {d: gm[g]['raw'][f'n_{d}'] for d in ('same', 'reverse', 'crossing')} }\n")
        print("| 키 | " + " | ".join(cols) + " |")
        print("|---|" + "---:|" * len(cols))
        for r in ("raw", "pca16_new"):
            print(f"| {r} | " + " | ".join(f(gm[g][r][m]) for m in cols) + " |")
        for fn in FAM:
            print(f"| {fn} | " + " | ".join(
                "–" if spread[fn][g][m] is None else
                f"{f(spread[fn][g][m]['mean'])} [{f(spread[fn][g][m]['min'])}–{f(spread[fn][g][m]['max'])}]"
                for m in cols) + " |")
    print("\n## per seq")
    for m in ("R@1", "AP", "R@100P", "R@100P_10m"):
        print(f"\n### {m}\n")
        print("| 시퀀스 | 양성 | raw | pca16_new | " + " | ".join(FAM) + " |")
        print("|---|---:|---:|---:|" + "---:|" * len(FAM))
        for k in per:
            print(f"| {k} | {per[k]['raw']['n_pos']} | {f(per[k]['raw'][m])} | {f(per[k]['pca16_new'][m])} | "
                  + " | ".join(f"{f(per_seq_spread[fn][k][m]['mean'])} "
                               f"[{f(per_seq_spread[fn][k][m]['min'])}–{f(per_seq_spread[fn][k][m]['max'])}]"
                               for fn in FAM) + " |")
    print("\n## retention")
    for key, d in ret.items():
        print(f"\n### {key} ({d['n_queries']})\n")
        print("| 키 | " + " | ".join(RT["conds"]) + " |")
        print("|---|" + "---:|" * len(RT["conds"]))
        print("| raw | " + " | ".join(f(d["raw"][c]) for c in RT["conds"]) + " |")
        for fn in FAM:
            print(f"| {fn} | " + " | ".join(f"{f(d[fn][c]['mean'])} [{f(d[fn][c]['min'])}–{f(d[fn][c]['max'])}]"
                                            for c in RT["conds"]) + " |")
    print("\n## equiv")
    for key, d in eqs.items():
        print(f"\n### {key}\n")
        for r, v in d.items():
            print(r, {k: (round(x, 4) if isinstance(x, float) else {kk: round(xx, 4) for kk, xx in x.items()})
                      for k, x in v.items()})
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("search")
    p = sub.add_parser("retention")
    p.add_argument("--n", type=int, default=300)
    p = sub.add_parser("equiv")
    p.add_argument("--n", type=int, default=100)
    sub.add_parser("report")
    a = ap.parse_args()
    return {"search": search, "retention": retention, "equiv": equiv, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
