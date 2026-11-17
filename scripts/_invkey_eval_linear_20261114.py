#!/usr/bin/env python3
"""invkey 진단 연구: 학습한 선형 키와 대조군을 held-out 21시퀀스에서 재고, 회전 점검·변형 점검을 한다 (2026-11-14).

키 (모두 t축 원형 상관 정렬 + 코사인, nfft 128. 정의는 scripts/_invkey_train_linear_20261114.py):
  raw      spec 805x16 그대로
  scaled   spec / s (행 척도, 학습 없음)
  pca16/64 W = 척도 맞춘 행의 2차 모멘트 상위 K 고유벡터 (학습 없음, 학습의 초기값)
  lin*     학습한 W (run_*.npz)
검색·PR 회계는 scripts/_poolalt_probe_20261114.py main()과 같다(질의 q >= 30, 후보 j <= q-30, 양성 GT <= 5 m,
거리는 float64 직접 차). mulran/Sejong02의 마지막 키프레임(전부 0)은 질의·지도에서 뺀다.

실행 (리포 루트):
  RUNS16="lin16_s0=run_final_K16_tau0.05_lr0.003_s0_e40.npz lin16_s1=run_final_K16_tau0.05_lr0.003_s1_e40.npz \
    lin16_s2=run_final_K16_tau0.05_lr0.003_s2_e40.npz"
  RUNS64="lin64_s0=run_final_K64_tau0.02_lr0.003_s0_e27.npz lin64_s1=run_final_K64_tau0.02_lr0.003_s1_e27.npz"
  # held-out 평가 (컨테이너). --runs 이름=파일 여러 개
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_eval_linear_20261114.py heldout --runs $RUNS16 $RUNS64
  # 점검 3: 스캔을 30도·90도 돌려 다시 계산한 키, 서술자를 2칸 순환 이동한 키와 원래 키의 정렬 유사도 (컨테이너)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_eval_linear_20261114.py equiv --n 20 --runs $RUNS16 $RUNS64
  # 변형 점검: scripts/_fullmap_probe_20261114.py (2)와 같은 질의·변형 (컨테이너), raw·pca16·학습 16
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_eval_linear_20261114.py deform --n 300 --runs $RUNS16
  # 모아서 metrics.json과 보고서 표 (호스트)
  python3 scripts/_invkey_eval_linear_20261114.py report
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

from _invkey_train_linear_20261114 import (  # noqa: E402
    DEV, NFFT, OUTD, REFS, chmod_write, encode, eval_keys, load_seq, split, to_freq_t)

MCD = "mcd/"
METRICS = ("R@1", "R@100", "AP", "max_F1", "R@100P", "R@100P_10m")


def parse_runs(items):
    out = {}
    for it in items or []:
        name, f = it.split("=", 1)
        out[name] = OUTD / f
    return out


def load_reps(runs, extra=False):
    """이름 -> (s, W) GPU 텐서. raw는 (None, None)."""
    import torch
    st = np.load(OUTD / "stats.npz")
    s = torch.as_tensor(st["s"], device=DEV)
    reps = {"raw": (None, None), "scaled": (s, None)}
    for K in (16, 64):
        z = np.load(OUTD / f"pca_K{K}.npz")
        assert np.array_equal(z["s"], st["s"])
        reps[f"pca{K}"] = (s, torch.as_tensor(z["W"], device=DEV))
    if extra:
        # 추가 대조군(요청 밖): 첫 주성분(공통 성분, 평균 방향에 가까움)을 뺀 2-17번째 고유벡터
        V = st["eigvecs_top64"]
        reps["pca16_d1"] = (s, torch.as_tensor(V[:, 1:17].T.astype(np.float32), device=DEV))
    for name, f in runs.items():
        z = np.load(f)
        assert np.array_equal(z["s"], st["s"]), f
        reps[name] = (s, torch.as_tensor(z["W"], device=DEV))
    return reps


# ---------------------------------------------------------------- held-out
def heldout(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    recs, train, dev, held = split()
    reps = load_reps(parse_runs(a.runs), extra=True)
    out = {"heldout": held, "reps": list(reps), "runs": a.runs, "per_seq": {}, "tf32": False}
    t0 = time.time()
    for k in held:
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
    # 점검 1의 참고: 컨테이너 기본값(TF32 켬)으로 raw를 기준 3시퀀스에서 다시 잰다
    torch.backends.cuda.matmul.allow_tf32 = True
    out["check1_tf32_on_raw"] = {}
    for k in REFS:
        spec, xyz, _ = load_seq(k)
        with torch.no_grad():
            out["check1_tf32_on_raw"][k] = eval_keys(encode(spec), xyz, chunk=16)
        torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    chmod_write(OUTD / "heldout_eval.json", text=json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- 점검 3: 회전
def equiv(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from _magspec_invariance_probe_20261114 import apply, yaw_mat
    from _pd_build_test_caches import _make_loader
    from _poolalt_probe_20261114 import aligned_sim, describe
    from _scatter_probe_20261114 import bank, occupancy
    reps = load_reps(parse_runs(a.runs))
    F = torch.from_numpy(bank()).to(DEV)
    res = {}
    for key in REFS:
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
            # roll_2: 서술자 자체를 방향 축으로 2칸(22.5도) 순환 이동 -> 모델의 순환 등변성만 본다
            X = np.stack([d0, d30, d90, np.roll(d0, 2, axis=-1), spec[o].astype(np.float32)])
            for r, (s, W) in reps.items():
                with torch.no_grad():
                    Fk, nk = to_freq_t(encode(X, s, W))
                    S = aligned_sim(Fk[:, :1], nk[:1], Fk, nk, NFFT)[0].cpu().numpy()
                    # 순위: 새로 계산한 키(원본·30도·90도)를 저장 지도에 넣어, 지도의 원래 키프레임 i가
                    # 20 m 밖 모든 키프레임보다 높은지 (변형 점검과 같은 1위 유지 정의)
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
            print(f"{key:22s} {r:10s} yaw_30 med {v['yaw_30']['median']:.4f} min {v['yaw_30']['min']:.4f}"
                  f"  yaw_90 med {v['yaw_90']['median']:.4f}  roll_2 min {v['roll_2']['min']:.6f}"
                  f"  other med {v['other']['median']:.4f} max {v['other']['max']:.4f}"
                  f"  far_max med {v['far_max']['median']:.4f}  keep "
                  + " ".join(f"{c} {100 * x:.0f}%" for c, x in v["top1_retention"].items()), flush=True)
        del db
        torch.cuda.empty_cache()
    chmod_write(OUTD / "equiv_check.json", text=json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- 변형 점검
def deform_check(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from _fullmap_probe_20261114 import CONDS, _make_loader, deform
    from _poolalt_probe_20261114 import aligned_sim, describe
    from _scatter_probe_20261114 import bank, occupancy
    allreps = load_reps(parse_runs(a.runs))
    reps = {r: v for r, v in allreps.items() if r in a.reps}
    F = torch.from_numpy(bank()).to(DEV)
    res = {}
    t0 = time.time()
    for key in REFS:
        spec, xyz64, ids = load_seq(key)
        n = len(spec)
        loader = _make_loader(*key.split("/"))
        db = {}
        with torch.no_grad():
            for r, (s, W) in reps.items():
                db[r] = to_freq_t(encode(spec, s, W))
        xyz = torch.as_tensor(xyz64, dtype=torch.float64, device=DEV)
        pairs = []
        for q in range(30, n - 1):
            d = (xyz[: q - 29] - xyz[q]).norm(dim=1)
            j = int(d.argmin())
            if d[j] <= 5.0 and (d > 20.0).any():
                pairs.append((q, j))
        if len(pairs) > a.n:
            pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, a.n).astype(int)]
        rng = np.random.default_rng(0)
        keep = {r: {c: [] for c in CONDS + ("real", "adjacent")} for r in reps}
        for kq, (q, j) in enumerate(pairs):
            if kq % 50 == 0:
                print(f"  {key} deform {kq}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
            A = loader[int(ids[q])]["points"]
            X = np.stack([describe(occupancy(deform(A, c, rng)), F)[1] for c in CONDS])
            neg = (xyz - xyz[q]).norm(dim=1) > 20.0
            for r, (s, W) in reps.items():
                Fdb, ndb = db[r]
                with torch.no_grad():
                    Fq, nq = to_freq_t(encode(X, s, W))
                    S = aligned_sim(Fq, nq, Fdb, ndb, NFFT)
                    S0 = aligned_sim(Fdb[:, q:q + 1], ndb[q:q + 1], Fdb, ndb, NFFT)[0]
                for ci, c in enumerate(CONDS):
                    keep[r][c].append(bool(S[ci, q] > S[ci][neg].max()))
                keep[r]["real"].append(bool(S0[j] > S0[neg].max()))
                keep[r]["adjacent"].append(bool(S0[q + 1] > S0[neg].max()))
        res[key] = {"n_queries": len(pairs),
                    "retention": {r: {c: float(np.mean(v)) for c, v in d.items()}
                                  for r, d in keep.items()}}
        print(f"\n{key} top-1 retention over {len(pairs)} queries")
        print(" " * 11 + "".join(f"{c:>10s}" for c in CONDS + ("real", "adjacent")))
        for r in reps:
            print(f" {r:10s}" + "".join(f"{100 * res[key]['retention'][r][c]:9.1f}%"
                                        for c in CONDS + ("real", "adjacent")), flush=True)
        del db
        torch.cuda.empty_cache()
    chmod_write(OUTD / "deform_check.json", text=json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- 호스트: 모으기
def groups_of(held, train_sensors, recs):
    return {"ref3": list(REFS),
            "seen_sensor": [k for k in held if recs[k]["sensor"] in train_sensors],
            "unseen_mcd": [k for k in held if k.startswith(MCD)]}


def report(a):
    recs, train, dev, held = split()
    H = json.load(open(OUTD / "heldout_eval.json"))
    dc = json.load(open(OUTD / "dev_controls.json"))
    train_sensors = {recs[k]["sensor"] for k in train}
    G = groups_of(held, train_sensors, recs)
    reps = H["reps"]
    per = H["per_seq"]
    gm = {}
    for g, ks in G.items():
        gm[g] = {}
        for r in reps:
            gm[g][r] = {m: float(np.mean([per[k][r][m] for k in ks])) for m in METRICS}
            for d in ("same", "reverse", "crossing"):
                nn = sum(per[k][r]["dir_n"][d] for k in ks)
                hh = sum(per[k][r]["dir_hit1"][d] for k in ks)
                gm[g][r][f"R@1_{d}_pooled"] = hh / nn if nn else None
                gm[g][r][f"n_{d}"] = nn
            nn = sum(per[k][r]["n_pos"] for k in ks)
            gm[g][r]["R@1_pooled"] = sum(per[k][r]["n_hit1"] for k in ks) / nn
    # 학습 seed 묶음
    fam = {}
    for r in reps:
        if r.startswith("lin"):
            fam.setdefault(r.split("_")[0], []).append(r)
    spread = {}
    for f, rs in fam.items():
        spread[f] = {g: {m: {"mean": float(np.mean([gm[g][r][m] for r in rs])),
                             "min": float(np.min([gm[g][r][m] for r in rs])),
                             "max": float(np.max([gm[g][r][m] for r in rs])),
                             "seeds": rs}
                         for m in METRICS + ("R@1_same_pooled", "R@1_reverse_pooled")}
                     for g in G}
    runs = {}
    for p in sorted(glob.glob(str(OUTD / "run_*.json"))):
        d = json.load(open(p))
        runs[d["name"]] = {k: d[k] for k in ("K", "tau", "lr", "seed", "epochs", "neg_per_anchor",
                                             "leak_check", "train_s")}
        runs[d["name"]]["dev_curve"] = [{"epoch": c["epoch"], "loss": c["loss"], **c["summary"]}
                                        for c in d["curve"]]
        runs[d["name"]]["step0_dev"] = d["step0_dev"]
        comp = {"site": {}, "sensor": {}, "direction": {}}
        for k, c in d["composition_total"].items():
            for grp, lab in (("site", d["sites"][k]), ("sensor", d["sensors"][k])):
                e = comp[grp].setdefault(lab, {x: 0 for x in c})
                for x, v in c.items():
                    e[x] += v
            for x, v in c.items():
                comp["direction"][x] = comp["direction"].get(x, 0) + v
        tot = sum(comp["direction"].values())
        runs[d["name"]]["composition"] = {
            "anchors_total_all_epochs": tot,
            "anchors_per_epoch": tot / d["epochs"],
            "direction_share": {x: v / tot for x, v in comp["direction"].items()},
            "site_share": {s: sum(v.values()) / tot for s, v in comp["site"].items()},
            "sensor_share": {s: sum(v.values()) / tot for s, v in comp["sensor"].items()},
            "by_site": comp["site"], "by_sensor": comp["sensor"]}
    out = {"date": "2026-11-14", "split": "data_survey.json candidates.A", "dev": dev,
           "train": train, "heldout": held, "groups": G, "heldout_per_seq": per,
           "heldout_group_means": gm, "learned_seed_spread": spread,
           "check1_tf32_on_raw": H.get("check1_tf32_on_raw"), "dev_controls": dc,
           "runs": runs}
    for f in ("equiv_check.json", "deform_check.json", "selection.json"):
        if (OUTD / f).is_file():
            out[f[:-5]] = json.load(open(OUTD / f))
    # 점검 요약
    ref_exp = {"kitti360/0004": (0.9577464788732394, 0.9994582881906825),
               "helipr/Roundabout01": (0.6472361809045226, 0.9723618090452261),
               "mulran/Sejong03": (0.9279393173198483, 1.0)}   # results/rawmag_probe_20261114/poolalt_*.json
    c1 = {k: {"R@1": per[k]["raw"]["R@1"], "R@100": per[k]["raw"]["R@100"],
              "expected": ref_exp[k], "n_pos": per[k]["raw"]["n_pos"],
              "query_diff_R@1": round((per[k]["raw"]["R@1"] - ref_exp[k][0]) * per[k]["raw"]["n_pos"]),
              "query_diff_R@1_tf32_on": round((H["check1_tf32_on_raw"][k]["R@1"] - ref_exp[k][0])
                                              * per[k]["raw"]["n_pos"])} for k in REFS}
    c2 = {}
    for name, d in runs.items():
        pk = dc["reps"][f"pca{d['K']}"]["per_seq"]
        c2[name] = all(d["step0_dev"][k][m] == pk[k][m] for k in pk for m in ("R@1", "AP", "R@100P"))
    c4 = {name: {"forbidden_touched": d["leak_check"]["forbidden_touched"],
                 "n_sequences_touched": len(d["leak_check"]["sequences_touched"])}
          for name, d in runs.items()}
    out["checks"] = {"check1_raw_reference": c1, "check2_step0_equals_pca_on_dev": c2,
                     "check3": "see equiv_check", "check4_leak": c4}
    (OUTD / "metrics.json").write_text(json.dumps(out, indent=1))
    # 표
    print("## group means")
    for g in G:
        print(f"\n### {g} ({len(G[g])})\n")
        print("| 키 | " + " | ".join(METRICS) + " | R@1 같은 | R@1 반대 |")
        print("|---|" + "---:|" * (len(METRICS) + 2))
        for r in reps:
            v = gm[g][r]
            f = lambda x: "–" if x is None else f"{100 * x:.1f}"  # noqa: E731
            print(f"| {r} | " + " | ".join(f(v[m]) for m in METRICS)
                  + f" | {f(v['R@1_same_pooled'])} | {f(v['R@1_reverse_pooled'])} |")
    print("\n## per seq R@1 / AP / R@100P")
    for m in ("R@1", "AP", "R@100P"):
        print(f"\n### {m}\n")
        print("| 시퀀스 | 양성 | " + " | ".join(reps) + " |")
        print("|---|---:|" + "---:|" * len(reps))
        for k in held:
            print(f"| {k} | {per[k]['raw']['n_pos']} | "
                  + " | ".join(f"{100 * per[k][r][m]:.1f}" for r in reps) + " |")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("heldout", "equiv", "deform"):
        p = sub.add_parser(c)
        p.add_argument("--runs", nargs="*", default=[])
        p.add_argument("--n", type=int, default=300 if c == "deform" else 20)
        p.add_argument("--reps", nargs="*", default=["raw", "pca16"])
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "heldout":
        return heldout(a)
    if a.cmd == "equiv":
        return equiv(a)
    if a.cmd == "deform":
        a.reps = list(a.reps) + [n.split("=")[0] for n in a.runs]
        return deform_check(a)
    return report(a)


if __name__ == "__main__":
    sys.exit(main())
