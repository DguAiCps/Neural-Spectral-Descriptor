#!/usr/bin/env python3
"""invkey 진단 연구: 주파수 등변 신경망(단계 1-3)을 선형 키·raw와 비교한다 (2026-11-15).

키 (모두 t축 원형 상관 정렬 + 코사인, nfft 128)
  raw            spec 805x16 그대로
  pca16          linear/pca_K16.npz (학습 없음; trainside에서만)
  lin16_s0..2    깨끗하게 학습한 선형 W (linear/run_final_K16_tau0.05_lr0.003_s*_e40.npz)
  aug16_s0..2    증강으로 학습한 선형 W (linear_aug/selection.json의 p·에폭)
  <단계>_s0..2   equiv/run_final_<단계>_*.npz (selection.json의 lr·에폭; 단계 목록은 best_stage.rows)
  <최선>aug_s0..2  equiv/run_finalaug_<최선 단계>_*_p0.25_*.npz
명령 (컨테이너, 리포 루트)
  heldout    held-out 21시퀀스 검색 지표 (scripts/_invkey_eval_linear_20261114.py heldout과 같은 프로토콜·chunk)
  trainside  학습 시퀀스 4개(센서마다 학습 목록 순서로 처음 나오는, 재방문 질의 1000개 이상·키프레임 1만 미만인
             것: kitti360/0000, helipr/Town01, mulran/KAIST01, nclt/2012-01-08)에서 같은 지표
  equiv      점검 3. (a)(c) scripts/_invkey_eval_linear_20261114.py equiv와 같은 고르기(기준 3시퀀스, default_rng(0)
             choice 100, 스캔마다 rng.integers 하나): 다시 계산한 서술자와 30도·90도 회전 스캔, 서술자 2칸 순환 이동,
             임의 키프레임의 정렬 유사도, 20 m 밖 최댓값, 1위 유지(fresh·yaw_30·yaw_90).
             (b) 같은 100개 저장 서술자의 주파수 입력 X[r, k]에 e^{-i 2 pi k d / 16}(d = 칸 사이 이동)를 곱해
             forward_freq에 넣고, 출력이 같은 위상만큼 도는지 상대 오차 ||Y(X e) - Y(X) e|| / ||Y(X)||.
             대조: st1 가중치에 시간 영역 점별 ReLU를 끼운 모델(등변이 아님).
  retention  scripts/_invkey_eval_linear_aug_20261115.py retention과 같은 질의 300개·같은 변형 난수 흐름. 조건
             obj_mix, near_bus, range_r40, all, occl_90, crop_50, crop_30, real (기준 3시퀀스).
  report     (호스트) 모든 수치·점검·선택·학습 구성을 equiv/metrics.json에 모으고 표를 찍는다.
실행
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_invkey_eval_equiv_20261115.py heldout
  docker exec ... trainside / equiv / retention
  python3 scripts/_invkey_eval_equiv_20261115.py report
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
    DEV, NFFT, OUTD as LIN, REFS, T, chmod_write, encode, eval_keys, load_seq, split, to_freq_t)
from _invkey_eval_linear_aug_20261115 import (  # noqa: E402
    METRICS, OUTA, aug_run_name, group_means, groups_of, max_abs_diff, query_pairs, spread_of)
from _invkey_train_equiv_20261115 import NF, OUTE, TAU  # noqa: E402

TRAIN_SIDE = ("kitti360/0000", "helipr/Town01", "mulran/KAIST01", "nclt/2012-01-08")
OLD_USE = ("occl_90", "crop_50", "crop_30")
NEW_USE = ("obj_mix", "near_bus", "range_r40", "all")
RET_CONDS = NEW_USE + OLD_USE
SHIFTS = (0.25, 0.37, 0.5, 1.5, 3.71, 7.5)          # 칸 (1칸 = 11.25도)
KEYS_M = METRICS + ("R@1_same_pooled", "R@1_reverse_pooled", "R@1_crossing_pooled", "R@1_pooled")


# ---------------------------------------------------------------- 키 목록
def stage_info():
    sel = json.load(open(OUTE / "selection.json"))
    stages = list(sel["best_stage"]["rows"])
    best = sel["best_stage"]["stage"]
    return sel, stages, best


def rep_files():
    sel, stages, best = stage_info()
    files = {"raw": None, "pca16": LIN / "pca_K16.npz"}
    for i in (0, 1, 2):
        files[f"lin16_s{i}"] = LIN / f"run_final_K16_tau0.05_lr0.003_s{i}_e40.npz"
    for i in (0, 1, 2):
        files[f"aug16_s{i}"] = OUTA / (aug_run_name(i) + ".npz")
    for st in stages:
        s = sel[st]
        for i in (0, 1, 2):
            files[f"{st}_s{i}"] = OUTE / f"run_final_{st}_tau{TAU}_lr{s['lr']}_s{i}_e{s['epochs']}.npz"
    s = sel[best]
    for i in (0, 1, 2):
        files[f"{best}aug_s{i}"] = OUTE / f"run_finalaug_{best}_tau{TAU}_lr{s['lr']}_p0.25_s{i}_e{s['epochs']}.npz"
    return files


def families(names):
    fam = {}
    for n in names:
        if n in ("raw", "pca16"):
            fam[n] = [n]
        else:
            fam.setdefault(n.rsplit("_s", 1)[0], []).append(n)
    return fam


def load_reps(names):
    """이름 -> (부호화 함수 X -> (n, R', 16) GPU 키, eval_keys chunk)."""
    import torch
    from _invkey_train_equiv_20261115 import load_model, model_encode
    files = rep_files()
    st = np.load(LIN / "stats.npz")
    s = torch.as_tensor(st["s"], device=DEV)
    reps, models = {}, {}
    for n in names:
        f = files[n]
        if n == "raw":
            reps[n] = (lambda X: encode(X), 16)
        elif n == "pca16" or n.startswith(("lin16", "aug16")):
            z = np.load(f)
            assert np.array_equal(z["s"], st["s"]), f
            W = torch.as_tensor(z["W"], device=DEV)
            assert W.shape == (16, 805)
            reps[n] = ((lambda X, W=W: encode(X, s, W)), 64)
        else:
            m, cfg = load_model(f)
            models[n] = m
            reps[n] = ((lambda X, m=m: model_encode(m, X)), 64)
    return reps, models, {n: (None if files[n] is None else str(files[n].relative_to(REPO))) for n in names}


def all_names():
    _, stages, best = stage_info()
    names = ["raw"] + [f"lin16_s{i}" for i in (0, 1, 2)] + [f"aug16_s{i}" for i in (0, 1, 2)]
    for st in stages:
        names += [f"{st}_s{i}" for i in (0, 1, 2)]
    names += [f"{best}aug_s{i}" for i in (0, 1, 2)]
    return names


# ---------------------------------------------------------------- held-out / 학습 쪽
def _eval_seqs(keys, names, out_name):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reps, _, files = load_reps(names)
    out = {"seqs": list(keys), "reps": names, "files": files, "per_seq": {}, "tf32": False}
    t0 = time.time()
    for k in keys:
        spec, xyz, _ = load_seq(k)
        out["per_seq"][k] = {}
        for rep, (enc, ch) in reps.items():
            with torch.no_grad():
                D = enc(spec)
                out["per_seq"][k][rep] = eval_keys(D, xyz, chunk=ch)
            del D
            torch.cuda.empty_cache()
        print(f"{k:22s} n {len(spec):6d}  " + "  ".join(
            f"{r} {100 * out['per_seq'][k][r]['R@1']:.1f}" for r in reps) + f"  {time.time() - t0:.0f}s",
            flush=True)
    out["elapsed_s"] = time.time() - t0
    chmod_write(OUTE / out_name, text=json.dumps(out, indent=1))
    return 0


def heldout(a):
    recs, train, dev, held = split()
    names = all_names()
    names.insert(1, "pca16")
    return _eval_seqs(held, names, "heldout_eval.json")


def trainside(a):
    recs, train, dev, held = split()
    assert set(TRAIN_SIDE) <= set(train)
    names = all_names()
    names.insert(1, "pca16")
    return _eval_seqs(TRAIN_SIDE, names, "trainside_eval.json")


# ---------------------------------------------------------------- 점검 3
def equiv(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from _magspec_invariance_probe_20261114 import apply, yaw_mat
    from _pd_build_test_caches import _make_loader
    from _poolalt_probe_20261114 import aligned_sim, describe
    from _scatter_probe_20261114 import bank, occupancy
    names = all_names()
    reps, models, files = load_reps(names)
    F = torch.from_numpy(bank()).to(DEV)
    res = {"files": files, "n": a.n, "per_seq": {}, "frac_shift": {}}
    t0 = time.time()
    k_idx = torch.arange(NF, device=DEV, dtype=torch.float64)
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
            db = {r: to_freq_t(enc(spec)) for r, (enc, _) in reps.items()}
        xyz_t = torch.as_tensor(xyz, dtype=torch.float64, device=DEV)
        for i in pick:
            pts = loader[int(ids[i])]["points"]
            d0 = describe(occupancy(pts), F)[1]
            d30 = describe(occupancy(apply(yaw_mat(30.0), pts)), F)[1]
            d90 = describe(occupancy(apply(yaw_mat(90.0), pts)), F)[1]
            o = int(rng.integers(n))
            far = (xyz_t - xyz_t[i]).norm(dim=1) > 20.0
            X = np.stack([d0, d30, d90, np.roll(d0, 2, axis=-1), spec[o].astype(np.float32)])
            for r, (enc, _) in reps.items():
                with torch.no_grad():
                    Fk, nk = to_freq_t(enc(X))
                    S = aligned_sim(Fk[:, :1], nk[:1], Fk, nk, NFFT)[0].cpu().numpy()
                    Sd = aligned_sim(Fk[:, :3], nk[:3], *db[r], NFFT)
                    fm = Sd[:, far].amax(1)
                for ci, c in enumerate(conds[:4]):
                    rows[r][c].append(float(S[ci + 1]))
                rows[r]["far_max"].append(float(fm[0]))
                for ci, c in enumerate(ranks):
                    keep[r][c].append(bool(Sd[ci, i] > fm[ci]))
        out = {r: {c: {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                   for c, v in d.items()} for r, d in rows.items()}
        for r in reps:
            out[r]["top1_retention"] = {c: float(np.mean(v)) for c, v in keep[r].items()}
        res["per_seq"][key] = out
        for r in reps:
            v = out[r]
            print(f"{key:22s} {r:12s} yaw_30 med {v['yaw_30']['median']:.4f}  yaw_90 min {v['yaw_90']['min']:.6f}"
                  f"  roll_2 min {v['roll_2']['min']:.6f}  keep "
                  + " ".join(f"{c} {100 * x:.0f}%" for c, x in v["top1_retention"].items()), flush=True)
        del db
        torch.cuda.empty_cache()

        # (b) 칸 사이 이동: 주파수 입력에 위상을 곱한다
        x = torch.as_tensor(spec[pick]).to(DEV).float()
        Xf = torch.fft.rfft(x, dim=-1)                                    # (100, 805, 9)
        fr = {}
        for r, m in models.items():
            with torch.no_grad():
                Y0r, Y0i, _ = m.forward_freq(Xf.real.contiguous(), Xf.imag.contiguous())
                Y0 = torch.complex(Y0r, Y0i).to(torch.complex128)
                yt = torch.fft.rfft(m(x)[0], dim=-1).to(torch.complex128)   # 시간 경로와 같은지
                nrm0 = Y0.abs().pow(2).sum((1, 2)).sqrt()
                d = {"time_vs_freq_path_rel": float(((yt - Y0).abs().pow(2).sum((1, 2)).sqrt() / nrm0).max())}
                for dd in SHIFTS:
                    ph = torch.exp(-2j * np.pi * k_idx * dd / T)          # (9,)
                    Xs = Xf * ph.to(torch.complex64)
                    Ysr, Ysi, _ = m.forward_freq(Xs.real.contiguous(), Xs.imag.contiguous())
                    Ys = torch.complex(Ysr, Ysi).to(torch.complex128)
                    e = (Ys - Y0 * ph).abs().pow(2).sum((1, 2)).sqrt() / nrm0
                    d[f"shift_{dd}"] = {"max": float(e.max()), "median": float(e.median())}
            fr[r] = d
        # 대조: st1_s0 가중치 + 시간 영역 점별 ReLU (Z -> irfft -> relu -> rfft -> A)
        ctrl = next((m for r, m in models.items() if r.startswith("st1_")), None)
        if ctrl is not None:
            blk = ctrl.blocks[0]

            def relu_model(Xr, Xi):
                sc = ctrl.s[None, :, None]
                Z = torch.complex(torch.einsum("hr,nrk->nhk", ctrl.P, Xr / sc),
                                  torch.einsum("hr,nrk->nhk", ctrl.P, Xi / sc))
                Z = torch.fft.rfft(torch.relu(torch.fft.irfft(Z, n=T, dim=-1)), dim=-1)
                Yr, Yi, _ = blk(Z.real.contiguous(), Z.imag.contiguous())
                return torch.complex(Yr, Yi).to(torch.complex128)
            with torch.no_grad():
                Y0 = relu_model(Xf.real.contiguous(), Xf.imag.contiguous())
                nrm0 = Y0.abs().pow(2).sum((1, 2)).sqrt()
                d = {}
                for dd in SHIFTS + (1.0, 2.0):
                    ph = torch.exp(-2j * np.pi * k_idx * dd / T)
                    Xs = Xf * ph.to(torch.complex64)
                    Ys = relu_model(Xs.real.contiguous(), Xs.imag.contiguous())
                    e = (Ys - Y0 * ph).abs().pow(2).sum((1, 2)).sqrt() / nrm0
                    d[f"shift_{dd}"] = {"max": float(e.max()), "median": float(e.median())}
            fr["control_st1_s0_time_relu"] = d
        res["frac_shift"][key] = fr
        for r, d in fr.items():
            print(f"  frac {key} {r:26s} " + "  ".join(
                f"{k}: {v['max']:.2e}" if isinstance(v, dict) else f"{k}: {v:.2e}" for k, v in d.items()),
                flush=True)
    res["elapsed_s"] = time.time() - t0
    chmod_write(OUTE / "equiv_check.json", text=json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- 1위 유지율
def retention(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from _fullmap_probe_20261114 import CONDS, _make_loader, deform
    from _invkey_augment_20261115 import augment, seed_for
    from _poolalt_probe_20261114 import aligned_sim, describe
    from _scatter_probe_20261114 import bank, occupancy
    _, stages, best = stage_info()
    names = [f"lin16_s{i}" for i in (0, 1, 2)] + [f"aug16_s{i}" for i in (0, 1, 2)]
    for st in stages:
        names += [f"{st}_s{i}" for i in (0, 1, 2)]
    names += [f"{best}aug_s{i}" for i in (0, 1, 2)]
    reps, _, files = load_reps(names)
    F = torch.from_numpy(bank()).to(DEV)
    res = {"n_requested": a.n, "files": files, "seqs": list(REFS), "conds": list(RET_CONDS) + ["real"],
           "per_seq": {}}
    t0 = time.time()
    for key in REFS:
        sensor = key.split("/")[0]
        spec, xyz64, ids = load_seq(key)
        n = len(spec)
        loader = _make_loader(*key.split("/"))
        db = {}
        with torch.no_grad():
            for r, (enc, _) in reps.items():
                db[r] = to_freq_t(enc(spec))
        xyz = torch.as_tensor(xyz64, dtype=torch.float64, device=DEV)
        pairs = query_pairs(xyz, n, a.n)
        rng = np.random.default_rng(0)          # deform_check의 난수 흐름
        keep = {r: {c: [] for c in list(RET_CONDS) + ["real"]} for r in reps}
        for kq, (q, j) in enumerate(pairs):
            if kq % 50 == 0:
                print(f"  {key} retention {kq}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
            sid = int(ids[q])
            A = loader[sid]["points"]
            dd = {c: deform(A, c, rng) for c in CONDS}       # 9개 모두: 난수 흐름 맞춤
            X = {c: describe(occupancy(dd[c]), F)[1] for c in OLD_USE}
            for c in NEW_USE:
                t = 0
                while True:
                    tag = f"ret_{c}" if t == 0 else f"ret_{c}_{t}"
                    aug, info = augment(A, np.random.default_rng(seed_for(key, sid, tag)), sensor, c)
                    if c != "obj_mix" or info["n_obj"] >= 1 or t >= 50:
                        break
                    t += 1
                X[c] = describe(occupancy(aug), F)[1]
            X = np.stack([X[c] for c in RET_CONDS])
            neg = (xyz - xyz[q]).norm(dim=1) > 20.0
            for r, (enc, _) in reps.items():
                Fdb, ndb = db[r]
                with torch.no_grad():
                    Fq, nq = to_freq_t(enc(X))
                    S = aligned_sim(Fq, nq, Fdb, ndb, NFFT)
                    S0 = aligned_sim(Fdb[:, q:q + 1], ndb[q:q + 1], Fdb, ndb, NFFT)[0]
                    negmax = S[:, neg].amax(1)
                for ci, c in enumerate(RET_CONDS):
                    keep[r][c].append(bool(S[ci, q] > negmax[ci]))
                keep[r]["real"].append(bool(S0[j] > S0[neg].max()))
        out = {"n_queries": len(pairs),
               "retention": {r: {c: float(np.mean(v)) for c, v in d.items()} for r, d in keep.items()}}
        res["per_seq"][key] = out
        print(f"\n{key} top-1 retention over {len(pairs)} queries")
        print(" " * 15 + "".join(f"{c:>11s}" for c in list(RET_CONDS) + ["real"]))
        for r in reps:
            print(f" {r:14s}" + "".join(f"{100 * out['retention'][r][c]:10.1f}%"
                                        for c in list(RET_CONDS) + ["real"]), flush=True)
        del db
        torch.cuda.empty_cache()
    res["elapsed_s"] = time.time() - t0
    chmod_write(OUTE / "retention_eval.json", text=json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- 호스트: 모으기
def _f(x):
    return "–" if x is None else f"{100 * x:.1f}"


def report(a):
    recs, train, dev, held = split()
    sel, stages, best = stage_info()
    OLDM = json.load(open(LIN / "metrics.json"))
    dc = json.load(open(LIN / "dev_controls.json"))
    old_eq = json.load(open(LIN / "equiv_check.json"))
    old_ret = json.load(open(OUTA / "retention_eval.json"))
    prepc = json.load(open(OUTE / "prep_checks.json"))
    H = json.load(open(OUTE / "heldout_eval.json"))
    TS = json.load(open(OUTE / "trainside_eval.json"))
    EQ = json.load(open(OUTE / "equiv_check.json"))
    RT = json.load(open(OUTE / "retention_eval.json"))
    train_sensors = {recs[k]["sensor"] for k in train}
    G = groups_of(held, train_sensors, recs)
    assert G == OLDM["groups"]
    reps = H["reps"]
    per = H["per_seq"]
    gm = group_means(per, G, reps)
    fam = families(reps)
    spread = {f: {g: {m: spread_of([gm[g][r][m] for r in rs]) for m in KEYS_M} for g in G}
              for f, rs in fam.items()}

    # 학습 실행
    runs = {}
    for p in sorted(glob.glob(str(OUTE / "run_*.json"))):
        d = json.load(open(p))
        r = {k: d.get(k) for k in ("stage", "tau", "lr", "seed", "p_aug", "weight_decay", "epochs",
                                    "neg_per_anchor", "leak_check", "train_s", "train_only_s", "dev_eval_s",
                                    "n_params", "aug_rows_drawn", "aug_rows_replaced", "aug_rows_replaced_frac")}
        r["dev_curve"] = [{"epoch": c["epoch"], "loss": c["loss"], **c["summary"],
                           **({"logg": c["logg"]} if c.get("logg") else {})} for c in d["curve"]]
        r["step0_dev"] = d["step0_dev"]
        tot = {}
        for k, c in d["composition_total"].items():
            for x, v in c.items():
                tot[x] = tot.get(x, 0) + v
        r["direction_share"] = {x: v / sum(tot.values()) for x, v in tot.items()}
        r["composition_total"] = d["composition_total"]
        runs[d["name"]] = r

    # 점검 1: 에폭 0 dev = PCA-16 대조군
    pk = dc["reps"]["pca16"]["per_seq"]
    c1 = {}
    for nme, r in runs.items():
        diffs = {f"{k}|{m}": r["step0_dev"][k][m] - pk[k][m] for k in pk for m in ("R@1", "AP", "R@100P")
                 if r["step0_dev"][k][m] != pk[k][m]}
        c1[nme] = {"equal": not diffs, "diffs": diffs}
    c1_all = all(v["equal"] for v in c1.values())
    # 점검 2: 누출
    c2 = {"train_dev_overlap": sorted(set(train) & set(dev)),
          "train_heldout_overlap": sorted(set(train) & set(held)),
          "dev_heldout_overlap": sorted(set(dev) & set(held)),
          "trainside_seqs_in_train": all(k in train for k in TRAIN_SIDE),
          "per_run": {n: {"forbidden_touched": r["leak_check"]["forbidden_touched"],
                          "n_forbidden": r["leak_check"]["n_forbidden"],
                          "sequences_touched_subset_of_train":
                              set(r["leak_check"]["sequences_touched"]) <= set(train),
                          "n_sequences_touched": len(r["leak_check"]["sequences_touched"])}
                      for n, r in runs.items()}}
    # 점검 4: raw·lin16 held-out = 기존 metrics.json; equiv raw·lin16 = 기존 equiv_check; 유지율 = 기존 retention
    c4 = {"heldout": {}, "equiv_vs_old": {}, "retention_vs_linear_aug": {}}
    for r in ["raw", "pca16"] + fam["lin16"]:
        d, nu = max_abs_diff({k: OLDM["heldout_per_seq"][k][r] for k in held}, {k: per[k][r] for k in held})
        c4["heldout"][r] = {"max_abs_diff": d, "n_unequal_fields": nu}
    for key in REFS:
        for r in ["raw"] + fam["lin16"]:
            o, nw = old_eq[key][r], EQ["per_seq"][key][r]
            c4["equiv_vs_old"][f"{key}|{r}"] = max_abs_diff(o, nw)
    for key in REFS:
        for r in fam["lin16"] + fam["aug16"]:
            o = old_ret["per_seq"][key]["retention"][r]
            nw = RT["per_seq"][key]["retention"][r]
            c4["retention_vs_linear_aug"][f"{key}|{r}"] = {c: [o[c], nw[c]] for c in RET_CONDS + ("real",)
                                                           if o[c] != nw[c]}
    c4_ok = {"heldout": all(v["n_unequal_fields"] == 0 for v in c4["heldout"].values()),
             "equiv": all(v[1] == 0 for v in c4["equiv_vs_old"].values()),
             "retention": all(not v for v in c4["retention_vs_linear_aug"].values())}
    # 점검 3 요약
    eqfam = families([r for r in EQ["per_seq"][REFS[0]]])
    c3 = {"a_roll2_yaw90_min_sim": {f: {key: {"roll_2_min": min(EQ["per_seq"][key][r]["roll_2"]["min"] for r in rs),
                                              "yaw_90_min": min(EQ["per_seq"][key][r]["yaw_90"]["min"] for r in rs)}
                                        for key in REFS} for f, rs in eqfam.items()},
          "b_frac_shift_max_rel_err": {}, "c_yaw30": {}}
    for key in REFS:
        for r, d in EQ["frac_shift"][key].items():
            f = r if r.startswith("control") else r.rsplit("_s", 1)[0]
            e = c3["b_frac_shift_max_rel_err"].setdefault(f, {}).setdefault(key, {})
            for k, v in d.items():
                val = v["max"] if isinstance(v, dict) else v
                e[k] = max(e.get(k, 0.0), val)
    for f, rs in eqfam.items():
        c3["c_yaw30"][f] = {key: {"sim_median": spread_of([EQ["per_seq"][key][r]["yaw_30"]["median"] for r in rs]),
                                  "top1_keep": spread_of([EQ["per_seq"][key][r]["top1_retention"]["yaw_30"]
                                                          for r in rs]),
                                  "other_median": spread_of([EQ["per_seq"][key][r]["other"]["median"] for r in rs]),
                                  "far_max_median": spread_of([EQ["per_seq"][key][r]["far_max"]["median"]
                                                               for r in rs])}
                            for key in REFS}

    # 학습 쪽 대 held-out 쪽
    tfam = families(TS["reps"])
    tper = TS["per_seq"]

    def pooled(perx, ks, r):
        return sum(perx[k][r]["n_hit1"] for k in ks) / sum(perx[k][r]["n_pos"] for k in ks)
    gain = {}
    for f, rs in tfam.items():
        gain[f] = {"train_side_per_seq_R@1": {k: spread_of([tper[k][r]["R@1"] for r in rs]) for k in TRAIN_SIDE},
                   "train_side_mean_R@1": spread_of([np.mean([tper[k][r]["R@1"] for k in TRAIN_SIDE]) for r in rs]),
                   "train_side_pooled_R@1": spread_of([pooled(tper, TRAIN_SIDE, r) for r in rs])}
        if f in fam:
            for g in G:
                gain[f][f"heldout_{g}_mean_R@1"] = spread[f][g]["R@1"]
                gain[f][f"heldout_{g}_pooled_R@1"] = spread[f][g]["R@1_pooled"]

    out = {"date": "2026-11-15",
           "config": {"tau": TAU, "epochs_sweep": 40, "lr_grid": [1e-3, 3e-3], "train_script":
                      "scripts/_invkey_train_equiv_20261115.py", "eval_script": "scripts/_invkey_eval_equiv_20261115.py",
                      "tf32": False, "aug_p": 0.25, "trainside": list(TRAIN_SIDE)},
           "selection": sel, "stages": stages, "best_stage": best,
           "split": OLDM["split"], "train": train, "dev": dev, "heldout": held, "groups": G,
           "keys": H["files"], "heldout_per_seq": per, "heldout_group_means": gm, "seed_spread": spread,
           "trainside_per_seq": tper, "train_vs_heldout": gain,
           "equiv_check": EQ, "retention": RT, "prep_checks": prepc,
           "checks": {"1_step0_equals_pca16_on_dev": {"all_equal": c1_all, "per_run": c1,
                                                      "key_diff_vs_pca16": prepc["init"]},
                      "2_leak": c2, "3_equivariance": c3,
                      "4_reproduction": {"ok": c4_ok, **c4}},
           "runs": runs,
           "elapsed_s": {"heldout": H.get("elapsed_s"), "trainside": TS.get("elapsed_s"),
                         "equiv": EQ.get("elapsed_s"), "retention": RT.get("elapsed_s")}}
    chmod_write(OUTE / "metrics.json", text=json.dumps(out, indent=1))

    # ---- 표
    print("checks: 1 all_equal", c1_all, {n: v["diffs"] for n, v in c1.items() if not v["equal"]})
    print("        2", {k: v for k, v in c2.items() if k != "per_run"},
          all(not v["forbidden_touched"] and v["sequences_touched_subset_of_train"] for v in c2["per_run"].values()))
    print("        4", c4_ok, {k: v for k, v in c4["retention_vs_linear_aug"].items() if v})
    print("        3a", json.dumps(c3["a_roll2_yaw90_min_sim"]))
    print("        3b", json.dumps(c3["b_frac_shift_max_rel_err"]))
    print("\n## 30도 회전 (sim 중앙값 / 1위 유지, seed 평균)\n")
    print("| 키 | " + " | ".join(f"{k} sim | {k} keep" for k in REFS) + " |")
    print("|---|" + "---:|" * (2 * len(REFS)))
    for f in eqfam:
        print(f"| {f} | " + " | ".join(f"{c3['c_yaw30'][f][k]['sim_median']['mean']:.4f} | "
                                       f"{100 * c3['c_yaw30'][f][k]['top1_keep']['mean']:.0f}" for k in REFS) + " |")
    print("\n## 무게 수")
    for nme, r in runs.items():
        if nme.startswith("final"):
            print(nme, r["n_params"], f"train {r['train_s']:.0f}s (only {r['train_only_s']:.0f}s, dev {r['dev_eval_s']:.0f}s)")
    print("\n## group means (seed mean [min–max])")
    for g in G:
        print(f"\n### {g} ({len(G[g])})\n")
        print("| 키 | " + " | ".join(KEYS_M) + " |")
        print("|---|" + "---:|" * len(KEYS_M))
        for fn in fam:
            print(f"| {fn} | " + " | ".join(
                f"{_f(spread[fn][g][m]['mean'])} [{_f(spread[fn][g][m]['min'])}–{_f(spread[fn][g][m]['max'])}]"
                for m in KEYS_M) + " |")
        print("n dirs", {d: gm[g]["raw"][f"n_{d}"] for d in ("same", "reverse", "crossing")})
    print("\n## per seq R@1 (seed mean)\n")
    print("| 시퀀스 | 양성 | " + " | ".join(fam) + " |")
    print("|---|---:|" + "---:|" * len(fam))
    for k in held:
        print(f"| {k} | {per[k]['raw']['n_pos']} | "
              + " | ".join(_f(np.mean([per[k][r]["R@1"] for r in rs])) for rs in fam.values()) + " |")
    for m in ("AP", "R@100P"):
        print(f"\n## per seq {m} (seed mean)\n")
        print("| 시퀀스 | " + " | ".join(fam) + " |")
        print("|---|" + "---:|" * len(fam))
        for k in held:
            print(f"| {k} | " + " | ".join(_f(np.mean([per[k][r][m] for r in rs])) for rs in fam.values()) + " |")
    print("\n## 학습 쪽 대 held-out 쪽 R@1 (seed 평균)\n")
    print("| 키 | " + " | ".join(TRAIN_SIDE) + " | 학습 4개 평균 | ref3 | seen 9 | mcd 12 | mcd 모음 |")
    print("|---|" + "---:|" * (len(TRAIN_SIDE) + 5))
    for f in tfam:
        g_ = gain[f]
        print(f"| {f} | " + " | ".join(_f(g_["train_side_per_seq_R@1"][k]["mean"]) for k in TRAIN_SIDE)
              + f" | {_f(g_['train_side_mean_R@1']['mean'])} | "
              + " | ".join(_f(g_[f'heldout_{g}_mean_R@1']['mean']) if f'heldout_{g}_mean_R@1' in g_ else "–"
                           for g in G)
              + f" | {_f(g_['heldout_unseen_mcd_pooled_R@1']['mean']) if 'heldout_unseen_mcd_pooled_R@1' in g_ else '–'} |")
    print("\n## retention (seed mean [min–max])")
    rfam = families(list(RT["files"]))
    for key, d in RT["per_seq"].items():
        cs = list(RET_CONDS) + ["real"]
        print(f"\n### {key} ({d['n_queries']})\n")
        print("| 키 | " + " | ".join(cs) + " |")
        print("|---|" + "---:|" * len(cs))
        for fn, rs in rfam.items():
            print(f"| {fn} | " + " | ".join(
                f"{_f(np.mean([d['retention'][r][c] for r in rs]))} "
                f"[{_f(min(d['retention'][r][c] for r in rs))}–{_f(max(d['retention'][r][c] for r in rs))}]"
                for c in cs) + " |")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("heldout")
    sub.add_parser("trainside")
    p = sub.add_parser("equiv")
    p.add_argument("--n", type=int, default=100)
    p = sub.add_parser("retention")
    p.add_argument("--n", type=int, default=300)
    sub.add_parser("report")
    a = ap.parse_args()
    return {"heldout": heldout, "trainside": trainside, "equiv": equiv, "retention": retention,
            "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
