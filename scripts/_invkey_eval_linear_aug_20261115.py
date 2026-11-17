#!/usr/bin/env python3
"""invkey 진단 연구: 증강으로 학습한 K=16 선형 키를 깨끗하게 학습한 키·raw와 비교한다 (2026-11-15).

키 (모두 t축 원형 상관 정렬 + 코사인, nfft 128; 정의는 scripts/_invkey_train_linear_20261114.py)
  raw          spec 805x16 그대로
  lin16_s0..2  깨끗하게 학습한 W (linear/run_final_K16_tau0.05_lr0.003_s*_e40.npz)
  aug16_s0..2  증강으로 학습한 W (linear_aug/selection.json의 p·에폭, run_final_*.npz)

heldout  held-out 21시퀀스 검색 지표. scripts/_invkey_eval_linear_20261114.py heldout과 같은 프로토콜·chunk.
retention
  기준 3시퀀스(REFS): deform_check와 같은 질의 300개(과거 5 m 안 최근접 j가 있고 20 m 밖 키프레임이 있는
    q를 고르게)에서, 질의 스캔에 변형을 적용해 다시 인코딩하고 지도(저장된 깨끗한 계수 전체)에서
    원래 q가 20 m 밖 모든 키프레임보다 높으면 1위 유지.
    기존 조건 occl_30·occl_90·crop_50·crop_30: deform_check와 같은 난수 흐름(시퀀스마다 default_rng(0),
      질의마다 _fullmap_probe CONDS 9개를 순서대로 deform)이라 변형 스캔이 deform_check와 같다.
      쓰지 않는 조건(drop_50 등)도 deform은 불러 난수 흐름을 맞추고 서술자는 계산하지 않는다.
    새 조건 obj_mix·near_bus·range_r40·beams_half·all: augment(), 시드 seed_for(시퀀스, scan_id, "ret_<조건>").
      obj_mix는 물체 수가 0으로 뽑히면 태그 "ret_obj_mix_<t>"로 다시 뽑는다(1개 이상).
      helipr(16줄)는 beams_half를 적용하지 않는다(값 없음으로 둔다).
    real: 저장된 q 키로 지도에서 실제 재방문 j가 20 m 밖 모든 키프레임보다 높은지(deform_check와 같음).
    recomputed: 변형 없이 다시 계산한 서술자(저장 float16과의 차이만 있는 점검).
    변형 스캔은 질의마다 한 번 만들어 모든 키에 같이 쓴다(키와 무관).
    유사도: 저장된 깨끗한 q 키와 그 near_bus·all 변형 키의 정렬 코사인, 그리고 20 m 넘게 떨어진 임의
      키프레임(질의마다 default_rng(1)로 하나, 키와 무관)과 q의 정렬 코사인.
  MCD 2시퀀스(--mcd): 새 조건과 real만. MCD는 전방 축이 없어 near_bus 방향이 무작위다.
report (호스트)  모든 수치·점검·선택·학습 구성을 linear_aug/metrics.json에 모으고 표를 찍는다.

실행 (리포 루트):
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_eval_linear_aug_20261115.py heldout
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_eval_linear_aug_20261115.py retention --n 300 --mcd mcd/kth_day_06 mcd/tuhh_day_02
  python3 scripts/_invkey_eval_linear_aug_20261115.py report
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
    DEV, NFFT, OUTD as LIN, REFS, chmod_write, encode, eval_keys, load_seq, split, to_freq_t)

OUTA = REPO / "results/invkey_study_20261114/linear_aug"
METRICS = ("R@1", "R@100", "AP", "max_F1", "R@100P", "R@100P_10m")
OLD_CONDS = ("occl_30", "occl_90", "crop_50", "crop_30")
NEW_CONDS = ("obj_mix", "near_bus", "range_r40", "beams_half", "all")
SIM_CONDS = ("near_bus", "all")
LIN_RUN = "run_final_K16_tau0.05_lr0.003_s{s}_e40.npz"


def aug_run_name(seed):
    sel = json.load(open(OUTA / "selection.json"))
    return f"run_final_K16_tau0.05_lr0.003_p{sel['p']}_s{seed}_e{sel['epochs']}"


def load_reps():
    """이름 -> (s, W) GPU 텐서, 파일 목록."""
    import torch
    st = np.load(LIN / "stats.npz")
    s = torch.as_tensor(st["s"], device=DEV)
    reps, files = {"raw": (None, None)}, {"raw": None}
    for i in (0, 1, 2):
        for name, f in ((f"lin16_s{i}", LIN / LIN_RUN.format(s=i)),
                        (f"aug16_s{i}", OUTA / (aug_run_name(i) + ".npz"))):
            z = np.load(f)
            assert np.array_equal(z["s"], st["s"]), f
            assert z["W"].shape == (16, 805), f
            reps[name] = (s, torch.as_tensor(z["W"], device=DEV))
            files[name] = str(f.relative_to(REPO))
    order = ["raw"] + [f"lin16_s{i}" for i in (0, 1, 2)] + [f"aug16_s{i}" for i in (0, 1, 2)]
    return {r: reps[r] for r in order}, files


# ---------------------------------------------------------------- held-out
def heldout(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    recs, train, dev, held = split()
    reps, files = load_reps()
    out = {"heldout": held, "reps": list(reps), "files": files, "per_seq": {}, "tf32": False}
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
    chmod_write(OUTA / "heldout_eval.json", text=json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- 1위 유지율·유사도
def query_pairs(xyz, n, nq):
    """deform_check와 같은 질의 고르기."""
    pairs = []
    for q in range(30, n - 1):
        d = (xyz[: q - 29] - xyz[q]).norm(dim=1)
        j = int(d.argmin())
        if d[j] <= 5.0 and (d > 20.0).any():
            pairs.append((q, j))
    if len(pairs) > nq:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, nq).astype(int)]
    return pairs


def _q(v):
    v = np.asarray(v, np.float64)
    return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(v.mean()),
            "p05": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)), "p95": float(np.percentile(v, 95))}


def retention(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from _fullmap_probe_20261114 import CONDS, _make_loader, deform
    from _invkey_augment_20261115 import augment, seed_for
    from _poolalt_probe_20261114 import aligned_sim, describe
    from _scatter_probe_20261114 import bank, occupancy
    reps, files = load_reps()
    F = torch.from_numpy(bank()).to(DEV)
    seqs = list(REFS) + list(a.mcd)
    res = {"n_requested": a.n, "files": files, "seqs": seqs, "per_seq": {}}
    t0 = time.time()
    for key in seqs:
        ref = key in REFS
        sensor = key.split("/")[0]
        spec, xyz64, ids = load_seq(key)
        n = len(spec)
        loader = _make_loader(*key.split("/"))
        db = {}
        with torch.no_grad():
            for r, (s, W) in reps.items():
                db[r] = to_freq_t(encode(spec, s, W))
        xyz = torch.as_tensor(xyz64, dtype=torch.float64, device=DEV)
        pairs = query_pairs(xyz, n, a.n)
        rng = np.random.default_rng(0)          # deform_check의 난수 흐름
        rng_far = np.random.default_rng(1)
        new = [c for c in NEW_CONDS if not (c == "beams_half" and sensor == "helipr")]
        conds = (list(OLD_CONDS) + ["recomputed"] if ref else []) + new
        keep = {r: {c: [] for c in conds + ["real"]} for r in reps}
        sims = {r: {c: [] for c in (["recomputed"] + list(SIM_CONDS) + ["far_random"])} for r in reps}
        ainfo = {c: {"frac_affected": [], "frac_replaced": [], "frac_removed": [], "n_objects_placed": [],
                     "near_placed": [], "redraws": [], "thin": []} for c in new}
        for kq, (q, j) in enumerate(pairs):
            if kq % 50 == 0:
                print(f"  {key} retention {kq}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
            sid = int(ids[q])
            A = loader[sid]["points"]
            X = []
            if ref:
                dd = {c: deform(A, c, rng) for c in CONDS}       # 9개 모두: 난수 흐름 맞춤
                X += [describe(occupancy(dd[c]), F)[1] for c in OLD_CONDS]
                X.append(describe(occupancy(A), F)[1])
            for c in new:
                t = 0
                while True:
                    tag = f"ret_{c}" if t == 0 else f"ret_{c}_{t}"
                    aug, info = augment(A, np.random.default_rng(seed_for(key, sid, tag)), sensor, c)
                    if c != "obj_mix" or info["n_obj"] >= 1 or t >= 50:
                        break
                    t += 1
                ai = ainfo[c]
                for f in ("frac_affected", "frac_replaced", "frac_removed", "thin"):
                    ai[f].append(info[f])
                ai["n_objects_placed"].append(sum(1 for o in info["objects"] if not o["near"]))
                ai["near_placed"].append(any(o["near"] for o in info["objects"]))
                ai["redraws"].append(t)
                X.append(describe(occupancy(aug), F)[1])
            X = np.stack(X)
            neg = (xyz - xyz[q]).norm(dim=1) > 20.0
            far_k = int(rng_far.choice(np.flatnonzero(neg.cpu().numpy())))
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
                if ref:
                    for c in ["recomputed"] + list(SIM_CONDS):
                        sims[r][c].append(float(S[conds.index(c), q]))
                    sims[r]["far_random"].append(float(S0[far_k]))
        out = {"n_queries": len(pairs), "conds": conds,
               "not_applied": ["beams_half (16-beam sensor)"] if sensor == "helipr" else [],
               "retention": {r: {c: float(np.mean(v)) for c, v in d.items()} for r, d in keep.items()},
               "aug_info": {c: {"frac_affected": _q(d["frac_affected"]),
                                "frac_replaced_mean": float(np.mean(d["frac_replaced"])),
                                "frac_removed_mean": float(np.mean(d["frac_removed"])),
                                "n_objects_placed_mean": float(np.mean(d["n_objects_placed"])),
                                "n_zero_objects_placed": int(np.sum(np.array(d["n_objects_placed"]) == 0)),
                                "near_placed_rate": float(np.mean(d["near_placed"])),
                                "redraws_total": int(np.sum(d["redraws"])),
                                "thin_counts": {str(f): int(np.sum(np.array(d["thin"]) == f))
                                                for f in sorted(set(d["thin"]))}}
                            for c, d in ainfo.items()}}
        if ref:
            out["similarity"] = {r: {c: _q(v) for c, v in d.items()} for r, d in sims.items()}
        res["per_seq"][key] = out
        print(f"\n{key} top-1 retention over {len(pairs)} queries")
        print(" " * 11 + "".join(f"{c:>11s}" for c in conds + ["real"]))
        for r in reps:
            print(f" {r:10s}" + "".join(f"{100 * out['retention'][r][c]:10.1f}%"
                                        for c in conds + ["real"]), flush=True)
        if ref:
            for r in reps:
                print(f" sim {r:10s} " + "  ".join(
                    f"{c} {out['similarity'][r][c]['median']:.4f}" for c in sims[r]), flush=True)
        del db
        torch.cuda.empty_cache()
        chmod_write(OUTA / "retention_eval.json", text=json.dumps(res, indent=1))
    res["elapsed_s"] = time.time() - t0
    chmod_write(OUTA / "retention_eval.json", text=json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- 호스트: 모으기
def groups_of(held, train_sensors, recs):
    return {"ref3": list(REFS),
            "seen_sensor": [k for k in held if recs[k]["sensor"] in train_sensors],
            "unseen_mcd": [k for k in held if k.startswith("mcd/")]}


def group_means(per, G, reps):
    """scripts/_invkey_eval_linear_20261114.py report()와 같은 묶음 평균."""
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
    return gm


def spread_of(vals):
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}


def max_abs_diff(x, y):
    """같은 모양 dict/list/수의 최대 절대 차(수가 아닌 값은 같아야 함). 다르면 (차, 불일치 수)."""
    if isinstance(x, dict):
        assert set(x) == set(y), (set(x) ^ set(y))
        rs = [max_abs_diff(x[k], y[k]) for k in x]
        return max([r[0] for r in rs], default=0.0), sum(r[1] for r in rs)
    if isinstance(x, list):
        assert len(x) == len(y)
        rs = [max_abs_diff(u, v) for u, v in zip(x, y)]
        return max([r[0] for r in rs], default=0.0), sum(r[1] for r in rs)
    if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and y is not None:
        return abs(float(x) - float(y)), int(x != y)
    return 0.0, int(x != y)


def report(a):
    recs, train, dev, held = split()
    OLDM = json.load(open(LIN / "metrics.json"))
    dc = json.load(open(LIN / "dev_controls.json"))
    sel = json.load(open(OUTA / "selection.json"))
    H = json.load(open(OUTA / "heldout_eval.json"))
    RT = json.load(open(OUTA / "retention_eval.json"))
    old_deform = json.load(open(LIN / "deform_check.json"))
    train_sensors = {recs[k]["sensor"] for k in train}
    G = groups_of(held, train_sensors, recs)
    assert G == OLDM["groups"]
    reps = H["reps"]
    per = H["per_seq"]
    gm = group_means(per, G, reps)
    fam = {"lin16": [f"lin16_s{i}" for i in (0, 1, 2)], "aug16": [f"aug16_s{i}" for i in (0, 1, 2)]}
    keys_m = METRICS + ("R@1_same_pooled", "R@1_reverse_pooled", "R@1_crossing_pooled", "R@1_pooled")
    spread = {f: {g: {m: spread_of([gm[g][r][m] for r in rs]) for m in keys_m} for g in G}
              for f, rs in fam.items()}
    # 학습 실행
    runs = {}
    for p in sorted(glob.glob(str(OUTA / "run_*.json"))):
        d = json.load(open(p))
        runs[d["name"]] = {k: d[k] for k in ("K", "tau", "lr", "seed", "p_aug", "epochs", "neg_per_anchor",
                                             "leak_check", "train_s", "aug_rows_drawn",
                                             "aug_rows_replaced", "aug_rows_replaced_frac")}
        runs[d["name"]]["dev_curve"] = [{"epoch": c["epoch"], "loss": c["loss"], **c["summary"]}
                                        for c in d["curve"]]
        runs[d["name"]]["step0_dev"] = d["step0_dev"]
        runs[d["name"]]["aug_file_check"] = d["aug_file_check"]
        tot = {}
        for k, c in d["composition_total"].items():
            for x, v in c.items():
                tot[x] = tot.get(x, 0) + v
        runs[d["name"]]["direction_share"] = {x: v / sum(tot.values()) for x, v in tot.items()}
        runs[d["name"]]["_full"] = d
    # 점검 (i): p = 0 재현
    old0 = json.load(open(LIN / "run_sweep_K16_tau0.05_lr0.003_s0_e40.json"))
    oldf0 = json.load(open(LIN / "run_final_K16_tau0.05_lr0.003_s0_e40.json"))
    new0 = runs["sweep_K16_tau0.05_lr0.003_p0.0_s0_e40"]["_full"]
    c_i = {}
    for lab, od in (("vs_linear_sweep_s0", old0), ("vs_linear_final_s0", oldf0)):
        dsum, nsum = max_abs_diff([c["summary"] for c in od["curve"]], [c["summary"] for c in new0["curve"]])
        dper, nper = max_abs_diff([c["per_seq"] for c in od["curve"]], [c["per_seq"] for c in new0["curve"]])
        dloss = max(abs(u["loss"] - v["loss"]) for u, v in zip(od["curve"][1:], new0["curve"][1:]))
        ep_diff = [c["epoch"] for u, c in zip(od["curve"], new0["curve"])
                   if max_abs_diff(u["per_seq"], c["per_seq"])[1]]
        c_i[lab] = {"summary_max_abs_diff": dsum, "summary_n_unequal": nsum,
                    "per_seq_max_abs_diff": dper, "per_seq_n_unequal": nper,
                    "loss_max_abs_diff": dloss, "epochs_with_any_difference": ep_diff,
                    "composition_equal": od["composition_total"] == new0["composition_total"],
                    "identical": nsum == 0 and nper == 0 and dloss == 0.0}
    # 점검 (ii): 에폭 0 = PCA-16 대조군
    pk = dc["reps"]["pca16"]["per_seq"]
    c_ii = {n: all(r["step0_dev"][k][m] == pk[k][m] for k in pk for m in ("R@1", "AP", "R@100P"))
            for n, r in runs.items()}
    # 점검 (iii): 누출
    aug_keys = sorted(json.load(open(REPO / "results/invkey_study_20261114/features_aug1_manifest.json"))
                      ["sequences"])
    c_iii = {"train_dev_overlap": sorted(set(train) & set(dev)),
             "train_heldout_overlap": sorted(set(train) & set(held)),
             "aug_feature_seqs_equal_train": aug_keys == sorted(train),
             "per_run": {n: {"forbidden_touched": r["leak_check"]["forbidden_touched"],
                             "n_forbidden": r["leak_check"]["n_forbidden"],
                             "sequences_touched_subset_of_train":
                                 set(r["leak_check"]["sequences_touched"]) <= set(train),
                             "n_sequences_touched": len(r["leak_check"]["sequences_touched"])}
                         for n, r in runs.items()}}
    # 점검 (iv): 깨끗한 학습 16·raw의 held-out 수치가 기존 metrics.json과 같은지
    c_iv = {}
    for r in ["raw"] + fam["lin16"]:
        d, nu = max_abs_diff({k: OLDM["heldout_per_seq"][k][r] for k in held}, {k: per[k][r] for k in held})
        c_iv[r] = {"max_abs_diff": d, "n_unequal_fields": nu}
    # 점검 (v): 기존 조건 유지율이 deform_check와 같은지
    c_v = {}
    for key in REFS:
        for r in ["raw"] + fam["lin16"]:
            o = old_deform[key]["retention"][r]
            nw = RT["per_seq"][key]["retention"][r]
            c_v[f"{key}|{r}"] = {c: [o[c], nw[c]] for c in OLD_CONDS + ("real",) if o[c] != nw[c]}
    c_v_ok = all(not v for v in c_v.values())
    for v in runs.values():
        v.pop("_full")
    out = {"date": "2026-11-15",
           "config": {"K": 16, "tau": 0.05, "lr": 3e-3, "epochs_sweep": 40, "init": "linear/pca_K16.npz",
                      "s": "linear/stats.npz", "p_grid": [0.25, 0.5, 0.75], "check_p0": True,
                      "aug_features": "/workspace/data/invkey_study_20261114/features_aug1 (cond all, 1 draw "
                                      "per keyframe, 33 train sequences)",
                      "aug_rule": "each pooled keyframe (anchor, positive, extra) replaced independently "
                                  "with prob p; mask rng default_rng([seed, 20261115])",
                      "tf32": False, "train_script": "scripts/_invkey_train_linear_aug_20261115.py",
                      "eval_script": "scripts/_invkey_eval_linear_aug_20261115.py"},
           "selection": sel, "split": OLDM["split"], "train": train, "dev": dev, "heldout": held,
           "groups": G, "keys": H["files"],
           "heldout_per_seq": per, "heldout_group_means": gm, "seed_spread": spread,
           "retention": RT,
           "checks": {"i_p0_reproduces_clean_run": c_i, "ii_step0_equals_pca16_on_dev": c_ii,
                      "iii_leak": c_iii, "iv_heldout_matches_linear_metrics": c_iv,
                      "v_old_conditions_match_deform_check": {"all_equal": c_v_ok, "diffs": c_v}},
           "runs": runs}
    chmod_write(OUTA / "metrics.json", text=json.dumps(out, indent=1))
    print(json.dumps(out["checks"]["i_p0_reproduces_clean_run"], indent=1))
    print("ii", c_ii)
    print("iii", {k: v for k, v in c_iii.items() if k != "per_run"},
          {n: (v["forbidden_touched"], v["sequences_touched_subset_of_train"], v["n_sequences_touched"])
           for n, v in c_iii["per_run"].items()})
    print("iv", c_iv)
    print("v all_equal", c_v_ok, {k: v for k, v in c_v.items() if v})
    f = lambda x: "–" if x is None else f"{100 * x:.1f}"  # noqa: E731
    print("\n## group means (seed mean [min–max])")
    for g in G:
        print(f"\n### {g} ({len(G[g])})\n")
        print("| 키 | " + " | ".join(keys_m) + " |")
        print("|---|" + "---:|" * len(keys_m))
        print("| raw | " + " | ".join(f(gm[g]["raw"][m]) for m in keys_m) + " |")
        for fn in fam:
            print(f"| {fn} | " + " | ".join(
                f"{f(spread[fn][g][m]['mean'])} [{f(spread[fn][g][m]['min'])}–{f(spread[fn][g][m]['max'])}]"
                for m in keys_m) + " |")
        print("n dirs", {d: gm[g]["raw"][f"n_{d}"] for d in ("same", "reverse", "crossing")})
    print("\n## per seq (R@1, AP, R@100P): raw | lin16 mean | aug16 mean | diff")
    for m in ("R@1", "AP", "R@100P", "R@100P_10m"):
        print(f"\n### {m}\n")
        print("| 시퀀스 | 양성 | raw | lin16 | aug16 | aug-lin | aug min–max | lin min–max |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k in held:
            lv = [per[k][r][m] for r in fam["lin16"]]
            av = [per[k][r][m] for r in fam["aug16"]]
            print(f"| {k} | {per[k]['raw']['n_pos']} | {f(per[k]['raw'][m])} | {f(np.mean(lv))} | "
                  f"{f(np.mean(av))} | {100 * (np.mean(av) - np.mean(lv)):+.1f} | {f(min(av))}–{f(max(av))} | "
                  f"{f(min(lv))}–{f(max(lv))} |")
    print("\n## retention")
    for key, d in RT["per_seq"].items():
        cs = d["conds"] + ["real"]
        print(f"\n### {key} ({d['n_queries']})\n")
        print("| 키 | " + " | ".join(cs) + " |")
        print("|---|" + "---:|" * len(cs))
        print("| raw | " + " | ".join(f(d["retention"]["raw"][c]) for c in cs) + " |")
        for fn, rs in fam.items():
            print(f"| {fn} | " + " | ".join(
                f"{f(np.mean([d['retention'][r][c] for r in rs]))} "
                f"[{f(min(d['retention'][r][c] for r in rs))}–{f(max(d['retention'][r][c] for r in rs))}]"
                for c in cs) + " |")
        print("aug_info", json.dumps({c: {"aff_med": round(v["frac_affected"]["median"], 3),
                                          "objs": round(v["n_objects_placed_mean"], 2),
                                          "zero_obj": v["n_zero_objects_placed"],
                                          "near": round(v["near_placed_rate"], 3),
                                          "redraws": v["redraws_total"], "thin": v["thin_counts"]}
                                      for c, v in d["aug_info"].items()}))
        if "similarity" in d:
            sc = list(d["similarity"]["raw"])
            print("\n| 키 | " + " | ".join(f"sim {c} med" for c in sc) + " |")
            print("|---|" + "---:|" * len(sc))
            for r in reps:
                print(f"| {r} | " + " | ".join(f"{d['similarity'][r][c]['median']:.4f}" for c in sc) + " |")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("heldout")
    p = sub.add_parser("retention")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--mcd", nargs="*", default=[])
    sub.add_parser("report")
    a = ap.parse_args()
    return {"heldout": heldout, "retention": retention, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
