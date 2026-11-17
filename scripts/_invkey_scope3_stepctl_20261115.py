#!/usr/bin/env python3
"""invkey 진단 연구: 학습 양(스텝 수)과 자료 구성(센서 다양성)의 혼동을 가르는 대조 (2026-11-15).

새 범위 선형 16 (a)는 264스텝(33에폭 x 8), 기존 선형 16은 약 5085스텝(40에폭 x 약 127)이었다.
  대조 1 (train1)  새 범위 (a) 구성(상한 없음, 장소 묶음 음성, scope3 s·PCA, tau 0.05, lr 3e-3)으로 625에폭
                   (= 5000스텝; (a)는 에폭마다 정확히 8스텝)까지 한 번에 학습하고 264·1304·2600·5000스텝에서
                   마지막 가중치와, 그 스텝까지의 개발용 최고 에폭(sel_R@1, 같으면 AP; 처음 나온 것) 가중치를
                   저장한다. 학습률이 고정이라 더 긴 실행의 앞부분은 짧은 실행과 같다(점검: 264스텝 가중치 =
                   scope3 final_a, 앞 33에폭 곡선 = final_a 곡선).
  대조 2 (train2)  기존 선형 16의 학습 구성(linear/ s·PCA, candidates A 분할, 25% 장소 상한, 다른 도시 음성,
                   tau 0.05, lr 3e-3)을 같은 seed로 다시 돌려 264·1300스텝에서 가중치를 저장한다(에폭 중간에서
                   멈춤). 점검: 끝난 에폭마다 손실·개발용 지표, 에폭 1 구성이 기존 run_final_K16 json과 같은지.
  학습 고리는 scripts/_invkey_train_linear_20261114.py train_one과 같은 순서(같은 난수 호출)의 사본이고,
  스텝 저장만 더했다.
eval   평가용 9 + kth 6 + 개발용 3의 검색 지표(eval_keys), 1위 유지율 occl_90·near_bus(kitti360/0004,
       mcd/tuhh_day_02; scope3 retention과 같은 질의·난수 흐름). 대조 모델 + 기존 선형 16 + 새 (a).
report (호스트) stepctl/metrics.json과 표.

실행 (리포 루트)
  for s in 0 1 2; do docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_scope3_stepctl_20261115.py train1 --seed $s & done
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_scope3_stepctl_20261115.py train2 --seeds 0 1 2
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_scope3_stepctl_20261115.py eval
  python3 scripts/_invkey_scope3_stepctl_20261115.py report
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _invkey_data_survey_20261114 import DIRS, classify  # noqa: E402,F401
from _invkey_eval_linear_aug_20261115 import (  # noqa: E402
    group_means, max_abs_diff, query_pairs, spread_of)
from _invkey_eval_scope3_20261115 import GROUPS, KTH, RET_SEQS, new_run  # noqa: E402
from _invkey_train_linear_20261114 import (  # noqa: E402
    DEV, NEG_GAP, NEG_M, NFFT, OUTD as LIN, TrainData, chmod_write, dev_groups, dev_summary, encode, eval_keys,
    load_seq, split, to_freq_t)
from _invkey_train_scope3_20261115 import DEVSET, EVAL, OUT3, TRAIN, split3  # noqa: E402

OUTS = OUT3 / "stepctl"
TAU, LR = 0.05, 3e-3
C1_STEPS = (264, 1304, 2600, 5000)
C2_STEPS = (264, 1300)
DEV_FIELDS = ("R@1", "R@100", "AP", "max_F1", "R@100P", "R@100P_10m", "n_pos")
OLD_RUN = "run_final_K16_tau0.05_lr0.003_s{s}_e40"


# ---------------------------------------------------------------- 학습 고리 (train_one 사본 + 스텝 저장)
def run_steps(td, dev_data, groups, s, W0, seed, max_steps, snap_steps, forbidden):
    import torch
    from _poolalt_probe_20261114 import aligned_sim
    from _invkey_train_linear_20261114 import to_freq_t as tf
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    W = torch.nn.Parameter(W0.clone())
    opt = torch.optim.Adam([W], lr=LR)
    curve, snaps = [], {}
    best = None
    comp_ep1 = None
    used_seq = np.zeros(len(td.keys), bool)
    negstat = np.zeros(3)

    def dev_eval(ep, loss, step):
        per = {}
        with torch.no_grad():
            for k, (spec16, xyz) in dev_data.items():
                per[k] = eval_keys(encode(spec16, s, W.detach()), xyz, chunk=64)
        summ = dev_summary(per, groups)
        curve.append({"epoch": ep, "step": step, "loss": loss, "summary": summ,
                      "per_seq": {k: {m: v[m] for m in DEV_FIELDS} for k, v in per.items()}})
        return summ

    dev_eval(0, None, 0)
    step, ep = 0, 0
    t0 = time.time()
    while step < max_steps:
        ep += 1
        batches, comp = td.epoch(rng)
        if comp_ep1 is None:
            comp_ep1 = comp.copy()
        losses, taken = [], []
        for b in batches:
            a_ids, p_ids, e_ids = td.batch_ids(b, rng)
            ids = torch.cat([a_ids, p_ids, e_ids])
            Bn = len(a_ids)
            used_seq[np.unique(td.seq[ids].cpu().numpy())] = True
            x = td.X[ids].float() / s[None, :, None]
            y = torch.einsum("kr,nrt->nkt", W, x)
            F, nrm = tf(y)
            S = aligned_sim(F[:, :Bn], nrm[:Bn], F, nrm, NFFT)
            with torch.no_grad():
                same = td.seq[a_ids][:, None] == td.seq[ids][None]
                gap = (td.loc[a_ids][:, None] - td.loc[ids][None]).abs()
                dist = (td.xyz[a_ids][:, None] - td.xyz[ids][None]).norm(dim=2)
                neg_same = same & (dist > NEG_M) & (gap > NEG_GAP)
                neg_city = td.city[a_ids][:, None] != td.city[ids][None]
                allow = neg_same | neg_city
                ar = torch.arange(Bn, device=DEV)
                assert not bool(allow[ar, Bn + ar].any())
                allow[ar, Bn + ar] = True
                negstat += [float(neg_same.sum()), float(neg_city.sum()), Bn]
            logits = (S / TAU).masked_fill(~allow, -torch.inf)
            loss = -logits.log_softmax(dim=1)[ar, Bn + ar].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            step += 1
            if step in snap_steps:
                snaps[step] = {"W": W.detach().clone(), "epoch": ep,
                               "epoch_end": len(losses) == len(batches)}
                taken.append(step)
            if step >= max_steps:
                break
        if len(losses) == len(batches):
            summ = dev_eval(ep, float(np.mean(losses)), step)
            key = (summ["sel_R@1_sensor_mean"], summ["AP_seq_mean"])
            if best is None or key > best[0]:
                best = (key, ep, step, W.detach().clone())
            for st in taken:
                snaps[st]["best"] = {"W": best[3], "epoch": best[1], "step": best[2],
                                     "sel": best[0][0], "AP": best[0][1]}
            if ep % 25 == 0 or taken:
                print(f"   ep {ep:4d} step {step:5d} loss {np.mean(losses):.4f} "
                      f"sel {summ['sel_R@1_sensor_mean']:.4f} AP {summ['AP_seq_mean']:.4f} "
                      f"best ep {best[1]} {best[0][0]:.4f}  {time.time() - t0:.0f}s", flush=True)
        else:
            curve.append({"epoch": ep, "step": step, "loss_partial_epoch": float(np.mean(losses)),
                          "batches_done": len(losses), "batches_in_epoch": len(batches)})
            print(f"   stop mid-epoch {ep} at step {step} ({len(losses)}/{len(batches)} batches)", flush=True)
    names = {td.keys[i] for i in np.flatnonzero(used_seq)}
    leak = sorted(names & set(forbidden))
    assert not leak, leak
    info = {"curve": curve,
            "composition_epoch1": {td.keys[i]: dict(zip(DIRS, map(int, comp_ep1[i])))
                                   for i in range(len(td.keys))},
            "neg_per_anchor": {"same_seq": negstat[0] / negstat[2], "other_group": negstat[1] / negstat[2]},
            "leak_check": {"sequences_touched": sorted(names), "forbidden_touched": leak},
            "steps": step, "epochs_started": ep, "train_s": time.time() - t0}
    return snaps, info


def save_snaps(prefix, snaps, s_np, best_too):
    meta = {}
    for st, d in sorted(snaps.items()):
        chmod_write(OUTS / f"{prefix}_st{st}_last.npz", arrays=dict(W=d["W"].cpu().numpy(), s=s_np))
        meta[st] = {"epoch": d["epoch"], "epoch_end": d["epoch_end"]}
        if best_too:
            b = d["best"]
            chmod_write(OUTS / f"{prefix}_st{st}_best.npz", arrays=dict(W=b["W"].cpu().numpy(), s=s_np))
            meta[st]["best"] = {k: b[k] for k in ("epoch", "step", "sel", "AP")}
    return meta


def train1(a):
    """대조 1: 새 범위 (a) 구성을 5000스텝까지."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    recs, unit_of, _ = split3()
    forbidden = list(DEVSET) + list(EVAL)
    st = np.load(OUT3 / "stats.npz")
    assert list(st["train_keys"]) == list(TRAIN)
    s = torch.as_tensor(st["s"], device=DEV)
    td = TrainData(recs, list(TRAIN), forbidden)
    td.w = np.ones(len(TRAIN))                 # arm a: 상한 없음 (scope3 site_weights(None)과 같음)
    td.site_budget = None
    unames = sorted({unit_of[k] for k in TRAIN})
    td.city = torch.as_tensor(np.array([unames.index(unit_of[k]) for k in TRAIN]), device=DEV)[td.seq]
    dev_data = {}
    for k in DEVSET:
        spec, xyz, _ = load_seq(k)
        dev_data[k] = (torch.as_tensor(spec).to(DEV), xyz)
    groups = dev_groups(recs, DEVSET)
    W0 = torch.as_tensor(np.load(OUT3 / "pca_K16.npz")["W"], device=DEV)
    print(f"== c1 seed {a.seed}: new scope arm a, tau {TAU} lr {LR}, steps {C1_STEPS}", flush=True)
    snaps, info = run_steps(td, dev_data, groups, s, W0, a.seed, max(C1_STEPS), set(C1_STEPS), forbidden)
    assert all(snaps[x]["epoch_end"] for x in C1_STEPS)
    info["snapshots"] = save_snaps(f"c1_s{a.seed}", snaps, st["s"], best_too=True)
    info.update({"control": 1, "seed": a.seed, "tau": TAU, "lr": LR, "scope": "new (scope3 arm a)",
                 "train": list(TRAIN), "dev": list(DEVSET), "s_pca": "scope3/stats.npz, scope3/pca_K16.npz",
                 "best_rule": "max dev sel_R@1 then AP over epochs 1..snapshot epoch (first occurrence)"})
    chmod_write(OUTS / f"run_c1_s{a.seed}.json", text=json.dumps(info, indent=1))
    return 0


def train2(a):
    """대조 2: 기존 선형 16 구성을 같은 seed로 1300스텝까지."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    recs, train_keys, dev, held = split()
    forbidden = list(held) + list(dev)
    st = np.load(LIN / "stats.npz")
    assert sorted(st["train_keys"].tolist()) == sorted(train_keys)
    s = torch.as_tensor(st["s"], device=DEV)
    td = TrainData(recs, train_keys, forbidden)    # 기존과 같은 25% 상한, 다른 도시 음성
    dev_data = {}
    for k in dev:
        spec, xyz, _ = load_seq(k)
        dev_data[k] = (torch.as_tensor(spec).to(DEV), xyz)
    groups = dev_groups(recs, dev)
    W0 = torch.as_tensor(np.load(LIN / "pca_K16.npz")["W"], device=DEV)
    for seed in a.seeds:
        print(f"== c2 seed {seed}: old 33-seq config, tau {TAU} lr {LR}, steps {C2_STEPS}", flush=True)
        snaps, info = run_steps(td, dev_data, groups, s, W0, seed, max(C2_STEPS), set(C2_STEPS), forbidden)
        info["snapshots"] = save_snaps(f"c2_s{seed}", snaps, st["s"], best_too=False)
        info.update({"control": 2, "seed": seed, "tau": TAU, "lr": LR, "scope": "old 33 train seqs (linear/)",
                     "train": train_keys, "dev": dev, "s_pca": "linear/stats.npz, linear/pca_K16.npz",
                     "sample_fraction": dict(zip(td.keys, map(float, td.w)))})
        chmod_write(OUTS / f"run_c2_s{seed}.json", text=json.dumps(info, indent=1))
    return 0


# ---------------------------------------------------------------- 평가
def model_list():
    """이름 -> (파일, s 출처). 행 묶음 이름은 이름에서 _s<seed>를 뗀 것."""
    m = {}
    for i in (0, 1, 2):
        m[f"old33_st5085_s{i}"] = (LIN / (OLD_RUN.format(s=i) + ".npz"), "old")
        for st in C2_STEPS:
            m[f"old33_st{st}_s{i}"] = (OUTS / f"c2_s{i}_st{st}_last.npz", "old")
        m[f"new3_st264sel_s{i}"] = (OUT3 / (new_run("a", i) + ".npz"), "new")
        for st in C1_STEPS:
            for kind in ("last", "best"):
                m[f"new3_st{st}{kind}_s{i}"] = (OUTS / f"c1_s{i}_st{st}_{kind}.npz", "new")
    return m


def load_reps():
    import torch
    so = np.load(LIN / "stats.npz")["s"]
    sn = np.load(OUT3 / "stats.npz")["s"]
    reps, files = {}, {}
    for name, (f, src) in model_list().items():
        z = np.load(f)
        s = so if src == "old" else sn
        assert np.array_equal(z["s"], s), f
        reps[name] = (torch.as_tensor(s, device=DEV), torch.as_tensor(z["W"], device=DEV))
        files[name] = str(f.relative_to(REPO))
    return reps, files


def evaluate(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from _fullmap_probe_20261114 import CONDS, _make_loader, deform
    from _invkey_augment_20261115 import augment, seed_for
    from _poolalt_probe_20261114 import aligned_sim, describe
    from _scatter_probe_20261114 import bank, occupancy
    split3()
    reps, files = load_reps()
    seqs = list(EVAL) + list(KTH) + list(DEVSET)
    out = {"reps": list(reps), "files": files, "per_seq": {}, "retention": {}, "tf32": False}
    t0 = time.time()
    for k in seqs:
        spec, xyz, _ = load_seq(k)
        out["per_seq"][k] = {}
        for rep, (s, W) in reps.items():
            with torch.no_grad():
                out["per_seq"][k][rep] = eval_keys(encode(spec, s, W), xyz, chunk=64)
            torch.cuda.empty_cache()
        print(f"{k:22s} done {time.time() - t0:.0f}s", flush=True)
    chmod_write(OUTS / "eval.json", text=json.dumps(out, indent=1))
    # 1위 유지율: scope3 retention과 같은 질의·난수 흐름, occl_90·near_bus만
    F = torch.from_numpy(bank()).to(DEV)
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
        conds = ("occl_90", "near_bus")
        keep = {r: {c: [] for c in conds + ("real",)} for r in reps}
        for kq, (q, j) in enumerate(pairs):
            sid = int(ids[q])
            A = loader[sid]["points"]
            dd = {c: deform(A, c, rng) for c in CONDS}
            aug, _ = augment(A, np.random.default_rng(seed_for(key, sid, "ret_near_bus")), sensor, "near_bus")
            X = np.stack([describe(occupancy(dd["occl_90"]), F)[1], describe(occupancy(aug), F)[1]])
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
        out["retention"][key] = {"n_queries": len(pairs),
                                 "retention": {r: {c: float(np.mean(v)) for c, v in d.items()}
                                               for r, d in keep.items()}}
        print(f"{key} retention done ({len(pairs)}) {time.time() - t0:.0f}s", flush=True)
        del db
        torch.cuda.empty_cache()
    out["elapsed_s"] = time.time() - t0
    chmod_write(OUTS / "eval.json", text=json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- 호스트: 모으기
def report(a):
    E = json.load(open(OUTS / "eval.json"))
    S3 = json.load(open(OUT3 / "metrics.json"))
    per, reps = E["per_seq"], E["reps"]
    G = {**GROUPS, "k360_no09": ["kitti360/0004", "kitti360/0006"], "mcd_kth": list(KTH),
         "dev_kitti": list(DEVSET)}
    gm = group_means(per, G, reps)
    fams = {}
    for r in reps:
        fams.setdefault(r.rsplit("_s", 1)[0], []).append(r)
    cols = {"kitti_k360": ("R@1_pooled", "AP", "R@100P", "R@1_same_pooled", "R@1_reverse_pooled",
                           "R@1_crossing_pooled"),
            "k360_no09": ("R@1_pooled", "AP", "R@100P"),
            "mcd_tuhh": ("R@1_pooled", "AP", "R@100P", "R@1_same_pooled", "R@1_reverse_pooled",
                         "R@1_crossing_pooled"),
            "mcd_kth": ("R@1_pooled",), "dev_kitti": ("R@1_pooled",)}
    table = {}
    for fn, rs in fams.items():
        table[fn] = {g: {m: spread_of([gm[g][r][m] for r in rs]) for m in ms} for g, ms in cols.items()}
        for key in RET_SEQS:
            for c in ("occl_90", "near_bus"):
                table[fn][f"ret|{key}|{c}"] = spread_of([E["retention"][key]["retention"][r][c] for r in rs])
    # 점검
    runs = {}
    for p in sorted(OUTS.glob("run_c*_s*.json")):
        d = json.load(open(p))
        runs[p.stem[4:]] = d
    chk = {}
    for i in (0, 1, 2):
        # 대조 1: 앞 33에폭 = scope3 final_a, 264스텝 가중치 = final_a 가중치
        d = runs[f"c1_s{i}"]
        ref = json.load(open(OUT3 / (new_run("a", i) + ".json")))
        mine = [c for c in d["curve"] if c["epoch"] <= 33]
        dd, nu = max_abs_diff([c["per_seq"] for c in ref["curve"]],
                              [{k: {m: v[m] for m in ref["curve"][0]["per_seq"][k]} for k, v in c["per_seq"].items()}
                               for c in mine])
        dl = max(abs(u["loss"] - v["loss"]) for u, v in zip(ref["curve"][1:], mine[1:]))
        w_eq = np.array_equal(np.load(OUTS / f"c1_s{i}_st264_last.npz")["W"],
                              np.load(OUT3 / (new_run("a", i) + ".npz"))["W"])
        chk[f"c1_s{i}"] = {"dev_curve_33ep_max_abs_diff": dd, "n_unequal": nu, "loss_max_abs_diff": dl,
                           "W_st264_equals_final_a": w_eq,
                           "comp_epoch1_equal": d["composition_epoch1"] == ref["composition_epoch1"],
                           "forbidden_touched": d["leak_check"]["forbidden_touched"]}
        # 대조 2: 끝난 에폭마다 기존 run과 같은지
        d = runs[f"c2_s{i}"]
        ref = json.load(open(LIN / (OLD_RUN.format(s=i) + ".json")))
        full = [c for c in d["curve"] if "summary" in c]
        dd, nu = max_abs_diff([c["per_seq"] for c in ref["curve"][:len(full)]], [c["per_seq"] for c in full])
        dl = max(abs(u["loss"] - v["loss"]) for u, v in zip(ref["curve"][1:len(full)], full[1:]))
        chk[f"c2_s{i}"] = {"full_epochs_compared": len(full) - 1, "dev_max_abs_diff": dd, "n_unequal": nu,
                           "loss_max_abs_diff": dl,
                           "comp_epoch1_equal": d["composition_epoch1"] == ref["composition_epoch1"],
                           "forbidden_touched": d["leak_check"]["forbidden_touched"],
                           "snapshots": d["snapshots"]}
    # 기존 수치 재현: 기존 선형 16·새 (a)의 검색·유지율
    rep_chk = {}
    for i in (0, 1, 2):
        for mine, old in ((f"old33_st5085_s{i}", f"lin16_old_s{i}"), (f"new3_st264sel_s{i}", f"new_a_s{i}")):
            dd, nu = max_abs_diff({k: S3["per_seq"][k][old] for k in per}, {k: per[k][mine] for k in per})
            rr = {key: {c: [S3["retention"]["per_seq"][key]["retention"][old][c],
                            E["retention"][key]["retention"][mine][c]] for c in ("occl_90", "near_bus", "real")}
                  for key in RET_SEQS}
            rep_chk[mine] = {"search_max_abs_diff": dd, "n_unequal": nu,
                             "retention_equal": all(v[0] == v[1] for x in rr.values() for v in x.values())}
    # 대조 1 곡선 요약
    c1 = {}
    for i in (0, 1, 2):
        cv = [c for c in runs[f"c1_s{i}"]["curve"] if "summary" in c]
        c1[f"s{i}"] = {"snapshots": runs[f"c1_s{i}"]["snapshots"],
                       "curve_every25": [{"epoch": c["epoch"], "step": c["step"],
                                          "loss": c["loss"], "sel": c["summary"]["sel_R@1_sensor_mean"],
                                          "AP": c["summary"]["AP_seq_mean"]}
                                         for c in cv if c["epoch"] % 25 == 0 or c["epoch"] in (1, 33, 163, 325)]}
    out = {"date": "2026-11-15", "tau": TAU, "lr": LR, "c1_steps": C1_STEPS, "c2_steps": C2_STEPS,
           "groups": G, "files": E["files"], "per_seq": per, "group_means": gm, "table": table,
           "retention": E["retention"], "checks": {"training_reproduces": chk, "eval_reproduces_scope3": rep_chk},
           "c1_curves": c1,
           "c2_snapshot_epochs": {f"s{i}": runs[f"c2_s{i}"]["snapshots"] for i in (0, 1, 2)},
           "neg_per_anchor": {k: v["neg_per_anchor"] for k, v in runs.items()},
           "raw_reference": {g: {m: S3["group_means"][g]["raw"][m] for m in ("R@1_pooled", "AP", "R@100P")}
                             for g in ("kitti_k360", "mcd_tuhh")}}
    chmod_write(OUTS / "metrics.json", text=json.dumps(out, indent=1))
    print(json.dumps(chk, indent=0, default=str)[:3000])
    print(json.dumps(rep_chk))
    f = lambda x: f"{100 * x['mean']:.1f} ({100 * x['min']:.1f}–{100 * x['max']:.1f})"  # noqa: E731
    order = ["new3_st264sel", "new3_st264last", "new3_st264best", "new3_st1304last", "new3_st1304best",
             "new3_st2600last", "new3_st2600best", "new3_st5000last", "new3_st5000best",
             "old33_st264", "old33_st1300", "old33_st5085"]
    hdr = ["K R@1모음", "K AP", "K R@100P", "K-09 R@1모음", "K-09 AP", "K-09 R@100P", "K 같은", "K 반대", "K 교차",
           "T R@1모음", "T AP", "T R@100P", "T 같은", "T 반대", "T 교차", "kth R@1모음", "dev R@1모음",
           "k360 occl90", "k360 near_bus", "tuhh occl90", "tuhh near_bus"]
    print("\n| 행 | " + " | ".join(hdr) + " |")
    print("|---|" + "---:|" * len(hdr))
    for fn in order:
        t = table[fn]
        vals = [t["kitti_k360"][m] for m in cols["kitti_k360"][:3]] + [t["k360_no09"][m] for m in cols["k360_no09"]] \
            + [t["kitti_k360"][m] for m in cols["kitti_k360"][3:]] + [t["mcd_tuhh"][m] for m in cols["mcd_tuhh"]] \
            + [t["mcd_kth"]["R@1_pooled"], t["dev_kitti"]["R@1_pooled"]] \
            + [t[f"ret|{k}|{c}"] for k in RET_SEQS for c in ("occl_90", "near_bus")]
        print(f"| {fn} | " + " | ".join(f(v) for v in vals) + " |")
    print("\nc1 curves (s0):", json.dumps(c1["s0"]["curve_every25"][:40]))
    for i in (0, 1, 2):
        print(f"c1 s{i} snaps", c1[f"s{i}"]["snapshots"])
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train1")
    p.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("train2")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p = sub.add_parser("eval")
    p.add_argument("--n", type=int, default=300)
    sub.add_parser("report")
    a = ap.parse_args()
    return {"train1": train1, "train2": train2, "eval": evaluate, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
