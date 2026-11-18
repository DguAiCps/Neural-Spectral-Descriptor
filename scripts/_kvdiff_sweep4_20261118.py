#!/usr/bin/env python3
"""차이 K/V 입력 4차 탐색(시드 1~3): 이웃 범위와 초기화를 두 조건 모두 같이 바꾸면 차이 계열과 절대 K/V(abskv)의 순위가 바뀌는가.

변형(1차 스크립트 scripts/_kvdiff_variants_20261118.py = V의 구현 그대로): sym(NSD), abskv, diffk_absv.
학습 설정은 모두 기본(512갱신, lr 1e-3, 게이트 고정, CFG x=log__n=ln__g=frozen__lr=1e-3__U=1536). 모두 로컬 GPU.

조건(조건마다 세 변형을 같은 그래프·표본·보정·학습 설정으로 학습):
 (A) 이웃 범위. 정제 그래프(DiffAttnConv 두 층과 복소 차이 전달)의 이웃 오프셋만 바꾸고, 시간 증거 plan(±5)·학습 표본
     (P.NEWPOOLS)·융합·lambda는 그대로다. 구현은 scripts/_dilated_neighbors_20261117.py(D)의 패치(temporal_edges,
     attach_transport, transport_difference, ContrastSpectralDifference.forward)를 import로 그대로 건다.
   d2   오프셋 ±2,±4,…,±10(D의 NSD_DIL=d2). 학습8 그래프·손실 보정·xstats·선택 결과는 기존
        results/pdv2_dilated_20261117/d2 의 것을 읽기만 한다(기존 NSD-D2와 같은 입력). 최종 판정 지연 2R+5 = 25.
        원고 구성(지연 15)의 대체가 아니라 진단이다.
   n3   오프셋 ±1,±2,±3. D를 NSD_DIL=d1로 import한 뒤 D의 모듈 전역(오프셋·칸·LOOKUP·LAG·결과 폴더)을 ±3으로 바꾼다.
        D의 함수는 호출 시점에 모듈 전역을 읽으므로 같은 코드가 ±3으로 동작한다. 학습8 그래프(D.graphs)와 손실 보정
        (D.calibrate, 이 조건의 균등 감산 초기 출력에 같은 절차로 적합)을 sweep4/n3 에 새로 만든다. 지연 2*3+5 = 11.
 (B) 초기화(이웃 ±5, 패치 없음, 손실 보정·그래프는 기본 조건과 같다).
   kv_init_x0.3 / kv_init_x3  두 층의 W_k·W_v 초기 가중치(편향 없음)를 0.3배 / 3배. W_q와 기하 MLP 최종층은 0 그대로라
                               초기 출력은 균등 감산과 같다(학습 전 검사).
   q_init      W_q를 0으로 두지 않고 모듈 기본 초기화(같은 시드의 run.source.factory가 만든 W_q, 0으로 지우기 전 값)를 쓴다.
               torch 난수 상태는 fork_rng로 보존한다. 초기 출력이 균등 감산과 달라지므로 init 검사는 이 조건에서만 기록만 하고
               통과를 요구하지 않는다. 손실 보정은 기본 조건의 기존 값을 그대로 쓴다.
 기본 조건(base) 값은 sweep2 base 원장(로컬 5080)을 읽기만 한다(재학습 없음).

이웃 모드(base/d2/n3)는 프로세스 전역 패치라 환경 변수 KV4_NB로 고른다(조건과 모드가 맞지 않으면 멈춘다).
학습(train_run)은 C.train(scripts/_report_model_controls_20261116.py)과 같은 순서·난수·계산이고 다음만 다르다:
 초기화 조건의 초기값 변경, q_init의 init 검사 비강제, sym 재현 검사는 d2에서만(기존 D2 체크포인트가 있는 시드 4).

docker exec [-e KV4_NB=d2|n3] -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_kvdiff_sweep4_20261118.py <action>
  n3_prepare | train --cond C --variant V --seed K | train_campaign --conds A,B | models --mode M --out FILE
  | pipe --role R --sequence S --models FILE | pipe_campaign --role R --models FILE [--prefix N] | summary
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _kvdiff_variants_20261118 as V  # noqa: E402  (C 전역을 sweep1로 바꾸므로 아래에서 다시 바꾼다)

C, S, P = V.C, V.S, V.P
run, prior = P.run, P.prior
ROOT = C.ROOT
OUT = ROOT / "results/kv_diff_improve_20261118/sweep4"
RUNS, LEDGERS = OUT / "runs", OUT / "ledgers"
SWEEP2 = ROOT / "results/kv_diff_improve_20261118/sweep2"
DIL = ROOT / "results/pdv2_dilated_20261117"
S_LEDGERS0 = S.LEDGERS          # 기존 탐색 원장(±5 uniform), D import 전에 잡아 둔다
VARS = ("sym", "abskv", "diffk_absv")
DIFF = ("sym", "diffk_absv")
SEEDS = (1, 2, 3)
CFG = V.CFG
TOTAL = 512
assert CFG == "x=log__n=ln__g=frozen__lr=1e-3__U=1536" and C.U == TOTAL and S.parse(CFG)["sch"] == "const"
COND = {"d2": "d2", "n3": "n3", "kv_init_x0.3": "base", "kv_init_x3": "base", "q_init": "base"}  # 조건 -> 이웃 모드
SCALE = {"kv_init_x0.3": 0.3, "kv_init_x3": 3.0}
EXTRA = [("sym", "d2", 4)]       # 기존 D2(시드 4) 수준 확인용
UNIFORM = {"d2": "uniform_d2", "n3": "uniform_n3", "kv_init_x0.3": "uniform_base", "kv_init_x3": "uniform_base", "q_init": "uniform_base"}

NB = os.environ.get("KV4_NB", "base")
assert NB in ("base", "d2", "n3"), NB
D = None
if NB in ("d2", "n3"):
    os.environ["NSD_DIL"] = "d2" if NB == "d2" else "d1"
    import _dilated_neighbors_20261117 as D  # noqa: E402  (import 때 패치가 걸리고 S/P 경로가 D의 폴더로 바뀐다)
    if NB == "n3":
        D.VAR = "n3"
        D.POS_OFFS = [1, 2, 3]
        D.OFFSETS = np.array(sorted([-o for o in D.POS_OFFS] + D.POS_OFFS), dtype=np.int64)
        D.R = int(D.OFFSETS.max())
        D.SLOT_OFFS = sorted(D.OFFSETS.tolist() + [0])
        D.NSLOT = len(D.SLOT_OFFS)
        D.SELF = D.SLOT_OFFS.index(0)
        D.LOOKUP = np.full(2 * D.R + 1, -1, np.int64)
        for _k, _o in enumerate(D.SLOT_OFFS):
            D.LOOKUP[_o + D.R] = _k
        D.LAG = 2 * D.R + 5
        D.VD = OUT / "n3"
        assert not D._LOOKUP_T
        S.OUT, S.RUNS, S.LEDGERS = D.VD, D.VD / "runs", D.VD / "ledgers"
        P.OUT = D.VD / "pilot"
        P.recal = types.SimpleNamespace(OUT=D.VD)
    else:
        assert D.VD == DIL / "d2" and D.LAG == 25 and D.OFFSETS.tolist() == [-10, -8, -6, -4, -2, 2, 4, 6, 8, 10]

C.OUT, C.LEDGERS = OUT, LEDGERS


def tag(variant: str, cond: str, seed: int) -> str:
    return f"{variant}@{cond}@s{seed}"


def ckpt(cond: str, variant: str, seed: int) -> Path:
    return RUNS / cond / variant / f"s{seed}" / f"u{TOTAL}.pth"


def build4(cond: str, variant: str, seed: int, stats: dict = None) -> nn.Module:
    """V.build(같은 순서·난수) 뒤 초기화 조건만 적용."""
    net = C.build(variant, seed, stats)
    with torch.no_grad():
        if cond in SCALE:
            for conv in net.convs:
                assert conv.W_k.bias is None and conv.W_v.bias is None
                conv.W_k.weight.mul_(SCALE[cond])
                conv.W_v.weight.mul_(SCALE[cond])
        elif cond == "q_init":
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                ref = run.source.factory(288)
            for conv, rc in zip(net.convs, ref.gnn.convs):
                assert float(conv.W_q.weight.abs().max()) == 0 and conv.W_q.bias is None and rc.W_q.bias is None
                conv.W_q.weight.copy_(rc.W_q.weight)
            del ref
    return net


# ----------------------------------------------------------------- n3 입력
def n3_prepare() -> None:
    assert NB == "n3"
    D.graphs()
    D.calibrate()


# ----------------------------------------------------------------- 학습
def train_run(cond: str, variant: str, seed: int) -> None:
    assert COND[cond] == NB, (cond, NB)
    cfg = S.parse(CFG)
    folder = RUNS / cond / variant / f"s{seed}"
    if (folder / "training_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    _, lam = P.selected()
    fpath = C.fit_path(variant)
    fitrec = run.READ(fpath)["fit"]
    net = build4(cond, variant, seed, S.xstats())
    if variant == "sym" and cond not in SCALE and cond != "q_init":  # S.build와 같은 초기 모델인지
        ref = S.build(cfg, seed, S.xstats())
        assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), ref.state_dict().values()))
        del ref
    assert not [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]
    for n_, p in net.named_parameters():
        p.requires_grad_(S.trainable(cfg, n_))
    graphs = {s: C.train_graph(variant, s) for s in C.TRAIN}
    with torch.no_grad():
        init = {s: bool(torch.equal(net(g), prior.filtered(g, -0.5))) for s, g in graphs.items()}
        init_err = {s: float((net(g) - prior.filtered(g, -0.5)).abs().max()) for s, g in graphs.items()}
    if cond != "q_init":
        assert init[C.TRAIN[0]], init_err
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    pls = {s: run.ARR(C.pool_dir(variant) / "train" / s.replace("/", "_") / "pool.npz") for s in C.TRAIN}
    plans = {s: {k: torch.from_numpy(d["plan_" + k]).cuda() for k in ("query", "candidate", "weight")} for s, d in pls.items()}
    batch = int(cfg["B"])
    records, updates, tick = [], 0, time.time()
    for epoch in range(1, TOTAL // 64 + 1):
        for ordinal, s in enumerate(C.TRAIN):
            data = pls[s]
            sets = [np.flatnonzero(data["eligible"] & (data["correct"] == c)) for c in (0, 1)]
            sets = [v for v in sets if len(v)]
            rng = np.random.default_rng((seed, epoch, ordinal))
            losses, grads = [], []
            for _ in range(8):
                n = batch // len(sets)
                chosen = rng.permutation(np.concatenate([rng.choice(v, n, replace=len(v) < n) for v in sets]))
                ix = torch.from_numpy(chosen).cuda()
                for group in opt.param_groups:
                    group["lr"] = S.lr_at(cfg, updates)
                opt.zero_grad(set_to_none=True)
                z = net(graphs[s])
                value = P.score(z, {k: v[ix] for k, v in plans[s].items()}, lam)
                logit = (value - fitrec["mean"][0]) * fitrec["coefficients"][0] / fitrec["scale"][0] + fitrec["intercept"]
                loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(data["correct"][chosen]).cuda().float())
                assert torch.isfinite(loss)
                loss.backward()
                g = torch.nn.utils.clip_grad_norm_(params, 1.0)
                assert torch.isfinite(g)
                opt.step()
                updates += 1
                losses.append(float(loss.detach()))
                grads.append(float(g))
            st = net.last_strengths.detach()
            records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)), grad=float(np.mean(grads)),
                                strength_mean=float(st.mean()), strength_min=float(st.min()), strength_max=float(st.max())))
    assert updates == TOTAL
    path = ckpt(cond, variant, seed)
    torch.save(dict(model_state_dict=net.state_dict(), updates=updates, cfg=cfg, variant=variant, seed=seed, cond=cond), path)
    seconds = time.time() - tick
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, initial[k])]
    repro = None
    old = S.RUNS / S.canon(CFG) / f"s{seed}" / f"u{TOTAL}.pth"
    if variant == "sym" and cond == "d2" and old.exists():  # 기존 NSD-D2 체크포인트(다른 GPU)와 비교
        o = torch.load(old, map_location="cuda", weights_only=False)["model_state_dict"]
        repro = dict(reference=str(old.relative_to(ROOT)), state_bitwise_equal_existing=all(torch.equal(v, o[k]) for k, v in net.state_dict().items()),
                     max_abs=max(float((v.float() - o[k].float()).abs().max()) for k, v in net.state_dict().items()))
    del graphs, plans, opt
    per = {}
    with torch.no_grad():
        for s in C.VAL:
            vname = s.replace("/", "_")
            graph, _, _ = prior.graph_inputs("val", s)
            data = run.ARR(C.pool_dir(variant) / "val" / vname / "pool.npz")
            row = {k: data[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist")}
            plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
            z = net(graph)
            v = P.batched(lambda pp: P.score(z, pp, lam).cpu().numpy(), plan, len(row["query_idx"])).astype(np.float64)
            per[s] = run.base.independent_metrics(row, v)
            del graph
    P.write(folder / "training_complete.json", dict(
        status="passed", cond=cond, neighbor_mode=NB, offsets=(D.OFFSETS.tolist() if D is not None else list(range(-5, 0)) + list(range(1, 6))),
        lag=(D.LAG if D is not None else 15), variant=variant, kind=C.VARIANTS[variant]["kind"], cfg=cfg, name=CFG, seed=seed,
        updates=updates, checkpoint=path.name, sha256=run.HASH(path), records=records, changed=changed,
        init_bitwise_equal_uniform=init, init_max_abs_vs_uniform=init_err, init_check_enforced=cond != "q_init",
        init_scale=SCALE.get(cond), trainable_parameters=sum(p.numel() for p in params), total_parameters=sum(p.numel() for p in net.parameters()),
        fit=fitrec, fit_path=str(fpath.relative_to(ROOT)), pools=str(C.pool_dir(variant).relative_to(ROOT)),
        train_inputs=str((P.recal.OUT / "train_inputs").relative_to(ROOT)), reproduction_vs_existing_d2=repro,
        val_fixed=dict(macro=run.base.macro(per)["macro"], sequences=per),
        seconds=seconds, torch=torch.__version__, gpu=torch.cuda.get_device_name()))
    print("TRAIN", cond, variant, seed, round(seconds, 1), "init", all(init.values()), round(max(init_err.values()), 6), "repro", repro,
          "valfixedAP", round(100 * run.base.macro(per)["macro"]["AP"], 3), flush=True)


def jobs(conds: list) -> list:
    out = [(c, v, k) for c in conds for v in VARS for k in SEEDS]
    return out + [(c, v, k) for v, c, k in EXTRA if c in conds]


def train_campaign(conds: list) -> None:
    logs = OUT / "logs" / "train"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    for c, v, k in jobs(conds):
        if (RUNS / c / v / f"s{k}" / "training_complete.json").exists():
            continue
        with (logs / f"{c}_{v}_s{k}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "train", "--cond", c, "--variant", v, "--seed", str(k)],
                           stdout=f, stderr=subprocess.STDOUT, check=True, env=dict(os.environ, KV4_NB=COND[c]))
        print("TRAINED", c, v, k, round(time.time() - tick, 1), flush=True)
    P.write(OUT / "logs" / f"train_campaign_{'_'.join(conds)}.done.json", dict(seconds=time.time() - tick))


# ----------------------------------------------------------------- 실제 검색 (C.pipe 경로, 모델 해석만 바꿈)
def model_spec(m: str):
    if m == "uniform_n3":
        assert NB == "n3"
        return "sym", None, None
    variant, cond, seed = m.split("@")
    assert variant in VARS and COND[cond] == NB, (m, NB)
    net = C.build(variant, int(seed[1:]))
    net.load_state_dict(torch.load(ckpt(cond, variant, int(seed[1:])), map_location="cuda", weights_only=False)["model_state_dict"],
                        strict=True)
    return variant, net, None


C.model_spec = model_spec


def write_models(mode: str, path: str) -> None:
    conds = [c for c, m in COND.items() if m == mode]
    names = [tag(v, c, k) for c, v, k in jobs(conds)] + (["uniform_n3"] if mode == "n3" else [])
    Path(path).write_text("\n".join(names) + "\n")


def pipe_campaign(role: str, models_file: str, workers: int, nprefix: int) -> None:
    seqs = C.VAL if role == "val" else C.TESTSEQ
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()

    def call(s):
        with (logs / f"pipe_{role}_{s.replace('/', '_')}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "pipe", "--role", role, "--sequence", s, "--models", models_file,
                            "--prefix", str(nprefix)], stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(concurrent.futures.as_completed([ex.submit(call, s) for s in seqs]), 1):
            fut.result()
            print("DONE", role, i, len(seqs), round(time.time() - tick, 1), flush=True)
    P.write(logs / f"pipe_campaign_{role}_{Path(models_file).stem}.done.json", dict(seconds=time.time() - tick, workers=workers, prefix=nprefix))


# ----------------------------------------------------------------- 집계 (sweep2·3 summary와 같은 규약)
def summary() -> None:
    import _pdv2_scratch_search_compare_20261116 as K

    assert NB == "base"
    keys, metrics = K.KEYS, C.METRICS
    oldd2 = f"{CFG}@s{{}}@u512"

    def led(role, name, t):
        if t == "uniform_base":
            return S_LEDGERS0 / role / name / "uniform.npz"
        if t == "uniform_d2":
            return DIL / "d2/ledgers" / role / name / "uniform.npz"
        if t.startswith("oldD2@s"):
            return DIL / "d2/ledgers" / role / name / (oldd2.format(t.split("@s")[1]) + ".npz")
        if "@base@" in t:
            return SWEEP2 / "ledgers" / role / name / (t + ".npz")
        return LEDGERS / role / name / (t + ".npz")

    arms = {c: {v: [tag(v, c, k) for k in SEEDS] for v in VARS} for c in list(COND) + ["base"]}
    arms["ref"] = {"uniform_base": ["uniform_base"], "uniform_d2": ["uniform_d2"], "uniform_n3": ["uniform_n3"],
                   "oldD2_nsd_s456": [f"oldD2@s{k}" for k in (4, 5, 6)], "oldD2_nsd_s4": ["oldD2@s4"], "sym_d2_s4": ["sym@d2@s4"]}
    alltags = sorted({t for a in arms.values() for ts in a.values() for t in ts})
    allseq = [s for v in K.CITY.values() for s in v]
    val_pm, rows = {}, {}
    for t in alltags:
        val_pm[t] = S.macro("val", {s: run.READ(led("val", s.replace("/", "_"), t).with_suffix(".json")) for s in C.VAL})
        for s in allseq:
            name = s.replace("/", "_")
            d = dict(np.load(K.DIRS / (name + ".npz")))
            with np.load(led("test", name, t)) as z:
                np.testing.assert_array_equal(z["query_idx"], d["query_idx"])
                rows.setdefault(t, {})[s] = K.metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"])

    def macro(t):
        cities = []
        for seqs in K.CITY.values():
            v = [rows[t][s] for s in seqs if rows[t][s] is not None]
            if v:
                cities.append({k: np.mean([x[k] for x in v]) for k in keys})
        return {k: float(np.mean([c[k] for c in cities])) for k in keys}
    test_pm = {t: macro(t) for t in alltags}

    def stat(pm, ts, ks):
        return {k: dict(mean=float(np.mean([pm[t][k] for t in ts])), sd=float(np.std([pm[t][k] for t in ts], ddof=1)) if len(ts) > 1 else None,
                        values=[pm[t][k] for t in ts]) for k in ks}
    val = {c: {v: stat(val_pm, ts, metrics) for v, ts in a.items()} for c, a in arms.items()}
    test = {c: {v: stat(test_pm, ts, keys) for v, ts in a.items()} for c, a in arms.items()}

    def boot(A, B):
        """시퀀스별 (A 시드 평균 - B 시드 평균), 도시 안 시퀀스 재표집 10,000회, 도시 평균, 난수 20261104(비교마다 새로)."""
        rng = np.random.default_rng(20261104)
        draws, city = [], {}
        for cname, seqs in K.CITY.items():
            avail = [s for s in seqs if rows[A[0]][s] is not None]
            if not avail:
                continue
            mat = np.array([[100 * (np.mean([rows[t][s][k] for t in A]) - np.mean([rows[t][s][k] for t in B])) for k in keys] for s in avail])
            city[cname] = dict(zip(keys, mat.mean(0).tolist()))
            idx = rng.integers(0, len(avail), size=(10000, len(avail)))
            draws.append(mat[idx].mean(1))
        b = np.stack(draws).mean(0)
        point = {k: float(np.mean([city[c][k] for c in city])) for k in keys}
        return dict(point_pp=point, conditional_95_pp={k: np.quantile(b[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(keys)},
                    city_delta_pp=city)

    def vdiff(A, B):
        return {k: 100 * (np.mean([val_pm[t][k] for t in A]) - np.mean([val_pm[t][k] for t in B])) for k in metrics}

    def comp(A, B):
        r = boot(A, B)
        for k in keys:
            assert abs(r["point_pp"][k] - 100 * (np.mean([test_pm[t][k] for t in A]) - np.mean([test_pm[t][k] for t in B]))) < 1e-9
        return dict(test46=r, val9_mean_diff_pp=vdiff(A, B))

    diffs = {}
    for c in COND:
        a, u = arms[c], arms["ref"][UNIFORM[c]]
        diffs[c] = {"sym-abskv": comp(a["sym"], a["abskv"]), "diffk_absv-abskv": comp(a["diffk_absv"], a["abskv"])}
        for v in VARS:
            diffs[c][f"{v}-{UNIFORM[c]}"] = comp(a[v], u)
            diffs[c][f"{v}@{c}-{v}@base"] = comp(a[v], arms["base"][v])
    diffs["base"] = {"sym-abskv": comp(arms["base"]["sym"], arms["base"]["abskv"]),
                     "diffk_absv-abskv": comp(arms["base"]["diffk_absv"], arms["base"]["abskv"])}
    for v in VARS:
        diffs["base"][f"{v}-uniform_base"] = comp(arms["base"][v], arms["ref"]["uniform_base"])
    out = dict(conditions={c: dict(neighbor_mode=m, init_scale=SCALE.get(c), uniform_reference=UNIFORM[c]) for c, m in COND.items()},
               variants=list(VARS), seeds=list(SEEDS), fixed=dict(weights=P.selected()[0].tolist(), lam=P.selected()[1]), arms=arms)
    out["val"], out["test"] = val, dict(arms=test, per_model=test_pm)
    out["val_per_model"] = val_pm
    out["differences"] = diffs
    out["metric_checks"] = {t: max(abs(S.macro("test", {s: run.READ(led("test", s.replace("/", "_"), t).with_suffix(".json"))
                                                        for s in C.TESTSEQ})[k] - test_pm[t][k]) for k in keys) for t in alltags}
    out["diff_family_ge_abskv"] = {c: {a: dict(val9_AP=val[c][a]["AP"]["mean"] >= val[c]["abskv"]["AP"]["mean"],
                                               test46_AP=test[c][a]["AP"]["mean"] >= test[c]["abskv"]["AP"]["mean"]) for a in DIFF}
                                   for c in list(COND) + ["base"]}
    out["abskv_better_than_base_abskv"] = {c: dict(val9_AP=val[c]["abskv"]["AP"]["mean"] > val["base"]["abskv"]["AP"]["mean"],
                                                   test46_AP=test[c]["abskv"]["AP"]["mean"] > test["base"]["abskv"]["AP"]["mean"])
                                           for c in COND}
    # base 값이 sweep2 요약과 같은지(같은 원장을 읽었는지)
    s2 = run.READ(SWEEP2 / "summary.json")
    out["base_matches_sweep2_summary"] = {v: dict(
        val9=max(abs(val["base"][v][k]["mean"] - s2["val"]["base"][v][k]["mean"]) for k in metrics),
        test46=max(abs(test["base"][v][k]["mean"] - s2["test"]["arms"]["base"][v][k]["mean"]) for k in keys)) for v in VARS}
    # D2 수준 확인: 기존 NSD-D2(서버 학습, 시드 4~6)와 이번 sym@d2(로컬, 시드 1~3, 시드 4)
    an = run.READ(DIL / "analysis.json")
    tr4 = run.READ(RUNS / "d2/sym/s4/training_complete.json")
    out["d2_level_check"] = dict(
        existing_analysis_test46_pct=dict(d2=an["all"]["summary"]["d2"], d2_uniform=an["all"]["summary"]["d2_uniform"]),
        existing_analysis_val9_pct=dict(d2=an["val9"]["d2"], d2_uniform=an["val9"]["d2_uniform"]),
        recomputed_here=dict(oldD2_nsd_s456=dict(val9=val["ref"]["oldD2_nsd_s456"], test46=test["ref"]["oldD2_nsd_s456"]),
                             uniform_d2=dict(val9=val["ref"]["uniform_d2"], test46=test["ref"]["uniform_d2"])),
        sym_d2_s123_minus_oldD2_s456=comp(arms["d2"]["sym"], arms["ref"]["oldD2_nsd_s456"]),
        sym_d2_s4_minus_oldD2_s4=comp(["sym@d2@s4"], ["oldD2@s4"]),
        sym_d2_s4_checkpoint_vs_existing=tr4["reproduction_vs_existing_d2"],
        note="기존 D2는 서버(스테이징) 학습, 이번은 로컬 RTX 5080. 같은 코드·입력이어도 GPU가 달라 비트 일치는 기대하지 않는다.")
    tr = {}
    for c, v, k in jobs(list(COND)):
        rec = run.READ(RUNS / c / v / f"s{k}" / "training_complete.json")
        assert rec["sha256"] == run.HASH(ckpt(c, v, k))
        tr[tag(v, c, k)] = dict(
            checkpoint=str(ckpt(c, v, k).relative_to(ROOT)), sha256=rec["sha256"], neighbor_mode=rec["neighbor_mode"], offsets=rec["offsets"],
            lag=rec["lag"], fit_path=rec["fit_path"], train_inputs=rec["train_inputs"], trainable_parameters=rec["trainable_parameters"],
            init_bitwise_equal_uniform=all(rec["init_bitwise_equal_uniform"].values()), init_max_abs=max(rec["init_max_abs_vs_uniform"].values()),
            init_check_enforced=rec["init_check_enforced"], gpu=rec["gpu"], run_seconds=rec["seconds"],
            final_loss=float(np.mean([x["loss"] for x in rec["records"][-8:]])),
            strength_range=[min(x["strength_min"] for x in rec["records"]), max(x["strength_max"] for x in rec["records"])],
            val_fixed_AP=rec["val_fixed"]["macro"]["AP"])
    out["training"] = tr
    out["training_by_condition"] = {c: {v: dict(final_loss_mean=float(np.mean([tr[tag(v, c, k)]["final_loss"] for k in SEEDS])),
                                                init_all_bitwise_uniform=all(tr[tag(v, c, k)]["init_bitwise_equal_uniform"] for k in SEEDS),
                                                init_max_abs=max(tr[tag(v, c, k)]["init_max_abs"] for k in SEEDS),
                                                val_fixed_AP_mean=float(np.mean([tr[tag(v, c, k)]["val_fixed_AP"] for k in SEEDS])))
                                    for v in VARS} for c in COND}
    n3fit = OUT / "n3/pilot/step2/loss_fit.json"
    out["n3_inputs"] = dict(fit=run.READ(n3fit) if n3fit.exists() else None,
                            train_inputs_summary=run.READ(OUT / "n3/train_inputs_summary.json"))
    pre = [x for s in C.TESTSEQ + C.VAL for t in [tag(v, "n3", k) for v in VARS for k in SEEDS]
           for x in run.READ(led("test" if s in C.TESTSEQ else "val", s.replace("/", "_"), t).with_suffix(".json"))["prefix"]]
    out["n3_prefix_check_q15"] = dict(n=len(pre), all_winner_equal=all(x["winner_equal"] for x in pre),
                                      all_plan_equal=all(x["plan_equal"] for x in pre),
                                      max_score_error=max([x["score_error"] for x in pre], default=None),
                                      max_output_error=max([x["output_max_error"] for x in pre], default=None),
                                      max_plan_frame_minus_query=max([x["max_plan_frame_minus_query"] for x in pre], default=None))
    out["conventions"] = dict(
        val9="S.macro('val') 파이프라인 집계, 시드 평균, SD ddof=1",
        test46="KITTI 07(양성 없음) 제외 45시퀀스 -> 6도시 평균 -> 시드 평균(compare 스크립트 metric), SD ddof=1",
        intervals="a-b: 시퀀스별 (a 시드 평균 - b 시드 평균), 도시 안 시퀀스 재표집 10,000회, 도시 평균, 2.5/97.5% 분위, 비교마다 난수 20261104",
        d2="D2(오프셋 ±2..±10)는 최종 판정 지연이 25 키프레임으로 원고 구성(15)과 달라 원고 구성의 대체가 아니라 진단이다. "
           "학습8 그래프·손실 보정·xstats·선택 결과는 기존 results/pdv2_dilated_20261117/d2를 읽기만 했고, 균등 감산(uniform_d2)과 "
           "기존 NSD-D2(oldD2, 서버 학습 시드 4~6) 원장도 그 폴더의 것을 읽었다. 접두 검사는 이번에 하지 않았다(기존 D2 실험에서 지연 25 검사).",
        n3="n3(오프셋 ±1,±2,±3)는 D 스크립트의 함수를 모듈 전역만 바꿔 썼다. 정제 출력 의존 i+6, 지연 2*3+5 = 11. 시간 증거 plan(±5)은 그대로라 "
           "판정 지연은 11이며 15 이하다. n3 원장은 C.pipe의 q+15 접두 검사(--prefix 1, 시퀀스·모델마다 한 점)를 함께 돌렸다. "
           "손실 보정은 n3 균등 감산 초기 출력에 D.calibrate 절차로 새로 적합(sweep4/n3/pilot/step2/loss_fit.json).",
        init="kv_init_x0.3/x3: build 직후 두 층 W_k·W_v 가중치에 배수(편향 없음). q_init: W_q를 같은 시드 factory의 0으로 지우기 전 값으로 "
             "(torch 난수 상태 보존). q_init은 초기 출력이 균등 감산과 달라 init 검사를 강제하지 않았고(기록만), 손실 보정은 기본 조건의 기존 값"
             "(results/pdv2_scratch_pilot_20261116/step2/loss_fit.json)을 그대로 썼다. 초기화 조건은 이웃 ±5·기본 그래프·기본 보정이다.",
        base="base(기본 조건)와 uniform_base는 각각 sweep2 base 원장과 기존 탐색(S.LEDGERS) uniform 원장을 읽기만 했다(재학습·재평가 없음).",
        prefix="d2·초기화 조건 원장은 접두 검사 생략(--prefix 0)", units="0~1 비율(차이·구간은 %p)")
    P.write(OUT / "summary.json", out)
    for c in list(COND) + ["base", "ref"]:
        print("==", c)
        for v in arms[c]:
            print(f"  {v:15s} val AP {100 * val[c][v]['AP']['mean']:.3f}", "test", {k: round(100 * test[c][v][k]["mean"], 3) for k in keys})
        for name, d in diffs.get(c, {}).items():
            e = d["test46"]
            print(f"    {name:28s}", {k: (round(e["point_pp"][k], 3), [round(x, 3) for x in e["conditional_95_pp"][k]]) for k in keys},
                  "val9 AP", round(d["val9_mean_diff_pp"]["AP"], 3))
    print("GE_ABSKV", out["diff_family_ge_abskv"])
    print("ABSKV>BASE", out["abskv_better_than_base_abskv"])
    print("D2CHECK", {k: out["d2_level_check"][k]["test46"]["point_pp"]["AP"] for k in ("sym_d2_s123_minus_oldD2_s456", "sym_d2_s4_minus_oldD2_s4")},
          out["d2_level_check"]["sym_d2_s4_checkpoint_vs_existing"])
    print("N3PREFIX", out["n3_prefix_check_q15"])
    print("CHECKS", max(out["metric_checks"].values()), out["base_matches_sweep2_summary"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--cond", choices=list(COND))
    p.add_argument("--conds")
    p.add_argument("--variant", choices=list(VARS))
    p.add_argument("--seed", type=int)
    p.add_argument("--mode", choices=["base", "d2", "n3"])
    p.add_argument("--out")
    p.add_argument("--role", default="val")
    p.add_argument("--sequence")
    p.add_argument("--models")
    p.add_argument("--prefix", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    P.e.configure()
    torch.set_num_threads(4)
    print("NB", NB, "offsets", D.OFFSETS.tolist() if D is not None else "±5", "lag", D.LAG if D is not None else 15, flush=True)
    if a.action == "n3_prepare":
        n3_prepare()
    elif a.action == "train":
        train_run(a.cond, a.variant, a.seed)
    elif a.action == "train_campaign":
        train_campaign(a.conds.split(","))
    elif a.action == "models":
        write_models(a.mode, a.out)
    elif a.action == "pipe":
        C.pipe(a.role, a.sequence, Path(a.models).read_text().split(), a.prefix)
    elif a.action == "pipe_campaign":
        pipe_campaign(a.role, a.models, a.workers, a.prefix)
    elif a.action == "summary":
        summary()
    else:
        raise SystemExit(a.action)
