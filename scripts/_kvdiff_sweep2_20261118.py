#!/usr/bin/env python3
"""학습 조건별 K/V 입력 변형 비교(시드 1~3).

변형(1차 스크립트 scripts/_kvdiff_variants_20261118.py = V의 구현 그대로): sym, abskv, diffk_absv, sum_kv.
학습 조건(조건마다 네 변형 모두, 모두 로컬 GPU에서 다시 학습):
  base          512갱신, lr 1e-3, 게이트 고정(1차·보고 모델 설정)
  u1024 / u1536 갱신 수만 1024 / 1536
  lr3e-3        lr 3e-3, 512갱신
  lr3e-4_u1536  lr 3e-4, 1536갱신
  gfree         게이트 학습(g=free), 512갱신
  gfree_u1536   게이트 학습, 1536갱신
데이터 분할·표본·손실 보정·융합 가중·lambda·난수 규약은 C(scripts/_report_model_controls_20261116.py)와 같다.

학습 실행(run)과 조건의 대응:
  run base / gfree : C.train 그대로(C의 전역 CFG·U·RUNS만 바꿔 호출).
  run long         : lr 1e-3 고정 게이트 1536갱신 한 번에서 512·1024·1536 저장 -> u1024, u1536.
  run gfree_long   : g=free 1536갱신에서 512·1536 저장 -> gfree_u1536.
  run lr3e-3, lr3e-4_u1536 : 아래 train_run.
  train_run은 C.train과 같은 순서·난수·계산이고 (1) 지정한 갱신 시점마다 체크포인트를 저장하고
  (2) sym 재현 대상(S.RUNS)이 없으면 재현 기록을 건너뛰는 것만 다르다. 학습률 일정이 const라 CFG의 U 항목은
  학습에 영향이 없다(S.lr_at은 sch=cos일 때만 U를 쓴다). long·gfree_long의 512 시점이 base·gfree(C.train)와
  비트 단위로 같은지 summary가 검사한다.
게이트: S.build는 g=free/frozen에서 모델을 똑같이 만들고(g=weak만 클래스 교체), 차이는 S.trainable의
gate.* 학습 여부뿐이다. 게이트 마지막 Linear는 0 초기화(LearnedSpectralDifference)라 초기 세기는 -0.5,
초기 출력은 균등 감산과 같다(C.train·train_run이 검사).

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_kvdiff_sweep2_20261118.py <action>
  train --run R --variant V --seed K | train_campaign --runs R1,R2 | models --conds A,B --out FILE
  | pipe --role R --sequence S --models FILE | pipe_campaign --role R --models FILE | summary
"""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
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
OUT = ROOT / "results/kv_diff_improve_20261118/sweep2"
RUNS, LEDGERS = OUT / "runs", OUT / "ledgers"
VARS = ("sym", "abskv", "diffk_absv", "sum_kv")
DIFF = ("sym", "diffk_absv", "sum_kv")
SEEDS = (1, 2, 3)
BASECFG = V.CFG
assert BASECFG == "x=log__n=ln__g=frozen__lr=1e-3__U=1536" and C.U == 512
LR3 = "x=log__n=ln__g=frozen__lr=3e-3__U=1536"
LR34 = "x=log__n=ln__g=frozen__lr=3e-4__U=1536"
FREE = "x=log__n=ln__g=free__lr=1e-3__U=1536"
# run -> (CFG, 총 갱신, 저장 시점, C.train 사용 여부)
RUNSPEC = {"base": (BASECFG, 512, (512,), True), "gfree": (FREE, 512, (512,), True),
           "long": (BASECFG, 1536, (512, 1024, 1536), False), "gfree_long": (FREE, 1536, (512, 1536), False),
           "lr3e-3": (LR3, 512, (512,), False), "lr3e-4_u1536": (LR34, 1536, (1536,), False)}
# 조건 -> (run, 체크포인트 갱신 수)
COND = {"base": ("base", 512), "u1024": ("long", 1024), "u1536": ("long", 1536), "lr3e-3": ("lr3e-3", 512),
        "lr3e-4_u1536": ("lr3e-4_u1536", 1536), "gfree": ("gfree", 512), "gfree_u1536": ("gfree_long", 1536)}
for _cfg, _total, _saves, _ in RUNSPEC.values():
    assert S.parse(_cfg)["sch"] == "const" and _total % 64 == 0 and max(_saves) == _total

C.OUT, C.LEDGERS = OUT, LEDGERS


def tag(variant: str, cond: str, seed: int) -> str:
    return f"{variant}@{cond}@s{seed}"


def ckpt(cond: str, variant: str, seed: int) -> Path:
    r, u = COND[cond]
    return RUNS / r / variant / f"s{seed}" / f"u{u}.pth"


# ----------------------------------------------------------------- 학습
def train_run(rname: str, variant: str, seed: int) -> None:
    """C.train과 같은 계산. 저장 시점을 여러 개 둘 수 있고 sym 재현 대상이 없으면 건너뛴다."""
    cfgname, total, saves, _ = RUNSPEC[rname]
    cfg = S.parse(cfgname)
    folder = RUNS / rname / variant / f"s{seed}"
    if (folder / "training_complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    _, lam = P.selected()
    fitrec = run.READ(C.fit_path(variant))["fit"]
    net = C.build(variant, seed, S.xstats())
    if variant == "sym":  # S.build와 같은 초기 모델인지
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
    assert init[C.TRAIN[0]], init_err
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    pls = {s: run.ARR(C.pool_dir(variant) / "train" / s.replace("/", "_") / "pool.npz") for s in C.TRAIN}
    plans = {s: {k: torch.from_numpy(d["plan_" + k]).cuda() for k in ("query", "candidate", "weight")} for s, d in pls.items()}
    batch = int(cfg["B"])
    records, hashes, repro, updates, tick = [], {}, {}, 0, time.time()
    for epoch in range(1, total // 64 + 1):
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
                if updates in saves:
                    path = folder / f"u{updates}.pth"
                    torch.save(dict(model_state_dict=net.state_dict(), updates=updates, cfg=cfg, variant=variant, seed=seed), path)
                    hashes[path.name] = run.HASH(path)
                    old = S.RUNS / S.canon(cfgname) / f"s{seed}" / f"u{updates}.pth"
                    if variant == "sym" and old.exists():
                        o = torch.load(old, map_location="cuda", weights_only=False)["model_state_dict"]
                        repro[path.name] = dict(state_bitwise_equal_existing=all(torch.equal(v, o[k]) for k, v in net.state_dict().items()),
                                                max_abs=max(float((v.float() - o[k].float()).abs().max()) for k, v in net.state_dict().items()))
            st = net.last_strengths.detach()
            records.append(dict(epoch=epoch, sequence=s, loss=float(np.mean(losses)), grad=float(np.mean(grads)),
                                strength_mean=float(st.mean()), strength_min=float(st.min()), strength_max=float(st.max())))
    assert updates == total and set(hashes) == {f"u{u}.pth" for u in saves}
    seconds = time.time() - tick
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, initial[k])]
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
        status="passed", run=rname, variant=variant, kind=C.VARIANTS[variant]["kind"], cfg=cfg, name=cfgname, seed=seed,
        updates=updates, saves=list(saves), checkpoints=hashes, records=records, changed=changed,
        init_bitwise_equal_uniform=init, init_max_abs_vs_uniform=init_err,
        trainable_parameters=sum(p.numel() for p in params), total_parameters=sum(p.numel() for p in net.parameters()),
        fit=fitrec, reproduction_vs_scratch_search=repro or None,
        val_fixed_final=dict(macro=run.base.macro(per)["macro"], sequences=per),
        seconds=seconds, torch=torch.__version__, gpu=torch.cuda.get_device_name()))
    print("TRAIN", rname, variant, seed, round(seconds, 1), "trainable", sum(p.numel() for p in params), "repro", repro,
          "valfixedAP", round(100 * run.base.macro(per)["macro"]["AP"], 3), flush=True)


def train(rname: str, variant: str, seed: int) -> None:
    cfgname, total, saves, use_c = RUNSPEC[rname]
    if use_c:
        C.CFG, C.U, C.RUNS = cfgname, total, RUNS / rname
        C.train(variant, seed)
    else:
        train_run(rname, variant, seed)


def train_campaign(runs: list) -> None:
    logs = OUT / "logs" / "train"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    for r in runs:
        for v in VARS:
            for k in SEEDS:
                if (RUNS / r / v / f"s{k}" / "training_complete.json").exists():
                    continue
                with (logs / f"{r}_{v}_s{k}.log").open("a") as f:
                    subprocess.run([sys.executable, __file__, "train", "--run", r, "--variant", v, "--seed", str(k)],
                                   stdout=f, stderr=subprocess.STDOUT, check=True)
                print("TRAINED", r, v, k, round(time.time() - tick, 1), flush=True)
    P.write(OUT / "logs" / f"train_campaign_{'_'.join(runs)}.done.json", dict(seconds=time.time() - tick))


# ----------------------------------------------------------------- 실제 검색 (C.pipe 경로, 모델 해석만 바꿈)
def load(cond: str, variant: str, seed: int) -> nn.Module:
    net = C.build(variant, seed)
    net.load_state_dict(torch.load(ckpt(cond, variant, seed), map_location="cuda", weights_only=False)["model_state_dict"], strict=True)
    return net


def model_spec(m: str):
    variant, cond, seed = m.split("@")
    assert variant in VARS and cond in COND
    return variant, load(cond, variant, int(seed[1:])), None


C.model_spec = model_spec


def write_models(conds: list, path: str) -> None:
    Path(path).write_text("\n".join(tag(v, c, k) for c in conds for v in VARS for k in SEEDS) + "\n")


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
    P.write(logs / f"pipe_campaign_{role}_{Path(models_file).stem}.done.json", dict(seconds=time.time() - tick, workers=workers))


# ----------------------------------------------------------------- 집계 (V.summary와 같은 규약, 조건별)
def summary() -> None:
    import _pdv2_scratch_search_compare_20261116 as K

    keys, metrics = K.KEYS, C.METRICS
    tags_ = {c: {v: [tag(v, c, k) for k in SEEDS] for v in VARS} for c in COND}
    led = lambda role, name, t: LEDGERS / role / name / (t + ".npz")  # noqa: E731
    out = dict(conditions={c: dict(run=r, updates=u, cfg=RUNSPEC[r][0]) for c, (r, u) in COND.items()},
               variants=list(VARS), seeds=list(SEEDS), fixed=dict(weights=P.selected()[0].tolist(), lam=P.selected()[1]))
    val, test, rows, per_model = {}, {}, {}, {}
    for c in COND:
        val[c] = {}
        for v, ts in tags_[c].items():
            per = {t: S.macro("val", {s: run.READ(led("val", s.replace("/", "_"), t).with_suffix(".json")) for s in C.VAL}) for t in ts}
            val[c][v] = {k: dict(mean=float(np.mean([per[t][k] for t in ts])), sd=float(np.std([per[t][k] for t in ts], ddof=1)),
                                 values=[per[t][k] for t in ts]) for k in metrics}
            for t in ts:
                for seqs in K.CITY.values():
                    for s in seqs:
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

    diffs = {}
    for c in COND:
        for ts in tags_[c].values():
            for t in ts:
                per_model[t] = macro(t)
        test[c] = {v: {k: dict(mean=float(np.mean([per_model[t][k] for t in ts])), sd=float(np.std([per_model[t][k] for t in ts], ddof=1)),
                           values=[per_model[t][k] for t in ts]) for k in keys} for v, ts in tags_[c].items()}
        diffs[c] = {}
        for a in VARS:
            for b in ("abskv", "sym"):
                if a == b:
                    continue
                A, B = tags_[c][a], tags_[c][b]
                rng = np.random.default_rng(20261104)
                draws, city = [], {}
                for cname, seqs in K.CITY.items():
                    avail = [s for s in seqs if rows[A[0]][s] is not None]
                    if not avail:
                        continue
                    mat = np.array([[100 * (np.mean([rows[t][s][k] for t in A]) - np.mean([rows[t][s][k] for t in B])) for k in keys]
                                    for s in avail])
                    city[cname] = dict(zip(keys, mat.mean(0).tolist()))
                    idx = rng.integers(0, len(avail), size=(10000, len(avail)))
                    draws.append(mat[idx].mean(1))
                boot = np.stack(draws).mean(0)
                diffs[c][f"{a}-{b}"] = dict(point_pp={k: 100 * (test[c][a][k]["mean"] - test[c][b][k]["mean"]) for k in keys},
                                            conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(keys)},
                                            val9_mean_diff_pp={k: 100 * (val[c][a][k]["mean"] - val[c][b][k]["mean"]) for k in metrics},
                                            city_delta_pp=city)
    out["val"], out["test"] = val, dict(arms=test, per_model=per_model)
    out["differences"] = diffs
    out["metric_checks"] = {t: max(abs(S.macro("test", {s: run.READ(led("test", s.replace("/", "_"), t).with_suffix(".json"))
                                                        for s in C.TESTSEQ})[k] - per_model[t][k]) for k in keys) for t in per_model}
    # 차이 계열이 같은 조건의 abskv 이상인지, 변형별 최고 조건
    out["diff_family_ge_abskv"] = {c: {a: dict(val9_AP=val[c][a]["AP"]["mean"] >= val[c]["abskv"]["AP"]["mean"],
                                               test46_AP=test[c][a]["AP"]["mean"] >= test[c]["abskv"]["AP"]["mean"]) for a in DIFF}
                                   for c in COND}
    out["best_condition"] = {v: dict(by_val9_AP=max(COND, key=lambda c: val[c][v]["AP"]["mean"]),
                                     by_test46_AP=max(COND, key=lambda c: test[c][v]["AP"]["mean"])) for v in VARS}
    # 학습 기록: 최종 손실(마지막 8기록 = 마지막 64갱신), 세기 범위(학습8 기록, 해당 갱신 수까지)
    tr = {}
    for c, (r, u) in COND.items():
        for v in VARS:
            for k in SEEDS:
                rec = run.READ(RUNS / r / v / f"s{k}" / "training_complete.json")
                recs = rec["records"][: u // 8]
                assert len(recs) == u // 8
                tr[tag(v, c, k)] = dict(
                    checkpoint=str(ckpt(c, v, k).relative_to(ROOT)), sha256=run.HASH(ckpt(c, v, k)),
                    trainable_parameters=rec["trainable_parameters"], init_bitwise_equal_uniform=all(rec["init_bitwise_equal_uniform"].values()),
                    init_max_abs=max(rec["init_max_abs_vs_uniform"].values()), gpu=rec["gpu"], run_seconds=rec["seconds"],
                    final_loss=float(np.mean([x["loss"] for x in recs[-8:]])),
                    strength_range=[min(x["strength_min"] for x in recs), max(x["strength_max"] for x in recs)],
                    reproduction_vs_scratch_search=rec["reproduction_vs_scratch_search"])
    out["training"] = tr
    out["training_by_condition"] = {c: {v: dict(final_loss_mean=float(np.mean([tr[tag(v, c, k)]["final_loss"] for k in SEEDS])),
                                                strength_range=[min(tr[tag(v, c, k)]["strength_range"][0] for k in SEEDS),
                                                                max(tr[tag(v, c, k)]["strength_range"][1] for k in SEEDS)],
                                                trainable_parameters=tr[tag(v, c, 1)]["trainable_parameters"]) for v in VARS} for c in COND}
    # 긴 학습의 512 시점 = C.train 512 학습(비트 단위)
    bit = {}
    for long_, short in (("long", "base"), ("gfree_long", "gfree")):
        for v in VARS:
            for k in SEEDS:
                a = torch.load(RUNS / long_ / v / f"s{k}" / "u512.pth", map_location="cpu", weights_only=False)["model_state_dict"]
                b = torch.load(RUNS / short / v / f"s{k}" / "u512.pth", map_location="cpu", weights_only=False)["model_state_dict"]
                bit[f"{long_}@u512=={short}:{v}@s{k}"] = list(a) == list(b) and all(torch.equal(t, b[n]) for n, t in a.items())
    out["long_run_u512_bitwise_equal_c_train"] = bit
    # 참고: 1차(다른 GPU, RTX PRO 6000)에서 학습한 같은 변형의 시드 1~3 평균
    s1 = run.READ(ROOT / "results/kv_diff_improve_20261118/sweep1/summary.json")
    out["reference_sweep1_other_gpu"] = {v: dict(val9_AP=s1["val"][v]["AP"]["mean"], test46={k: s1["test"]["arms"][v][k]["mean"] for k in keys})
                                         for v in VARS}
    out["conventions"] = dict(
        val9="S.macro('val') 파이프라인 집계, 시드 평균, SD ddof=1",
        test46="KITTI 07(양성 없음) 제외 45시퀀스 -> 6도시 평균 -> 시드 평균(compare 스크립트 metric), SD ddof=1",
        intervals="같은 조건 안 a-b: 시퀀스별 (a 시드 평균 - b 시드 평균), 도시 안 시퀀스 재표집 10,000회, 도시 평균, 2.5/97.5% 분위, 쌍마다 난수 20261104",
        training="final_loss는 해당 갱신 수까지의 마지막 8기록(64갱신) 평균, strength_range는 학습8 기록(시퀀스 블록마다 마지막 순전파)의 최소·최대",
        prefix="원장 접두 검사 생략(--prefix 0)", units="0~1 비율(차이·구간은 %p)")
    P.write(OUT / "summary.json", out)
    for c in COND:
        print("==", c)
        for v in VARS:
            print(f"  {v:11s} val AP {100 * val[c][v]['AP']['mean']:.3f}", "test", {k: round(100 * test[c][v][k]["mean"], 3) for k in keys},
                  "loss", round(out["training_by_condition"][c][v]["final_loss_mean"], 5),
                  "eta", [round(x, 3) for x in out["training_by_condition"][c][v]["strength_range"]])
        for name, d in diffs[c].items():
            print(f"    {name:18s}", {k: (round(d["point_pp"][k], 3), [round(x, 3) for x in d["conditional_95_pp"][k]]) for k in keys},
                  "val9 AP", round(d["val9_mean_diff_pp"]["AP"], 3))
    print("BEST", out["best_condition"])
    print("GE_ABSKV", out["diff_family_ge_abskv"])
    print("CHECKS", max(out["metric_checks"].values()), all(bit.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--run", choices=list(RUNSPEC))
    p.add_argument("--runs")
    p.add_argument("--variant", choices=list(VARS))
    p.add_argument("--seed", type=int)
    p.add_argument("--conds")
    p.add_argument("--out")
    p.add_argument("--role", default="val")
    p.add_argument("--sequence")
    p.add_argument("--models")
    p.add_argument("--prefix", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    P.e.configure()
    torch.set_num_threads(4)
    if a.action == "train":
        train(a.run, a.variant, a.seed)
    elif a.action == "train_campaign":
        train_campaign(a.runs.split(","))
    elif a.action == "models":
        write_models(a.conds.split(","), a.out)
    elif a.action == "pipe":
        C.pipe(a.role, a.sequence, Path(a.models).read_text().split(), a.prefix)
    elif a.action == "pipe_campaign":
        pipe_campaign(a.role, a.models, a.workers, a.prefix)
    elif a.action == "summary":
        summary()
    else:
        raise SystemExit(a.action)
