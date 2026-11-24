#!/usr/bin/env python3
"""학습 센서 계열을 하나로 줄였을 때의 K/V 입력 변형 비교(진단, 시드 1~3).

변형(1차 스크립트 scripts/_kvdiff_variants_20261118.py = V의 구현 그대로): sym, abskv, diffk_absv.
학습 조건(모두 로컬 GPU, 512갱신, lr 1e-3, 게이트 고정, 기본 CFG = 보고 모델 설정):
  kitti_only   학습 시퀀스 kitti/01, 02, 05, 06
  helipr_only  학습 시퀀스 helipr/Bridge01~04
바꾸는 것은 학습 시퀀스 목록뿐이다. 표본(P.NEWPOOLS), 손실 보정(학습8 초기 출력으로 적합한 기존 값), 입력 통계(S.xstats, 학습8),
융합 가중·lambda, 검증9·시험46 분할은 그대로다.
학습(train_run)은 C.train(scripts/_report_model_controls_20261116.py)과 같은 계산이고 다음만 다르다.
  - 시퀀스 루프가 조건의 4시퀀스(`for ordinal, s in enumerate(TRAIN_COND)`, ordinal 0~3).
  - epoch 수 = 512 // (8 * 4) = 16 (시퀀스당 8갱신, 총 512갱신 유지). 난수는 C와 같은 (seed, epoch, ordinal).
  - sym 재현 검사(기존 학습8 체크포인트와 비교)는 해당 없음.
비교 기준 전체 학습8(base)은 sweep2 원장(`{variant}@base@s{k}`)을 읽기만 한다. 균등 감산은 S.LEDGERS의 uniform 원장.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_kvdiff_sweep3_20261118.py <action>
  train --cond C --variant V --seed K | train_campaign | models --out FILE
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
OUT = ROOT / "results/kv_diff_improve_20261118/sweep3"
RUNS, LEDGERS = OUT / "runs", OUT / "ledgers"
SWEEP2 = ROOT / "results/kv_diff_improve_20261118/sweep2"
VARS = ("sym", "abskv", "diffk_absv")
SEEDS = (1, 2, 3)
CFG = V.CFG
TOTAL = 512
assert CFG == "x=log__n=ln__g=frozen__lr=1e-3__U=1536" and C.U == TOTAL and S.parse(CFG)["sch"] == "const"
TRAIN8 = list(C.TRAIN)
CONDS = {"kitti_only": ["kitti/01", "kitti/02", "kitti/05", "kitti/06"],
         "helipr_only": ["helipr/Bridge01", "helipr/Bridge02", "helipr/Bridge03", "helipr/Bridge04"]}
for _seqs in CONDS.values():
    assert set(_seqs) <= set(TRAIN8) and TOTAL % (8 * len(_seqs)) == 0

C.OUT, C.LEDGERS = OUT, LEDGERS


def tag(variant: str, cond: str, seed: int) -> str:
    return f"{variant}@{cond}@s{seed}"


def ckpt(cond: str, variant: str, seed: int) -> Path:
    return RUNS / cond / variant / f"s{seed}" / f"u{TOTAL}.pth"


# ----------------------------------------------------------------- 학습
def train_run(cond: str, variant: str, seed: int) -> None:
    """C.train과 같은 계산, 학습 시퀀스만 CONDS[cond]."""
    seqs = CONDS[cond]
    cfg = S.parse(CFG)
    folder = RUNS / cond / variant / f"s{seed}"
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
    graphs = {s: C.train_graph(variant, s) for s in seqs}
    with torch.no_grad():
        init = {s: bool(torch.equal(net(g), prior.filtered(g, -0.5))) for s, g in graphs.items()}
        init_err = {s: float((net(g) - prior.filtered(g, -0.5)).abs().max()) for s, g in graphs.items()}
    assert init[seqs[0]], init_err
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    pls = {s: run.ARR(C.pool_dir(variant) / "train" / s.replace("/", "_") / "pool.npz") for s in seqs}
    plans = {s: {k: torch.from_numpy(d["plan_" + k]).cuda() for k in ("query", "candidate", "weight")} for s, d in pls.items()}
    batch = int(cfg["B"])
    epochs = TOTAL // (8 * len(seqs))
    records, updates, tick = [], 0, time.time()
    for epoch in range(1, epochs + 1):
        for ordinal, s in enumerate(seqs):
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
    path = folder / f"u{TOTAL}.pth"
    torch.save(dict(model_state_dict=net.state_dict(), updates=updates, cfg=cfg, variant=variant, seed=seed, train=seqs), path)
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
        status="passed", cond=cond, train_sequences=seqs, epochs=epochs, variant=variant, kind=C.VARIANTS[variant]["kind"],
        cfg=cfg, name=CFG, seed=seed, updates=updates, checkpoint=path.name, sha256=run.HASH(path), records=records,
        changed=changed, init_bitwise_equal_uniform=init, init_max_abs_vs_uniform=init_err,
        trainable_parameters=sum(p.numel() for p in params), total_parameters=sum(p.numel() for p in net.parameters()),
        fit=fitrec, fit_path=str(C.fit_path(variant).relative_to(ROOT)), pools=str(C.pool_dir(variant).relative_to(ROOT)),
        val_fixed=dict(macro=run.base.macro(per)["macro"], sequences=per),
        seconds=seconds, torch=torch.__version__, gpu=torch.cuda.get_device_name()))
    print("TRAIN", cond, variant, seed, round(seconds, 1), "trainable", sum(p.numel() for p in params),
          "valfixedAP", round(100 * run.base.macro(per)["macro"]["AP"], 3), flush=True)


def train_campaign() -> None:
    logs = OUT / "logs" / "train"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    for c in CONDS:
        for v in VARS:
            for k in SEEDS:
                if (RUNS / c / v / f"s{k}" / "training_complete.json").exists():
                    continue
                with (logs / f"{c}_{v}_s{k}.log").open("a") as f:
                    subprocess.run([sys.executable, __file__, "train", "--cond", c, "--variant", v, "--seed", str(k)],
                                   stdout=f, stderr=subprocess.STDOUT, check=True)
                print("TRAINED", c, v, k, round(time.time() - tick, 1), flush=True)
    P.write(OUT / "logs" / "train_campaign.done.json", dict(seconds=time.time() - tick))


# ----------------------------------------------------------------- 실제 검색 (C.pipe 경로, 모델 해석만 바꿈)
def model_spec(m: str):
    variant, cond, seed = m.split("@")
    assert variant in VARS and cond in CONDS
    net = C.build(variant, int(seed[1:]))
    net.load_state_dict(torch.load(ckpt(cond, variant, int(seed[1:])), map_location="cuda", weights_only=False)["model_state_dict"],
                        strict=True)
    return variant, net, None


C.model_spec = model_spec


def write_models(path: str) -> None:
    Path(path).write_text("\n".join(tag(v, c, k) for c in CONDS for v in VARS for k in SEEDS) + "\n")


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


# ----------------------------------------------------------------- 집계 (sweep2 summary와 같은 규약 + 센서 묶음)
def summary() -> None:
    import _pdv2_scratch_search_compare_20261116 as K

    keys, metrics = K.KEYS, C.METRICS
    allc = list(CONDS) + ["base"]
    tags_ = {c: {v: [tag(v, c, k) for k in SEEDS] for v in VARS} for c in allc}
    tags_["uniform"] = {"uniform": ["uniform"]}

    def led(role, name, t):
        if t == "uniform":
            return S.LEDGERS / role / name / "uniform.npz"
        root = SWEEP2 / "ledgers" if "@base@" in t else LEDGERS
        return root / role / name / (t + ".npz")

    # 센서 묶음: scripts/_kv_subset_analysis_20261117.py H1 주 정의의 시퀀스(학습 계열 = kitti·kitti360·helipr,
    # 미학습 = mulran·nclt·mcd)에서 학습 계열을 KITTI·KITTI-360과 HeLiPR로 나눔. 도시 구조는 restrict()와 같다.
    def restrict(groups, seqs):
        return {c: [s for s in v if s in seqs] for c, v in groups.items() if any(s in seqs for s in v)}
    allseq = [s for v in K.CITY.values() for s in v]
    GROUPS = {"all": K.CITY,
              "kitti_kitti360": restrict(K.CITY, [s for s in allseq if s.split("/")[0] in ("kitti", "kitti360")]),
              "helipr": restrict(K.CITY, [s for s in allseq if s.split("/")[0] == "helipr"]),
              "unseen_sensors": restrict(K.CITY, [s for s in allseq if s.split("/")[0] in ("mulran", "nclt", "mcd")])}

    out = dict(conditions={**{c: dict(train=CONDS[c], updates=TOTAL, cfg=CFG, ledgers=str(LEDGERS.relative_to(ROOT))) for c in CONDS},
                           "base": dict(train=TRAIN8, updates=TOTAL, cfg=CFG, ledgers=str((SWEEP2 / "ledgers").relative_to(ROOT)),
                                        note="sweep2 base(로컬 5080) 원장 재사용, 재학습 없음"),
                           "uniform": dict(ledgers=str(S.LEDGERS.relative_to(ROOT)), note="학습 없음, 기존 원장")},
               variants=list(VARS), seeds=list(SEEDS), fixed=dict(weights=P.selected()[0].tolist(), lam=P.selected()[1]),
               groups={g: {c: list(v) for c, v in d.items()} for g, d in GROUPS.items()})
    val, rows = {}, {}
    for c, arms in tags_.items():
        val[c] = {}
        for v, ts in arms.items():
            per = {t: S.macro("val", {s: run.READ(led("val", s.replace("/", "_"), t).with_suffix(".json")) for s in C.VAL}) for t in ts}
            val[c][v] = {k: dict(mean=float(np.mean([per[t][k] for t in ts])),
                                 sd=float(np.std([per[t][k] for t in ts], ddof=1)) if len(ts) > 1 else None,
                                 values=[per[t][k] for t in ts]) for k in metrics}
            for t in ts:
                for s in allseq:
                    name = s.replace("/", "_")
                    d = dict(np.load(K.DIRS / (name + ".npz")))
                    with np.load(led("test", name, t)) as z:
                        np.testing.assert_array_equal(z["query_idx"], d["query_idx"])
                        rows.setdefault(t, {})[s] = K.metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"])

    def macro(t, groups):
        cities = []
        for seqs in groups.values():
            v = [rows[t][s] for s in seqs if rows[t][s] is not None]
            if v:
                cities.append({k: np.mean([x[k] for x in v]) for k in keys})
        return {k: float(np.mean([c[k] for c in cities])) for k in keys}

    per_model = {g: {t: macro(t, groups) for arms in tags_.values() for ts in arms.values() for t in ts} for g, groups in GROUPS.items()}
    test = {g: {c: {v: {k: dict(mean=float(np.mean([per_model[g][t][k] for t in ts])),
                                sd=float(np.std([per_model[g][t][k] for t in ts], ddof=1)) if len(ts) > 1 else None,
                                values=[per_model[g][t][k] for t in ts]) for k in keys}
                    for v, ts in arms.items()} for c, arms in tags_.items()} for g in GROUPS}

    def seqmat(A, B, avail):
        return np.array([[100 * (np.mean([rows[t][s][k] for t in A]) - np.mean([rows[t][s][k] for t in B])) for k in keys] for s in avail])

    def boot(A, B, groups, A2=None, B2=None):
        """도시 안 시퀀스 재표집 10,000회, 도시 평균, 난수 20261104(비교마다 새로). A2/B2가 있으면 같은 색인으로 (A-B)-(A2-B2)."""
        rng = np.random.default_rng(20261104)
        draws, city = [], {}
        for cname, seqs in groups.items():
            avail = [s for s in seqs if rows[A[0]][s] is not None]
            if not avail:
                continue
            mat = seqmat(A, B, avail)
            if A2 is not None:
                mat = mat - seqmat(A2, B2, avail)
            city[cname] = dict(zip(keys, mat.mean(0).tolist()))
            idx = rng.integers(0, len(avail), size=(10000, len(avail)))
            draws.append(mat[idx].mean(1))
        b = np.stack(draws).mean(0)
        point = {k: float(np.mean([city[c][k] for c in city])) for k in keys}
        return dict(point_pp=point, conditional_95_pp={k: np.quantile(b[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(keys)},
                    city_delta_pp=city)

    pairs = [("sym", "abskv"), ("diffk_absv", "abskv"), ("sym", "uniform"), ("abskv", "uniform"), ("diffk_absv", "uniform")]
    diffs, cross = {}, {}
    for c in list(CONDS) + ["base"]:
        diffs[c] = {}
        for a, b in pairs:
            A = tags_[c][a]
            B = tags_["uniform"]["uniform"] if b == "uniform" else tags_[c][b]
            Bv = val["uniform"]["uniform"] if b == "uniform" else val[c][b]
            diffs[c][f"{a}-{b}"] = dict(
                test46={g: boot(A, B, groups) for g, groups in GROUPS.items()},
                val9_mean_diff_pp={k: 100 * (val[c][a][k]["mean"] - Bv[k]["mean"]) for k in metrics})
            # 점추정 검산: 도시 평균의 차 = 시드 평균 매크로의 차
            for g in GROUPS:
                bm = test[g]["uniform"]["uniform"] if b == "uniform" else test[g][c][b]
                for k in keys:
                    assert abs(diffs[c][f"{a}-{b}"]["test46"][g]["point_pp"][k] - 100 * (test[g][c][a][k]["mean"] - bm[k]["mean"])) < 1e-9
    # 조건 간 차이의 차이: (a-b)@cond − (a-b)@base, 같은 시퀀스 색인으로 짝지은 재표집
    for c in CONDS:
        cross[c] = {}
        for a, b in pairs[:2]:
            cross[c][f"({a}-{b})@{c} - ({a}-{b})@base"] = dict(
                test46={g: boot(tags_[c][a], tags_[c][b], groups, tags_["base"][a], tags_["base"][b]) for g, groups in GROUPS.items()},
                val9_mean_diff_pp={k: diffs[c][f"{a}-{b}"]["val9_mean_diff_pp"][k] - diffs["base"][f"{a}-{b}"]["val9_mean_diff_pp"][k]
                                   for k in metrics})
    out["val"], out["test"] = val, dict(groups=test, per_model=per_model)
    out["differences"], out["difference_vs_base"] = diffs, cross
    out["metric_checks"] = {t: max(abs(S.macro("test", {s: run.READ(led("test", s.replace("/", "_"), t).with_suffix(".json"))
                                                        for s in C.TESTSEQ})[k] - per_model["all"][t][k]) for k in keys)
                            for t in per_model["all"]}
    # sweep2 base 값과 일치하는지(같은 원장을 읽었는지)
    s2 = run.READ(SWEEP2 / "summary.json")
    out["base_matches_sweep2_summary"] = {v: dict(
        val9=max(abs(val["base"][v][k]["mean"] - s2["val"]["base"][v][k]["mean"]) for k in metrics),
        test46=max(abs(test["all"]["base"][v][k]["mean"] - s2["test"]["arms"]["base"][v][k]["mean"]) for k in keys)) for v in VARS}
    tr = {}
    for c in CONDS:
        for v in VARS:
            for k in SEEDS:
                rec = run.READ(RUNS / c / v / f"s{k}" / "training_complete.json")
                assert rec["sha256"] == run.HASH(ckpt(c, v, k)) and rec["train_sequences"] == CONDS[c]
                tr[tag(v, c, k)] = dict(
                    checkpoint=str(ckpt(c, v, k).relative_to(ROOT)), sha256=rec["sha256"], epochs=rec["epochs"],
                    trainable_parameters=rec["trainable_parameters"], init_bitwise_equal_uniform=all(rec["init_bitwise_equal_uniform"].values()),
                    init_max_abs=max(rec["init_max_abs_vs_uniform"].values()), gpu=rec["gpu"], run_seconds=rec["seconds"],
                    final_loss=float(np.mean([x["loss"] for x in rec["records"][-8:]])),
                    strength_range=[min(x["strength_min"] for x in rec["records"]), max(x["strength_max"] for x in rec["records"])],
                    val_fixed_AP=rec["val_fixed"]["macro"]["AP"])
    out["training"] = tr
    out["training_by_condition"] = {c: {v: dict(final_loss_mean=float(np.mean([tr[tag(v, c, k)]["final_loss"] for k in SEEDS])),
                                                strength_range=[min(tr[tag(v, c, k)]["strength_range"][0] for k in SEEDS),
                                                                max(tr[tag(v, c, k)]["strength_range"][1] for k in SEEDS)])
                                        for v in VARS} for c in CONDS}
    out["conventions"] = dict(
        val9="S.macro('val') 파이프라인 집계, 시드 평균, SD ddof=1",
        test46="KITTI 07(양성 없음) 제외 45시퀀스 -> 6도시 평균 -> 시드 평균(compare 스크립트 metric), SD ddof=1",
        groups="센서 묶음은 _kv_subset_analysis_20261117.py H1 주 정의의 시퀀스 묶음(학습 계열을 KITTI·KITTI-360 / HeLiPR로 나눔). "
               "묶음 안 집계는 해당 시퀀스가 있는 도시만 도시 평균(daejeon은 묶음에 따라 helipr 9 또는 mulran 9만)",
        intervals="같은 조건 안 a-b: 시퀀스별 (a 시드 평균 - b 시드 평균), 도시 안 시퀀스 재표집 10,000회, 도시 평균, 2.5/97.5% 분위, "
                  "비교·묶음마다 난수 20261104. difference_vs_base는 같은 시퀀스 색인으로 두 조건의 차를 함께 재표집(짝지음)",
        rng="학습 난수 (seed, epoch, ordinal), ordinal은 조건의 4시퀀스 목록 안 순번 0~3, epoch 1~16",
        prefix="원장 접두 검사 생략(--prefix 0)", units="0~1 비율(차이·구간은 %p)")
    P.write(OUT / "summary.json", out)
    for c in list(CONDS) + ["base", "uniform"]:
        print("==", c)
        for v in tags_[c]:
            print(f"  {v:11s} val AP {100 * val[c][v]['AP']['mean']:.3f}", "test", {k: round(100 * test["all"][c][v][k]["mean"], 3) for k in keys})
        if c in diffs:
            for name, d in diffs[c].items():
                for g in GROUPS:
                    e = d["test46"][g]
                    print(f"    {name:18s} {g:15s}", {k: (round(e["point_pp"][k], 3), [round(x, 3) for x in e["conditional_95_pp"][k]]) for k in keys})
                print(f"    {name:18s} val9 AP {d['val9_mean_diff_pp']['AP']:.3f}")
    for c, dd in cross.items():
        for name, d in dd.items():
            for g in GROUPS:
                e = d["test46"][g]
                print(f"  {name} {g:15s} AP", round(e["point_pp"]["AP"], 3), [round(x, 3) for x in e["conditional_95_pp"]["AP"]])
    print("CHECKS", max(out["metric_checks"].values()), out["base_matches_sweep2_summary"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--cond", choices=list(CONDS))
    p.add_argument("--variant", choices=list(VARS))
    p.add_argument("--seed", type=int)
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
        train_run(a.cond, a.variant, a.seed)
    elif a.action == "train_campaign":
        train_campaign()
    elif a.action == "models":
        write_models(a.out)
    elif a.action == "pipe":
        C.pipe(a.role, a.sequence, Path(a.models).read_text().split(), a.prefix)
    elif a.action == "pipe_campaign":
        pipe_campaign(a.role, a.models, a.workers, a.prefix)
    elif a.action == "summary":
        summary()
    else:
        raise SystemExit(a.action)
