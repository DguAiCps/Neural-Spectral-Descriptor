#!/usr/bin/env python3
"""차이 K/V 최종 비교(시드 4,5,6, base 조건): sym, abskv, absk, diffk_absv, sum_kv.

base 조건 = 512갱신, lr 1e-3, 게이트 고정(CFG x=log__n=ln__g=frozen__lr=1e-3__U=1536 @512). 변형 구현은
scripts/_kvdiff_variants_20261118.py(V)의 build/KVConv 그대로, 학습·검색은 scripts/_report_model_controls_20261116.py(C)의
train / pipe 그대로(C 전역 OUT·RUNS·LEDGERS만 이 폴더로). 데이터 분할·표본·손실 보정·융합 가중·lambda·난수 규약은 C와 같다.
초기 출력 = 균등 감산 검사는 C.train이 한다. 모두 로컬 GPU(RTX 5080)에서 학습한다.

6시드 참고 집계는 시드 1~3에 sweep2 base 원장(같은 로컬 GPU, 같은 C.train)을 쓴다. sweep2에 없는 absk 시드 1~3은
이 폴더에서 같은 방식으로 로컬 학습한다(sweep1 absk는 다른 GPU에서 학습됨).

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_kvdiff_final_s456_20261118.py <action>
  train --variant V --seed K | train_campaign | models | pipe --role R --sequence S | pipe_campaign --role R | summary
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

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _kvdiff_variants_20261118 as V  # noqa: E402  (C 전역을 sweep1로 바꾸므로 아래에서 다시 바꾼다)

C, S, P = V.C, V.S, V.P
run = P.run
ROOT, CFG, U = C.ROOT, C.CFG, C.U
assert CFG == "x=log__n=ln__g=frozen__lr=1e-3__U=1536" and U == 512
OUT = ROOT / "results/kv_diff_improve_20261118/final_s456"
SW2 = ROOT / "results/kv_diff_improve_20261118/sweep2"
CONTROLS = ROOT / "results/pdv2_report_model_20261116/controls/summary.json"
C.OUT, C.RUNS, C.LEDGERS = OUT, OUT / "runs", OUT / "ledgers"
VARS = ("sym", "abskv", "absk", "diffk_absv", "sum_kv")
SEEDS = (4, 5, 6)
EXTRA = {"absk": (1, 2, 3)}  # 6시드 참고용: sweep2 base에 없는 로컬 시드 1~3
MODELS = OUT / "models.txt"
GROUPS = ("all", "forward_only", "reverse_only")
PAIRS = [("sym", "abskv"), ("absk", "abskv"), ("diffk_absv", "abskv"), ("sum_kv", "abskv"),
         ("absk", "sym"), ("diffk_absv", "sym"), ("sum_kv", "sym")] + [(v, "uniform") for v in VARS]
# 2×2 분해에 필요한 나머지 방향(위 쌍의 부호 반대): K 효과 at V=abs, V 효과 at K=abs
DECOMP_PAIRS = [("abskv", "diffk_absv"), ("abskv", "absk")]


def jobs() -> list:
    return [(v, k) for v in VARS for k in SEEDS] + [(v, k) for v, ks in EXTRA.items() for k in ks]


def tag(v: str, k: int) -> str:
    return f"{v}@s{k}@u{U}"


# ----------------------------------------------------------------- 학습
def train_campaign() -> None:
    logs = OUT / "logs" / "train"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    for v, k in jobs():
        if (C.RUNS / v / f"s{k}" / "training_complete.json").exists():
            continue
        with (logs / f"{v}_s{k}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "train", "--variant", v, "--seed", str(k)],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        print("TRAINED", v, k, round(time.time() - tick, 1), flush=True)
    P.write(OUT / "logs" / "train_campaign.done.json", dict(seconds=time.time() - tick))


# ----------------------------------------------------------------- 실제 검색 (C.pipe; C.model_spec -> C.load -> V.build)
def pipe_campaign(role: str, workers: int, nprefix: int) -> None:
    seqs = C.VAL if role == "val" else C.TESTSEQ
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()

    def call(s):
        with (logs / f"pipe_{role}_{s.replace('/', '_')}.log").open("a") as f:
            subprocess.run([sys.executable, __file__, "pipe", "--role", role, "--sequence", s, "--prefix", str(nprefix)],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(concurrent.futures.as_completed([ex.submit(call, s) for s in seqs]), 1):
            fut.result()
            print("DONE", role, i, len(seqs), round(time.time() - tick, 1), flush=True)
    P.write(logs / f"pipe_campaign_{role}.done.json", dict(seconds=time.time() - tick, workers=workers))


# ----------------------------------------------------------------- 집계 (C.summary의 방향별 규약)
def ledger(role: str, name: str, t: str) -> Path:
    if t == "uniform":
        return S.LEDGERS / role / name / "uniform.npz"
    if "@base@" in t:
        return SW2 / "ledgers" / role / name / (t + ".npz")
    return C.LEDGERS / role / name / (t + ".npz")


def summary() -> None:
    import _pdv2_scratch_search_compare_20261116 as K

    keys, metrics = K.KEYS, C.METRICS
    sets = {"s456": {v: [tag(v, k) for k in SEEDS] for v in VARS},
            "s123456": {v: ([tag(v, k) for k in EXTRA[v]] if v in EXTRA else [f"{v}@base@s{k}" for k in (1, 2, 3)])
                        + [tag(v, k) for k in SEEDS] for v in VARS}}
    for arms in sets.values():
        arms["uniform"] = ["uniform"]
    every = sorted({t for arms in sets.values() for ts in arms.values() for t in ts})
    out = dict(config=CFG, updates=U, variants=list(VARS), seeds=list(SEEDS),
               fixed=dict(weights=P.selected()[0].tolist(), lam=P.selected()[1]), arms=sets)
    valpm = {t: S.macro("val", {s: run.READ(ledger("val", s.replace("/", "_"), t).with_suffix(".json")) for s in C.VAL}) for t in every}
    rows = {}
    for t in every:
        for seqs in K.CITY.values():
            for s in seqs:
                name = s.replace("/", "_")
                d = dict(np.load(K.DIRS / (name + ".npz")))
                with np.load(ledger("test", name, t)) as z:
                    np.testing.assert_array_equal(z["query_idx"], d["query_idx"])
                    masks = dict(all=None, forward_only=d["group"] == 0, reverse_only=d["group"] == 1)
                    rows.setdefault(t, {})[s] = {g: K.metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"], m)
                                                 for g, m in masks.items()}

    def stat(vals):
        return dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None, values=list(map(float, vals)))

    res = {}
    for sname, arms in sets.items():
        val = {arm: {k: stat([valpm[t][k] for t in ts]) for k in metrics} for arm, ts in arms.items()}
        test = {}
        for g in GROUPS:
            def macro(t):
                cities = []
                for seqs in K.CITY.values():
                    v = [rows[t][s][g] for s in seqs if rows[t][s][g] is not None]
                    if v:
                        cities.append({k: np.mean([x[k] for x in v]) for k in keys})
                return {k: float(np.mean([c[k] for c in cities])) for k in keys}
            per_model = {t: macro(t) for ts in arms.values() for t in ts}
            st = {arm: {k: stat([per_model[t][k] for t in ts]) for k in keys} for arm, ts in arms.items()}
            comp = {}
            for a, b in PAIRS + DECOMP_PAIRS:
                rng = np.random.default_rng(20261104)
                draws, city = [], {}
                for cname, seqs in K.CITY.items():
                    avail = [s for s in seqs if rows[arms[a][0]][s][g] is not None]
                    if not avail:
                        continue
                    mat = np.array([[100 * (np.mean([rows[t][s][g][k] for t in arms[a]]) - np.mean([rows[t][s][g][k] for t in arms[b]]))
                                     for k in keys] for s in avail])
                    city[cname] = dict(zip(keys, mat.mean(0).tolist()))
                    idx = rng.integers(0, len(avail), size=(10000, len(avail)))
                    draws.append(mat[idx].mean(1))
                boot = np.stack(draws).mean(0)
                comp[f"{a}-{b}"] = dict(point_pp={k: 100 * (st[a][k]["mean"] - st[b][k]["mean"]) for k in keys},
                                        conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(keys)},
                                        val9_mean_diff_pp={k: 100 * (val[a][k]["mean"] - val[b][k]["mean"]) for k in metrics},
                                        city_delta_pp=city)
            test[g] = dict(per_model=per_model, arms=st, differences=comp)
        # 2×2 분해(시험 AP 등, 점추정 %p): K 효과 = 절대K - 차이K, V 효과 = 절대V - 차이V
        dec = {}
        for g in GROUPS:
            m = {v: {k: test[g]["arms"][v][k]["mean"] for k in keys} for v in ("sym", "absk", "diffk_absv", "abskv")}
            dec[g] = {k: dict(K_effect_at_Vdiff=100 * (m["absk"][k] - m["sym"][k]), K_effect_at_Vabs=100 * (m["abskv"][k] - m["diffk_absv"][k]),
                              V_effect_at_Kdiff=100 * (m["diffk_absv"][k] - m["sym"][k]), V_effect_at_Kabs=100 * (m["abskv"][k] - m["absk"][k]),
                              interaction=100 * (m["abskv"][k] - m["diffk_absv"][k] - m["absk"][k] + m["sym"][k])) for k in keys}
        mv = {v: val[v]["AP"]["mean"] for v in ("sym", "absk", "diffk_absv", "abskv")}
        dec["val9_AP"] = dict(K_effect_at_Vdiff=100 * (mv["absk"] - mv["sym"]), K_effect_at_Vabs=100 * (mv["abskv"] - mv["diffk_absv"]),
                              V_effect_at_Kdiff=100 * (mv["diffk_absv"] - mv["sym"]), V_effect_at_Kabs=100 * (mv["abskv"] - mv["absk"]))
        res[sname] = dict(val=val, test=test, decomposition_2x2_pp=dec)
    out.update(res)
    out["val_per_model"] = valpm
    # 검산: 시험46 파이프라인 집계(S.macro)와 compare 경로(전체)
    out["metric_checks"] = {t: max(abs(S.macro("test", {s: run.READ(ledger("test", s.replace("/", "_"), t).with_suffix(".json"))
                                                        for s in C.TESTSEQ})[k] - res["s123456"]["test"]["all"]["per_model"][t][k]) for k in keys)
                            for t in every}
    # 참고: 기존 보고 체크포인트(다른 GPU) controls의 nsd-abskv 시드 4~6
    ctl = run.READ(CONTROLS)
    out["reference_controls_other_gpu"] = {
        g: dict(controls_nsd_minus_abskv=ctl["test"][g]["differences"]["nsd-abskv"],
                controls_arms={a: {k: ctl["test"][g]["arms"][a][k]["mean"] for k in keys} for a in ("nsd", "abskv")},
                local_s456_sym_minus_abskv=dict(point_pp=res["s456"]["test"][g]["differences"]["sym-abskv"]["point_pp"],
                                                conditional_95_pp=res["s456"]["test"][g]["differences"]["sym-abskv"]["conditional_95_pp"]))
        for g in GROUPS}
    out["reference_controls_other_gpu"]["val9"] = dict(
        controls_nsd_minus_abskv_pp={k: 100 * (ctl["val"]["nsd"][k]["mean"] - ctl["val"]["abskv"][k]["mean"]) for k in metrics},
        local_s456_sym_minus_abskv_pp=res["s456"]["test"]["all"]["differences"]["sym-abskv"]["val9_mean_diff_pp"])
    tr = {}
    for v, k in jobs():
        r = run.READ(C.RUNS / v / f"s{k}" / "training_complete.json")
        tr[f"{v}@s{k}"] = dict(sha256=r["sha256"], local_sha256=run.HASH(C.RUNS / v / f"s{k}" / f"u{U}.pth"),
                               trainable_parameters=r["trainable_parameters"], total_parameters=r["total_parameters"],
                               init_bitwise_equal_uniform=all(r["init_bitwise_equal_uniform"].values()),
                               init_max_abs=max(r["init_max_abs_vs_uniform"].values()), seconds=r["seconds"], gpu=r["gpu"],
                               final_loss=float(np.mean([x["loss"] for x in r["records"][-8:]])),
                               reproduction_vs_scratch_search=r["reproduction_vs_scratch_search"], val_fixed_AP=r["val_fixed"]["macro"]["AP"])
    out["training"] = tr
    out["conventions"] = dict(
        val9="S.macro('val') 파이프라인 집계, 시드 평균, SD ddof=1",
        test46="KITTI 07(양성 없음) 제외 45시퀀스 -> 6도시 평균 -> 시드 평균(compare 스크립트 metric), SD ddof=1. "
               "방향별은 리뷰 방향 표지(group 0 forward_only, 1 reverse_only; C.summary와 같다)",
        intervals="a-b: 시퀀스별 (a 시드 평균 - b 시드 평균), 도시 안 시퀀스 재표집 10,000회, 도시 평균, 2.5/97.5% 분위, "
                  "쌍·방향 묶음·시드 묶음마다 난수 20261104 새로 시작",
        seeds="s456 = 최종 비교(시드 4,5,6, 이번 로컬 학습). s123456 = 참고용 6시드: 시드 1~3은 구성 탐색(sweep1·sweep2 선택)에 쓴 시드이며 "
              "sweep2 base 원장(로컬 RTX 5080, C.train)을 쓴다. absk 시드 1~3은 sweep2에 없어 이 폴더에서 같은 방식으로 로컬 학습했다",
        uniform="기존 탐색 원장(S.LEDGERS) uniform",
        decomposition="2×2: K 효과 = 절대K - 차이K(absk-sym, abskv-diffk_absv), V 효과 = 절대V - 차이V(diffk_absv-sym, abskv-absk), 점추정 %p",
        reference="reference_controls_other_gpu: results/pdv2_report_model_20261116/controls/summary.json의 nsd-abskv(다른 GPU 학습 체크포인트, 시드 4~6)",
        prefix="원장 접두 검사 생략(--prefix 0)", units="0~1 비율(차이·구간은 %p)")
    P.write(OUT / "summary.json", out)
    for sname in sets:
        print("====", sname)
        for v in sets[sname]:
            print(f"  {v:11s} val AP {100 * res[sname]['val'][v]['AP']['mean']:.3f}",
                  "test", {k: round(100 * res[sname]["test"]["all"]["arms"][v][k]["mean"], 3) for k in keys})
        for g in GROUPS:
            print("  --", g)
            for c, d in res[sname]["test"][g]["differences"].items():
                print(f"    {c:20s}", {k: (round(d["point_pp"][k], 3), [round(x, 3) for x in d["conditional_95_pp"][k]]) for k in keys})
        print("  DECOMP", res[sname]["decomposition_2x2_pp"]["all"]["AP"], res[sname]["decomposition_2x2_pp"]["val9_AP"])
    print("CHECKS", max(out["metric_checks"].values()))
    print("TRAIN", {k: (v["gpu"], v["init_bitwise_equal_uniform"], v["sha256"] == v["local_sha256"]) for k, v in tr.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("--variant", choices=list(VARS))
    p.add_argument("--seed", type=int)
    p.add_argument("--role", default="val")
    p.add_argument("--sequence")
    p.add_argument("--prefix", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    P.e.configure()
    torch.set_num_threads(4)
    if a.action == "train":
        assert (a.variant, a.seed) in jobs()
        C.train(a.variant, a.seed)
    elif a.action == "train_campaign":
        train_campaign()
    elif a.action == "models":
        MODELS.parent.mkdir(parents=True, exist_ok=True)
        MODELS.write_text("\n".join(tag(v, k) for v, k in jobs()) + "\n")
    elif a.action == "pipe":
        C.pipe(a.role, a.sequence, MODELS.read_text().split(), a.prefix)
    elif a.action == "pipe_campaign":
        assert a.workers <= 2
        pipe_campaign(a.role, a.workers, a.prefix)
    elif a.action == "summary":
        summary()
    else:
        raise SystemExit(a.action)
