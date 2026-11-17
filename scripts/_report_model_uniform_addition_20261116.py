#!/usr/bin/env python3
"""균등 더하기(이웃 평활) 대조를 새 고정 설정(융합 (2,1,2,2), λ=0.25)에서 다시 계산한다.

정의는 scripts/_pdv2_uniform_addition_control.py(2026-11-10)와 같다: 균등 감산과 같은 transport_difference·두 층·균등 주의에서
세기만 η>0(한 층 = (1−η)·자기 + η·이웃 평균). 신경망·학습 없음. 세기 후보 {0.0625, 0.125, 0.25, 0.5, 1}, 검증9 파이프라인 집계
AP 최대(1e-9 안 동률이면 작은 세기)로 하나를 고르고 시험46에는 그 세기만 보고한다. 시험셋은 선택에 쓰지 않는다.

검색·점수 경로는 scripts/_report_model_ablations_20261116.py의 worker와 같다(재선택 융합으로 다시 계산한 후보 정적 점수,
pipeline.winners, prior.base.prior.plan, P.score(z, plan, λ)).

재현 검사
- 같은 경로로 S 원장의 uniform(새 설정 균등 감산)을 승자·점수 비트 단위로 재현한다(검증9·시험46).
- 옛 설정(P.CURRENT_W, λ=0.25, 옛 후보 정적 점수)으로 옛 균등 더하기 원장(/workspace/data/nsd_results/pdv2_uniform_addition_20261110)을
  재현해 조건 구현을 확인한다(검증: 다섯 세기, 시험: +0.0625). 옛 검증 identity도 재현한다.
- 시험 identity(정제 없음)는 단위 B 원장(ablations/ledgers/test/<seq>/identity.npz, 새 설정)을 그대로 쓴다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_report_model_uniform_addition_20261116.py <action>
  worker --role val|test --sequence S
  campaign --role val|test      (순차 1개)
  select                        검증9 세기 선택
  finish                        집계·짝지은 구간 → ablations/uniform_addition_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
ABL = ROOT / "results/pdv2_report_model_20261116/ablations"
OUT = ABL / "uniform_addition"
LED = OUT / "ledgers"
SUMMARY = ABL / "uniform_addition_summary.json"
OLD = Path("/workspace/data/nsd_results/pdv2_uniform_addition_20261110")
STRENGTHS = (0.0625, 0.125, 0.25, 0.5, 1.0)
KEYS = ("AP", "max_F1", "R@100P", "R@1")
BOOT_SEED = 20261104  # 원고 부록 짝지은 구간표(Table paired_current)와 같은 난수


def tag(eta: float) -> str:
    return "add_" + f"{eta:g}".replace(".", "p")


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


def read(path: Path):
    return json.loads(Path(path).read_text())


def sequences(role: str) -> list:
    spec = read(ROOT / "results/pdv2_diffattn_contrast_test_20261104/protocol.json")
    return spec["val"] if role == "val" else spec["test"]


def worker(role: str, sequence: str) -> None:
    import torch
    import _pdv2_scratch_search_20261116 as S
    P, run, prior, pipeline = S.P, S.run, S.prior, S.pipeline
    name = sequence.replace("/", "_")
    folder = LED / role / name
    folder.mkdir(parents=True, exist_ok=True)
    src = (P.PIPE / "val" / name) if role == "val" else (P.TEST / "test" / name)
    sledger = S.LEDGERS / role / name
    cache, pos, minimum, travel, bank, _, _, _ = P.inputs(role, sequence)
    graph, _, _ = prior.graph_inputs(role, sequence)
    raw = pipeline.raw_key(cache["descriptors"])
    new_w, lam = P.selected()
    assert [float(x) for x in new_w] == [2.0, 1.0, 2.0, 2.0] and lam == 0.25
    poolpath = src / "candidate_pool.npz"
    assert run.HASH(poolpath) == run.READ(poolpath.with_suffix(".json"))["sha256"]
    base_pool = run.ARR(poolpath)
    q = base_pool["query"]
    off = base_pool["offset"]
    static = []
    for n, t in enumerate(q):
        ids = base_pool["ids"][off[n]:off[n + 1]]
        static.append(new_w[0] * run.base.minmax(raw[ids] @ raw[t]) + new_w[1] * run.base.minmax(bank[0][ids] @ bank[0][t]))
    new_pool = dict(base_pool, static=np.concatenate(static).astype(np.float32))

    def row_of(j):
        assert (j <= q - 30).all()
        return dict(query_idx=q, j_star=j, min_past_gt_dist=minimum[q],
                    top1_geo_dist=np.linalg.norm(pos[q] - pos[j], axis=1), group=np.zeros(len(q), np.int64))

    def score(z, plan, lam_=lam):
        return P.batched(lambda pp: P.score(z, pp, lam_).cpu().numpy(), plan, len(q)).astype(np.float64)

    def metrics(row, value):
        m = run.base.independent_metrics(row, value)
        o = run.white.prior.BASE.exp.metrics(value, row)
        for k in KEYS:
            if m[k] is not None:
                np.testing.assert_allclose(m[k], o[k], atol=1e-12, rtol=0)
        return m

    def compare(tagname, row, value, ref_path):
        r = run.ARR(ref_path)
        rec = dict(reference=str(ref_path), query_equal=bool(np.array_equal(q, r["query_idx"])),
                   winners_equal=bool(np.array_equal(row["j_star"], r["j_star"])),
                   score_bitwise_equal=bool(np.array_equal(value, r["score"])),
                   score_max_error=float(np.abs(value - r["score"]).max()))
        write(folder / ("check_" + tagname + ".json"), rec)
        return rec

    def evaluate(z, weights=new_w, pool=new_pool, lam_=lam):
        change = run.base.variant_inputs(z.cpu().numpy(), 8)
        j = pipeline.winners(pool, change, weights)
        return row_of(j), score(z, prior.base.prior.plan(travel, q, j), lam_)

    tick = time.time()
    with torch.no_grad():
        # 새 설정 균등 감산 재현(S 원장과 비트 단위)
        row, value = evaluate(prior.filtered(graph, -0.5))
        rec = compare("uniform", row, value, sledger / "uniform.npz")
        assert rec["query_equal"] and rec["winners_equal"] and rec["score_bitwise_equal"], rec
        etas = STRENGTHS if role == "val" else (read(OUT / "selection.json")["selected_eta"],)
        for eta in etas:
            z = prior.filtered(graph, float(eta))
            row, value = evaluate(z)
            np.savez_compressed(folder / (tag(eta) + ".npz"), **row, score=value)
            write(folder / (tag(eta) + ".json"), dict(metrics=metrics(row, value), sha256=run.HASH(folder / (tag(eta) + ".npz"))))
            # 옛 설정 재현(조건 구현 확인)
            oldref = OLD / ("val_eval/val" if role == "val" else "test_eval/test") / name / (tag(eta) + ".npz")
            if oldref.exists():
                orow, ovalue = evaluate(z, P.CURRENT_W, base_pool, 0.25)
                rec = compare("old_" + tag(eta), orow, ovalue, oldref)
                assert rec["query_equal"] and rec["winners_equal"] and rec["score_max_error"] < 8e-6, rec
            del z
        if role == "val":
            # 정제 없음(새 설정): pipeline.worker의 identity와 같은 정의(z=원래 복소 특징, 정렬 뱅크=입력 뱅크)
            z = graph.spectrum
            j = pipeline.winners(new_pool, bank, new_w)
            row = row_of(j)
            value = score(z, prior.base.prior.plan(travel, q, j))
            np.savez_compressed(folder / "identity.npz", **row, score=value)
            write(folder / "identity.json", dict(metrics=metrics(row, value), sha256=run.HASH(folder / "identity.npz")))
            jo = pipeline.winners(base_pool, bank, P.CURRENT_W)
            rec = compare("old_identity", row_of(jo), score(z, prior.base.prior.plan(travel, q, jo), 0.25), src / "identity.npz")
            assert rec["query_equal"] and rec["winners_equal"] and rec["score_max_error"] < 8e-6, rec
    write(folder / "timing.json", dict(seconds=time.time() - tick))
    print("WORKER", role, sequence, round(time.time() - tick, 1), flush=True)


def campaign(role: str) -> None:
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    for i, s in enumerate(sequences(role), 1):
        if (LED / role / s.replace("/", "_") / "timing.json").exists():
            continue
        with (logs / f"{role}_{s.replace('/', '_')}.log").open("a") as f:
            r = subprocess.run([sys.executable, __file__, "worker", "--role", role, "--sequence", s], stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(s)
        print("DONE", i, s, round(time.time() - tick, 1), flush=True)
    write(logs / f"campaign_{role}.done.json", dict(seconds=time.time() - tick))


def select() -> None:
    import _pdv2_scratch_search_20261116 as S
    run = S.run
    table = {}
    for t_ in ["uniform", "identity"] + [tag(e) for e in STRENGTHS]:
        per = {}
        for s in sequences("val"):
            name = s.replace("/", "_")
            p = (S.LEDGERS / "val" / name / "uniform.json") if t_ == "uniform" else (LED / "val" / name / (t_ + ".json"))
            per[s] = read(p)["metrics"]
        table[t_] = run.base.macro(per)["macro"]
    ref = read(S.OUT / "agg.json")["val"]["models"]["uniform"]["macro"]
    for k in KEYS:
        assert abs(table["uniform"][k] - ref[k]) < 1e-12
    chosen = sorted(STRENGTHS, key=lambda e: (-round(table[tag(e)]["AP"], 9), e))[0]
    write(OUT / "selection.json", dict(
        selected_eta=chosen, selected_tag=tag(chosen),
        rule="검증9 파이프라인 집계(run.base.macro) AP 최대 세기 하나(1e-9 안 동률이면 작은 세기). 시험셋 미사용. _pdv2_uniform_addition_control.select와 같은 규칙, 새 고정 설정.",
        val_macro_percent={t_: {k: 100 * v[k] for k in KEYS} for t_, v in table.items()}, test_used=False))
    print("SELECTED", chosen, {t_: round(100 * v["AP"], 3) for t_, v in table.items()}, flush=True)


def finish() -> None:
    from _pr_eval_lib import classify_queries, pr_sweep
    S_OUT = ROOT / "results/pdv2_scratch_search_20261116"
    CITY = read(ROOT / "results/pdv2_scratch_pilot_20261116/defaultbn_3seed/city.json")
    seqs = sequences("test")
    sel = read(OUT / "selection.json")
    add = sel["selected_tag"]

    def path_of(cond, s):
        name = s.replace("/", "_")
        if cond == "uniform":
            return S_OUT / "ledgers/test" / name / "uniform.npz"
        if cond == "identity":
            return ABL / "ledgers/test" / name / "identity.npz"
        return LED / "test" / name / (cond + ".npz")

    def metric(score, top1, minimum):  # _report_model_ablations_20261116.finish와 같은 독립 재계산
        qc = classify_queries(minimum)
        if not qc["n_pos"]:
            return None
        v = pr_sweep(score, top1, qc)
        return {**{k: float(v[k]) for k in KEYS[:3]}, "R@1": float(np.mean(top1[qc["positive"]] <= 5))}

    rows, maxerr = {}, 0.0
    for c in ("uniform", "identity", add):
        rows[c] = {}
        for s in seqs:
            p = path_of(c, s)
            with np.load(p) as z:
                rows[c][s] = metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"])
            ref = read(p.with_suffix(".json"))["metrics"]
            if rows[c][s] is not None:
                maxerr = max(maxerr, max(abs(rows[c][s][k] - ref[k]) for k in KEYS))
    assert maxerr < 1e-9, maxerr

    def macro(c):
        return {k: float(np.mean([np.mean([rows[c][s][k] for s in v if rows[c][s] is not None]) for v in CITY.values()])) for k in KEYS}

    def city(c):
        return {cy: {k: 100 * float(np.mean([rows[c][s][k] for s in v if rows[c][s] is not None])) for k in KEYS} for cy, v in CITY.items()}

    means = {c: macro(c) for c in rows}
    # 기준 수치 재현: 균등 감산(공통 안내), 정제 없음(단위 B 요약)
    brief = {"AP": 89.198, "max_F1": 89.244, "R@100P": 68.820, "R@1": 91.488}
    for k in KEYS:
        assert round(100 * means["uniform"][k], 3) == brief[k], (k, means["uniform"][k])
    nr = read(ABL / "no_refining_summary.json")["mean"]["identity"]
    for k in KEYS:
        assert abs(means["identity"][k] - nr[k]) < 1e-12, k

    def paired(a, b):
        rng = np.random.default_rng(BOOT_SEED)
        draws, city_delta = [], {}
        for cy, v in CITY.items():
            avail = [s for s in v if rows[a][s] is not None and rows[b][s] is not None]
            mat = np.array([[100 * (rows[a][s][k] - rows[b][s][k]) for k in KEYS] for s in avail])
            city_delta[cy] = dict(zip(KEYS, mat.mean(0).tolist()))
            idx = rng.integers(0, len(avail), size=(10000, len(avail)))
            draws.append(mat[idx].mean(1))
        boot = np.stack(draws).mean(0)
        return dict(delta_pp={k: float(np.mean([city_delta[c][k] for c in city_delta])) for k in KEYS}, city_delta_pp=city_delta,
                    conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(KEYS)})

    repro = {}
    for role in ("val", "test"):
        for s in sequences(role):
            for p in (LED / role / s.replace("/", "_")).glob("check_*.json"):
                r = read(p)
                e = repro.setdefault(f"{role}/{p.stem[6:]}", dict(sequences=0, winners_equal=True, score_bitwise_equal=True, score_max_error=0.0))
                e["sequences"] += 1
                e["winners_equal"] &= r["winners_equal"]
                e["score_bitwise_equal"] &= r["score_bitwise_equal"]
                e["score_max_error"] = max(e["score_max_error"], r["score_max_error"])
    write(SUMMARY, dict(
        status="passed", script="scripts/_report_model_uniform_addition_20261116.py",
        definition="균등 더하기: 균등 감산과 같은 두 층 transport_difference, 균등 주의, 세기 η>0(한 층 = (1−η)자기 + η·이웃 평균). 학습 없음. 후보·융합 (2,1,2,2)·정렬·시간 수락·λ=0.25는 균등 감산과 같다.",
        selection=sel, test_macro_percent={c: {k: 100 * means[c][k] for k in KEYS} for c in means},
        labels={"uniform": "균등 감산(새 설정, S 원장)", "identity": "정제 없음(단위 B 원장)", add: "균등 더하기(검증 선택 세기)"},
        comparisons={"add-uniform": paired(add, "uniform"), "add-identity": paired(add, "identity"), "uniform-identity": paired("uniform", "identity")},
        city_percent={c: city(c) for c in means},
        reproduction=repro, independent_metric_max_error=maxerr,
        aggregation="양성 있는 45시퀀스 -> 6도시 균등. 결정적 조건이라 시드 평균 없음.",
        bootstrap=dict(replicates=10000, seed=BOOT_SEED, unit="도시 안 시퀀스 복원추출, 도시 균등(_report_model_ablations_20261116.paired와 같은 구현)"),
        test_used_for_selection=False))
    print(json.dumps({c: {k: round(100 * means[c][k], 3) for k in KEYS} for c in means}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("worker", "campaign", "select", "finish"))
    ap.add_argument("--role", choices=("val", "test"))
    ap.add_argument("--sequence")
    a = ap.parse_args()
    if a.action == "worker":
        import torch
        import _pdv2_scratch_search_20261116 as S
        S.P.e.configure()  # 단위 B·S와 같은 결정적 설정(TF32 끔 등)
        torch.set_num_threads(4)
        worker(a.role, a.sequence)
    elif a.action == "campaign":
        campaign(a.role)
    else:
        globals()[a.action]()
