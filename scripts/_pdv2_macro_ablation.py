#!/usr/bin/env python3
"""현재 NSD(diff, matching_only)의 3단계 거시 대조를 시험46·3시작점에 적용한다.

- no_alignment: 반환 장소를 후보 합집합의 두 코사인 점수(원키·BEV 크기 색인)만으로 고른다.
  복소 정렬 두 채널의 재점수를 쓰지 않으며, 최종 수락 점수는 그 장소에 대해 원래 식으로 계산한다.
- no_temporal: 반환 장소는 그대로 두고, 최종 수락 점수에서 시간 이웃의 가중치를 0, 중심 쌍을 1로 둔다.

체크포인트·설정·후보 합집합·융합 가중치는 기존 시험 실행(`results/pdv2_diffattn_contrast_test_20261104`)과 같고
재선택하지 않는다. 시험셋은 대조 측정에만 쓰고 설정 선택에 쓰지 않는다.
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import _pdv2_contrast_pipeline as pipeline
import _pdv2_contrast_pipeline_test as T

run, ROOT, e = pipeline.run, pipeline.ROOT, pipeline.e
SOURCE = T.OUT
OUT = ROOT / "results/pdv2_macro_ablation_20261106"
CONDITIONS = ("no_alignment", "no_temporal")
SEEDS = (1, 2, 3)


def protocol() -> dict:
    """대조 정의와 원천 해시를 실행 전에 고정한다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    paths = [Path(__file__), Path(pipeline.__file__), Path(T.__file__),
             SOURCE / "protocol.json", SOURCE / "summary.json", SOURCE / "integrated_validation.json"]
    sources = {str(p.relative_to(ROOT)): run.HASH(p) for p in paths}
    if path.exists():
        spec = run.READ(path)
        for name, digest in spec["sources"].items():
            assert run.HASH(ROOT / name) == digest, name
        return spec
    spec = dict(
        created_unix=time.time(),
        model="diff/matching_only, 시작점 3개(diff_s1~s3) 기존 체크포인트",
        conditions=dict(
            no_alignment="반환 장소 = 후보 합집합에서 고정 코사인 부분합(w_key·minmax 원키 + w_aux·minmax BEV 크기)의 최대. 복소 정렬 두 채널 재점수 없음. 수락 점수는 원래 시간 증거 식.",
            no_temporal="반환 장소 = NSD 원 승자. 수락 점수 = 같은 스펙트럼 점수식에서 중심 쌍 가중치 1, 시간 이웃 가중치 0.",
        ),
        fixed="체크포인트·후보 합집합·융합 가중치(aux_half)·점수 척도는 기존 시험과 같고 재선택하지 않는다.",
        checks="NSD 원장의 승자 재계산 일치와 원 점수 재현(<8e-6), 조건별 접두 입력(q+15) 재계산 2질의/시퀀스, 독립 지표 두 구현 대조.",
        intervals=dict(replicates=10000, seed=20261104, method="기존 integrated_report와 같은 도시 안 양성 시퀀스 복원추출, 세 시작점 평균 차이, 도시 동일 비중"),
        sources=sources,
    )
    run.WRITE(path, spec)
    return spec


def batched_score(z: torch.Tensor, plan: dict) -> np.ndarray:
    """기존 최종 수락 점수식을 128질의씩 계산한다."""
    parts = []
    for start in range(0, len(plan["query"]), 128):
        pp = {k: torch.from_numpy(v[start:start + 128]).cuda() for k, v in plan.items()}
        parts.append(e.prior.score(z, pp).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def center_only(plan: dict) -> dict:
    """시간 이웃의 가중치를 0, 중심 쌍을 1로 둔다."""
    weight = np.zeros_like(plan["weight"])
    weight[..., 0] = 1.0
    return dict(plan, weight=weight)


def static_winner(key: np.ndarray, aux: np.ndarray, weights: np.ndarray, q: int) -> int:
    """기존 후보 합집합 규칙과 두 코사인 부분합만으로 반환 장소를 고른다."""
    ids = np.unique(np.r_[run.base.prefix_topk(key, q, q - 29, 100), run.base.prefix_topk(aux, q, q - 29, 100)])
    score = weights[0] * run.base.minmax(key[ids] @ key[q]) + weights[1] * run.base.minmax(aux[ids] @ aux[q])
    return int(ids[np.argmax(score)])


def metrics(row: dict, value: np.ndarray) -> dict:
    """두 독립 지표 구현이 같은지 확인한다."""
    metric = run.base.independent_metrics(row, value)
    other = run.white.prior.BASE.exp.metrics(value, row)
    for m in run.base.METRICS:
        if metric[m] is not None:
            np.testing.assert_allclose(metric[m], other[m], atol=1e-12, rtol=0)
    return metric


def worker(sequence: str) -> None:
    """한 시퀀스의 세 시작점에서 두 조건을 계산하고 검사한다."""
    spec = T.activate()
    protocol()
    name = sequence.replace("/", "_")
    src = SOURCE / "test" / name
    folder = OUT / "test" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        return
    cache, pos, minimum, travel, bank, _, _, _ = run.inputs("test", sequence)
    graph, _, audit = e.prior.graph_inputs("test", sequence)
    raw = pipeline.raw_key(cache["descriptors"])
    weights = run.cal.fusion_weights(spec["confidence"])[run.cal.FUSIONS.index("aux_half")]
    poolpath = src / "candidate_pool.npz"
    assert run.HASH(poolpath) == run.READ(poolpath.with_suffix(".json"))["sha256"]
    pool = run.ARR(poolpath)
    q = pool["query"]
    offsets = pool["offset"]
    static_j = np.asarray([pool["ids"][a:b][np.argmax(pool["static"][a:b])] for a, b in zip(offsets[:-1], offsets[1:])], np.int64)
    assert (static_j <= q - 30).all()
    out = dict(metrics={}, reproduction={}, prefix_checks=[], input_audit=audit,
               candidate_pool_sha256=run.HASH(poolpath))
    with torch.no_grad():
        for seed in SEEDS:
            tag = f"diff_s{seed}"
            case = spec["cases"][tag]
            refpath = src / (tag + ".npz")
            assert run.HASH(refpath) == run.READ(refpath.with_suffix(".json"))["sha256"]
            ref = run.ARR(refpath)
            net = pipeline.load_model(case)
            z = pipeline.transform(net, graph, "diff")
            change = run.base.variant_inputs(z.cpu().numpy(), 8)
            winners = pipeline.winners(pool, change, weights)
            np.testing.assert_array_equal(winners, ref["j_star"])
            plan = e.prior.base.prior.plan(travel, q, ref["j_star"])
            full = batched_score(z, plan)
            error = float(abs(full - ref["score"]).max())
            assert error < 8e-6, error
            out["reproduction"][tag] = dict(winners_exact=True, score_error=error)
            base_row = {k: ref[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")}
            rows = {
                "no_temporal": (base_row, batched_score(z, center_only(plan))),
                "no_alignment": (
                    dict(base_row, j_star=static_j, top1_geo_dist=np.linalg.norm(pos[q] - pos[static_j], axis=1)),
                    batched_score(z, e.prior.base.prior.plan(travel, q, static_j)),
                ),
            }
            for cond, (row, value) in rows.items():
                key = f"{cond}_s{seed}"
                out["metrics"][key] = metrics(row, value)
                np.savez_compressed(folder / (key + ".npz"), **row, score=value)
            # 접두 입력 검사: q+15까지의 관측만으로 같은 승자·점수가 나오는지 확인한다.
            for at in (len(q) // 3, 2 * len(q) // 3):
                t = int(q[at])
                end = t + 16
                small, _, _ = e.prior.graph_inputs("test", sequence, end)
                short = pipeline.transform(net, small, "diff")
                skey = pipeline.raw_key(cache["descriptors"][:end])
                np.testing.assert_array_equal(skey, raw[:end])
                short_bank = run.base.variant_inputs(short.cpu().numpy(), 8)
                winner = run.cal.candidates(skey, (bank[0][:end], *short_bank[1:]), weights[None], t)
                assert int(winner[0]) == int(ref["j_star"][at])
                sw = static_winner(skey, bank[0][:end], weights, t)
                assert sw == int(static_j[at]), (sequence, t, sw, static_j[at])
                for cond, j, transform_plan in (("no_temporal", int(winner[0]), center_only),
                                                ("no_alignment", sw, lambda p: p)):
                    pp = transform_plan(e.prior.base.prior.plan(travel[:end], np.array([t]), np.array([j])))
                    check = batched_score(short, pp)
                    expected = rows[cond][1][at:at + 1]
                    err = float(abs(check - expected).max())
                    assert err < 8e-6, (cond, sequence, t, err)
                    out["prefix_checks"].append(dict(condition=f"{cond}_s{seed}", query=t, winner=j, score_error=err))
                del small, short
            del net, z, change
            print("EVALUATED", sequence, seed, {k: v["AP"] for k, v in out["metrics"].items() if k.endswith(f"s{seed}")}, flush=True)
    for key in out["metrics"]:
        out.setdefault("sha256", {})[key] = run.HASH(folder / (key + ".npz"))
    run.WRITE(folder / "summary.json", out)


def campaign(workers: int) -> None:
    """시퀀스별 작업을 별도 프로세스로 병렬 실행한다."""
    spec = T.activate()
    protocol()
    todo = [s for s in spec["test"] if not (OUT / "test" / s.replace("/", "_") / "summary.json").exists()]

    def one(sequence):
        result = subprocess.run([sys.executable, __file__, "worker", "--sequence", sequence],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(sequence + "\n" + result.stderr[-3000:])
        return sequence

    with ThreadPoolExecutor(workers) as pool:
        for done in pool.map(one, todo):
            print("DONE", done, flush=True)


def intervals(per: dict, a: str, b: str, plan: dict) -> dict:
    """기존 integrated_report와 같은 도시 안 시퀀스 재표집 구간을 계산한다."""
    rng = np.random.default_rng(plan["seed"])
    draws, city_delta = [], {}
    for city, sequences in run.current.CITY.items():
        available = [s for s in sequences if per[f"{a}_s1"]["sequences"][s]["n_pos"]]
        matrix = np.array([[np.mean([100 * (per[f"{a}_s{k}"]["sequences"][s][m] - per[f"{b}_s{k}"]["sequences"][s][m])
                                     for k in SEEDS]) for m in run.base.METRICS] for s in available])
        city_delta[city] = dict(zip(run.base.METRICS, matrix.mean(0).tolist()))
        index = rng.integers(0, len(available), size=(plan["replicates"], len(available)))
        draws.append(matrix[index].mean(1))
    boot = np.stack(draws).mean(0)
    return dict(city_delta_pp=city_delta,
                conditional_95_pp={m: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, m in enumerate(run.base.METRICS)})


def finish() -> None:
    """시퀀스→도시6→시작점3으로 집계하고 NSD와의 짝지은 구간을 계산한다."""
    spec = T.activate()
    plan = protocol()
    per = {f"{c}_s{s}": {} for c in ("diff", *CONDITIONS) for s in SEEDS}
    count = 0
    checks = 0
    for sequence in spec["test"]:
        name = sequence.replace("/", "_")
        meta = run.READ(OUT / "test" / name / "summary.json")
        source = run.READ(SOURCE / "test" / name / "summary.json")
        count += len(run.ARR(SOURCE / "test" / name / "candidate_pool.npz")["query"])
        checks += len(meta["prefix_checks"])
        for key, digest in meta["sha256"].items():
            assert run.HASH(OUT / "test" / name / (key + ".npz")) == digest
        for s in SEEDS:
            per[f"diff_s{s}"][sequence] = source["metrics"][f"diff_s{s}"]
            for c in CONDITIONS:
                per[f"{c}_s{s}"][sequence] = meta["metrics"][f"{c}_s{s}"]
    assert count == 195726
    results = {tag: dict(**run.current.adoption.previous.macro(rows, run.current.CITY), sequences=rows) for tag, rows in per.items()}
    published = run.READ(SOURCE / "summary.json")
    means, stds = {}, {}
    for kind in ("diff", *CONDITIONS):
        vals = {m: [results[f"{kind}_s{s}"]["macro"][m] for s in SEEDS] for m in run.base.METRICS}
        means[kind] = {m: float(np.mean(v)) for m, v in vals.items()}
        stds[kind] = {m: float(np.std(v, ddof=1)) for m, v in vals.items()}
    for m in run.base.METRICS:
        assert abs(means["diff"][m] - published["mean"]["diff"][m]) < 1e-12
    comparisons = {f"diff-{c}": dict(delta_pp={m: 100 * (means["diff"][m] - means[c][m]) for m in run.base.METRICS},
                                     **intervals(results, "diff", c, plan["intervals"])) for c in CONDITIONS}
    report = dict(status="passed", queries_per_case=count, prefix_checks=checks, mean=means, std=stds,
                  per_case=results, comparisons=comparisons, test_used_for_selection=False,
                  sources={str(p.relative_to(ROOT)): run.HASH(p) for p in (OUT / "protocol.json", Path(__file__))})
    run.WRITE(OUT / "summary.json", report)
    print({k: {m: round(100 * v, 3) for m, v in d.items()} for k, d in means.items()})
    print({k: {m: [round(x, 3) for x in v] for m, v in d["conditional_95_pp"].items()} for k, d in comparisons.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("protocol", "worker", "campaign", "finish"))
    parser.add_argument("--sequence")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    e.configure() if hasattr(e, "configure") else None
    if args.action == "worker":
        worker(args.sequence)
    elif args.action == "campaign":
        campaign(args.workers)
    elif args.action == "finish":
        finish()
    else:
        protocol()
