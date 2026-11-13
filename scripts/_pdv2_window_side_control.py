#!/usr/bin/env python3
"""시간 창 방향 대조: 균일 감산의 이웃을 과거 5개 또는 미래 5개로만 줄인 학습 없는 시간 전달.

원고 서론의 설계 이유("대칭 창은 순방향·역방향 주행에서 같은 구간을 덮고, 한쪽 창은 역방향에서 반대쪽을 덮는다")를
원고와 같은 평가 경로에서 확인한다. 갱신식·두 층·균일 주의·후보·융합 가중치·정렬·거리 맞춤 시간 판정은 균일 감산
(fixed, 앞뒤 5개, 세기 -0.5)과 같고, 전달에 쓰는 이웃 칸만 한쪽으로 줄인다. 시간 판정의 앞뒤 이웃은 그대로다.
세기는 한쪽 창마다 검증9의 AP로 고르고, 시험46에는 고른 세기와 -0.5를 보고한다. 방향별 층화는 표S7과 같은 라벨을 쓴다.

산출물은 D 드라이브(컨테이너 /workspace/data)에 둔다.
사용법(컨테이너 ring_baseline 안, 작업 폴더 /workspace/Neural-Spectral-Codec):
  python scripts/_pdv2_window_side_control.py val       # 검증9: 두 창 x 세기 4개 + fixed 재현, 세기 선택
  python scripts/_pdv2_window_side_control.py test      # 시험46: 고른 세기와 -0.5 + fixed 재현, 방향별 분석
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

OUT = Path(os.environ.get("NSD_SIDE_OUT", "/workspace/data/nsd_results/pdv2_window_side_20261111"))
VAL = OUT / "val_eval"
TEST = OUT / "test_eval"
OLD_VAL = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104"
OLD_TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
DIRECTIONS = ROOT / "results/pdv2_review_direction_yaw_20261105"
SIDES = ("past", "future")
STRENGTHS = (-0.125, -0.25, -0.5, -1.0)
METRICS = ("AP", "max_F1", "R@100P", "R@1")
GROUPS = ("all", "forward_only", "reverse_only")
BOOTSTRAP = 10000
BOOT_SEED = 20261111
FIXED = dict(kind="fixed", seed=0, checkpoint=None)


def READ(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def WRITE(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


def tag(side: str, eta: float) -> str:
    return f"{side}_{abs(eta):g}".replace(".", "p")


def case(side: str, eta: float) -> dict:
    return dict(kind=f"{side}{eta:+g}", seed=0, checkpoint=None)


def patch_transform(pipeline) -> None:
    """파이프라인 변환에 한쪽 창의 균일 감산만 추가한다. 칸 0~4는 과거, 6~10은 미래 이웃이다."""
    original = pipeline.transform

    def transform(net, graph, kind: str):
        side = next((s for s in SIDES if kind.startswith(s)), None)
        if side is None:
            return original(net, graph, kind)
        eta = float(kind[len(side):])
        assert eta < 0, eta
        full = graph.transport_mask
        mask = full.clone()
        if side == "past":
            mask[:, 6:] = 0
        else:
            mask[:, :5] = 0
        graph.transport_mask = mask
        try:
            return pipeline.e.prior.filtered(graph, eta)
        finally:
            graph.transport_mask = full

    pipeline.transform = transform


def copy_pool(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".json"):
        target = dst / ("candidate_pool" + suffix)
        if not target.exists():
            shutil.copy2(src / ("candidate_pool" + suffix), target)


def run_parallel(action: str, sequences: list, logs: Path) -> None:
    logs.mkdir(parents=True, exist_ok=True)

    def call(sequence):
        with (logs / (sequence.replace("/", "_") + ".log")).open("a") as stream:
            subprocess.run([sys.executable, __file__, action, "--sequence", sequence],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, s) for s in sequences]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            print(action.upper(), "DONE", i, len(futures), flush=True)


# ---------------------------------------------------------------- 검증
def val_spec() -> dict:
    path = VAL / "protocol.json"
    if path.exists():
        return READ(path)
    spec = dict(READ(OLD_VAL / "protocol.json"))
    cases = {"fixed": FIXED}
    cases.update({tag(s, eta): case(s, eta) for s in SIDES for eta in STRENGTHS})
    spec.update(created_unix=time.time(), cases=cases, scope="시간 창 방향 대조의 검증 세기 선택. 시험 미사용.")
    WRITE(path, spec)
    return spec


def val_worker(sequence: str) -> None:
    import _pdv2_contrast_pipeline as P

    P.e.configure()
    P.OUT = VAL
    P.run.OUT = VAL
    P.prepare = val_spec
    patch_transform(P)
    name = sequence.replace("/", "_")
    copy_pool(OLD_VAL / "val" / name, VAL / "val" / name)
    P.worker("val", sequence)


def val() -> None:
    import _pdv2_contrast_pipeline as P

    spec = val_spec()
    run_parallel("val_worker", spec["val"], VAL / "logs")
    rows = {s: READ(VAL / "val" / s.replace("/", "_") / "summary.json") for s in spec["val"]}
    table = {t: P.run.base.macro({s: r["metrics"][t] for s, r in rows.items()})["macro"] for t in spec["cases"]}
    reference = READ(OLD_VAL / "summary.json")["mean"]["fixed"]
    for m in METRICS:  # 기존 검증 보고값 재현
        assert abs(table["fixed"][m] - reference[m]) < 1e-12, m
    chosen = {}
    for side in SIDES:  # AP 최대, 1e-9 안의 동률이면 약한 세기
        chosen[side] = sorted(STRENGTHS, key=lambda eta: (-round(table[tag(side, eta)]["AP"], 9), abs(eta)))[0]
    WRITE(OUT / "selection.json", dict(
        selected=chosen,
        rule="검증9 파이프라인 집계 AP 최대인 세기를 창마다 하나 고른다. 시험셋은 쓰지 않는다.",
        val_macro_percent={t: {m: 100 * v[m] for m in METRICS} for t, v in table.items()},
        test_used=False))
    print("SELECTED", chosen, {t: round(100 * v["AP"], 3) for t, v in table.items()}, flush=True)


# ---------------------------------------------------------------- 시험
def test_arms() -> dict:
    """시험 조건: 창마다 검증에서 고른 세기와, 대칭 창과 같은 세기 -0.5."""
    chosen = READ(OUT / "selection.json")["selected"]
    arms = {}
    for side in SIDES:
        arms[side + "_selected"] = tag(side, chosen[side])
        arms[side + "_same"] = tag(side, -0.5)
    return arms


def test_spec() -> dict:
    path = TEST / "protocol.json"
    if path.exists():
        return READ(path)
    chosen = READ(OUT / "selection.json")["selected"]
    spec = dict(READ(OLD_TEST / "protocol.json"))
    cases = {"fixed": FIXED}
    for side in SIDES:
        for eta in sorted({chosen[side], -0.5}):
            cases[tag(side, eta)] = case(side, eta)
    spec.update(created_unix=time.time(), cases=cases,
                test_policy="검증에서 고른 세기와 -0.5의 시험 보고. 설정 선택 없음.")
    WRITE(path, spec)
    return spec


def test_worker(sequence: str) -> None:
    import _pdv2_contrast_pipeline_test as T

    T.OUT = TEST
    T.prepare = test_spec
    T.pipeline.e.configure()
    patch_transform(T.pipeline)
    name = sequence.replace("/", "_")
    folder = TEST / "test" / name
    copy_pool(OLD_TEST / "test" / name, folder)
    T.worker("test", sequence)
    new = np.load(folder / "fixed.npz")
    ref = np.load(OLD_TEST / "test" / name / "fixed.npz")
    for field in ("query_idx", "j_star"):
        np.testing.assert_array_equal(new[field], ref[field])
    error = float(np.abs(new["score"] - ref["score"]).max())
    assert error < 8e-6, error
    WRITE(folder / "reproduction_check.json", dict(status="passed", score_error=error))


def test() -> None:
    assert (OUT / "selection.json").exists()
    spec = test_spec()
    run_parallel("test_worker", spec["test"], TEST / "logs")
    analyze()


# ---------------------------------------------------------------- 분석
def analyze() -> None:
    import _pdv2_pilot as pilot
    import _review_direction_metrics as direction

    spec = test_spec()
    arms = dict(symmetric="fixed", **test_arms())
    rows = {}  # 시퀀스 -> 조건 -> 방향 -> 지표
    for s in spec["test"]:
        name = s.replace("/", "_")
        assert READ(TEST / "test" / name / "reproduction_check.json")["status"] == "passed"
        with np.load(DIRECTIONS / "directions" / (name + ".npz")) as d:
            label = {k: d[k] for k in ("query_idx", "group")}
        masks = dict(all=None, forward_only=label["group"] == 0, reverse_only=label["group"] == 1)
        rows[s] = {}
        for arm, t in arms.items():
            with np.load(TEST / "test" / name / (t + ".npz")) as z:
                np.testing.assert_array_equal(z["query_idx"], label["query_idx"])
                rows[s][arm] = {g: direction.metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"], m)
                                for g, m in masks.items()}

    def macro(arm: str, group: str, metric: str, picks=None, minus: str = None) -> float:
        """양성이 있는 시퀀스 -> 도시 -> 양성이 있는 도시 평균. picks가 있으면 도시 안에서 재표집한다."""
        cities = []
        for c, seqs in pilot.CITY.items():
            valid = [s for s in seqs if rows[s][arm][group] is not None]
            if not valid:
                continue
            v = np.array([rows[s][arm][group][metric] - (rows[s][minus][group][metric] if minus else 0)
                          for s in valid])
            cities.append(v.mean() if picks is None else v[picks[c]].mean())
        return float(np.mean(cities))

    reference = READ(OLD_TEST / "summary.json")["mean"]["fixed"]
    old_direction = READ(DIRECTIONS / "direction_summary.json")["methods"]["fixed"]["mean_std"]
    for m in METRICS:  # 기존 보고값 재현: 전체와 방향별 균일 감산
        assert abs(macro("symmetric", "all", m) - reference[m]) < 1e-12, m
        for g in ("forward_only", "reverse_only"):
            assert abs(macro("symmetric", g, m) - old_direction[g][m]["mean"]) < 1e-12, (g, m)

    rng = np.random.default_rng(BOOT_SEED)
    result = dict(selection=READ(OUT / "selection.json"), arms=arms, groups={})
    for g in GROUPS:
        sizes = {c: sum(rows[s]["symmetric"][g] is not None for s in seqs) for c, seqs in pilot.CITY.items()}
        draws = [{c: rng.integers(0, n, n) for c, n in sizes.items() if n} for _ in range(BOOTSTRAP)]
        block = dict(percent={a: {m: 100 * macro(a, g, m) for m in METRICS} for a in arms}, minus_symmetric={})
        for a in arms:
            if a == "symmetric":
                continue
            block["minus_symmetric"][a] = {}
            for m in METRICS:
                boot = np.array([macro(a, g, m, p, "symmetric") for p in draws])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                block["minus_symmetric"][a][m] = dict(mean_pp=100 * macro(a, g, m, None, "symmetric"),
                                                      ci95_pp=[100 * lo, 100 * hi])
        result["groups"][g] = block
    result.update(
        aggregation="시험46 중 해당 방향 양성이 있는 시퀀스 -> 도시 -> 양성이 있는 도시 평균. 결정적 조건이라 시드 평균 없음.",
        bootstrap=dict(replicates=BOOTSTRAP, seed=BOOT_SEED, scheme="도시 안 시퀀스 수 유지 짝지은 복원추출"),
        test_used_for_selection=False)
    WRITE(OUT / "result.json", result)
    print(json.dumps({g: {a: round(v["AP"], 2) for a, v in b["percent"].items()}
                      for g, b in result["groups"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("val", "val_worker", "test", "test_worker", "analyze"))
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "val_worker":
        val_worker(a.sequence)
    elif a.action == "test_worker":
        test_worker(a.sequence)
    else:
        globals()[a.action]()
