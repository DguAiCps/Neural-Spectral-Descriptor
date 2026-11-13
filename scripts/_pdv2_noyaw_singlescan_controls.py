#!/usr/bin/env python3
"""학습 없는 두 대조: yaw 정렬 없는 균일 감산, 단일 스캔 NSD.

1. fixed_noyaw: 원고 균일 감산(fixed, 세기 -0.5)과 같되 이웃 복소 스펙트럼을 공통 yaw 좌표계로 옮기지 않는다
   (transport_difference(transport=False)). 세기는 다시 고르지 않는다.
2. single: 시간 정보를 모두 뺀 NSD. 반환 장소는 정제 없는 복소 스펙트럼(원고 identity 승자), 수락 점수는
   같은 점수식에서 중심 쌍 가중치 1, 시간 이웃 가중치 0(원고 표3 no_temporal과 같은 방식).

후보 합집합·융합 가중치·정렬·점수식은 원고 시험 실행과 같다. 시험셋은 보고만 하고 설정 선택에 쓰지 않는다.
산출물은 D 드라이브(컨테이너 /workspace/data)에 둔다.
사용법(컨테이너 ring_baseline 안, 작업 폴더 /workspace/Neural-Spectral-Codec):
  python scripts/_pdv2_noyaw_singlescan_controls.py prepare
  python scripts/_pdv2_noyaw_singlescan_controls.py run       # 검증9·시험46 두 대조
  python scripts/_pdv2_noyaw_singlescan_controls.py analyze
"""
import argparse
import concurrent.futures
import hashlib
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

OUT = Path(
    os.environ.get(
        "NSD_CTRL_OUT", "/workspace/data/nsd_results/pdv2_noyaw_singlescan_20261110"
    )
)
OLD_VAL = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104"
OLD_TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
METRICS = ("AP", "max_F1", "R@100P", "R@1")
BOOTSTRAP = 10000
BOOT_SEED = 20261111
FIXED = dict(kind="fixed", seed=0, checkpoint=None)
NOYAW = dict(kind="noyaw-0.5", seed=0, checkpoint=None)

PREREGISTRATION = dict(
    purpose="Fable 5.2 검토의 권고 2번. (1) 초록·방법이 내세우는 yaw 정렬이 균일 감산에서 실제로 기여하는지, (2) 시간 정보를 모두 뺀 NSD가 단일 스캔 BEVPlace++와 비교해 어디에 있는지 잰다.",
    fixed_noyaw="fixed와 같은 두 층·균일 주의·세기 -0.5·에너지 정규화. 이웃 복소 스펙트럼을 공통 yaw 좌표계로 옮기는 위상 곱만 뺀다(transport=False). 세기 재선택 없음.",
    single="반환 장소 = 원고 identity(정제 없음) 승자. 수락 점수 = 정제 없는 복소 스펙트럼으로 같은 점수식에서 중심 쌍 가중치 1, 시간 이웃 가중치 0. identity 원장의 전체 점수를 같은 코드로 재현해 경로를 확인한다.",
    report="검증9(파이프라인 검증 집계)과 시험46(양성 45시퀀스 -> 도시6 균등) AP·최대F1·R@100P·R@1. 시험 비교: fixed-fixed_noyaw, single-BEVPlace++ 원래 점수(세 시드 평균), NSD(diff 세 시작점 평균)-single, identity-single. 도시 안 시퀀스 수를 유지한 10,000회 짝지은 시퀀스 재표집 백분위 95% 구간. 도시별 값.",
    decision="fixed-fixed_noyaw: 지표별 구간 하한>0이면 'yaw 정렬 우위', 상한<0이면 '무정렬 우위', 그 외 '구분 안 됨'. single-BEVPlace++는 같은 규칙으로 'NSD 단일 스캔 우위/BEVPlace++ 우위/구분 안 됨'.",
    exposure="시험셋은 과거 개발 노출이 있는 탐색적 평가이며 설정 선택에 쓰지 않는다. 결과를 보고 정의·지표·규칙을 바꾸지 않는다.",
)


def READ(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def WRITE(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


def HASH(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare() -> dict:
    path = OUT / "protocol.json"
    if path.exists():
        return READ(path)
    sources = [
        Path(__file__),
        ROOT / "scripts/_pdv2_contrast_pipeline.py",
        ROOT / "scripts/_pdv2_contrast_pipeline_test.py",
        ROOT / "scripts/_pdv2_macro_ablation.py",
        ROOT / "scripts/_pdv2_diffattn_complex_difference.py",
        ROOT / "src/gnn/spectral_difference.py",
        OLD_VAL / "protocol.json",
        OLD_TEST / "protocol.json",
        OLD_TEST / "summary.json",
    ]
    spec = dict(
        created_unix=time.time(),
        preregistration=PREREGISTRATION,
        sources={str(p.relative_to(ROOT)): HASH(p) for p in sources},
    )
    WRITE(path, spec)
    return spec


def patch_transform(pipeline) -> None:
    original = pipeline.transform

    def transform(net, graph, kind: str):
        if kind.startswith("noyaw"):
            return pipeline.e.prior.filtered(graph, float(kind[5:]), transport=False)
        return original(net, graph, kind)

    pipeline.transform = transform


def copy_pool(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".json"):
        target = dst / ("candidate_pool" + suffix)
        if not target.exists():
            shutil.copy2(src / ("candidate_pool" + suffix), target)


def spec_for(role: str) -> dict:
    folder = OUT / f"{role}_eval"
    path = folder / "protocol.json"
    if path.exists():
        return READ(path)
    parent = OLD_VAL if role == "val" else OLD_TEST
    spec = dict(READ(parent / "protocol.json"))
    spec.update(
        created_unix=time.time(),
        cases={"fixed": FIXED, "fixed_noyaw": NOYAW},
        parent_protocol_sha256=HASH(OUT / "protocol.json"),
        scope="yaw 정렬 없는 균일 감산 대조. 설정 선택 없음.",
    )
    WRITE(path, spec)
    return spec


# ---------------------------------------------------------------- yaw 정렬 없는 감산
def noyaw_worker(role: str, sequence: str) -> None:
    name = sequence.replace("/", "_")
    folder = OUT / f"{role}_eval" / role / name
    if role == "val":
        import _pdv2_contrast_pipeline as P

        P.e.configure()
        P.OUT = OUT / "val_eval"
        P.run.OUT = P.OUT
        P.prepare = lambda: spec_for("val")
        patch_transform(P)
        copy_pool(OLD_VAL / "val" / name, folder)
        P.worker("val", sequence)
        old = OLD_VAL / "val" / name
    else:
        import _pdv2_contrast_pipeline_test as T

        T.OUT = OUT / "test_eval"
        T.prepare = lambda: spec_for("test")
        T.pipeline.e.configure()
        patch_transform(T.pipeline)
        copy_pool(OLD_TEST / "test" / name, folder)
        T.worker("test", sequence)
        old = OLD_TEST / "test" / name
    new = np.load(folder / "fixed.npz")
    ref = np.load(old / "fixed.npz")
    for field in ("query_idx", "j_star"):
        np.testing.assert_array_equal(new[field], ref[field])
    error = float(np.abs(new["score"] - ref["score"]).max())
    assert error < 8e-6, error
    WRITE(folder / "reproduction_check.json", dict(status="passed", fixed_score_error=error))


# ---------------------------------------------------------------- 단일 스캔 NSD
def single_worker(role: str, sequence: str) -> None:
    import torch
    import _pdv2_contrast_pipeline as P
    import _pdv2_contrast_pipeline_test as T
    import _pdv2_macro_ablation as M

    P.e.configure()
    if role == "test":
        T.activate()
    name = sequence.replace("/", "_")
    folder = OUT / "single" / role
    target = folder / (name + ".json")
    if target.exists():
        return
    old = (OLD_VAL / "val" if role == "val" else OLD_TEST / "test") / name
    refpath = old / "identity.npz"
    assert P.run.HASH(refpath) == P.run.READ(refpath.with_suffix(".json"))["sha256"]
    ref = P.run.ARR(refpath)
    _, _, _, travel, _, _, _, _ = P.run.inputs(role, sequence)
    graph, _, _ = P.e.prior.graph_inputs(role, sequence)
    q = ref["query_idx"]
    with torch.no_grad():
        z = graph.spectrum
        plan = P.e.prior.base.prior.plan(travel, q, ref["j_star"])
        full = M.batched_score(z, plan)
        error = float(abs(full - ref["score"]).max())
        assert error < 8e-6, error
        value = M.batched_score(z, M.center_only(plan))
    row = {k: ref[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")}
    metric = M.metrics(row, value)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / (name + ".npz"), **row, score=value)
    WRITE(
        target,
        dict(
            metrics=metric,
            identity_reproduction_score_error=error,
            sha256=HASH(folder / (name + ".npz")),
            identity_sha256=HASH(refpath),
        ),
    )
    print("SINGLE", role, sequence, metric, flush=True)


# ---------------------------------------------------------------- 실행
def run_all() -> None:
    prepare()
    val = spec_for("val")["val"]
    test = spec_for("test")["test"]
    jobs = (
        [("noyaw_worker", "val", s) for s in val]
        + [("single_worker", "val", s) for s in val]
        + [("noyaw_worker", "test", s) for s in test]
        + [("single_worker", "test", s) for s in test]
    )
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def call(job):
        action, role, sequence = job
        log = logs / f"{action}_{role}_{sequence.replace('/', '_')}.log"
        with log.open("a") as stream:
            subprocess.run(
                [sys.executable, __file__, action, "--role", role, "--sequence", sequence],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, j) for j in jobs]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            if i % 10 == 0 or i == len(futures):
                print("DONE", i, len(futures), flush=True)
    analyze()


def analyze() -> None:
    import _pdv2_contrast_pipeline as P
    import _pdv2_pilot as pilot

    # 검증
    vspec = spec_for("val")
    vrows = {
        s: READ(OUT / "val_eval/val" / s.replace("/", "_") / "summary.json")["metrics"]
        for s in vspec["val"]
    }
    vold = {
        s: READ(OLD_VAL / "val" / s.replace("/", "_") / "summary.json")["metrics"]
        for s in vspec["val"]
    }
    vsingle = {
        s: READ(OUT / "single/val" / (s.replace("/", "_") + ".json"))["metrics"]
        for s in vspec["val"]
    }
    val_macro = {
        "identity": P.run.base.macro({s: r["identity"] for s, r in vold.items()})["macro"],
        "fixed": P.run.base.macro({s: r["fixed"] for s, r in vrows.items()})["macro"],
        "fixed_noyaw": P.run.base.macro({s: r["fixed_noyaw"] for s, r in vrows.items()})["macro"],
        "single": P.run.base.macro(vsingle)["macro"],
    }
    reference = READ(OLD_VAL / "summary.json")["mean"]
    for m in METRICS:
        assert abs(val_macro["fixed"][m] - reference["fixed"][m]) < 1e-12, m

    # 시험
    tspec = spec_for("test")
    old_summary = READ(OLD_TEST / "summary.json")
    per = {}
    for s in tspec["test"]:
        name = s.replace("/", "_")
        assert READ(OUT / "test_eval/test" / name / "reproduction_check.json")["status"] == "passed"
        new = READ(OUT / "test_eval/test" / name / "summary.json")["metrics"]
        old = READ(OLD_TEST / "test" / name / "summary.json")["metrics"]
        single = READ(OUT / "single/test" / (name + ".json"))["metrics"]
        row = dict(
            fixed=new["fixed"],
            fixed_noyaw=new["fixed_noyaw"],
            identity=old["identity"],
            single=single,
        )
        per[s] = row
    cities = {c: [s for s in seqs if per[s]["fixed"]["n_pos"]] for c, seqs in pilot.CITY.items()}
    assert sum(len(v) for v in cities.values()) == 45
    order = [s for seqs in cities.values() for s in seqs]
    index = [np.array([order.index(s) for s in seqs]) for seqs in cities.values()]

    def seed_mean(label: str, s: str, m: str) -> float:
        return float(
            np.mean([old_summary["per_case"][f"{label}_s{k}"]["sequences"][s][m] for k in (1, 2, 3)])
        )

    vec = {}
    for arm in ("fixed", "fixed_noyaw", "identity", "single"):
        vec[arm] = {m: np.array([per[s][arm][m] for s in order], float) for m in METRICS}
    vec["nsd"] = {m: np.array([seed_mean("diff", s, m) for s in order]) for m in METRICS}
    vec["bev_native"] = {m: np.array([seed_mean("bev_native", s, m) for s in order]) for m in METRICS}

    def macro(v: np.ndarray, picks=None) -> float:
        if picks is None:
            return float(np.mean([v[ix].mean() for ix in index]))
        return float(np.mean([v[ix[p]].mean() for ix, p in zip(index, picks)]))

    for arm, key in (("fixed", "fixed"), ("identity", "identity"), ("nsd", "diff"), ("bev_native", "bev_native")):
        for m in METRICS:
            assert abs(macro(vec[arm][m]) - old_summary["mean"][key][m]) < 1e-9, (arm, m)

    rng = np.random.default_rng(BOOT_SEED)
    picks = [[rng.integers(0, len(ix), len(ix)) for ix in index] for _ in range(BOOTSTRAP)]
    pairs = {
        "fixed-fixed_noyaw": ("fixed", "fixed_noyaw"),
        "single-bev_native": ("single", "bev_native"),
        "nsd-single": ("nsd", "single"),
        "identity-single": ("identity", "single"),
    }
    comparisons = {}
    for label, (a, b) in pairs.items():
        comparisons[label] = {}
        for m in METRICS:
            d = vec[a][m] - vec[b][m]
            boot = np.array([macro(d, p) for p in picks])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            comparisons[label][m] = dict(mean_pp=100 * macro(d), ci95_pp=[100 * lo, 100 * hi])

    def verdict(label: str, pos: str, neg: str) -> dict:
        out = {}
        for m in METRICS:
            lo, hi = comparisons[label][m]["ci95_pp"]
            out[m] = pos if lo > 0 else neg if hi < 0 else "구분 안 됨"
        return out

    city = {
        c: {
            arm: {m: 100 * float(np.mean(vec[arm][m][ix])) for m in METRICS}
            for arm in vec
        }
        for c, ix in zip(cities, index)
    }
    report = dict(
        val_macro_percent={a: {m: 100 * v[m] for m in METRICS} for a, v in val_macro.items()},
        test_macro_percent={a: {m: 100 * macro(vec[a][m]) for m in METRICS} for a in vec},
        comparisons=comparisons,
        verdict={
            "fixed-fixed_noyaw": verdict("fixed-fixed_noyaw", "yaw 정렬 우위", "무정렬 우위"),
            "single-bev_native": verdict("single-bev_native", "NSD 단일 스캔 우위", "BEVPlace++ 우위"),
        },
        city_percent=city,
        aggregation="시험46 중 양성 45시퀀스 -> 도시6 균등. NSD·BEVPlace++는 시퀀스별 세 시드 평균 뒤 집계.",
        bootstrap=dict(replicates=BOOTSTRAP, seed=BOOT_SEED, scheme="도시 안 시퀀스 수 유지 짝지은 복원추출"),
        test_used_for_selection=False,
    )
    WRITE(OUT / "result.json", report)
    print(json.dumps(dict(val=report["val_macro_percent"], test=report["test_macro_percent"],
                          verdict=report["verdict"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "run", "noyaw_worker", "single_worker", "analyze"))
    p.add_argument("--role", choices=("val", "test"))
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "noyaw_worker":
        noyaw_worker(a.role, a.sequence)
    elif a.action == "single_worker":
        single_worker(a.role, a.sequence)
    elif a.action == "run":
        run_all()
    else:
        globals()[a.action]()
