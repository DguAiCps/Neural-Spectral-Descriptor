#!/usr/bin/env python3
"""균일 더하기(이웃 평활) 대조: 원고의 균일 감산(fixed, 세기 -0.5)에서 세기의 부호만 바꾼 학습 없는 시간 전달.

서론 4문단의 주장("Simply aggregating them can reinforce what already makes neighboring places look alike")을
원고와 같은 평가 경로에서 확인한다. 갱신식은 감산과 같은 z_i <- z_i + eta * mean_j(z_j - z_i)이고, eta>0이면
(1-eta) z_i + eta * mean_j z_j 로 이웃과 섞는 평활이 된다. 후보·융합 가중치·정렬·시간 판정·두 층·균일 주의는
균일 감산과 같다. 세기는 검증9에서만 고르고, 시험46에는 고른 세기 하나만 보고한다.

산출물은 C 드라이브 용량 때문에 D 드라이브(컨테이너 /workspace/data)에 둔다.
사용법(컨테이너 ring_baseline 안, 작업 폴더 /workspace/Neural-Spectral-Codec):
  python scripts/_pdv2_uniform_addition_control.py prepare   # 사전 등록(실행 전)
  python scripts/_pdv2_uniform_addition_control.py val       # 검증9: 세기 5개 + fixed 재현
  python scripts/_pdv2_uniform_addition_control.py select    # 사전 등록 규칙으로 세기 고정
  python scripts/_pdv2_uniform_addition_control.py test      # 시험46: 고른 세기 + fixed 재현
  python scripts/_pdv2_uniform_addition_control.py analyze   # 사전 등록한 비교·구간
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
        "NSD_ADD_OUT", "/workspace/data/nsd_results/pdv2_uniform_addition_20261110"
    )
)
VAL = OUT / "val_eval"
TEST = OUT / "test_eval"
OLD_VAL = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104"
OLD_TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
STRENGTHS = (0.0625, 0.125, 0.25, 0.5, 1.0)
METRICS = ("AP", "max_F1", "R@100P", "R@1")
BOOTSTRAP = 10000
BOOT_SEED = 20261110

PREREGISTRATION = dict(
    purpose="서론 4문단의 '이웃을 모으면 비슷한 장소가 더 비슷해진다' 주장을 원고 평가 경로에서 확인한다. 원고 표3·부록 Stage-2 대조의 균일 감산(fixed) 줄과 부호만 다른 대칭 대조다.",
    operation="transport_difference를 그대로 쓰고, 균일 시간 주의·두 층·층별 같은 세기 eta>0을 준다. 한 층은 z_i <- (1-eta) z_i + eta * 이웃 평균(공통 yaw 좌표계, 스캔별 에너지 정규화 포함)이다. 신경망·학습 없음. 원래 색인 후보 합집합, 융합 가중치(aux_half), 두 정렬 점수, 거리 맞춤 시간 판정은 균일 감산과 같다.",
    strengths="eta in {0.0625, 0.125, 0.25, 0.5, 1.0}. 감산 세기 탐색(-0.0625~-0.5, 이후 -1~-4)과 같은 크기 중 평활 범위(0<eta<=1)만 쓴다. eta>1은 자기 가중치가 음수라 평활이 아니어서 제외한다.",
    selection="검증9에서 파이프라인 검증 집계(run.base.macro, 양성 없는 KITTI09 제외 규칙 동일)의 AP가 가장 높은 세기 하나를 고른다. 1e-9 안의 동률이면 작은 세기. 시험셋은 선택에 쓰지 않는다.",
    test="고른 세기 하나만 시험46에서 평가한다. fixed를 함께 다시 계산해 기존 원장과 승자 일치·점수 오차 8e-6 미만을 확인한다. identity는 기존 시험 원장을 쓴다.",
    report="시험 6도시 매크로(양성 있는 45시퀀스 -> 도시6 균등) AP·최대F1·R@100P·R@1. 비교: 더하기-감산(주), 더하기-갱신없음(보조), 감산-갱신없음(참고). 도시 안 시퀀스 수를 유지한 10,000회 짝지은 시퀀스 재표집 백분위 95% 구간. 도시별 값. 검증 다섯 세기 모두 보고한다.",
    decision="AP·최대F1·R@100P 각각에서 더하기-감산 구간 상한<0이면 '감산 우위', 하한>0이면 '더하기 우위', 그 외 '구분 안 됨'. 더하기-갱신없음도 같은 방식으로 '갱신없음보다 낮음/높음/구분 안 됨'. 서론 주장의 지지: 세 지표 모두 '감산 우위'면 지지, 그중 AP에서 더하기-갱신없음 상한<0이면 '모으면 오히려 나빠진다'까지 지지. 세 지표 중 하나라도 '더하기 우위'면 주장과 반대. 그 외는 부분적.",
    exposure="시험셋은 과거 개발 노출이 있는 탐색적 평가이며 이번 실행은 설정 선택에 쓰지 않는다. 결과를 보고 세기·지표·규칙을 바꾸지 않는다.",
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


def tag(eta: float) -> str:
    return "add_" + f"{eta:g}".replace(".", "p")


def case(eta: float) -> dict:
    return dict(kind=f"add{eta:+g}", seed=0, checkpoint=None)


FIXED = dict(kind="fixed", seed=0, checkpoint=None)


def prepare() -> dict:
    """사전 등록과 출처 해시를 실행 전에 고정한다."""
    path = OUT / "protocol.json"
    if path.exists():
        return READ(path)
    sources = [
        Path(__file__),
        ROOT / "scripts/_pdv2_contrast_pipeline.py",
        ROOT / "scripts/_pdv2_contrast_pipeline_test.py",
        ROOT / "scripts/_pdv2_diffattn_complex_difference.py",
        ROOT / "src/gnn/spectral_difference.py",
        OLD_VAL / "protocol.json",
        OLD_TEST / "protocol.json",
        OLD_TEST / "summary.json",
    ]
    spec = dict(
        created_unix=time.time(),
        preregistration=PREREGISTRATION,
        strengths=list(STRENGTHS),
        sources={str(p.relative_to(ROOT)): HASH(p) for p in sources},
    )
    WRITE(path, spec)
    return spec


def patch_transform(pipeline) -> None:
    """파이프라인 변환에 학습 없는 양의 세기(평활)만 추가한다."""
    original = pipeline.transform

    def transform(net, graph, kind: str):
        if kind.startswith("add"):
            eta = float(kind[3:])
            assert 0 < eta <= 1, eta
            return pipeline.e.prior.filtered(graph, eta)
        return original(net, graph, kind)

    pipeline.transform = transform


def copy_pool(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".json"):
        target = dst / ("candidate_pool" + suffix)
        if not target.exists():
            shutil.copy2(src / ("candidate_pool" + suffix), target)


# ---------------------------------------------------------------- 검증
def val_spec() -> dict:
    path = VAL / "protocol.json"
    if path.exists():
        return READ(path)
    spec = dict(READ(OLD_VAL / "protocol.json"))
    cases = {"fixed": FIXED}
    cases.update({tag(eta): case(eta) for eta in STRENGTHS})
    spec.update(
        created_unix=time.time(),
        cases=cases,
        parent_protocol_sha256=HASH(OUT / "protocol.json"),
        scope="균일 더하기 대조의 검증 세기 선택. 시험 미사용.",
    )
    WRITE(path, spec)
    return spec


def val_worker(sequence: str) -> None:
    import _pdv2_contrast_pipeline as P

    P.e.configure()
    spec = val_spec()
    P.OUT = VAL
    P.run.OUT = VAL
    P.prepare = val_spec
    patch_transform(P)
    name = sequence.replace("/", "_")
    copy_pool(OLD_VAL / "val" / name, VAL / "val" / name)
    P.worker("val", sequence)
    assert sequence in spec["val"]
    rec = READ(VAL / "val" / name / "summary.json")
    assert "fixed" in rec["reproduction"], rec["reproduction"].keys()


def run_parallel(action: str, sequences: list, logs: Path) -> None:
    logs.mkdir(parents=True, exist_ok=True)

    def call(sequence):
        with (logs / (sequence.replace("/", "_") + ".log")).open("a") as stream:
            subprocess.run(
                [sys.executable, __file__, action, "--sequence", sequence],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, s) for s in sequences]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            print(action.upper(), "DONE", i, len(futures), flush=True)


def val() -> None:
    prepare()
    spec = val_spec()
    run_parallel("val_worker", spec["val"], VAL / "logs")
    select()


def select() -> None:
    """검증 집계로 세기 하나를 고정한다(시험 미사용)."""
    import _pdv2_contrast_pipeline as P

    spec = val_spec()
    rows = {s: READ(VAL / "val" / s.replace("/", "_") / "summary.json") for s in spec["val"]}
    old = {
        s: READ(OLD_VAL / "val" / s.replace("/", "_") / "summary.json") for s in spec["val"]
    }
    table = {}
    for t in spec["cases"]:
        table[t] = P.run.base.macro({s: r["metrics"][t] for s, r in rows.items()})["macro"]
    table["identity"] = P.run.base.macro(
        {s: r["metrics"]["identity"] for s, r in old.items()}
    )["macro"]
    reference = READ(OLD_VAL / "summary.json")["mean"]
    for m in METRICS:  # 기존 검증 보고값 재현
        assert abs(table["fixed"][m] - reference["fixed"][m]) < 1e-12, m
        assert abs(table["identity"][m] - reference["identity"][m]) < 1e-12, m
    ranked = sorted(STRENGTHS, key=lambda eta: (-round(table[tag(eta)]["AP"], 9), eta))
    chosen = ranked[0]
    WRITE(
        OUT / "selection.json",
        dict(
            selected_eta=chosen,
            selected_tag=tag(chosen),
            rule=PREREGISTRATION["selection"],
            val_macro_percent={
                t: {m: 100 * v[m] for m in METRICS} for t, v in table.items()
            },
            test_used=False,
        ),
    )
    print("SELECTED", chosen, {t: round(100 * v["AP"], 3) for t, v in table.items()}, flush=True)


# ---------------------------------------------------------------- 시험
def test_spec() -> dict:
    path = TEST / "protocol.json"
    if path.exists():
        return READ(path)
    chosen = READ(OUT / "selection.json")["selected_eta"]
    spec = dict(READ(OLD_TEST / "protocol.json"))
    spec.update(
        created_unix=time.time(),
        cases={"fixed": FIXED, tag(chosen): case(chosen)},
        parent_protocol_sha256=HASH(OUT / "protocol.json"),
        selection_sha256=HASH(OUT / "selection.json"),
        test_policy="검증에서 고정한 균일 더하기 세기 하나의 시험 보고. 설정 선택 없음.",
    )
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
    WRITE(
        folder / "reproduction_check.json",
        dict(status="passed", fixed=dict(winners_exact=True, score_error=error)),
    )


def test() -> None:
    assert (OUT / "selection.json").exists()
    spec = test_spec()
    run_parallel("test_worker", spec["test"], TEST / "logs")
    analyze()


# ---------------------------------------------------------------- 분석
def analyze() -> None:
    import _pdv2_pilot as pilot

    spec = test_spec()
    chosen = READ(OUT / "selection.json")["selected_tag"]
    per = {}
    for s in spec["test"]:
        name = s.replace("/", "_")
        assert READ(TEST / "test" / name / "reproduction_check.json")["status"] == "passed"
        new = READ(TEST / "test" / name / "summary.json")["metrics"]
        old = READ(OLD_TEST / "test" / name / "summary.json")["metrics"]
        per[s] = dict(add=new[chosen], fixed=new["fixed"], identity=old["identity"])
    cities = {
        c: [s for s in seqs if per[s]["fixed"]["n_pos"]] for c, seqs in pilot.CITY.items()
    }
    assert sum(len(v) for v in cities.values()) == 45
    order = [s for seqs in cities.values() for s in seqs]
    index = [np.array([order.index(s) for s in seqs]) for seqs in cities.values()]
    vec = {
        arm: {m: np.array([per[s][arm][m] for s in order], float) for m in METRICS}
        for arm in ("add", "fixed", "identity")
    }

    def macro(v: np.ndarray, picks=None) -> float:
        if picks is None:
            return float(np.mean([v[ix].mean() for ix in index]))
        return float(np.mean([v[ix[p]].mean() for ix, p in zip(index, picks)]))

    reference = READ(OLD_TEST / "summary.json")["mean"]
    for arm in ("fixed", "identity"):
        for m in METRICS:
            assert abs(macro(vec[arm][m]) - reference[arm][m]) < 1e-12, (arm, m)
    rng = np.random.default_rng(BOOT_SEED)
    picks = [[rng.integers(0, len(ix), len(ix)) for ix in index] for _ in range(BOOTSTRAP)]
    pairs = {"add-fixed": ("add", "fixed"), "add-identity": ("add", "identity"),
             "fixed-identity": ("fixed", "identity")}
    comparisons = {}
    for label, (a, b) in pairs.items():
        comparisons[label] = {}
        for m in METRICS:
            d = vec[a][m] - vec[b][m]
            boot = np.array([macro(d, p) for p in picks])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            comparisons[label][m] = dict(
                mean_pp=100 * macro(d), ci95_pp=[100 * lo, 100 * hi]
            )

    def verdict(label: str, m: str, below: str, above: str) -> str:
        lo, hi = comparisons[label][m]["ci95_pp"]
        return below if hi < 0 else above if lo > 0 else "구분 안 됨"

    primary = ("AP", "max_F1", "R@100P")
    versus_fixed = {m: verdict("add-fixed", m, "감산 우위", "더하기 우위") for m in primary}
    versus_identity = {
        m: verdict("add-identity", m, "갱신없음보다 낮음", "갱신없음보다 높음") for m in primary
    }
    if any(v == "더하기 우위" for v in versus_fixed.values()):
        claim = "주장과 반대"
    elif all(v == "감산 우위" for v in versus_fixed.values()):
        claim = (
            "지지(모으면 오히려 나빠짐까지)"
            if versus_identity["AP"] == "갱신없음보다 낮음"
            else "지지(감산이 더하기보다 나음)"
        )
    else:
        claim = "부분적"
    city = {
        c: {
            arm: {m: 100 * float(np.mean([per[s][arm][m] for s in seqs])) for m in METRICS}
            for arm in ("add", "fixed", "identity")
        }
        for c, seqs in cities.items()
    }
    report = dict(
        selected=READ(OUT / "selection.json"),
        test_macro_percent={
            arm: {m: 100 * macro(vec[arm][m]) for m in METRICS}
            for arm in ("add", "fixed", "identity")
        },
        comparisons=comparisons,
        verdict=dict(versus_fixed=versus_fixed, versus_identity=versus_identity, claim=claim),
        city_percent=city,
        aggregation="시험46 중 양성 45시퀀스 -> 도시6 균등. 결정적 조건이라 시드 평균 없음.",
        bootstrap=dict(replicates=BOOTSTRAP, seed=BOOT_SEED, scheme="도시 안 시퀀스 수 유지 짝지은 복원추출"),
        test_used_for_selection=False,
    )
    WRITE(OUT / "result.json", report)
    print(json.dumps(dict(test=report["test_macro_percent"], verdict=report["verdict"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "action",
        choices=("prepare", "val", "val_worker", "select", "test", "test_worker", "analyze"),
    )
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "val_worker":
        val_worker(a.sequence)
    elif a.action == "test_worker":
        test_worker(a.sequence)
    else:
        globals()[a.action]()
