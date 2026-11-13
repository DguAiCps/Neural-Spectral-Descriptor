#!/usr/bin/env python3
"""같은 오수락 수에서 NSD와 BEVPlace++의 정답 수락을 비교한다(시험 점수에서 고른 단일 전역 임계값).

방법·시작점마다 시험 46시퀀스의 적격 질의를 합쳐 임계값 하나를 훑고, 합산 오수락 ≤ k(보조: 합산 정밀도 ≥ p)를
만족하는 임계값 가운데 정답 수락(TP)이 가장 많은 것을 고른다. 임계값은 시험 점수에서 고르며 두 방법에 같은 규칙을
쓴다. 검증 고정 임계값이 아니다. 판정·집계는 scripts/_pr_eval_lib.py(classify_queries, pr_sweep,
operating_point)와 scripts/_pdv2_pilot.py(CITY, TEST)를 그대로 쓰고, 도시 평균·평균±SD는 검증 고정 임계값
분석(scripts/_review_v23_fixed_threshold_20261107.py)의 city_mean·mean_sd를 가져다 쓴다.

protocol: 실행 전에 규칙을 OUT/protocol.json에 고정한다(이미 있으면 덮어쓰지 않는다).
run: protocol 해시 확인 → 원장 276개 해시 대조 → 정산 검증(기존 고정 τ의 수치 재현) → 기준별 선택과 독립 재계산
     → 짝지은 재표집 구간 → OUT/result.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import numpy as np

from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import MATCH_RADIUS_M, classify_queries, operating_point, pr_sweep
from _review_v23_fixed_threshold_20261107 import city_mean, mean_sd

REPO = Path(__file__).resolve().parents[1]
OUT = Path("/mnt/d/NSD_datasets/nsd_results/pdv2_matched_false_accepts_20261112")
FIXED = REPO / "results/pdv2_review_closure_20261107/threshold"
MAIN = REPO / "results/pdv2_diffattn_contrast_test_20261104"
SEEDS = (1, 2, 3)
METHODS = ("nsd", "bevplace")
SCORE = {"nsd": "score", "bevplace": "score_negl2"}
BUDGETS = (0, 5, 10, 30, 100)
FLOORS = (0.999, 0.995, 0.99)
CRITERIA = tuple([(f"FA<={k}", "budget", k) for k in BUDGETS] +
                 [(f"P>={100 * p:g}%", "precision", p) for p in FLOORS])
FIXED_POINTS = ("P99", "P100", "F1")
B, BOOT_SEED, CHUNK = 10000, 20261104, 50
SEQS = [f"{s}/{q}" for s, q in TEST]
# 지시문에 적힌 임시 계산값(검증 안 됨, 세 시작점 평균 TP). 일치 여부만 본다.
PROVISIONAL_TP = {"nsd": {0: 7107, 5: 10105, 10: 14954, 30: 21478, 100: 24547},
                  "bevplace": {0: 6049, 5: 8906, 10: 10931, 30: 15919, 100: 20335}}
BUILT: dict = {}     # 재표집 작업자가 fork로 물려받는다: (method, seed) -> 곡선
WEIGHTS = None       # 재표집 가중치 (B, 46)
CITY_COLS: list = []


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO))


def ledger_path(method: str, seed: int, sensor: str, seq: str) -> str:
    name = f"{sensor}_{seq}"
    if method == "nsd":
        return f"results/pdv2_diffattn_contrast_test_20261104/test/{name}/diff_s{seed}.npz"
    return f"results/pdv2_review_response_20261104/bev/test/s{seed}/{name}.npz"


# ------------------------------------------------------------------ protocol

def protocol() -> None:
    target = OUT / "protocol.json"
    if target.exists():
        print("PROTOCOL_EXISTS", target, digest(target)); return
    spec = dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        purpose="방법·시작점마다 시험 46시퀀스 합산 sweep에서 같은 합산 오수락 예산(보조: 같은 합산 정밀도 하한)을 만족하는 "
                "임계값 가운데 정답 수락이 가장 많은 것을 골라 NSD와 BEVPlace++를 같은 오수락 수에서 비교한다.",
        threshold_source="시험 점수. 두 방법에 같은 규칙으로 시험 합산 sweep에서 고른다. 검증 고정 임계값이 아니며 배치 가능한 "
                         "운영점이 아니다(같은 오수락 수에서의 사후 비교).",
        inputs=dict(
            ledgers="results/pdv2_review_closure_20261107/threshold/summary.json의 test_ledgers_sha256 276개(46시퀀스×2방법×3시작점). "
                    "실행 시 모든 npz sha256을 이 값과 대조한다.",
            nsd="results/pdv2_diffattn_contrast_test_20261104/test/<sensor_seq>/diff_s{1,2,3}.npz, 점수 키 score",
            bevplace="results/pdv2_review_response_20261104/bev/test/s{1,2,3}/<sensor_seq>.npz, 점수 키 score_negl2(−L2 서술자 거리)",
            common_keys=["top1_geo_dist", "min_past_gt_dist", "query_idx"],
            pairing="시퀀스마다 6개 원장의 query_idx·min_past_gt_dist가 비트 단위로 같아야 한다.",
        ),
        accounting=dict(
            library="scripts/_pr_eval_lib.py classify_queries/pr_sweep/operating_point 그대로, 도시 대응 scripts/_pdv2_pilot.py CITY·TEST",
            query_class="양성 d_min≤5 m, 음성 d_min>25 m, 5<d_min≤25 m 질의는 제외",
            acceptance="유한 점수 ≥ τ 이면 수락(같은 점수는 함께 수락), NaN 점수는 수락 안 함",
            TP="양성 질의, 수락, 반환 1등 ≤5 m",
            FP="수락했는데 반환 1등 >5 m (양성·음성 질의 모두) = 오수락",
            recall_denominator="양성 질의 수 전체(τ 무관)",
            empty_accept="수락 0개면 정밀도 1.0, 재현율 0(+inf 표지점)",
        ),
        selection=dict(
            pool="방법·시작점마다 46시퀀스(KITTI07 포함) 적격 질의를 합친 sweep 하나. 후보 τ는 합산 적격 질의의 고유 유한 점수와 +inf 표지점(pr_sweep과 같은 후보).",
            budgets=list(BUDGETS),
            budget_rule="합산 FP ≤ k 인 τ 가운데 합산 TP 최대. 같은 TP면 가장 높은 τ(가장 적은 수락).",
            precision_floors=list(FLOORS),
            precision_rule="합산 정밀도 TP/(TP+FP) ≥ p 인 τ 가운데 합산 TP 최대. 같은 TP면 가장 높은 τ. (보조)",
            per="방법×시작점마다 독립. 시작점 평균 τ를 만들지 않는다.",
            infeasible="조건을 만족하는 점이 +inf 표지점뿐이면 τ=+inf(수락 0, TP 0)로 보고",
        ),
        metrics=dict(
            pooled="46시퀀스 합산 TP, FP(양성 질의 오반환·음성 질의 분리), 수락 수, 합산 정밀도, 합산 재현율 TP/전체양성",
            city="표2 집계: 양성 있는 시퀀스 재현율 → 도시 평균 → 6도시 동일가중 평균(_review_v23_fixed_threshold_20261107.city_mean), 도시별 재현율",
            seeds="시작점 1·2·3의 평균과 SD(ddof=1, mean_sd)",
            difference="NSD 시작점 평균 − BEVPlace++ 시작점 평균(합산 재현율, 도시 재현율, 도시별 재현율, TP)",
        ),
        intervals=dict(
            procedure="원고 짝지은 구간(scripts/_pdv2_contrast_pipeline_report.py)과 같은 재표집: 난수 np.random.default_rng(20261104)를 한 번 만들고 "
                      "CITY 순서로 도시마다 양성 있는 시퀀스(같은 순서) 목록에서 rng.integers(0, n, size=(10000, n))로 시퀀스 수를 유지하며 복원 추출. "
                      "두 방법·세 시작점·모든 기준이 같은 표본을 쓴다(짝지음). 표본마다 시작점별로 계산해 세 시작점을 평균한 뒤 NSD−BEVPlace++ 차이를 만들고 "
                      "np.quantile(차이, [0.025, 0.975])를 95% 구간으로 보고한다(백분율 점).",
            replicates=B,
            seed=BOOT_SEED,
            primary="reselect: 표본마다 그 표본의 합산 sweep(뽑힌 횟수만큼 시퀀스 질의를 중복)에서 같은 규칙으로 τ를 다시 고른다. "
                    "이유: 비교 대상이 '같은 오수락 수를 지키는 최선의 임계값에서의 재현율'이므로 임계값 선택이 추정량의 일부이고, "
                    "τ를 고정하면 재표집 표본에서 오수락 수가 k를 벗어나 같은 오수락 수 비교가 아니게 된다.",
            secondary="fixed_tau: 전체 시험에서 고른 τ를 표본에서도 고정(원고의 다른 구간처럼 고정 운영점의 시퀀스별 값을 재표집). "
                      "τ 선택 변동을 넣지 않으며, 표본의 오수락 수는 k와 다를 수 있다(표본 오수락 분위수를 함께 보고).",
            zero_positive_sequences="양성이 없는 시퀀스(KITTI07)는 재현율 재표집 대상이 아니다. reselect에서는 오수락 예산 계산에 쓰이도록 "
                                    "모든 표본에 한 번씩 고정 포함하고, fixed_tau 표본 오수락 계산에도 같은 가중치를 쓴다. 재현율에는 영향이 없다(양성 0).",
            pooled_recall_in_replicate="표본 합산 TP / 표본 합산 양성 수(뽑힌 횟수 가중)",
            city_recall_in_replicate="도시마다 뽑힌 시퀀스 재현율의 평균 → 6도시 평균",
            scope="고정한 모델(시작점 3개)과 표집된 도시에 조건부인 구간. 학습 변동·도시 밖 일반화·다중비교를 다루지 않는다.",
        ),
        reconciliation=dict(
            what="freeze.json의 고정 τ(P99, P100, F1; 방법×시작점)를 이 스크립트의 판정 경로로 같은 시험 원장에 적용해 기존 summary.json의 "
                 "시작점별 합산 TP·FP·FP(양성/음성)·수락 수·합산 정밀도·합산 재현율·도시 재현율(6도시 평균과 도시별)·시퀀스별 TP·FP, "
                 "시작점 평균·SD와 false_accepts.json의 질의 종류별·도시별·시퀀스별 오수락 수를 재현하는지 대조한다. "
                 "정수는 정확히, 실수는 절대 차 ≤1e-12.",
            curve="같은 τ에서 합산 곡선(시퀀스별 누적표)의 TP·FP가 operating_point 합과 같은지 확인한다.",
            gate="정산이 하나라도 어긋나면 분석을 하지 않고 멈춘다.",
        ),
        verification=dict(
            independent="선택마다 곡선 코드와 독립인 계산(TP·FP 점수를 따로 정렬하고 고유 점수마다 searchsorted로 센 sweep)에서 같은 규칙으로 고른 τ·TP·FP와 대조. "
                        "예산 기준은 닫힌 꼴(오수락 점수 (k+1)번째보다 큰 TP 점수 개수)과도 대조. 선택한 τ에서 operating_point 합과 곡선 값이 같아야 한다.",
            bootstrap_identity="모든 시퀀스 가중치 1인 표본에서 재표집 코드가 전체 시험 선택·재현율을 그대로 내는지 확인",
            bootstrap_draws="같은 표본 가중치로 results/pdv2_diffattn_contrast_test_20261104/summary.json의 시퀀스별 지표 차이(diff−bev_native)를 "
                            "재표집해 integrated_report.json의 원고 구간(NSD minus BEVPlace++, 4지표)을 1e-9 안에서 재현하는지 확인",
        ),
        provisional_reference=dict(note="지시문의 임시 계산(검증 안 됨, 시작점 평균 TP). 결과와 일치 여부만 보고한다(차 ≤0.5면 일치).",
                                   TP=PROVISIONAL_TP),
        fixed_rules_note="예산 목록·정밀도 하한·선택 규칙·재표집 절차는 이 파일로 고정하며 시험 결과를 보고 바꾸지 않는다. 한 번 실행한다.",
        disclosure="이 분석은 이미 공개된 시험 결과에 대한 사후 분석이며 맹검이 아니다. 지시문에 임시 계산값(오수락 예산별 시작점 평균 TP)이 이미 적혀 있었다.",
        outputs=dict(dir=str(OUT), protocol="protocol.json", result="result.json", log="run_<UTC>.log"),
        run_command="cd <리포> && OPENBLAS_NUM_THREADS=2 python3 scripts/_paper_matched_false_accepts.py run 2>&1 | tee <OUT>/run_<UTC>.log "
                    "(호스트 CPU, 방법×시작점 6개 작업자 병렬)",
        code_sha256={rel(Path(__file__)): digest(Path(__file__)),
                     "scripts/_pr_eval_lib.py": digest(REPO / "scripts/_pr_eval_lib.py"),
                     "scripts/_pdv2_pilot.py": digest(REPO / "scripts/_pdv2_pilot.py"),
                     "scripts/_review_v23_fixed_threshold_20261107.py": digest(REPO / "scripts/_review_v23_fixed_threshold_20261107.py"),
                     "scripts/_pdv2_contrast_pipeline_report.py": digest(REPO / "scripts/_pdv2_contrast_pipeline_report.py")},
        reference_sha256={**{rel(FIXED / n): digest(FIXED / n) for n in
                             ("summary.json", "freeze.json", "false_accepts.json", "verification.json", "protocol.json", "done.json")},
                          **{rel(MAIN / n): digest(MAIN / n) for n in ("summary.json", "integrated_report.json")}},
        environment=dict(python=platform.python_version(), numpy=np.__version__, host="WSL 호스트 파이썬(CPU)"),
    )
    write(target, spec)
    print(target, digest(target))


# ------------------------------------------------------------------- loading

def load(expected: dict) -> dict:
    """원장 276개의 sha256을 기존 summary.json 값과 대조하고 점수·반환 거리·라벨을 읽는다."""
    wanted = {ledger_path(m, s, sensor, q) for m in METHODS for s in SEEDS for sensor, q in TEST}
    assert wanted == set(expected), "원장 목록이 summary.json과 다르다"
    data = {(m, s): {} for m in METHODS for s in SEEDS}
    for sensor, q in TEST:
        key, ref = f"{sensor}/{q}", None
        for m in METHODS:
            for s in SEEDS:
                path = ledger_path(m, s, sensor, q)
                assert digest(REPO / path) == expected[path], path
                with np.load(REPO / path) as z:
                    row = dict(score=z[SCORE[m]], dist=z["top1_geo_dist"], query_idx=z["query_idx"],
                               mpgd=z["min_past_gt_dist"])
                assert row["score"].dtype == np.float64 and row["dist"].dtype == np.float64
                if ref is None:
                    ref = row
                np.testing.assert_array_equal(row["query_idx"], ref["query_idx"])
                np.testing.assert_array_equal(row["mpgd"], ref["mpgd"])
                row["qc"] = classify_queries(row["mpgd"])
                data[(m, s)][key] = row
    for rows in data.values():
        assert sum(len(r["query_idx"]) for r in rows.values()) == 195726
    return data


# ------------------------------------------------------------------- curves

def build(rows: dict, seqs: list) -> dict:
    """합산 sweep과 고유 임계값마다 시퀀스별 누적 TP·FP 표. pr_sweep과 같은 순서·동점 규칙이며 그 결과와 대조한다."""
    score = np.concatenate([rows[k]["score"] for k in seqs])
    dist = np.concatenate([rows[k]["dist"] for k in seqs])
    sid = np.concatenate([np.full(len(rows[k]["score"]), i) for i, k in enumerate(seqs)])
    qc = classify_queries(np.concatenate([rows[k]["mpgd"] for k in seqs]))
    sweep = pr_sweep(score, dist, qc)
    pos, neg = qc["positive"], qc["negative"]
    correct = np.isfinite(dist) & (dist <= MATCH_RADIUS_M)
    fin = (pos | neg) & np.isfinite(score)
    order = np.flatnonzero(fin)[np.argsort(-score[fin], kind="stable")]
    s_sorted = score[order]
    last = (np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True]) if len(order)
            else np.empty(0, dtype=np.int64))
    is_tp, is_fp, sid_sorted = (pos & correct)[order], (neg | (pos & ~correct))[order], sid[order]
    tp_seq = np.zeros((len(seqs), len(last) + 1))
    fp_seq = np.zeros((len(seqs), len(last) + 1))
    for i in range(len(seqs)):
        mine = sid_sorted == i
        tp_seq[i, 1:] = np.cumsum(is_tp & mine)[last]
        fp_seq[i, 1:] = np.cumsum(is_fp & mine)[last]
    thresholds = np.r_[np.inf, s_sorted[last]]
    tp, fp = tp_seq.sum(0), fp_seq.sum(0)
    n_pos = qc["n_pos"]
    acc = tp + fp
    assert np.array_equal(thresholds, sweep["thresholds"])
    assert np.array_equal(tp / max(n_pos, 1), sweep["recall"])
    assert np.array_equal(np.where(acc > 0, tp / np.maximum(acc, 1), 1.0), sweep["precision"])
    npos_seq = np.array([rows[k]["qc"]["n_pos"] for k in seqs], dtype=np.float64)
    return dict(thresholds=thresholds, tp_seq=tp_seq, fp_seq=fp_seq, tp=tp, fp=fp, n_pos=n_pos, npos_seq=npos_seq)


def choose(tp: np.ndarray, fp: np.ndarray, kind: str, value: float) -> np.ndarray:
    """곡선(마지막 축이 τ 내림차순)에서 조건을 만족하는 점 가운데 TP 최대, 같은 TP면 가장 앞(가장 높은 τ)의 인덱스."""
    if kind == "budget":
        ok = fp <= value
    else:
        acc = tp + fp
        ok = np.where(acc > 0, tp / np.maximum(acc, 1), 1.0) >= value
    best = np.where(ok, tp, -1).max(axis=-1, keepdims=True)
    return np.argmax(ok & (tp == best), axis=-1)


def index_at(thresholds: np.ndarray, tau: float) -> int:
    """고정 τ의 수락 집합에 해당하는 곡선 점(τ 이상인 고유 임계값의 마지막)."""
    return int(np.sum(thresholds >= tau)) - 1


def independent(rows: dict, seqs: list, kind: str, value: float) -> dict:
    """곡선 코드와 독립인 선택: TP·FP 점수를 따로 정렬해 고유 점수마다 searchsorted로 센다."""
    score = np.concatenate([rows[k]["score"] for k in seqs])
    dist = np.concatenate([rows[k]["dist"] for k in seqs])
    pos = np.concatenate([rows[k]["qc"]["positive"] for k in seqs])
    neg = np.concatenate([rows[k]["qc"]["negative"] for k in seqs])
    fin = (pos | neg) & np.isfinite(score)
    good = np.isfinite(dist) & (dist <= MATCH_RADIUS_M)
    tp_scores = np.sort(score[fin & pos & good])
    fp_scores = np.sort(score[fin & ~(pos & good)])
    cand = np.r_[np.inf, np.unique(score[fin])[::-1]]
    tp = (len(tp_scores) - np.searchsorted(tp_scores, cand, side="left")).astype(np.float64)
    fp = (len(fp_scores) - np.searchsorted(fp_scores, cand, side="left")).astype(np.float64)
    i = int(choose(tp, fp, kind, value))
    out = dict(tau=float(cand[i]), TP=int(tp[i]), FP=int(fp[i]))
    if kind == "budget":   # 닫힌 꼴: (k+1)번째로 높은 오수락 점수보다 큰 TP 점수 개수
        k = int(value)
        bound = fp_scores[::-1][k] if len(fp_scores) > k else -np.inf
        out["closed_form_TP"] = int(np.sum(tp_scores > bound))
    return out


def at_tau(rows: dict, tau: float) -> dict:
    """고정 τ에서 시퀀스별 operating_point를 합산하고 도시 재현율을 만든다."""
    per = {}
    for k, r in rows.items():
        qc = r["qc"]
        p = operating_point(r["score"], r["dist"], qc, tau)
        accepted = np.isfinite(r["score"]) & (r["score"] >= tau)
        wrong = ~(np.isfinite(r["dist"]) & (r["dist"] <= MATCH_RADIUS_M))
        per[k] = dict(TP=p["TP"], FP=p["FP"], n_pos=qc["n_pos"],
                      FP_positive_query=int((accepted & wrong & qc["positive"]).sum()),
                      FP_negative_query=int((accepted & qc["negative"]).sum()))
        assert per[k]["FP"] == per[k]["FP_positive_query"] + per[k]["FP_negative_query"]
    tot = {f: sum(c[f] for c in per.values()) for f in ("TP", "FP", "n_pos", "FP_positive_query", "FP_negative_query")}
    acc = tot["TP"] + tot["FP"]
    macro, cities = city_mean({k: c["TP"] / c["n_pos"] for k, c in per.items() if c["n_pos"]})
    return dict(tau=float(tau), tau_hex=float(tau).hex(), **tot, accepted=acc,
                precision=tot["TP"] / acc if acc else 1.0, recall=tot["TP"] / tot["n_pos"],
                city_macro_recall=macro, city_recall=cities,
                city_FP={c: sum(per[s]["FP"] for s in names) for c, names in CITY.items()},
                sequence={k: dict(TP=c["TP"], FP=c["FP"]) for k, c in per.items()})


# ------------------------------------------------------------ reconciliation

def reconcile(data: dict, built: dict) -> dict:
    """기존 검증 고정 τ를 이 스크립트의 판정 경로로 적용해 기존 summary.json·false_accepts.json을 재현하는지 본다."""
    summary, freeze, fa = (read(FIXED / n) for n in ("summary.json", "freeze.json", "false_accepts.json"))
    mismatches, n_checks, worst = [], 0, 0.0

    def same(label, mine, ref, exact):
        nonlocal n_checks, worst
        n_checks += 1
        if exact:
            ok = int(mine) == int(ref) and float(mine) == float(ref)
        else:
            d = abs(float(mine) - float(ref)); worst = max(worst, d); ok = d <= 1e-12
        if not ok:
            mismatches.append(dict(check=label, mine=mine, reference=ref))

    for m in METHODS:
        for point in FIXED_POINTS:
            mine_seeds = {}
            for s in SEEDS:
                frozen = freeze["thresholds"][m][str(s)][point]
                tau = frozen["tau"]
                assert float(tau).hex() == frozen["tau_hex"]
                mine = at_tau(data[(m, s)], tau)
                mine_seeds[s] = mine
                ref = summary["methods"][m][point]["per_seed"][str(s)]
                tag = f"{m}/s{s}/{point}"
                same(tag + "/tau", mine["tau"], ref["tau"], False)
                b, i = built[(m, s)], index_at(built[(m, s)]["thresholds"], tau)
                same(tag + "/curve_TP", b["tp"][i], mine["TP"], True)
                same(tag + "/curve_FP", b["fp"][i], mine["FP"], True)
                for f in ("TP", "FP", "FP_positive_query", "FP_negative_query", "accepted", "n_pos"):
                    same(f"{tag}/{f}", mine[f], ref["pooled"][f], True)
                for f in ("precision", "recall"):
                    same(f"{tag}/{f}", mine[f], ref["pooled"][f], False)
                same(tag + "/city_macro_recall", mine["city_macro_recall"], ref["city_macro_recall"], False)
                for c in CITY:
                    same(f"{tag}/city_recall/{c}", mine["city_recall"][c], ref["city_recall"][c], False)
                    same(f"{tag}/city_FP/{c}", mine["city_FP"][c], ref["city"][c]["FP"], True)
                for k in SEQS:
                    for f in ("TP", "FP"):
                        same(f"{tag}/seq/{k}/{f}", mine["sequence"][k][f], ref["sequence"][k][f], True)
                fref = fa["per_seed"][m][str(s)][point]
                same(tag + "/fa/positive_wrong_place", mine["FP_positive_query"], fref["query_kind"].get("positive_wrong_place", 0), True)
                same(tag + "/fa/negative", mine["FP_negative_query"], fref["query_kind"].get("negative", 0), True)
                for c in CITY:
                    same(f"{tag}/fa/city/{c}", mine["city_FP"][c], fref["city"].get(c, 0), True)
                for k in SEQS:
                    same(f"{tag}/fa/seq/{k}", mine["sequence"][k]["FP"], fref["sequence"].get(k, 0), True)
                assert set(fref["sequence"]) <= set(SEQS) and set(fref["city"]) <= set(CITY)
            agg = summary["methods"][m][point]["mean_sd"]
            for f, src in (("TP", "TP"), ("FP", "FP"), ("precision", "precision"), ("recall", "recall"),
                           ("city_macro_recall", "city_macro_recall")):
                ms = mean_sd([mine_seeds[s][f] for s in SEEDS])
                same(f"{m}/{point}/mean/{f}", ms["mean"], agg[src]["mean"], False)
                same(f"{m}/{point}/sd/{f}", ms["sd"], agg[src]["sd"], False)
            for c in CITY:
                ms = mean_sd([mine_seeds[s]["city_recall"][c] for s in SEEDS])
                same(f"{m}/{point}/mean/city_recall/{c}", ms["mean"], agg["city_recall"][c]["mean"], False)
    return dict(status="passed" if not mismatches else "failed", checks=n_checks, max_abs_float_diff=worst,
                mismatches=mismatches[:50], n_mismatches=len(mismatches))


# ----------------------------------------------------------------- bootstrap

def draws(npos: dict, seqs: list, city: dict, replicates: int, seed: int) -> tuple:
    """원고 짝지은 구간과 같은 표본: 도시 순서대로 양성 있는 시퀀스를 복원 추출한 횟수(가중치)."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(seqs)))
    cols = []
    for names in city.values():
        available = [q for q in names if npos[q]]
        index = rng.integers(0, len(available), size=(replicates, len(available)))
        for j, q in enumerate(available):
            weights[:, seqs.index(q)] = (index == j).sum(1)
        cols.append(np.array([seqs.index(q) for q in available]))
    zero = [i for i, q in enumerate(seqs) if not npos[q]]
    weights[:, zero] = 1.0     # 양성 없는 시퀀스는 모든 표본에 한 번씩(오수락 예산에만 영향)
    return weights, cols, zero


def city_macro(weights: np.ndarray, rec: np.ndarray, cols: list) -> np.ndarray:
    """weights (R, S), rec (S,) 또는 (S, R): 도시마다 뽑힌 시퀀스 재현율 평균 → 6도시 평균."""
    if rec.ndim == 1:
        per_city = [weights[:, c] @ rec[c] / len(c) for c in cols]
    else:
        per_city = [np.einsum("rs,sr->r", weights[:, c], rec[c]) / len(c) for c in cols]
    return np.mean(per_city, axis=0)


def replicate_curves(b: dict, weights: np.ndarray, fixed_index: dict, cols: list, criteria=CRITERIA) -> dict:
    """한 방법·시작점의 표본별 값. reselect: 표본 곡선에서 다시 고른다. fixed_tau: 전체 시험 τ를 고정한다."""
    n = b["npos_seq"]
    rec_of = lambda t: np.divide(t, n[:, None] if t.ndim == 2 else n, out=np.zeros_like(t),
                                 where=(n[:, None] if t.ndim == 2 else n) > 0)
    denom = weights @ n
    out = {}
    for name, kind, value in criteria:
        i = fixed_index[name]
        t, f = b["tp_seq"][:, i], b["fp_seq"][:, i]
        out[name] = dict(fixed_pooled=(weights @ t) / denom, fixed_city=city_macro(weights, rec_of(t), cols),
                         fixed_FP=weights @ f, reselect_pooled=np.empty(len(weights)),
                         reselect_city=np.empty(len(weights)), reselect_FP=np.empty(len(weights)),
                         reselect_TP=np.empty(len(weights)))
    for start in range(0, len(weights), CHUNK):
        w = weights[start:start + CHUNK]
        tp, fp = w @ b["tp_seq"], w @ b["fp_seq"]
        rows = np.arange(len(w))
        for name, kind, value in criteria:
            i = choose(tp, fp, kind, value)
            o = out[name]
            o["reselect_TP"][start:start + len(w)] = tp[rows, i]
            o["reselect_FP"][start:start + len(w)] = fp[rows, i]
            o["reselect_pooled"][start:start + len(w)] = tp[rows, i] / denom[start:start + len(w)]
            o["reselect_city"][start:start + len(w)] = city_macro(w, rec_of(b["tp_seq"][:, i]), cols)
    return out


def boot_job(job: tuple) -> tuple:
    m, s, fixed_index, expect = job
    b = BUILT[(m, s)]
    ones = np.ones((1, b["tp_seq"].shape[0]))
    check = replicate_curves(b, ones, fixed_index, CITY_COLS)   # 가중치 1 표본 = 전체 시험
    for name, *_ in CRITERIA:
        for v in ("fixed", "reselect"):
            assert abs(check[name][f"{v}_pooled"][0] - expect[name]["recall"]) < 1e-12, (m, s, name, v)
            assert abs(check[name][f"{v}_city"][0] - expect[name]["city_macro_recall"]) < 1e-12, (m, s, name, v)
            assert check[name][f"{v}_FP"][0] == expect[name]["FP"], (m, s, name, v)
        assert check[name]["reselect_TP"][0] == expect[name]["TP"]
    t0 = time.time()
    res = replicate_curves(b, WEIGHTS, fixed_index, CITY_COLS)
    print(f"  bootstrap {m} s{s} done {time.time() - t0:.0f}s", flush=True)
    return m, s, res


def interval(values: np.ndarray) -> list:
    return [100 * float(x) for x in np.quantile(values, [0.025, 0.975])]


def manuscript_draw_check(weights: np.ndarray, cols: list) -> dict:
    """같은 가중치로 원고의 NSD minus BEVPlace++ 구간(시퀀스별 지표 차이 재표집)을 재현하는지 본다."""
    per = read(MAIN / "summary.json")["per_case"]
    target = read(MAIN / "integrated_report.json")["comparisons"]["diff-bev_native"]["conditional_95_pp"]
    out = {}
    for m, ref in target.items():
        d = np.array([np.mean([100 * (per[f"diff_s{s}"]["sequences"][k][m] - per[f"bev_native_s{s}"]["sequences"][k][m])
                               for s in SEEDS]) if per["diff_s1"]["sequences"][k]["n_pos"] else 0.0 for k in SEQS])
        mine = np.quantile(city_macro(weights, d, cols), [0.025, 0.975]).tolist()
        out[m] = dict(recomputed=mine, manuscript=ref, max_abs_diff=max(abs(a - b) for a, b in zip(mine, ref)))
        assert out[m]["max_abs_diff"] < 1e-9, (m, out[m])
    return out


# ---------------------------------------------------------------------- run

def run() -> None:
    global WEIGHTS, CITY_COLS
    t_start = time.time()
    spec_path = OUT / "protocol.json"
    spec = read(spec_path)
    me = rel(Path(__file__))
    script_now = digest(Path(__file__))
    for path, value in spec["code_sha256"].items():
        if path != me:
            assert digest(REPO / path) == value, path
    for path, value in spec["reference_sha256"].items():
        assert digest(REPO / path) == value, path
    done = read(FIXED / "done.json")
    for n in ("summary.json", "freeze.json", "false_accepts.json", "verification.json"):
        assert done["outputs"][rel(FIXED / n)] == digest(FIXED / n), n   # 11-07 산출물 그대로
    assert read(FIXED / "verification.json")["status"] == "passed"
    print("protocol", digest(spec_path), "script unchanged since protocol:", script_now == spec["code_sha256"][me], flush=True)

    expected = read(FIXED / "summary.json")["test_ledgers_sha256"]
    data = load(expected)
    print(f"ledgers: {len(expected)} sha256 match", flush=True)
    built = {key: build(rows, SEQS) for key, rows in data.items()}
    print("curves built", {f"{m}_s{s}": len(b["thresholds"]) for (m, s), b in built.items()}, flush=True)

    rec = reconcile(data, built)
    print("reconciliation", rec["status"], rec["checks"], "checks, max float diff", rec["max_abs_float_diff"], flush=True)
    result = dict(
        description=("같은 오수락 수 비교. 방법·시작점마다 시험 46시퀀스 적격 질의를 합친 sweep에서 합산 오수락 ≤ k "
                     "(보조: 합산 정밀도 ≥ p)를 만족하는 임계값 가운데 정답 수락이 가장 많은 것을 골랐다. 임계값은 시험 점수에서 "
                     "선택했으며(두 방법에 같은 규칙), 검증에서 고정한 임계값이 아니다. 따라서 배치 가능한 운영점의 성능이 아니라 "
                     "같은 오수락 수에서 두 점수가 정답을 얼마나 앞에 두는지의 사후 비교다. 검증 고정 임계값 결과는 "
                     "results/pdv2_review_closure_20261107/threshold/summary.json에 있다."),
        threshold_selected_on="test", validation_fixed=False,
        protocol_sha256=digest(spec_path), script=me, script_sha256=script_now,
        script_unchanged_since_protocol=script_now == spec["code_sha256"][me],
        ledger_check=dict(n=len(expected), all_sha256_match=True, source=rel(FIXED / "summary.json")),
        reconciliation=rec)
    if rec["status"] != "passed":
        result["status"] = "reconciliation_failed"
        write(OUT / "result.json", result)
        raise SystemExit("정산 검증 실패: 분석을 하지 않았다")

    criteria, fixed_index, expect = {}, {}, {}
    for name, kind, value in CRITERIA:
        block = dict(kind=kind, value=value, methods={})
        for m in METHODS:
            per_seed = {}
            for s in SEEDS:
                b = built[(m, s)]
                i = int(choose(b["tp"], b["fp"], kind, value))
                r = at_tau(data[(m, s)], float(b["thresholds"][i]))
                assert r["TP"] == b["tp"][i] and r["FP"] == b["fp"][i]
                assert r["FP"] <= value if kind == "budget" else r["precision"] >= value
                ind = independent(data[(m, s)], SEQS, kind, value)
                assert ind["tau"] == r["tau"] and ind["TP"] == r["TP"] and ind["FP"] == r["FP"], (m, s, name, ind)
                if kind == "budget":
                    assert ind["closed_form_TP"] == r["TP"], (m, s, name, ind)
                r["curve_index"] = i
                r["next_threshold"] = (dict(tau=float(b["thresholds"][i + 1]), TP=int(b["tp"][i + 1]),
                                            FP=int(b["fp"][i + 1])) if i + 1 < len(b["thresholds"]) else None)
                per_seed[str(s)] = r
                fixed_index.setdefault((m, s), {})[name] = i
                expect.setdefault((m, s), {})[name] = r
            pick = lambda f: mean_sd([per_seed[str(s)][f] for s in SEEDS])
            block["methods"][m] = dict(
                mean_sd=dict(**{f: pick(f) for f in ("recall", "TP", "FP", "FP_positive_query", "FP_negative_query",
                                                    "accepted", "precision", "city_macro_recall")},
                             city_recall={c: mean_sd([per_seed[str(s)]["city_recall"][c] for s in SEEDS]) for c in CITY},
                             city_FP={c: mean_sd([per_seed[str(s)]["city_FP"][c] for s in SEEDS]) for c in CITY}),
                per_seed=per_seed)
        a, z = block["methods"]["nsd"]["mean_sd"], block["methods"]["bevplace"]["mean_sd"]
        block["difference_nsd_minus_bevplace"] = dict(
            pooled_recall_pp=100 * (a["recall"]["mean"] - z["recall"]["mean"]),
            city_macro_recall_pp=100 * (a["city_macro_recall"]["mean"] - z["city_macro_recall"]["mean"]),
            TP=a["TP"]["mean"] - z["TP"]["mean"], FP=a["FP"]["mean"] - z["FP"]["mean"],
            city_recall_pp={c: 100 * (a["city_recall"][c]["mean"] - z["city_recall"][c]["mean"]) for c in CITY})
        criteria[name] = block
        d = block["difference_nsd_minus_bevplace"]
        print(f"{name:10s} NSD rec {100*a['recall']['mean']:6.2f} TP {a['TP']['mean']:8.1f} FP {a['FP']['mean']:6.1f} "
              f"city {100*a['city_macro_recall']['mean']:6.2f}±{100*a['city_macro_recall']['sd']:.2f} | "
              f"BEV rec {100*z['recall']['mean']:6.2f} TP {z['TP']['mean']:8.1f} FP {z['FP']['mean']:6.1f} "
              f"city {100*z['city_macro_recall']['mean']:6.2f}±{100*z['city_macro_recall']['sd']:.2f} | "
              f"diff pooled {d['pooled_recall_pp']:+.2f} city {d['city_macro_recall_pp']:+.2f}", flush=True)

    npos = {k: data[("nsd", 1)][k]["qc"]["n_pos"] for k in SEQS}
    WEIGHTS, CITY_COLS, zero = draws(npos, SEQS, CITY, B, BOOT_SEED)
    assert [SEQS[i] for i in zero] == ["kitti/07"]
    draw_check = manuscript_draw_check(WEIGHTS, CITY_COLS)
    print("bootstrap draws reproduce manuscript NSD-BEVPlace++ intervals:",
          {m: round(v["max_abs_diff"], 15) for m, v in draw_check.items()}, flush=True)
    BUILT.update(built)
    jobs = [(m, s, fixed_index[(m, s)], expect[(m, s)]) for m in METHODS for s in SEEDS]
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        boots = {(m, s): res for m, s, res in pool.map(boot_job, jobs)}
    for name, *_ in CRITERIA:
        block = criteria[name]
        block["intervals_95_pp"] = {}
        for variant in ("reselect", "fixed"):
            seed_mean = {m: {f: np.mean([boots[(m, s)][name][f"{variant}_{f}"] for s in SEEDS], axis=0)
                             for f in ("pooled", "city", "FP")} for m in METHODS}
            block["intervals_95_pp"][variant if variant == "reselect" else "fixed_tau"] = dict(
                pooled_recall=interval(seed_mean["nsd"]["pooled"] - seed_mean["bevplace"]["pooled"]),
                city_macro_recall=interval(seed_mean["nsd"]["city"] - seed_mean["bevplace"]["city"]),
                replicate_FP_seed_mean_q025_q50_q975={m: [float(x) for x in np.quantile(seed_mean[m]["FP"], [.025, .5, .975])]
                                                      for m in METHODS})
        iv = block["intervals_95_pp"]
        print(f"{name:10s} reselect pooled [{iv['reselect']['pooled_recall'][0]:+.2f}, {iv['reselect']['pooled_recall'][1]:+.2f}] "
              f"city [{iv['reselect']['city_macro_recall'][0]:+.2f}, {iv['reselect']['city_macro_recall'][1]:+.2f}] | "
              f"fixed pooled [{iv['fixed_tau']['pooled_recall'][0]:+.2f}, {iv['fixed_tau']['pooled_recall'][1]:+.2f}] "
              f"city [{iv['fixed_tau']['city_macro_recall'][0]:+.2f}, {iv['fixed_tau']['city_macro_recall'][1]:+.2f}]", flush=True)

    provisional = {}
    for m in METHODS:
        for k, v in PROVISIONAL_TP[m].items():
            mean = criteria[f"FA<={k}"]["methods"][m]["mean_sd"]["TP"]["mean"]
            provisional[f"{m}/FA<={k}"] = dict(provisional=v, recomputed_seed_mean=mean, match=abs(mean - v) <= 0.5)
    result.update(
        status="passed",
        criteria=criteria,
        intervals_design=dict(replicates=B, seed=BOOT_SEED, primary="reselect", secondary="fixed_tau",
                              zero_positive_fixed=[SEQS[i] for i in zero],
                              manuscript_draw_check=draw_check,
                              note=spec["intervals"]["primary"] + " / " + spec["intervals"]["secondary"]),
        provisional_comparison=provisional,
        elapsed_s=time.time() - t_start)
    write(OUT / "result.json", result)
    print("provisional", {k: (v["provisional"], round(v["recomputed_seed_mean"], 2), v["match"]) for k, v in provisional.items()})
    print(OUT / "result.json", digest(OUT / "result.json"), f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("protocol", "run"))
    globals()[parser.parse_args().action]()
