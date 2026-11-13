#!/usr/bin/env python3
"""무지연(지연 0) NSD: 시간 정제와 수락의 시간 증거를 모두 과거만으로 계산하고, 같은 절차로 추가 학습해 시험46에서 잰다.

현재 NSD는 정제가 앞뒤 5키프레임(두 층이라 i+10)을, 수락의 거리 대응 시간 증거가 질의 앞뒤 5키프레임(q+5)을 읽어서
처리 시점 t에 q=t-15를 질의한다. 과거 전용 학습(`_pdv2_window_side_learned.py`)은 정제 이웃만 과거로 줄이고 시간
증거는 앞뒤 그대로 두어 5키프레임 지연이 남았다. 여기서는 두 쪽 모두 과거만 쓰므로 q=t다.

- 정제: `_pdv2_window_side_learned.past_graph`(시간 간선 src<tgt만 남기고 전달 칸 6~10을 끈 사본)를 그대로 쓴다.
- 시간 증거: `_pdv2_diffattn_spectral_gain.plan`이 모듈 전역 이름으로 부르는 `neighbor_plans`의 결과에서 질의 쪽
  오프셋 +1~+5 칸을 없는 이웃(valid=False)으로 바꾼다. 이후 계산은 기존 plan 그대로다: 없는 칸은 가중치 0, 같은 후보
  중복 계산에서 빠지고, 남은 칸과 중심을 다시 정규화해 x11/15, 중심에 4/15를 더한다. 빠진 칸의 색인은 질의 q와 후보
  j로 둔다(가중치 0). 이것은 q까지 자른 입력에 원래 plan을 돌렸을 때(미래 오프셋이 시퀀스 밖이라 없는 이웃) 가중치와
  같고 가중치가 있는 칸의 색인도 같다.
- 그대로 둔 것: β=11/15, 방향 최댓값, 후보 쪽 대응 창 [j*-10, j*+10], 융합 가중치, 보정 W, 75:25 혼합, 점수 척도.
- 학습: `_pdv2_window_side_learned.train`(과거 창, 전체 학습 diff, 시작점 3, 512갱신)을 그대로 부르고, 학습 풀의 시간
  대응(plan_*)만 무지연 plan으로 바꿔 준다. 손실 척도(loss_fit)·추론 척도·W는 기존 과거 전용 실험처럼 다시 맞추지 않는다.
  같은 경로에 앞뒤 plan을 넣어 다시 학습하면 기존 과거 전용 체크포인트와 비트 단위로 같아야 한다.
- 질의: 기존 공통 질의(q=30..n-16)를 그대로 쓴다. 무지연에서는 q=t이므로 마지막 15프레임도 질의할 수 있지만 짝지은
  비교를 위해 넣지 않는다. 시험셋으로 고르는 설정은 없다.

산출물은 D 드라이브(컨테이너 /workspace/data)에 둔다.
사용법(컨테이너 ring_baseline 안, 작업 폴더 /workspace/Neural-Spectral-Codec):
  python scripts/_pdv2_zero_lag.py protocol    # 실행 전 protocol.json 저장
  python scripts/_pdv2_zero_lag.py train_all   # 무지연 학습 3개 + 앞뒤 plan 재학습 3개(기존 과거 전용 체크포인트와 비트 대조)
  python scripts/_pdv2_zero_lag.py test        # 시험46 평가, 재현·접두 절단 검사, 분석
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

import _pdv2_window_side_learned as W  # noqa: E402

READ, WRITE, HASH = W.READ, W.WRITE, W.HASH
OUT = Path(os.environ.get("NSD_ZERO_LAG_OUT", "/workspace/data/nsd_results/pdv2_zero_lag_20261112"))
TEST = OUT / "test_eval"
PAST = W.OUT  # 기존 과거 전용 학습: 정제만 과거, 시간 증거는 앞뒤
TRAIN_RUNS = {"past": OUT / "train_zero_lag", "both": OUT / "train_bidirectional_repro"}
STARTS, METRICS, GROUPS = W.STARTS, W.METRICS, W.GROUPS
BOOTSTRAP, BOOT_SEED = W.BOOTSTRAP, W.BOOT_SEED
DENSE = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])  # neighbor_plans의 arc_dense 질의 오프셋 순서
FUTURE = DENSE > 0
NEGATIVE_CONTROLS = ("diff_s1", "past_diff_s1")  # 앞뒤 증거 조건: 접두 절단에서 점수가 달라져야 한다
EVIDENCE = {"mode": None}  # "past"(무지연) 또는 "both"(기존 앞뒤)
ORIGINAL = {}

ARMS = {
    "symmetric_nsd": [f"diff_s{k}" for k in STARTS],
    "past_refine_both_evidence": [f"past_diff_s{k}" for k in STARTS],
    "Z1_zero_lag_learned": [f"zl_past_diff_s{k}" for k in STARTS],
    "Z1b_zero_lag_eval_of_past_refine_model": [f"zl_bidirtrained_past_diff_s{k}" for k in STARTS],
    "Z2_zero_lag_no_refine": ["zl_identity"],
    "Z3_zero_lag_past_uniform": ["zl_past_fixed"],
}
CONTRASTS = {
    "Z1_minus_symmetric": [(f"zl_past_diff_s{k}", f"diff_s{k}") for k in STARTS],
    "Z2_minus_symmetric": [("zl_identity", f"diff_s{k}") for k in STARTS],
    "Z3_minus_symmetric": [("zl_past_fixed", f"diff_s{k}") for k in STARTS],
    "Z1b_minus_symmetric": [(f"zl_bidirtrained_past_diff_s{k}", f"diff_s{k}") for k in STARTS],
    "Z1_minus_Z3": [(f"zl_past_diff_s{k}", "zl_past_fixed") for k in STARTS],
    "Z1_minus_Z2": [(f"zl_past_diff_s{k}", "zl_identity") for k in STARTS],
    "Z1_minus_past_refine_both_evidence": [(f"zl_past_diff_s{k}", f"past_diff_s{k}") for k in STARTS],
    "Z1_minus_Z1b": [(f"zl_past_diff_s{k}", f"zl_bidirtrained_past_diff_s{k}") for k in STARTS],
    "past_refine_both_evidence_minus_symmetric": [(f"past_diff_s{k}", f"diff_s{k}") for k in STARTS],
}


def past_checkpoint(k: int) -> Path:
    return PAST / "runs/past" / f"s{k}" / "diff" / "epoch8.pth"


def zero_lag_checkpoint(k: int) -> Path:
    return TRAIN_RUNS["past"] / "runs/past" / f"s{k}" / "diff" / "epoch8.pth"


def reference_ledger(tag: str, name: str):
    """기존 원장을 재사용하는 참조 조건의 경로. 무지연 조건은 None."""
    if tag.startswith("diff_s"):
        return W.OLD_TEST / "test" / name / f"{tag}.npz"
    if tag.startswith("past_diff_s"):
        return PAST / "test_eval/test" / name / f"{tag}.npz"
    return None


# ---------------------------------------------------------------- 무지연 시간 증거
def modules():
    """학습·평가가 쓰는 같은 모듈. R을 먼저 불러 학습 손실 척도 교체(R.score)를 기존과 같게 둔다."""
    import _pdv2_diffattn_contrast_recalibrated as R

    sg = R.e.prior.base.prior  # _pdv2_diffattn_spectral_gain: plan
    ORIGINAL.setdefault("plan", sg.plan)
    ORIGINAL.setdefault("neighbor_plans", sg.neighbor_plans)
    return R.e, sg


def past_neighbor_plans(path, query, candidate) -> dict:
    """기존 neighbor_plans 결과에서 질의 쪽 미래 오프셋 +1~+5를 없는 이웃으로 바꾼다. 다른 계산은 그대로다."""
    result = ORIGINAL["neighbor_plans"](path, query, candidate)
    q, j = np.asarray(query, np.int64), np.asarray(candidate, np.int64)
    qn, jn, valid = (v.copy() for v in result["arc_dense"])
    assert np.array_equal(qn - q[None, :, None], np.broadcast_to(DENSE, qn.shape))
    qn[..., FUTURE] = q[None, :, None]
    jn[..., FUTURE] = j[None, :, None]
    valid[..., FUTURE] = False
    result["arc_dense"] = result["arc_unique"] = (qn, jn, valid)
    return result


def zero_lag_plan(travel, q, j) -> dict:
    """기존 plan을 그대로 부르되 그 안의 neighbor_plans만 과거 전용으로 바꾼다."""
    _, sg = modules()
    sg.neighbor_plans = past_neighbor_plans  # plan()은 모듈 전역 이름으로 부른다
    try:
        plan = ORIGINAL["plan"](travel, q, j)
    finally:
        sg.neighbor_plans = ORIGINAL["neighbor_plans"]
    assert (plan["query"] <= np.asarray(q)[:, None, None]).all()  # 질의 쪽은 q 이후 프레임을 가리키지 않는다
    return plan


def evidence_plan(travel, q, j) -> dict:
    mode = EVIDENCE["mode"]
    assert mode in ("past", "both"), mode
    return zero_lag_plan(travel, q, j) if mode == "past" else ORIGINAL["plan"](travel, q, j)


def plan_stats(plan: dict, q: np.ndarray) -> dict:
    w = plan["weight"]
    return dict(center_weight_mean=float(w[..., 0].mean()),
                weighted_neighbors_mean=float((w[..., 1:] > 0).sum(-1).mean()),
                max_query_offset=int((plan["query"] - q[:, None, None]).max()))


# ---------------------------------------------------------------- 학습
def install_train_plans() -> dict:
    """W.train이 읽는 학습 풀의 plan_*을 EVIDENCE 방식으로 다시 계산해 돌려준다. 나머지 풀 내용은 그대로다."""
    e, _ = modules()
    sequences = READ(W.RECAL / "protocol.json")["train"]
    names = {s.replace("/", "_"): s for s in sequences}
    pools = (e.prior.PARENT / "pools/train").resolve()
    original = ORIGINAL.setdefault("ARR", e.run.ARR)
    cache, audit = {}, {}

    def arr(path):
        data = original(path)
        path = Path(path)
        if path.name != "pool.npz" or path.parent.parent.resolve() != pools:
            return data
        name, mode = path.parent.name, EVIDENCE["mode"]
        if (name, mode) not in cache:
            travel = e.run.inputs("train", names[name])[3]
            q, j = data["query_idx"], data["j_star"]
            again = ORIGINAL["plan"](travel, q, j)
            exact = all(np.array_equal(again[k], data["plan_" + k]) for k in again)
            assert exact, name  # 저장된 앞뒤 plan을 같은 함수로 정확히 다시 만든다
            plan = zero_lag_plan(travel, q, j) if mode == "past" else again
            cache[(name, mode)] = plan
            audit[f"{mode}/{names[name]}"] = dict(queries=len(q), stored_plan_recomputed_exact=exact,
                                                  **plan_stats(plan, q))
        data.update({"plan_" + k: v for k, v in cache[(name, mode)].items()})
        return data

    e.run.ARR = arr
    for mode in ("past", "both"):
        EVIDENCE["mode"] = mode
        for s in sequences:
            arr(e.prior.PARENT / "pools/train" / s.replace("/", "_") / "pool.npz")
    EVIDENCE["mode"] = None
    return audit


def train_all() -> None:
    """무지연 학습 3개와 앞뒤 plan 재학습 3개. 재학습은 기존 과거 전용 체크포인트와 비트 단위로 같아야 한다."""
    import torch

    assert (OUT / "protocol.json").exists()
    audit = install_train_plans()
    WRITE(OUT / "train_plans.json", audit)
    for mode in ("past", "both"):
        EVIDENCE["mode"] = mode
        W.OUT = TRAIN_RUNS[mode]  # W.run_folder가 쓰는 출력 위치
        for k in STARTS:
            W.train("past", k, "diff")
    EVIDENCE["mode"] = None
    W.OUT = PAST

    def state(path):
        return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]

    rows, training = {}, {}
    for k in STARTS:
        old = past_checkpoint(k)
        assert HASH(old) == READ(old.parent / "training_complete.json")["sha256"]
        a = state(TRAIN_RUNS["both"] / "runs/past" / f"s{k}" / "diff" / "epoch8.pth")
        b, z = state(old), state(zero_lag_checkpoint(k))
        assert a.keys() == b.keys() == z.keys()
        rows[f"past_diff_s{k}"] = dict(
            exact=all(torch.equal(a[n], b[n]) for n in a),
            max_abs=max(float((a[n].double() - b[n].double()).abs().max()) for n in a),
            original=str(old),
            zero_lag_max_abs_vs_original=max(float((z[n].double() - b[n].double()).abs().max()) for n in z))
        record = READ(zero_lag_checkpoint(k).parent / "training_complete.json")
        losses = np.array([r["loss"] for r in record["records"]]).reshape(8, -1)
        training[f"zl_past_diff_s{k}"] = dict(
            finite=bool(np.isfinite(losses).all()), epoch_mean_loss=losses.mean(1).tolist(),
            final_strength_mean=float(np.mean([r["strength_mean"] for r in record["records"][-losses.shape[1]:]])),
            seconds=record["seconds"], init_error=record["init_error"])
    status = "passed" if all(r["exact"] for r in rows.values()) else "failed"
    report = dict(training=dict(status=status, checkpoints=rows), zero_lag_training=training)
    WRITE(OUT / "reproduction.json", report)
    print("REPRODUCTION", json.dumps(report, ensure_ascii=False), flush=True)
    assert status == "passed", rows
    assert all(v["finite"] for v in training.values())


# ---------------------------------------------------------------- 시험
def checkpoint_case(kind: str, k: int, path: Path, evidence: str) -> dict:
    digest = HASH(path)
    assert READ(path.parent / "training_complete.json")["sha256"] == digest
    return dict(kind=kind, seed=k, checkpoint=str(path), checkpoint_sha256=digest, evidence=evidence)


def test_spec() -> dict:
    """기존 시험 명세에서 사례만 바꾼다. evidence는 시간 증거 방식이다(past=무지연, both=기존 앞뒤)."""
    path = TEST / "protocol.json"
    if path.exists():
        return READ(path)
    old = READ(W.OLD_TEST / "protocol.json")
    cases = {}
    for k in STARTS:
        case = dict(old["cases"][f"diff_s{k}"], evidence="both")
        assert HASH(ROOT / case["checkpoint"]) == case["checkpoint_sha256"]
        cases[f"diff_s{k}"] = case
        cases[f"past_diff_s{k}"] = checkpoint_case("past_diff", k, past_checkpoint(k), "both")
        cases[f"zl_past_diff_s{k}"] = checkpoint_case("past_diff", k, zero_lag_checkpoint(k), "past")
        cases[f"zl_bidirtrained_past_diff_s{k}"] = checkpoint_case("past_diff", k, past_checkpoint(k), "past")
    cases["zl_identity"] = dict(kind="identity", seed=0, checkpoint=None, evidence="past")
    cases["zl_past_fixed"] = dict(kind="past_fixed", seed=0, checkpoint=None, evidence="past")
    spec = dict(old)
    spec.update(created_unix=time.time(), cases=cases,
                test_policy="고정 설정의 시험 보고. 설정 선택 없음. 조건은 한 번 실행.")
    WRITE(path, spec)
    return spec


def install_eval(pipeline) -> None:
    """과거 창(W.patch) 위에, 사례의 evidence로 시간 증거 plan을 고르는 연결을 더한다."""
    _, sg = modules()
    W.patch(pipeline)
    load = pipeline.load_model

    def load_model(case: dict):
        EVIDENCE["mode"] = case["evidence"]  # worker는 사례마다 load_model → transform → plan 순서로 부른다
        return load(case)

    pipeline.load_model = load_model
    sg.plan = evidence_plan


def prefix_checks(pipeline, sequence: str, folder: Path) -> list:
    """q까지만 준 입력(길이 q+1)으로 반환 장소와 수락 점수를 다시 계산해 전체 입력 원장과 대조한다."""
    import torch

    spec = test_spec()
    run, e, sg = pipeline.run, pipeline.e, modules()[1]
    cache, _, _, travel, bank, _, _, _ = run.inputs("test", sequence)
    graph, _, _ = e.prior.graph_inputs("test", sequence)
    weights = run.cal.fusion_weights(spec["confidence"])[run.cal.FUSIONS.index("aux_half")]
    queries = np.load(folder / "zl_identity.npz")["query_idx"]
    positions = sorted({len(queries) // 3, 2 * len(queries) // 3, len(queries) - 1})
    prefixes = {}
    for at in positions:
        end = int(queries[at]) + 1
        small, short_travel, _ = e.prior.graph_inputs("test", sequence, end)
        prefixes[end] = (small, short_travel)

    def cuda(plan):
        return {k: torch.from_numpy(v).cuda() for k, v in plan.items()}

    tags = [t for t in spec["cases"] if t.startswith("zl_")] + list(NEGATIVE_CONTROLS)
    rows = []
    with torch.no_grad():
        for tag in tags:
            case = spec["cases"][tag]
            kind = case["kind"]
            net = pipeline.load_model(case)
            z = pipeline.transform(net, graph, kind)
            with np.load(folder / f"{tag}.npz") as ledger:
                ledger = {k: ledger[k] for k in ("query_idx", "j_star", "score")}
            for at in positions:
                t = int(ledger["query_idx"][at])
                end = t + 1
                small, short_travel = prefixes[end]
                short = pipeline.transform(net, small.clone(), kind)
                skey = pipeline.raw_key(cache["descriptors"][:end])
                short_bank = (tuple(v[:end] for v in bank) if kind == "identity"
                              else run.base.variant_inputs(short.cpu().numpy(), 8))
                winner = int(run.cal.candidates(skey, (bank[0][:end], *short_bank[1:]), weights[None], t)[0])
                full_plan = sg.plan(travel, np.array([t]), ledger["j_star"][at:at + 1])
                plan = sg.plan(short_travel, np.array([t]), np.array([winner]))
                original = ORIGINAL["plan"](short_travel, np.array([t]), np.array([winner]))
                score = float(e.prior.score(short, cuda(plan)).cpu().numpy()[0])
                score_original = float(e.prior.score(short, cuda(original)).cpu().numpy()[0])
                rows.append(dict(
                    condition=tag, evidence=case["evidence"], query=t, prefix_frames=end, full_frames=len(z),
                    winner_full=int(ledger["j_star"][at]), winner_prefix=winner,
                    coefficient_error=float((short - z[:end]).abs().max()),
                    plan_equal=all(np.array_equal(full_plan[k], plan[k]) for k in plan),
                    max_plan_frame=int(max(plan["query"].max(), plan["candidate"].max())),
                    score_full=float(ledger["score"][at]),
                    score_error=abs(score - float(ledger["score"][at])),
                    original_plan_on_prefix_score_error=abs(score_original - float(ledger["score"][at]))))
                if case["evidence"] == "past":
                    r = rows[-1]
                    assert r["winner_full"] == r["winner_prefix"], r
                    assert r["coefficient_error"] < 4e-6 and r["plan_equal"] and r["max_plan_frame"] <= t, r
                    assert r["score_error"] < 8e-6 and r["original_plan_on_prefix_score_error"] < 8e-6, r
            del net, z
    return rows


def test_worker(sequence: str) -> None:
    """한 시퀀스의 사례를 평가하고 참조 원장 재현, 증거만 바뀐 사례의 승자 일치, 접두 절단 검사를 기록한다."""
    import _pdv2_contrast_pipeline_test as T

    T.OUT = TEST
    T.prepare = test_spec
    T.pipeline.e.configure()
    install_eval(T.pipeline)
    name = sequence.replace("/", "_")
    folder = TEST / "test" / name
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".json"):
        dst = folder / ("candidate_pool" + suffix)
        if not dst.exists():
            shutil.copy2(W.OLD_TEST / "test" / name / ("candidate_pool" + suffix), dst)
    T.worker("test", sequence)
    target = folder / "zero_lag_check.json"
    if target.exists():
        assert READ(target)["status"] == "passed"
        return
    spec = test_spec()
    reproduction = {}
    for tag in spec["cases"]:
        ref = reference_ledger(tag, name)
        if ref is None:
            continue
        new, old = np.load(folder / f"{tag}.npz"), np.load(ref)
        for field in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist"):
            np.testing.assert_array_equal(new[field], old[field])
        error = float(np.abs(new["score"] - old["score"]).max())
        assert error < 8e-6, (tag, error)
        reproduction[tag] = dict(winners_exact=True, score_error=error, reference=str(ref))
    # 시간 증거만 바뀐 사례는 반환 장소가 기존 원장과 같아야 한다.
    same_winners = {"zl_identity": W.OLD_TEST / "test" / name / "identity.npz",
                    "zl_past_fixed": PAST / "test_eval/test" / name / "past_fixed.npz"}
    same_winners.update({f"zl_bidirtrained_past_diff_s{k}": PAST / "test_eval/test" / name / f"past_diff_s{k}.npz"
                         for k in STARTS})
    winners = {}
    for tag, ref in same_winners.items():
        new, old = np.load(folder / f"{tag}.npz"), np.load(ref)
        np.testing.assert_array_equal(new["query_idx"], old["query_idx"])
        np.testing.assert_array_equal(new["j_star"], old["j_star"])
        winners[tag] = dict(winners_equal_to=str(ref), score_changed=bool(np.abs(new["score"] - old["score"]).max() > 8e-6))
    # 공통 질의 범위: 지연 15 기준의 q=30..n-16. 무지연에서도 과거 오프셋 q-5..q-1이 모두 시퀀스 안이다.
    cache = T.inputs("test", sequence)[0]
    n = len(cache["descriptors"])
    q = np.load(folder / "zl_identity.npz")["query_idx"]
    np.testing.assert_array_equal(q, np.arange(30, n - 15))
    query_range = dict(frames=n, first=int(q[0]), last=int(q[-1]), count=len(q), past_offsets_in_range=bool(q.min() - 5 >= 0))
    prefix = prefix_checks(T.pipeline, sequence, folder)
    WRITE(target, dict(status="passed", reproduction=reproduction, evidence_only_winners=winners,
                       query_range=query_range, prefix=prefix))
    print("CHECKED", sequence, flush=True)


def test() -> None:
    report = READ(OUT / "reproduction.json")
    assert report["training"]["status"] == "passed"
    spec = test_spec()
    logs = TEST / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tick = time.time()

    def call(sequence):
        with (logs / (sequence.replace("/", "_") + ".log")).open("a") as stream:
            subprocess.run([sys.executable, __file__, "test_worker", "--sequence", sequence],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, s) for s in spec["test"]]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            print("TEST_WORKER DONE", i, len(futures), round(time.time() - tick, 1), flush=True)
    analyze()


# ---------------------------------------------------------------- 분석
def analyze() -> None:
    """방향별 6도시 매크로, 시작점 평균·표준편차, 도시 안 시퀀스 짝지은 재표집 차이. 집계·재표집은 W.analyze와 같다."""
    import _pdv2_pilot as pilot
    import _review_direction_metrics as direction

    spec = test_spec()
    tags = list(spec["cases"])
    rows, rows_new = {}, {}  # 시퀀스 -> 사례 -> 방향 -> 지표. rows_new는 참조 사례를 이 스크립트로 다시 계산한 원장
    ledger_errors, prefix, winners, total = [], [], {}, 0
    for s in spec["test"]:
        name = s.replace("/", "_")
        check = READ(TEST / "test" / name / "zero_lag_check.json")
        assert check["status"] == "passed"
        ledger_errors += [v["score_error"] for v in check["reproduction"].values()]
        prefix += [dict(r, sequence=s) for r in check["prefix"]]
        winners[s] = check["evidence_only_winners"]
        with np.load(W.DIRECTIONS / "directions" / (name + ".npz")) as d:
            label = {k: d[k] for k in ("query_idx", "group")}
        total += len(label["query_idx"])
        masks = dict(all=None, forward_only=label["group"] == 0, reverse_only=label["group"] == 1)
        rows[s], rows_new[s] = {}, {}
        for t in tags:
            ref = reference_ledger(t, name)
            for store, path in ((rows, ref or TEST / "test" / name / f"{t}.npz"),
                                (rows_new, TEST / "test" / name / f"{t}.npz" if ref else None)):
                if path is None:
                    continue
                with np.load(path) as z:
                    np.testing.assert_array_equal(z["query_idx"], label["query_idx"])
                    store[s][t] = {g: direction.metric(z["score"], z["top1_geo_dist"], z["min_past_gt_dist"], m)
                                   for g, m in masks.items()}
    assert total == 195726, total

    valid = {g: {c: [s for s in seqs if rows[s]["diff_s1"][g] is not None] for c, seqs in pilot.CITY.items()}
             for g in GROUPS}
    for s in rows:
        for g in GROUPS:
            assert len({rows[s][t][g] is None for t in tags}) == 1

    def table(source):
        return {g: {t: {m: {c: np.array([source[s][t][g][m] for s in seqs]) for c, seqs in valid[g].items() if seqs}
                        for m in METRICS} for t in source[spec["test"][0]]} for g in GROUPS}

    values, values_new = table(rows), table(rows_new)

    def macro(by_city: dict, picks: dict = None):
        parts = [v.mean() if picks is None else v[picks[c]].mean(axis=1) for c, v in by_city.items()]
        return np.mean(parts, axis=0)

    # 재현: 다시 계산한 참조 원장과 재사용 원장이 기존 보고값(대칭 diff, 과거 전용 학습 diff)과 같다.
    side = READ(PAST / "result.json")["groups"]
    old = READ(W.OLD_TEST / "summary.json")
    gaps = {}
    for g in GROUPS:
        for m in METRICS:
            for arm, prefix_tag in (("symmetric_learned_diff", "diff_s"), ("past_learned_diff", "past_diff_s")):
                for i, k in enumerate(STARTS):
                    reported = side[g]["percent"][arm][m]["per_start"][i]
                    for label, source in (("reused", values), ("recomputed", values_new)):
                        gaps[f"{label}_{prefix_tag}{k}_{g}_{m}"] = abs(100 * macro(source[g][f"{prefix_tag}{k}"][m]) - reported)
    for m in METRICS:
        for k in STARTS:
            gaps[f"summary_diff_s{k}_{m}"] = 100 * abs(macro(values_new["all"][f"diff_s{k}"][m])
                                                       - old["per_case"][f"diff_s{k}"]["macro"][m])
    max_gap = float(max(gaps.values()))
    assert max_gap < 1e-9, max(gaps.items(), key=lambda kv: kv[1])

    rng = np.random.default_rng(BOOT_SEED)
    result = dict(arms=ARMS, contrasts={k: [list(p) for p in v] for k, v in CONTRASTS.items()}, groups={})
    for g in GROUPS:
        sizes = {c: len(seqs) for c, seqs in valid[g].items() if seqs}
        picks = {c: rng.integers(0, n, (BOOTSTRAP, n)) for c, n in sizes.items()}
        block = dict(sequences_per_city=sizes, percent={}, differences_pp={})
        for name, arm_tags in ARMS.items():
            block["percent"][name] = {}
            for m in METRICS:
                per = [100 * float(macro(values[g][t][m])) for t in arm_tags]
                block["percent"][name][m] = dict(mean=float(np.mean(per)), per_start=per,
                                                 sd=float(np.std(per, ddof=1)) if len(per) > 1 else 0.0)
        for name, pairs in CONTRASTS.items():
            block["differences_pp"][name] = {}
            for m in METRICS:
                diff = {c: np.mean([values[g][a][m][c] - values[g][b][m][c] for a, b in pairs], axis=0)
                        for c in sizes}
                boot = 100 * macro(diff, picks)
                block["differences_pp"][name][m] = dict(
                    mean_pp=100 * float(macro(diff)),
                    per_start_pp=[100 * float(macro(values[g][a][m]) - macro(values[g][b][m])) for a, b in pairs],
                    ci95_pp=[float(v) for v in np.percentile(boot, [2.5, 97.5])])
        result["groups"][g] = block

    # 기존 실험의 같은 대조(과거 전용 학습 - 대칭 학습)가 구간까지 같다.
    for g in GROUPS:
        for m in METRICS:
            a = result["groups"][g]["differences_pp"]["past_refine_both_evidence_minus_symmetric"][m]
            b = side[g]["differences_pp"]["past_minus_symmetric_learned_diff"][m]
            gaps[f"contrast_{g}_{m}"] = max(abs(a["mean_pp"] - b["mean_pp"]),
                                            *[abs(x - y) for x, y in zip(a["ci95_pp"], b["ci95_pp"])])
    max_gap = float(max(gaps.values()))
    assert max_gap < 1e-9, max(gaps.items(), key=lambda kv: kv[1])

    zl = [r for r in prefix if r["evidence"] == "past"]
    negative = [r for r in prefix if r["evidence"] == "both"]
    causality = dict(
        rule="질의 q까지의 접두 입력(q+1프레임)으로 그래프·정제·후보 검색·시간 대응·수락 점수를 다시 계산해 전체 입력 원장과 대조",
        zero_lag=dict(checks=len(zl), sequences=len({r["sequence"] for r in zl}),
                      conditions=sorted({r["condition"] for r in zl}),
                      winners_equal=all(r["winner_full"] == r["winner_prefix"] for r in zl),
                      plans_equal=all(r["plan_equal"] for r in zl),
                      max_plan_frame_minus_query=max(r["max_plan_frame"] - r["query"] for r in zl),
                      max_coefficient_error=max(r["coefficient_error"] for r in zl),
                      max_score_error=max(r["score_error"] for r in zl),
                      max_original_plan_on_prefix_score_error=max(r["original_plan_on_prefix_score_error"] for r in zl)),
        negative_controls={c: dict(checks=sum(r["condition"] == c for r in negative),
                                   score_changed=sum(r["condition"] == c and r["score_error"] > 8e-6 for r in negative),
                                   winner_changed=sum(r["condition"] == c and r["winner_full"] != r["winner_prefix"]
                                                      for r in negative),
                                   coefficient_changed=sum(r["condition"] == c and r["coefficient_error"] > 4e-6
                                                           for r in negative),
                                   max_score_error=max(r["score_error"] for r in negative if r["condition"] == c))
                           for c in NEGATIVE_CONTROLS})
    training = {}
    for t, c in spec["cases"].items():
        if c["checkpoint"] and t.startswith(("zl_past_diff", "past_diff")):
            record = READ(Path(c["checkpoint"]).parent / "training_complete.json")
            losses = np.array([r["loss"] for r in record["records"]]).reshape(8, -1)
            training[t] = dict(finite=bool(np.isfinite(losses).all()), epoch_mean_loss=losses.mean(1).tolist(),
                               final_epoch_strength_mean=float(np.mean(
                                   [r["strength_mean"] for r in record["records"][-losses.shape[1]:]])),
                               init_error=record["init_error"], causal_check=record["causal_check"],
                               seconds=record["seconds"])
    reproduction = READ(OUT / "reproduction.json")
    reproduction["evaluation"] = dict(
        status="passed", ledger_max_score_error=max(ledger_errors), max_macro_or_ci_gap_pp=max_gap,
        rule="대칭 diff·과거 전용 학습 diff를 이 스크립트 경로로 다시 계산해 기존 원장(승자 일치, 점수 차 8e-6 미만)과 기존 "
             "보고값(시작점별 매크로, 과거-대칭 대조의 평균·구간, 1e-9%p 미만)에 대조",
        evidence_only_winners_equal=True)
    WRITE(OUT / "reproduction.json", reproduction)
    result.update(
        sanity_checks=dict(ledger_max_score_error=max(ledger_errors), max_reproduction_gap_pp=max_gap,
                           queries=total, evidence_only_cases_keep_winners=True),
        causality=causality, training=training, reproduction=reproduction,
        script_sha256=HASH(Path(__file__)),
        protocol_script_sha256=READ(OUT / "protocol.json")["sources"]["scripts/_pdv2_zero_lag.py"],
        aggregation="시험46 중 해당 방향 양성이 있는 시퀀스 -> 도시 -> 양성이 있는 도시(6) 평균 -> 시작점 3개 평균(SD는 시작점 간, ddof=1).",
        bootstrap=dict(replicates=BOOTSTRAP, seed=BOOT_SEED,
                       scheme="도시 안 시퀀스 수 유지 짝지은 복원추출. 시작점은 고정하고 차이는 시작점 평균. 방향 묶음 순서와 난수 사용 순서는 기존 과거 전용 학습 분석과 같다."),
        test_used_for_selection=False)
    WRITE(OUT / "result.json", result)
    print(json.dumps({g: {a: round(v["AP"]["mean"], 2) for a, v in b["percent"].items()}
                      for g, b in result["groups"].items()}, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- 사전 기록
def protocol() -> None:
    path = OUT / "protocol.json"
    if path.exists():
        print("PROTOCOL EXISTS", path, flush=True)
        return
    sources = [Path(__file__), Path(W.__file__), ROOT / "src/retrieval/sequence_evidence.py",
               ROOT / "src/retrieval/context_spectral_gain.py", ROOT / "scripts/_pdv2_diffattn_spectral_gain.py",
               ROOT / "scripts/_pdv2_diffattn_complex_difference.py", ROOT / "scripts/_pdv2_diffattn_contrast_recalibrated.py",
               ROOT / "scripts/_pdv2_contrast_pipeline.py", ROOT / "scripts/_pdv2_contrast_pipeline_test.py",
               W.RECAL / "protocol.json", W.OLD_TEST / "protocol.json", PAST / "result.json"]
    sources += [W.PRETRAIN / f"s{k}/octave288/checkpoints/checkpoint_epoch_20.pth" for k in STARTS]
    sources += [past_checkpoint(k) for k in STARTS]
    WRITE(path, dict(
        created_unix=time.time(),
        purpose="미래 관측을 전혀 쓰지 않는 무지연(q=t) NSD를 원고와 같은 학습 절차로 학습해 시험46에서 대칭 NSD와 짝지어 비교한다.",
        zero_lag_definition=dict(
            refinement="기존 과거 전용 그래프(_pdv2_window_side_learned.past_graph): 시간 간선 src<tgt만, 전달 칸 6~10 끔. 정제 계수 z'_i는 i-10..i에만 의존.",
            temporal_evidence="수락 점수의 거리 대응 시간 증거에서 질의 쪽 오프셋을 u∈{-5..-1}만 쓴다. neighbor_plans 결과의 u=+1..+5 칸을 valid=False(기존 '없는 이웃' 처리)로 바꾸고 색인은 q·j로 둔다. 이후 중복 후보 가중치, 정규화, x11/15, 중심 +4/15는 기존 plan 그대로.",
            unchanged="β=11/15, 정·역방향 최댓값, 후보 쪽 대응 창 [j*-10, j*+10], 융합 가중치(1,0.5,2,1), 보정 W, 75:25 혼합, (30*c_spec+c_col)/2, 추론 점수 척도, 후보 j<=q-30, 원키·보조 색인 100+100.",
            query_time="q=t(지연 0). 후보 쪽 프레임은 j+10<=q-20이라 모두 과거.",
            new_hyperparameters="없음"),
        conditions=dict(
            Z1="zl_past_diff_s{1,2,3}: 과거 정제 + 과거 증거로 추가 학습한 모델, 과거 정제 + 과거 증거로 평가",
            Z2="zl_identity: 정제 없음 + 과거 증거, 학습 없음",
            Z3="zl_past_fixed: 과거 균일 감산 -0.5 두 층 + 과거 증거, 학습 없음",
            Z1b="zl_bidirtrained_past_diff_s{1,2,3}(추가 조건): 기존 과거 전용 학습 모델(앞뒤 증거로 학습)을 과거 증거로 평가. 학습 목표의 증거 차이를 분리한다.",
            references="diff_s{1,2,3}(대칭 NSD), past_diff_s{1,2,3}(과거 정제 + 앞뒤 증거 학습 모델). 분석은 기존 원장을 재사용하고, 같은 스크립트 경로로 다시 계산해 재현을 확인한다."),
        training=dict(
            procedure="_pdv2_window_side_learned.train('past', k, 'diff')를 그대로 호출: 사전학습 시드 k 20에포크 시작점, ContrastSpectralDifference(diff) 균등 감산 -0.5 초기화, 학습8, 8에포크x8시퀀스x8갱신=512, 배치 64(정답/오답 균등), 실행 시드0 표본 순서(45000+1000*epoch+ordinal), AdamW lr1e-4·감쇠1e-5, 기울기 절단 1, BCE.",
            only_change="학습 풀의 plan_query/plan_candidate/plan_weight를 무지연 plan으로 바꿔 준다(저장 plan을 같은 함수로 정확히 다시 만드는지 먼저 확인). 표본, 정답 라벨(원키 승자 기준), 그래프 사본은 기존 과거 전용과 같다.",
            score_scale="학습 BCE의 손실 척도(_pdv2_diffattn_contrast_recalibrated의 loss_fit, 대칭 감산 -0.5 초기 출력에 학습8에서 맞춘 양의 선형 변환)와 추론 척도(calibration_recenter), 보정 W는 조건에 따라 다시 맞추지 않는다. 기존 과거 전용 학습과 같은 원칙.",
            reproduction="같은 경로에 앞뒤 plan을 넣어 다시 학습한 3개가 기존 과거 전용 체크포인트와 비트 단위로 같아야 한다."),
        evaluation=dict(
            queries="기존 공통 질의(각 시퀀스 q=30..n-16, 시험46 합 195,726개)와 같은 후보 풀·정답·방향 라벨. 무지연에서는 마지막 15프레임도 질의할 수 있으나 짝지은 비교를 위해 넣지 않는다.",
            metrics=list(METRICS), groups=list(GROUPS),
            direction_labels=str(W.DIRECTIONS / "directions"),
            aggregation="해당 방향 양성이 있는 시퀀스 -> 도시 -> 6도시 평균 -> 시작점 3개 평균(SD ddof=1).",
            bootstrap=f"도시 안 양성 시퀀스 수 유지 짝지은 복원추출 {BOOTSTRAP}회, 시드 {BOOT_SEED}(기존 과거 전용 학습 분석과 같은 난수 순서). 시작점 고정, 차이는 시작점 평균.",
            contrasts={k: [list(p) for p in v] for k, v in CONTRASTS.items()}),
        checks=dict(
            causality="무지연 사례 전부와 시퀀스 46개 각각 질의 3개(1/3, 2/3, 마지막 위치)에서 q+1프레임 접두 입력으로 다시 계산: 반환 장소 같음, 정제 계수 차 <4e-6, plan 같음·q 이후 프레임 미참조, 점수 차 <8e-6. 원래 plan을 접두 입력에 돌린 점수도 같아야 한다. 대조로 diff_s1·past_diff_s1(앞뒤 증거)은 점수가 달라져야 한다.",
            reproduction="대칭 diff AP 89.43/F1 89.33/R@100P 69.65/R@1 91.73, 과거 전용 학습 diff AP 82.80±0.12/F1 83.94±0.09/R@100P 50.37±0.09/R@1 86.60±0.09와 그 대조 구간을 재현.",
            evidence_only="Z2·Z3·Z1b는 시간 증거만 바뀌므로 반환 장소가 기존 identity·past_fixed·past_diff 원장과 같아야 한다.",
            training_log="모든 갱신의 손실·기울기 유한(W.train의 assert), 에포크 평균 손실 추이 기록."),
        selection="시험셋으로 고르는 설정 없음. 조건은 한 번 실행한다.",
        train=READ(W.RECAL / "protocol.json")["train"],
        test=READ(W.OLD_TEST / "protocol.json")["test"],
        sources={(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)): HASH(p) for p in sources}))
    print("PROTOCOL WRITTEN", path, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("protocol", "train_all", "test", "test_worker", "analyze"))
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "test_worker":
        test_worker(a.sequence)
    else:
        globals()[a.action]()
