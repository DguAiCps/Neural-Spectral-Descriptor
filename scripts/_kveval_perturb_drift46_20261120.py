#!/usr/bin/env python3
"""오도메트리 교란(B1·B2)의 드리프트 크기를 시험46 전 시퀀스에서 다시 잰다(학습·평가 없음, CPU).

교란은 scripts/_kveval_perturb_20261118.py의 reintegrate·seed_of·조건 정의를 그대로 불러 같은 실현을 만든다.
기준 궤적이 둘이다.
  (a) 교란이 만든 드리프트: 교란 궤적 대 교란 전 KISS-ICP 궤적(캐시 poses). 같은 몸체 좌표·같은 시작 자세라 정렬이 필요 없다.
  (b) KISS-ICP 자체 드리프트: KISS-ICP 궤적 대 참값(캐시 poses_gt). 데이터셋마다 참값 몸체 좌표가 라이다 좌표와 다르다
      (KITTI는 카메라 좌표 등). MCD 캐시의 poses_gt는 위치만 있고 회전이 단위 행렬이다.
  (c) 참고: 교란 궤적 대 참값(교란 + KISS-ICP 자체 드리프트), (b)와 같은 정렬.
지표 세 개(정의 전문은 출력 json의 definitions).
  end: 끝점 위치 오차 / 기준 궤적 주행 거리, 시작 자세가 같은 (a)에만. 기존 inputs 진단 end_drift_frac_of_travel과 같은 정의(검산).
  end_pos: 같은 비율이되 기준 궤적 주행 첫 100 m 위치로 강체 정렬한 뒤. 회전 없는 참값에도 쓸 수 있어 (a)(b)(c)를 같은 정의로 잰다.
  trel: KITTI 오도메트리 방식 상대 병진 오차(%). 시작 키프레임 10개마다, 구간 길이 100,200,...,800 m(기준 궤적 주행 거리),
        E = (기준 상대 자세)^-1 (추정 상대 자세)의 병진 크기 / 구간 길이의 평균. 시퀀스가 짧아 해당 길이 구간이 없으면 그 길이는 빠진다.

실행(컨테이너 안, GPU 불필요):
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_kveval_perturb_drift46_20261120.py
출력: results/kv_eval_conditions_20261118/perturb/drift46.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _kveval_perturb_20261118 as K  # noqa: E402
import _report_model_common_20261116 as RC  # noqa: E402

OUT = K.OUT / "drift46.json"
LENGTHS = np.arange(100.0, 801.0, 100.0)
STEP = 10
CONDS = [n for n, c in K.COND.items() if c.params.get("family") in ("B1", "B2")]


def travel(p: np.ndarray) -> np.ndarray:
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(p[:, :3, 3], axis=0), axis=1))]


def trel(ref: np.ndarray, est: np.ndarray) -> float:
    """KITTI 방식 상대 병진 오차 평균(%)."""
    tr = travel(ref)
    inv = np.linalg.inv(ref)
    inv_e = np.linalg.inv(est)
    errs = []
    for i in range(0, len(ref), STEP):
        js = np.searchsorted(tr, tr[i] + LENGTHS)
        for L, j in zip(LENGTHS, js):
            if j >= len(ref):
                continue
            e = np.linalg.inv(inv[i] @ ref[j]) @ (inv_e[i] @ est[j])
            errs.append(np.linalg.norm(e[:3, 3]) / L)
    return float(100 * np.mean(errs)) if errs else float("nan")


def end_frac(ref: np.ndarray, est: np.ndarray) -> float:
    return float(np.linalg.norm(est[-1, :3, 3] - ref[-1, :3, 3]) / travel(ref)[-1])


def rotvec(r: np.ndarray) -> np.ndarray:
    """회전 행렬 묶음의 회전 벡터(축 x 각, rad)."""
    a = np.arccos(np.clip((np.trace(r, axis1=1, axis2=2) - 1) / 2, -1, 1))
    w = np.stack([r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]], 1)
    s = np.sin(a)
    return np.where(s[:, None] > 1e-6, w * (a / np.maximum(2 * s, 1e-12))[:, None], 0.0)


def body_rotation(odo: np.ndarray, gt: np.ndarray) -> tuple:
    """참값 몸체 좌표 = M @ 라이다 몸체 좌표 가 되게 하는 고정 회전 M(손-눈 보정을 회전만으로).
    키프레임 5개 간격 상대 자세의 병진(m)과 회전 벡터(rad x 10 m)를 함께 Kabsch로 맞춘다.
    병진만 쓰면 직진 축 둘레 회전이 정해지지 않으므로 회전 축(대부분 수직)을 함께 쓴다."""
    g = 5
    ro = np.linalg.inv(odo[:-g]) @ odo[g:]
    rg = np.linalg.inv(gt[:-g]) @ gt[g:]
    vo = np.r_[ro[:, :3, 3], 10 * rotvec(ro[:, :3, :3])]
    vg = np.r_[rg[:, :3, 3], 10 * rotvec(rg[:, :3, :3])]
    u, _, vt = np.linalg.svd(vg.T @ vo)
    d = np.sign(np.linalg.det(u @ vt))
    m = u @ np.diag([1, 1, d]) @ vt
    e = np.linalg.inv(rg[:, :3, :3]) @ (m @ ro[:, :3, :3] @ m.T)
    rot_err = np.degrees(np.arccos(np.clip((np.trace(e, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return m, float(np.median(rot_err))


def has_rotation(gt: np.ndarray) -> bool:
    """참값에 자세 회전이 들어 있는가(MCD 캐시의 poses_gt는 회전이 모두 단위 행렬이다)."""
    return bool(np.abs(gt[:, :3, :3] - np.eye(3)).max() > 1e-6)


def start_aligned(est: np.ndarray, gt: np.ndarray, metres: float = 100.0) -> np.ndarray:
    """위치만으로 맞춘 추정 위치: 참값 주행 첫 metres 구간의 위치로 강체 변환(Kabsch, 척도 없음)을 맞춘다."""
    k = int(np.searchsorted(travel(gt), metres)) + 1
    a, b = est[:k, :3, 3], gt[:k, :3, 3]
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((b - cb).T @ (a - ca))
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return (est[:, :3, 3] - ca) @ r.T + cb


def end_frac_pos(est: np.ndarray, gt: np.ndarray) -> float:
    """참값 비교용 끝점 드리프트: 첫 100 m 위치 정렬 뒤 끝점 위치 오차 / 참값 주행 거리(참값 자세 회전 불필요)."""
    p = start_aligned(est, gt)
    return float(np.linalg.norm(p[-1] - gt[-1, :3, 3]) / travel(gt)[-1])


def to_gt_frame(odo: np.ndarray, gt: np.ndarray, m: np.ndarray) -> np.ndarray:
    """라이다 궤적을 참값 몸체 좌표로 옮기고 시작 자세를 G_0에 맞춘다."""
    x = np.eye(4)
    x[:3, :3] = m.T
    o = odo @ x                                  # 몸체 회전 변경
    return gt[0] @ np.linalg.inv(o[0]) @ o       # 시작 자세 정렬


def stats(v) -> dict:
    v = np.asarray(v, float)
    q1, med, q3 = np.quantile(v, [.25, .5, .75])
    return dict(median=float(med), q1=float(q1), q3=float(q3), mean=float(v.mean()), min=float(v.min()), max=float(v.max()), n=int(len(v)))


def main() -> None:
    P = K.ctx()["P"]
    per = {c: {} for c in CONDS}
    kiss = {}
    for s in RC.SEQS:
        tup = P.inputs("test", s)
        cache = tup[0]
        odo = np.asarray(cache["poses"], np.float64)
        gt = np.asarray(cache["poses_gt"], np.float64)
        rot = has_rotation(gt)
        if rot:
            m, fit = body_rotation(odo, gt)
        kiss[s] = dict(n=len(odo), travel_gt_m=float(travel(gt)[-1]), travel_kiss_m=float(travel(odo)[-1]),
                       end_frac_pos=end_frac_pos(odo, gt),
                       trel_pct=trel(gt, to_gt_frame(odo, gt, m)) if rot else None,
                       body_fit_median_step_rot_err_deg=fit if rot else None)
        for c in CONDS:
            new, _ = K.COND[c].inputs(s, tup)
            pert = np.asarray(new[0]["poses"], np.float64)
            rec = dict(end_frac=end_frac(odo, pert), end_frac_pos=end_frac_pos(pert, odo), trel_pct=trel(odo, pert))
            # 기존 하네스 진단과 같은 정의인지 검산
            saved = json.loads((K.OUT / c / "inputs/test" / (s.replace("/", "_") + ".json")).read_text())["diag"]
            assert abs(rec["end_frac"] - saved["end_drift_frac_of_travel"]) < 1e-9, (c, s)
            rec.update(vs_gt_end_frac_pos=end_frac_pos(pert, gt),
                       vs_gt_trel_pct=trel(gt, to_gt_frame(pert, gt, m)) if rot else None)
            per[c][s] = rec
        print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kiss[s].items()}, flush=True)
    res = dict(
        definitions=dict(
            perturbation_vs_kiss="교란 궤적 대 교란 전 KISS-ICP 궤적(같은 몸체 좌표, 같은 시작 자세 P_0)",
            kiss_vs_gt="KISS-ICP 궤적 대 참값(cache poses_gt). trel은 몸체 회전 M을 손-눈 방식(상대 병진+회전 벡터 Kabsch)으로 맞추고 외부 보정 병진은 무시",
            perturbed_vs_gt="교란 궤적 대 참값(kiss_vs_gt와 같은 정렬). 교란과 KISS-ICP 자체 오차가 함께 들어간다",
            end_frac="끝점 위치 오차 / 기준 궤적 주행 거리(마지막 키프레임), 시작 자세를 그대로 맞춤. 교란 대 KISS-ICP에만 정의된다. "
                     "기존 inputs 진단 end_drift_frac_of_travel(kitti/00 예비 값 11%의 정의)과 같다(검산 assert 통과)",
            end_frac_pos="끝점 위치 오차 / 기준 궤적 주행 거리, 단 기준 궤적 주행 첫 100 m의 위치로 강체 정렬(Kabsch, 척도 없음)한 뒤. "
                         "참값 자세 회전이 필요 없어 46시퀀스 모두에 같은 정의로 KISS-ICP 대 참값과 교란 대 KISS-ICP를 잰다",
            trel_pct="KITTI 오도메트리 방식 상대 병진 오차(%): 시작 키프레임 10개마다, 구간 100~800 m(100 m 간격, 기준 궤적 주행 거리)의 평균. "
                     "참값 비교는 참값에 자세 회전이 있는 시퀀스만(MCD 12개는 poses_gt 회전이 단위 행렬이라 None)",
            keyframes="키프레임은 캐시 생성 때 교란 전 KISS-ICP 오도메트리로 고른 그대로(교란 뒤 다시 고르지 않음). 정답 위치는 참값"),
        sequences=list(RC.SEQS),
        kiss_icp=dict(per_sequence=kiss, summary={k: stats([kiss[s][k] for s in RC.SEQS if kiss[s][k] is not None])
                                                  for k in ("end_frac_pos", "trel_pct")}),
        conditions={c: dict(params=K.COND[c].params, per_sequence=per[c],
                            summary={k: stats([per[c][s][k] for s in RC.SEQS if per[c][s][k] is not None])
                                     for k in ("end_frac", "end_frac_pos", "trel_pct", "vs_gt_end_frac_pos", "vs_gt_trel_pct")})
                    for c in CONDS})
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
