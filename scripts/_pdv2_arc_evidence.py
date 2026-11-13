#!/usr/bin/env python3
"""같은 관측 예산에서 프레임 수와 주행 거리로 시간 쌍을 정렬한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_evidence as exp
from _pdv2_no_context_ablation import sha256, write_json
from retrieval.spectral_evidence import correlation_profiles

OUT = REPO / "results/pdv2_arc_evidence_20261018"
METHODS = ("frame5", "candidate_frame10", "arc_match10")


def arc_neighbor(path: np.ndarray, center: np.ndarray, signed_distance: np.ndarray,
                 radius: int = 10) -> np.ndarray:
    """주어진 거리와 가장 가까운 과거/미래 이웃을 제한 창에서 고른다."""
    center = np.asarray(center, dtype=np.int64)
    target = path[center]+signed_distance
    left = np.maximum(0, center-radius)
    right = np.minimum(len(path)-1, center+radius)
    lower = np.clip(np.searchsorted(path, target, side="left")-1, left, right)
    upper = np.clip(lower+1, left, right)
    return np.where(np.abs(path[lower]-target) <= np.abs(path[upper]-target), lower, upper)


def scores(bank: tuple, travel: np.ndarray, q: np.ndarray, j: np.ndarray,
           channel: str, method: str) -> tuple:
    """현재와 앞뒤 두 쌍의 피크 평균을 정/역방향에서 비교한다."""
    center = correlation_profiles(bank, q, j, channel).max(1)
    choices, errors = [], []
    for direction in (1, -1):
        total, count = center.copy(), np.ones(len(q))
        for step in (-5, 5):
            qn = q+step
            valid = (qn >= 0) & (qn < len(travel))
            qsafe = np.clip(qn, 0, len(travel)-1)
            distance = travel[qsafe]-travel[q]
            if method == "arc_match10":
                jn = arc_neighbor(travel, j, direction*distance)
                valid &= jn != j
            else:
                jn = j+direction*step*(2 if method == "candidate_frame10" else 1)
                valid &= (jn >= 0) & (jn < len(travel))
            total[valid] += correlation_profiles(bank, qn[valid], jn[valid], channel).max(1)
            count[valid] += 1
            errors.extend(np.abs(np.abs(travel[jn[valid]]-travel[j[valid]])-np.abs(distance[valid])).tolist())
        choices.append(total/count)
    return np.maximum(*choices), np.asarray(errors)


def main() -> None:
    """미리 정한 세 대응 규칙의 최종 NSD 지표를 모두 보고한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    spec = exp.protocol(True)
    write_json(OUT / "protocol.json", {"script_sha256": sha256(Path(__file__)),
        "input_protocol_sha256": sha256(exp.OUT / "protocol.json"), "test_used": False,
        "methods": METHODS, "scope": "시간25%혼합개선뒤후속물리거리대조. 질의±5고정, 후보±5/±10/질의이동거리와가까운±10안후보. 혼합25%고정. KISS-ICP 누적이동거리만사용. 후보중심j<=q-30, 참조j+10<=q-20, 추가관측지연없음. 시간모델과큰투영3시드. 시험미사용."})
    rows, error_rows = {}, {}
    for sensor, seq in exp.old.base.VAL:
        sequence = sensor+"/"+seq
        _, _, paths, _ = exp.old.base.inputs("val", sequence)
        cache = exp.arrays(paths["cache"])
        pos = cache["poses"][:, :3, 3]
        travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
        with np.load(paths["spec"]) as z:
            bank = exp.old.base.bank_for(exp.OP, z)
        result, error_rows[sequence] = {}, {}
        for seed in (1, 2, 3):
            channel = exp.read(exp.old.base.BASE / f"s{seed}/confidence_selection.json")["ops"][exp.OP]["decision"]
            result[str(seed)], error_rows[sequence][str(seed)] = {}, {}
            for model in spec["config"]["models"]:
                a = exp.arrays(exp.OUT / f"s{seed}/{model}_{sensor}_{seq}.npz")
                q, j = a["query_idx"], a["j_star"]
                result[str(seed)][model], error_rows[sequence][str(seed)][model] = {}, {}
                output = {k: a[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")}
                for method in METHODS:
                    chunks, errors = [], []
                    for start in range(0, len(q), 512):
                        sl = slice(start, start+512)
                        value, error = scores(bank, travel, q[sl], j[sl], channel, method)
                        chunks.append(value)
                        errors.extend(error)
                    temporal = np.concatenate(chunks)
                    if method == "frame5":
                        np.testing.assert_allclose(temporal, a["temporal_independent"], atol=1e-12)
                    output[method] = .75*a["raw"]+.25*temporal
                    result[str(seed)][model][method] = exp.metrics(output[method], a)
                    error_rows[sequence][str(seed)][model][method] = np.quantile(errors, [.1, .5, .9]).tolist()
                path = OUT / f"s{seed}/{model}_{sensor}_{seq}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **output)
        rows[sequence] = result
        write_json(OUT / f"{sensor}_{seq}.json", {"results": result, "test_used": False})
        print("ARC", sequence, flush=True)
    values = {str(seed): {model: {} for model in spec["config"]["models"]} for seed in (1, 2, 3)}
    for seed in values:
        for model in values[seed]:
            for method in METHODS:
                families = []
                for fam in ("kitti", "helipr", "mulran"):
                    selected = [r[seed][model][method] for seq, r in rows.items() if seq.startswith(fam) and r[seed][model][method]["n_pos"]]
                    families.append({m: float(np.mean([r[m] for r in selected])) for m in exp.METRICS})
                values[seed][model][method] = {m: float(np.mean([r[m] for r in families])) for m in exp.METRICS}
    means = {model: {method: {m: float(np.mean([values[s][model][method][m] for s in values])) for m in exp.METRICS} for method in METHODS} for model in values["1"]}
    write_json(OUT / "summary.json", {"per_seed": values, "mean": means, "distance_mismatch_quantiles": error_rows, "test_used": False})
    write_json(OUT / "status.json", {"status": "complete", "test_used": False})
    print("ARC_SUMMARY", means, flush=True)


if __name__ == "__main__":
    main()
