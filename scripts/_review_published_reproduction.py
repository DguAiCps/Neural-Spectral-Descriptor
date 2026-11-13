#!/usr/bin/env python3
"""공개 정답·공식 평가를 재검산하고 논문 수치와 대조한다."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_published_reproduction_20261106"
NATIVE = ROOT / "results/pdv2_review_native_20261105"
SOURCES = {}


def track(path: Path) -> Path:
    """읽은 파일의 해시를 보존한다."""
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path: Path) -> dict:
    """JSON과 원천 해시를 읽는다."""
    return json.loads(track(path).read_text())


def write(name: str, value: dict) -> None:
    """JSON 결과를 기록한다."""
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def seqot() -> dict:
    """공개 자세로 15m 정답 전체를 재생성하고 최종 예측을 재채점한다."""
    a = np.load(track(OUT / "sources/seqot_poses_database.npy"))
    b = np.load(track(OUT / "sources/seqot_poses_query.npy"))
    gt = np.load(track(ROOT / "_external/seqot_official_eval/gt_120108_120205.npy"), allow_pickle=True)
    assert a.shape == (28127, 4, 4) and b.shape == (28239, 4, 4)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    source = ROOT / "_external/SeqOT/data_prepararion/gen_ground_truth.py"
    track(source)
    raw = cKDTree(a[:, :3, 3]).query_ball_point(b[::5, :3, 3], r=15, return_sorted=True)
    rebuilt = [np.array(ids, np.int64)[np.linalg.norm(a[ids, :3, 3] - b[i * 5, :3, 3], axis=1) < 15]
               for i, ids in enumerate(raw)]
    different = [i for i in range(len(gt)) if not np.array_equal(np.asarray(gt[i], np.int64), rebuilt[i])]
    without_same_index = [ids[ids != i * 5] for i, ids in enumerate(rebuilt)]
    same_after_exclusion = all(np.array_equal(np.asarray(gt[i], np.int64), ids)
                               for i, ids in enumerate(without_same_index))
    assert same_after_exclusion
    prediction = np.load(track(NATIVE / "seqot/author/native/predicted_L2_dis.npz"))["arr_0"].reshape(-1, 22, 3)
    assert np.array_equal(prediction[:, 0, 0], np.arange(0, len(b), 5))
    order = np.argsort(prediction[:, :, 2], axis=1)
    indices = np.take_along_axis(prediction[:, :, 1], order, axis=1).astype(np.int64)
    result = {"same_as_released_array": not different, "queries": len(gt),
              "different_query_rows": len(different), "different_query_ids_first20": [i * 5 for i in different[:20]],
              "total_rebuilt_positive_pairs": sum(len(x) for x in rebuilt),
              "released_positive_pairs": sum(len(x) for x in gt),
              "same_after_excluding_equal_raw_index": same_after_exclusion,
              "equal_index_exclusion_is_not_same_scan": "서로 다른 날짜의 지도·질의다. 동일 원시 프레임 번호는 동일 스캔을 뜻하지 않는다. 제외 이유는 저자 근거가 확인되지 않았다.",
              "rule": "공개 자세의 3D거리<15m, 모든 원시 지도 프레임, 매5번째 질의",
              "author_instruction": "https://github.com/BIT-MJY/SeqOT/issues/1#issuecomment-1342137823",
              "evaluation": {}}
    original = read(NATIVE / "seqot/author/native/report.json")
    for label, truth in [("released", gt), ("rebuilt_15m", rebuilt)]:
        positive = [i for i, ids in enumerate(truth) if np.asarray(ids).any()]
        counts = {str(k): sum(bool(np.isin(indices[i, :k], truth[i]).any()) for i in positive)
                  for k in (1, 5, 20)}
        metrics = {k: v / len(positive) for k, v in counts.items()}
        result["evaluation"][label] = {"positive_queries": len(positive), "hits": counts, "recall": metrics}
        if label == "released":
            assert counts == original["hits"] and metrics == original["official_recall"]
    result["paper_reference"] = original["published"]
    result["paper_delta_pp"] = {str(k): 100 * (result["evaluation"]["released"]["recall"][str(k)] - original["published"][f"r{k}"])
                                 for k in (1, 5, 20)}
    result["interpretation"] = "공개 배열은3D15m 기준에서 동일 원시 프레임 번호 쌍을 뺀 것과 전체 일치한다. 현재 공개 생성기에는 이 제외가 없으며 논문 overlap>0.3와 같다는 증명은 아니다."
    write("seqot_label_reconstruction.json", result)
    return result


def imlpr() -> dict:
    """원시 자세로 정답을 재구성하고 공개 질의 평가 함수를 다시 실행한다."""
    source = track(ROOT / "_external/ImLPR/eval/evaluate.py")
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_process_query")
    namespace = {"np": np, "KDTree": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    official = namespace["_process_query"]
    rows = []
    for sequence in ("DCC01", "DCC02", "DCC03"):
        with np.load(track(NATIVE / f"imlpr/inputs/{sequence}/index.npz")) as z:
            positions, timestamps = z["poses"][:, :2, 3], z["timestamps"]
        n = len(timestamps)
        neighbors = cKDTree(positions).query_ball_point(positions, r=10)
        # 원래 pickle을 재사용하지 않고 공식 XY10m 정답을 독립 재구성한다.
        metadata = [{"timestamp": int(timestamps[i]), 0: list(map(int, neighbors[i]))} for i in range(n)]
        seconds = timestamps.astype(np.float64) / 1e9
        positive = np.array([any(seconds[j] < seconds[i] - 60 for j in neighbors[i]) for i in range(n)])
        eligible = positive & (seconds - seconds.min() >= 90)
        for k in (15, 25):
            path = NATIVE / f"imlpr/k{k}/author/native_batch1_{sequence}.npz"
            previous = read(path.with_suffix(".json"))
            with np.load(track(path)) as z:
                distances, indices = z["distances"], z["indices"]
                old_cells = z["tp_fp_tn_fn"]
            cells, cumulative = np.zeros((4, 1000)), np.zeros(30)
            valid, direct_hits, excluded = 0, 0, 0
            thresholds = np.linspace(0, 1, 1000)
            for i in range(n):
                class CachedTree:
                    def query(self, vector, k):
                        assert k == 200
                        return distances[i:i + 1], indices[i:i + 1]
                output = official(i, np.zeros(1), metadata[i], CachedTree(), 30,
                                  max(int(round(n / 1000)), 1), None, thresholds, 0,
                                  database_meta=metadata, is_intra=True, init_time=float(seconds.min()))
                if output is None:
                    excluded += int(eligible[i])
                    continue
                recall, _, tp, fp, tn, fn, count = output
                cumulative += recall
                cells += np.array([tp, fp, tn, fn])
                valid += count
                allowed = indices[i][seconds[indices[i]] < seconds[i] - 60]
                direct_hits += int(np.linalg.norm(positions[i] - positions[allowed[0]]) <= 10)
            assert np.array_equal(cells, old_cells), (sequence, k)
            recall_curve = np.cumsum(cumulative) / valid
            assert direct_hits == int(cumulative[0])
            precision = cells[0] / (cells[0] + cells[1] + 1e-8)
            recall = cells[0] / (cells[0] + cells[3] + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            values = {"r1": float(recall_curve[0]), "r5": float(recall_curve[4]), "f1": float(f1.max())}
            assert abs(values["r1"] - previous["r1"]) < 1e-12
            assert abs(values["f1"] - previous["official_max_f1"]) < 1e-12
            rows.append({"sequence": sequence, "k": k, "frames": n, "positive_queries": int(eligible.sum()),
                         "evaluated_queries": valid, "top200_filter_excluded_positives": excluded,
                         "metrics": values, "published": previous["published"],
                         "delta_pp": {metric: 100 * (values[metric] - previous["published"][metric]) for metric in ("r1", "f1")},
                         "official_replay_passed": True})
    result = {"rows": rows, "author_weights_only": True, "no_new_inference_or_training": True,
              "input_limitation": "공식 가공자료 다운로드 한도 초과. 재생성4275장 대 논문4328장.",
              "macro": {str(k): {m: float(np.mean([r["metrics"][m] for r in rows if r["k"] == k]))
                                  for m in ("r1", "r5", "f1")} for k in (15, 25)}}
    write("imlpr_official_replay.json", result)
    return result


def ring() -> dict:
    """논문에 명시한10m와 공식 기본 영역·전체 영역을 모두 검산한다."""
    sets = read(NATIVE / "evaluation_sets.json")
    reference = read(NATIVE / "protocol.json")["paper_table_II_nclt_reference_only"]
    results = []
    for folder in (NATIVE, NATIVE / "maxfix"):
        summary = read(folder / "summary.json")
        for region, groups in sets.items():
            q = np.array([x["xy"] for x in groups["query"]])
            m = np.array([x["xy"] for x in groups["map"]])
            geo = np.linalg.norm(q[:, None] - m[None], axis=2)
            for model in ("ring", "ringpp"):
                scores = np.load(track(folder / f"pair_dists_{region}_{model}.npy"))
                selected = np.argmin(scores, axis=1)
                good = geo[np.arange(len(q)), selected] <= 10
                positive = geo.min(axis=1) <= 10
                r1, positive_r1 = float(good.mean()), float(good[positive].mean())
                existing = summary["variants"][f"{region}_{model}"]["10.0"]
                assert r1 == existing["official_R@1"] and positive_r1 == existing["positive_only_R@1"]
                results.append({"model": model, "variant": "released" if folder == NATIVE else "exactmax",
                                "region": region, "map": len(m), "queries": len(q), "positive_queries": int(positive.sum()),
                                "R@1": r1, "positive_only_R@1": positive_r1, "reference_R@1": reference[model]["R@1"],
                                "delta_to_reference_pp": 100 * (r1 - reference[model]["R@1"])})
    result = {"rows": results, "radius_basis": "논문VII-B3은 지도20m·질의5m·재방문10m를 명시하며 표III의 NCLT RING++ 세 값은 표II와 같다. 정확 입력 범위·목록은 여전히 미확인.",
              "not_adopted": "논문 값에 맞는 반경·영역을 선택하지 않는다. 공개 기본 test와 전체 궤적을 모두 보고한다."}
    write("ring_published_scope.json", result)
    return result


def main() -> None:
    """선택 없는 재현 검사를 수행한다."""
    OUT.mkdir(exist_ok=True)
    results = {"seqot": seqot(), "imlpr": imlpr(), "ring": ring(),
               "helios": read(NATIVE / "helios/author_full_published_comparison.json")}
    results["sources_sha256"] = SOURCES
    results["passed"] = True
    write("verified_published_comparison.json", results)
    print(json.dumps({"passed": True, "seqot_label_match": results["seqot"]["same_as_released_array"],
                      "seqot_paper_delta_pp": results["seqot"]["paper_delta_pp"],
                      "imlpr_macro": results["imlpr"]["macro"], "sources": len(SOURCES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
