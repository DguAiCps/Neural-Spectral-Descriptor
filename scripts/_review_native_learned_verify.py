#!/usr/bin/env python3
"""학습형 native 예측 원장을 공식 함수 없이 독립 재계산한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_seqot() -> list:
    root = OUT / "seqot"
    protocol = json.loads((root / "protocol.json").read_text())
    truth = np.load(REPO / protocol["ground_truth"], allow_pickle=True)
    rows = []
    for report_path in root.glob("*/native/report.json"):
        report = json.loads(report_path.read_text())
        path = report_path.parent / "predicted_L2_dis.npz"
        pred = np.load(path)["arr_0"].reshape(-1, 22, 3)
        assert sha(path) == report["prediction_sha256"]
        positive = np.asarray([np.asarray(g).any() for g in truth])
        ids = pred[:, :, 1].astype(np.int64)
        hit = {}
        for k in (1, 5, 20):
            hit[str(k)] = sum(bool(set(ids[i, :k]) & set(truth[i])) for i in range(len(truth)) if positive[i])
            assert hit[str(k)] == report["hits"][str(k)]
            assert hit[str(k)] / positive.sum() == report["official_recall"][str(k)]
        rows.append({"model": report["model"], "positive": int(positive.sum()), "hits": hit, "prediction_sha256": sha(path), "passed": True})
    return rows


def independent_pair(indices: np.ndarray, distances: np.ndarray, query: dict,
                     database: dict, db_key: int, intra: bool, past_s: int,
                     timestamps_in_metadata: bool) -> tuple:
    def timestamp(row):
        return float(row["timestamp"] if timestamps_in_metadata else Path(row["query"]).stem) / 1e9
    db_times = np.asarray([timestamp(database[i]) for i in range(len(database))])
    evaluated = np.zeros(len(query), bool)
    good = np.zeros(len(query), bool)
    hits = np.zeros(30, np.int64)
    confusion = np.zeros((4, 1000), np.int64)
    levels = np.linspace(0, 1, 1000)
    fixed_denominator = 0
    for q in range(len(query)):
        correct = np.asarray(query[q].get(db_key, []), np.int64)
        candidate = indices[q]
        distance = distances[q]
        if intra:
            time = timestamp(query[q])
            if time - db_times.min() < 90:
                continue
            correct = correct[db_times[correct] < time - past_s]
            old = db_times[candidate] < time - past_s
            candidate, distance = candidate[old], distance[old]
        if not len(correct):
            continue
        fixed_denominator += 1
        if not len(candidate):
            continue
        evaluated[q] = True
        rank_hit = np.isin(candidate[:30], correct)
        matched = np.flatnonzero(rank_hit)
        if len(matched):
            hits[matched[0]:] += 1
        good[q] = rank_hit[0]
        accepted = distance[0] < levels
        confusion[0] += accepted & good[q]
        confusion[1] += accepted & ~good[q]
        confusion[2] += ~accepted & ~good[q]
        confusion[3] += ~accepted & good[q]
    return evaluated, good, hits, confusion, fixed_denominator


def check_helios() -> list:
    root = OUT / "helios"
    queries = pickle.loads((root / "data/query.pickle").read_bytes())
    databases = pickle.loads((root / "data/database.pickle").read_bytes())
    protocol = json.loads((root / "protocol.json").read_text())
    rows = []
    for model in ("author", "v2_s1", "v2_s2", "v2_s3"):
        report_path = root / model / "native_report.json"
        if not report_path.exists():
            report_path = root / model / "native_partial_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        for pair in report["pairs"]:
            db, name = pair["database"], pair["query"]
            path = root / model / "native_pairs" / f"{db}__{name}.npz"
            z = np.load(path)
            assert sha(path) == pair["prediction_sha256"]
            db_key = protocol["database"].index(db)
            q = queries[protocol["query"].index(name)]
            evaluated, good, hits, cells, fixed = independent_pair(z["indices"], z["distances"], q, databases[db_key], db_key, db == name, 30, False)
            np.testing.assert_array_equal(evaluated, z["evaluated"])
            np.testing.assert_array_equal(good, z["first_correct"])
            np.testing.assert_array_equal(cells, z["tp_fp_tn_fn"])
            np.testing.assert_array_equal(hits, z["recall_hits"])
            assert int(evaluated.sum()) == pair["positive_evaluated"]
            assert hits[0] / evaluated.sum() == pair["r1"]
            assert hits[4] / evaluated.sum() == pair["r5"]
            rows.append({"model": model, "database": db, "query": name, "positive": int(evaluated.sum()), "fixed_positive": fixed, "passed": True})
    return rows


def check_imlpr() -> list:
    root = OUT / "imlpr"
    rows = []
    for path in root.glob("k*/*/native*DCC*.json"):
        report = json.loads(path.read_text())
        seq = report["sequence"]
        meta = pickle.loads((root / "inputs" / seq / "metadata.pickle").read_bytes())
        z = np.load(path.with_suffix(".npz"))
        evaluated, good, hits, cells, fixed = independent_pair(z["indices"], z["distances"], meta, meta, 0, True, 60, True)
        np.testing.assert_array_equal(cells, z["tp_fp_tn_fn"])
        assert int(evaluated.sum()) == report["official_evaluated_queries"]
        assert fixed == report["fixed_positive_denominator"]
        assert int(hits[0]) == report["hits_at1"]
        assert hits[0] / evaluated.sum() == report["r1"]
        assert hits[4] / evaluated.sum() == report["r5"]
        p = cells[0] / (cells[0] + cells[1] + 1e-8)
        r = cells[0] / (cells[0] + cells[3] + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        assert float(f1.max()) == report["official_max_f1"]
        rows.append({"model": report["model"], "sequence": seq, "k": report["k"], "inference_batch": report.get("inference_batch", 4), "positive": int(evaluated.sum()), "passed": True})
    return rows


def main() -> None:
    report = {"seqot": check_seqot(), "helios": check_helios(), "imlpr": check_imlpr(),
              "scope": "완료한 원장을 공식 평가함수 없이 재계산; 없는 결과는 완료로 간주하지 않음"}
    (OUT / "learned_independent_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print({k: len(v) for k, v in report.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
