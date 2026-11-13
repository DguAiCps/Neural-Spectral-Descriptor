#!/usr/bin/env python3
"""저자 HeLiPR Town 평가 pickle이 공개 생성기와 같은 정답·시각 단위·3m 표본인지 확인한다.

MulRan 저자 입력이 압축본에 없어서, 같은 공개 생성기로 만든 재생성 MulRan 정답의 타당성을
간접 확인하는 보조 진단이다. 성능 평가나 설정 선택에 쓰지 않는다. CPU만 쓴다.

    docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
        python3 scripts/_review_imlpr_author_town_check.py --attempt 01
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _review_imlpr_author_eval import OFFICIAL, OUT, RUN, safe_load, sha, write  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attempt", required=True)
    a = ap.parse_args()
    data = OUT / "author_data" / f"attempt_{a.attempt}" / "files/data"
    ar = RUN / f"attempt_{a.attempt}"
    db, q = safe_load(data / "db_helipr_town.pickle"), safe_load(data / "query_helipr_town.pickle")
    folders = sorted(p.name for p in (data / "validation").iterdir() if p.name.startswith("Town"))
    work = ar / "town_check" / "work"
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL / "preprocess"))
    import generate_test_sets_img as G
    cwd = os.getcwd()
    os.chdir(work)
    try:
        G.construct_query_and_database_sets(str(data / "validation") + "/", "", folders, "poses.txt")
    finally:
        os.chdir(cwd)
    gdb = pickle.load((ar / "town_check/data/db_img.pickle").open("rb"))
    gq = pickle.load((ar / "town_check/data/query_img.pickle").open("rb"))
    rows = []
    for i, folder in enumerate(folders):
        a_db, a_q = list(db[i].values()), list(q[i].values())
        g_db, g_q = list(gdb[i].values()), list(gq[i].values())
        same_paths = [x["query"] for x in a_db] == [x["query"] for x in g_db]
        ts = np.array([float(x["timestamp"]) for x in a_db])
        xy = np.array([[x["northing"], x["easting"]] for x in a_db])
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        pair = {}
        for j in range(len(folders)):
            raw = sum(sorted(x.get(j, [])) == sorted(y.get(j, [])) for x, y in zip(a_q, g_q))
            tj = np.array([float(x["timestamp"]) for x in db[j].values()])
            eff = 0
            for x, y in zip(a_q, g_q):
                cut = float(x["timestamp"]) / 1e9 - 60
                fa = sorted(k for k in x.get(j, []) if tj[k] / 1e9 < cut) if i == j else sorted(x.get(j, []))
                fb = sorted(k for k in y.get(j, []) if tj[k] / 1e9 < cut) if i == j else sorted(y.get(j, []))
                eff += fa == fb
            pair[folders[j]] = {"raw_equal": int(raw), "effective_equal": int(eff), "n": len(a_q)}
        rows.append({"folder": folder, "n": len(a_db), "same_paths_as_generator": same_paths,
                     "timestamp_unit": "ns" if ts.min() > 1e17 else "other", "timestamp_min": float(ts.min()),
                     "positives_vs_generator": pair,
                     "consecutive_step_m": {"min": float(step.min()), "median": float(np.median(step)),
                                            "below_3m": int((step < 3.0).sum()), "below_2_9m": int((step < 2.9).sum())},
                     "db_equals_query_entries": all({k: v for k, v in x.items() if not isinstance(k, int)} ==
                                                    {k: v for k, v in y.items() if not isinstance(k, int)} for x, y in zip(a_db, a_q))})
    out = {"purpose": "저자 Town 평가 pickle 대 공개 generate_test_sets_img(저자 poses.txt 입력) 정답 비교",
           "inputs_sha256": {"db_helipr_town.pickle": sha(data / "db_helipr_town.pickle"),
                             "query_helipr_town.pickle": sha(data / "query_helipr_town.pickle"),
                             "generate_test_sets_img.py": sha(OFFICIAL / "preprocess/generate_test_sets_img.py")},
           "folders": folders, "rows": rows,
           "note": "같은 세션 정답은 평가 함수의 60초 필터 뒤(effective)로도 비교한다. 생성기의 60.0 필터는 ns 시각에서는 사실상 걸리지 않는다"}
    write(ar / "town_check.json", out)
    print(json.dumps(out, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
