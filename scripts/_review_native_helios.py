#!/usr/bin/env python3
"""저자 공개 Roundabout4센서 자료로 HeLiOS native 평가를 수행한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tarfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105/helios"
OFFICIAL = REPO / "_external/HeLiOS"
NAMES = [f"Roundabout{n:02d}-{sensor}" for n in (1, 2, 3) for sensor in ("Aeva", "Avia", "Ouster", "Velodyne")]
DATABASES = ["Roundabout01-Aeva", "Roundabout01-Ouster"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def protocol() -> None:
    """공식 소스·모델·두 지도·12질의 세션을 결과 전에 고정한다."""
    inventory = json.loads((REPO / "results/pdv2_review_selection_full_20261105/retained_source_inventory.json").read_text())
    models = {"author": {"path": "_external/HeLiOS/weights/HeLiOS.pth",
                          "sha256": sha(OFFICIAL / "weights/HeLiOS.pth")}}
    for s in (1, 2, 3):
        p = inventory["methods"]["helios"]["seeds"][str(s)]["original_choice"]
        models[f"v2_s{s}"] = {"path": p["path"].replace("/workspace/Neural-Spectral-Codec/", ""), "sha256": p["sha256"]}
    sources = ["eval/evaluate.py", "datasets/pnv_raw.py", "datasets/base_datasets.py",
               "datasets/preprocess/generate_test_sets.py", "config/model_helios.txt", "models/helios.py"]
    d = {"official_repo": "https://github.com/minwoo0611/HeLiOS",
         "paper": "https://minwoo0611.github.io/assets/pdf/icra2025-helios.pdf",
         "source_sha256": {p: sha(OFFICIAL / p) for p in sources}, "models": models,
         "database": DATABASES, "query": NAMES, "threshold_overlap": 0.5,
         "input": "저자 공개 5m/8192점 float32 점군과 공개 overlap matrix를 그대로 사용",
         "metadata_column_alias": "공개CSV x/y를 공식생성기의 northing/easting으로 이름만 연결",
         "protocol": "공식 process_query: 200NN, intra 시작90초·과거30초; 양성질의만; L2",
         "native_metric": "쌍별 R@1/R@5 -> Narrow(Aeva,Avia)6쌍/Wide(Ouster,Velodyne)6쌍 평균",
         "published_table_I": {"Roundabout01-Ouster": {"Narrow": [.700, .852], "Wide": [.912, .946]},
                                "Roundabout01-Aeva": {"Narrow": [.806, .885], "Wide": [.849, .940]}},
         "execution_only": "공식 evaluate_dataset의 query_embeddings>4 break를 호출하지 않고 공식 추론/process_query를12세션 각각 실행; batch 1, score 함수 그대로",
         "selection": "기존v2검증선택본·저자공개본 고정; native시험으로 선택 안 함",
         "contamination": "저자학습 DCC/KAIST/Riverside는 우리 대전 시험과 장소 겹침. 이 native Roundabout 진단은 주표 가중치 교체에 사용 안 함"}
    dest = OUT / "protocol.json"
    if dest.exists():
        assert json.loads(dest.read_text()) == d
    else:
        write(dest, d)


def check() -> dict:
    d = json.loads((OUT / "protocol.json").read_text())
    for p, h in d["source_sha256"].items():
        assert sha(OFFICIAL / p) == h, p
    return d


def prepare() -> None:
    """저자 파일을 안전하게 풀고 공식 순서의 overlap 정답을 만든다."""
    import pandas as pd
    check()
    source = Path(os.environ.get("HELIOS_NATIVE_SOURCE", "/mnt/d/NSD_datasets/_review_native_20261105/helios"))
    data = OUT / "data"
    records, paths, all_sets = {}, {}, []
    for name in NAMES:
        src = source / "data_validation" / name
        csv = src / "trajectory.csv"
        df = pd.read_csv(csv, dtype={"timestamp": str})
        target = data / "validation" / name
        target.mkdir(parents=True, exist_ok=True)
        archive = src / "LiDAR.tar.gz"
        available = archive.with_suffix(".gz.complete").exists()
        if available and not (target / "extracted.json").exists():
            with tarfile.open(archive) as stream:
                for member in stream:
                    assert member.name == "LiDAR" or member.name.startswith("LiDAR/"), member.name
                    assert ".." not in Path(member.name).parts and not member.issym() and not member.islnk()
                    if member.isfile():
                        dest = target / member.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with dest.open("wb") as f:
                            f.write(stream.extractfile(member).read())
            write(target / "extracted.json", {"archive_sha256": sha(archive)})
        dataset = {}
        hashes = []
        for i, row in df.iterrows():
            rel = f"validation/{name}/LiDAR/{row['timestamp']}.bin"
            path = data / rel
            if available:
                a = np.fromfile(path, np.float32).reshape(-1, 4)
                assert a.shape == (8192, 4) and np.isfinite(a).all()
            dataset[int(i)] = {"query": rel, "northing": float(row["x"]), "easting": float(row["y"])}
            if available:
                hashes.append(sha(path))
        all_sets.append(dataset)
        records[name] = {"n_frames": len(df), "csv_sha256": sha(csv), "point_sha256": hashes,
                         "available": available,
                         "archive_sha256": json.loads((target / "extracted.json").read_text())["archive_sha256"] if available else None}
        print("INPUT", name, len(df), flush=True)
    overlap_path = source / "data_overlap/overlap_matrix_validation_Roundabout.txt"
    overlap = np.loadtxt(overlap_path, dtype=np.float32)
    offsets = np.r_[0, np.cumsum([len(x) for x in all_sets])]
    assert overlap.shape == (offsets[-1], offsets[-1]), (overlap.shape, offsets)
    assert np.isfinite(overlap).all() and overlap.min() >= 0 and overlap.max() <= 1
    database_sets = [all_sets[NAMES.index(name)] for name in DATABASES]
    for q, query in enumerate(all_sets):
        for db, name in enumerate(DATABASES):
            k = NAMES.index(name)
            gt = overlap[offsets[q]:offsets[q + 1], offsets[k]:offsets[k + 1]] > .5
            for i in query:
                query[i][db] = np.flatnonzero(gt[i]).tolist()
    for name, obj in [("database", database_sets), ("query", all_sets)]:
        with (data / f"{name}.pickle").open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    write(OUT / "input_manifest.json", {"sequences": records, "overlap_sha256": sha(overlap_path),
          "overlap_shape": list(overlap.shape), "folders_order": NAMES,
          "database_pickle_sha256": sha(data / "database.pickle"), "query_pickle_sha256": sha(data / "query.pickle")})


def official():
    """경로만 연결하고 공식 추론·평가 모듈을 읽는다."""
    sys.path[:0] = [str(REPO / "scripts"), str(OFFICIAL), str(OFFICIAL / "docker/thirdparty/SparseTransformer")]
    from _helios_official_train import official_path
    official_path()
    import eval.evaluate as E
    from misc.utils import TrainingParams
    E.DATASET_FOLDER = str(OUT / "data")
    params = TrainingParams(str(OFFICIAL / "config/config_baseline_helios.txt"), str(OFFICIAL / "config/model_helios.txt"))
    return E, params


def infer(model_name: str, sequence: str, smoke: bool) -> None:
    """공식 get_latent_vectors를256개 단위로 실행해 재개점을 남긴다."""
    import torch
    d = check()
    E, params = official()
    p = REPO / d["models"][model_name]["path"]
    assert sha(p) == d["models"][model_name]["sha256"]
    model = E.model_factory(params.model_params)
    model.load_state_dict(torch.load(p, map_location="cpu", weights_only=False), strict=True)
    model.cuda().eval()
    with (OUT / "data/query.pickle").open("rb") as f:
        dataset = pickle.load(f)[NAMES.index(sequence)]
    chosen = np.unique(np.linspace(0, len(dataset) - 1, 8, dtype=int)) if smoke else np.arange(len(dataset))
    root = OUT / model_name / ("smoke" if smoke else "desc") / sequence
    root.mkdir(parents=True, exist_ok=True)
    if (root / "done.json").exists():
        return
    start = time.time()
    for first in range(0, len(chosen), 256):
        target = root / f"shard_{first:06d}.npz"
        if target.exists():
            continue
        indices = chosen[first:first + 256]
        subset = {int(i): dataset[int(i)] for i in indices}
        desc = E.get_latent_vectors(model, subset, "cuda", params)
        assert desc.shape == (len(indices), 8448) and np.isfinite(desc).all()
        assert np.allclose(np.linalg.norm(desc, axis=1), 1, atol=1e-5)
        tmp = target.with_name(target.stem + ".tmp.npz")
        np.savez_compressed(tmp, indices=indices, desc=desc)
        tmp.replace(target)
        status = {"model": model_name, "sequence": sequence, "frames": first + len(indices), "total": len(chosen),
                  "elapsed_s": time.time() - start, "pid": os.getpid(), "gpu": torch.cuda.get_device_name()}
        write(root / "status.json", status)
        print(json.dumps(status), flush=True)
    write(root / "done.json", status)


def score(model_name: str) -> None:
    """공식 질의 함수를 실행하고 top200에서 독립 Recall을 검산한다."""
    from sklearn.neighbors import KDTree
    check()
    E, _ = official()
    with (OUT / "data/query.pickle").open("rb") as f:
        queries = pickle.load(f)
    with (OUT / "data/database.pickle").open("rb") as f:
        databases = pickle.load(f)
    vectors = {}
    for name in NAMES:
        directory = OUT / model_name / "desc" / name
        if not (directory / "done.json").exists():
            continue
        done = json.loads((directory / "done.json").read_text())
        a = np.zeros((done["total"], 8448), np.float32)
        seen = np.zeros(len(a), bool)
        for p in directory.glob("shard_*.npz"):
            z = np.load(p)
            a[z["indices"]] = z["desc"]
            seen[z["indices"]] = True
        assert seen.all()
        vectors[name] = a
    results = []
    thresholds = np.linspace(0, 1, 1000)
    for db, db_name in enumerate(DATABASES):
        if db_name not in vectors:
            continue
        database = vectors[db_name]
        meta = list(databases[db].values())
        tree = KDTree(database)
        for qi, name in enumerate(NAMES):
            if name not in vectors:
                continue
            distance, index = tree.query(vectors[name], k=200)
            for probe in [0, len(vectors[name]) // 2, len(vectors[name]) - 1]:
                direct_distance, direct_index = tree.query(vectors[name][probe:probe + 1], k=200)
                assert np.array_equal(direct_index[0], index[probe])
                assert np.array_equal(direct_distance[0], distance[probe])
            intra = db_name == name
            start_time = float(Path(meta[0]["query"]).stem) / 1e9 if intra else None
            sums = np.zeros(30)
            direct = np.zeros(30)
            cells = np.zeros((4, 1000))
            valid = 0
            evaluated = np.zeros(len(vectors[name]), bool)
            first_correct = np.zeros(len(vectors[name]), bool)
            class CachedTree:
                def query(self, x, k):
                    assert k == 200
                    return distance[current:current + 1], index[current:current + 1]
            for current in range(len(vectors[name])):
                result = E.process_query(current, vectors[name][current], queries[qi][current], CachedTree(), 30,
                    max(int(round(len(database) / 1000)), 1), database, thresholds, db,
                    database_meta=meta, is_intra=intra, init_time=start_time)
                if result is None:
                    continue
                rec, _, tp, fp, tn, fn, count = result
                valid += count
                evaluated[current] = True
                sums += rec
                cells += np.asarray([tp, fp, tn, fn])
                allowed = index[current]
                if intra:
                    cutoff = float(Path(queries[qi][current]["query"]).stem) / 1e9 - 30
                    allowed = np.asarray([j for j in allowed if float(Path(meta[j]["query"]).stem) / 1e9 < cutoff])
                correct = np.isin(allowed[:30], queries[qi][current][db])
                first_correct[current] = bool(correct[0])
                matched = np.flatnonzero(correct)
                if len(matched):
                    direct[matched[0]:] += 1
            official_recall = np.cumsum(sums) / max(valid, 1)
            independent = direct / max(valid, 1)
            assert np.array_equal(official_recall, independent)
            precision = np.divide(cells[0], cells[0] + cells[1], out=np.zeros(1000), where=(cells[0] + cells[1]) != 0)
            recall = np.divide(cells[0], cells[0] + cells[3], out=np.zeros(1000), where=(cells[0] + cells[3]) != 0)
            f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(1000), where=(precision + recall) != 0)
            row = {"database": db_name, "query": name, "positive_evaluated": valid,
                   "r1": float(official_recall[0]), "r5": float(official_recall[4]),
                   "official_max_f1": float(f1.max()), "independent_recall_equal": True}
            results.append(row)
            pair_dir = OUT / model_name / "native_pairs"
            pair_dir.mkdir(exist_ok=True)
            pair_file = pair_dir / f"{db_name}__{name}.npz"
            np.savez_compressed(pair_file, indices=index, distances=distance,
                                evaluated=evaluated, first_correct=first_correct,
                                tp_fp_tn_fn=cells, recall_hits=direct, thresholds=thresholds)
            row["prediction_sha256"] = sha(pair_file)
            write(OUT / model_name / "score_progress.json", {"pairs": results, "completed": len(results), "total": 24})
            print(json.dumps(row), flush=True)
    if len(results) != 24:
        write(OUT / model_name / "native_partial_report.json", {"model": model_name, "pairs": results,
              "completed_pairs": len(results), "required_pairs": 24,
              "note": "다운로드/추론 완료쌍만의 진단. 논문Table I의6쌍평균을 대신하지 않음"})
        return
    groups = {db: {group: {metric: float(np.mean([r[metric] for r in results if r["database"] == db and r["query"].split("-")[-1] in sensors]))
                          for metric in ["r1", "r5"]} for group, sensors in [("Narrow", ["Aeva", "Avia"]), ("Wide", ["Ouster", "Velodyne"])]} for db in DATABASES}
    write(OUT / model_name / "native_report.json", {"model": model_name, "pairs": results, "groups": groups,
          "published": check()["published_table_I"], "protocol_sha256": sha(OUT / "protocol.json"),
          "input_manifest_sha256": sha(OUT / "input_manifest.json"), "checks_passed": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["protocol", "prepare", "infer", "score"])
    p.add_argument("--model", default="author", choices=["author", "v2_s1", "v2_s2", "v2_s3"])
    p.add_argument("--sequence", choices=NAMES)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "protocol":
        protocol()
    elif a.stage == "prepare":
        prepare()
    elif a.stage == "infer":
        infer(a.model, a.sequence, a.smoke)
    else:
        score(a.model)
