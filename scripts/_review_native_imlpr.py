#!/usr/bin/env python3
"""공식 MulRan 경로로 ImLPR 일반화 성능과 k15/k25 차이를 분리한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105/imlpr"
OFFICIAL = REPO / "_external/ImLPR"
sys.path[:0] = [str(REPO / "scripts"), str(REPO), str(REPO / "src")]
NAMES = ["DCC01", "DCC02", "DCC03"]


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
    """시험 수치에 무관하게 공개 코드·논문 단일요인 두 조건을 고정한다."""
    inventory = json.loads((REPO / "results/pdv2_review_selection_full_20261105/retained_source_inventory.json").read_text())
    models = {"author": {"path": "_external/ImLPR/weights/ImLPR_default.pth",
                          "sha256": sha(OFFICIAL / "weights/ImLPR_default.pth")}}
    for s in (1, 2, 3):
        v = inventory["methods"]["imlpr"]["seeds"][str(s)]["original_choice"]
        models[f"v2_s{s}"] = {"path": v["checkpoint"].replace("/workspace/Neural-Spectral-Codec/", ""), "sha256": v["sha256"]}
    source = [OFFICIAL / "preprocess/generate_riv_images.py", OFFICIAL / "eval/evaluate.py",
              OFFICIAL / "preprocess/generate_test_sets_img.py", REPO / "scripts/_imlpr_official_riv.py",
              REPO / "scripts/_imlpr_desc.py", REPO / "src/data/mulran_loader.py"]
    d = {"official_repo": "https://github.com/minwoo0611/ImLPR",
         "paper": "https://arxiv.org/abs/2505.18364", "models": models, "sequences": NAMES,
         "published_Table7_D": {"DCC01": {"r1": .909, "f1": .965}, "DCC02": {"r1": .861, "f1": .945}, "DCC03": {"r1": .825, "f1": .918}},
         "source_sha256": {str(p.relative_to(REPO)): sha(p) for p in source},
         "factors": {"k15": "공식공개생성코드 k_neighbors=15", "k25": "논문App.A의 MulRan k=25; 이 값만 바꿈"},
         "fixed_preprocessing": "원시OS1-64 single scan, ring=원시index%64, H64 W1022, range120m, intensity min(4096)/4096*255, normal max_neighbor15m",
         "sampling": "전체 원시스캔 대응GT에서 공식 select_frame_indices 3m; GT는 공식MulRan100Hz csv와scan시간 최근접(기존loader)으로 연결",
         "labels": "공식 평가생성기와 같은 XY거리<=10m, 같은세션 timestamp<q-60초; query90초 이후; 공개 process_query의 top200 후 과거필터",
         "scope_limit": "저자 가공tar는공개다운로드한도초과. 원시공식센서/논문규약 재생성; 저자가쓴가공파일과byte동일하다고주장안함. GT시간연결은공식헬퍼의index정렬가정대신명시적으로연결",
         "selection": "두 사전고정조건 모두보고; 시험성능으로설정/가중치선택안함",
         "contamination": "저자HeLiPR대전학습과MulRanDCC시험은장소겹침. 논문의native일반화진단이며우리장소분리주표에편입안함"}
    p = OUT / "protocol.json"
    if p.exists():
        assert json.loads(p.read_text()) == d
    else:
        write(p, d)


def check() -> dict:
    d = json.loads((OUT / "protocol.json").read_text())
    for p, h in d["source_sha256"].items():
        assert sha(REPO / p) == h, p
    return d


def prepare(sequence: str, limit: int = 0) -> None:
    """공식3m샘플과 동일스캔의 두 k조건 RIV를 CPU에서 만든다."""
    from _imlpr_official_riv import official_generator, fast_normals, fast_image, fast_range, SENSORS
    from _pd_build_test_caches import _make_loader
    from sklearn.neighbors import KDTree
    check()
    gen = official_generator()
    loader = _make_loader("mulran", sequence)
    poses = loader.scan_poses
    pose_dicts = [{"x": float(t[0, 3]), "y": float(t[1, 3]), "z": float(t[2, 3])} for t in poses]
    selected = np.asarray(gen.select_frame_indices(pose_dicts, 3.0))
    root = OUT / "inputs" / sequence
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.npz"
    np.savez_compressed(index_path, scan_ids=selected, poses=poses[selected], timestamps=loader.scan_timestamps_ns[selected])
    c = SENSORS["mulran"]
    tree = KDTree(poses[selected, :2, 3])
    true = tree.query_radius(poses[selected, :2, 3], r=10)
    times = loader.scan_timestamps_ns[selected]
    metadata = {i: {"query": f"{sequence}/{i:06d}.npy", "northing": float(poses[sid, 0, 3]),
        "easting": float(poses[sid, 1, 3]), "timestamp": int(times[i]),
        0: [int(j) for j in true[i] if float(times[j]) < float(times[i]) - 60e9]}
        for i, sid in enumerate(selected)}
    with (root / "metadata.pickle").open("wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    smoke_ids = set(np.unique(np.linspace(0, len(selected) - 1, 3, dtype=int)).tolist())
    check_path = root / "official_input_checks.json"
    checks = json.loads(check_path.read_text())["checks"] if check_path.exists() else []
    start = time.time()
    for i, sid in enumerate(selected):
        if limit and i >= limit:
            break
        targets = {k: OUT / f"k{k}/data/validation" / sequence / f"{i:06d}.npy" for k in (15, 25)}
        if all(p.exists() for p in targets.values()):
            continue
        raw = loader.scan_files[int(sid)]
        xyz, refl, ring, _ = gen.read_bin_file(str(raw), "mulran")
        raw_hash = sha(raw)
        for k, target in targets.items():
            if target.exists():
                continue
            normal = fast_normals(xyz, k=k, max_neighbor_m=15)
            img = np.stack([fast_image(xyz, refl, ring, c), fast_range(xyz, ring, c), fast_image(xyz, normal, ring, c)], axis=-1).astype(np.float32)
            if i in smoke_ids:
                official_normal = gen.compute_normals(xyz, k=k, max_neighbor_m=15)
                kw = dict(image_width=1022, image_height_initial=64, image_height_final=64, v_mode="ring", fov_up_deg=16., fov_down_deg=16., invert_vertical=False)
                official_img = np.stack([gen.pointCloudToImage(xyz, refl, ring, **kw),
                    gen.pointCloudToRangeImage(xyz, ring, lidar_type="mulran", max_range_m=120, **kw),
                    gen.pointCloudToImage(xyz, official_normal, ring, **kw)], axis=-1)
                error = float(np.max(np.abs(img - official_img)))
                assert error < 1e-4, (sequence, i, k, error)
                checks.append({"frame": i, "scan_id": int(sid), "k": k, "max_abs_error_official": error, "raw_sha256": raw_hash})
                write(root / "official_input_checks.json", {"checks": checks})
            assert img.shape == (64, 1022, 3) and np.isfinite(img).all()
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.stem + ".tmp.npy")
            np.save(tmp, img)
            tmp.replace(target)
        if i % 25 == 0:
            status = {"sequence": sequence, "frames": i + 1, "total": len(selected), "elapsed_s": time.time() - start, "pid": os.getpid()}
            write(root / "status.json", status)
            print(json.dumps(status), flush=True)
    if not limit:
        write(root / "done.json", {"sequence": sequence, "n_frames": len(selected), "index_sha256": sha(index_path), "elapsed_s": time.time() - start})


def infer(model_name: str, sequence: str, k: int, smoke: bool) -> None:
    """공식 영상읽기와 모델을256장 재개단위로 실행한다."""
    import torch
    from _imlpr_desc import load_model
    d = check()
    torch.hub.set_dir(str(REPO / "_external/imlpr_official_pdv2/torch_hub/hub"))
    torch.set_num_threads(2)
    path = REPO / d["models"][model_name]["path"]
    assert sha(path) == d["models"][model_name]["sha256"]
    model = load_model(path, "cuda")
    import eval.evaluate as E
    E.RESIZE_TO_HELIPR = True
    with (OUT / "inputs" / sequence / "metadata.pickle").open("rb") as f:
        metadata = pickle.load(f)
    chosen = np.arange(min(8, len(metadata))) if smoke else np.arange(len(metadata))
    root = OUT / f"k{k}" / model_name / ("smoke_batch1" if smoke else "desc_batch1") / sequence
    root.mkdir(parents=True, exist_ok=True)
    if (root / "done.json").exists():
        return
    start = time.time()
    probes = np.unique(chosen[np.linspace(0, len(chosen) - 1, 3, dtype=int)])
    probe_data = {int(i): metadata[int(i)] for i in probes}
    single = E._get_latent_vectors(model, probe_data, "cuda", str(OUT / f"k{k}/data"), batch_size=1)
    grouped = E._get_latent_vectors(model, probe_data, "cuda", str(OUT / f"k{k}/data"), batch_size=3)
    error = float(np.max(np.abs(single - grouped)))
    write(root / "batch_check.json", {"indices": probes.tolist(), "max_abs_error": error, "limit": 1e-5,
                                      "within_limit": error < 1e-5, "used_inference_batch": 1,
                                      "policy": "배치간 차이 진단만 기록. 모든native조건을batch1로고정하며기존batch4결과보존"})
    for first in range(0, len(chosen), 256):
        target = root / f"shard_{first:06d}.npz"
        if target.exists():
            continue
        indices = chosen[first:first + 256]
        subset = {int(i): metadata[int(i)] for i in indices}
        for row in subset.values():
            assert (OUT / f"k{k}/data/validation" / row["query"]).exists()
        desc = E._get_latent_vectors(model, subset, "cuda", str(OUT / f"k{k}/data"), batch_size=1)
        assert desc.shape == (len(indices), 8448) and np.isfinite(desc).all()
        tmp = target.with_name(target.stem + ".tmp.npz")
        np.savez_compressed(tmp, indices=indices, desc=desc)
        tmp.replace(target)
        status = {"model": model_name, "sequence": sequence, "k": k, "frames": first + len(indices), "total": len(chosen),
                  "elapsed_s": time.time() - start, "gpu": torch.cuda.get_device_name(), "pid": os.getpid(), "batch": 1}
        write(root / "status.json", status)
        print(json.dumps(status), flush=True)
    write(root / "done.json", status)


def score(model_name: str, sequence: str, k: int) -> None:
    """공식질의함수의분모·점수를유지하고 독립정수 Recall을 검산한다."""
    from sklearn.neighbors import KDTree
    sys.path.insert(0, str(OFFICIAL))
    import eval.evaluate as E
    check()
    root = OUT / f"k{k}" / model_name / "desc_batch1" / sequence
    done = json.loads((root / "done.json").read_text())
    desc = np.zeros((done["total"], 8448), np.float32)
    seen = np.zeros(len(desc), bool)
    for p in root.glob("shard_*.npz"):
        z = np.load(p)
        desc[z["indices"]] = z["desc"]
        seen[z["indices"]] = True
    assert seen.all() and np.isfinite(desc).all()
    with (OUT / "inputs" / sequence / "metadata.pickle").open("rb") as f:
        metadata = pickle.load(f)
    meta = list(metadata.values())
    times = np.array([float(m["timestamp"]) / 1e9 for m in meta])
    tree = KDTree(desc)
    distances, indices = tree.query(desc, k=200)
    for probe in [0, len(desc) // 2, len(desc) - 1]:
        direct_distance, direct_index = tree.query(desc[probe:probe + 1], k=200)
        assert np.array_equal(direct_index[0], indices[probe])
        assert np.array_equal(direct_distance[0], distances[probe])
    thresholds = np.linspace(0, 1, 1000)
    cells = np.zeros((4, 1000))
    cumulative = np.zeros(30)
    valid, hits, denominator = 0, 0, 0
    rows = []
    class CachedTree:
        def query(self, x, k):
            assert k == 200
            return distances[current:current + 1], indices[current:current + 1]
    for current in range(len(desc)):
        result = E._process_query(current, desc[current], meta[current], CachedTree(), 30,
            max(int(round(len(desc) / 1000)), 1), desc, thresholds, 0,
            database_meta=meta, is_intra=True, init_time=float(times.min()))
        if len(meta[current][0]) and times[current] - times.min() >= 90:
            denominator += 1
        if result is None:
            rows.append([current, -1, float("nan"), 0])
            continue
        rec, _, tp, fp, tn, fn, count = result
        valid += count
        cumulative += rec
        cells += [tp, fp, tn, fn]
        allowed = indices[current][times[indices[current]] < times[current] - 60]
        assert len(allowed)
        direct = int(int(allowed[0]) in meta[current][0])
        assert direct == int(rec[0])
        hits += direct
        rows.append([current, int(allowed[0]), float(np.linalg.norm(desc[current].astype(np.float64) - desc[allowed[0]].astype(np.float64))), direct])
    recall_curve = np.cumsum(cumulative) / max(valid, 1)
    assert recall_curve[0] == hits / max(valid, 1)
    precision = np.divide(cells[0], cells[0] + cells[1], out=np.zeros(1000), where=(cells[0] + cells[1]) != 0)
    recall = np.divide(cells[0], cells[0] + cells[3], out=np.zeros(1000), where=(cells[0] + cells[3]) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(1000), where=(precision + recall) != 0)
    # 공개 _get_recall_for_pair가 쓰는 epsilon까지 유지한다.
    official_precision = cells[0] / (cells[0] + cells[1] + 1e-8)
    official_recall = cells[0] / (cells[0] + cells[3] + 1e-8)
    official_f1 = 2 * official_precision * official_recall / (official_precision + official_recall + 1e-8)
    assert abs(float(f1.max()) - float(official_f1.max())) < 1e-7
    fixed_recall = cells[0] / max(denominator, 1)
    fixed_f1 = np.divide(2 * precision * fixed_recall, precision + fixed_recall, out=np.zeros(1000), where=(precision + fixed_recall) != 0)
    target = OUT / f"k{k}" / model_name / f"native_batch1_{sequence}.json"
    write(target, {"model": model_name, "sequence": sequence, "k": k, "n_frames": len(desc), "inference_batch": 1,
          "official_evaluated_queries": valid, "fixed_positive_denominator": denominator, "hits_at1": hits,
          "r1": float(recall_curve[0]), "r5": float(recall_curve[4]), "official_max_f1": float(official_f1.max()),
          "independent_max_f1_without_epsilon": float(f1.max()),
          "fixed_denominator_max_f1": float(fixed_f1.max()), "independent_recall_equal": True,
          "published": check()["published_Table7_D"][sequence], "protocol_sha256": sha(OUT / "protocol.json")})
    np.savez_compressed(target.with_suffix(".npz"), predictions=np.asarray(rows), thresholds=thresholds, tp_fp_tn_fn=cells,
                        indices=indices, distances=distances)
    print(target.read_text(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["protocol", "prepare", "infer", "score"])
    p.add_argument("--sequence", choices=NAMES)
    p.add_argument("--model", default="author", choices=["author", "v2_s1", "v2_s2", "v2_s3"])
    p.add_argument("--k", type=int, choices=[15, 25], default=15)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "protocol":
        protocol()
    elif a.stage == "prepare":
        prepare(a.sequence, a.limit)
    elif a.stage == "infer":
        infer(a.model, a.sequence, a.k, a.smoke)
    else:
        score(a.model, a.sequence, a.k)
