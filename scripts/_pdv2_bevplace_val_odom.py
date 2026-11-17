#!/usr/bin/env python3
"""선택된 공식 BEVPlace++를 NSD와 같은 KISS-ICP 검증 질의로 평가한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

import _bevplace_official_bev as bev
from _bevplace_official_dump import _official_model, ledger_l2
from _pdv2_pilot import VAL
from _pr_eval_lib import classify_queries, pr_sweep

OUT = REPO / "results/pdv2_bevplace_val_odom_20261018"
LOG = REPO / "logs/bevplace_val_odom_20261018"
ROLE = "val_odom"
NSD = REPO / "results/pdv2_sequence_evidence_20261018"
SELECTION = REPO / "results/pdv2_bevplace_official/train_selection.json"
METRICS = ("AP", "max_F1", "R@100P", "R@1")
EXPECTED_SHA = "b0dd3a16c329bd8fc847e4f44ad5b848d98e821772cd6e39097f41940f90a96f"
bev.ROLES[ROLE] = (VAL, bev.ODOM, "val_", "poses_gt")


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """진행 상태가 부분 파일로 보이지 않도록 교체 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare() -> dict:
    """재선택 없이 체크포인트·입력·평가 조건을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        protocol = read(path)
        for source, digest in protocol["sources"].items():
            assert bev.sha256(REPO / source) == digest, source
        return protocol
    selection = read(SELECTION)
    assert selection["finished"]
    best = selection["best"]
    assert best["epoch"] == 4 and best["sha256"] == EXPECTED_SHA
    checkpoint = REPO / best["checkpoint"]
    assert checkpoint.stat().st_size == 16462465
    assert bev.sha256(checkpoint) == EXPECTED_SHA
    sources = [Path(__file__), SELECTION, checkpoint,
               checkpoint.parent / "flags.json", NSD / "summary.json"]
    sources += [REPO / "scripts" / name for name in (
        "_bevplace_official_bev.py", "_bevplace_official_dump.py",
        "_pd_build_test_caches.py", "_pr_eval_lib.py", "_pdv2_pilot.py")]
    sources += [bev.OFFICIAL / name for name in (
        "REIN.py", "main.py", "nclt_dataset.py", "datasets/gen_bev_images.py")]
    for sensor, seq in VAL:
        sources.append(bev.cache_path(ROLE, sensor, seq))
        sources += [NSD / f"s{s}/standard_{sensor}_{seq}.npz" for s in (1, 2, 3)]
    protocol = {
        "created_unix": time.time(), "checkpoint": best,
        "source_server": "cps2 (SSH joms-gpu)",
        "remote_checkpoint": "/home/qkrrlans00/Neural-Spectral-Codec/" + best["checkpoint"],
        "official_commit": bev.OFFICIAL_COMMIT,
        "role": ROLE, "roster": [s + "/" + q for s, q in VAL],
        "query_rule": "30 <= q < N-15", "candidate_rule": "j <= q-30",
        "input": "NSD와 같은 KISS-ICP 키프레임의 원시 스캔 한 장. GT는 평가 정답에만 사용.",
        "preprocessing": "공식 getBEV와 필터·cv2.imwrite, 201x201 점 밀도 영상, 공식 /256 입력",
        "retrieval": "공식 8192차원 특징과 FAISS IndexFlatL2. 검색마다 과거 접두만 색인에 존재.",
        "score": "score_negl2", "infer_batch": 32, "loader_threads": 4,
        "aggregation": "양성 있는 시퀀스 평균 -> KITTI/HeLiPR/MulRan 동일 비중 평균",
        "selection": "기존 선택 에포크 4 유지. 새 체크포인트·NSD 설정·임계값 선택 없음.",
        "test_used": False,
        "sources": {str(p.relative_to(REPO)): bev.sha256(p) for p in sources},
    }
    write(path, protocol)
    return protocol


def image_audit(sensor: str, seq: str) -> dict:
    """모든 PNG를 검사하고 복사·생성 입력 표본을 원시 스캔에서 재현한다."""
    import cv2
    from _pd_build_test_caches import _make_loader

    name = bev.seq_name(sensor, seq)
    directory = bev.BEV / ROLE / name
    ids = np.load(directory / "scan_ids.npy")
    wanted = [f"{sid:06d}.png" for sid in ids]
    assert sorted(os.listdir(directory / "bev_imgs")) == wanted
    digest = hashlib.sha256()
    for filename in wanted:
        path = directory / "bev_imgs" / filename
        raw = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert image is not None and image.shape == (201, 201)
        assert image.dtype == np.uint8
        digest.update(filename.encode())
        digest.update(hashlib.sha256(raw).digest())
    old_ids = np.load(bev.BEV / "val" / name / "scan_ids.npy")
    reused = np.isin(ids, old_ids)
    selected = set()
    for mask in (reused, ~reused):
        indices = np.flatnonzero(mask)
        if len(indices):
            selected.update(int(indices[i]) for i in (0, len(indices) // 2, len(indices) - 1))
    loader = _make_loader(sensor, seq)
    get_bev = bev.official_getBEV()
    checks = []
    with np.load(bev.cache_path(ROLE, sensor, seq)) as cache:
        gt = cache["poses_gt"]
    for index in sorted(selected):
        item = loader[int(ids[index])]
        if item.get("pose") is not None:
            np.testing.assert_allclose(item["pose"][:3, 3], gt[index, :3, 3], rtol=0, atol=1e-2)
        expected = bev.bev_image(item["points"], get_bev)
        temporary = directory / "_audit.png"
        assert cv2.imwrite(str(temporary), expected)
        actual = cv2.imread(str(directory / "bev_imgs" / wanted[index]), 0)
        np.testing.assert_array_equal(cv2.imread(str(temporary), 0), actual)
        temporary.unlink()
        checks.append({"scan_id": int(ids[index]), "reused": bool(reused[index]), "pixel_max_error": 0})
    return {"frames": len(ids), "ordered_image_sha256": digest.hexdigest(),
            "raw_regeneration": checks, "reused": int(reused.sum())}


def metrics(arrays: dict) -> dict:
    """공통 원장으로 최종 판정 지표를 계산한다."""
    query_class = classify_queries(arrays["min_past_gt_dist"])
    curve = pr_sweep(arrays["score_negl2"], arrays["top1_geo_dist"], query_class)
    positive = query_class["positive"]
    result = {key: curve[key] for key in METRICS if key != "R@1"}
    result["R@1"] = float(np.mean(arrays["top1_geo_dist"][positive] <= 5)) if positive.any() else None
    result["n_pos"] = query_class["n_pos"]
    result["n_queries"] = len(positive)
    return result


def evaluate(sequence: str) -> None:
    """공식 추론과 검색을 실행하고 동일 질의·과거 입력을 확인한다."""
    import faiss
    import torch

    protocol = prepare()
    sensor, seq = sequence.split("/")
    target = OUT / f"{sensor}_{seq}.json"
    if target.exists():
        return
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    checkpoint = REPO / protocol["checkpoint"]["checkpoint"]
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint_data["epoch"] == 4
    assert abs(checkpoint_data["best_score"] - protocol["checkpoint"]["family_macro_r1"]) < 1e-12
    inputs = image_audit(sensor, seq)
    module, dataset_module, _ = _official_model(checkpoint, 4, 32, "cuda")
    dataset = dataset_module.InferDataset(seq=bev.seq_name(sensor, seq), dataset_path=str(bev.BEV / ROLE) + "/")
    with np.load(bev.cache_path(ROLE, sensor, seq)) as cache:
        ids = cache["scan_ids"]
        positions = cache["poses_gt"][:, :3, 3].astype(np.float64)
        np.testing.assert_array_equal(dataset.poses, cache["poses_gt"][:, :3, :].reshape(-1, 12))
    assert [Path(p).name for p in dataset.imgs_path] == [f"{sid:06d}.png" for sid in ids]
    tick = time.perf_counter()
    descriptors = module.infer(dataset).astype(np.float32)
    encode_s = time.perf_counter() - tick
    assert descriptors.shape == (len(ids), 8192) and np.isfinite(descriptors).all()
    np.testing.assert_allclose(np.linalg.norm(descriptors, axis=1), 1., rtol=0, atol=1e-5)
    feature_path = OUT / f"descriptors_{sensor}_{seq}.npy"
    np.save(feature_path, descriptors)
    tick = time.perf_counter()
    arrays = ledger_l2(descriptors, positions)
    search_s = time.perf_counter() - tick
    keep = arrays["query_idx"] < len(ids) - 15
    arrays = {key: value[keep] for key, value in arrays.items()}
    with np.load(NSD / f"s1/standard_{sensor}_{seq}.npz") as reference:
        for key in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(arrays[key], reference[key])
    assert (arrays["j_star"] <= arrays["query_idx"] - 30).all()
    np.testing.assert_array_equal(arrays["n_candidates"], arrays["query_idx"] - 29)
    audits = []
    count = len(arrays["query_idx"])
    for row in sorted(set((0, count // 3, 2 * count // 3, count - 1))):
        q = int(arrays["query_idx"][row])
        index = faiss.IndexFlatL2(8192)
        index.add(descriptors[:q - 29])
        distance, candidate = index.search(descriptors[q:q + 1], 1)
        assert candidate[0, 0] == arrays["j_star"][row]
        assert -float(np.sqrt(max(float(distance[0, 0]), 0.))) == arrays["score_negl2"][row]
        brute = np.sum((descriptors[:q - 29].astype(np.float64) - descriptors[q]) ** 2, axis=1)
        assert int(brute.argmin()) == arrays["j_star"][row]
        error = abs(-np.sqrt(brute.min()) - arrays["score_negl2"][row])
        assert error < 1e-5
        audits.append({"q": q, "independent_prefix_exact": True, "float64_distance_error": error})
    ledger = OUT / f"dump_bevplace_{sensor}_{seq}.npz"
    meta = {"schema": "pr-dump-v1", "method": "bevplace", "role": ROLE,
            "checkpoint_sha256": EXPECTED_SHA, "primary_score": "score_negl2",
            "n_frames": len(ids), "n_queries": len(arrays["query_idx"]),
            "protocol_sha256": bev.sha256(OUT / "protocol.json")}
    np.savez_compressed(ledger, meta=json.dumps(meta), **arrays)
    write(target, {"sequence": sequence, "metrics": metrics(arrays), "images": inputs,
                   "retrieval_audit": audits, "timing_s": {"encode": encode_s, "search": search_s},
                   "ledger": str(ledger.relative_to(REPO)), "ledger_sha256": bev.sha256(ledger),
                   "descriptors_sha256": bev.sha256(feature_path)})
    print("EVALUATED", sequence, metrics(arrays), flush=True)


def report() -> None:
    """검증 원장을 재검산하고 고정 NSD의 같은 검증 결과와 비교한다."""
    prepare()
    records, total, input_checks, search_checks = {}, 0, 0, 0
    for sensor, seq in VAL:
        row = read(OUT / f"{sensor}_{seq}.json")
        path = REPO / row["ledger"]
        assert bev.sha256(path) == row["ledger_sha256"]
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files if key != "meta"}
        assert metrics(arrays) == row["metrics"]
        records[sensor + "/" + seq] = row["metrics"]
        total += row["metrics"]["n_queries"]
        input_checks += len(row["images"]["raw_regeneration"])
        search_checks += len(row["retrieval_audit"])
    families = {}
    for family in ("kitti", "helipr", "mulran"):
        chosen = [r for name, r in records.items() if name.startswith(family) and r["n_pos"]]
        families[family] = {metric: float(np.mean([r[metric] for r in chosen])) for metric in METRICS}
    macro = {metric: float(np.mean([r[metric] for r in families.values()])) for metric in METRICS}
    nsd = read(NSD / "summary.json")["mean_std"]["standard"]
    comparison = {method: nsd[method] for method in ("raw", "arc_sparse_b25", "arc_unique_b100")}
    delta = {method: {metric: 100 * (row[metric]["mean"] - macro[metric]) for metric in METRICS}
             for method, row in comparison.items()}
    summary = {"bevplace": {"seed": 1, "epoch": 4, "per_sequence": records, "families": families, "macro": macro},
               "nsd_three_seed": comparison, "nsd_minus_bevplace_pp": delta, "n_queries": total,
               "scope": "같은 KISS-ICP 검증 9·3계열 최종 판정. 기존 가중치 고정, 시험 사용·재선택 없음."}
    write(OUT / "summary.json", summary)
    with (OUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "seeds", *METRICS])
        for method, row in comparison.items():
            writer.writerow(["NSD " + method, 3, *[100 * row[m]["mean"] for m in METRICS]])
        writer.writerow(["BEVPlace++ official", 1, *[100 * macro[m] for m in METRICS]])
    assert total == 99236
    write(OUT / "completion_validation.json", {"status": "passed", "ledgers": 9, "rows": total,
        "image_regeneration_checks": input_checks, "independent_search_checks": search_checks,
        "same_nsd_queries": True, "frozen_sources": True, "test_used": False,
        "summary_sha256": bev.sha256(OUT / "summary.json")})
    print("FINAL", summary, flush=True)


def run() -> None:
    """누락 영상을 채우고 시퀀스별 공식 추론·검색을 실행한다."""
    protocol = prepare()
    tick = time.time()
    bev.cmd_index(ROLE)

    def generate(sequence: str, shard: int) -> None:
        with (LOG / f"images_{sequence.replace('/', '_')}_{shard}.log").open("w") as handle:
            subprocess.run([sys.executable, str(Path(__file__)), "gen", "--sequence", sequence,
                            "--shard", str(shard)], stdout=handle, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(generate, sequence, shard) for sequence in protocol["roster"] for shard in range(4)]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            write(OUT / "status.json", {"status": "images", "done": count, "total": len(futures)})
            print("IMAGES", count, len(futures), flush=True)
    assert bev.cmd_status([ROLE])
    for count, sequence in enumerate(protocol["roster"], 1):
        with (LOG / f"evaluate_{sequence.replace('/', '_')}.log").open("w") as handle:
            subprocess.run([sys.executable, str(Path(__file__)), "evaluate", "--sequence", sequence],
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        write(OUT / "status.json", {"status": "evaluate", "done": count, "total": len(VAL)})
        print("EVALUATE", count, len(VAL), sequence, flush=True)
    report()
    write(OUT / "status.json", {"status": "complete", "runtime_s": time.time() - tick})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "gen", "evaluate", "report"))
    parser.add_argument("--sequence", choices=[s + "/" + q for s, q in VAL])
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if args.action == "gen":
        bev.cmd_gen(ROLE, args.sequence, args.shard, 4)
    elif args.action == "evaluate":
        evaluate(args.sequence)
    else:
        globals()[args.action]()
