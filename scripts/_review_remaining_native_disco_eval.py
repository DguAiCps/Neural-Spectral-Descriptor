#!/usr/bin/env python3
"""서버에서 공개 DiSCO 모델 추론→30세션쌍 평가→독립 검산을 단계별로 수행한다.

encode는 GPU를 쓰므로 호스트 감독기가 /tmp/nsd_native_reproduction_gpu0.lock을 잡은 뒤에만 부른다.
docker exec -e CUDA_VISIBLE_DEVICES=0 -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 -u scripts/_review_remaining_native_disco_eval.py encode --dates 2012-05-26
score·verify는 CPU만 쓴다(-e CUDA_VISIBLE_DEVICES=). 순서는
scripts/_review_remaining_native_disco_supervise.py가 관리한다.
공개 생성기의3≤d<3.5m 표본과 상한만 제거한 진단을 구별해 모두 기록한다.
후자는 원 논문 재현이나 새 주표 설정으로 채택하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/pdv2_remaining_native_20261106/disco"
PROTOCOL = REPO / "results/pdv2_reproduction_orchestration_20261106/disco_protocol"
DATES = ["2012-03-17", "2012-05-26", "2012-06-15", "2012-08-20",
         "2012-09-28", "2012-10-28"]
EXPECTED_SHA = "8665e356f0aab2cf03fdc47aa2b3dd2d32be7a2b420bad35ac2c994f45d7808c"
PAPER = {"r1_percent": 89.08, "r1_percentile_percent": 99.14}
RADIUS = 1.5
NEIGHBORS = 25


def write(path: Path, value: dict) -> None:
    """JSON을 원자적으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def hits(raw: Path) -> np.ndarray:
    """공식 NCLT 디코더와 같은 float64 연산으로 점을 읽는다."""
    from _disco_official_prep import official_filter
    packed = np.fromfile(raw, dtype=np.dtype([
        ("xyz", "<u2", (3,)), ("intensity", "u1"), ("laser", "u1")]))
    points = packed["xyz"].astype(np.float64) * 0.005 + -100.0
    points[:, 2] = -points[:, 2]
    return official_filter(points)


def encode(date: str, model, cfg, ev, gputransform, device: dict | None = None) -> None:
    """두 표본 목록의 합집합을 한 번 추론하고 공식 구현과 검산한다."""
    import torch
    from _disco_official_prep import (generator_functions, official_polar,
                                      polar_agrees, polar_numpy)
    dest = OUT / "native_data" / date
    target = OUT / "encoded" / date
    if (target / "complete.json").exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    primary = json.loads((dest / "selection.json").read_text())["rows"]
    diagnostic = json.loads((dest / "selection_no_upper_bound.json").read_text())["rows"]
    union = {str(row["timestamp"]): row for row in primary + diagnostic}
    keys = sorted(union, key=int)
    functions, gen_cfg = generator_functions(gputransform)
    batch_size = cfg.EVAL_BATCH_SIZE * (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY)
    assert batch_size == 7
    checks, descriptors, spot_images = [], [], []
    pick = set(np.linspace(0, max(0, len(keys)-1), min(3, len(keys))).round().astype(int))
    started = time.time()
    model.eval()
    pending = []
    for i, key in enumerate(keys):
        raw = dest / "velodyne_sync" / (key + ".bin")
        filtered = hits(raw)
        assert len(filtered), f"공개 생성기가 처리할 수 없는 빈 스캔: {raw}"
        polar = official_polar(filtered, gputransform, gen_cfg)
        reference = polar_numpy(filtered.T.flatten().astype(np.float32), len(filtered))
        reference = reference.reshape(-1, 3)[:, 2].reshape(polar.shape)
        ok, delta = polar_agrees(polar, reference)
        assert ok and polar.sum() > 1, (date, key, delta)
        if i in pick:
            slow_hits = functions["load_lidar_file_nclt"](str(raw))
            assert np.array_equal(filtered, slow_hits), (date, key, "원시 디코더 차이")
            slow_polar = functions["load_pc_file"](str(raw))
            assert np.array_equal(polar, slow_polar), (date, key, "공식 입력 차이")
            checks.append({"timestamp": int(key), "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                           "official_hits_exact": True, "official_polar_exact": True,
                           "cpu_cuda_differing_cells": delta})
        if i < 9:
            spot_images.append(polar)
        pending.append(polar)
        if len(pending) == batch_size or i == len(keys)-1:
            with torch.no_grad():
                tensor = torch.from_numpy(np.array(pending, dtype=np.float32)).cuda()
                descriptors.append(model(tensor)[0].detach().cpu().numpy().reshape(len(pending), -1))
            pending = []
        if (i+1) % 100 == 0:
            write(OUT / "pipeline_status.json", {"stage": "encoding", "date": date,
                  "done": i+1, "total": len(keys), "elapsed_s": time.time()-started})
    desc = np.concatenate(descriptors) if descriptors else np.empty((0, 1024), dtype=np.float32)
    assert desc.shape == (len(keys), 1024) and np.isfinite(desc).all()
    # 공식 추론 드라이버의 완전 배치와 나머지 배치를 함께 확인한다.
    inference_check = None
    if len(spot_images) == 9:
        old_loader = ev.load_pc_files
        ev.load_pc_files = lambda names: [spot_images[int(name)] for name in names]
        try:
            exact = ev.get_latent_vectors(model, {i: {"query": str(i), "heading": 0.0} for i in range(9)})[0]
        finally:
            ev.load_pc_files = old_loader
            model.eval()
        # 배치 크기에 따른 부동소수점 오차만 허용한다.
        assert np.allclose(exact, desc[:9], atol=2e-5, rtol=2e-5)
        inference_check = float(np.max(np.abs(exact-desc[:9])))
    distinct = len(np.unique(np.round(desc, 6), axis=0)) if len(desc) else 0
    assert not len(desc) or distinct >= 0.9 * len(desc)
    np.save(target / "descriptors.npy", desc)
    write(target / "rows.json", {"keys": keys, "rows": [union[key] for key in keys]})
    write(target / "complete.json", {"date": date, "n": len(keys), "descriptor_dim": 1024,
          "checkpoint_sha256": EXPECTED_SHA, "checks": checks, "all_inputs_cpu_cuda_checked": True,
          "official_inference_max_abs_difference": inference_check,
          "distinct_descriptors": distinct, "device": device, "elapsed_s": time.time()-started})


def recheck_encoded(date: str, model, gputransform, count: int = 14) -> dict:
    """앞선 판에서 저장한 서술자 일부를 현재 판 경로로 다시 추론해 세션 간 일관성을 확인한다."""
    import torch
    from _disco_official_prep import generator_functions, official_polar
    target = OUT / "encoded" / date
    keys = json.loads((target / "rows.json").read_text())["keys"]
    stored = np.load(target / "descriptors.npy")
    pick = np.unique(np.linspace(0, len(keys)-1, min(count, len(keys))).round().astype(int))
    _, gen_cfg = generator_functions(gputransform)
    images = [official_polar(hits(OUT / "native_data" / date / "velodyne_sync" / (keys[i] + ".bin")),
                             gputransform, gen_cfg) for i in pick]
    model.eval()
    with torch.no_grad():
        again = np.concatenate([model(torch.from_numpy(np.array(images[s:s+7], dtype=np.float32)).cuda())[0]
                                .detach().cpu().numpy().reshape(-1, 1024) for s in range(0, len(images), 7)])
    difference = float(np.max(np.abs(again - stored[pick])))
    return {"date": date, "stored_complete_json_has_device": "device" in json.loads((target / "complete.json").read_text()),
            "rechecked_indices": pick.tolist(), "rechecked_timestamps": [int(keys[i]) for i in pick],
            "max_abs_difference": difference, "tolerance": "np.allclose atol=2e-5 rtol=2e-5 (공식 배치 검산과 같음)",
            "passed": bool(np.allclose(again, stored[pick], atol=2e-5, rtol=2e-5)), "unix": time.time()}


def score(variant: str, ev) -> dict:
    """원본 KDTree·1.5m 정답·양성 질의 분모·30쌍 평균을 확인한다."""
    import torch
    from sklearn.neighbors import KDTree
    databases, queries, db_xy, query_xy = [], [], [], []
    for date in DATES:
        target = OUT / "encoded" / date
        desc = np.load(target / "descriptors.npy")
        meta = json.loads((target / "rows.json").read_text())
        lookup = {int(key): i for i, key in enumerate(meta["keys"])}
        filename = "selection.json" if variant == "released" else "selection_no_upper_bound.json"
        rows = json.loads((OUT / "native_data" / date / filename).read_text())["rows"]
        outside = [r for r in rows if not r["test_region"]]
        inside = [r for r in rows if r["test_region"]]
        databases.append(desc[[lookup[r["timestamp"]] for r in outside + inside]])
        queries.append(desc[[lookup[r["timestamp"]] for r in inside]])
        db_xy.append(np.array([r["xy"] for r in outside + inside]).reshape(-1, 2))
        query_xy.append(np.array([r["xy"] for r in inside]).reshape(-1, 2))
    rows, qsets = [], [{} for _ in DATES]
    for m, db in enumerate(databases):
        if not len(db):
            continue
        geo_tree = KDTree(db_xy[m])
        for n, query in enumerate(queries):
            if m == n or not len(query):
                continue
            truths = geo_tree.query_radius(query_xy[n], r=1.5)
            for i, truth in enumerate(truths):
                qsets[n].setdefault(i, {})[m] = truth.tolist()
    for m, db in enumerate(databases):
        for n, query in enumerate(queries):
            if m == n:
                continue
            entry = {"database": DATES[m], "query": DATES[n], "database_frames": len(db), "query_frames": len(query)}
            if len(db) < 25 or not len(query):
                entry.update(status="official_evaluator_unusable", reason="지도25개 미만 또는 질의 없음")
                rows.append(entry)
                continue
            valid = np.array([len(qsets[n][i][m]) > 0 for i in range(len(query))])
            if not valid.any():
                entry.update(status="official_evaluator_unusable", reason="정답 있는 질의 없음")
                rows.append(entry)
                continue
            distances, indices = KDTree(db).query(query, k=25)
            success = np.zeros((len(query), 25), dtype=bool)
            for i in range(len(query)):
                success[i] = np.isin(indices[i], qsets[n][i][m])
            curve = np.maximum.accumulate(success[valid], axis=1).mean(0)*100
            top_percent = max(int(round(len(db)/100)), 1)
            opr = float(success[valid, :top_percent].any(1).mean()*100)
            # 원본 get_recall 그대로 호출하되 별도 출력인 yaw 추정만 생략한다.
            # 장소 검색·거리·정답 목록·분모·집계 코드는 변경하지 않는다.
            old_phase = ev.phase_corr
            ev.phase_corr = lambda *args: (torch.tensor([0.0]), None)
            try:
                original = ev.get_recall(m, n, databases, queries, qsets,
                    [np.zeros((len(x), 1)) for x in databases],
                    [np.zeros((len(x), 1)) for x in queries],
                    [np.zeros((len(x), 1)) for x in databases],
                    [np.zeros((len(x), 1)) for x in queries], None)
            finally:
                ev.phase_corr = old_phase
            assert np.allclose(original[0], curve, atol=1e-10, rtol=0)
            assert abs(original[2]-opr) < 1e-10
            entry.update(status="complete", positive_queries=int(valid.sum()), correct_top1=int(success[valid, 0].sum()),
                         r1_percent=float(curve[0]), r1_percentile_percent=opr,
                         recall_at_1_to_25_percent=curve.tolist(), one_percent_k=top_percent,
                         effective_one_percent_k=min(25, top_percent), official_metric_agrees=True)
            rows.append(entry)
            ledgers = OUT / "ledgers" / variant
            ledgers.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ledgers / f"{DATES[m]}__{DATES[n]}.npz", distances=distances, indices=indices,
                                positive=valid, success=success, query_xy=query_xy[n], database_xy=db_xy[m])
    completed = [r for r in rows if r["status"] == "complete"]
    return {"variant": variant, "pairs": rows, "valid_pairs": len(completed), "required_pairs": 30,
            "complete_30_pair_evaluation": len(completed) == 30,
            "mean_over_valid_pairs": {"r1_percent": float(np.mean([r["r1_percent"] for r in completed])),
                                      "r1_percentile_percent": float(np.mean([r["r1_percentile_percent"] for r in completed]))} if completed else None}


def rel(path: Path) -> str:
    """리포 안이면 상대 경로, 밖(임시 검사 폴더)이면 절대 경로를 돌려준다."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    """파일 SHA-256을 1MiB씩 읽어 계산한다."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ready_dates() -> list[str]:
    """자료 준비가 끝났고 아직 추론하지 않은 날짜(락을 얻은 시점 기준)."""
    return [d for d in DATES if (OUT / "native_data" / d / "complete.json").exists()
            and not (OUT / "encoded" / d / "complete.json").exists()]


def run_encode(dates: list[str]) -> None:
    """자료가 준비된 날짜만 GPU로 추론한다. 호출자가 GPU0 락을 잡고 있어야 한다.

    `--dates auto`는 락을 얻은 뒤 준비된 날짜를 모두 고른다. 대기 중 새로 준비된 날짜도 한 번에 처리한다.
    """
    if dates == ["auto"]:
        dates = ready_dates()
        print(json.dumps({"auto_dates": dates}, ensure_ascii=False), flush=True)
        if not dates:
            return
    import torch
    from _disco_official_dump import official_model
    torch.set_num_threads(2)
    # 공개 FP32 경로와 검산한다. TF32 사용 시 배치 크기만으로 약1e-4 차이가 났다.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ck = OUT.parent / "weights/disco_author_model.ckpt"
    assert sha256(ck) == EXPECTED_SHA
    cfg, _, ev, model = official_model(ck)
    import gputransform
    device = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "name": torch.cuda.get_device_name(0),
              "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}
    write(OUT / "evaluation_protocol.json", {"dates": DATES, "checkpoint_sha256": EXPECTED_SHA,
          "primary": "released 3<=d<3.5m sampler", "diagnostic": "d>=3m; upper-bound freeze diagnosis, not original benchmark",
          "geometry_radius_m": RADIUS, "queries": "x(-50,150),y(-250,150); self session excluded",
          "metric": "official raw1024D L2 KDTree; positive queries only; ordered30pair macro mean",
          "precision": "FP32; TF32 disabled; full and partial official batches checked",
          "paper": PAPER, "gpu_serialization": "호스트 flock /tmp/nsd_native_reproduction_gpu0.lock 안에서만 encode 실행",
          "remaining_limits": "저자의 생성된 스캔 목록은 공개되지 않아 원시 자료에서 재생성한다. PR-AUC·yaw 정확도는 이 실행의 산출물이 아니다."})
    consistency = PROTOCOL / "encode_consistency.json"
    if not consistency.exists() and (OUT / "encoded" / DATES[0] / "complete.json").exists():
        # 03-17은 더 이전 판에서 추론했으므로 같은 프로세스·설정으로 일부를 다시 추론해 기록한다.
        # 이 검사의 오류는 주 추론을 막지 않고 기록만 한다. verify가 통과 여부를 검산 항목으로 본다.
        try:
            recheck = recheck_encoded(DATES[0], model, gputransform)
        except Exception as exc:
            recheck = {"date": DATES[0], "passed": False, "max_abs_difference": None,
                       "error": f"{type(exc).__name__}: {exc}"[:500], "traceback": traceback.format_exc()[-2000:]}
        recheck["device"] = device
        write(consistency, recheck)
        print(json.dumps({"encode_consistency": {k: recheck[k] for k in ["date", "max_abs_difference", "passed"]}},
                         ensure_ascii=False), flush=True)
    for date in dates:
        if not (OUT / "native_data" / date / "complete.json").exists():
            raise RuntimeError(f"{date}: 자료 준비 전 encode 요청")
        encode(date, model, cfg, ev, gputransform, device)
    encoded = [d for d in DATES if (OUT / "encoded" / d / "complete.json").exists()]
    write(OUT / "pipeline_status.json", {"stage": "encoded", "encoded": encoded,
          "waiting": [d for d in DATES if d not in encoded], "device": device, "updated_unix": time.time()})


def run_score() -> dict:
    """여섯 세션 추론이 끝난 뒤 CPU에서 공식 get_recall과 독립 벡터 집계를 대조한다."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from _disco_official_dump import official_model
    from _disco_official_prep import fake_gputransform_module
    torch.set_num_threads(2)
    missing = [d for d in DATES if not (OUT / "encoded" / d / "complete.json").exists()]
    if missing:
        raise RuntimeError(f"추론 미완료 세션: {missing}")
    _, _, ev, _ = official_model(None, cpu_only=True, fake=fake_gputransform_module())
    assert str(ev.device) == "cpu"
    report = {"released": score("released", ev), "no_upper_bound_diagnostic": score("no_upper_bound", ev),
              "scope": "공개 모델·원시 자료 재생성에 의한 재현 진단; 주표 변경 또는 추가 학습 아님"}
    write(OUT / "report.json", report)
    write(OUT / "pipeline_status.json", {"stage": "scored", "report": rel(OUT / "report.json"),
                                         "updated_unix": time.time()})
    return report


def official_selection(names: list[str], gt_path: Path, upper_bound: bool = True) -> list[dict]:
    """공식 생성기 함수 원문(AST)과 pandas 읽기로 표본을 다시 고른다.

    `sample()`의 numpy 구현과 독립적인 대조다. 상한 제거 진단은 같은 반복에서
    check_submap_test 대신 거리≥SUBMAP_INTERVAL_TEST만 쓴다.
    """
    import ast
    import importlib.util
    import pandas as pd
    from _disco_official_prep import GENERATOR
    spec = importlib.util.spec_from_file_location("disco_gen_config_verify", GENERATOR.parent / "config.py")
    gcfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gcfg)
    tree = ast.parse(GENERATOR.read_text(), str(GENERATOR))
    want = {"check_in_test_set", "check_submap_test", "find_closest_timestamp"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in want]
    region = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "p" for t in node.targets)]
    assert {node.name for node in body} == want and len(region) == 1
    ns = {"np": np, "cfg": gcfg}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(GENERATOR), "exec"), ns)
    df = pd.read_csv(gt_path, header=0, names=["timestamp", "northing", "easting", "height", "roll", "pitch", "yaw"],
                     low_memory=False)
    df["timestamp"] = df["timestamp"].astype(int)
    stamps = df["timestamp"].values
    rows, first, prev = [], False, None
    for name in sorted(Path(n).name for n in names):
        ts = int(os.path.splitext(name)[0])
        loc = ns["find_closest_timestamp"](stamps, ts)
        northing, easting, yaw = df["northing"][loc], df["easting"][loc], df["yaw"][loc]
        if np.isnan(float(northing)) or np.isnan(float(easting)):
            continue
        elif not first:
            prev, first = (float(northing), float(easting)), True
        if upper_bound:
            take = ns["check_submap_test"](float(northing), float(easting), prev[0], prev[1])
        else:
            take = np.sqrt((prev[0]-float(northing))**2 + (prev[1]-float(easting))**2) >= gcfg.SUBMAP_INTERVAL_TEST
        if take:
            rows.append({"timestamp": ts, "xy": [float(northing), float(easting)], "yaw": float(yaw),
                         "test_region": bool(ns["check_in_test_set"](float(northing), float(easting), region[0]))})
            prev = (float(northing), float(easting))
    return rows


def scan_names(date: str) -> list[str]:
    """공식 생성기가 os.listdir로 볼 velodyne_sync 파일 이름 전체를 돌려준다."""
    members = OUT / "native_data" / date / "members.json"
    if members.exists():
        return [Path(name).name for name in json.loads(members.read_text())["members"]]
    listing = PROTOCOL / "inputs" / f"{date}_velodyne_sync_names.txt"
    return listing.read_text().split()


def verify_selection(dates: list[str]) -> dict:
    """저장한 두 표본 목록을 공식 함수 재생과 전수 비교한다."""
    out = {}
    for date in dates:
        names = scan_names(date)
        gt = OUT / "native_data" / date / "groundtruth.csv"
        row = {"all_scans": len(names), "groundtruth_sha256": sha256(gt)}
        for variant, filename, upper in [("released", "selection.json", True),
                                         ("no_upper_bound", "selection_no_upper_bound.json", False)]:
            stored = json.loads((OUT / "native_data" / date / filename).read_text())["rows"]
            replay = official_selection(names, gt, upper)
            same = [r["timestamp"] for r in stored] == [r["timestamp"] for r in replay]
            # np.loadtxt와 pandas 파서는 마지막 자리에서 다를 수 있어 좌표는 차이 크기를 기록한다.
            xy_diff = float(max((abs(a - b) for s, r in zip(stored, replay) for a, b in zip(s["xy"], r["xy"])),
                                default=0.0)) if same else None
            exact = same and xy_diff <= 1e-9 and all(s["yaw"] == r["yaw"] and s["test_region"] == r["test_region"]
                                                     for s, r in zip(stored, replay))
            row[variant] = {"stored": len(stored), "replayed": len(replay), "timestamps_equal": same,
                            "xy_max_abs_difference_m": xy_diff, "xy_yaw_region_exact": exact,
                            "queries_in_test_region": sum(r["test_region"] for r in stored),
                            "selection_sha256": sha256(OUT / "native_data" / date / filename)}
        out[date] = row
    return out


def brute_force_pair(db: np.ndarray, query: np.ndarray, db_xy: np.ndarray, query_xy: np.ndarray) -> dict:
    """KDTree·sklearn·공식 코드 없이 정답과 L2 순위를 float64로 다시 계산한다."""
    geometry = np.sqrt(((query_xy[:, None, :] - db_xy[None, :, :])**2).sum(-1))
    positive = geometry <= RADIUS
    db64, q64 = db.astype(np.float64), query.astype(np.float64)
    approx = (q64**2).sum(1)[:, None] + (db64**2).sum(1)[None, :] - 2 * q64 @ db64.T
    order = np.argsort(approx, axis=1, kind="stable")[:, :NEIGHBORS + 5]
    exact = np.stack([((db64[order[i]] - q64[i])**2).sum(1) for i in range(len(q64))])
    resorted = np.take_along_axis(order, np.argsort(exact, axis=1, kind="stable"), 1)[:, :NEIGHBORS]
    distance = np.sqrt(np.sort(exact, axis=1)[:, :NEIGHBORS])
    return {"positive": positive, "indices": resorted, "distances": distance,
            "boundary_pairs_1e-9m": int((np.abs(geometry - RADIUS) < 1e-9).sum())}


def verify_retrieval(report: dict) -> dict:
    """저장 서술자와 표본 목록만으로 30쌍 지표를 다시 계산해 보고서·원장과 비교한다."""
    desc, lookup = {}, {}
    for date in DATES:
        desc[date] = np.load(OUT / "encoded" / date / "descriptors.npy")
        keys = json.loads((OUT / "encoded" / date / "rows.json").read_text())["keys"]
        lookup[date] = {int(k): i for i, k in enumerate(keys)}
    out = {}
    for variant, key, filename in [("released", "released", "selection.json"),
                                   ("no_upper_bound", "no_upper_bound_diagnostic", "selection_no_upper_bound.json")]:
        sel = {d: json.loads((OUT / "native_data" / d / filename).read_text())["rows"] for d in DATES}
        stored_pairs = {(p["database"], p["query"]): p for p in report[key]["pairs"]}
        pairs, worst = [], {"r1": 0.0, "opr": 0.0, "curve": 0.0}
        for m in DATES:
            db_rows = [r for r in sel[m] if not r["test_region"]] + [r for r in sel[m] if r["test_region"]]
            for n in DATES:
                if m == n:
                    continue
                q_rows = [r for r in sel[n] if r["test_region"]]
                stored = stored_pairs[(m, n)]
                entry = {"database": m, "query": n, "database_frames": len(db_rows), "query_frames": len(q_rows)}
                if len(db_rows) < NEIGHBORS or not q_rows:
                    entry.update(status="unusable", stored_status=stored["status"],
                                 status_agrees=stored["status"] != "complete")
                    pairs.append(entry)
                    continue
                db = desc[m][[lookup[m][r["timestamp"]] for r in db_rows]]
                query = desc[n][[lookup[n][r["timestamp"]] for r in q_rows]]
                result = brute_force_pair(db, query, np.array([r["xy"] for r in db_rows]),
                                          np.array([r["xy"] for r in q_rows]))
                valid = result["positive"].any(1)
                if not valid.any():
                    entry.update(status="unusable", stored_status=stored["status"],
                                 status_agrees=stored["status"] != "complete")
                    pairs.append(entry)
                    continue
                success = np.take_along_axis(result["positive"], result["indices"], 1)
                curve = np.maximum.accumulate(success[valid], axis=1).mean(0) * 100
                k = max(int(round(len(db_rows) / 100.0)), 1)
                opr = float(success[valid, :k].any(1).mean() * 100)
                entry.update(status="complete", stored_status=stored["status"],
                             positive_queries=int(valid.sum()), correct_top1=int(success[valid, 0].sum()),
                             r1_percent=float(curve[0]), r1_percentile_percent=opr,
                             boundary_pairs_1e9m=result["boundary_pairs_1e-9m"])
                if stored["status"] != "complete":
                    entry.update(status_agrees=False, positive_mask_equal=False, top1_index_mismatches=len(q_rows),
                                 top1_nontie_mismatches=len(q_rows), top25_order_mismatches=len(q_rows))
                    pairs.append(entry)
                    continue
                ledger = np.load(OUT / "ledgers" / variant / f"{m}__{n}.npz")
                top1_same = ledger["indices"][:, 0] == result["indices"][:, 0]
                # 서술자가 같아 거리가 같은 경우만 동률로 보고, 그 밖의 top-1 차이는 실패로 센다.
                nontie = 0
                for i in np.flatnonzero(~top1_same):
                    theirs = np.sqrt(((db[ledger["indices"][i, 0]].astype(np.float64) - query[i])**2).sum())
                    nontie += int(abs(theirs - result["distances"][i, 0]) > 1e-9 * max(1.0, theirs))
                order_same = (ledger["indices"] == result["indices"]).all(1)
                entry.update(status_agrees=True,
                             r1_abs_difference=abs(float(curve[0]) - stored["r1_percent"]),
                             opr_abs_difference=abs(opr - stored["r1_percentile_percent"]),
                             curve_max_abs_difference=float(np.max(np.abs(curve - np.array(stored["recall_at_1_to_25_percent"])))),
                             positive_mask_equal=bool(np.array_equal(valid, ledger["positive"])),
                             top1_index_mismatches=int((~top1_same).sum()), top1_nontie_mismatches=nontie,
                             top25_order_mismatches=int((~order_same).sum()),
                             max_distance_abs_difference=float(np.max(np.abs(ledger["distances"] - result["distances"]))))
                for name, field in [("r1", "r1_abs_difference"), ("opr", "opr_abs_difference"),
                                    ("curve", "curve_max_abs_difference")]:
                    worst[name] = max(worst[name], entry[field])
                pairs.append(entry)
        done = [p for p in pairs if p["status"] == "complete"]
        stored_mean = report[key]["mean_over_valid_pairs"] or {}
        mean_r1 = float(np.mean([p["r1_percent"] for p in done])) if done else None
        mean_opr = float(np.mean([p["r1_percentile_percent"] for p in done])) if done else None
        out[variant] = {"pairs": pairs, "complete_pairs": len(done),
                        "independent_mean_r1_percent": mean_r1, "independent_mean_r1_percentile_percent": mean_opr,
                        "mean_abs_difference": max(abs(mean_r1 - stored_mean["r1_percent"]),
                                                   abs(mean_opr - stored_mean["r1_percentile_percent"])) if done and stored_mean else None,
                        "worst_abs_difference": worst,
                        "all_statuses_agree": all(p["status_agrees"] for p in pairs),
                        "all_positive_masks_equal": all(p["positive_mask_equal"] for p in done),
                        "top1_index_mismatches": sum(p["top1_index_mismatches"] for p in done),
                        "top1_nontie_mismatches": sum(p["top1_nontie_mismatches"] for p in done),
                        "geometry_boundary_pairs_1e9m": sum(p["boundary_pairs_1e9m"] for p in done),
                        "top25_order_mismatches": sum(p["top25_order_mismatches"] for p in done)}
    return out


def verify_inputs() -> dict:
    """가중치·원시 스캔·서술자 해시와 합집합 구성을 확인한다."""
    out = {"checkpoint_sha256": sha256(OUT.parent / "weights/disco_author_model.ckpt"), "dates": {}}
    for date in DATES:
        native = json.loads((OUT / "native_data" / date / "complete.json").read_text())
        encoded = json.loads((OUT / "encoded" / date / "complete.json").read_text())
        keys = json.loads((OUT / "encoded" / date / "rows.json").read_text())["keys"]
        union = sorted({str(r["timestamp"]) for f in ["selection.json", "selection_no_upper_bound.json"]
                        for r in json.loads((OUT / "native_data" / date / f).read_text())["rows"]}, key=int)
        recorded = {row["name"]: row["sha256"] for row in native["raw_sha256"]}
        mismatched = [key for key in keys if recorded.get(key + ".bin")
                      != sha256(OUT / "native_data" / date / "velodyne_sync" / (key + ".bin"))]
        desc = np.load(OUT / "encoded" / date / "descriptors.npy")
        out["dates"][date] = {"union_keys_equal_encoded": keys == union, "encoded": len(keys),
                              "raw_files_rehashed": len(keys), "raw_sha256_mismatches": len(mismatched),
                              "descriptor_shape": list(desc.shape), "descriptor_finite": bool(np.isfinite(desc).all()),
                              "descriptors_sha256": sha256(OUT / "encoded" / date / "descriptors.npy"),
                              "official_inference_max_abs_difference": encoded.get("official_inference_max_abs_difference"),
                              "official_polar_spot_checks": len(encoded.get("checks", [])),
                              "device": encoded.get("device"), "network_bytes": native.get("network_bytes"),
                              "archive_bytes": native.get("archive_bytes")}
    return out


def verdict(metrics: dict) -> list[str]:
    """논문 값과의 비교를 미리 정한 문장 틀로 쓴다(값에 따라 설정을 고르지 않는다)."""
    primary, diagnostic = metrics["primary_released_sampler"], metrics["diagnostic_no_upper_bound"]
    if not primary["complete_30_pair_evaluation"]:
        return ["30쌍이 모두 완료되지 않아 논문 평균과 직접 비교하지 않는다"]
    # 논문 표기 정밀도(소수 둘째 자리) 안에서 같을 때만 수치 재현으로 본다.
    reproduced = abs(primary["paper_minus_ours_r1_pp"]) <= 0.005 and abs(primary["paper_minus_ours_r1_percentile_pp"]) <= 0.005
    gap, gap_opr = primary["paper_minus_ours_r1_pp"], primary["paper_minus_ours_r1_percentile_pp"]
    closer = abs(diagnostic["paper_minus_ours_r1_pp"]) < abs(gap)
    lines = [f"공개 가중치·공개 생성기 재생 조건(주 진단)의 R@1 {primary['mean_r1_percent']:.2f}%·R@1% "
             f"{primary['mean_r1_percentile_percent']:.2f}%는 논문 {PAPER['r1_percent']}%·{PAPER['r1_percentile_percent']}%"
             + ("와 논문 표기 정밀도 안에서 같다" if reproduced else
                f"와 {-gap:+.2f}%p·{-gap_opr:+.2f}%p 다르다. 논문 수치의 재현은 확인되지 않았다"),
             "공식 디코더·극좌표 입력·get_latent_vectors·get_recall과의 일치는 확인했으므로 공개 코드 실행은 재현됐다. 우리 분할 재학습 결과가 아니다",
             f"평가 전 고정한 상한 제거 진단은 R@1 {diagnostic['mean_r1_percent']:.2f}%·R@1% {diagnostic['mean_r1_percentile_percent']:.2f}%다. "
             + ("R@1의 논문과의 차이가 주 진단보다 작지만 원 논문 재현값으로 대체하지 않는다" if closer else
                "R@1의 논문과의 차이가 주 진단보다 커서, 상한 때문에 표본 선택이 멈추는 현상으로 논문과의 차이를 설명할 수 없다"),
             "남은 원인 후보는 저자가 생성한 표본 목록(pickle) 미공개와 공개 가중치(epoch 9, iter 67719)와 논문 최종 가중치의 동일성 미확인이다. "
             "어느 쪽도 이 실행으로 검정하지 않았고, 시험 결과에 맞춘 설정 탐색은 하지 않았다"]
    return lines


def build_report(verification: dict) -> dict:
    """원장 폴더의 보고서와 검산을 합쳐 root가 읽을 최종 요약을 만든다."""
    report = json.loads((OUT / "report.json").read_text())
    scripts = ["_review_remaining_native_disco_data.py", "_review_remaining_native_disco_eval.py",
               "_review_remaining_native_disco_supervise.py", "_disco_official_prep.py", "_disco_official_dump.py"]
    metrics = {}
    for key, label in [("released", "primary_released_sampler"), ("no_upper_bound_diagnostic", "diagnostic_no_upper_bound")]:
        block = report[key]
        mean = block["mean_over_valid_pairs"] or {}
        metrics[label] = {"valid_pairs": block["valid_pairs"], "required_pairs": 30,
                          "complete_30_pair_evaluation": block["complete_30_pair_evaluation"],
                          "failed_pairs": [f"{p['database']}->{p['query']}: {p['reason']}" for p in block["pairs"]
                                           if p["status"] != "complete"],
                          "mean_r1_percent": mean.get("r1_percent"),
                          "mean_r1_percentile_percent": mean.get("r1_percentile_percent"),
                          "paper_minus_ours_r1_pp": PAPER["r1_percent"] - mean["r1_percent"] if mean else None,
                          "paper_minus_ours_r1_percentile_pp": PAPER["r1_percentile_percent"] - mean["r1_percentile_percent"] if mean else None,
                          "positive_queries_total": sum(p.get("positive_queries", 0) for p in block["pairs"])}
    side = PROTOCOL / "side_analysis.json"
    operations = PROTOCOL / "operations.json"
    local = PROTOCOL / "local_check.json"
    summary = {"task": "DiSCO 저자 공개 가중치의 원 논문 NCLT 6세션 30쌍 평가 재현",
               "classification": "공개 모델·공개 코드·공식 원시 자료 재생성에 의한 논문 조건 재현 시도. 우리 분할 재학습 아님",
               "paper": PAPER, "metrics": metrics, "verdict_ko": verdict(metrics),
               "max_one_percent_k": max(p["one_percent_k"] for key in ["released", "no_upper_bound_diagnostic"]
                                        for p in report[key]["pairs"] if p["status"] == "complete"),
               "interpretation_guard": ["primary는 공개 생성기 3≤d<3.5m 그대로다. 상한 제거 진단은 평가 전 고정했고 원 논문 재현값으로 대체하지 않는다",
                                        "빈 질의·정답 없는 쌍은 실패로 표시하고 평균에서 0으로 메우지 않는다. 30쌍이 모두 완료되지 않으면 공식 평균과 같은 값이 아니다",
                                        "R@1%는 공식 get_recall처럼 KDTree 상위25 안에서만 판정한다(지도 2,500장 초과 시 1%가 25를 넘음)",
                                        "yaw 추정·PR-AUC는 산출하지 않았다"],
               "commands": {"supervisor": "setsid nohup python3 scripts/_review_remaining_native_disco_supervise.py (서버 호스트)",
                            "fetch_retry": "docker exec ring_baseline python3 -u scripts/_review_remaining_native_disco_data.py fetch --dates DATE",
                            "encode": "flock -x /tmp/nsd_native_reproduction_gpu0.lock docker exec -e CUDA_VISIBLE_DEVICES=0 -e NVIDIA_TF32_OVERRIDE=0 ring_baseline python3 -u scripts/_review_remaining_native_disco_eval.py encode --dates auto (03-17은 앞선 평가기 판에서 추론했고 complete.json에 device 필드가 없다)",
                            "score": "docker exec -e CUDA_VISIBLE_DEVICES= ring_baseline python3 -u scripts/_review_remaining_native_disco_eval.py score",
                            "verify": "docker exec -e CUDA_VISIBLE_DEVICES= ring_baseline python3 -u scripts/_review_remaining_native_disco_eval.py verify"},
               "code_sha256": {name: sha256(REPO / "scripts" / name) for name in scripts if (REPO / "scripts" / name).exists()},
               "sources": {"report": rel(OUT / "report.json"),
                           "ledgers": rel(OUT / "ledgers"),
                           "verification": rel(PROTOCOL / "verification.json")},
               "verification_passed": verification["passed"], "verification": {
                   k: v for k, v in verification.items() if k != "retrieval"},
               "verification_retrieval_summary": {v: {k: x for k, x in block.items() if k != "pairs"}
                                                  for v, block in verification["retrieval"].items()},
               "operations": json.loads(operations.read_text()) if operations.exists() else None,
               "local_check": json.loads(local.read_text()) if local.exists() else None,
               "side_analysis": json.loads(side.read_text()) if side.exists() else None}
    return summary


def report_text(summary: dict) -> str:
    """root가 바로 읽을 한국어 요약을 만든다."""
    lines = ["DiSCO 저자 공개 가중치 원 논문 재현 보고 (NCLT 6세션 순서쌍 30개, 1.5m, raw1024D L2)", "",
             "판정:"] + [f"  - {x}" for x in summary.get("verdict_ko", [])] + [""]
    for label, name in [("primary_released_sampler", "주 진단: 공개 생성기 3≤d<3.5m"),
                        ("diagnostic_no_upper_bound", "별도 진단: 상한 제거 3m 간격")]:
        m = summary["metrics"][label]
        lines.append(f"[{name}] 완료 쌍 {m['valid_pairs']}/30, 양성 질의 합 {m['positive_queries_total']}")
        if m["mean_r1_percent"] is not None:
            lines.append(f"  R@1 {m['mean_r1_percent']:.4f}% (논문 {PAPER['r1_percent']}%, 논문−본 실행 {m['paper_minus_ours_r1_pp']:+.4f}%p)")
            lines.append(f"  R@1% {m['mean_r1_percentile_percent']:.4f}% (논문 {PAPER['r1_percentile_percent']}%, 논문−본 실행 {m['paper_minus_ours_r1_percentile_pp']:+.4f}%p)")
        if m["failed_pairs"]:
            lines.append("  실패 쌍(평균 제외, 0으로 채우지 않음): " + "; ".join(m["failed_pairs"]))
        if not m["complete_30_pair_evaluation"]:
            lines.append("  30쌍이 모두 완료되지 않아 공식 30쌍 평균과 같은 값이 아니다.")
    v = summary["verification"]
    lines += ["", f"독립 검산 통과: {summary['verification_passed']}",
              f"  가중치 SHA-256 {v['inputs']['checkpoint_sha256']}"]
    recheck = v["inputs"].get("encode_consistency_2012-03-17", {})
    lines.append(f"  03-17 앞선 판 서술자 재추론 일관성: {recheck.get('passed')} "
                 f"({len(recheck.get('rechecked_indices', []))}장, 최대 차 {recheck.get('max_abs_difference')})")
    for date, row in v["inputs"]["dates"].items():
        sel = v["selection"][date]
        lines.append(f"  {date}: 전체 {sel['all_scans']}스캔, 주 표본 {sel['released']['stored']}(질의 {sel['released']['queries_in_test_region']}),"
                     f" 진단 표본 {sel['no_upper_bound']['stored']}(질의 {sel['no_upper_bound']['queries_in_test_region']}),"
                     f" 추론 {row['encoded']}, 원시 해시 불일치 {row['raw_sha256_mismatches']}, 공식 함수 표본 재생 일치 "
                     f"{sel['released']['xy_yaw_region_exact'] and sel['no_upper_bound']['xy_yaw_region_exact']}")
    for variant, block in summary["verification_retrieval_summary"].items():
        lines.append(f"  {variant}: 독립 L2 전수 재계산 완료 쌍 {block['complete_pairs']}, 최대 차 {block['worst_abs_difference']},"
                     f" top1 색인 불일치 {block['top1_index_mismatches']}, 상위25 순서 불일치 {block['top25_order_mismatches']}")
    local = summary.get("local_check")
    if local:
        lines.append(f"  로컬 산술 검산(local_check.json, 작은 JSON만): 통과 {local['passed']}, 쌍별 R@1 재계산 최대 차 "
                     + ", ".join(f"{k} {v['r1_from_counts_max_abs_difference']:.1e}" for k, v in local["variants"].items()))
    lines += ["", "해석 경계:"] + [f"  - {x}" for x in summary["interpretation_guard"]]
    if "max_one_percent_k" in summary:
        k = summary["max_one_percent_k"]
        lines.append(f"  - 이번 실행의 1% 상위 수는 최대 {k}" + ("라 상위25 제한은 작동하지 않았다" if k <= NEIGHBORS
                                                       else "라 상위25 제한이 작동한 쌍이 있다"))
    if summary.get("operations"):
        lines += ["", "실행 경과(operations.json):"] + [f"  - {x}" for x in summary["operations"].get("summary_ko", [])]
    if summary.get("side_analysis"):
        lines += ["", "SC++/SeqOT 조건 대응(자료 대기 중 분석, 근거는 side_analysis.json):"]
        lines += [f"  - {x}" for x in summary["side_analysis"].get("summary_ko", [])]
    return "\n".join(lines) + "\n"


def run_verify(dates: list[str], selection_only: bool, report_only: bool = False) -> dict:
    """독립 검산을 수행하고 지정 결과 폴더에 보고서를 쓴다.

    `report_only`는 저장된 verification.json을 다시 계산하지 않고 보고서(report.json·report_ko.txt)만 다시 만든다.
    """
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    if report_only:
        verification = json.loads((PROTOCOL / "verification.json").read_text())
        summary = build_report(verification)
        summary["verification_sha256"] = sha256(PROTOCOL / "verification.json")
        write(PROTOCOL / "report.json", summary)
        (PROTOCOL / "report_ko.txt").write_text(report_text(summary))
        return verification
    selection = verify_selection(dates)
    if selection_only:
        result = {"selection": selection, "passed": all(
            row[v]["xy_yaw_region_exact"] for row in selection.values() for v in ["released", "no_upper_bound"])}
        write(PROTOCOL / "selection_replay.json", result)
        return result
    report = json.loads((OUT / "report.json").read_text())
    inputs = verify_inputs()
    retrieval = verify_retrieval(report)
    consistency = json.loads((PROTOCOL / "encode_consistency.json").read_text()) \
        if (PROTOCOL / "encode_consistency.json").exists() else {"passed": False, "reason": "재추론 기록 없음"}
    inputs["encode_consistency_2012-03-17"] = consistency
    checks = {
        "encode_consistency_2012-03-17": bool(consistency.get("passed")),
        "checkpoint": inputs["checkpoint_sha256"] == EXPECTED_SHA,
        "selection_replay": all(row[v]["xy_yaw_region_exact"] for row in selection.values()
                                for v in ["released", "no_upper_bound"]),
        "inputs": all(row["union_keys_equal_encoded"] and row["raw_sha256_mismatches"] == 0 and row["descriptor_finite"]
                      for row in inputs["dates"].values()),
        "retrieval": all(block["all_statuses_agree"] and block["all_positive_masks_equal"]
                         and block["top1_nontie_mismatches"] == 0
                         and (block["mean_abs_difference"] is None or block["mean_abs_difference"] < 1e-9)
                         and max(block["worst_abs_difference"].values()) < 1e-9 for block in retrieval.values()),
    }
    verification = {"selection": selection, "inputs": inputs, "retrieval": retrieval, "checks": checks,
                    "passed": all(checks.values())}
    write(PROTOCOL / "verification.json", verification)
    summary = build_report(verification)
    write(PROTOCOL / "report.json", summary)
    (PROTOCOL / "report_ko.txt").write_text(report_text(summary))
    return verification


def main() -> None:
    """encode(GPU, 호스트 락 안), score(CPU), verify(CPU) 중 하나를 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["encode", "score", "verify"])
    parser.add_argument("--dates", nargs="+", default=DATES)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="verify: 저장된 검산으로 보고서만 다시 만든다")
    args = parser.parse_args()
    if args.stage == "encode":
        run_encode(args.dates)
    elif args.stage == "score":
        report = run_score()
        print(json.dumps({k: v["mean_over_valid_pairs"] for k, v in report.items() if isinstance(v, dict)},
                         ensure_ascii=False), flush=True)
    else:
        result = run_verify(args.dates, args.selection_only, args.report_only)
        print(json.dumps({"passed": result["passed"]}, ensure_ascii=False), flush=True)
        if not result["passed"]:
            raise SystemExit("독립 검산 실패")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write(OUT / "pipeline_error.json", {"type": type(exc).__name__, "message": str(exc), "argv": sys.argv,
                                            "traceback": traceback.format_exc(), "unix": time.time()})
        raise
