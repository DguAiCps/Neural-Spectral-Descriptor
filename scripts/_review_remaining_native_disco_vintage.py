#!/usr/bin/env python3
"""DiSCO 공개 가중치 업로드 당시 전처리(자차 5m 상자 제외, 8e79286)와 현재판(1m)의 원인 대조.

사전 고정 조건은 results/pdv2_reproduction_orchestration_20261106/disco_protocol/
preprocess_vintage_20261107/protocol.json에 있다. 같은 가중치·표본·30쌍·양성 목록을 유지하고
load_lidar_file_nclt의 상자 반폭만 1m→5m로 바꾼다. 1m 서술자·원장은 기존 결과를 재사용한다.

서버 호스트(표준 라이브러리만)에서 단계 전체를 감독한다.
cd /home/qkrrlans00/Neural-Spectral-Codec && setsid nohup python3 -u \
    scripts/_review_remaining_native_disco_vintage.py run \
    > results/pdv2_reproduction_orchestration_20261106/disco_protocol/preprocess_vintage_20261107/run.log 2>&1 &
encode는 호스트 flock /tmp/nsd_native_reproduction_gpu0.lock 안에서만 컨테이너로 부른다.
score·verify는 CPU만 쓴다(-e CUDA_VISIBLE_DEVICES=). 다시 실행하면 완료 표시가 있는 단계는 건너뛴다.
이 진단은 주표 설정이나 원 논문 재현값으로 채택하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "results/pdv2_remaining_native_20261106/disco"
VINTAGE = REPO / "results/pdv2_reproduction_orchestration_20261106/disco_protocol/preprocess_vintage_20261107"
ENCODED = VINTAGE / "encoded_ego5m"
LEDGERS = VINTAGE / "ledgers" / "ego5m"
HISTORICAL = VINTAGE / "generator_8e79286.py"
HISTORICAL_SHA = "3b1ce65f6b9ce147e5fcd2a86b847ffd29d8395f33c056148df1a53d194a10f4"
LOCK = Path("/tmp/nsd_native_reproduction_gpu0.lock")
DATES = ["2012-03-17", "2012-05-26", "2012-06-15", "2012-08-20", "2012-09-28", "2012-10-28"]
VARIANTS = [("released", "selection.json", "released"),
            ("no_upper_bound", "selection_no_upper_bound.json", "no_upper_bound_diagnostic")]
HALF_EGO5M = 5.
HALF_EGO1M = 1.
MIN_FREE_GIB = 20
EXEC = ["docker", "exec", "-w", "/workspace/Neural-Spectral-Codec"]
THREADS = ["-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1"]
CPU_ENV = ["-e", "CUDA_VISIBLE_DEVICES="] + THREADS
GPU_ENV = ["-e", "CUDA_VISIBLE_DEVICES=0", "-e", "NVIDIA_TF32_OVERRIDE=0"] + THREADS
SELF = ["ring_baseline", "python3", "-u", "scripts/_review_remaining_native_disco_vintage.py"]


def write(path: Path, value: dict) -> None:
    """JSON을 원자적으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read(path: Path) -> dict | None:
    """쓰는 중인 파일은 건너뛰고 JSON을 읽는다."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def sha256(path: Path) -> str:
    """파일 SHA-256을 1MiB씩 읽어 계산한다."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status(stage: str, **detail) -> None:
    """status.json에 현재 단계와 사건 기록을 남긴다."""
    path = VINTAGE / "status.json"
    state = read(path) or {"events": []}
    state.update(stage=stage, updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), updated_unix=time.time(), **detail)
    state["events"] = (state.get("events", []) + [{"time": state["updated"], "unix": state["updated_unix"],
                                                   "stage": stage, **detail}])[-40:]
    write(path, state)


# ------------------------------------------------------------ 전처리(컨테이너)
def decode(raw: Path):
    """공식 NCLT 디코더와 같은 float64 연산(scripts/_review_remaining_native_disco_eval.py:45)."""
    import numpy as np
    packed = np.fromfile(raw, dtype=np.dtype([("xyz", "<u2", (3,)), ("intensity", "u1"), ("laser", "u1")]))
    points = packed["xyz"].astype(np.float64) * 0.005 + -100.0
    points[:, 2] = -points[:, 2]
    return points


def box_filter(p, half: float):
    """scripts/_disco_official_prep.py:303 official_filter에서 자차 상자 반폭만 인자로 받는다."""
    import numpy as np
    x, y, z = p[:, 0], p[:, 1], -p[:, 2]
    keep = (np.abs(x) < 70.) & (z > -20.) & (z < -2.) & (np.abs(y) < 70.) & \
        ~((np.abs(x) < half) & (np.abs(y) < half))
    hits = np.stack([x[keep] / 70., y[keep] / 70., z[keep] / 20.], axis=1)
    hits[:, 2] = -hits[:, 2]
    return hits, keep


def historical_functions(gputransform_mod):
    """8e79286 원문 생성기의 convert/load_lidar_file_nclt/load_pc_file만 AST로 실행한다."""
    import ast
    import importlib.util
    import struct
    import numpy as np
    from _disco_official_prep import GENERATOR
    assert sha256(HISTORICAL) == HISTORICAL_SHA
    source = HISTORICAL.read_text()
    assert "not(np.abs(x) < 5. and np.abs(y) < 5.)" in source
    spec = importlib.util.spec_from_file_location("disco_gen_config_8e79286", GENERATOR.parent / "config.py")
    gcfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gcfg)
    tree = ast.parse(source, str(HISTORICAL))
    want = {"convert", "load_lidar_file_nclt", "load_pc_file"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert {n.name for n in body} == want
    ns = {"np": np, "struct": struct, "os": os, "cfg": gcfg, "gputransform": gputransform_mod}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(HISTORICAL), "exec"), ns)
    return ns


def encode_date(date: str, model, ev, gputransform, gen_cfg, history, device: dict) -> dict:
    """1m 기존 추론과 같은 합집합 순서로 5m 입력을 추론하고 검산·시간을 기록한다."""
    import numpy as np
    import torch
    from _disco_official_prep import official_polar, polar_agrees, polar_numpy
    target = ENCODED / date
    if (target / "complete.json").exists():
        return read(target / "complete.json")
    target.mkdir(parents=True, exist_ok=True)
    meta1 = json.loads((DATA / "encoded" / date / "rows.json").read_text())
    keys = meta1["keys"]
    stored1 = np.load(DATA / "encoded" / date / "descriptors.npy")
    assert stored1.shape == (len(keys), 1024)
    started = time.time()
    t_pre = t_diag = t_gpu = t_check = 0.0
    # protocol_amendment_1.json: scan_counts.json에 미리 기록된 점 1개 이하 스캔만 허용한다.
    allowed = {str(r["timestamp"]) for r in degenerate_scans() if r["date"] == date}
    degenerate_seen = []
    status("encoding", date=date, done=0, total=len(keys))
    pick = set(np.linspace(0, len(keys) - 1, 3).round().astype(int).tolist())
    checks, descriptors, pending, spot5, spot1 = [], [], [], [], []
    removed, cells, rings, rel_cells = [], [], [], []
    cpu_cuda_max = 0
    model.eval()
    for i, key in enumerate(keys):
        raw = DATA / "native_data" / date / "velodyne_sync" / (key + ".bin")
        t = time.perf_counter()
        points = decode(raw)
        hits5, keep5 = box_filter(points, HALF_EGO5M)
        degenerate = len(hits5) <= 1
        assert not degenerate or key in allowed, f"5m 필터 뒤 빈 스캔(사전 목록 밖): {raw}"
        polar5 = official_polar(hits5, gputransform, gen_cfg)
        reference = polar_numpy(hits5.T.flatten().astype(np.float32), len(hits5)).reshape(-1, 3)[:, 2]
        ok, delta = polar_agrees(polar5, reference.reshape(polar5.shape))
        assert ok and (degenerate or polar5.sum() > 1), (date, key, delta)
        if degenerate:
            degenerate_seen.append({"timestamp": int(key), "ego5m_points": int(len(hits5)),
                                    "ego5m_occupied_cells": int(polar5.sum()),
                                    "treatment": "official_polar 빈 입력 규칙(전부 0 영상)" if not len(hits5) else "공식 경로 그대로"})
        cpu_cuda_max = max(cpu_cuda_max, delta)
        t_pre += time.perf_counter() - t
        # 진단 통계: 같은 원시 점의 1m 입력과의 차이. 추론 필수 경로가 아니므로 시간을 따로 잰다.
        t = time.perf_counter()
        hits1, keep1 = box_filter(points, HALF_EGO1M)
        assert not (keep5 & ~keep1).any()
        gone = keep1 & ~keep5
        gx, gy = points[gone, 0], points[gone, 1]
        assert ((np.abs(gx) < 5.) & (np.abs(gy) < 5.) & ~((np.abs(gx) < 1.) & (np.abs(gy) < 1.))).all()
        polar1 = official_polar(hits1, gputransform, gen_cfg)
        assert (polar5 <= polar1).all(), (date, key, "점 제거로 늘어난 칸")
        changed = polar1 != polar5
        removed.append(int(gone.sum()))
        cells.append(int(changed.sum()))
        rel_cells.append(float(changed.sum() / max(polar1.sum(), 1)))
        rings.append(int(np.nonzero(changed)[1].max()) if changed.any() else -1)
        t_diag += time.perf_counter() - t
        if i in pick and degenerate and not len(hits5):
            # 8e79286 원문은 0점 스캔에서 IndexError를 낸다(protocol_amendment_1.json). 그 동작을 기록한다.
            try:
                history["load_lidar_file_nclt"](str(raw))
                raised = None
            except IndexError as exc:
                raised = f"IndexError: {exc}"
            checks.append({"timestamp": int(key), "raw_sha256": sha256(raw), "historical_raises": raised,
                           "historical_hits_exact": None, "historical_polar_exact": None})
        elif i in pick:
            t = time.perf_counter()
            slow = history["load_lidar_file_nclt"](str(raw))
            assert np.array_equal(hits5, slow), (date, key, "8e79286 원문 디코더·필터 차이")
            slow_polar = history["load_pc_file"](str(raw))
            assert np.array_equal(polar5, slow_polar), (date, key, "8e79286 원문 입력 차이")
            checks.append({"timestamp": int(key), "raw_sha256": sha256(raw), "historical_hits_exact": True,
                           "historical_polar_exact": True, "cpu_cuda_differing_cells": delta,
                           "points_removed_vs_ego1m": int(gone.sum())})
            t_check += time.perf_counter() - t
        if i < 9:
            spot5.append(polar5)
            spot1.append(polar1)
        pending.append(polar5)
        if len(pending) == 7 or i == len(keys) - 1:
            t = time.perf_counter()
            with torch.no_grad():
                tensor = torch.from_numpy(np.array(pending, dtype=np.float32)).cuda()
                out = model(tensor)[0]
                torch.cuda.synchronize()
                descriptors.append(out.detach().cpu().numpy().reshape(len(pending), -1))
            t_gpu += time.perf_counter() - t
            pending = []
        if (i + 1) % 250 == 0:
            status("encoding", date=date, done=i + 1, total=len(keys))
    desc = np.concatenate(descriptors)
    assert desc.shape == (len(keys), 1024) and np.isfinite(desc).all()
    t = time.perf_counter()
    # 공식 get_latent_vectors(완전 배치 7 + 나머지 2)와 같은지, 같은 9장의 1m 재추론이 저장값과 같은지.
    old_loader = ev.load_pc_files
    ev.load_pc_files = lambda names: [spot5[int(name)] for name in names]
    try:
        exact = ev.get_latent_vectors(model, {i: {"query": str(i), "heading": 0.0} for i in range(9)})[0]
    finally:
        ev.load_pc_files = old_loader
        model.eval()
    assert np.allclose(exact, desc[:9], atol=2e-5, rtol=2e-5)
    with torch.no_grad():
        again1 = np.concatenate([model(torch.from_numpy(np.array(spot1[s:s + 7], dtype=np.float32)).cuda())[0]
                                 .detach().cpu().numpy().reshape(-1, 1024) for s in range(0, 9, 7)])
    assert np.allclose(again1, stored1[:9], atol=2e-5, rtol=2e-5), (date, "1m 재추론이 저장값과 다름")
    t_check += time.perf_counter() - t
    distinct = len(np.unique(np.round(desc, 6), axis=0))
    assert distinct >= 0.9 * len(desc)
    shift = np.sqrt(((desc.astype(np.float64) - stored1.astype(np.float64))**2).sum(1))
    np.save(target / "descriptors.npy", desc)
    write(target / "rows.json", {"keys": keys, "rows": meta1["rows"]})
    result = {"date": date, "n": len(keys), "descriptor_dim": 1024, "ego_box_half_m": HALF_EGO5M,
              "checkpoint_sha256": "8665e356f0aab2cf03fdc47aa2b3dd2d32be7a2b420bad35ac2c994f45d7808c",
              "historical_generator_sha256": HISTORICAL_SHA, "checks": checks,
              "all_inputs_cpu_cuda_checked": True, "cpu_cuda_max_differing_cells": cpu_cuda_max,
              "official_inference_max_abs_difference": float(np.max(np.abs(exact - desc[:9]))),
              "ego1m_reencode_max_abs_difference": float(np.max(np.abs(again1 - stored1[:9]))),
              "distinct_descriptors": distinct, "device": device, "degenerate_scans": degenerate_seen,
              "input_change": {"points_removed_mean": float(np.mean(removed)), "points_removed_max": int(max(removed)),
                               "scans_with_removed_points": int(sum(r > 0 for r in removed)),
                               "polar_cells_changed_mean": float(np.mean(cells)), "polar_cells_changed_max": int(max(cells)),
                               "polar_cells_changed_fraction_of_ego1m_occupied_mean": float(np.mean(rel_cells)),
                               "scans_with_changed_polar": int(sum(c > 0 for c in cells)),
                               "max_changed_ring_index": int(max(rings)),
                               "descriptor_l2_shift_median": float(np.median(shift)),
                               "descriptor_l2_shift_p90": float(np.percentile(shift, 90)),
                               "descriptor_l2_shift_max": float(shift.max())},
              "timing_s": {"preprocess_ego5m_filter_cuda_polar_cpu_check": t_pre, "gpu_forward_synchronized": t_gpu,
                           "diagnostic_ego1m_comparison": t_diag, "official_and_historical_checks": t_check,
                           "date_total_wall": time.time() - started}}
    write(target / "complete.json", result)
    return result


def run_encode() -> None:
    """GPU0 락 안에서 6세션 5m 입력을 추론한다. 호출자(run)가 flock을 잡는다."""
    process_start = time.time()
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO / "scripts"))
    from _disco_official_dump import official_model
    from _disco_official_prep import generator_functions
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ck = DATA.parent / "weights/disco_author_model.ckpt"
    assert sha256(ck) == "8665e356f0aab2cf03fdc47aa2b3dd2d32be7a2b420bad35ac2c994f45d7808c"
    t = time.time()
    cfg, _, ev, model = official_model(ck)
    import gputransform
    model_load = time.time() - t
    _, gen_cfg = generator_functions(gputransform)
    history = historical_functions(gputransform)
    device = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "name": torch.cuda.get_device_name(0),
              "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"), "torch": torch.__version__}
    results = {}
    for date in DATES:
        status("encoding", date=date)
        results[date] = encode_date(date, model, ev, gputransform, gen_cfg, history, device)
    write(VINTAGE / "encode_timing.json", {
        "process_import_to_end_s": time.time() - process_start, "model_load_s": model_load,
        "sessions": {d: r["timing_s"] for d, r in results.items()},
        "sum_gpu_forward_s": sum(r["timing_s"]["gpu_forward_synchronized"] for r in results.values()),
        "sum_preprocess_s": sum(r["timing_s"]["preprocess_ego5m_filter_cuda_polar_cpu_check"] for r in results.values()),
        "sum_diagnostic_s": sum(r["timing_s"]["diagnostic_ego1m_comparison"] for r in results.values()),
        "sum_checks_s": sum(r["timing_s"]["official_and_historical_checks"] for r in results.values()),
        "scans": sum(r["n"] for r in results.values()), "device": device})


def run_changed_check(count: int = 9) -> None:
    """검산 보강(지표 무관): 세션마다 저장 서술자의 5m−1m L2 변화가 가장 큰 9장을 골라
    8e79286 원문 load_pc_file·공식 get_latent_vectors(7+2 배치)와 저장 5m 서술자를 대조한다.

    encode의 공식 배치 대조는 각 세션 첫 9장이었고 그중 5m에서 바뀐 영상은 6세션 합 4장뿐이었다.
    호출자가 GPU0 락을 잡는다.
    """
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO / "scripts"))
    from _disco_official_dump import official_model
    from _disco_official_prep import generator_functions, official_polar
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.time()
    _, _, ev, model = official_model(DATA.parent / "weights/disco_author_model.ckpt")
    import gputransform
    _, gen_cfg = generator_functions(gputransform)
    history = historical_functions(gputransform)
    out = {}
    for date in DATES:
        keys = json.loads((ENCODED / date / "rows.json").read_text())["keys"]
        d5 = np.load(ENCODED / date / "descriptors.npy")
        d1 = np.load(DATA / "encoded" / date / "descriptors.npy")
        shift = np.sqrt(((d5.astype(np.float64) - d1.astype(np.float64))**2).sum(1))
        pick = np.argsort(-shift, kind="stable")[:count]
        images, exact_hist = [], []
        for i in pick:
            raw = DATA / "native_data" / date / "velodyne_sync" / (keys[i] + ".bin")
            hits5, _ = box_filter(decode(raw), HALF_EGO5M)
            img = official_polar(hits5, gputransform, gen_cfg)
            if len(hits5):
                exact_hist.append(bool(np.array_equal(img, history["load_pc_file"](str(raw)))))
            images.append(img)
        old_loader = ev.load_pc_files
        ev.load_pc_files = lambda names: [images[int(name)] for name in names]
        try:
            official = ev.get_latent_vectors(model, {i: {"query": str(i), "heading": 0.0} for i in range(len(pick))})[0]
        finally:
            ev.load_pc_files = old_loader
            model.eval()
        diff = float(np.max(np.abs(official - d5[pick])))
        out[date] = {"indices": pick.tolist(), "timestamps": [int(keys[i]) for i in pick],
                     "descriptor_l2_shift": shift[pick].round(6).tolist(),
                     "historical_polar_exact": exact_hist, "official_vs_stored_ego5m_max_abs_difference": diff,
                     "passed": bool(all(exact_hist) and np.allclose(official, d5[pick], atol=2e-5, rtol=2e-5))}
    write(VINTAGE / "changed_check.json", {
        "purpose": "검산 보강(지표 무관). 선택 기준은 저장 서술자의 5m−1m L2 변화 크기뿐이고 검색 결과를 보지 않는다",
        "sessions": out, "passed": all(v["passed"] for v in out.values()), "elapsed_s": time.time() - started})


def run_scale() -> None:
    """CPU만: 서술자 변화 크기를 견줄 척도로 원장의 1위 L2 거리와 1·2위 간격 중앙값을 잰다(protocol.json 사전 지정 통계)."""
    import numpy as np
    out = {}
    for variant, _, _ in VARIANTS:
        for label, folder in [("ego1m", DATA / "ledgers" / variant), ("ego5m", LEDGERS / variant)]:
            top1, gap = [], []
            for path in sorted(folder.glob("*.npz")):
                z = np.load(path)
                v = z["positive"]
                top1.append(z["distances"][v, 0])
                gap.append(z["distances"][v, 1] - z["distances"][v, 0])
            top1, gap = np.concatenate(top1), np.concatenate(gap)
            out[f"{variant}_{label}"] = {"ledgers": len(list(folder.glob("*.npz"))), "positive_query_instances": int(len(top1)),
                                         "top1_l2_median": float(np.median(top1)), "top1_top2_gap_median": float(np.median(gap))}
    shift = np.concatenate([np.sqrt(((np.load(ENCODED / d / "descriptors.npy").astype(np.float64)
                                      - np.load(DATA / "encoded" / d / "descriptors.npy").astype(np.float64))**2).sum(1))
                            for d in DATES])
    out["descriptor_l2_shift_all_scans"] = {"median": float(np.median(shift)), "p90": float(np.percentile(shift, 90)),
                                            "unchanged_scans": int((shift == 0).sum()), "scans": int(len(shift))}
    write(VINTAGE / "descriptor_scale.json", out)


def degenerate_scans() -> list[dict]:
    """scan_counts.json에 기록된 5m 필터 뒤 점 1개 이하 스캔(protocol_amendment_1.json)."""
    counts = read(VINTAGE / "scan_counts.json")
    assert counts is not None, "scan_counts.json 없음(scan_counts 단계 먼저)"
    return counts["ego5m_degenerate_le1_point"]


def run_scan_counts() -> None:
    """CPU만: 합집합 전 스캔의 원시·1m·5m 필터 뒤 점 수를 세고 표본 역할(지도·질의)을 기록한다."""
    import numpy as np
    out, degenerate = {}, []
    for date in DATES:
        keys = json.loads((DATA / "encoded" / date / "rows.json").read_text())["keys"]
        roles = {}
        for variant, filename, _ in VARIANTS:
            for r in json.loads((DATA / "native_data" / date / filename).read_text())["rows"]:
                roles.setdefault(str(r["timestamp"]), []).append(f"{variant}:{'query+map' if r['test_region'] else 'map'}")
        n1, n5 = [], []
        for key in keys:
            points = decode(DATA / "native_data" / date / "velodyne_sync" / (key + ".bin"))
            a, b = len(box_filter(points, HALF_EGO1M)[0]), len(box_filter(points, HALF_EGO5M)[0])
            n1.append(a)
            n5.append(b)
            if b <= 1:
                degenerate.append({"date": date, "timestamp": int(key), "raw_points": int(len(points)),
                                   "ego1m_points": a, "ego5m_points": b, "roles": roles.get(key, [])})
        n1, n5 = np.array(n1), np.array(n5)
        out[date] = {"scans": len(keys), "ego1m_points_min": int(n1.min()), "ego5m_points_min": int(n5.min()),
                     "ego1m_points_median": float(np.median(n1)), "ego5m_points_median": float(np.median(n5)),
                     "ego5m_le1": int((n5 <= 1).sum()), "ego5m_lt100": int((n5 < 100).sum()),
                     "ego1m_lt100": int((n1 < 100).sum())}
    write(VINTAGE / "scan_counts.json", {"sessions": out, "ego5m_degenerate_le1_point": degenerate,
                                         "note": "8e79286 원문 load_lidar_file_nclt는 점 0개면 hits[:, 2]에서 IndexError, 1개면 그대로 처리한다"})


# ------------------------------------------------------------ 집계(컨테이너, CPU)
def load_sets(encoded_dir: Path, filename: str, exclude: frozenset = frozenset()):
    """scripts/_review_remaining_native_disco_eval.py:156-169와 같은 지도·질의 배열을 만든다.

    exclude는 protocol_amendment_1.json 민감도 분석에서만 쓰며 (날짜, 타임스탬프)를 양쪽에서 뺀다.
    """
    import numpy as np
    databases, queries, db_xy, query_xy = [], [], [], []
    for date in DATES:
        desc = np.load(encoded_dir / date / "descriptors.npy")
        meta = json.loads((encoded_dir / date / "rows.json").read_text())
        lookup = {int(key): i for i, key in enumerate(meta["keys"])}
        rows = [r for r in json.loads((DATA / "native_data" / date / filename).read_text())["rows"]
                if (date, r["timestamp"]) not in exclude]
        outside = [r for r in rows if not r["test_region"]]
        inside = [r for r in rows if r["test_region"]]
        databases.append(desc[[lookup[r["timestamp"]] for r in outside + inside]])
        queries.append(desc[[lookup[r["timestamp"]] for r in inside]])
        db_xy.append(np.array([r["xy"] for r in outside + inside]).reshape(-1, 2))
        query_xy.append(np.array([r["xy"] for r in inside]).reshape(-1, 2))
    return databases, queries, db_xy, query_xy


def score_variant(encoded_dir: Path, filename: str, ev, ledger_dir: Path | None,
                  exclude: frozenset = frozenset()) -> dict:
    """scripts/_review_remaining_native_disco_eval.py:152 score()와 같은 집계(서술자 폴더만 인자)."""
    import numpy as np
    import torch
    from sklearn.neighbors import KDTree
    databases, queries, db_xy, query_xy = load_sets(encoded_dir, filename, exclude)
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
            curve = np.maximum.accumulate(success[valid], axis=1).mean(0) * 100
            top_percent = max(int(round(len(db) / 100)), 1)
            opr = float(success[valid, :top_percent].any(1).mean() * 100)
            # 원본 get_recall 그대로 호출하되 별도 출력인 yaw 추정만 생략한다.
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
            assert abs(original[2] - opr) < 1e-10
            entry.update(status="complete", positive_queries=int(valid.sum()), correct_top1=int(success[valid, 0].sum()),
                         r1_percent=float(curve[0]), r1_percentile_percent=opr,
                         recall_at_1_to_25_percent=curve.tolist(), one_percent_k=top_percent,
                         effective_one_percent_k=min(25, top_percent), official_metric_agrees=True)
            rows.append(entry)
            if ledger_dir is not None:
                ledger_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(ledger_dir / f"{DATES[m]}__{DATES[n]}.npz", distances=distances, indices=indices,
                                    positive=valid, success=success, query_xy=query_xy[n], database_xy=db_xy[m])
    completed = [r for r in rows if r["status"] == "complete"]
    return {"pairs": rows, "valid_pairs": len(completed), "required_pairs": 30,
            "complete_30_pair_evaluation": len(completed) == 30,
            "mean_over_valid_pairs": {"r1_percent": float(np.mean([r["r1_percent"] for r in completed])),
                                      "r1_percentile_percent": float(np.mean([r["r1_percentile_percent"] for r in completed]))}
            if completed else None}


def scorer_equivalence(fresh: dict, stored: dict) -> dict:
    """새 집계기를 1m 저장 서술자에 적용한 결과가 기존 report.json과 같은지 쌍마다 비교한다."""
    import numpy as np
    by = {(p["database"], p["query"]): p for p in stored["pairs"]}
    worst, statuses, counts = 0.0, True, True
    for p in fresh["pairs"]:
        q = by[(p["database"], p["query"])]
        statuses &= p["status"] == q["status"]
        if p["status"] == "complete":
            counts &= p["positive_queries"] == q["positive_queries"] and p["correct_top1"] == q["correct_top1"]
            worst = max(worst, abs(p["r1_percent"] - q["r1_percent"]),
                        abs(p["r1_percentile_percent"] - q["r1_percentile_percent"]),
                        float(np.max(np.abs(np.array(p["recall_at_1_to_25_percent"]) - q["recall_at_1_to_25_percent"]))))
    mean = max(abs(fresh["mean_over_valid_pairs"][k] - stored["mean_over_valid_pairs"][k])
               for k in ["r1_percent", "r1_percentile_percent"])
    return {"pairs_compared": len(fresh["pairs"]), "statuses_equal": bool(statuses), "counts_equal": bool(counts),
            "pair_metric_max_abs_difference": worst, "mean_max_abs_difference": mean,
            "passed": bool(statuses and counts and worst < 1e-10 and mean < 1e-10)}


def run_score() -> None:
    """1m(기존)·5m(새) 서술자를 같은 집계기로 30쌍×2변형 평가한다."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.time()
    import torch
    sys.path.insert(0, str(REPO / "scripts"))
    from _disco_official_dump import official_model
    from _disco_official_prep import fake_gputransform_module
    torch.set_num_threads(2)
    missing = [d for d in DATES if not (ENCODED / d / "complete.json").exists()]
    if missing:
        raise RuntimeError(f"5m 추론 미완료 세션: {missing}")
    _, _, ev, _ = official_model(None, cpu_only=True, fake=fake_gputransform_module())
    assert str(ev.device) == "cpu"
    stored = json.loads((DATA / "report.json").read_text())
    out = {"ego1m_scorer_equivalence": {}, "ego5m": {}}
    for variant, filename, key in VARIANTS:
        fresh1 = score_variant(DATA / "encoded", filename, ev, None)
        out["ego1m_scorer_equivalence"][variant] = scorer_equivalence(fresh1, stored[key])
        out["ego5m"][variant] = score_variant(ENCODED, filename, ev, LEDGERS / variant)
    # protocol_amendment_1.json 민감도: 5m 점 1개 이하 스캔을 두 조건 모두 지도·질의에서 뺀다.
    exclude = frozenset((r["date"], r["timestamp"]) for r in degenerate_scans())
    out["sensitivity_exclude_degenerate"] = {"excluded": sorted(map(list, exclude)), "variants": {}}
    for variant, filename, _ in VARIANTS:
        member = {(d, r["timestamp"]) for d in DATES
                  for r in json.loads((DATA / "native_data" / d / filename).read_text())["rows"]}
        if not exclude & member:
            out["sensitivity_exclude_degenerate"]["variants"][variant] = {"affected": False}
            continue
        blocks = {label: score_variant(folder, filename, ev, None, exclude)
                  for label, folder in [("ego1m", DATA / "encoded"), ("ego5m", ENCODED)]}
        out["sensitivity_exclude_degenerate"]["variants"][variant] = {
            "affected": True, **{label: {k: b[k] for k in ["valid_pairs", "complete_30_pair_evaluation",
                                                           "mean_over_valid_pairs"]}
                                 | {"positive_queries_total": sum(p.get("positive_queries", 0) for p in b["pairs"])}
                                 for label, b in blocks.items()}}
    out["elapsed_s"] = time.time() - started
    out["scope"] = "공개 가중치의 역사판 전처리 원인 대조. 주표·원 논문 재현값 아님"
    write(VINTAGE / "score.json", out)


# ------------------------------------------------------------ 검산·보고(컨테이너, CPU)
def verify_ego5m(score: dict) -> dict:
    """5m 원장을 float64 독립 L2 전수 재계산과 비교하고, 양성 마스크를 1m 원장과 비교한다."""
    import numpy as np
    sys.path.insert(0, str(REPO / "scripts"))
    from _review_remaining_native_disco_eval import brute_force_pair
    out = {}
    for variant, filename, _ in VARIANTS:
        databases, queries, db_xy, query_xy = load_sets(ENCODED, filename)
        stored = {(p["database"], p["query"]): p for p in score["ego5m"][variant]["pairs"]}
        pairs, worst = [], {"r1": 0.0, "opr": 0.0, "curve": 0.0}
        for m, dm in enumerate(DATES):
            for n, dn in enumerate(DATES):
                if m == n:
                    continue
                p = stored[(dm, dn)]
                entry = {"database": dm, "query": dn, "stored_status": p["status"]}
                if len(databases[m]) < 25 or not len(queries[n]):
                    entry.update(status="unusable", status_agrees=p["status"] != "complete")
                    pairs.append(entry)
                    continue
                result = brute_force_pair(databases[m], queries[n], db_xy[m], query_xy[n])
                valid = result["positive"].any(1)
                if not valid.any() or p["status"] != "complete":
                    # 독립 계산과 저장 상태가 다르면 평균에서 빼고 불일치로 남긴다.
                    entry.update(status="unusable" if not valid.any() else "status_mismatch",
                                 status_agrees=(not valid.any()) and p["status"] != "complete")
                    pairs.append(entry)
                    continue
                success = np.take_along_axis(result["positive"], result["indices"], 1)
                curve = np.maximum.accumulate(success[valid], axis=1).mean(0) * 100
                k = max(int(round(len(databases[m]) / 100.0)), 1)
                opr = float(success[valid, :k].any(1).mean() * 100)
                ledger = np.load(LEDGERS / variant / f"{dm}__{dn}.npz")
                ledger1 = np.load(DATA / "ledgers" / variant / f"{dm}__{dn}.npz")
                top1_same = ledger["indices"][:, 0] == result["indices"][:, 0]
                nontie = 0
                for i in np.flatnonzero(~top1_same):
                    theirs = np.sqrt(((databases[m][ledger["indices"][i, 0]].astype(np.float64) - queries[n][i])**2).sum())
                    nontie += int(abs(theirs - result["distances"][i, 0]) > 1e-9 * max(1.0, theirs))
                entry.update(status="complete", status_agrees=True, positive_queries=int(valid.sum()),
                             correct_top1=int(success[valid, 0].sum()),
                             r1_abs_difference=abs(float(curve[0]) - p["r1_percent"]),
                             opr_abs_difference=abs(opr - p["r1_percentile_percent"]),
                             curve_max_abs_difference=float(np.max(np.abs(curve - np.array(p["recall_at_1_to_25_percent"])))),
                             positive_mask_equal_ledger=bool(np.array_equal(valid, ledger["positive"])),
                             positive_mask_equal_ego1m=bool(np.array_equal(valid, ledger1["positive"])),
                             query_xy_equal_ego1m=bool(np.array_equal(ledger["query_xy"], ledger1["query_xy"])),
                             database_xy_equal_ego1m=bool(np.array_equal(ledger["database_xy"], ledger1["database_xy"])),
                             top1_index_mismatches=int((~top1_same).sum()), top1_nontie_mismatches=nontie,
                             top25_order_mismatches=int((~(ledger["indices"] == result["indices"]).all(1)).sum()),
                             max_distance_abs_difference=float(np.max(np.abs(ledger["distances"] - result["distances"]))))
                for name, field in [("r1", "r1_abs_difference"), ("opr", "opr_abs_difference"),
                                    ("curve", "curve_max_abs_difference")]:
                    worst[name] = max(worst[name], entry[field])
                pairs.append(entry)
        done = [p for p in pairs if p["status"] == "complete"]
        mean_r1 = float(np.mean([p["correct_top1"] / p["positive_queries"] * 100 for p in done])) if done else None
        stored_mean = score["ego5m"][variant]["mean_over_valid_pairs"]
        out[variant] = {"pairs": pairs, "complete_pairs": len(done), "independent_mean_r1_percent": mean_r1,
                        "mean_r1_abs_difference": abs(mean_r1 - stored_mean["r1_percent"]) if done else None,
                        "worst_abs_difference": worst,
                        "all_statuses_agree": all(p["status_agrees"] for p in pairs),
                        "all_positive_masks_equal_ledger": all(p["positive_mask_equal_ledger"] for p in done),
                        "all_positive_masks_equal_ego1m": all(p["positive_mask_equal_ego1m"] for p in done),
                        "all_xy_equal_ego1m": all(p["query_xy_equal_ego1m"] and p["database_xy_equal_ego1m"] for p in done),
                        "top1_index_mismatches": sum(p["top1_index_mismatches"] for p in done),
                        "top1_nontie_mismatches": sum(p["top1_nontie_mismatches"] for p in done),
                        "top25_order_mismatches": sum(p["top25_order_mismatches"] for p in done)}
    return out


def contrast(score: dict) -> dict:
    """사전 지정 통계: 쌍별 Δ, 질의 엇갈림, 세션 하나 뺀 평균 Δ, 논문 차이 설명 비율."""
    import numpy as np
    stored = json.loads((DATA / "report.json").read_text())
    paper = {"r1_percent": 89.08, "r1_percentile_percent": 99.14}
    out = {}
    for variant, filename, key in VARIANTS:
        one = {(p["database"], p["query"]): p for p in stored[key]["pairs"]}
        five = {(p["database"], p["query"]): p for p in score["ego5m"][variant]["pairs"]}
        rows, flips_gain, flips_loss = [], 0, 0
        for pair, p5 in five.items():
            p1 = one[pair]
            if p5["status"] != "complete" or p1["status"] != "complete":
                rows.append({"database": pair[0], "query": pair[1], "status": "incomplete",
                             "ego1m_status": p1["status"], "ego5m_status": p5["status"]})
                continue
            l1 = np.load(DATA / "ledgers" / variant / f"{pair[0]}__{pair[1]}.npz")
            l5 = np.load(LEDGERS / variant / f"{pair[0]}__{pair[1]}.npz")
            assert np.array_equal(l1["positive"], l5["positive"])
            v = l1["positive"]
            s1, s5 = l1["success"][v, 0], l5["success"][v, 0]
            gain, loss = int((~s1 & s5).sum()), int((s1 & ~s5).sum())
            flips_gain += gain
            flips_loss += loss
            rows.append({"database": pair[0], "query": pair[1], "status": "complete",
                         "positive_queries": p5["positive_queries"],
                         "correct_top1_ego1m": p1["correct_top1"], "correct_top1_ego5m": p5["correct_top1"],
                         "r1_ego1m": p1["r1_percent"], "r1_ego5m": p5["r1_percent"],
                         "delta_r1_pp": p5["r1_percent"] - p1["r1_percent"],
                         "r1_percentile_ego1m": p1["r1_percentile_percent"], "r1_percentile_ego5m": p5["r1_percentile_percent"],
                         "delta_r1_percentile_pp": p5["r1_percentile_percent"] - p1["r1_percentile_percent"],
                         "queries_wrong_to_right": gain, "queries_right_to_wrong": loss})
        done = [r for r in rows if r["status"] == "complete"]
        m1 = stored[key]["mean_over_valid_pairs"]
        m5 = score["ego5m"][variant]["mean_over_valid_pairs"]
        delta = {k: m5[k] - m1[k] for k in ["r1_percent", "r1_percentile_percent"]}
        gap = {k: paper[k] - m1[k] for k in delta}
        loso = {}
        for s in DATES:
            keep = [r for r in done if s not in (r["database"], r["query"])]
            loso[s] = {"pairs": len(keep), "delta_r1_pp": float(np.mean([r["delta_r1_pp"] for r in keep])),
                       "delta_r1_percentile_pp": float(np.mean([r["delta_r1_percentile_pp"] for r in keep]))}
        d = np.array([r["delta_r1_pp"] for r in done])
        dp = np.array([r["delta_r1_percentile_pp"] for r in done])
        # protocol_amendment_1.json: 5m 점 1개 이하 스캔의 질의 결과와 지도 항목이 1위로 뽑힌 수.
        degenerate = []
        for scan in degenerate_scans():
            dd, ts = scan["date"], scan["timestamp"]
            sel = json.loads((DATA / "native_data" / dd / filename).read_text())["rows"]
            inside = [r for r in sel if r["test_region"]]
            ordered = [r for r in sel if not r["test_region"]] + inside
            q_idx = next((i for i, r in enumerate(inside) if r["timestamp"] == ts), None)
            db_idx = next((i for i, r in enumerate(ordered) if r["timestamp"] == ts), None)
            if q_idx is None and db_idx is None:
                continue
            info = {"date": dd, "timestamp": ts, "query_index": q_idx, "database_index": db_idx,
                    "as_query": [], "as_database_top1": []}
            for other in DATES:
                if other == dd:
                    continue
                if q_idx is not None:
                    l1 = np.load(DATA / "ledgers" / variant / f"{other}__{dd}.npz")
                    l5 = np.load(LEDGERS / variant / f"{other}__{dd}.npz")
                    info["as_query"].append({"database": other, "positive": bool(l1["positive"][q_idx]),
                                             "top1_correct_ego1m": bool(l1["success"][q_idx, 0]),
                                             "top1_correct_ego5m": bool(l5["success"][q_idx, 0])})
                if db_idx is not None:
                    l1 = np.load(DATA / "ledgers" / variant / f"{dd}__{other}.npz")
                    l5 = np.load(LEDGERS / variant / f"{dd}__{other}.npz")
                    v = l1["positive"]
                    info["as_database_top1"].append({"query": other,
                                                     "positive_queries_top1_ego1m": int((l1["indices"][v, 0] == db_idx).sum()),
                                                     "positive_queries_top1_ego5m": int((l5["indices"][v, 0] == db_idx).sum())})
            degenerate.append(info)
        sens = score.get("sensitivity_exclude_degenerate", {}).get("variants", {}).get(variant, {"affected": False})
        if sens.get("affected"):
            s1, s5 = sens["ego1m"]["mean_over_valid_pairs"], sens["ego5m"]["mean_over_valid_pairs"]
            sens = {**sens, "delta_ego5m_minus_ego1m_pp": {k: s5[k] - s1[k] for k in s1},
                    "explained_fraction_of_paper_gap": {k: (s5[k] - s1[k]) / (paper[k] - s1[k]) for k in s1}}
        out[variant] = {
            "degenerate_scans": degenerate, "sensitivity_exclude_degenerate": sens,
            "complete_pairs_ego1m": stored[key]["valid_pairs"], "complete_pairs_ego5m": score["ego5m"][variant]["valid_pairs"],
            "ego1m": m1, "ego5m": m5, "delta_ego5m_minus_ego1m_pp": delta,
            "paper_minus_ego1m_pp": gap, "paper_minus_ego5m_pp": {k: paper[k] - m5[k] for k in delta},
            "explained_fraction_of_paper_gap": {k: delta[k] / gap[k] for k in delta},
            "pairs_r1_up_down_same": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())],
            "pairs_r1_percentile_up_down_same": [int((dp > 0).sum()), int((dp < 0).sum()), int((dp == 0).sum())],
            "pair_delta_r1_pp_min_median_max": [float(d.min()), float(np.median(d)), float(d.max())],
            "queries_wrong_to_right": flips_gain, "queries_right_to_wrong": flips_loss,
            "positive_queries_total": int(sum(r["positive_queries"] for r in done)),
            "leave_one_session_out_delta": loso,
            "leave_one_session_out_delta_r1_range_pp": [min(v["delta_r1_pp"] for v in loso.values()),
                                                        max(v["delta_r1_pp"] for v in loso.values())],
            "pairs": rows}
    return out


def classify(fraction: float, ours: dict) -> str:
    """protocol.json predeclared_interpretation 규칙을 그대로 적용한다."""
    if abs(89.08 - ours["r1_percent"]) <= 0.005 and abs(99.14 - ours["r1_percentile_percent"]) <= 0.005:
        return "numeric_reproduction"
    if fraction < 0:
        return "explained_fraction_lt_0"
    if fraction < 1:
        return "explained_fraction_0_to_1"
    return "explained_fraction_ge_1"


def run_verify() -> None:
    """검산·대조 통계·보고서를 만든다."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.time()
    import numpy as np
    score = json.loads((VINTAGE / "score.json").read_text())
    protocol = json.loads((VINTAGE / "protocol.json").read_text())
    retrieval = verify_ego5m(score)
    inputs = {"checkpoint_sha256": sha256(DATA.parent / "weights/disco_author_model.ckpt"),
              "historical_generator_sha256": sha256(HISTORICAL), "dates": {}}
    for date in DATES:
        c5 = json.loads((ENCODED / date / "complete.json").read_text())
        k1 = json.loads((DATA / "encoded" / date / "rows.json").read_text())["keys"]
        k5 = json.loads((ENCODED / date / "rows.json").read_text())["keys"]
        d5 = np.load(ENCODED / date / "descriptors.npy")
        inputs["dates"][date] = {
            "keys_equal_ego1m": k1 == k5, "encoded": len(k5),
            "selection_sha256_equal_protocol": all(
                sha256(DATA / "native_data" / date / f) == protocol["held_fixed"]["selection_sha256"][v][date]
                for v, f, _ in VARIANTS),
            "ego1m_descriptors_sha256_equal_protocol": sha256(DATA / "encoded" / date / "descriptors.npy")
            == protocol["reused_ego1m"]["descriptors_sha256"][date],
            "ego5m_descriptors_sha256": sha256(ENCODED / date / "descriptors.npy"),
            "ego5m_shape": list(d5.shape), "ego5m_finite": bool(np.isfinite(d5).all()),
            "historical_spot_checks": len(c5["checks"]), "all_inputs_cpu_cuda_checked": c5["all_inputs_cpu_cuda_checked"],
            "official_inference_max_abs_difference": c5["official_inference_max_abs_difference"],
            "ego1m_reencode_max_abs_difference": c5["ego1m_reencode_max_abs_difference"],
            "max_changed_ring_index": c5["input_change"]["max_changed_ring_index"]}
    equiv = score["ego1m_scorer_equivalence"]
    checks = {
        "checkpoint": inputs["checkpoint_sha256"] == "8665e356f0aab2cf03fdc47aa2b3dd2d32be7a2b420bad35ac2c994f45d7808c",
        "historical_generator": inputs["historical_generator_sha256"] == HISTORICAL_SHA,
        "inputs_fixed": all(r["keys_equal_ego1m"] and r["selection_sha256_equal_protocol"]
                            and r["ego1m_descriptors_sha256_equal_protocol"] and r["ego5m_finite"]
                            and r["historical_spot_checks"] == 3 and r["all_inputs_cpu_cuda_checked"]
                            for r in inputs["dates"].values()),
        "official_inference": all(r["official_inference_max_abs_difference"] < 2e-5
                                  and r["ego1m_reencode_max_abs_difference"] < 2e-5 for r in inputs["dates"].values()),
        "changed_rings_within_box": all(r["max_changed_ring_index"] <= 4 for r in inputs["dates"].values()),
        "ego1m_scorer_equivalence": all(v["passed"] for v in equiv.values()),
        "ego5m_retrieval": all(b["all_statuses_agree"] and b["all_positive_masks_equal_ledger"]
                               and b["all_positive_masks_equal_ego1m"] and b["all_xy_equal_ego1m"]
                               and b["top1_nontie_mismatches"] == 0 and b["mean_r1_abs_difference"] < 1e-9
                               and max(b["worst_abs_difference"].values()) < 1e-9 for b in retrieval.values()),
    }
    verification = {"inputs": inputs, "ego1m_scorer_equivalence": equiv, "ego5m_retrieval": retrieval,
                    "checks": checks, "passed": all(checks.values()), "elapsed_s": time.time() - started}
    write(VINTAGE / "verification.json", verification)
    stats = contrast(score)
    write(VINTAGE / "contrast.json", stats)
    status("verified", passed=verification["passed"])


def local_check() -> dict:
    """작은 JSON(score·contrast·report.json 1m)만으로 쌍별 R@1과 30쌍 평균을 다시 계산한다(numpy 없음)."""
    score = json.loads((VINTAGE / "score.json").read_text())
    stats = json.loads((VINTAGE / "contrast.json").read_text())
    stored1 = json.loads((DATA / "report.json").read_text())
    out = {}
    for variant, _, key in VARIANTS:
        row = {}
        for label, block in [("ego1m", stored1[key]), ("ego5m", score["ego5m"][variant])]:
            done = [p for p in block["pairs"] if p["status"] == "complete"]
            worst = max(abs(p["correct_top1"] / p["positive_queries"] * 100 - p["r1_percent"]) for p in done)
            mean = sum(p["correct_top1"] / p["positive_queries"] * 100 for p in done) / len(done)
            row[label] = {"complete_pairs": len(done), "positive_queries": sum(p["positive_queries"] for p in done),
                          "correct_top1": sum(p["correct_top1"] for p in done),
                          "r1_from_counts_max_abs_difference": worst, "mean_r1_from_counts": mean,
                          "mean_abs_difference": abs(mean - block["mean_over_valid_pairs"]["r1_percent"]),
                          "contrast_mean_abs_difference": abs(mean - stats[variant][label]["r1_percent"])}
        delta = row["ego5m"]["mean_r1_from_counts"] - row["ego1m"]["mean_r1_from_counts"]
        row["delta_r1_pp_from_counts"] = delta
        row["delta_abs_difference"] = abs(delta - stats[variant]["delta_ego5m_minus_ego1m_pp"]["r1_percent"])
        row["passed"] = (row["ego1m"]["complete_pairs"] == row["ego5m"]["complete_pairs"] == 30
                         and row["ego1m"]["positive_queries"] == row["ego5m"]["positive_queries"]
                         and max(row[x][k] for x in ["ego1m", "ego5m"] for k in
                                 ["r1_from_counts_max_abs_difference", "mean_abs_difference",
                                  "contrast_mean_abs_difference"]) < 1e-9 and row["delta_abs_difference"] < 1e-9)
        out[variant] = row
    result = {"where": "로컬 호스트, 서버에서 받은 작은 JSON만", "variants": out,
              "passed": all(v["passed"] for v in out.values()), "unix": time.time()}
    write(VINTAGE / "local_check.json", result)
    return result


def interpretation(metrics: dict, change: dict, scale: dict | None) -> list[str]:
    """사전 규칙과 고정 문장 틀로 원인 설명력을 적는다(값에 따라 설정을 고르지 않는다)."""
    p, s = metrics["released"], metrics["no_upper_bound"]
    dp, fp = p["delta_ego5m_minus_ego1m_pp"], p["explained_fraction_of_paper_gap"]
    lo, hi = p["leave_one_session_out_delta_r1_range_pp"]
    lines = [f"H1(가중치 공개 시점 5m 판과 현재 1m 판의 불일치) 검정: 주 대조에서 5m 판은 R@1 {dp['r1_percent']:+.2f}%p, "
             f"R@1% {dp['r1_percentile_percent']:+.2f}%p로 논문과의 차이를 "
             + ("줄인다" if dp["r1_percent"] > 0 else "줄이지 않고 키운다")
             + f"(논문 차이 설명 비율 R@1 {fp['r1_percent']*100:+.1f}%, R@1% {fp['r1_percentile_percent']*100:+.1f}%)"]
    sens = s.get("sensitivity_exclude_degenerate", {})
    same = (s["delta_ego5m_minus_ego1m_pp"]["r1_percent"] > 0) == (dp["r1_percent"] > 0)
    lines.append(f"보조 대조(상한 제거)는 R@1 {s['delta_ego5m_minus_ego1m_pp']['r1_percent']:+.2f}%p, "
                 f"R@1% {s['delta_ego5m_minus_ego1m_pp']['r1_percentile_percent']:+.2f}%p"
                 + (f", 0점 스캔을 두 조건 모두 뺀 민감도는 R@1 {sens['delta_ego5m_minus_ego1m_pp']['r1_percent']:+.2f}%p"
                    if sens.get("affected") else "")
                 + ("로 주 대조와 방향이 같다" if same else "로 주 대조와 방향이 다르다"))
    sign = "모두 음수" if hi < 0 else "모두 양수" if lo > 0 else "부호가 섞여"
    lines.append(f"세션 하나를 뺀 6가지 20쌍 평균 ΔR@1이 {sign}다({lo:+.2f}~{hi:+.2f}%p). 쌍별로는 증가 {p['pairs_r1_up_down_same'][0]}·"
                 f"감소 {p['pairs_r1_up_down_same'][1]}·동일 {p['pairs_r1_up_down_same'][2]}쌍, 질의 오답→정답 "
                 f"{p['queries_wrong_to_right']}·정답→오답 {p['queries_right_to_wrong']}개다")
    if change and scale:
        removed = sum(c["points_removed_mean"] for c in change.values()) / len(change)
        cells = sum(c["polar_cells_changed_fraction_of_ego1m_occupied_mean"] for c in change.values()) / len(change)
        shift = scale["descriptor_l2_shift_all_scans"]
        ref = scale["released_ego1m"]
        lines.append(f"입력 변화는 작지 않다: 5m 판은 스캔당 평균 약 {removed:.0f}점을 더 지워 링0~4에서 1m 점유 칸의 평균 "
                     f"{cells*100:.1f}%를 바꾸고, 서술자 L2 이동 중앙값 {shift['median']:.3f}(변화 없는 스캔 {shift['unchanged_scans']}/"
                     f"{shift['scans']})은 1m 주 대조 1위 L2 거리 중앙값 {ref['top1_l2_median']:.3f}, 1·2위 간격 중앙값 "
                     f"{ref['top1_top2_gap_median']:.3f}과 견줄 크기다. 변화가 작아 지표가 그대로였던 것이 아니라, 바뀐 입력에서 공개 가중치가 덜 맞는다")
    lines += ["판정(사전 규칙): R@1·R@1% 모두 '5m이 1m보다 낮다. 현재 1m 판 사용은 우리 수치를 낮춘 원인이 아니다'. "
              "causal_audit H1은 논문 대비 R@1 7.03%p 차이의 설명에서 배제하며, 이 요인으로 설명한 부분은 0이다",
              "공개 가중치가 1m 입력에서 더 나은 것은 가중치가 1m 판(또는 비슷한 근거리 처리)으로 학습됐을 가능성과 어긋나지 않지만, "
              "학습 판은 저자 기록이 없어 확정하지 않는다",
              "남은 후보는 H2 공개 가중치와 논문 가중치의 동일성 미확인, H3 저자 표본 목록 미공개, H4 논문 시기 전처리(필터 없음·스캔별 정규화·80m 서술), "
              "H5 원 환경(PyTorch 1.4)이다. 이번 실행은 이들을 검정하지 않았다"]
    return lines


def build_report() -> None:
    """작은 JSON만으로 report.json·report_ko.txt를 만든다(서버·로컬 어디서나)."""
    protocol = json.loads((VINTAGE / "protocol.json").read_text())
    verification = json.loads((VINTAGE / "verification.json").read_text())
    stats = json.loads((VINTAGE / "contrast.json").read_text())
    local = read(VINTAGE / "local_check.json")
    changed = read(VINTAGE / "changed_check.json")
    timing = read(VINTAGE / "timing.json") or {}
    encode_timing = read(VINTAGE / "encode_timing.json") or {}
    score = json.loads((VINTAGE / "score.json").read_text())
    change = {d: json.loads((ENCODED / d / "complete.json").read_text())["input_change"]
              for d in DATES if (ENCODED / d / "complete.json").exists()}
    if len(change) == len(DATES):
        write(VINTAGE / "input_change.json", change)
    else:
        change = read(VINTAGE / "input_change.json") or change
    metrics, classes = {}, {}
    for variant, _, _ in VARIANTS:
        s = stats[variant]
        metrics[variant] = {k: s[k] for k in s if k not in ("pairs", "leave_one_session_out_delta")}
        classes[variant] = {k: classify(s["explained_fraction_of_paper_gap"][k], s["ego5m"])
                            for k in ["r1_percent", "r1_percentile_percent"]}
    report = {"task": protocol["task"], "classification": protocol["classification"],
              "protocol_sha256": sha256(VINTAGE / "protocol.json"),
              "protocol_amendment_1_sha256": sha256(VINTAGE / "protocol_amendment_1.json"),
              "amendment_1_summary": json.loads((VINTAGE / "protocol_amendment_1.json").read_text())["rule"],
              "scan_counts": read(VINTAGE / "scan_counts.json"),
              "verification_sha256": sha256(VINTAGE / "verification.json"),
              "verification_passed": verification["passed"], "verification_checks": verification["checks"],
              "local_check": local,
              "changed_check": {k: v for k, v in changed.items() if k != "sessions"} | {
                  "max_abs_difference": max(s["official_vs_stored_ego5m_max_abs_difference"] for s in changed["sessions"].values()),
                  "historical_polar_exact": sum(sum(s["historical_polar_exact"]) for s in changed["sessions"].values()),
                  "historical_polar_checked": sum(len(s["historical_polar_exact"]) for s in changed["sessions"].values()),
                  "images": sum(len(s["indices"]) for s in changed["sessions"].values())} if changed else None,
              "all_checks_passed": bool(verification["passed"] and (changed or {}).get("passed")
                                        and (local or {}).get("passed")),
              "paper": {"r1_percent": 89.08, "r1_percentile_percent": 99.14},
              "metrics": metrics, "predeclared_class": classes,
              "class_meaning": protocol["predeclared_interpretation"],
              "input_change": change, "timing": {"host": timing, "container_encode": encode_timing,
                                                 "score_elapsed_s": score.get("elapsed_s"),
                                                 "verify_elapsed_s": verification.get("elapsed_s")},
              "code_sha256": {name: sha256(REPO / "scripts" / name) for name in
                              ["_review_remaining_native_disco_vintage.py", "_review_remaining_native_disco_eval.py",
                               "_disco_official_prep.py", "_disco_official_dump.py"]},
              "commands": {"run": "cd /home/qkrrlans00/Neural-Spectral-Codec && setsid nohup python3 -u scripts/_review_remaining_native_disco_vintage.py run > .../preprocess_vintage_20261107/run.log 2>&1 &",
                           "encode": "flock -x /tmp/nsd_native_reproduction_gpu0.lock docker exec -e CUDA_VISIBLE_DEVICES=0 -e NVIDIA_TF32_OVERRIDE=0 -e OMP_NUM_THREADS=2 ring_baseline python3 -u scripts/_review_remaining_native_disco_vintage.py encode",
                           "score": "docker exec -e CUDA_VISIBLE_DEVICES= ring_baseline python3 -u scripts/_review_remaining_native_disco_vintage.py score",
                           "verify": "docker exec -e CUDA_VISIBLE_DEVICES= ring_baseline python3 -u scripts/_review_remaining_native_disco_vintage.py verify",
                           "report": "python3 scripts/_review_remaining_native_disco_vintage.py report"},
              "guard": protocol["predeclared_interpretation"]["guard"],
              "descriptor_scale": read(VINTAGE / "descriptor_scale.json"),
              "interpretation_ko": interpretation(metrics, change, read(VINTAGE / "descriptor_scale.json")),
              "separate_from_main_table": True}
    write(VINTAGE / "report.json", report)
    (VINTAGE / "report_ko.txt").write_text(report_text(report, stats))


def report_text(report: dict, stats: dict) -> str:
    """root가 읽을 한국어 요약."""
    names = {"released": "주 대조: 공개 생성기 3≤d<3.5m", "no_upper_bound": "보조 대조: 상한 제거 3m"}
    meaning = {"numeric_reproduction": "논문 표기 정밀도 안에서 같다",
               "explained_fraction_lt_0": "5m이 1m보다 낮다. 현재 1m 판 사용은 우리 수치를 낮춘 원인이 아니다",
               "explained_fraction_0_to_1": "논문 차이의 해당 비율만 설명한다. 나머지는 미설명",
               "explained_fraction_ge_1": "논문 차이 이상 변한다. 재현값으로 대체하지 않고 원인 후보로만 기록"}
    lines = ["DiSCO 전처리 판 원인 대조: 공개 가중치 업로드 당시 자차 5m 제외(8e79286) 대 현재 1m 제외(0e598a8~HEAD)",
             "같은 가중치(SHA-256 8665e356…)·같은 6세션 30쌍·같은 표본·같은 양성 목록. 바꾼 것은 상자 반폭 한 줄뿐이다.",
             "공통 분할 주표와 무관한 원인 대조 진단이며 주표·주 진단(1m)을 바꾸지 않는다.", "",
             f"독립 검산 통과: {report['verification_passed']} "
             + ", ".join(f"{k}={v}" for k, v in report["verification_checks"].items()),
             "로컬 산술 검산(local_check.json): " + (f"통과 {report['local_check']['passed']}" if report.get("local_check")
                                                  else "아직 실행 전(서버 보고서)"),
             "검산 보강(changed_check.json, 세션마다 서술자 변화 최대 9장): " + (
                 f"통과 {report['changed_check']['passed']}, 공식 get_latent_vectors 대 저장 5m 최대 차 "
                 f"{report['changed_check']['max_abs_difference']:.1e}, 8e79286 원문 극좌표 일치 "
                 f"{report['changed_check']['historical_polar_exact']}/{report['changed_check']['historical_polar_checked']}장"
                 f"(대조 영상 {report['changed_check']['images']}장, 0점 스캔은 원문이 IndexError라 극좌표 대조에서 제외)"
                 if report.get("changed_check") else "실행 전"),
             f"전체 검산 통과: {report['all_checks_passed']}", "", "원인 설명력:"] + \
        [f"  - {x}" for x in report.get("interpretation_ko", [])] + [""]
    for variant in ["released", "no_upper_bound"]:
        m = report["metrics"][variant]
        lines += [f"[{names[variant]}] 완료 쌍 1m {m['complete_pairs_ego1m']}/30, 5m {m['complete_pairs_ego5m']}/30, 양성 질의 합 {m['positive_queries_total']}",
                  f"  R@1: 1m {m['ego1m']['r1_percent']:.4f}% → 5m {m['ego5m']['r1_percent']:.4f}% "
                  f"(Δ {m['delta_ego5m_minus_ego1m_pp']['r1_percent']:+.4f}%p, 논문 89.08−5m {m['paper_minus_ego5m_pp']['r1_percent']:+.4f}%p)",
                  f"  R@1%: 1m {m['ego1m']['r1_percentile_percent']:.4f}% → 5m {m['ego5m']['r1_percentile_percent']:.4f}% "
                  f"(Δ {m['delta_ego5m_minus_ego1m_pp']['r1_percentile_percent']:+.4f}%p, 논문 99.14−5m {m['paper_minus_ego5m_pp']['r1_percentile_percent']:+.4f}%p)",
                  f"  논문 차이 설명 비율 R@1 {m['explained_fraction_of_paper_gap']['r1_percent']:+.3f}, "
                  f"R@1% {m['explained_fraction_of_paper_gap']['r1_percentile_percent']:+.3f} → 사전 규칙: "
                  f"R@1 {meaning[report['predeclared_class'][variant]['r1_percent']]}; "
                  f"R@1% {meaning[report['predeclared_class'][variant]['r1_percentile_percent']]}",
                  f"  쌍별 R@1 증가/감소/동일 {m['pairs_r1_up_down_same']}, Δ 최소/중앙/최대 "
                  + "/".join(f"{x:+.3f}" for x in m["pair_delta_r1_pp_min_median_max"])
                  + f"%p, 질의 오답→정답 {m['queries_wrong_to_right']}·정답→오답 {m['queries_right_to_wrong']}",
                  f"  세션 하나 뺀 20쌍 평균 ΔR@1 범위 {m['leave_one_session_out_delta_r1_range_pp'][0]:+.4f}~"
                  f"{m['leave_one_session_out_delta_r1_range_pp'][1]:+.4f}%p (쌍이 독립이 아니어서 유의확률은 주장하지 않음)"]
        for scan in m.get("degenerate_scans", []):
            q = [x for x in scan["as_query"] if x["positive"]]
            lines.append(f"  5m 0점 스캔 {scan['date']} {scan['timestamp']}(전부 0 영상으로 추론): 양성 질의로 {len(q)}쌍, "
                         f"1위 정답 1m {sum(x['top1_correct_ego1m'] for x in q)}·5m {sum(x['top1_correct_ego5m'] for x in q)}; "
                         f"다른 양성 질의가 이 지도 항목을 1위로 고른 수 1m {sum(x['positive_queries_top1_ego1m'] for x in scan['as_database_top1'])}"
                         f"·5m {sum(x['positive_queries_top1_ego5m'] for x in scan['as_database_top1'])}")
        sens = m.get("sensitivity_exclude_degenerate", {})
        if sens.get("affected"):
            lines.append(f"  민감도(그 스캔을 두 조건 모두 제외): R@1 1m {sens['ego1m']['mean_over_valid_pairs']['r1_percent']:.4f}% → "
                         f"5m {sens['ego5m']['mean_over_valid_pairs']['r1_percent']:.4f}% (Δ {sens['delta_ego5m_minus_ego1m_pp']['r1_percent']:+.4f}%p), "
                         f"R@1% Δ {sens['delta_ego5m_minus_ego1m_pp']['r1_percentile_percent']:+.4f}%p, 완료 쌍 "
                         f"{sens['ego1m']['valid_pairs']}/{sens['ego5m']['valid_pairs']}")
        lines.append("")
    change = report["input_change"]
    if change:
        lines.append("입력 변화(세션별): 스캔당 제거 점 평균, 바뀐 극좌표 칸 평균(1m 점유 대비), 최대 바뀐 링, 서술자 L2 이동 중앙값")
        for d, c in change.items():
            lines.append(f"  {d}: {c['points_removed_mean']:.0f}점, {c['polar_cells_changed_mean']:.1f}칸"
                         f"({c['polar_cells_changed_fraction_of_ego1m_occupied_mean']*100:.2f}%), 링≤{c['max_changed_ring_index']}, "
                         f"L2 {c['descriptor_l2_shift_median']:.4f}")
    t = report["timing"]
    host, enc = t.get("host") or {}, t.get("container_encode") or {}
    if host:
        lines += ["", f"시간: GPU0 락 대기 합 {host.get('lock_wait_s_total', float('nan')):.1f}s, 락 보유 합(컨테이너 시작~종료) "
                      f"{host.get('lock_hold_s_total', float('nan')):.1f}s, 시도 "
                  + "; ".join(f"{a['attempt']}회 보유 {a['lock_hold_s']:.1f}s·완료 {len(a['completed_sessions'])}세션"
                              for a in host.get("encode_attempts", []) if a.get("lock_hold_s") is not None),
                  f"  컨테이너 안: 모델 적재 {enc.get('model_load_s', float('nan')):.1f}s, 전처리(5m 필터·CUDA 극좌표·CPU 대조) {enc.get('sum_preprocess_s', float('nan')):.1f}s, "
                  f"GPU 순전파(동기화) {enc.get('sum_gpu_forward_s', float('nan')):.1f}s, 1m 비교 진단 {enc.get('sum_diagnostic_s', float('nan')):.1f}s, 공식·역사판 검산 {enc.get('sum_checks_s', float('nan')):.1f}s, 스캔 {enc.get('scans')}",
                  f"  CPU 집계 {t.get('score_elapsed_s') or float('nan'):.1f}s, CPU 검산 {t.get('verify_elapsed_s') or float('nan'):.1f}s. root 추정 GPU 2~3분과 비교: 실제 락 보유 합 {host.get('lock_hold_s_total', float('nan'))/60:.2f}분"
                  " (세션별 전처리·순전파 합은 6세션 complete.json 기준, 모델 적재는 마지막 시도)"]
        cc = host.get("changed_check_lock") or {}
        if cc.get("lock_hold_s") is not None:
            total = host.get("lock_hold_s_total", 0) + cc["lock_hold_s"]
            lines.append(f"  검산 보강 락 보유 {cc['lock_hold_s']:.1f}s(대기 {cc['lock_wait_s']:.1f}s). GPU0 락 보유 총 {total:.1f}s = {total/60:.2f}분")
    lines += ["", "해석 경계:", f"  - {report['guard']}",
              "  - 가중치가 실제로 어느 전처리 판으로 학습됐는지는 저자 기록이 없어 이 대조로 직접 확정하지 않는다",
              "  - 논문 80m 서술·평가 시 무작위 회전·저자 표본 목록은 바꾸지 않았다"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------ 감독(서버 호스트, 표준 라이브러리)
def free_gib() -> float:
    """서버 루트 파일 시스템 여유."""
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 2**30


def stage(name: str, command: list[str], log: Path) -> None:
    """컨테이너 단계를 실행하고 실패하면 기록 뒤 중단한다."""
    with open(log, "ab") as handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {name}: {' '.join(command)}\n".encode())
        handle.flush()
        rc = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
    if rc:
        status(f"{name}_failed", returncode=rc, log=str(log.relative_to(REPO)))
        raise SystemExit(f"{name} 실패 rc={rc}")


def run() -> None:
    """encode(GPU0 락)→score→verify→report 순서로 실행한다. 완료 단계는 건너뛴다."""
    VINTAGE.mkdir(parents=True, exist_ok=True)
    assert (VINTAGE / "protocol.json").exists(), "사전 고정 protocol.json 없음"
    status("started", host_pid=os.getpid(), free_gib=round(free_gib(), 1))
    if free_gib() < MIN_FREE_GIB + 1:
        status("blocked_disk", free_gib=round(free_gib(), 1))
        raise SystemExit("서버 여유 20GiB 보존 조건 미충족")
    log = VINTAGE / "stages.log"
    timing = read(VINTAGE / "timing.json") or {}
    if not (VINTAGE / "scan_counts.json").exists():
        status("scan_counts")
        stage("scan_counts", EXEC + CPU_ENV + SELF + ["scan_counts"], log)
    if not all((ENCODED / d / "complete.json").exists() for d in DATES):
        attempts = timing.setdefault("encode_attempts", [])
        marker = VINTAGE / f"lock_acquired_attempt{len(attempts) + 1}.txt"
        command = ["flock", "-x", str(LOCK), "bash", "-c", 'date +%s.%N > "$0"; exec "$@"', str(marker)] + \
            EXEC + GPU_ENV + SELF + ["encode"]
        queued = time.time()
        status("encode_queued", host_pid=os.getpid(), lock=str(LOCK), attempt=len(attempts) + 1)
        try:
            stage("encode", command, log)
        finally:
            finished = time.time()
            acquired = float(marker.read_text().strip()) if marker.exists() else None
            attempts.append({"attempt": len(attempts) + 1, "queued_unix": queued, "lock_acquired_unix": acquired,
                             "finished_unix": finished, "lock_wait_s": acquired - queued if acquired else None,
                             "lock_hold_s": finished - acquired if acquired else None,
                             "completed_sessions": [d for d in DATES if (ENCODED / d / "complete.json").exists()]})
            timing["lock_hold_s_total"] = sum(a["lock_hold_s"] or 0 for a in attempts)
            timing["lock_wait_s_total"] = sum(a["lock_wait_s"] or 0 for a in attempts)
            write(VINTAGE / "timing.json", timing)
        status("encoded", lock_wait_s=round(timing["lock_wait_s_total"], 1),
               lock_hold_s=round(timing["lock_hold_s_total"], 1))
    if not (VINTAGE / "score.json").exists():
        status("scoring")
        t = time.time()
        stage("score", EXEC + CPU_ENV + SELF + ["score"], log)
        timing["score_wall_s"] = time.time() - t
        write(VINTAGE / "timing.json", timing)
    if not (VINTAGE / "verification.json").exists():
        status("verifying")
        t = time.time()
        stage("verify", EXEC + CPU_ENV + SELF + ["verify"], log)
        timing["verify_wall_s"] = time.time() - t
        write(VINTAGE / "timing.json", timing)
    if not (VINTAGE / "changed_check.json").exists():
        marker = VINTAGE / "lock_acquired_changed_check.txt"
        queued = time.time()
        status("changed_check_queued", lock=str(LOCK))
        try:
            stage("changed_check", ["flock", "-x", str(LOCK), "bash", "-c", 'date +%s.%N > "$0"; exec "$@"', str(marker)]
                  + EXEC + GPU_ENV + SELF + ["changed_check"], log)
        finally:
            finished = time.time()
            acquired = float(marker.read_text().strip()) if marker.exists() else None
            timing["changed_check_lock"] = {"queued_unix": queued, "lock_acquired_unix": acquired, "finished_unix": finished,
                                            "lock_wait_s": acquired - queued if acquired else None,
                                            "lock_hold_s": finished - acquired if acquired else None}
            write(VINTAGE / "timing.json", timing)
    if not (VINTAGE / "descriptor_scale.json").exists():
        stage("scale", EXEC + CPU_ENV + SELF + ["scale"], log)
    stage("report", EXEC + CPU_ENV + SELF + ["report"], log)
    verification = read(VINTAGE / "verification.json") or {}
    status("complete", passed=verification.get("passed"), host_pid=None)


def main() -> None:
    """단계 하나를 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "encode", "score", "verify", "report", "local", "scan_counts",
                                          "changed_check", "scale"])
    args = parser.parse_args()
    if args.stage == "run":
        run()
    elif args.stage == "scale":
        run_scale()
    elif args.stage == "scan_counts":
        run_scan_counts()
    elif args.stage == "changed_check":
        run_changed_check()
    elif args.stage == "encode":
        run_encode()
    elif args.stage == "score":
        run_score()
    elif args.stage == "verify":
        run_verify()
    elif args.stage == "report":
        build_report()
    else:
        print(json.dumps({"local_check_passed": local_check()["passed"]}, ensure_ascii=False), flush=True)
        build_report()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (0, None):
            write(VINTAGE / f"error_{sys.argv[1] if len(sys.argv) > 1 else 'main'}.json",
                  {"type": type(exc).__name__, "message": str(exc), "argv": sys.argv,
                   "traceback": traceback.format_exc(), "unix": time.time()})
        raise
