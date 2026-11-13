#!/usr/bin/env python3
"""주표46시퀀스의 RING++ 격자 max만 수정해 별도 원장을 만든다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_ringpp_maxfix_20261105"
CHUNK = 128
sys.path.insert(0, str(REPO / "scripts"))
import _ring_official_lib as rol
from _pdv2_pilot import CITY, TEST


def sha256(path: Path) -> str:
    """파일 SHA256을 계산한다."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: np.ndarray) -> str:
    """연속 배열의 바이트 해시를 계산한다."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """진행 기록을 원자적으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def write_npz(path: Path, **arrays) -> None:
    """완성된 묶음만 노출한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def parts(sequence: str) -> tuple:
    """고정 시험 목록의 센서와 시퀀스를 반환한다."""
    sensor, seq = sequence.split("/")
    assert (sensor, seq) in TEST, sequence
    return sensor, seq


def seq_dir(stage: str, sequence: str) -> Path:
    """단계별 전용 저장 폴더를 반환한다."""
    return OUT / stage / sequence.replace("/", "_")


def keyframes(sequence: str) -> tuple:
    """기존 키프레임과 평가 자세를 그대로 연다."""
    sensor, seq = parts(sequence)
    path = rol.split_cache("odom", "test", sensor, seq)
    ids, poses = rol.load_keyframes(path)
    if (OUT / "protocol.json").exists():
        protocol = json.loads((OUT / "protocol.json").read_text())
        record = next(r for r in protocol["sequences"] if (r["sensor"], r["sequence"]) == (sensor, seq))
        assert sha256(path) == record["cache_sha256"], str(path)
        assert array_sha(ids) == record["scan_ids_sha256"], sequence
    return path, ids, poses


def prepare_protocol() -> None:
    """입력·알고리즘·평가·집계 조건을 결과 전에 고정한다."""
    inputs = []
    for sensor, seq in TEST:
        path, ids, _ = keyframes(f"{sensor}/{seq}")
        inputs.append({"sensor": sensor, "sequence": seq, "frames": len(ids),
                       "cache": str(path.relative_to(REPO)), "cache_sha256": sha256(path),
                       "scan_ids_sha256": array_sha(ids)})
    sources = ["scripts/_review_native_ring_maxfix.py", "scripts/_ring_official_lib.py",
               "scripts/_ring_official_pr.py", "scripts/_causal_ring_pr.py",
               "scripts/_pdv2_main_baselines_official.py", "scripts/_pr_eval_lib.py",
               "_external/RING/utils/core.py", "_external/RING/utils/circorr2.py",
               "_external/RING/utils/extract_local_descriptor.py"]
    protocol = {
        "purpose": "현재 Table2 공통46시퀀스 RING++ 결정적 격자 max 재평가",
        "only_algorithm_change": "공식 point features를 유지하고 비원자 grid max만 검증된 np.fmax.at으로 교체",
        "base_evidence": "results/pdv2_review_native_20261105/maxfix/verification.json",
        "selection": "시험 성능으로 설정을 선택하지 않음; 반경·키프레임·후보·점수·집계 고정",
        "queries": "30 <= q < N-15", "candidates": "j <= q-30; 전체 과거 후보",
        "geometry": "기존 poses_gt 3D; 양성<=5m, 음성>25m, 중간질의 PR 제외",
        "aggregation": "45양성시퀀스 -> 6도시 -> 동일 가중 평균; KITTI07 제외",
        "expected_common_queries": 195726, "expected_common_positives": 61682,
        "expected_keyframes": sum(r["frames"] for r in inputs), "sequences": inputs,
        "source_sha256": {p: sha256(REPO / p) for p in sources},
        "chunk_size": CHUNK, "dtype": "정규화 points·특징BEV·TIRING float32",
        "preservation": "기존 공식 원장·주표·공통 코드·그림 보존; 새 결과 폴더만 사용",
        "cost_policy": "기존 Table3 시간을 수정본 실측으로 간주하지 않음",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    dest = OUT / "protocol.json"
    if dest.exists():
        old = json.loads(dest.read_text())
        for key in ("sequences", "source_sha256", "expected_keyframes", "chunk_size"):
            assert old[key] == protocol[key], f"고정 조건 변경: {key}"
        return
    write_json(dest, protocol)
    print(json.dumps({"sequences": len(inputs), "frames": protocol["expected_keyframes"]}), flush=True)


def check_protocol(aggregation: bool = False) -> dict:
    """고정한 핵심 소스가 바뀌지 않았는지 확인한다."""
    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, expected in protocol["source_sha256"].items():
        # 공통 집계는 정본 소스가 있는 로컬에서 수행한다. 서버의 옛 집계 파일은
        # 특징 생성·검색에서 호출하지도 덮어쓰지도 않는다.
        if not aggregation and path == "scripts/_pdv2_main_baselines_official.py":
            continue
        assert sha256(REPO / path) == expected, path
    return protocol


def prepare_points(sequence: str, smoke: bool = False) -> None:
    """기존 로더·공식 전처리 직후 점을 손실 없이 저장한다."""
    check_protocol()
    sensor, seq = parts(sequence)
    cache, ids, _ = keyframes(sequence)
    selected = np.linspace(0, len(ids) - 1, 8, dtype=np.int64) if smoke else np.arange(len(ids))
    stage = "smoke_points" if smoke else "points"
    directory = seq_dir(stage, sequence)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "done.json").exists():
        print("POINTS_SKIP", sequence, flush=True)
        return
    loader = rol.make_loader(sensor, seq)
    files = getattr(loader, "frame_files", None)
    if files is None:
        files = loader.scan_files
    start = time.time()
    n_points = 0
    for offset in range(0, len(selected), CHUNK):
        index = selected[offset:offset + CHUNK]
        path = directory / f"shard_{offset:06d}.npz"
        if path.exists():
            with np.load(path) as z:
                assert np.array_equal(z["scan_ids"], ids[index])
                n_points += int(z["offsets"][-1])
            continue
        points, offsets, raw_paths, raw_hashes, point_hashes = [], [0], [], [], []
        for frame in index:
            scan_id = int(ids[frame])
            raw_path = Path(files[scan_id])
            _, pcn = rol.official_points(sensor, loader[scan_id]["points"])
            pcn = np.ascontiguousarray(pcn, dtype=np.float32)
            assert pcn.ndim == 2 and pcn.shape[1] == 3 and np.isfinite(pcn).all()
            points.append(pcn)
            offsets.append(offsets[-1] + len(pcn))
            raw_paths.append(str(raw_path.relative_to(rol.DATA_ROOT)))
            raw_hashes.append(sha256(raw_path))
            point_hashes.append(array_sha(pcn))
        write_npz(path, points=np.concatenate(points), offsets=np.asarray(offsets, np.int64),
                  keyframe_indices=index, scan_ids=ids[index], raw_paths=np.asarray(raw_paths),
                  raw_sha256=np.asarray(raw_hashes), normalized_sha256=np.asarray(point_hashes))
        n_points += offsets[-1]
        progress = {"stage": stage, "sequence": sequence, "frames": offset + len(index),
                    "total": len(selected), "elapsed_s": time.time() - start,
                    "host": socket.gethostname(), "pid": os.getpid()}
        write_json(directory / "status.json", progress)
        print(json.dumps(progress), flush=True)
    write_json(directory / "done.json", {
        "stage": stage, "sequence": sequence, "frames": len(selected), "total_points": n_points,
        "scan_ids_sha256": array_sha(ids[selected]), "keyframe_cache_sha256": sha256(cache),
        "shards": {p.name: sha256(p) for p in sorted(directory.glob("shard_*.npz"))},
        "elapsed_s": time.time() - start, "input_definition": "기존 로더 -> 공식 read_pc/normalize_pc; float32 무손실",
    })


def generate(sequence: str, smoke: bool = False, direct: bool = False) -> None:
    """검증된 결정적 max와 공식 라돈을 위한 특징 격자를 생성한다."""
    import torch
    import _review_native_ring_maxfix as fix

    check_protocol()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    fix.install_maxfix()
    core = rol.import_ring_core()
    sensor, seq = parts(sequence)
    cache, ids, _ = keyframes(sequence)
    selected = np.linspace(0, len(ids) - 1, 8, dtype=np.int64) if smoke else np.arange(len(ids))
    directory = seq_dir("smoke_bev" if smoke else "bev", sequence)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "done.json").exists():
        print("BEV_SKIP", sequence, flush=True)
        return
    loader = rol.make_loader(sensor, seq) if direct else None
    points_dir = seq_dir("smoke_points" if smoke else "points", sequence)
    files = getattr(loader, "frame_files", None) if loader is not None else None
    if loader is not None and files is None:
        files = loader.scan_files
    start = time.time()
    timing = []
    count = []
    for offset in range(0, len(selected), CHUNK):
        index = selected[offset:offset + CHUNK]
        target = directory / f"shard_{offset:06d}.npz"
        if target.exists():
            with np.load(target) as z:
                assert np.array_equal(z["scan_ids"], ids[index])
                count.extend(z["n_points"].tolist())
            continue
        points_path = points_dir / f"shard_{offset:06d}.npz"
        if not direct:
            packed = np.load(points_path)
            np.testing.assert_array_equal(packed["scan_ids"], ids[index])
            all_points, offsets = packed["points"], packed["offsets"]
        occ = np.zeros((len(index), 1, 120, 120), np.uint8)
        feat = np.zeros((len(index), 6, 120, 120), np.float32)
        point_hashes, raw_hashes, n_points = [], [], []
        for k, frame in enumerate(index):
            t = time.time()
            if direct:
                _, pcn = rol.official_points(sensor, loader[int(ids[frame])]["points"])
                pcn = np.ascontiguousarray(pcn, dtype=np.float32)
                raw_hashes.append(sha256(Path(files[int(ids[frame])])))
            else:
                pcn = all_points[offsets[k]:offsets[k + 1]]
                assert array_sha(pcn) == str(packed["normalized_sha256"][k])
                raw_hashes.append(str(packed["raw_sha256"][k]))
            point_hashes.append(array_sha(pcn))
            n_points.append(len(pcn))
            with torch.no_grad():
                if len(pcn):
                    occ[k] = core.generate_bev_occ(pcn).cpu().numpy()
                if len(pcn) >= 30:
                    feat[k] = core.generate_bev_feats(pcn).cpu().numpy()
            rol.check_bev(occ[k], pcn, feat[k] if len(pcn) >= 30 else None)
            timing.append(time.time() - t)
            if k % 32 == 31:
                write_json(directory / "status.json", {
                    "stage": "smoke_bev" if smoke else "bev", "sequence": sequence,
                    "frames": offset + k + 1, "persisted_frames": offset, "total": len(selected),
                    "elapsed_s": time.time() - start, "per_frame_s": float(np.mean(timing)),
                    "host": socket.gethostname(), "pid": os.getpid(),
                })
        count.extend(n_points)
        write_npz(target, scan_ids=ids[index], keyframe_indices=index, bev_ring=occ,
                  bev_ringpp=feat, n_points=np.asarray(n_points, np.int64),
                  normalized_sha256=np.asarray(point_hashes), raw_sha256=np.asarray(raw_hashes))
        if not direct:
            packed.close()
        status = {"stage": "smoke_bev" if smoke else "bev", "sequence": sequence,
                  "frames": offset + len(index), "total": len(selected),
                  "elapsed_s": time.time() - start, "per_frame_s": float(np.mean(timing)),
                  "host": socket.gethostname(), "pid": os.getpid(), "gpu": torch.cuda.get_device_name(0)}
        write_json(directory / "status.json", status)
        print(json.dumps(status), flush=True)
    info = {"sensor": sensor, "sequence": seq, "n_frames": len(selected),
            "scan_ids_sha256": array_sha(ids[selected]), "keyframe_cache_sha256": sha256(cache),
            "table_row": rol.DATASET_TABLE[sensor], "ring_commit": rol.RING_COMMIT,
            "maxfix": "scripts/_review_native_ring_maxfix.py:install_maxfix",
            "per_frame_s": float(np.mean(timing)) if timing else None,
            "point_count_median": float(np.median(count)), "elapsed_s": time.time() - start,
            "input_mode": "direct existing loader" if direct else "lossless normalized point cache",
            "shards": {p.name: sha256(p) for p in sorted(directory.glob("shard_*.npz"))}}
    write_json(directory / "done.json", info)


def dump(sequence: str) -> None:
    """공식 검색·기존 과거 후보 규칙으로 원장을 생성한다."""
    import torch
    import _ring_official_pr as driver

    check_protocol()
    audit_path = seq_dir("bev", sequence) / "input_verification.json"
    assert json.loads(audit_path.read_text())["passed"], str(audit_path)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    driver.SHARD = CHUNK
    driver.bev_dir = lambda condition, sensor, seq: seq_dir("bev", f"{sensor}/{seq}")
    rol.OUT_ROOT = OUT
    driver.cmd_dump("odom", sequence, ["ringpp"], False, False)
    sensor, seq = parts(sequence)
    path = OUT / "odom" / f"dump_ringpp_{sensor}_{seq}.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    meta["pipeline"] = meta["pipeline"].replace("generate_bev_feats", "generate_bev_feats with deterministic np.fmax.at grid aggregation")
    meta["maxfix_protocol_sha256"] = sha256(OUT / "protocol.json")
    meta["cost_scope"] = "진행 시간만 기록; Table3 실측으로 사용하지 않음"
    arrays["meta"] = json.dumps(meta)
    write_npz(path, **arrays)
    write_json(seq_dir("ledger_status", sequence) / "done.json", {
        "sequence": sequence, "ledger": str(path.relative_to(REPO)), "sha256": sha256(path),
        "queries": len(arrays["query_idx"]), "complete": True,
        "host": socket.gethostname(), "pid": os.getpid(),
    })


def audit_input(sequence: str, smoke: bool = False) -> None:
    """이전 공식 BEV의 스캔ID·점수·점유격자와 새 입력을 대조한다."""
    sensor, seq = parts(sequence)
    directory = seq_dir("smoke_bev" if smoke else "bev", sequence)
    original = REPO / "_external/ring_official_cache/odom" / f"{sensor}_{seq}"
    old_blocks = {}
    checked = 0
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as new:
            occ = new["bev_ring"]
            for k, frame in enumerate(new["keyframe_indices"]):
                old_start = int(frame) // 1000 * 1000
                if old_start not in old_blocks:
                    with np.load(original / f"shard_{old_start:06d}.npz") as z:
                        old_blocks[old_start] = {name: z[name] for name in ("scan_ids", "n_points", "bev_ring")}
                old = old_blocks[old_start]
                old_index = int(frame) - old_start
                assert int(old["scan_ids"][old_index]) == int(new["scan_ids"][k]), (sequence, int(frame), "scan_ids")
                assert int(old["n_points"][old_index]) == int(new["n_points"][k]), (sequence, int(frame), "n_points")
                assert np.array_equal(old["bev_ring"][old_index], occ[k]), (sequence, int(frame), "occupancy")
                checked += 1
    done = json.loads((directory / "done.json").read_text())
    assert checked == done["n_frames"], (checked, done["n_frames"])
    write_json(directory / "input_verification.json", {
        "passed": True, "sequence": sequence, "frames": checked,
        "checks": ["기존공식 scan_ids 동일", "정규화 점 개수 동일", "기존공식 occupancy 비트 동일"],
        "reference": str(original.relative_to(REPO)),
    })
    print("INPUT_AUDIT", sequence, checked, "passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["protocol", "points", "generate", "audit", "dump"])
    parser.add_argument("--sequence")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    if args.stage == "protocol":
        prepare_protocol()
    elif args.stage == "points":
        prepare_points(args.sequence, args.smoke)
    elif args.stage == "generate":
        generate(args.sequence, args.smoke, args.direct)
    elif args.stage == "audit":
        audit_input(args.sequence, args.smoke)
    else:
        dump(args.sequence)
