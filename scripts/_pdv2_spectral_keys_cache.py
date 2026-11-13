#!/usr/bin/env python3
"""기존 키프레임과 센서 설정으로 18/22주파수 키의 공통 계수를 추출한다."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal_derivatives as base
from _pd_build_test_caches import _effective_config, _make_desc_encoder, _make_loader
from _nclt_fov_fix import fixed_config
from _pdv2_matching_mine import cache_path
from _pdv2_no_context_ablation import sha256, write_json
from encoding.frequency_key import frequency_key

OUT = REPO / "results/pdv2_spectral_keys_20261025"
CACHE = REPO / "data/pdv2_spectral_keys_20261025"
LOG = REPO / "logs/spectral_keys_20261025"


def source_path(role: str, sequence: str) -> Path:
    """학습 GT와 검증·시험 KISS의 확정 캐시를 반환한다."""
    return (
        cache_path(sequence)
        if role == "train"
        else base.exp.old.base.inputs(role, sequence)[2]["cache"]
    )


def output_path(role: str, sequence: str) -> Path:
    """원본과 분리한 낮은 주파수 캐시 경로를 반환한다."""
    return CACHE / role / (sequence.replace("/", "_") + ".npz")


def freeze_cache() -> dict:
    """자료 추출 전에 정의·센서 설정·소스를 동결한다."""
    path = OUT / "cache_protocol.json"
    if path.exists():
        spec = base.read(path)
        for name, value in spec["runtime"].items():
            assert sha256(REPO / name) == value, name
        return spec
    parent = base.read(
        REPO / "results/pdv2_geometry_interaction_20261023/protocol.json"
    )
    files = [
        Path(__file__),
        REPO / "src/encoding/frequency_key.py",
        REPO / "src/encoding/spectral_encoder.py",
        REPO / "src/encoding/range_image.py",
        REPO / "scripts/_pd_build_test_caches.py",
        REPO / "scripts/_pd_build_val_layouts.py",
        REPO / "scripts/_nclt_fov_fix.py",
        REPO / "configs/training_place_disjoint.yaml",
        REPO / "train_multi_dataset.py",
    ]
    files.extend((REPO / "src/data").glob("*loader.py"))
    spec = {
        "runtime": {str(p.relative_to(REPO)): sha256(p) for p in files},
        "train": parent["train"],
        "val": parent["val"],
        "test": [s + "/" + q for s, q in base.fixed.TEST],
        "definition": "현재 투영·보간·16높이행·360방향·로그 크기 변환까지 동일. "
        "0..17/0..21 주파수를 행 우선으로 저장. 평균·표준편차 집계만 생략한다. "
        "모든 스캔에서 원288 복원과 캐시를 대조. 정답으로 추출 설정을 고르지 않음.",
        "nclt_window": [-10.67, 30.67],
        "test_used_for_selection": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(path, spec)
    return spec


def worker(role: str, sequence: str, limit: int = 0) -> None:
    """같은 스캔에서 절단 전 계수를 추출하고 기존288 키 복원을 검사한다."""
    spec = freeze_cache()
    assert sequence in spec[role]
    if role == "test":
        assert (OUT / "test_freeze.json").exists(), "시험 설정 동결 후 추출한다"
    path = output_path(role, sequence)
    if limit:
        path = CACHE / "smoke" / (role + "_" + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(".json").exists():
        report = base.read(path.with_suffix(".json"))
        assert sha256(path) == report["sha256"]
        assert sha256(source_path(role, sequence)) == report["source_sha256"]
        return
    assert not path.exists(), "완료되지 않은 캐시를 덮어쓰지 않는다"
    original = base.exp.arrays(source_path(role, sequence))
    sensor, seq = sequence.split("/")
    cfg = fixed_config() if sensor == "nclt" else _effective_config()
    enc = _make_desc_encoder(cfg, sensor, "cuda").eval()
    assert not enc.normalize_channels and enc.output_dim == 288
    loader = _make_loader(sensor, seq)
    ids = original["scan_ids"][:limit] if limit else original["scan_ids"]
    low = np.empty((len(ids), 16, 22), np.float32)
    errors = np.empty(len(ids), np.float64)
    point_hash = hashlib.sha256()
    started = time.time()
    with torch.no_grad():
        for row, sid in enumerate(ids):
            item = loader[int(sid)]
            points = np.ascontiguousarray(item["points"])
            point_hash.update(np.asarray([sid, *points.shape], np.int64).tobytes())
            point_hash.update(points.tobytes())
            mags = enc.compute_fft_magnitudes(points)
            assert mags.shape == (16, 181) and np.isfinite(mags).all()
            low[row] = mags[:, :22]
            recovered = (
                enc._bin_fft_magnitudes(
                    torch.from_numpy(mags).cuda(), enc._compute_bin_edges(enc.alpha)
                )
                .cpu()
                .numpy()
            )
            ref = original["descriptors"][row]
            errors[row] = float(abs(recovered - ref).max())
            assert errors[row] <= 2e-6 * max(float(abs(ref).max()), 1.0), (
                sequence,
                sid,
                errors[row],
            )
            if row in (0, len(ids) // 2, len(ids) - 1):
                np.testing.assert_allclose(
                    enc.encode_points(points).cpu().numpy(),
                    recovered,
                    atol=1e-6,
                    rtol=1e-6,
                )
            if (row + 1) % 1000 == 0:
                print(
                    "SPECTRAL_CACHE",
                    role,
                    sequence,
                    row + 1,
                    len(ids),
                    round(time.time() - started, 1),
                    flush=True,
                )
    np.savez_compressed(path, low22=low, scan_ids=ids, reconstruction_error=errors)
    for k in (18, 22):
        assert frequency_key(low, k).shape == (len(ids), 16 * k)
    write_json(
        path.with_suffix(".json"),
        {
            "sha256": sha256(path),
            "role": role,
            "sequence": sequence,
            "source": str(source_path(role, sequence).relative_to(REPO)),
            "source_sha256": sha256(source_path(role, sequence)),
            "cache_protocol_sha256": sha256(OUT / "cache_protocol.json"),
            "frames": len(ids),
            "reconstruction_max_error": float(errors.max()),
            "point_arrays_sha256": point_hash.hexdigest(),
            "seconds": time.time() - started,
            "test_used_for_selection": False,
            "limit": limit,
        },
    )
    print(
        "SPECTRAL_CACHE_COMPLETE",
        role,
        sequence,
        len(ids),
        round(time.time() - started, 1),
        flush=True,
    )


def main() -> None:
    """학습·검증 또는 동결 후 시험의 캐시를 독립 프로세스 네 개로 추출한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "worker", "run"))
    parser.add_argument("--role", choices=("train", "val", "test"))
    parser.add_argument("--sequence")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "worker":
        worker(args.role, args.sequence, args.limit)
        return
    spec = freeze_cache()
    if args.action == "freeze":
        return
    LOG.mkdir(parents=True, exist_ok=True)
    roles = [args.role] if args.role else ["train", "val"]
    jobs = [(role, seq) for role in roles for seq in spec[role]]

    def invoke(job: tuple) -> None:
        role, seq = job
        log = LOG / ("cache_" + role + "_" + seq.replace("/", "_") + ".log")
        with log.open("a") as stream:
            p = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "worker",
                    "--role",
                    role,
                    "--sequence",
                    seq,
                ],
                cwd=REPO,
                env=os.environ.copy(),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if p.returncode:
            raise RuntimeError(log.read_text()[-5000:])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(invoke, job) for job in jobs]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            write_json(
                OUT / ("cache_status_" + "_".join(roles) + ".json"),
                {"done": count, "total": len(jobs), "roles": roles},
            )
            print("CACHE_JOB", count, len(jobs), flush=True)


if __name__ == "__main__":
    main()
