#!/usr/bin/env python3
"""torch_radon 2D 평행빔 순방향 커널을 numpy로 옮겨 서버 GPU 빌드의 TING 캐시와 비교한다(CPU).

    docker exec -e OMP_NUM_THREADS=1 -e CUDA_VISIBLE_DEVICES= -w /workspace/Neural-Spectral-Codec ring_baseline \
        python3 scripts/_review_ring_followup_radon_replica.py

커널: _external/RING/utils/torch-radon/src/forward.cu:12-121 (광선 시작·방향 회전, 볼륨 절단, 반올림 단계 수,
반 칸 정렬 첫 표본, cudaFilterModeLinear·cudaAddressModeBorder 텍스처 표본 합 × hypot(vx, vy)).
CUDA 선형 필터는 보간 가중치를 8비트 소수부로 양자화하므로 양자화 유무 두 복제본을 함께 쓴다.
BEV는 _ring_official_lib.occupancy_numpy(voxelocc 커널 전사, 기존 check_bev로 GPU와 일치 확인)로 만든다.
대상: 정렬한 원시 목록 1,413개에서 np.linspace(0, 1412, 48)로 고정한 스캔.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring/kernels/radon_replica.json"
NATIVE = REPO / "results/pdv2_review_native_20261105"
sys.path.insert(0, str(REPO / "scripts"))
import _ring_official_lib as rol  # noqa: E402


def sample(tex: np.ndarray, x: np.ndarray, y: np.ndarray, quantize: bool) -> np.ndarray:
    """비정규 좌표 쌍선형 텍스처 읽기(텍셀 중심 +0.5, 경계 밖 0)."""
    h, w = tex.shape
    xb, yb = x - np.float32(0.5), y - np.float32(0.5)
    j0, i0 = np.floor(xb).astype(np.int64), np.floor(yb).astype(np.int64)
    a, b = (xb - j0).astype(np.float32), (yb - i0).astype(np.float32)
    if quantize:
        a, b = np.round(a * 256) / 256, np.round(b * 256) / 256
    pad = np.zeros((h + 2, w + 2), np.float32)
    pad[1:-1, 1:-1] = tex
    i0c, j0c = np.clip(i0 + 1, 0, h + 1), np.clip(j0 + 1, 0, w + 1)
    i1c, j1c = np.clip(i0 + 2, 0, h + 1), np.clip(j0 + 2, 0, w + 1)
    ok00 = (i0 >= -1) & (i0 < h) & (j0 >= -1) & (j0 < w)
    v = ((1 - a) * (1 - b) * pad[i0c, j0c] + a * (1 - b) * pad[i0c, j1c]
         + (1 - a) * b * pad[i1c, j0c] + a * b * pad[i1c, j1c])
    return np.where(ok00, v, 0).astype(np.float32)


def radon_replica(img: np.ndarray, angles: np.ndarray, quantize: bool) -> np.ndarray:
    """ParallelBeam(det_count=120, Volume2D(120,120)).forward의 단일 채널 복제."""
    h, w = img.shape
    det = 120
    f = np.float32
    sx = (np.arange(det, dtype=f) - f(det * 0.5) + f(0.5))
    sy = f(np.hypot(w * 0.5, h * 0.5))
    out = np.zeros((len(angles), det), np.float32)
    for k, ang in enumerate(angles.astype(f)):
        cs, sn = f(np.cos(ang)), f(np.sin(ang))
        rsx, rsy = sx * cs + sy * sn, -sx * sn + sy * cs
        ex, ey = sx, -sy
        rdx, rdy = ex * cs + ey * sn - rsx, -ex * sn + ey * cs - rsy
        rsx, rsy = rsx + f(0.5 * w), rsy + f(0.5 * h)
        dx = np.where(rdx >= 0, np.maximum(rdx, f(1e-6)), np.minimum(rdx, f(-1e-6)))
        dy = np.where(rdy >= 0, np.maximum(rdy, f(1e-6)), np.minimum(rdy, f(-1e-6)))
        axm, axp, aym, ayp = -rsx / dx, (w - rsx) / dx, -rsy / dy, (h - rsy) / dy
        a_s = np.maximum(np.minimum(axp, axm), np.minimum(ayp, aym))
        a_e = np.minimum(np.maximum(axp, axm), np.maximum(ayp, aym))
        valid = a_s <= a_e
        rsx, rsy = rsx + rdx * a_s, rsy + rdy * a_s
        rdx, rdy = rdx * (a_e - a_s), rdy * (a_e - a_s)
        m = np.maximum(np.abs(rdx), np.abs(rdy))
        n_steps = np.rint(m).astype(np.int64)
        vx, vy = rdx / m, rdy / m
        norm = np.hypot(vx, vy)
        ymajor = np.abs(rdy) >= np.abs(rdx)
        step = np.where(ymajor, (f(0.5) - rsy + np.rint(rsy)) / vy + (vy < 0),
                        (f(0.5) - rsx + np.rint(rsx)) / vx + (vx < 0)).astype(f)
        rsx, rsy = rsx + step * vx, rsy + step * vy
        acc = np.zeros(det, np.float32)
        for j in range(int(n_steps[valid].max()) if valid.any() else 0):
            live = valid & (j < n_steps)
            acc += np.where(live, sample(img, rsx, rsy, quantize), 0)
            rsx, rsy = rsx + vx, rsy + vy
        out[k] = np.where(valid, acc * norm, 0)
    return out


def main() -> None:
    """고정 48개 스캔의 복제 TING과 GPU 캐시 TING을 비교한다."""
    rol.import_ring()
    from datasets.NCLTDataset import NCLTPointCloudLoader

    raw = (NATIVE / "raw_files.txt").read_text().split()
    picks = np.linspace(0, len(raw) - 1, 48).round().astype(int)
    loader = NCLTPointCloudLoader()
    angles = np.linspace(0, 2 * np.pi, 120).astype(np.float32)
    rows = []
    for idx in picks:
        rel = raw[idx]
        pc_norm = loader.normalize_pc(loader.read_pc(str(NATIVE / "native_data" / rel))[:, :3])
        bev = rol.occupancy_numpy(pc_norm).astype(np.float32)
        with np.load(NATIVE / "descriptor_cache" / (rel.replace("/", "_").replace(".bin", ".npz"))) as z:
            cached = z["ring"][0].astype(np.float64)
        row = {"raw_path": rel, "occupied_cells": int(bev.sum())}
        for q in (False, True):
            sino = radon_replica(bev, angles, q)
            ting = np.abs(np.fft.fft(sino.astype(np.float64), axis=-1, norm="ortho"))
            key = "quantized" if q else "exact"
            row[key] = {"ting_rel_l2_vs_gpu_cache": float(np.linalg.norm(ting - cached) / np.linalg.norm(cached)),
                        "ting_max_abs_diff": float(np.abs(ting - cached).max()),
                        "ting_corr": float(np.corrcoef(ting.ravel(), cached.ravel())[0, 1])}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "purpose": "서버 torch_radon CUDA 빌드가 공개 커널 알고리즘과 같은 값을 내는지 확인; 설정 변경 없음",
        "selection_rule": "sorted raw_files.txt, np.linspace(0, 1412, 48).round()",
        "rows": rows,
        "summary": {k: {"max_rel_l2": max(r[k]["ting_rel_l2_vs_gpu_cache"] for r in rows),
                        "min_corr": min(r[k]["ting_corr"] for r in rows)} for k in ("exact", "quantized")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
