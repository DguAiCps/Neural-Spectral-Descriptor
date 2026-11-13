#!/usr/bin/env python3
"""공식 NCLT 점 읽기(z 뒤집기) 뒤의 좌표계에서 레이저 앙각·빔 원점·지면 높이를 실제 스캔으로 잰다.

호스트에서 원시 스캔 6개만 읽는다(torch 없음).
    python3 scripts/_review_ring_followup_sensor_frame.py --nclt-root /mnt/d/NSD_datasets/nclt

공식 NCLT 설정은 z를 뒤집은 뒤 z > 1.0 m, z < 20 m만 남긴다(_external/RING/datasets/NCLTDataset.py:45,51,79).
KITTI·MulRan은 음수 기준(−1.0, −0.9)이므로 이 양수 기준이 지면 제거로 타당한지 확인한다.
레이저 번호마다 z = a + r·tan(e)를 최소제곱으로 맞춰 앙각 e와 빔 원점 높이 a를 얻는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring/sensor_frame.json"
RAW_LIST = REPO / "results/pdv2_review_native_20261105/raw_files.txt"
HDL32E_ELEVATION_DEG = (-30.67, 10.67)
DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("i", "u1"), ("l", "u1")])


def main() -> None:
    """정렬 목록의 고정 6개 스캔에서 통계를 낸다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nclt-root", type=Path, default=Path("/mnt/d/NSD_datasets/nclt"))
    args = parser.parse_args()
    files = RAW_LIST.read_text().split()
    picks = [files[i] for i in np.linspace(0, len(files) - 1, 6).round().astype(int)]
    rows = []
    for rel in picks:
        path = args.nclt_root / rel
        raw = np.fromfile(path, dtype=DTYPE)
        xyz = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float64) * 0.005 - 100.0
        keep = ~np.all(np.isclose(xyz, 0), axis=1)
        xyz, laser = xyz[keep], raw["l"][keep]
        z = -xyz[:, 2]                                      # load_pc_nclt의 z 뒤집기
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        fits = []
        for k in range(32):
            m = laser == k
            if m.sum() < 20:
                continue
            coef, *_ = np.linalg.lstsq(np.stack([np.ones(m.sum()), r[m]], 1), z[m], rcond=None)
            fits.append((float(coef[0]), float(np.degrees(np.arctan(coef[1])))))
        fits = np.array(fits)
        band = (z > 1.0) & (z < 20.0) & (np.abs(xyz[:, 0]) < 70) & (np.abs(xyz[:, 1]) < 70)
        rows.append({"raw_path": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "points": int(len(z)), "lasers_fitted": int(len(fits)),
                     "fitted_elevation_deg_min_max": [float(fits[:, 1].min()), float(fits[:, 1].max())],
                     "fitted_beam_origin_z_m_median": float(np.median(fits[:, 0])),
                     "z_quantiles_05_25_50_75_95": [float(v) for v in np.quantile(z, [.05, .25, .5, .75, .95])],
                     "fraction_z_below_minus_0p5": float((z < -0.5).mean()),
                     "fraction_kept_by_official_band": float(band.mean())})
        print(json.dumps(rows[-1]), flush=True)
    report = {
        "purpose": "공식 NCLT 높이 대역(z>1, z<20)이 뒤집은 좌표계에서 무엇을 남기는지 확인; 설정 변경 없음",
        "hdl32e_nominal_elevation_deg": HDL32E_ELEVATION_DEG,
        "rows": rows,
        "interpretation": (
            "뒤집은 좌표계의 레이저 앙각이 HDL-32E 공칭 범위를 위아래로 뒤집은 −10.67°~+30.67°이면 센서가 거꾸로 "
            "장착된 것이고, 빔 원점이 z≈+1m이면 점은 센서 중심이 아닌 원점에서 표현된다. 이 경우 z>1.0은 센서 높이 "
            "이상 구조물만 남기는 지면 제거로 작동한다."),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
