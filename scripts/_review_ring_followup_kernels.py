#!/usr/bin/env python3
"""서버 컨테이너의 공식 RING CUDA 확장(torch_radon, voxelfeat)을 실제 NCLT 스캔에서 독립 계산과 대조한다.

서버 호스트에서 GPU0 락을 잡고 실행한다.
    flock -x /tmp/nsd_native_reproduction_gpu0.lock docker exec -e CUDA_VISIBLE_DEVICES=0 \
        -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_review_ring_followup_kernels.py

대상 스캔은 정렬한 원시 목록 1,413개에서 np.linspace(0, 1412, 12)로 고정한다(결과와 무관한 규칙).
1) generate_RING의 torch_radon ParallelBeam 투영: 반경 58칸 원 밖을 0으로 둔 BEV에서 질량 보존과
   skimage.transform.radon(공식 CPU 경로 generate_RING_cpu가 쓰는 구현)과의 상관을 잰다.
2) generate_RING TIRING을 기존 서술자 캐시와 비교한다(결정적 재현 여부).
3) voxelfeat.GPUFeatureExtractor의 6개 점 특징을 공식 파이썬 기준식
   utils/extract_local_descriptor.py:83 calculate_features와 점마다 비교한다.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring/kernels"
NATIVE = REPO / "results/pdv2_review_native_20261105"
sys.path.insert(0, str(REPO / "scripts"))
import _ring_official_lib as rol  # noqa: E402


def digest(path: Path) -> str:
    """파일의 SHA256을 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """고정 12개 스캔에서 세 가지 대조를 실행하고 report.json을 쓴다."""
    import torch
    from skimage.transform import radon as sk_radon

    torch.set_num_threads(4)
    core = rol.import_ring_core()
    from datasets.NCLTDataset import NCLTPointCloudLoader
    import voxelfeat
    from utils.extract_local_descriptor import calculate_features

    OUT.mkdir(parents=True, exist_ok=True)
    raw = (NATIVE / "raw_files.txt").read_text().split()
    picks = np.linspace(0, len(raw) - 1, 12).round().astype(int)
    loader = NCLTPointCloudLoader()
    n = 120
    yy, xx = np.mgrid[:n, :n]
    inside = (xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2 <= 58.0 ** 2   # 두 구현의 내접원 안
    angles = np.linspace(0, 2 * np.pi, n).astype(np.float32)
    deg = np.degrees(angles.astype(np.float64))
    rows = []
    for idx in picks:
        rel = raw[idx]
        pc = loader.read_pc(str(NATIVE / "native_data" / rel))[:, :3]
        pc_norm = loader.normalize_pc(pc)
        with torch.no_grad():
            occ = core.generate_bev_occ(pc_norm)
            bev = occ.cpu().numpy()[0].astype(np.float64)
            # (1) 반경 58칸 원 마스크 BEV: torch_radon vs skimage
            masked = torch.from_numpy((bev * inside)[None].astype(np.float32)).to(occ.device)
            sino_gpu = core.generate_RING(masked)[0].cpu().numpy()[0].astype(np.float64)   # (angle, det)
            tiring = core.generate_RING(occ)[1].cpu().numpy()
        cand = {}
        for tr in (False, True):
            img = (bev * inside).T if tr else bev * inside
            for sign in (1, -1):
                sk = sk_radon(img, theta=sign * deg, circle=True).T             # (angle, det)
                for flip in (False, True):
                    s = sk[:, ::-1] if flip else sk
                    c = np.corrcoef(s.ravel(), sino_gpu.ravel())[0, 1]
                    rel_l2 = np.linalg.norm(s - sino_gpu) / max(np.linalg.norm(sino_gpu), 1e-12)
                    cand[f"transpose={tr},angle_sign={sign},det_flip={flip}"] = (float(c), float(rel_l2))
        best = max(cand, key=lambda k: cand[k][0])
        mass = float((bev * inside).sum())
        proj_mass = sino_gpu.sum(axis=1)
        # (2) 캐시 TIRING과 비교
        key = rel.replace("/", "_").replace(".bin", ".npz")
        with np.load(NATIVE / "descriptor_cache" / key) as z:
            cached = z["ring"]
        # (3) voxelfeat 6특징 vs 파이썬 기준식
        idx_nn, _, _, eig, vec = core.build_neighbors_NN(pc_norm[:, :3], 30)
        gpu = voxelfeat.GPUFeatureExtractor(pc_norm[:, :3].flatten(), len(pc_norm), 13, 30,
                                            idx_nn.flatten().astype(np.int32), eig.flatten()).get_features()
        gpu = gpu.reshape(-1, 13)[:, [0, 1, 3, 10, 11, 12]].astype(np.float64)
        ref = np.stack([calculate_features(pc_norm[:, :3], idx_nn[i], eig[i], None)
                        for i in range(len(pc_norm))]).astype(np.float64)
        both = np.isfinite(gpu) & np.isfinite(ref)
        diff = np.abs(gpu - ref)
        scale = np.maximum(np.abs(ref), 1e-6)
        rows.append({
            "raw_path": rel, "n_points_norm": int(len(pc_norm)),
            "occupied_cells": int(bev.sum()), "occupied_outside_r58_circle": int((bev * ~inside).sum()),
            "radon_best_convention": best, "radon_best_corr": cand[best][0], "radon_best_rel_l2": cand[best][1],
            "radon_all_conventions": cand,
            "projection_mass_rel_err_max": float(np.abs(proj_mass - mass).max() / max(mass, 1e-12)),
            "tiring_cache_max_abs_diff": float(np.abs(tiring - cached).max()),
            "tiring_cache_bitwise_equal": bool(np.array_equal(tiring, cached)),
            "feature_nonfinite_mismatch": int((np.isfinite(gpu) != np.isfinite(ref)).sum()),
            "feature_max_abs_diff_per_channel": [float(diff[:, c][both[:, c]].max()) for c in range(6)],
            "feature_max_rel_diff_per_channel": [float((diff / scale)[:, c][both[:, c]].max()) for c in range(6)],
            "feature_rel_diff_q999_per_channel": [float(np.quantile((diff / scale)[:, c][both[:, c]], .999))
                                                  for c in range(6)],
            "feature_frac_rel_diff_gt_1e-3": [float(((diff / scale)[:, c][both[:, c]] > 1e-3).mean())
                                              for c in range(6)],
        })
        print(json.dumps({k: rows[-1][k] for k in ("raw_path", "radon_best_corr", "radon_best_rel_l2",
                                                    "projection_mass_rel_err_max", "tiring_cache_bitwise_equal",
                                                    "feature_max_rel_diff_per_channel")}), flush=True)
    report = {
        "purpose": "서버 공식 CUDA 확장의 실제 스캔 출력 대조; 설정·결과 선택 없음",
        "selection_rule": "sorted raw_files.txt, np.linspace(0, 1412, 12).round()",
        "feature_order": ["change_of_curvature", "omnivariance", "eigen_entropy(normalized)", "2d_linearity",
                          "max_height_diff", "height_variance"],
        "rows": rows,
        "summary": {
            "radon_min_best_corr": min(r["radon_best_corr"] for r in rows),
            "radon_max_best_rel_l2": max(r["radon_best_rel_l2"] for r in rows),
            "radon_best_conventions": sorted({r["radon_best_convention"] for r in rows}),
            "projection_mass_rel_err_max": max(r["projection_mass_rel_err_max"] for r in rows),
            "tiring_cache_all_bitwise_equal": all(r["tiring_cache_bitwise_equal"] for r in rows),
            "tiring_cache_max_abs_diff": max(r["tiring_cache_max_abs_diff"] for r in rows),
            "feature_max_rel_diff": max(max(r["feature_max_rel_diff_per_channel"]) for r in rows),
            "feature_rel_diff_q999_max": max(max(r["feature_rel_diff_q999_per_channel"]) for r in rows),
            "feature_frac_rel_diff_gt_1e-3_max": max(max(r["feature_frac_rel_diff_gt_1e-3"]) for r in rows),
            "feature_nonfinite_mismatch": sum(r["feature_nonfinite_mismatch"] for r in rows),
            "occupied_outside_circle_fraction_mean": float(np.mean(
                [r["occupied_outside_r58_circle"] / max(r["occupied_cells"], 1) for r in rows])),
        },
        "gpu": torch.cuda.get_device_name(0),
        "script_sha256": digest(Path(__file__)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
