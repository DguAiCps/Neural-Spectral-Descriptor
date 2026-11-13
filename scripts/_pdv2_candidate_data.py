#!/usr/bin/env python3
"""학습 8시퀀스의 주행 추정 키프레임과 현재 스펙트럼을 준비한다."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
OUT = REPO / "results/pdv2_candidate_learning_20261014"
from _pdv2_pilot import TRAIN, VAL
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_pair_evaluate import ArrayCache


def train_path(sequence: str) -> Path:
    """학습 전용 캐시 경로를 반환한다."""
    assert tuple(sequence.split("/")) in TRAIN
    return OUT / "data" / (sequence.replace("/", "_") + ".npz")


def prepare(sequence: str) -> None:
    """모든 원시 스캔에서 KISS-ICP를 실행하고 같은 선택·인코딩을 적용한다."""
    import torch
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP
    from keyframe.selector import KeyframeSelector
    from _pd_build_test_caches import _make_loader, _effective_config, _make_desc_encoder, CONFIG
    from run_kitti_operating_point import _make_encoder, _load_config, _project_nsd_layout
    from _pbev_branch_fusion import BP
    from _height_candidate_fusion import make_height_grids
    from encoding.phase_features import _phase_sketch, _pool_rows
    from _pdv2_matching_mine import cache_path, spec_path
    torch.set_num_threads(1)
    target = train_path(sequence)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert json.loads(target.with_suffix(".json").read_text())["sha256"] == sha256(target)
        return
    sensor, seq = sequence.split("/")
    cfg = _effective_config()
    encoder = _make_desc_encoder(cfg, sensor, "cuda")
    layout_encoder = _make_encoder(_load_config(CONFIG), "cpu")
    selector = KeyframeSelector(**{k: cfg["keyframe"][k] for k in
                                 ("distance_threshold", "rotation_threshold", "overlap_threshold", "temporal_threshold")})
    kiss_cfg = load_config(None)
    kiss_cfg.data.deskew = False
    kiss_cfg.data.min_range = 1.
    kiss_cfg.data.max_range = 100.
    kiss_cfg.registration.max_num_threads = 2
    odom = KissICP(kiss_cfg)
    loader = _make_loader(sensor, seq)
    # 원시 스캔 하나의 특징만 재사용한다. 키프레임 선택·자세는 새 주행 추정이다.
    old_path, spectrum_path = cache_path(sequence), spec_path(sequence)
    with np.load(old_path) as z:
        old_ids, old_raw = z["scan_ids"], z["descriptors"]
    with np.load(spectrum_path) as z:
        np.testing.assert_array_equal(old_ids, z["scan_ids"])
        old_spec = {k: z[k] for k in ("sk_re", "sk_im", "raw_height")}
    old_index = {int(sid): i for i, sid in enumerate(old_ids)}
    rows = {k: [] for k in ("scan_ids", "poses", "poses_gt", "timestamps", "descriptors",
                            "nsd_layouts", "sk_re", "sk_im", "raw_height")}
    all_odom, all_gt, all_stamps = [], [], []
    started = time.time()
    reused = 0
    for sid in range(len(loader)):
        item = loader[sid]
        points = item["points"]
        odom.register_frame(np.asarray(points)[:, :3].astype(np.float64), np.empty(0))
        pose = odom.last_pose.copy()
        all_odom.append(pose)
        all_gt.append(item["pose"])
        all_stamps.append(float(item["timestamp"]))
        selected, kf, _ = selector.process_scan(sid, points, pose, float(item["timestamp"]))
        if selected:
            if sid in old_index:
                i = old_index[sid]
                raw = old_raw[i]
                re, im, height = [old_spec[k][i] for k in ("sk_re", "sk_im", "raw_height")]
                reused += 1
                if reused <= 3:
                    with torch.no_grad():
                        np.testing.assert_allclose(encoder.encode_points(points).cpu().numpy(), raw, atol=1e-6, rtol=1e-6)
            else:
                with torch.no_grad():
                    raw = encoder.encode_points(points).cpu().numpy()
                bev, _ = BP.project(points, keep_intensity=False)
                sketch = _phase_sketch(_pool_rows(np.maximum(bev, 0.), 16), 12)
                re, im = sketch.real, sketch.imag
                _, height, _, _ = make_height_grids(points)
                height = height.astype(np.float16)
            layout = _project_nsd_layout(layout_encoder, points, n_layout_sectors=60)
            values = (sid, pose, item["pose"], float(item["timestamp"]), raw, layout, re, im, height)
            for key, value in zip(rows, values):
                rows[key].append(value)
            # 기존 process_dataset과 같이 저장 점군을 비워 같은 선택 규약을 유지한다.
            kf.points = np.empty((0, 3), np.float32)
        if sid % 1000 == 0:
            write_json(target.with_suffix(".progress.json"), {"sequence": sequence, "scans": sid + 1,
                       "total_scans": len(loader), "keyframes": len(rows["scan_ids"]), "runtime_s": time.time() - started})
            print("DATA", sequence, sid, len(loader), len(rows["scan_ids"]), round(time.time() - started, 1), flush=True)
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays["keyframe_ids"] = np.arange(len(rows["scan_ids"]))
    assert arrays["descriptors"].shape[1] == 288
    assert all(np.isfinite(v).all() for v in arrays.values())
    temp = target.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(target)
    np.savez_compressed(target.with_name(target.stem + "_odometry.npz"),
                        odom_poses=all_odom, gt_poses=all_gt, timestamps=all_stamps)
    write_json(target.with_suffix(".json"), {"sequence": sequence, "n_scans": len(loader),
               "n_keyframes": len(arrays["scan_ids"]), "reused_single_scan_features": reused,
               "kiss_config": kiss_cfg.model_dump(), "runtime_s": time.time() - started,
               "sha256": sha256(target), "script_sha256": sha256(Path(__file__)),
               "sources": {str(p.relative_to(REPO)): sha256(p) for p in (old_path, spectrum_path)}})
    print("DATA_DONE", sequence, flush=True)


def load_data(role: str, sequence: str) -> tuple:
    """정답 배열을 입력 캐시와 분리하여 학습·검증 자료만 반환한다."""
    if role == "train":
        path = train_path(sequence)
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        spec = {k: arrays.pop(k) for k in ("sk_re", "sk_im", "raw_height")}
        layout = arrays.pop("nsd_layouts")
        paths = {"cache": path}
        min_past = None
    else:
        assert role == "val" and tuple(sequence.split("/")) in VAL
        from _pdv2_lag15_campaign import inputs
        _, _, paths, min_past = inputs("val", sequence)
        with np.load(paths["cache"]) as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(paths["layouts"]) as z:
            np.testing.assert_array_equal(arrays["scan_ids"], z["scan_ids"])
            layout = z["nsd_layouts"]
        with np.load(paths["spec"]) as z:
            np.testing.assert_array_equal(arrays["scan_ids"], z["scan_ids"])
            spec = {k: z[k] for k in ("sk_re", "sk_im", "raw_height")}
    gt = arrays.pop("poses_gt")
    return ArrayCache(arrays), layout, spec, gt, min_past, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    prepare(parser.parse_args().sequence)
