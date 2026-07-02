#!/usr/bin/env python3
"""Build a MulRan descriptor cache (mirrors _build_nclt_cache) using the new MulRanLoader.
Produces npz {descriptors(288), poses, timestamps, scan_ids, keyframe_ids, fft_magnitudes, nsd_layouts}
in the SAME format as KITTI/NCLT caches, so the existing dump/aliasrate/whitening/identifiability
analyses run unchanged. Encoder elevation set to Ouster OS1-64 [-16.6, 16.6]."""
import sys, os, argparse
sys.path.insert(0, "src")
import numpy as np, torch
from data.mulran_loader import MulRanLoader
from keyframe.selector import KeyframeSelector
from run_kitti_operating_point import _make_encoder, _project_nsd_layout
from evaluate_dump import _apply_encoder_preset, _load_config

def build(root, seq, config, cache_dir, device, layout_sectors, stride):
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"mulran_operating_{seq}_layout{layout_sectors}_stride{stride}.npz")
    if os.path.exists(out):
        print("exists", out); return out
    loader = MulRanLoader(root, seq, lazy_load=True)
    print(f"[{seq}] scans={len(loader)}")
    encoder = _make_encoder(config, device)
    k = config["keyframe"]
    selector = KeyframeSelector(
        distance_threshold=k.get("distance_threshold", 0.8),
        rotation_threshold=k.get("rotation_threshold", 20.0),
        overlap_threshold=k.get("overlap_threshold", 0.65),
        temporal_threshold=k.get("temporal_threshold", 30.0),
        voxel_size=k.get("voxel_size", 0.2),
        max_keyframes=k.get("max_keyframes", 10000000),
    )
    D, P, T, S, K, FF, NL = [], [], [], [], [], [], []
    for sid in range(0, len(loader), stride):
        if sid % 500 == 0:
            print(f"[{seq}] scan {sid}/{len(loader)} kf={len(S)}", flush=True)
        item = loader[sid]
        selected, kf, _ = selector.process_scan(scan_id=sid, points=item["points"], pose=item["pose"], timestamp=item["timestamp"])
        if not selected:
            continue
        D.append(encoder.encode_points(item["points"]).detach().cpu().numpy().astype(np.float32))
        FF.append(encoder.compute_fft_magnitudes(item["points"]).astype(np.float32))
        NL.append(_project_nsd_layout(encoder, item["points"], n_layout_sectors=layout_sectors))
        P.append(kf.pose.astype(np.float64)); T.append(float(kf.timestamp))
        S.append(int(sid)); K.append(int(kf.keyframe_id))
    np.savez_compressed(out,
        descriptors=np.asarray(D, dtype=np.float32), poses=np.asarray(P, dtype=np.float64),
        timestamps=np.asarray(T, dtype=np.float64), scan_ids=np.asarray(S, dtype=np.int64),
        keyframe_ids=np.asarray(K, dtype=np.int64), fft_magnitudes=np.asarray(FF, dtype=np.float32),
        nsd_layouts=np.asarray(NL, dtype=np.float32))
    print(f"WROTE {out}  keyframes={len(S)}  desc_shape={np.asarray(D).shape}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_multi_dataset.yaml")
    ap.add_argument("--root", default="/rise/RISE1/workspace/data/mulran")
    ap.add_argument("--sequences", nargs="+", default=["DCC03"])
    ap.add_argument("--cache-dir", default="data/preprocessed_mulran_nointerdiff")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layout-sectors", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    cfg = _apply_encoder_preset(_load_config(a.config), "no_interdiff")
    cfg["encoding"]["elevation_range"] = [-16.6, 16.6]   # Ouster OS1-64
    for s in a.sequences:
        build(a.root, s, cfg, a.cache_dir, a.device, a.layout_sectors, a.stride)
