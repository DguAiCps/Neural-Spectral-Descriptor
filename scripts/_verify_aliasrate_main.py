#!/usr/bin/env python3
"""AliasRate at the MAIN operating point: 288D magnitude key vs 416D retrieval key.

Same definition as scripts/compute_aliasrate.py (eps=0.1, gamma=25m, 2D-xy far pairs,
200k sampled pairs). Reports per-sequence and per-sensor, 3-seed mean for the 416D key.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
CKPTS = {
    "seed1": REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    "seed2": REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    "seed3": REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
}
SENSORS = {"kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
           "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"]}
EPS, GAMMA, NPAIRS = 0.1, 25.0, 200_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def aliasrate(desc, poses, rng):
    d = desc.astype(np.float64)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    xy = poses[:, :2, 3]
    n = d.shape[0]
    i = rng.integers(0, n, NPAIRS); j = rng.integers(0, n, NPAIRS)
    keep = i != j; i, j = i[keep], j[keep]
    geo = np.linalg.norm(xy[i] - xy[j], axis=1)
    dd = np.linalg.norm(d[i] - d[j], axis=1)
    far = geo >= GAMMA
    coll = (dd <= EPS) & far
    return 100.0 * coll.sum() / max(int(far.sum()), 1)


def main():
    cfg = _apply_encoder_preset(yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True; cfg["gnn"]["gate_initial_alpha"] = 0.0625

    # raw (288D) is seed-independent (closed-form encoder) -> compute once
    raw_ar, fin_ar = {}, {}
    models = {s: _make_model(cfg, c, DEVICE) for s, c in CKPTS.items() if c.exists()}
    for sensor, seqs in SENSORS.items():
        for seq in seqs:
            cpath = CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz"
            if not cpath.exists():
                print("MISS", cpath); continue
            cache = np.load(cpath); desc = cache["descriptors"].astype(np.float32); poses = cache["poses"]
            rng0 = np.random.default_rng(0)
            raw_ar[f"{sensor}/{seq}"] = aliasrate(desc, poses, rng0)
            fins = []
            for s, model in models.items():
                graph = _build_eval_graph(
                    keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=desc, cache=cache,
                    config=cfg, device=DEVICE, temporal_edge_mode="bidirectional",
                    temporal_direction_mode="none", similarity_min_k=0, phase_features=None, sensor_key=sensor)
                with torch.no_grad():
                    emb = model(graph.to(DEVICE)).detach().cpu().numpy()
                fins.append(aliasrate(emb, poses, np.random.default_rng(0)))
            fin_ar[f"{sensor}/{seq}"] = float(np.mean(fins))
            print(f"{sensor}/{seq:<12} raw288={raw_ar[f'{sensor}/{seq}']:.2f}%  "
                  f"final416={fin_ar[f'{sensor}/{seq}']:.2f}%", flush=True)

    # per-sensor macro
    def macro(d):
        out = {}
        for sensor, seqs in SENSORS.items():
            vals = [d[f"{sensor}/{s}"] for s in seqs if f"{sensor}/{s}" in d]
            out[sensor] = float(np.mean(vals)) if vals else None
        return out
    report = {"per_seq_raw288": raw_ar, "per_seq_final416": fin_ar,
              "per_sensor_raw288": macro(raw_ar), "per_sensor_final416": macro(fin_ar)}
    (REPO / "results/_verify_aliasrate_main.json").write_text(json.dumps(report, indent=2))
    print("\n=== per-sensor macro AliasRate (%) ===")
    for s in SENSORS:
        print(f"{s:<8} raw288={report['per_sensor_raw288'][s]:.2f}  final416={report['per_sensor_final416'][s]:.2f}")
    print("WROTE results/_verify_aliasrate_main.json")


if __name__ == "__main__":
    main()
