#!/usr/bin/env python3
"""A+B evaluation: closed-form diagonal re-metrization + edge rebuild (two-pass).

Variants (all sharing the trained 800D checkpoints; no retraining):
  P   baseline        paper graph (raw-d edges, threshold 0.993), frozen key
  A   re-metrize      paper graph, key block re-metrized by the fixed diag_wccn
                      vector (gnn.key_remetrize enabled; ckpt lacks key_scale so
                      the init vector stays fixed — closed-form)
  B0  density control top-10 similarity edges from raw d, re-metrized key
  B1  edges@d_rm      top-10 similarity edges from the re-metrized key d_rm
  B2  two-pass        top-10 similarity edges from the pass-1 416D key (A output)

B0 isolates the effect of switching threshold->top-k densification from the
effect of a better edge key (B1/B2). Node features stay raw d everywhere; only
edge construction and the final key assembly change.
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes
from run_kitti_operating_point import _find_queries, _topk_cosine, _score, _normalize

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
RVEC = REPO / "artifacts/key_remetrize_r.npy"
OUT = REPO / "results/_remetrize_twopass"
CKPTS = {
    "seed1": REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    "seed2": REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    "seed3": REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
    "s5": REPO / "checkpoints/remetrize_twopass_s1/best_model.pth",
}
SENSORS = {
    "kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
DTH, SKIP = 5.0, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK_EDGE_THRESHOLD = -1.0   # accept-all; density controlled by similarity_max_k


def r1(emb, poses):
    normed = _normalize(emb)
    q = _find_queries(poses, DTH, SKIP)
    ranked = [_topk_cosine(normed, i, 1 + 2 * SKIP, SKIP)[:1] for i, _ in q]
    return _score(poses, q, ranked, [1], DTH)["R@1"]


def build(cfg, cache, poses, edge_desc, sensor, threshold=None):
    cfg_g = copy.deepcopy(cfg)
    if threshold is not None:
        cfg_g["keyframe"].setdefault("graph", {})["similarity_threshold"] = threshold
    return _build_eval_graph(
        keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=edge_desc,
        cache=cache, config=cfg_g, device=DEVICE, temporal_edge_mode="bidirectional",
        temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
        sensor_key=sensor)


def aggregates(per_seq, counts):
    seqs = list(per_seq.keys())
    vals = np.array([per_seq[s] for s in seqs])
    n = np.array([counts[s] for s in seqs], dtype=float)
    macro = {}
    for s, v in per_seq.items():
        macro.setdefault(s.split("/")[0], []).append(v)
    macros = np.array([np.mean(v) for v in macro.values()])
    return {
        "R^q": float((vals * n).sum() / n.sum()),
        "R^s": float(vals.mean()),
        "sigma_cross": float(np.sqrt(((macros - macros.mean()) ** 2).mean())),
        "R_min": float(macros.min()),
    }


def main(seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    base_cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    base_cfg["gnn"]["use_residual_gate"] = True
    base_cfg["gnn"]["gate_initial_alpha"] = 0.0625

    rm_cfg = copy.deepcopy(base_cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(RVEC)}
    r = np.load(RVEC).astype(np.float32)

    for seed in seeds:
        ckpt = CKPTS[seed]
        model_base = _make_model(base_cfg, ckpt, DEVICE)
        model_rm = _make_model(rm_cfg, ckpt, DEVICE)
        ks = model_rm.gnn.key_scale if hasattr(model_rm, "gnn") else model_rm.key_scale
        assert ks is not None and np.allclose(ks.detach().cpu().numpy(), r, atol=1e-6), \
            "key_scale must stay at the fixed diag_wccn init (closed-form mode)"

        per = {v: {} for v in ("P", "A", "B0", "B1", "B2")}
        counts = {}
        for sensor, seqs in SENSORS.items():
            for seq in seqs:
                name = f"{sensor}/{seq}"
                cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
                d_raw = cache["descriptors"].astype(np.float32)
                poses = cache["poses"]
                counts[name] = len(_find_queries(poses, DTH, SKIP))
                d_rm = _normalize(d_raw * r[None, :])

                g_paper = build(base_cfg, cache, poses, d_raw, sensor)
                with torch.no_grad():
                    per["P"][name] = r1(model_base(g_paper.to(DEVICE)).cpu().numpy(), poses)
                    embA = model_rm(g_paper.to(DEVICE)).cpu().numpy()
                per["A"][name] = r1(embA, poses)

                for variant, edge_desc in (("B0", d_raw), ("B1", d_rm), ("B2", embA)):
                    g = build(rm_cfg, cache, poses, edge_desc, sensor,
                              threshold=TOPK_EDGE_THRESHOLD)
                    with torch.no_grad():
                        per[variant][name] = r1(model_rm(g.to(DEVICE)).cpu().numpy(), poses)
                print(f"[{seed}] {name:<20} " + "  ".join(
                    f"{v}={per[v][name]:.3f}" for v in ("P", "A", "B0", "B1", "B2")), flush=True)

        report = {v: {"per_seq": per[v], **aggregates(per[v], counts)} for v in per}
        out = OUT / f"{seed}.json"
        json.dump(report, open(out, "w"), indent=1)
        print(f"[{seed}] " + " | ".join(
            f"{v}: R^q={report[v]['R^q']:.4f} R_min={report[v]['R_min']:.4f} "
            f"sig={report[v]['sigma_cross']:.4f}" for v in per), flush=True)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["seed1"])
