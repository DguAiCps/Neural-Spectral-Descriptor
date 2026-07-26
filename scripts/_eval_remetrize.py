#!/usr/bin/env python3
"""Evaluate the learned re-metrization head checkpoint vs baseline / fixed diag_wccn.

Loads the step-2 checkpoint WITH the key_remetrize head enabled (so the trained
key_scale loads), evaluates the deployed 416D key coarse Recall@1 on the 9 VAL
sequences with the SAME protocol as _metric_probe_416.py, and prints the
comparison against the known baseline (frozen) and fixed diag_wccn numbers.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
CKPT = REPO / "checkpoints/remetrize_s1_20260706/best_model.pth"
VAL = {"kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
       "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"]}
DTH, SKIP = 5.0, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QC = {"kitti/00": 632, "kitti/05": 377, "kitti/08": 235, "nclt/2012-01-08": 1834,
      "nclt/2013-01-10": 181, "helipr/Town01": 1586, "mulran/DCC03": 2344,
      "mulran/KAIST03": 2909, "mulran/Riverside03": 2796}
# reference numbers from _metric_probe_416.py (same protocol)
REF = {"baseline": {"R^q": 0.7248, "R_min": 0.3669, "sigma": 0.2264},
       "fixed_diag_wccn": {"R^q": 0.7376, "R_min": 0.4011, "sigma": 0.2164}}


def norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def r1(key, pos):
    r = norm(key.astype(np.float64)); n = len(r); c = nq = 0
    for q in range(n):
        idx = np.where(np.abs(np.arange(n) - q) >= SKIP)[0]
        if idx.size == 0:
            continue
        geo = np.linalg.norm(pos[idx] - pos[q], axis=1)
        if not np.any(geo < DTH):
            continue
        nq += 1
        nn = idx[np.linalg.norm(r[idx] - r[q], axis=1).argmin()]
        if np.linalg.norm(pos[nn] - pos[q]) < DTH:
            c += 1
    return c / max(nq, 1)


def summarize(ps):
    ks = list(ps)
    rq = sum(ps[k] * QC[k] for k in ks) / sum(QC[k] for k in ks)
    rs = float(np.mean([ps[k] for k in ks]))
    sens = {}
    for k in ks:
        sens.setdefault(k.split("/")[0], []).append(ps[k])
    macro = {s: float(np.mean(v)) for s, v in sens.items()}
    mbar = np.mean(list(macro.values()))
    sig = float(np.sqrt(np.sum([(macro[s] - mbar) ** 2 for s in macro]) / (len(macro) - 1)))
    return rq, rs, sig, min(macro.values()), macro


def main():
    cfg = _apply_encoder_preset(yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": "artifacts/key_remetrize_r.npy"}
    model = _make_model(cfg, CKPT, DEVICE)

    # diagnostic: how far did the learned key_scale drift from the warm-start?
    core = model.gnn if hasattr(model, "gnn") else model
    ks = core.key_scale.detach().cpu().numpy()
    r0 = np.load(REPO / "artifacts/key_remetrize_r.npy")
    corr = float(np.corrcoef(ks, r0)[0, 1])
    print(f"key_scale: learned vs warm-start corr={corr:.4f}, "
          f"||learned-init||/||init||={np.linalg.norm(ks-r0)/np.linalg.norm(r0):.4f}, "
          f"learned min/max={ks.min():.3f}/{ks.max():.3f}", flush=True)

    ps = {}
    for sensor, seqs in VAL.items():
        for seq in seqs:
            z = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
            pos = z["poses"][:, :3, 3].astype(np.float64)
            graph = _build_eval_graph(keyframes=_cache_to_keyframes(z), poses=z["poses"],
                descriptors=z["descriptors"].astype(np.float32), cache=z, config=cfg, device=DEVICE,
                temporal_edge_mode="bidirectional", temporal_direction_mode="none",
                similarity_min_k=0, phase_features=None, sensor_key=sensor)
            with torch.no_grad():
                emb = model(graph.to(DEVICE)).detach().cpu().numpy()
            ps[f"{sensor}/{seq}"] = r1(emb, pos)
            print(f"  {sensor}/{seq:<12} {ps[f'{sensor}/{seq}']:.3f}", flush=True)

    rq, rs, sig, rmin, macro = summarize(ps)
    print("\n=== 416D key coarse Recall@1 (learned re-metrization head, best_model epoch69) ===")
    print(f"  R^q={rq:.4f} R^s={rs:.4f} sigma_cross={sig:.4f} R_min={rmin:.4f}")
    print(f"  macro: " + " ".join(f"{s}={v:.3f}" for s, v in macro.items()))
    print("\n=== COMPARISON (same protocol) ===")
    print(f"{'variant':<22}{'R^q':>8}{'R_min':>8}{'sigma':>8}")
    print(f"{'baseline (frozen)':<22}{REF['baseline']['R^q']:>8.4f}{REF['baseline']['R_min']:>8.4f}{REF['baseline']['sigma']:>8.4f}")
    print(f"{'fixed diag_wccn':<22}{REF['fixed_diag_wccn']['R^q']:>8.4f}{REF['fixed_diag_wccn']['R_min']:>8.4f}{REF['fixed_diag_wccn']['sigma']:>8.4f}")
    print(f"{'learned head':<22}{rq:>8.4f}{rmin:>8.4f}{sig:>8.4f}")
    print(f"\nlearned vs baseline:      R^q {rq-REF['baseline']['R^q']:+.4f}  R_min {rmin-REF['baseline']['R_min']:+.4f}")
    print(f"learned vs fixed diag_wccn: R^q {rq-REF['fixed_diag_wccn']['R^q']:+.4f}  R_min {rmin-REF['fixed_diag_wccn']['R_min']:+.4f}")


if __name__ == "__main__":
    main()
