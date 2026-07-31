#!/usr/bin/env python3
"""Step-1b: does the closed-form d re-metrization survive in the deployed 416D key?

Refits the metrics on TRAIN (same as _metric_probe.py), then for each VAL seq and
each of the 3 main checkpoints, builds the 416D key = [normalize(metric(d)) ; context]
(context taken from the actual model forward) and evaluates coarse Recall@1/R_min.
Decides whether re-metrizing the frozen d channel helps the actual retrieval key.
"""
from __future__ import annotations
import glob, json, sys
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
VAL = {"kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
       "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"]}
DTH, SKIP = 5.0, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QC = {"kitti/00": 632, "kitti/05": 377, "kitti/08": 235, "nclt/2012-01-08": 1834,
      "nclt/2013-01-10": 181, "helipr/Town01": 1586, "mulran/DCC03": 2344,
      "mulran/KAIST03": 2909, "mulran/Riverside03": 2796}


def norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def fit_metrics():
    valset = {(s, q) for s, qs in VAL.items() for q in qs}
    deltas, alld = [], []
    for f in sorted(glob.glob(str(CACHE / "*_operating_*_layout60_stride1.npz"))):
        base = Path(f).name.replace("_operating_", "|").replace("_layout60_stride1.npz", "")
        sensor, seq = base.split("|")
        if (sensor, seq) in valset:
            continue
        z = np.load(f); d = z["descriptors"].astype(np.float64); pos = z["poses"][:, :3, 3].astype(np.float64)
        alld.append(d); n = len(d)
        for q in range(SKIP, n):
            prior = np.arange(0, q - SKIP)
            if prior.size == 0:
                continue
            geo = np.linalg.norm(pos[prior] - pos[q], axis=1)
            p = prior[geo < DTH]
            if p.size == 0:
                continue
            m = p[np.argmin(np.linalg.norm(pos[p] - pos[q], axis=1))]
            deltas.append(d[q] - d[m])
    deltas = np.array(deltas); alld = np.concatenate(alld, 0)
    Wd = 0.5 * np.mean(deltas ** 2, 0); eps = 1e-6 * Wd.mean()
    r_wccn = 1.0 / np.sqrt(Wd + eps); r_wccn /= r_wccn.mean()
    Sw = 0.5 * (deltas.T @ deltas) / len(deltas)
    Sw = 0.9 * Sw + 0.1 * np.eye(Sw.shape[0]) * np.trace(Sw) / Sw.shape[0]
    ev, evec = np.linalg.eigh(Sw); ev = np.maximum(ev, 1e-8)
    L = evec @ np.diag(ev ** -0.5) @ evec.T
    return {"r_wccn": r_wccn, "L": L, "mu": alld.mean(0)}, len(deltas)


def metric_d(d, name, M):
    if name == "baseline":
        return d
    if name == "diag_wccn":
        return d * M["r_wccn"]
    if name == "full_wccn":
        return (d - M["mu"]) @ M["L"].T
    raise ValueError(name)


def r1(key, pos):
    r = norm(key); n = len(r); correct = 0; nq = 0
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
            correct += 1
    return correct / max(nq, 1)


def summarize(per_seq):
    ks = list(per_seq)
    rq = sum(per_seq[k] * QC[k] for k in ks) / sum(QC[k] for k in ks)
    rs = float(np.mean([per_seq[k] for k in ks]))
    sens = {}
    for k in ks:
        sens.setdefault(k.split("/")[0], []).append(per_seq[k])
    macro = {s: float(np.mean(v)) for s, v in sens.items()}
    mbar = np.mean(list(macro.values()))
    sig = float(np.sqrt(np.sum([(macro[s] - mbar) ** 2 for s in macro]) / (len(macro) - 1)))
    return {"R^q": round(rq, 4), "R^s": round(rs, 4), "sigma_cross": round(sig, 4),
            "R_min": round(min(macro.values()), 4), "macro": {s: round(v, 4) for s, v in macro.items()}}


def main():
    M, npairs = fit_metrics()
    print(f"fit on {npairs} train pairs", flush=True)
    cfg = _apply_encoder_preset(yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True; cfg["gnn"]["gate_initial_alpha"] = 0.0625
    models = {s: _make_model(cfg, c, DEVICE) for s, c in CKPTS.items() if c.exists()}

    # cache context per (seed, seq)
    variants = ["baseline", "diag_wccn", "full_wccn"]
    acc = {v: {k: [] for k in QC} for v in variants}
    for s, model in models.items():
        for sensor, seqs in VAL.items():
            for seq in seqs:
                z = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
                d = z["descriptors"].astype(np.float64); pos = z["poses"][:, :3, 3].astype(np.float64)
                graph = _build_eval_graph(keyframes=_cache_to_keyframes(z), poses=z["poses"], descriptors=z["descriptors"].astype(np.float32),
                    cache=z, config=cfg, device=DEVICE, temporal_edge_mode="bidirectional",
                    temporal_direction_mode="none", similarity_min_k=0, phase_features=None, sensor_key=sensor)
                with torch.no_grad():
                    emb = model(graph.to(DEVICE)).detach().cpu().numpy().astype(np.float64)
                ctx = emb[:, 288:416]  # already g*normalize(c)
                for v in variants:
                    dm = norm(metric_d(d, v, M))
                    key = np.concatenate([dm, ctx], axis=1)
                    acc[v][f"{sensor}/{seq}"].append(r1(key, pos))
    report = {}
    print("\n=== 416D key (context) coarse Recall@1, 3-seed mean ===")
    for v in variants:
        per_seq = {k: float(np.mean(acc[v][k])) for k in QC}
        summ = summarize(per_seq)
        report[v] = {"per_seq": {k: round(x, 4) for k, x in per_seq.items()}, "summary": summ}
        print(f"\n[{v}] R^q={summ['R^q']} R^s={summ['R^s']} sigma={summ['sigma_cross']} R_min={summ['R_min']}")
        print("  macro: " + " ".join(f"{s}={x}" for s, x in summ["macro"].items()))
        for k in QC:
            print(f"    {k:<20} {per_seq[k]:.3f}")
    out = Path("results/metric_probe_416.json")
    out.write_text(json.dumps(report, indent=2))
    print("\nWROTE", out)


if __name__ == "__main__":
    main()
