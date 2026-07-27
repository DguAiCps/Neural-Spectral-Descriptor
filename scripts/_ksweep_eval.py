#!/usr/bin/env python3
"""Table 33 (tab:ksweep): candidate-pool sensitivity at fixed deployed weights.

Per seed and validation sequence, loads the final B3 416D embeddings (dumped
once per seed into results/_final416_{seed}/), rebuilds the three top-800
candidate channels, computes per-candidate distances ONCE on the K=800 union,
and derives every smaller K by subset masking (exact: _topk_cosine returns
candidates sorted by descending cosine).

Deployed fusion weights, no per-K recalibration (PAPER_MAIN_B3): KITTI fused
score = d_emb + 0.5*d_bev; NCLT/HeLiPR/MulRan use d_emb only (range weight 0
everywhere, so range-sketch distances are never computed).

Rows: K in {50,100,200,400,800} plus 'primary' (416D single-channel pool,
K=800). Metrics per row: R@1, union-size median/p95 (as fraction of map too),
GT-in-pool share.

Verification: at K=800, KITTI R@1 must equal b3_rerank_{seed}.json
grid['phase_sketch_fusion_bev0.5_range0'] and non-KITTI bev0_range0.
--verify-direct additionally recomputes kitti/05 at K=200 from scratch and
compares with the masked derivation.

Run inside the pyg container (dump step needs GPU):
  python scripts/_ksweep_eval.py --seed seed1 seed2 seed3
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import (_phase_sketch, _phase_sketch_keys,
    _phase_sketch_distances_fft, _pool_rows)
from run_kitti_operating_point import _find_queries, _topk_cosine, _score, _normalize

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
OUT = REPO / "results/_remetrize_twopass"
SENSORS = {
    "kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
BEV_DIRS = {"kitti": "preprocessed_kitti_bev_layout", "nclt": "preprocessed_nclt_bev_layout",
            "helipr": "preprocessed_helipr_bev_layout", "mulran": "preprocessed_mulran_bev_layout"}
DTH, SKIP = 5.0, 30
KS = [50, 100, 200, 400, 800]
W_BEV = {"kitti": 0.5, "nclt": 0.0, "helipr": 0.0, "mulran": 0.0}


def ensure_dump(seed):
    """Dump final B3 416D embeddings for a seed (mirrors run_b3_rerank.py)."""
    dump_dir = REPO / f"results/_final416_{seed}"
    missing = [(s, q) for s, qs in SENSORS.items() for q in qs
               if not (dump_dir / f"{s}_{q}.npz").exists()]
    if not missing:
        return
    import torch, yaml
    from evaluate_kitti_checkpoint import (_apply_encoder_preset, _make_model,
        _build_eval_graph, _cache_to_keyframes)
    from build_and_train_edge_classifier import edge_features, range_phase_flat, EdgeMLP, N_FEAT
    from run_b3_rerank import CKPTS, KEEP_K
    RVEC = REPO / "artifacts/key_remetrize_r.npy"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(RVEC)}
    r = np.load(RVEC).astype(np.float32)
    model = _make_model(rm_cfg, CKPTS[seed], device)
    blob = torch.load(REPO / "artifacts/edge_classifier.pt", map_location="cpu", weights_only=False)
    clf = EdgeMLP(N_FEAT); clf.load_state_dict(blob["state_dict"]); clf.eval().to(device)
    mu, sd = blob["mu"].astype(np.float32), blob["sd"].astype(np.float32)
    dump_dir.mkdir(parents=True, exist_ok=True)

    for sensor, seq in missing:
        cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
        d_raw = cache["descriptors"].astype(np.float32)
        poses = cache["poses"]
        graph = _build_eval_graph(
            keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
            cache=cache, config=rm_cfg, device=device, temporal_edge_mode="bidirectional",
            temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
            sensor_key=sensor)
        with torch.no_grad():
            embA = model(graph.to(device)).cpu().numpy()
        d_rm = _normalize(d_raw * r[None, :])
        x_phase = range_phase_flat(cache["nsd_layouts"])
        src, dst, feats, _ = edge_features(embA, d_rm, x_phase, poses, want_labels=False)
        with torch.no_grad():
            logits = clf(torch.from_numpy((feats - mu) / sd).float().to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
        n = len(embA)
        probs2 = probs.reshape(n, -1)
        pick = np.argsort(-probs2, axis=1)[:, :KEEP_K]
        src2 = np.repeat(np.arange(n), KEEP_K)
        dst2 = dst.reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
        cos2 = feats[:, 0].reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
        prob2 = probs2[np.arange(n)[:, None], pick].reshape(-1)
        l2_2 = np.linalg.norm(embA[src2] - embA[dst2], axis=1)
        cfg_t = copy.deepcopy(rm_cfg)
        cfg_t["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.0
        g = _build_eval_graph(
            keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
            cache=cache, config=cfg_t, device=device, temporal_edge_mode="bidirectional",
            temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
            sensor_key=sensor)
        import torch as _t
        dev = g.edge_index.device
        ei_new = _t.from_numpy(np.stack([src2, dst2])).long().to(dev)
        attr_new = _t.from_numpy(np.stack([
            np.zeros_like(cos2), np.zeros_like(cos2), cos2,
            np.log1p(l2_2) / 5.0, prob2], axis=1).astype(np.float32)).to(dev)
        g.edge_index = _t.cat([g.edge_index, ei_new], dim=1)
        g.edge_attr = _t.cat([g.edge_attr, attr_new], dim=0)
        g.edge_type = _t.cat([g.edge_type, _t.ones(ei_new.shape[1], dtype=_t.long, device=dev)])
        with torch.no_grad():
            embB3 = model(g.to(device)).cpu().numpy()
        np.savez(dump_dir / f"{sensor}_{seq}.npz", emb416=embB3, poses=poses)
        print(f"[dump/{seed}] {sensor}/{seq} n={n}", flush=True)


def sweep_seq(sensor, seq, emb416, poses, verify_direct=False):
    cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
    stride = "_stride1" if sensor == "nclt" else ""
    bev = _pool_rows(np.load(REPO / "data" / BEV_DIRS[sensor] /
        f"{sensor}_bev_layout_{seq}_s60_max_r1-80_z-3-5_h8{stride}.npz")["bev_layouts"].astype(np.float32), 16, "max")
    normed = _normalize(emb416)
    range_sketch = _phase_sketch(cache["nsd_layouts"].astype(np.float32), 4)
    bev_sketch = _phase_sketch(bev, 8)
    range_keys = _phase_sketch_keys(range_sketch)
    bev_keys = _phase_sketch_keys(bev_sketch)
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    wb = W_BEV[sensor]
    n_map = len(poses)

    rows = {k: {"ranked": [], "union": [], "gt": []} for k in KS + ["primary"]}
    for qi, _ in queries:
        ce = _topk_cosine(normed, qi, 800, SKIP)
        cr = _topk_cosine(range_keys, qi, 800, SKIP)
        cb = _topk_cosine(bev_keys, qi, 800, SKIP)
        u = np.unique(np.concatenate([ce, cr, cb]))
        emb_d = 1.0 - (normed[u] @ normed[qi])
        bev_d = _phase_sketch_distances_fft(bev_sketch[qi], bev_sketch[u], n_sectors=60) \
            if wb > 0 else np.zeros_like(emb_d)
        fused_u = emb_d + wb * bev_d
        geo_u = np.linalg.norm(positions[u] - positions[qi], axis=1)

        for K in KS:
            pool = np.unique(np.concatenate([ce[:K], cr[:K], cb[:K]]))
            sel = np.searchsorted(u, pool)
            top1 = pool[np.argmin(fused_u[sel])]
            rows[K]["ranked"].append(np.array([top1]))
            rows[K]["union"].append(len(pool))
            rows[K]["gt"].append(bool((geo_u[sel] < DTH).any()))
        sel = np.searchsorted(u, np.sort(ce))
        top1 = np.sort(ce)[np.argmin(fused_u[sel])]
        rows["primary"]["ranked"].append(np.array([top1]))
        rows["primary"]["union"].append(len(ce))
        rows["primary"]["gt"].append(bool((geo_u[sel] < DTH).any()))

    out = {}
    for k, d in rows.items():
        sc = _score(poses, queries, d["ranked"], [1], DTH)
        un = np.array(d["union"])
        out[str(k)] = {
            "r1": sc["R@1"], "n_q": len(queries), "n_map": n_map,
            "union_med": float(np.median(un)), "union_p95": float(np.percentile(un, 95)),
            "gt_in_pool": float(np.mean(d["gt"])),
        }

    if verify_direct:
        K = 200
        ranked = []
        for qi, _ in queries:
            ce = _topk_cosine(normed, qi, K, SKIP)
            cr = _topk_cosine(range_keys, qi, K, SKIP)
            cb = _topk_cosine(bev_keys, qi, K, SKIP)
            pool = np.unique(np.concatenate([ce, cr, cb]))
            emb_d = 1.0 - (normed[pool] @ normed[qi])
            bev_d = _phase_sketch_distances_fft(bev_sketch[qi], bev_sketch[pool], n_sectors=60) \
                if wb > 0 else np.zeros_like(emb_d)
            ranked.append(np.array([pool[np.argmin(emb_d + wb * bev_d)]]))
        direct = _score(poses, queries, ranked, [1], DTH)["R@1"]
        assert abs(direct - out["200"]["r1"]) < 1e-12, (direct, out["200"]["r1"])
        print(f"  verify-direct K=200 {sensor}/{seq}: {direct:.4f} == masked OK", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", nargs="+", default=["seed1"])
    ap.add_argument("--verify-direct", action="store_true")
    args = ap.parse_args()
    for seed in args.seed:
        ensure_dump(seed)
        ref = json.load(open(OUT / f"b3_rerank_{seed}.json"))
        report = {}
        for sensor, seqs in SENSORS.items():
            for seq in seqs:
                name = f"{sensor}/{seq}"
                z = np.load(REPO / f"results/_final416_{seed}/{sensor}_{seq}.npz")
                res = sweep_seq(sensor, seq, z["emb416"].astype(np.float32), z["poses"],
                                verify_direct=(args.verify_direct and name == "kitti/05"))
                # K=800 anchor vs committed grid
                key = "phase_sketch_fusion_bev0.5_range0" if sensor == "kitti" \
                    else "phase_sketch_fusion_bev0_range0"
                anchor = ref[name]["grid"][key]
                ok = abs(res["800"]["r1"] - anchor) < 5e-4
                report[name] = res
                print(f"[ksweep/{seed}] {name:<20} " +
                      " ".join(f"K{k}={res[str(k)]['r1']:.3f}" for k in KS) +
                      f" prim={res['primary']['r1']:.3f} anchor{'OK' if ok else 'FAIL'}", flush=True)
        json.dump(report, open(OUT / f"ksweep_{seed}.json", "w"), indent=1)
        print("saved", OUT / f"ksweep_{seed}.json", flush=True)


if __name__ == "__main__":
    main()
