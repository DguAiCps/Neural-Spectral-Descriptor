#!/usr/bin/env python3
"""B3 rerank on the pd-protocol NEW VAL sequences (place-disjoint retrain).

Same pipeline as scripts/run_b3_rerank.py — pass-1 embeddings on the paper
graph -> top-30 cosine candidates per node -> edge classifier -> top-10
similarity edges injected on a temporal-only graph -> forward -> phase-sketch
rerank grid — but wired to the pd2026 protocol artifacts:

  --variant pd    checkpoints/pd_s{1,2,3}   + artifacts/key_remetrize_r_pd2026.npy
                  + artifacts/edge_classifier_pd2026.pt
                  val = kitti 09, nclt 2012-01-08, helipr Roundabout03,
                        mulran Sejong03
  --variant pdnn  checkpoints/pdnn_s{1,2,3} + *_pdnn2026 artifacts, val minus nclt

Node descriptors/poses come from the pd2026 val caches; nsd/bev layouts from
data/preprocessed_pd2026/vallayouts_*.npz (scripts/_pd_build_val_layouts.py).
NEW VAL sequences only — never test. Never touches the frozen paper-main
artifacts or results. Results to results/pd2026/b3_rerank_<variant>_<seed>.json.
"""
from __future__ import annotations
import argparse, copy, glob, json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import (_apply_encoder_preset, _make_model, _build_eval_graph,
    _cache_to_keyframes, _recall_with_phase_sketch_fusion, _pool_rows)
from run_kitti_operating_point import _find_queries, _topk_cosine, _score, _normalize
from build_and_train_edge_classifier import edge_features, range_phase_flat, EdgeMLP, N_FEAT

PD_CACHE = REPO / "data/preprocessed_pd2026"
OUT = REPO / "results/pd2026"
DTH, SKIP, KEEP_K = 5.0, 30, 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PD_VAL_SEQS = {
    "kitti": ["09"], "nclt": ["2012-01-08"],
    "helipr": ["Roundabout03"], "mulran": ["Sejong03"],
}
VARIANTS = {
    "pd": dict(
        ckpt_dir="pd_{seed}",
        rvec=REPO / "artifacts/key_remetrize_r_pd2026.npy",
        cls=REPO / "artifacts/edge_classifier_pd2026.pt",
        config=REPO / "configs/training_place_disjoint.yaml",
        seqs=PD_VAL_SEQS,
    ),
    "pdnn": dict(
        ckpt_dir="pdnn_{seed}",
        rvec=REPO / "artifacts/key_remetrize_r_pdnn2026.npy",
        cls=REPO / "artifacts/edge_classifier_pdnn2026.pt",
        config=REPO / "configs/training_place_disjoint_no_nclt.yaml",
        seqs={k: v for k, v in PD_VAL_SEQS.items() if k != "nclt"},
    ),
}


def _pd_val_cache(sensor: str, seq: str) -> np.lib.npyio.NpzFile:
    hits = sorted(glob.glob(str(PD_CACHE / f"cache_*_{sensor}_val_{seq}.npz")))
    assert len(hits) == 1, (sensor, seq, hits)
    return np.load(hits[0])


def r1(emb, poses):
    normed = _normalize(emb)
    q = _find_queries(poses, DTH, SKIP)
    ranked = [_topk_cosine(normed, i, 1 + 2 * SKIP, SKIP)[:1] for i, _ in q]
    return _score(poses, q, ranked, [1], DTH)["R@1"]


def run_seed(variant: str, seed: str):
    V = VARIANTS[variant]
    ckpt = REPO / "checkpoints" / V["ckpt_dir"].format(seed=seed) / "best_model.pth"
    cfg = _apply_encoder_preset(yaml.safe_load(open(V["config"])), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(V["rvec"])}
    r = np.load(V["rvec"]).astype(np.float32)
    model = _make_model(rm_cfg, ckpt, DEVICE)

    blob = torch.load(V["cls"], map_location="cpu", weights_only=False)
    clf = EdgeMLP(N_FEAT)
    clf.load_state_dict(blob["state_dict"])
    clf.eval().to(DEVICE)
    mu, sd = blob["mu"].astype(np.float32), blob["sd"].astype(np.float32)

    report = {}
    for sensor, seqs in V["seqs"].items():
        for seq in seqs:
            name = f"{sensor}/{seq}"
            cache = _pd_val_cache(sensor, seq)
            lay = np.load(PD_CACHE / f"vallayouts_{sensor}_{seq}.npz")
            assert (lay["keyframe_ids"] == cache["keyframe_ids"]).all(), name
            nsd_layouts = lay["nsd_layouts"].astype(np.float32)
            d_raw = cache["descriptors"].astype(np.float32)
            poses = cache["poses"]
            graph = _build_eval_graph(
                keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
                cache=cache, config=rm_cfg, device=DEVICE, temporal_edge_mode="bidirectional",
                temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
                sensor_key=sensor)
            with torch.no_grad():
                embA = model(graph.to(DEVICE)).cpu().numpy()
            coarse_pass1 = r1(embA, poses)
            d_rm = _normalize(d_raw * r[None, :])
            x_phase = range_phase_flat(nsd_layouts)
            src, dst, feats, _ = edge_features(embA, d_rm, x_phase, poses, want_labels=False)
            with torch.no_grad():
                logits = clf(torch.from_numpy((feats - mu) / sd).float().to(DEVICE))
                probs = torch.sigmoid(logits).cpu().numpy()

            # per-node top-K by classifier probability
            n = len(embA)
            probs2 = probs.reshape(n, -1)
            pick = np.argsort(-probs2, axis=1)[:, :KEEP_K]
            src2 = np.repeat(np.arange(n), KEEP_K)
            dst2 = dst.reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
            cos2 = feats[:, 0].reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
            prob2 = probs2[np.arange(n)[:, None], pick].reshape(-1)
            l2_2 = np.linalg.norm(embA[src2] - embA[dst2], axis=1)

            # temporal-only base graph, then inject classifier edges (attr layout
            # mirrors the builder: [0, 0, cos, log1p(l2)/5, posterior])
            cfg_t = copy.deepcopy(rm_cfg)
            cfg_t["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.0
            g = _build_eval_graph(
                keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
                cache=cache, config=cfg_t, device=DEVICE, temporal_edge_mode="bidirectional",
                temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
                sensor_key=sensor)
            dev = g.edge_index.device
            ei_new = torch.from_numpy(np.stack([src2, dst2])).long().to(dev)
            attr_new = torch.from_numpy(np.stack([
                np.zeros_like(cos2), np.zeros_like(cos2), cos2,
                np.log1p(l2_2) / 5.0, prob2], axis=1).astype(np.float32)).to(dev)
            g.edge_index = torch.cat([g.edge_index, ei_new], dim=1)
            g.edge_attr = torch.cat([g.edge_attr, attr_new], dim=0)
            g.edge_type = torch.cat(
                [g.edge_type, torch.ones(ei_new.shape[1], dtype=torch.long, device=dev)])

            with torch.no_grad():
                embB3 = model(g.to(DEVICE)).cpu().numpy()
            coarse = r1(embB3, poses)
            bev = _pool_rows(lay["bev_layouts"].astype(np.float32), 16, "max")
            res = _recall_with_phase_sketch_fusion(
                embeddings=embB3, range_layouts=nsd_layouts,
                bev_layouts=bev, poses=poses, k_values=[1], distance_threshold=DTH,
                skip_frames=SKIP, n_coarse=800, range_freqs=4, bev_freqs=8, n_sectors=60,
                sketch_bev_weights=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
                sketch_range_weights=[0.0, 0.5, 1.0, 2.0, 4.0],
                rerank_mode="sketch_fft", fusion_norm="raw")
            grid = {k: v["R@1"] for k, v in res.items() if not k.startswith("_")}
            report[name] = {"coarse_pass1": coarse_pass1, "coarse": coarse,
                            "grid": grid, "best": max(max(grid.values()), coarse)}
            print(f"[b3-{variant}-{seed}] {name:<22} pass1={coarse_pass1:.3f} "
                  f"coarse={coarse:.3f} best={report[name]['best']:.3f}", flush=True)

    macro = {
        "coarse_pass1": float(np.mean([v["coarse_pass1"] for v in report.values()])),
        "coarse": float(np.mean([v["coarse"] for v in report.values()])),
        "best": float(np.mean([v["best"] for v in report.values()])),
    }
    report["_macro"] = macro
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"b3_rerank_{variant}_{seed}.json"
    json.dump(report, open(out_path, "w"), indent=1)
    print(f"[b3-{variant}-{seed}] macro coarse={macro['coarse']:.3f} "
          f"best={macro['best']:.3f} -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--seeds", nargs="+", default=["s1", "s2", "s3"])
    args = ap.parse_args()
    for seed in args.seeds:
        run_seed(args.variant, seed)


if __name__ == "__main__":
    main()
