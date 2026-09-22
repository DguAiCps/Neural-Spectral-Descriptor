#!/usr/bin/env python3
"""Stored-spectrum branch fusion on the pd-protocol NEW VAL sequences.

Exactly the paper-main branch protocol -- frozen embedding via the two-pass
classifier-selected-edge pipeline, candidate union of two top-100 cosine
indexes, per-query min-max 4-channel fusion over the paper WEIGHTS grid --
but wired to the place-disjoint retrain artifacts:

  --variant pd    checkpoints/pd_s{1,2,3}   + artifacts/key_remetrize_r_pd2026.npy
                  + artifacts/edge_classifier_pd2026.pt
                  val = kitti 09, nclt 2012-01-08, helipr Roundabout03,
                        mulran Sejong03
  --variant pdnn  checkpoints/pdnn_s{1,2,3} + *_pdnn2026 artifacts, val minus nclt

Both operating points are evaluated per sequence:
  pbev128_main -- NSD    (BEV 16x8 sketch, scripts/_pbev_branch_fusion.variant_inputs)
  hspec12      -- NSD-H  (height-polar 20x12, scripts/_hspec_branch_fusion.hspec_inputs)

Descriptors/poses from data/preprocessed_pd2026/cache_*_val_*.npz, nsd layouts
from vallayouts_*.npz (scripts/_pd_build_val_layouts.py), sketches/height grids
from valspec_*.npz (scripts/_pd_build_val_spec_caches.py). evaluate(), WEIGHTS,
and the variant input builders are imported unchanged from the paper scripts;
only make_model/nsd_embedding are re-wired (variant checkpoint, omega, and edge
classifier instead of the frozen paper artifacts). NEW VAL only -- never test.
Results to results/pd2026/branch_fusion_<variant>_seed<seed>.json.
"""
from __future__ import annotations
import argparse
import copy
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _height_candidate_fusion import (  # noqa: E402
    WEIGHTS, evaluate, keyframes_from,
)
from _pbev_branch_fusion import variant_inputs  # noqa: E402
from _hspec_branch_fusion import hspec_inputs  # noqa: E402
from evaluate_kitti_checkpoint import (  # noqa: E402
    _apply_encoder_preset, _build_eval_graph, _make_model,
)
from run_kitti_operating_point import _normalize  # noqa: E402
from build_and_train_edge_classifier import (  # noqa: E402
    EdgeMLP, N_FEAT, edge_features, range_phase_flat,
)

PD_CACHE = REPO / "data/preprocessed_pd2026"
OUT = REPO / "results/pd2026"

JOBS_PD = {
    "K09": ("kitti", "09"),
    "N12PD": ("nclt", "2012-01-08"),
    "RA3": ("helipr", "Roundabout03"),
    "SJ3": ("mulran", "Sejong03"),
}
VARIANTS = {
    "pd": dict(
        ckpt_dir="pd_s{seed}",
        rvec=REPO / "artifacts/key_remetrize_r_pd2026.npy",
        cls=REPO / "artifacts/edge_classifier_pd2026.pt",
        config=REPO / "configs/training_place_disjoint.yaml",
        tags=["K09", "N12PD", "RA3", "SJ3"],
    ),
    "pdnn": dict(
        ckpt_dir="pdnn_s{seed}",
        rvec=REPO / "artifacts/key_remetrize_r_pdnn2026.npy",
        cls=REPO / "artifacts/edge_classifier_pdnn2026.pt",
        config=REPO / "configs/training_place_disjoint_no_nclt.yaml",
        tags=["K09", "RA3", "SJ3"],
    ),
}


def _pd_val_cache(sensor: str, seq: str) -> np.lib.npyio.NpzFile:
    hits = sorted(glob.glob(str(PD_CACHE / f"cache_*_{sensor}_val_{seq}.npz")))
    assert len(hits) == 1, (sensor, seq, hits)
    return np.load(hits[0])


def make_model_pd(variant: str, seed: int, device: str):
    V = VARIANTS[variant]
    cfg = _apply_encoder_preset(yaml.safe_load(open(V["config"])), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    cfg = copy.deepcopy(cfg)
    cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(V["rvec"])}
    ckpt = REPO / "checkpoints" / V["ckpt_dir"].format(seed=seed) / "best_model.pth"
    model = _make_model(cfg, ckpt, device)
    blob = torch.load(V["cls"], map_location="cpu", weights_only=False)
    clf = EdgeMLP(N_FEAT)
    clf.load_state_dict(blob["state_dict"])
    clf.eval().to(device)
    r = np.load(V["rvec"]).astype(np.float32)
    return cfg, model, clf, blob, r, ckpt


def nsd_embedding_pd(cache, sensor: str, d288: np.ndarray, layouts: np.ndarray,
                     cfg, model, clf, clf_blob, r: np.ndarray, device: str):
    """scripts/_height_candidate_fusion.nsd_embedding with the omega vector
    passed in (variant artifact) instead of the frozen paper artifact."""
    graph = _build_eval_graph(
        keyframes=keyframes_from(cache, d288), poses=cache["poses"],
        descriptors=d288, cache=cache, config=cfg, device=device,
        temporal_edge_mode="bidirectional", temporal_direction_mode="none",
        similarity_min_k=0, phase_features=None, sensor_key=sensor,
    )
    with torch.no_grad():
        emb_a = model(graph.to(device)).detach().cpu().numpy()

    d_rm = _normalize(d288 * r[None])
    src, dst, feats, _ = edge_features(
        emb_a, d_rm, range_phase_flat(layouts), cache["poses"], want_labels=False,
    )
    mu = clf_blob["mu"].astype(np.float32)
    sd = clf_blob["sd"].astype(np.float32)
    with torch.no_grad():
        logits = clf(torch.from_numpy((feats - mu) / sd).float().to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    keep_k, n = 10, len(emb_a)
    probs2 = probs.reshape(n, -1)
    pick = np.argsort(-probs2, axis=1)[:, :keep_k]
    src2 = np.repeat(np.arange(n), keep_k)
    dst2 = dst.reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
    cos2 = feats[:, 0].reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
    prob2 = probs2[np.arange(n)[:, None], pick].reshape(-1)
    l2_2 = np.linalg.norm(emb_a[src2] - emb_a[dst2], axis=1)

    cfg_t = copy.deepcopy(cfg)
    cfg_t["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.0
    graph_b = _build_eval_graph(
        keyframes=keyframes_from(cache, d288), poses=cache["poses"],
        descriptors=d288, cache=cache, config=cfg_t, device=device,
        temporal_edge_mode="bidirectional", temporal_direction_mode="none",
        similarity_min_k=0, phase_features=None, sensor_key=sensor,
    )
    dev = graph_b.edge_index.device
    edge_index = torch.from_numpy(np.stack([src2, dst2])).long().to(dev)
    edge_attr = torch.from_numpy(np.stack([
        np.zeros_like(cos2), np.zeros_like(cos2), cos2,
        np.log1p(l2_2) / 5.0, prob2,
    ], axis=1).astype(np.float32)).to(dev)
    graph_b.edge_index = torch.cat([graph_b.edge_index, edge_index], dim=1)
    graph_b.edge_attr = torch.cat([graph_b.edge_attr, edge_attr], dim=0)
    graph_b.edge_type = torch.cat([
        graph_b.edge_type,
        torch.ones(edge_index.shape[1], dtype=torch.long, device=dev),
    ])
    with torch.no_grad():
        emb_b = model(graph_b.to(device)).detach().cpu().numpy()
    return _normalize(emb_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-k", type=int, default=100)
    args = ap.parse_args()
    V = VARIANTS[args.variant]
    tags = args.tags or V["tags"]
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, model, clf, clf_blob, r, ckpt = make_model_pd(
        args.variant, args.seed, args.device)
    result_path = OUT / f"branch_fusion_{args.variant}_seed{args.seed}.json"
    report = json.loads(result_path.read_text()) if result_path.exists() else {}
    report["_meta"] = {
        "variant": args.variant, "seed": args.seed,
        "checkpoint": str(ckpt), "rvec": str(V["rvec"]), "classifier": str(V["cls"]),
        "config": str(V["config"]), "pool_k": args.pool_k,
        "protocol": "paper-main branch fusion (union top-100 x2, min-max 4ch, "
                    "WEIGHTS grid) on pd NEW VAL only",
    }

    for tag in tags:
        if tag in report and {"pbev128_main", "hspec12"} <= set(
                report[tag].get("variants", {})):
            print(f"[{tag}] already done, skipping", flush=True)
            continue
        sensor, sequence = JOBS_PD[tag]
        print(f"\n=== {args.variant} seed{args.seed} {tag} "
              f"({sensor}/{sequence}) ===", flush=True)
        cache = _pd_val_cache(sensor, sequence)
        d288 = cache["descriptors"].astype(np.float32)
        lay = np.load(PD_CACHE / f"vallayouts_{sensor}_{sequence}.npz")
        assert (lay["scan_ids"] == cache["scan_ids"]).all(), tag
        spec = np.load(PD_CACHE / f"valspec_{sensor}_{sequence}.npz")
        assert (spec["scan_ids"] == cache["scan_ids"]).all(), tag
        layouts = lay["nsd_layouts"].astype(np.float32)
        sk = spec["sk_re"].astype(np.float32) + 1j * spec["sk_im"].astype(np.float32)
        raw_height = spec["raw_height"].astype(np.float32)

        nsd = nsd_embedding_pd(cache, sensor, d288, layouts,
                               cfg, model, clf, clf_blob, r, args.device)

        variants = report.get(tag, {}).get("variants", {})
        for name, (hkey, hfft, hnorm, column_bank) in (
                ("pbev128_main", variant_inputs(sk, 8)),
                ("hspec12", hspec_inputs(raw_height, 12))):
            if name in variants:
                continue
            print(f"[{tag}] variant={name}", flush=True)
            vals = evaluate(cache["poses"], nsd, hkey, hfft, hnorm,
                            column_bank, args.pool_k)
            variants[name] = vals
            best = max(WEIGHTS, key=lambda w: vals[str(w)])
            print(f"  nsd={vals['nsd']:.4f} branch={vals['height']:.4f} "
                  f"best={vals[str(best)]:.4f} w={best} "
                  f"oracle={vals['oracle_union']:.4f} q={vals['n_queries']}",
                  flush=True)
        report[tag] = {"sensor": sensor, "sequence": sequence,
                       "pool_k": args.pool_k, "variants": variants}
        result_path.write_text(json.dumps(report, indent=2))
    print("PD_BRANCH_FUSION_DONE", flush=True)


if __name__ == "__main__":
    main()
