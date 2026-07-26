#!/usr/bin/env python3
"""Deployed-config component-ablation eval (tab:ablation re-run).

Evaluates the RAW 416D retrieval key (cat(288D magnitude, 128D gated ctx);
NO key re-metrization, NO B3 edges, NO phase rerank) with cosine Recall@1 on
the KITTI+NCLT+HeLiPR analysis subset (6 sequences, 4,845 queries), protocol
5 m / 30-frame skip — identical to the legacy 672D tab:ablation protocol and
to scripts/_eval_remetrize.py / _metric_probe_416.py.

Each variant is evaluated with ITS OWN config so the graph restriction that
was active at training applies identically at eval graph construction:
  - v3: keyframe.graph.similarity_max_k=0 -> no similarity edges
  - v4: keyframe.graph.edge_type_filter=similarity_only -> temporal dropped
  - v1/v2/v5: graph identical to baseline; the delta lives in the model/mining

Usage (inside nvcr.io/nvidia/pyg:26.01-py3, repo at /workspace/...):
  python scripts/_eval_deployed_ablation.py baseline v1 v2 v3 v4 v5
  # smoke: single variant, custom ckpt/config, one sequence
  python scripts/_eval_deployed_ablation.py v3 \
      --config configs/_smoke_abl_v3.yaml \
      --checkpoint checkpoints/_smoke_abl_v3/best_model.pth \
      --sequences kitti/00 --out results/deployed_ablation_smoke.json

Results merge into results/deployed_ablation.json (one key per variant).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import (  # noqa: E402
    _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes,
)
from run_kitti_operating_point import (  # noqa: E402
    _find_queries, _topk_cosine, _normalize,
)
from keyframe.graph_manager import filter_edges_by_type  # noqa: E402

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTH, SKIP = 5.0, 30

# KITTI+NCLT+HeLiPR analysis subset (matches tab:ablation, 4,845 queries)
SEQUENCES = [
    ("kitti", "00"), ("kitti", "05"), ("kitti", "08"),
    ("nclt", "2012-01-08"), ("nclt", "2013-01-10"),
    ("helipr", "Town01"),
]
EXPECTED_QC = {
    "kitti/00": 632, "kitti/05": 377, "kitti/08": 235,
    "nclt/2012-01-08": 1834, "nclt/2013-01-10": 181, "helipr/Town01": 1586,
}

VARIANTS = {
    "baseline": {
        "label": "deployed baseline (seed1, no retrain)",
        "config": "configs/training_multi_dataset.yaml",
        "checkpoint": "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    },
    "v0": {
        "label": "retrained un-ablated control",
        "config": "configs/_abl_deployed_v0.yaml",
        "checkpoint": "checkpoints/abl_deployed_v0/best_model.pth",
    },
    "v1": {
        "label": "standard attention (on h_j)",
        "config": "configs/_abl_deployed_v1.yaml",
        "checkpoint": "checkpoints/abl_deployed_v1/best_model.pth",
    },
    "v2": {
        "label": "no edge bias",
        "config": "configs/_abl_deployed_v2.yaml",
        "checkpoint": "checkpoints/abl_deployed_v2/best_model.pth",
    },
    "v3": {
        "label": "temporal edges only",
        "config": "configs/_abl_deployed_v3.yaml",
        "checkpoint": "checkpoints/abl_deployed_v3/best_model.pth",
    },
    "v4": {
        "label": "similarity edges only",
        "config": "configs/_abl_deployed_v4.yaml",
        "checkpoint": "checkpoints/abl_deployed_v4/best_model.pth",
    },
    "v5": {
        "label": "mining on refined descriptors",
        "config": "configs/_abl_deployed_v5.yaml",
        "checkpoint": "checkpoints/abl_deployed_v5/best_model.pth",
    },
}


def load_variant_config(config_path: Path) -> dict:
    """Load a variant YAML and mirror the mandatory CLI overrides."""
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(config_path)), "no_interdiff"
    )
    cfg["gnn"]["use_residual_gate"] = True          # --use-gated-context
    cfg["gnn"]["gate_initial_alpha"] = 0.0625       # --gate-initial-alpha
    return cfg


def cosine_r1(key: np.ndarray, poses: np.ndarray) -> tuple[float, int]:
    """Cosine Recall@1, canonical tab:ablation protocol.

    Queries: _find_queries (causal — a true match exists >= SKIP frames
    earlier; yields the 4,845-query subset). Ranking: _topk_cosine over all
    frames with |idx - q| >= SKIP. Success: top-1 within DTH meters.
    Identical to the pass-1 coarse scoring in _dump_b3_yaw_perquery.py.
    """
    normed = _normalize(key)
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    correct = 0
    for q_idx, _ in queries:
        ranked = _topk_cosine(normed, q_idx, 1 + 2 * SKIP, SKIP)
        if len(ranked) == 0:
            continue
        if np.linalg.norm(positions[ranked[0]] - positions[q_idx]) < DTH:
            correct += 1
    return correct / max(len(queries), 1), len(queries)


def eval_variant(variant: str, config_path: Path, ckpt_path: Path,
                 sequences: list[tuple[str, str]]) -> dict:
    cfg = load_variant_config(config_path)
    edge_type_filter = (
        cfg["keyframe"].get("graph", {}).get("edge_type_filter", "both")
    )
    model = _make_model(cfg, ckpt_path, DEVICE)
    core = model.gnn if hasattr(model, "gnn") else model
    assert getattr(core, "key_scale", None) is None, (
        "raw-d eval requires key_remetrize disabled")

    per_seq = {}
    for sensor, seq in sequences:
        z = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
        graph = _build_eval_graph(
            keyframes=_cache_to_keyframes(z), poses=z["poses"],
            descriptors=z["descriptors"].astype(np.float32), cache=z,
            config=cfg, device=DEVICE,
            temporal_edge_mode="bidirectional", temporal_direction_mode="none",
            similarity_min_k=0, phase_features=None, sensor_key=sensor,
        )
        graph = filter_edges_by_type(graph, edge_type_filter)
        n_t = int((graph.edge_type == 0).sum())
        n_s = int((graph.edge_type == 1).sum())
        with torch.no_grad():
            emb = model(graph.to(DEVICE)).detach().cpu().numpy()
        r1, nq = cosine_r1(emb, z["poses"])
        key = f"{sensor}/{seq}"
        per_seq[key] = {"recall@1": r1, "n_queries": nq,
                        "edges_temporal": n_t, "edges_similarity": n_s}
        exp = EXPECTED_QC.get(key)
        flag = "" if exp is None or exp == nq else f"  [WARN expected nq={exp}]"
        print(f"  {key:<18} R@1={r1:.4f}  nq={nq}  "
              f"edges(t={n_t:,}, s={n_s:,}){flag}", flush=True)

    total_q = sum(v["n_queries"] for v in per_seq.values())
    avg = sum(v["recall@1"] * v["n_queries"] for v in per_seq.values()) / max(total_q, 1)
    return {
        "label": VARIANTS.get(variant, {}).get("label", variant),
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "edge_type_filter": edge_type_filter,
        "protocol": {"distance_threshold_m": DTH, "skip_frames": SKIP,
                     "metric": "cosine", "key": "raw 416D (no re-metrization)"},
        "per_sequence": per_seq,
        "n_queries_total": total_q,
        "avg_recall@1_query_weighted": avg,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variants", nargs="+", choices=sorted(VARIANTS),
                    help="variants to evaluate")
    ap.add_argument("--config", type=Path, default=None,
                    help="override config YAML (single-variant runs only)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="override checkpoint path (single-variant runs only)")
    ap.add_argument("--sequences", nargs="*", default=None,
                    help="subset like kitti/00 nclt/2013-01-10 (default: all 6)")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results/deployed_ablation.json")
    args = ap.parse_args()

    if (args.config or args.checkpoint) and len(args.variants) != 1:
        ap.error("--config/--checkpoint overrides require exactly one variant")

    sequences = SEQUENCES
    if args.sequences:
        wanted = set(args.sequences)
        sequences = [(s, q) for s, q in SEQUENCES if f"{s}/{q}" in wanted]
        missing = wanted - {f"{s}/{q}" for s, q in sequences}
        if missing:
            ap.error(f"unknown sequences: {sorted(missing)}")

    results = {}
    if args.out.exists():
        results = json.loads(args.out.read_text())

    for variant in args.variants:
        spec = VARIANTS[variant]
        config_path = args.config or (REPO / spec["config"])
        ckpt_path = args.checkpoint or (REPO / spec["checkpoint"])
        if not ckpt_path.exists():
            print(f"[SKIP] {variant}: checkpoint missing: {ckpt_path}")
            continue
        print(f"\n=== {variant}: {spec['label']} ===", flush=True)
        results[variant] = eval_variant(variant, config_path, ckpt_path, sequences)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"  -> merged into {args.out}")

    base = results.get("baseline", {}).get("avg_recall@1_query_weighted")
    print("\n=== SUMMARY (query-weighted avg R@1, raw 416D key) ===")
    for variant in sorted(results):
        r = results[variant]
        delta = ""
        if base is not None and variant != "baseline":
            delta = f"  d={r['avg_recall@1_query_weighted'] - base:+.4f}"
        print(f"  {variant:<9} {r['avg_recall@1_query_weighted']:.4f}"
              f"  (nq={r['n_queries_total']}){delta}  {r['label']}")


if __name__ == "__main__":
    main()
