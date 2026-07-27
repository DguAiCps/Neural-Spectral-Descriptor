#!/usr/bin/env python3
"""High-upside height-key probe with frozen NSD context and cyclic alignment.

This is an exploratory script.  It never fits on the held-out sequences.  The
reported weight sweep is an oracle diagnostic used to decide which *fixed*
fusion family should subsequently be calibrated on the training split.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from data.kitti_loader import KITTILoader
from data.nclt_loader import NCLTLoader
from data.helipr_loader import HeLiPRLoader
try:
    from data.mulran_loader_old import MulRanLoader
except ImportError:
    from data.mulran_loader import MulRanLoader
from evaluate_kitti_checkpoint import _apply_encoder_preset, _build_eval_graph, _make_model
from keyframe.selector import Keyframe
from run_kitti_operating_point import (
    _find_queries, _make_encoder, _normalize, _project_nsd_layout, _topk_cosine,
)
from build_and_train_edge_classifier import (
    EdgeMLP, N_FEAT, edge_features, range_phase_flat,
)


ROOT = Path("/rise/RISE1/workspace/data")
CACHE_OUT = REPO / "results/height_sota/cache"
RESULT_OUT = REPO / "results/height_sota"
DTH, SKIP = 5.0, 30

JOBS = {
    "K00": ("kitti", "00", "data/preprocessed/cache_de201724_kitti_val_00.npz"),
    "K05": ("kitti", "05", "data/preprocessed/cache_de201724_kitti_val_05.npz"),
    "K08": ("kitti", "08", "data/preprocessed/cache_de201724_kitti_val_08.npz"),
    "K01_TR": ("kitti", "01", "data/preprocessed_train/cache_de201724_kitti_val_01.npz"),
    "K02_TR": ("kitti", "02", "data/preprocessed_train/cache_de201724_kitti_val_02.npz"),
    "K06_TR": ("kitti", "06", "data/preprocessed_train/cache_de201724_kitti_val_06.npz"),
    "K07_TR": ("kitti", "07", "data/preprocessed_train/cache_de201724_kitti_val_07.npz"),
    "N12": ("nclt", "2012-01-08", "data/preprocessed_nclt_fix8/nclt_operating_2012-01-08_layout60_stride1.npz"),
    "N13": ("nclt", "2013-01-10", "data/preprocessed_nclt_fix8/nclt_operating_2013-01-10_layout60_stride1.npz"),
    "N0511_TR": ("nclt", "2012-05-11", "data/preprocessed_nclt_fix8/nclt_operating_2012-05-11_layout60_stride1.npz"),
    "N0804_TR": ("nclt", "2012-08-04", "data/preprocessed_nclt_fix8/nclt_operating_2012-08-04_layout60_stride1.npz"),
    "TOWN": ("helipr", "Town01", "data/preprocessed_cross_sensor_operating/helipr_operating_Town01_layout60_stride1.npz"),
    "DCC": ("mulran", "DCC03", "data/preprocessed/cache_dabab5b5_mulran_val_DCC03.npz"),
    "KAI": ("mulran", "KAIST03", "data/preprocessed/cache_dabab5b5_mulran_val_KAIST03.npz"),
    "RIV": ("mulran", "Riverside03", "data/preprocessed/cache_dabab5b5_mulran_val_Riverside03.npz"),
}


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def bin288(m: np.ndarray) -> np.ndarray:
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 181]
    means, stds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = m[..., lo:hi]
        mean = band.mean(-1)
        means.append(mean)
        stds.append(np.sqrt(((band - mean[..., None]) ** 2).mean(-1) + 1e-8))
    # Preserve the encoder's native scale for the frozen GNN/BatchNorm.  Cosine
    # retrieval normalizes later; normalizing here silently destroys the
    # checkpoint's input distribution.
    return np.concatenate([
        np.stack(means, -1).reshape(len(m), -1),
        np.stack(stds, -1).reshape(len(m), -1),
    ], axis=1).astype(np.float32)


def octave_key(grid: np.ndarray) -> np.ndarray:
    n_freq = grid.shape[-1] // 2 + 1
    edges = [0, 1, 2, 4, 8, 16, n_freq]
    spec = np.abs(np.fft.rfft(grid, axis=-1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = spec[..., lo:hi]
        out.extend((band.mean(-1), band.std(-1)))
    return l2(np.concatenate(out, axis=-1).astype(np.float32))


def make_height_grids(points: np.ndarray, n_rings: int = 20, n_sectors: int = 120):
    xyz = points[:, :3]
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    keep = (radius > 0.5) & (radius < 80.0) & (xyz[:, 2] > -3.0) & (xyz[:, 2] < 25.0)
    xyz, radius = xyz[keep], radius[keep]
    z = xyz[:, 2].astype(np.float32)
    ground = float(np.percentile(z, 5)) if len(z) else 0.0
    z_ground = z - ground
    theta = (np.arctan2(xyz[:, 1], xyz[:, 0]) + np.pi) / (2 * np.pi)
    rings = np.clip((radius / 80.0 * n_rings).astype(int), 0, n_rings - 1)
    sectors = np.clip((theta * n_sectors).astype(int), 0, n_sectors - 1)

    height = np.zeros((n_rings, n_sectors), np.float32)
    raw_height = np.zeros((n_rings, n_sectors), np.float32)
    density = np.zeros_like(height)
    np.maximum.at(height, (rings, sectors), z_ground)
    np.maximum.at(raw_height, (rings, sectors), z)
    np.add.at(density, (rings, sectors), 1.0)
    density = np.log1p(density)

    # Exact original 20x60 key, retained as a protocol regression control.
    sec60 = np.clip((theta * 60).astype(int), 0, 59)
    legacy = np.full((n_rings, 60), -3.0, np.float32)
    np.maximum.at(legacy, (rings, sec60), z)
    return height, raw_height, density, legacy


def loader_for(sensor: str, sequence: str):
    if sensor == "kitti":
        return KITTILoader(str(ROOT / "kitti/dataset"), sequence, lazy_load=True)
    if sensor == "nclt":
        return NCLTLoader(str(ROOT / "nclt"), sequence, lazy_load=True)
    if sensor == "helipr":
        return HeLiPRLoader(str(ROOT / "helipr" / sequence), lazy_load=True)
    return MulRanLoader(str(ROOT / "mulran"), sequence, lazy_load=True)


def height_cache(tag: str, sensor: str, sequence: str,
                 cache: np.lib.npyio.NpzFile, encoder):
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    path = CACHE_OUT / f"{tag}_height120.npz"
    if path.exists():
        z = np.load(path)
        if (np.array_equal(z["scan_ids"], cache["scan_ids"])
                and "nsd_layout" in z.files and "raw_height" in z.files):
            return z
        print(f"[{tag}] stale height cache; rebuilding", flush=True)

    loader = loader_for(sensor, sequence)
    heights, raw_heights, densities, legacy, layouts = [], [], [], [], []
    for i, scan_id in enumerate(cache["scan_ids"]):
        if i % 500 == 0:
            print(f"[{tag}] height {i}/{len(cache['scan_ids'])}", flush=True)
        points = loader[int(scan_id)]["points"]
        h, h_raw, d, old = make_height_grids(points)
        heights.append(h.astype(np.float16))
        raw_heights.append(h_raw.astype(np.float16))
        densities.append(d.astype(np.float16))
        legacy.append(octave_key(old).astype(np.float32))
        layouts.append(_project_nsd_layout(encoder, points, 60).astype(np.float16))
    heights = np.asarray(heights, dtype=np.float16)
    raw_heights = np.asarray(raw_heights, dtype=np.float16)
    densities = np.asarray(densities, dtype=np.float16)
    legacy = np.asarray(legacy, dtype=np.float32)
    layouts = np.asarray(layouts, dtype=np.float16)
    np.savez_compressed(path, scan_ids=cache["scan_ids"], height=heights,
                        raw_height=raw_heights,
                        density=densities, legacy_key=legacy,
                        nsd_layout=layouts)
    return np.load(path)


def keyframes_from(cache: np.lib.npyio.NpzFile, descriptors: np.ndarray):
    return [
        Keyframe(
            keyframe_id=int(cache["keyframe_ids"][i]),
            scan_id=int(cache["scan_ids"][i]),
            points=np.empty((0, 3), dtype=np.float32),
            pose=cache["poses"][i],
            timestamp=float(cache["timestamps"][i]),
            descriptor=descriptors[i],
        )
        for i in range(len(descriptors))
    ]


def make_model(device: str, seed: int):
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")),
        "no_interdiff",
    )
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    cfg = copy.deepcopy(cfg)
    cfg["gnn"]["key_remetrize"] = {
        "enabled": True,
        "init_path": str(REPO / "artifacts/key_remetrize_r.npy"),
    }
    ckpt = {
        1: REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
        2: REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
        3: REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
    }[seed]
    model = _make_model(cfg, ckpt, device)
    blob = torch.load(REPO / "artifacts/edge_classifier.pt",
                      map_location="cpu", weights_only=False)
    clf = EdgeMLP(N_FEAT)
    clf.load_state_dict(blob["state_dict"])
    clf.eval().to(device)
    return cfg, model, clf, blob


def nsd_embedding(cache, sensor: str, d288: np.ndarray, layouts: np.ndarray,
                  cfg, model, clf, clf_blob, device: str):
    graph = _build_eval_graph(
        keyframes=keyframes_from(cache, d288), poses=cache["poses"],
        descriptors=d288, cache=cache, config=cfg, device=device,
        temporal_edge_mode="bidirectional", temporal_direction_mode="none",
        similarity_min_k=0, phase_features=None, sensor_key=sensor,
    )
    with torch.no_grad():
        emb_a = model(graph.to(device)).detach().cpu().numpy()

    r = np.load(REPO / "artifacts/key_remetrize_r.npy").astype(np.float32)
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


def align_bank(grid: np.ndarray, center: bool):
    x = grid.astype(np.float32)
    if center:
        x = x - x.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(x.reshape(len(x), -1), axis=1)
    fft = np.fft.rfft(x, axis=-1).astype(np.complex64)
    return fft, np.clip(norm, 1e-8, None)


def align_scores(fft: np.ndarray, norm: np.ndarray, query: int, candidates: np.ndarray):
    corr = np.fft.irfft(fft[candidates] * np.conj(fft[query]),
                        n=(fft.shape[-1] - 1) * 2, axis=-1)
    corr = corr.sum(axis=tuple(range(1, corr.ndim - 1))).max(axis=-1)
    return corr / (norm[candidates] * norm[query])


def column_align_bank(grid: np.ndarray):
    """FFT bank for exact SC-style mean column-cosine cyclic alignment."""
    x = grid.astype(np.float32)
    col_norm = np.linalg.norm(x, axis=1, keepdims=True)
    valid = (col_norm[:, 0] > 1e-8).astype(np.float32)
    x = x / np.clip(col_norm, 1e-8, None)
    return (np.fft.rfft(x, axis=-1).astype(np.complex64),
            np.fft.rfft(valid, axis=-1).astype(np.complex64))


def column_align_scores(bank, query: int, candidates: np.ndarray):
    fft, mask_fft = bank
    n_sector = (fft.shape[-1] - 1) * 2
    corr = np.fft.irfft(fft[candidates] * np.conj(fft[query]),
                        n=n_sector, axis=-1).sum(axis=1)
    count = np.fft.irfft(mask_fft[candidates] * np.conj(mask_fft[query]),
                         n=n_sector, axis=-1)
    score = np.where(count > 0.5, corr / np.maximum(count, 1.0), -1.0)
    return score.max(axis=-1)


def minmax(x: np.ndarray):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / max(hi - lo, 1e-8)


WEIGHTS = []
for wa in (0.0, 0.25, 0.5, 1.0, 2.0):
    for wc in (0.0, 0.25, 0.5, 1.0, 2.0):
        WEIGHTS.append((0.0, 1.0, wa, wc))
for wn in (0.25, 0.5, 1.0, 2.0):
    for wa in (0.0, 0.5, 1.0, 2.0):
        for wc in (0.0, 0.5, 1.0, 2.0):
            WEIGHTS.append((wn, 1.0, wa, wc))
WEIGHTS = sorted(set(WEIGHTS))


def evaluate(poses, nsd, hkey, hfft, hnorm, column_bank, pool_k: int):
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    correct = {str(w): 0 for w in WEIGHTS}
    correct.update({"nsd": 0, "height": 0, "oracle_union": 0})
    gt_pool = 0
    for qi, (q, _) in enumerate(queries):
        cn = _topk_cosine(nsd, q, pool_k + 2 * SKIP, SKIP)[:pool_k]
        ch = _topk_cosine(hkey, q, pool_k + 2 * SKIP, SKIP)[:pool_k]
        candidates = np.unique(np.concatenate([cn, ch]))
        good = np.linalg.norm(positions[candidates] - positions[q], axis=1) < DTH
        correct["oracle_union"] += bool(good.any())
        gt_pool += bool(good.any())

        sn = minmax(nsd[candidates] @ nsd[q])
        sh = minmax(hkey[candidates] @ hkey[q])
        sa = minmax(align_scores(hfft, hnorm, q, candidates))
        sd = minmax(column_align_scores(column_bank, q, candidates))
        for w in WEIGHTS:
            score = w[0] * sn + w[1] * sh + w[2] * sa + w[3] * sd
            correct[str(w)] += bool(good[int(np.argmax(score))])

        correct["nsd"] += np.linalg.norm(
            positions[_topk_cosine(nsd, q, 1 + 2 * SKIP, SKIP)[0]] - positions[q]) < DTH
        correct["height"] += np.linalg.norm(
            positions[_topk_cosine(hkey, q, 1 + 2 * SKIP, SKIP)[0]] - positions[q]) < DTH
        if qi and qi % 500 == 0:
            print(f"  fusion queries {qi}/{len(queries)}", flush=True)
    n = max(len(queries), 1)
    return {k: float(v / n) for k, v in correct.items()} | {"n_queries": len(queries)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=list(JOBS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pool-k", type=int, default=100)
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--variants", nargs="*", default=None,
                    choices=("legacy240", "raw_height240", "height240",
                             "height_density480"))
    args = ap.parse_args()
    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    cfg, model, clf, clf_blob = make_model(args.device, args.seed)
    encoder = _make_encoder(cfg, args.device)
    suffix = "" if args.seed == 1 else f"_seed{args.seed}"
    result_path = RESULT_OUT / f"height_candidate_fusion{suffix}.json"
    report = json.loads(result_path.read_text()) if result_path.exists() else {}
    for tag in args.tags:
        sensor, sequence, cache_path = JOBS[tag]
        print(f"\n=== {tag} ===", flush=True)
        cache = np.load(REPO / cache_path)
        d288 = bin288(cache["fft_magnitudes"].astype(np.float32))
        hc = height_cache(tag, sensor, sequence, cache, encoder)
        height = hc["height"].astype(np.float32)
        raw_height = hc["raw_height"].astype(np.float32)
        density = hc["density"].astype(np.float32)
        keys = {
            "legacy240": l2(hc["legacy_key"].astype(np.float32)),
            "raw_height240": octave_key(raw_height),
            "height240": octave_key(height),
            "height_density480": l2(np.concatenate([
                octave_key(height), octave_key(density)], axis=1)),
        }
        nsd = nsd_embedding(
            cache, sensor, d288, hc["nsd_layout"].astype(np.float32),
            cfg, model, clf, clf_blob, args.device,
        )
        hfft, hnorm = align_bank(height[:, None], center=True)
        column_bank = column_align_bank(raw_height)
        variants = {}
        selected = set(args.variants or keys)
        for name, hkey in keys.items():
            if name not in selected:
                continue
            print(f"[{tag}] variant={name}", flush=True)
            variants[name] = evaluate(cache["poses"], nsd, hkey,
                                      hfft, hnorm, column_bank, args.pool_k)
            vals = variants[name]
            best_name = max(WEIGHTS, key=lambda w: vals[str(w)])
            print(f"  nsd={vals['nsd']:.4f} h={vals['height']:.4f} "
                  f"best={vals[str(best_name)]:.4f} w={best_name} "
                  f"oracle={vals['oracle_union']:.4f}", flush=True)
        report[tag] = {"sensor": sensor, "sequence": sequence,
                       "pool_k": args.pool_k, "variants": variants}
        result_path.write_text(json.dumps(report, indent=2))
    print("HEIGHT_CANDIDATE_FUSION_DONE", flush=True)


if __name__ == "__main__":
    main()
