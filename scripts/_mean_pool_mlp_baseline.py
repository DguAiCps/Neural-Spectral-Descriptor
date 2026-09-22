#!/usr/bin/env python3
"""Train a compact temporal mean-pool + MLP baseline on the configured train split.

The baseline has no graph message passing, edge selector, or external
pretraining.  It maps the concatenation of a 288D invariant key and the mean
of its fixed +/-5 temporal neighbours to a 416D retrieval embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from run_kitti_operating_point import _find_queries, _normalize, _topk_cosine  # noqa: E402


TRAIN_CACHE = {
    "helipr": ("08b9d745", ["Town02", "Roundabout01", "Bridge01", "KAIST04", "DCC04", "Riverside04"]),
    "kitti": ("de201724", ["01", "02", "06", "07"]),
    "nclt": ("3a73ece2", ["2012-05-11", "2012-08-04", "2012-11-04", "2012-11-16", "2013-02-23"]),
    "mulran": ("dabab5b5", ["DCC01", "DCC02", "KAIST01", "KAIST02", "Riverside01", "Riverside02", "Sejong01", "Sejong02", "Sejong03"]),
}
EVAL_SEQUENCES = [
    ("kitti", "00"), ("kitti", "05"), ("kitti", "08"),
    ("nclt", "2012-01-08"), ("nclt", "2013-01-10"),
    ("helipr", "Town01"),
]
DTH, SKIP, POS_RADIUS, NEG_RADIUS = 5.0, 30, 5.0, 25.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bin288(magnitudes: np.ndarray) -> np.ndarray:
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 181]
    means, stds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = magnitudes[..., lo:hi]
        mean = band.mean(-1)
        means.append(mean)
        stds.append(np.sqrt(((band - mean[..., None]) ** 2).mean(-1) + 1e-8))
    return np.concatenate([
        np.stack(means, -1).reshape(len(magnitudes), -1),
        np.stack(stds, -1).reshape(len(magnitudes), -1),
    ], axis=1).astype(np.float32)


def temporal_mean(keys: np.ndarray, radius: int = 5) -> np.ndarray:
    """Mean pool fixed index neighbours without accessing poses or labels."""
    n = len(keys)
    pooled = np.empty_like(keys)
    for index in range(n):
        lo, hi = max(0, index - radius), min(n, index + radius + 1)
        neighbors = np.arange(lo, hi)
        neighbors = neighbors[neighbors != index]
        pooled[index] = keys[neighbors].mean(axis=0) if len(neighbors) else keys[index]
    return pooled


def features_from_cache(cache: np.lib.npyio.NpzFile) -> np.ndarray:
    keys = bin288(cache["fft_magnitudes"].astype(np.float32))
    return np.concatenate([keys, temporal_mean(keys)], axis=1).astype(np.float32)


class MeanPoolMLP(nn.Module):
    def __init__(self, input_dim: int = 576, hidden_dim: int = 128, output_dim: int = 416):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(x), dim=1)


def train_cache_paths() -> list[tuple[str, Path]]:
    result = []
    for sensor, (cache_key, sequences) in TRAIN_CACHE.items():
        for sequence in sequences:
            path = REPO / "data/preprocessed" / f"cache_{cache_key}_{sensor}_{sequence}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            result.append((f"{sensor}/{sequence}", path))
    return result


def build_triplets(positions: np.ndarray, segment_offsets: list[tuple[int, int]],
                   rng: np.random.Generator, requested: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors, positives, negatives = [], [], []
    candidates = []
    for begin, end in segment_offsets:
        local = positions[begin:end]
        tree = cKDTree(local)
        for index, near in enumerate(tree.query_ball_point(local, POS_RADIUS)):
            valid = [j for j in near if abs(j - index) >= SKIP]
            if valid:
                candidates.append((begin, end, index, np.asarray(valid, dtype=np.int64)))
    if not candidates:
        raise RuntimeError("No train anchors have a non-temporal <5m positive")
    picks = rng.integers(0, len(candidates), size=requested)
    for pick in picks:
        begin, end, local_index, positive_choices = candidates[int(pick)]
        anchor = begin + local_index
        positive = begin + int(rng.choice(positive_choices))
        for _ in range(100):
            negative = int(rng.integers(begin, end))
            if (abs(negative - anchor) >= SKIP
                    and np.linalg.norm(positions[negative] - positions[anchor]) >= NEG_RADIUS):
                break
        else:
            continue
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    if not anchors:
        raise RuntimeError("Unable to draw any >=25m negatives")
    return (np.asarray(anchors), np.asarray(positives), np.asarray(negatives))


def recall_at_1(embedding: np.ndarray, poses: np.ndarray) -> tuple[float, int]:
    key = _normalize(embedding)
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    hits = 0
    for query, _ in queries:
        ranked = _topk_cosine(key, query, 1 + 2 * SKIP, SKIP)
        hits += int(len(ranked) and np.linalg.norm(positions[ranked[0]] - positions[query]) < DTH)
    return hits / max(len(queries), 1), len(queries)


def train_and_evaluate(seed: int, args, train_features: np.ndarray,
                       train_positions: np.ndarray, offsets: list[tuple[int, int],],
                       train_metadata: list[dict]) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    anchors, positives, negatives = build_triplets(
        train_positions, offsets, rng, args.triplets
    )
    device = torch.device(args.device)
    feature_tensor = torch.from_numpy(train_features).to(device)
    model = MeanPoolMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(anchors))
        epoch_losses = []
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            anchor = model(feature_tensor[torch.from_numpy(anchors[selected]).to(device)])
            positive = model(feature_tensor[torch.from_numpy(positives[selected]).to(device)])
            negative = model(feature_tensor[torch.from_numpy(negatives[selected]).to(device)])
            loss = torch.relu(args.margin + (anchor * negative).sum(1) - (anchor * positive).sum(1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
        print(f"[seed {seed}] epoch {epoch + 1}/{args.epochs} loss={losses[-1]:.5f}", flush=True)

    checkpoint_root = args.checkpoint_root
    if not checkpoint_root.is_absolute():
        checkpoint_root = REPO / checkpoint_root
    checkpoint_dir = checkpoint_root / f"mean_pool_mlp_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "final_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(), "seed": seed,
        "architecture": "concat(288D invariant key, +/-5 temporal mean) -> Linear(576,128) -> ReLU -> Linear(128,416)",
        "final_epoch": args.epochs,
    }, checkpoint)

    per_sequence, total, weighted = {}, 0, 0.0
    model.eval()
    with torch.no_grad():
        for sensor, sequence in EVAL_SEQUENCES:
            cache_path = REPO / "data/preprocessed_cross_sensor_operating" / f"{sensor}_operating_{sequence}_layout60_stride1.npz"
            cache = np.load(cache_path)
            features = torch.from_numpy(features_from_cache(cache)).to(device)
            embedding = model(features).cpu().numpy()
            recall, n_queries = recall_at_1(embedding, cache["poses"])
            tag = f"{sensor}/{sequence}"
            per_sequence[tag] = {
                "recall_at_1": recall,
                "n_queries": n_queries,
                "cache_path": str(cache_path.relative_to(REPO)),
                "cache_sha256": sha256(cache_path),
            }
            weighted += recall * n_queries
            total += n_queries
            print(f"[seed {seed}] eval {tag}: {recall:.4f}", flush=True)
    return {
        "seed": seed,
        "checkpoint": str(checkpoint.relative_to(REPO)),
        "checkpoint_sha256": sha256(checkpoint),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "final_epoch": args.epochs,
        "loss_by_epoch": losses,
        "training_triplets": int(len(anchors)),
        "training_cache": train_metadata,
        "query_weighted_recall_at_1": weighted / total,
        "n_queries_total": total,
        "per_sequence": per_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--triplets", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--checkpoint-root", type=Path,
                        default=REPO / "checkpoints/supplementary_b4")
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/supplementary/mean_pool_mlp_3seed.json")
    args = parser.parse_args()
    if args.epochs < 1 or args.triplets < 1 or args.batch_size < 1:
        raise ValueError("epochs, triplets, and batch-size must be positive")
    paths = train_cache_paths()
    features, positions, offsets, metadata = [], [], [], []
    offset = 0
    for tag, path in paths:
        cache = np.load(path)
        local_features = features_from_cache(cache)
        local_positions = cache["poses"][:, :3, 3].astype(np.float32)
        features.append(local_features)
        positions.append(local_positions)
        offsets.append((offset, offset + len(local_features)))
        metadata.append({"sequence": tag, "path": str(path.relative_to(REPO)),
                         "sha256": sha256(path), "n_keyframes": len(local_features)})
        offset += len(local_features)
    train_features = np.concatenate(features)
    train_positions = np.concatenate(positions)
    report = {
        "protocol": {
            "name": "B4 mean-pool + MLP simple learned baseline",
            "input": "concat(288D invariant key, mean of fixed +/-5 temporal neighbour keys)",
            "model": "Linear(576,128) -> ReLU -> Linear(128,416), L2-normalized output",
            "training": "fixed final epoch; triplet loss on configured training split only",
            "triplet_positive": f"same sequence, <{POS_RADIUS:g}m, |index gap| >= {SKIP}",
            "triplet_negative": f"same sequence, >= {NEG_RADIUS:g}m, |index gap| >= {SKIP}",
            "evaluation": "six-sequence raw-416D diagnostic protocol; clean poses, <5m, 30-frame skip",
            "gru_trigger": "Run GRU only if the 3-seed mean is within 1.0 percentage point of the NSD v0 reference (0.5442).",
        },
        "provenance": {
            "command": shlex.join(sys.argv),
            "script_sha256": sha256(Path(__file__)),
            "seeds": args.seeds,
            "epochs": args.epochs,
            "triplets_per_seed": args.triplets,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "margin": args.margin,
        },
        "seeds": {},
    }
    for seed in args.seeds:
        report["seeds"][str(seed)] = train_and_evaluate(
            seed, args, train_features, train_positions, offsets, metadata
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    values = [row["query_weighted_recall_at_1"] for row in report["seeds"].values()]
    report["three_seed_summary"] = {
        "query_weighted_recall_at_1": {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values,
        },
        "nsd_v0_reference": 0.544203646370829,
        "gru_required_by_predeclared_trigger": bool(np.mean(values) >= 0.534203646370829),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
