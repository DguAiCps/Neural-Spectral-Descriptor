"""Fine-tune BEVPlace++ on the NSD multi-sensor train set.

Goal: defend the σ_cross argument by showing that, even when BEVPlace++ is
fine-tuned on the same multi-sensor corpus NSD trains on, NSD retains its
cross-sensor consistency advantage (or, if not, report that honestly).

Pipeline:
  1. Load all 24 train sequences via existing data loaders.
  2. Stratify a 30k-keyframe subset across sensors (~7.5k per sensor).
  3. Pre-compute BEV images (uint8 200x200) -> ~1.2GB in RAM.
  4. Initialize BEVPlace++ REIN from the official KITTI checkpoint.
  5. Fine-tune for 5 epochs with Adam lr=1e-5, triplet loss margin 0.3,
     random negatives (no hard mining for speed; mining adds ~3x cost).
  6. Save checkpoint to baselines/weights/bevplace_finetune.pth for eval.

Usage (in container):
    python scripts/finetune_bevplace.py \
        --config configs/training_multi_dataset.yaml \
        --output baselines/weights/bevplace_finetune.pth \
        --epochs 5 --subset-per-sensor 7500
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

# Project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "baselines" / "_bevplace_official"))

from baselines.bevplace import _bev_image, _WEIGHTS_PATH


def create_loader(dataset_type: str, root: str, sequence: str):
    if dataset_type == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(root, sequence, lazy_load=True)
    if dataset_type == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(root, sequence, lazy_load=True)
    if dataset_type == "helipr":
        from data.helipr_loader import HeLiPRLoader
        seq_path = os.path.join(root, sequence, sequence)
        return HeLiPRLoader(seq_path, lazy_load=True)
    if dataset_type == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(root, sequence, lazy_load=True)
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def keyframes_with_poses(loader, keyframe_stride: int):
    """Iterate the loader at a stride, returning (idx, pose, points).

    Stride mimics our keyframe selector cheaply: every Nth scan is treated
    as a keyframe candidate. We don't need exact NSD keyframe semantics for
    BEVPlace++ training; pose-based triplets are what matters.
    """
    n = len(loader)
    idxs = list(range(0, n, keyframe_stride))
    for idx in idxs:
        d = loader[idx]
        yield idx, d["pose"], d["points"]


def collect_subset(config: dict,
                   subset_per_sensor: int,
                   bev_size: int = 200,
                   max_range: float = 50.0,
                   keyframe_stride: int = 5,
                   verbose: bool = True) -> dict:
    """Walk every train sequence, generate BEV, sub-sample per-sensor.

    Returns a dict of {sensor_name -> (bev_uint8: (N, S, S), poses: (N, 4, 4))}.
    Sensor names: HDL-64E, HDL-32E, VLP-16, OS1-64.
    """
    sensor_of = {"kitti": "HDL-64E", "nclt": "HDL-32E",
                 "helipr": "VLP-16", "mulran": "OS1-64"}

    accum = {s: {"bev": [], "pose": []} for s in sensor_of.values()}
    train = config["data"]["datasets"]["train"]
    rng = np.random.default_rng(0)

    for ds in train:
        sensor = sensor_of[ds["type"]]
        for seq in ds["sequences"]:
            t0 = time.time()
            try:
                loader = create_loader(ds["type"], ds["root"], seq)
            except Exception as e:
                if verbose:
                    print(f"[skip] {ds['type']}/{seq}: {e}")
                continue
            n = 0
            for idx, pose, pts in keyframes_with_poses(loader, keyframe_stride):
                bev = _bev_image(pts, img_size=bev_size, max_range=max_range)
                # store one channel (we'll broadcast later) as uint8
                ch = (bev[0] * 256).clip(0, 255).astype(np.uint8)
                accum[sensor]["bev"].append(ch)
                accum[sensor]["pose"].append(np.asarray(pose, dtype=np.float32))
                n += 1
            if verbose:
                print(f"  [{ds['type']}/{seq}] {n} keyframes in {time.time()-t0:.1f}s "
                      f"({sensor}, total {len(accum[sensor]['bev'])})")

    # Subsample per sensor.
    per_sensor = {}
    for s, d in accum.items():
        if not d["bev"]:
            continue
        all_bev = np.stack(d["bev"], axis=0)
        all_pose = np.stack(d["pose"], axis=0)
        n = all_bev.shape[0]
        if n > subset_per_sensor:
            sel = rng.choice(n, size=subset_per_sensor, replace=False)
            sel.sort()
            all_bev = all_bev[sel]
            all_pose = all_pose[sel]
        per_sensor[s] = {"bev": all_bev, "pose": all_pose}
        if verbose:
            print(f"[subset] {s}: kept {all_bev.shape[0]} keyframes")
    return per_sensor


class TripletDataset(Dataset):
    """Per-sensor triplets: anchor / positive (<5m) / negative (>10m)."""

    def __init__(self, per_sensor: dict, num_neg: int = 4,
                 pos_thresh: float = 5.0, neg_thresh: float = 10.0,
                 seed: int = 0):
        self.entries = []  # list of (sensor, anchor_idx, pos_choices, neg_choices)
        for sensor, d in per_sensor.items():
            bev = d["bev"]
            pose = d["pose"]
            xy = pose[:, :2, 3]
            n = bev.shape[0]
            for i in range(n):
                dists = np.linalg.norm(xy - xy[i], axis=1)
                pos = np.where((dists < pos_thresh) & (np.arange(n) != i))[0]
                neg = np.where(dists > neg_thresh)[0]
                if len(pos) == 0 or len(neg) < num_neg:
                    continue
                self.entries.append((sensor, i, pos, neg))
        self.per_sensor = per_sensor
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        sensor, ai, pos, neg = self.entries[idx]
        bev = self.per_sensor[sensor]["bev"]
        pi = int(self.rng.choice(pos))
        ni = self.rng.choice(neg, size=self.num_neg, replace=False).astype(np.int64)
        # Convert to 3-channel float32 [0, 1)
        a = self._to_tensor(bev[ai])
        p = self._to_tensor(bev[pi])
        ns = torch.stack([self._to_tensor(bev[k]) for k in ni], dim=0)
        return a, p, ns

    @staticmethod
    def _to_tensor(img_u8: np.ndarray) -> torch.Tensor:
        arr = img_u8.astype(np.float32) / 256.0
        return torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)


def collate_triplets(batch):
    a = torch.stack([b[0] for b in batch], dim=0)
    p = torch.stack([b[1] for b in batch], dim=0)
    n = torch.cat([b[2] for b in batch], dim=0)
    return a, p, n


def triplet_loss(q: torch.Tensor, p: torch.Tensor, ns: torch.Tensor,
                 margin: float = 0.3) -> torch.Tensor:
    """Hardest-negative triplet loss (matches official BEVPlace2 main.py)."""
    pos_d = (q - p).pow(2).sum().sqrt()
    neg_d = (q.unsqueeze(0) - ns).pow(2).sum(dim=1).sqrt()
    return F.relu(pos_d - neg_d + margin).max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_multi_dataset.yaml")
    parser.add_argument("--output", default="baselines/weights/bevplace_finetune.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-neg", type=int, default=4)
    parser.add_argument("--subset-per-sensor", type=int, default=7500)
    parser.add_argument("--keyframe-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("[1/4] Collecting train data + generating BEV images...")
    t0 = time.time()
    per_sensor = collect_subset(
        config, subset_per_sensor=args.subset_per_sensor,
        keyframe_stride=args.keyframe_stride, verbose=True,
    )
    total = sum(d["bev"].shape[0] for d in per_sensor.values())
    print(f"  Total: {total} keyframes across {len(per_sensor)} sensors "
          f"in {time.time()-t0:.1f}s")

    print("[2/4] Building triplet dataset...")
    ds = TripletDataset(per_sensor, num_neg=args.num_neg, seed=args.seed)
    print(f"  Triplets: {len(ds)}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, collate_fn=collate_triplets,
                        pin_memory=True)

    print("[3/4] Initializing BEVPlace++ from KITTI checkpoint...")
    from REIN import REIN  # vendored from BEVPlace2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = REIN().to(device)
    ckpt = torch.load(_WEIGHTS_PATH, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"  Loaded weights from {_WEIGHTS_PATH}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-3)

    print(f"[4/4] Fine-tuning for {args.epochs} epochs...")
    n_iter_per_epoch = len(loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        t_epoch = time.time()
        for batch_idx, (qb, pb, nb) in enumerate(loader):
            B = qb.shape[0]
            x = torch.cat([qb, pb, nb], dim=0).to(device, non_blocking=True)
            _, _, gd = model(x)
            gd_q, gd_p, gd_n = torch.split(gd, [B, B, nb.shape[0]])
            loss = 0.0
            for i in range(B):
                neg_slice = gd_n[i * args.num_neg:(i + 1) * args.num_neg]
                loss = loss + triplet_loss(gd_q[i], gd_p[i], neg_slice)
            loss = loss / B
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batch += 1
            if batch_idx % 200 == 0:
                print(f"  epoch {epoch} [{batch_idx}/{n_iter_per_epoch}] "
                      f"loss={loss.item():.4f}", flush=True)
        avg = epoch_loss / max(n_batch, 1)
        print(f"  ==> Epoch {epoch} done in {time.time()-t_epoch:.1f}s, "
              f"avg loss={avg:.4f}")

        # Save after each epoch (in case of interruption).
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                    "avg_loss": avg, "config": vars(args)}, args.output)
        print(f"  Saved checkpoint -> {args.output}")

    print("[done] Fine-tuning complete.")


if __name__ == "__main__":
    main()
