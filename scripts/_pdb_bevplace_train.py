#!/usr/bin/env python3
"""pd2026 baseline retrain: BEVPlace++ on the place-disjoint TRAIN split only.

Inherits the paper-era fine-tune recipe (scripts/finetune_bevplace.py):
  * init from the official KITTI checkpoint (baselines/weights/bevplace_kitti.pth.tar)
  * BEV 200x200 uint8 (max-z, +/-50 m, z in [-3, 5]), scan stride 5
  * triplets: positive < 5 m, negatives > 10 m, 4 random negatives,
    hardest-negative triplet loss margin 0.3
  * Adam lr 1e-5, weight_decay 1e-3, batch 4, 5 epochs, subset 7500/sensor

Differences from the paper-era script (both deliberate):
  1. Train data = configs/training_place_disjoint.yaml TRAIN split ONLY
     (kitti 01/02/06/07, helipr Bridge01-03+Roundabout01-02, mulran
     Sejong01/02, nclt 5 sessions). No test-place sequences.
  2. Triplet pos/neg pools are restricted to the SAME sequence. The paper-era
     script pooled per sensor across sequences, but KITTI odometry frames are
     per-sequence local and the MulRan loader re-centers each sequence on its
     first scan, so cross-sequence coordinates are not comparable (false
     positives/negatives). Same-sequence pooling removes that noise.
  3. Early stop / checkpoint selection on the pd VAL split (kitti 09,
     nclt 2012-01-08, helipr Roundabout03, mulran Sejong03; keyframes from the
     committed pd2026 val caches): after each epoch, mean intra-session R@1
     (5 m / skip 30) is computed and the best-val epoch is kept.

Usage (inside docker ring_baseline):
    python3 scripts/_pdb_bevplace_train.py \
        --output baselines/weights/bevplace_pd2026.pth

Smoke (no GPU-hours):
    python3 scripts/_pdb_bevplace_train.py --max-scans-per-seq 200 \
        --subset-per-sensor 150 --max-steps 20 --skip-val --epochs 1 \
        --output /tmp/smoke_bevplace_pd.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts"),
                str(REPO / "baselines" / "_bevplace_official")]

# Inherited paper-era pieces.
from finetune_bevplace import collate_triplets, create_loader, triplet_loss  # noqa: E402
from baselines.bevplace import _bev_image  # noqa: E402

CONFIG = REPO / "configs/training_place_disjoint.yaml"
INIT_WEIGHTS = REPO / "baselines/weights/bevplace_kitti.pth.tar"

# pd VAL split -> committed pd2026 val caches (keyframe roster identical to
# the NSD pd stack). Keyed by (dataset_type, sequence).
VAL_CACHES = {
    ("kitti", "09"): "data/preprocessed_pd2026/cache_fecdaef7_kitti_val_09.npz",
    ("nclt", "2012-01-08"): "data/preprocessed_pd2026/cache_aff53e59_nclt_val_2012-01-08.npz",
    ("helipr", "Roundabout03"): "data/preprocessed_pd2026/cache_ad22d735_helipr_val_Roundabout03.npz",
    ("mulran", "Sejong03"): "data/preprocessed_pd2026/cache_50223b37_mulran_val_Sejong03.npz",
}


def collect_pd_train(config: dict, subset_per_sensor: int,
                     keyframe_stride: int, max_scans_per_seq: int | None,
                     seed: int) -> list:
    """Walk the pd TRAIN split; return per-sequence groups of BEV + xy.

    Mirrors finetune_bevplace.collect_subset (stride, uint8 BEV channel) but
    keeps sequences separate so triplets never cross sequence frames.
    """
    rng = np.random.default_rng(seed)
    groups = []
    for ds in config["data"]["datasets"]["train"]:
        for seq in ds["sequences"]:
            t0 = time.time()
            try:
                loader = create_loader(ds["type"], ds["root"], seq)
            except Exception as e:
                print(f"[skip] {ds['type']}/{seq}: {e}")
                continue
            idxs = list(range(0, len(loader), keyframe_stride))
            if max_scans_per_seq is not None:
                idxs = idxs[:max_scans_per_seq]
            bevs, xys = [], []
            for idx in idxs:
                d = loader[idx]
                ch = (_bev_image(d["points"])[0] * 256).clip(0, 255).astype(np.uint8)
                bevs.append(ch)
                xys.append(np.asarray(d["pose"], dtype=np.float64)[:2, 3])
            groups.append({"sensor": ds["type"], "seq": seq,
                           "bev": np.stack(bevs), "xy": np.stack(xys)})
            print(f"  [{ds['type']}/{seq}] {len(bevs)} frames "
                  f"in {time.time() - t0:.1f}s", flush=True)

    # Per-sensor cap (inherited 7500/sensor), applied proportionally per seq.
    for sensor in sorted({g["sensor"] for g in groups}):
        sensor_groups = [g for g in groups if g["sensor"] == sensor]
        total = sum(g["bev"].shape[0] for g in sensor_groups)
        if total <= subset_per_sensor:
            print(f"[subset] {sensor}: kept all {total}")
            continue
        for g in sensor_groups:
            n = g["bev"].shape[0]
            keep = max(2, int(round(subset_per_sensor * n / total)))
            if keep < n:
                sel = np.sort(rng.choice(n, size=keep, replace=False))
                g["bev"], g["xy"] = g["bev"][sel], g["xy"][sel]
        print(f"[subset] {sensor}: {total} -> "
              f"{sum(g['bev'].shape[0] for g in sensor_groups)}")
    return groups


class PDTripletDataset(Dataset):
    """Same-sequence triplets: anchor / positive (<5 m) / negatives (>10 m)."""

    def __init__(self, groups: list, num_neg: int = 4, pos_thresh: float = 5.0,
                 neg_thresh: float = 10.0, seed: int = 0):
        self.groups = groups
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)
        self.entries = []  # (group_idx, anchor_idx, pos_idx[], neg_idx[])
        for gi, g in enumerate(groups):
            xy = g["xy"]
            n = xy.shape[0]
            for i in range(n):
                dists = np.linalg.norm(xy - xy[i], axis=1)
                pos = np.where((dists < pos_thresh) & (np.arange(n) != i))[0]
                neg = np.where(dists > neg_thresh)[0]
                if len(pos) == 0 or len(neg) < num_neg:
                    continue
                self.entries.append((gi, i, pos.astype(np.int32),
                                     neg.astype(np.int32)))

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def _to_tensor(img_u8: np.ndarray) -> torch.Tensor:
        arr = img_u8.astype(np.float32) / 256.0
        return torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)

    def __getitem__(self, idx):
        gi, ai, pos, neg = self.entries[idx]
        bev = self.groups[gi]["bev"]
        pi = int(self.rng.choice(pos))
        ni = self.rng.choice(neg, size=self.num_neg, replace=False)
        a = self._to_tensor(bev[ai])
        p = self._to_tensor(bev[pi])
        ns = torch.stack([self._to_tensor(bev[int(k)]) for k in ni], dim=0)
        return a, p, ns


def precompute_val_bevs(config: dict) -> list:
    """BEV stacks + poses for the pd VAL keyframe rosters (from val caches)."""
    val_sets = []
    for ds in config["data"]["datasets"]["val"]:
        for seq in ds["sequences"]:
            cache_rel = VAL_CACHES.get((ds["type"], seq))
            if cache_rel is None:
                print(f"[val skip] no pd cache for {ds['type']}/{seq}")
                continue
            with np.load(REPO / cache_rel, allow_pickle=False) as c:
                scan_ids = c["scan_ids"].astype(np.int64)
                poses = c["poses"].astype(np.float64)
            loader = create_loader(ds["type"], ds["root"], seq)
            bevs = np.stack([
                (_bev_image(loader[int(s)]["points"])[0] * 256)
                .clip(0, 255).astype(np.uint8) for s in scan_ids])
            val_sets.append({"name": f"{ds['type']}/{seq}", "bev": bevs,
                             "poses": poses})
            print(f"[val] {ds['type']}/{seq}: {len(scan_ids)} keyframes",
                  flush=True)
    return val_sets


@torch.no_grad()
def val_recall_at_1(model, val_sets: list, device: str,
                    batch_size: int = 8) -> dict:
    from baselines.eval_utils import compute_recall_multi_k
    model.eval()
    per_seq = {}
    for vs in val_sets:
        descs = []
        bev = vs["bev"]
        for s in range(0, bev.shape[0], batch_size):
            chunk = bev[s:s + batch_size].astype(np.float32) / 256.0
            x = torch.from_numpy(chunk).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            _, _, gd = model(x)
            descs.append(gd.cpu().numpy())
        descs = np.concatenate(descs, axis=0).astype(np.float32)
        recalls, n_q = compute_recall_multi_k(
            descs, vs["poses"], k_values=[1], distance_threshold=5.0,
            skip_frames=30)
        per_seq[vs["name"]] = {"r1": float(recalls[1]), "n_queries": int(n_q)}
    model.train()
    valid = [v["r1"] for v in per_seq.values() if v["n_queries"] > 0]
    return {"mean_r1": float(np.mean(valid)) if valid else 0.0,
            "per_seq": per_seq}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output",
                        default=str(REPO / "baselines/weights/bevplace_pd2026.pth"))
    parser.add_argument("--init-weights", default=str(INIT_WEIGHTS),
                        help="official KITTI checkpoint (paper-era recipe), "
                             "or 'imagenet' for no checkpoint init")
    # Inherited paper-era hyperparameters.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-neg", type=int, default=4)
    parser.add_argument("--subset-per-sensor", type=int, default=7500)
    parser.add_argument("--keyframe-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    # pd additions.
    parser.add_argument("--patience", type=int, default=0,
                        help="stop after N epochs without val improvement "
                             "(0 = run all epochs, keep best-val checkpoint)")
    # Smoke-test controls.
    parser.add_argument("--max-scans-per-seq", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-val", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("[1/4] Collecting pd TRAIN split + generating BEV images...")
    t0 = time.time()
    groups = collect_pd_train(config, args.subset_per_sensor,
                              args.keyframe_stride, args.max_scans_per_seq,
                              args.seed)
    total = sum(g["bev"].shape[0] for g in groups)
    print(f"  Total: {total} frames across {len(groups)} sequences "
          f"in {time.time() - t0:.1f}s")

    print("[2/4] Building same-sequence triplet dataset...")
    ds = PDTripletDataset(groups, num_neg=args.num_neg, seed=args.seed)
    print(f"  Triplets: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, collate_fn=collate_triplets,
                        pin_memory=True)

    val_sets = [] if args.skip_val else precompute_val_bevs(config)

    print("[3/4] Initializing BEVPlace++...")
    from REIN import REIN  # vendored from BEVPlace2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = REIN().to(device)
    if args.init_weights != "imagenet":
        ckpt = torch.load(args.init_weights, map_location=device,
                          weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded init weights from {args.init_weights}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-3)

    print(f"[4/4] Training for up to {args.epochs} epochs...")
    best = {"epoch": 0, "val_mean_r1": -1.0}
    epochs_since_best = 0
    step_times = []
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batch = 0.0, 0
        t_epoch = time.time()
        for batch_idx, (qb, pb, nb) in enumerate(loader):
            t_step = time.perf_counter()
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
            step_times.append(time.perf_counter() - t_step)
            epoch_loss += float(loss.item())
            n_batch += 1
            if batch_idx % 20 == 0 or (args.max_steps and n_batch <= args.max_steps):
                print(f"  epoch {epoch} [{batch_idx}/{len(loader)}] "
                      f"loss={loss.item():.4f} "
                      f"({step_times[-1]:.2f}s/step)", flush=True)
            if args.max_steps and n_batch >= args.max_steps:
                print(f"  [smoke] stopping after {n_batch} steps; "
                      f"mean {np.mean(step_times):.3f}s/step")
                stop = True
                break
        avg = epoch_loss / max(n_batch, 1)
        print(f"  ==> Epoch {epoch} done in {time.time() - t_epoch:.1f}s, "
              f"avg loss={avg:.4f}")

        val = val_recall_at_1(model, val_sets, device) if val_sets else None
        if val:
            print(f"  val mean R@1={val['mean_r1']:.4f} "
                  f"{json.dumps(val['per_seq'])}", flush=True)

        state = {"state_dict": model.state_dict(), "epoch": epoch,
                 "avg_loss": avg, "val": val, "config": vars(args)}
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        torch.save(state, args.output + ".last")
        val_metric = val["mean_r1"] if val else -avg  # no-val: lowest loss
        if val_metric > best["val_mean_r1"]:
            best = {"epoch": epoch, "val_mean_r1": val_metric}
            epochs_since_best = 0
            torch.save(state, args.output)
            print(f"  Saved best checkpoint (epoch {epoch}) -> {args.output}")
        else:
            epochs_since_best += 1
            if args.patience and epochs_since_best >= args.patience:
                print(f"  Early stop: no val improvement for {args.patience} epochs")
                stop = True
        if stop:
            break

    print(f"[done] best epoch {best['epoch']} "
          f"(val mean R@1 {best['val_mean_r1']:.4f}); "
          f"mean step time {np.mean(step_times):.3f}s "
          f"over {len(step_times)} steps")


if __name__ == "__main__":
    main()
