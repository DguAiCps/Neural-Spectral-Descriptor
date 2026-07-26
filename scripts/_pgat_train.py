#!/usr/bin/env python3
"""
P-GAT baseline training driver on OUR split.

Thin replacement for _external/P-GAT/training/train.py that imports their model
(PoseGAT) and loss weighting (engine.trainer.weight_loss) untouched, and
replicates their training loop on the tensors built by
scripts/_pgat_make_tensors.py. A separate driver is needed because:

  - their data path expects one dense global adjacency per dataset dir; ours is
    recomputed per batch from stored 2D positions (identical values, see
    _pgat_common.block_scores_switches);
  - their "validation" phase is broken (it re-runs do_train with optimizer
    steps on the train loader); here HAS_VALIDATION runs a real no-grad pass
    over held-out sequences;
  - their AttentionalGraph.forward mutates its input in place, which breaks
    autograd on current PyTorch; training uses the functionally-identical
    out-of-place replication in _pgat_common.pgat_streams, verified against the
    native forward at startup (no_grad).

Faithfully replicated semantics:
  - one epoch = one shuffled pass over all train subgraph indices (their
    DataLoader over TensorDataset(subgraph_idx, dataset_id); here
    dataset_id == sequence);
  - random_pairing: with prob POS_RATE pick a positive from paired.json, else a
    uniformly random other subgraph of the same sequence (their negative branch
    can still sample an overlapping subgraph; GT labels stay correct);
  - BCELoss(reduction='none') on scores in (0, 1), weighted by their
    weight_loss (switch matrix zeroes the 10-50 m band; padding masked;
  TRAIN.WEIGHT is carried but unused - their weighting code is commented out);
  - Adam with SOLVER.LR, no weight decay (their train.py never passes WD);
  - batches whose pose tensor contains NaN are skipped (their guard);
  - checkpoints named attentional_graph<epoch>.pt / attentional_graph.pt.

Config: yacs yml (configs/_pgat_train.yml / configs/_pgat_smoke.yml), same key
names as their attentional_graph/config_init/defaults.py where applicable.
DATA_DIR is a single directory (our tensor layout), not a list.

Usage (inside the pyg container):
  python scripts/_pgat_train.py --config_file configs/_pgat_train.yml
  python scripts/_pgat_train.py --config_file configs/_pgat_smoke.yml \
      SOLVER.EPOCH 2 SOLVER.BATCHSIZE 16
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from yacs.config import CfgNode as CN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pgat_common import (REPO, build_model, pgat_forward,  # noqa: E402
                          verify_against_native, gather_subgraph_features,
                          block_scores_switches)
from attentional_graph.engine.trainer import weight_loss  # noqa: E402


def default_config():
    c = CN()
    c.DATA_DIR = "_external/pgat_data/"
    c.OUTPUT_DIR = "_external/pgat_runs/run/"
    c.SAVE_MODEL = True
    c.SEED = 0
    c.MODEL = CN()
    c.MODEL.ATTENTION_GRAPH = CN()
    c.MODEL.ATTENTION_GRAPH.POSE_DIM = 2
    c.MODEL.ATTENTION_GRAPH.FEATURE_DIM = 544
    c.MODEL.ATTENTION_GRAPH.KEYPOINT_HIDDEN_DIM = [32, 64, 128]
    c.MODEL.ATTENTION_GRAPH.NUM_HEADS = 4
    c.MODEL.ATTENTION_GRAPH.NUM_LAYERS = 5
    c.MODEL.ATTENTION_GRAPH.INCLUDE_POSE = True
    c.MODEL.ATTENTION_GRAPH.DROPOUT = 0.0
    c.SOLVER = CN()
    c.SOLVER.EPOCH = 100
    c.SOLVER.LR = 1e-4
    c.SOLVER.BATCHSIZE = 128
    c.TRAIN = CN()
    c.TRAIN.SAVE_INTERVAL = 25
    c.TRAIN.WEIGHT = [1.0, 0.0]
    c.TRAIN.POS_RATE = 0.3
    c.TRAIN.HAS_VALIDATION = True
    c.TRAIN.MODEL_PARAM = ""
    return c


class SplitData:
    """Per-split (train/val) tensors, sequence == P-GAT dataset_id."""

    def __init__(self, data_dir, split):
        split_dir = os.path.join(data_dir, split)
        with open(os.path.join(split_dir, "paired.json")) as f:
            paired_raw = json.load(f)
        self.names = sorted(paired_raw.keys())
        self.features, self.xy = [], []
        self.sub_nodes, self.sub_poses, self.sub_masks = [], [], []
        self.paired = []
        for name in self.names:
            z = np.load(os.path.join(split_dir, f"{name}.npz"))
            self.features.append(torch.from_numpy(z["features"]))
            self.xy.append(torch.from_numpy(z["xy"]))
            self.sub_nodes.append(torch.from_numpy(z["sub_nodes"]))
            self.sub_poses.append(torch.from_numpy(z["sub_poses"]))
            self.sub_masks.append(torch.from_numpy(z["sub_masks"]))
            self.paired.append({int(k): v for k, v in paired_raw[name].items()})
        self.num_subgraphs = [len(n) for n in self.sub_nodes]

    def index_loader(self, batch_size, shuffle):
        sub_idx, seq_idx = [], []
        for sid, count in enumerate(self.num_subgraphs):
            sub_idx.extend(range(count))
            seq_idx.extend([sid] * count)
        return DataLoader(
            TensorDataset(torch.tensor(sub_idx), torch.tensor(seq_idx)),
            batch_size=batch_size, shuffle=shuffle, num_workers=0,
        )


def random_pairing(split, index_batch, pos_rate, rng):
    """Replication of engine.trainer.random_pairing on our layout."""
    target_index = []
    for i in range(index_batch[0].shape[0]):
        q = int(index_batch[0][i])
        sid = int(index_batch[1][i])
        positives = split.paired[sid].get(q, [])
        if rng.uniform(0, 1) < pos_rate and len(positives) > 0:
            target_index.append(rng.choice(positives))
        else:
            t = q
            while t == q:
                t = rng.randrange(split.num_subgraphs[sid])
            target_index.append(t)
    return torch.tensor(target_index)


def create_pairs(split, index_batch, pos_rate, pos_thresh, neg_thresh, rng):
    """Replication of engine.trainer.create_pairs (batch assembly)."""
    q_sub, seq_ids = index_batch
    t_sub = random_pairing(split, index_batch, pos_rate, rng)
    feats, poses, masks, scores, switches = [], [], [], [], []
    for i in range(len(q_sub)):
        sid = int(seq_ids[i])
        q, t = int(q_sub[i]), int(t_sub[i])
        nodes = torch.stack([split.sub_nodes[sid][q], split.sub_nodes[sid][t]])
        msk = torch.stack([split.sub_masks[sid][q], split.sub_masks[sid][t]])
        feats.append(gather_subgraph_features(split.features[sid], nodes, msk))
        poses.append(torch.stack([split.sub_poses[sid][q],
                                  split.sub_poses[sid][t]]))
        masks.append(msk)
        sc, sw = block_scores_switches(
            split.xy[sid], nodes[0:1], msk[0:1], nodes[1:2], msk[1:2],
            pos_thresh, neg_thresh)
        scores.append(sc[0])
        switches.append(sw[0])
    return (torch.stack(feats), torch.stack(poses), torch.stack(scores),
            torch.stack(masks), torch.stack(switches))


def run_epoch(cfg, model, split, loader, meta, criterion, device,
              optimizer=None, rng=None):
    """One pass over `loader`. optimizer=None -> no-grad validation pass."""
    training = optimizer is not None
    model.train() if training else model.eval()
    losses = []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for index_batch in loader:
            features, poses, scores, masks, switches = create_pairs(
                split, index_batch, cfg.TRAIN.POS_RATE,
                meta["pos_thresh"], meta["neg_thresh"], rng)
            if torch.isnan(poses).any():
                continue  # their NaN-pose guard
            features = features.to(device)
            poses = poses.to(device)
            masks = masks.to(device)
            scores = scores.to(device)
            switches = switches.to(device)
            if training:
                optimizer.zero_grad()
            output, _, _ = pgat_forward(model, features, poses, masks)
            # numerical guard: fp32 roundoff in the cosine can leave the
            # (0, 1) band their /2.0001 shift is meant to guarantee; BCELoss
            # asserts on inputs outside [0, 1]. No-op except on such elements.
            output = output.clamp(1e-6, 1.0 - 1e-6)
            loss = criterion(output.flatten(start_dim=1),
                             scores.flatten(start_dim=1))
            loss = weight_loss(loss, scores, cfg.TRAIN.WEIGHT, masks,
                               switches, device)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser(description="P-GAT baseline training driver")
    ap.add_argument("--config_file", default="configs/_pgat_train.yml")
    ap.add_argument("opts", nargs=argparse.REMAINDER, default=None,
                    help="yacs-style KEY VALUE overrides")
    args = ap.parse_args()

    cfg = default_config()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    train_rng = random.Random(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_dir = cfg.DATA_DIR if os.path.isabs(cfg.DATA_DIR) \
        else os.path.join(REPO, cfg.DATA_DIR)
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    assert meta["feature_dim"] == cfg.MODEL.ATTENTION_GRAPH.FEATURE_DIM, \
        (meta["feature_dim"], cfg.MODEL.ATTENTION_GRAPH.FEATURE_DIM)

    train_data = SplitData(data_dir, "train")
    train_loader = train_data.index_loader(cfg.SOLVER.BATCHSIZE, shuffle=True)
    print(f"train: {len(train_data.names)} sequences, "
          f"{sum(train_data.num_subgraphs)} subgraphs")
    val_data = None
    if cfg.TRAIN.HAS_VALIDATION:
        val_data = SplitData(data_dir, "val")
        val_loader = val_data.index_loader(cfg.SOLVER.BATCHSIZE, shuffle=False)
        print(f"val:   {len(val_data.names)} sequences, "
              f"{sum(val_data.num_subgraphs)} subgraphs")

    ag = cfg.MODEL.ATTENTION_GRAPH
    model = build_model(ag.POSE_DIM, ag.FEATURE_DIM, ag.KEYPOINT_HIDDEN_DIM,
                        ag.INCLUDE_POSE, ag.NUM_HEADS, ag.NUM_LAYERS,
                        ag.DROPOUT).to(device)
    if cfg.TRAIN.MODEL_PARAM:
        model.load_state_dict(torch.load(cfg.TRAIN.MODEL_PARAM,
                                         map_location=device))
        print(f"resumed from {cfg.TRAIN.MODEL_PARAM}")
    criterion = torch.nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    # verify the out-of-place forward replication against their native forward
    probe_batch = next(iter(train_loader))
    f, p, _, m, _ = create_pairs(train_data, probe_batch, cfg.TRAIN.POS_RATE,
                                 meta["pos_thresh"], meta["neg_thresh"],
                                 random.Random(0))
    ok, diff = verify_against_native(model, f.to(device), p.to(device),
                                     m.to(device))
    print(f"forward replication check vs native PoseGAT.forward: "
          f"max|diff|={diff:.2e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise RuntimeError("replicated forward does not match native forward")

    out_dir = cfg.OUTPUT_DIR if os.path.isabs(cfg.OUTPUT_DIR) \
        else os.path.join(REPO, cfg.OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yml"), "w") as f_out:
        f_out.write(cfg.dump())
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f_out:
        json.dump({"data_meta": meta, "config_file": args.config_file}, f_out,
                  indent=2)

    curve_path = os.path.join(out_dir, "loss_curve.txt")
    with open(curve_path, "w") as f_out:
        f_out.write("epoch, train_loss" +
                    (", val_loss\n" if cfg.TRAIN.HAS_VALIDATION else "\n"))

    for epoch in range(1, cfg.SOLVER.EPOCH + 1):
        train_loss = run_epoch(cfg, model, train_data, train_loader, meta,
                               criterion, device, optimizer=optimizer,
                               rng=train_rng)
        line = f"{epoch}, {train_loss:.6f}"
        if cfg.TRAIN.HAS_VALIDATION:
            # epoch-stable pairing RNG so the val loss is comparable across epochs
            val_loss = run_epoch(cfg, model, val_data, val_loader, meta,
                                 criterion, device, optimizer=None,
                                 rng=random.Random(10_000 + epoch))
            line += f", {val_loss:.6f}"
        print(f"epoch {line}")
        with open(curve_path, "a") as f_out:
            f_out.write(line + "\n")
        if epoch % cfg.TRAIN.SAVE_INTERVAL == 0:
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"attentional_graph{epoch}.pt"))
    if cfg.SAVE_MODEL:
        torch.save(model.state_dict(),
                   os.path.join(out_dir, "attentional_graph.pt"))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
