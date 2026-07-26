#!/usr/bin/env python3
"""Batched two-stage SeqOT training driver on OUR keyframe split.

Imports the official SeqOT modules UNCHANGED via sys.path
(_external/SeqOT: modules/seqTransformerCat.py featureExtracter,
modules/gem.py GeM, tools/loss.py triplet_loss) and trains them on the
range-image data produced by scripts/_seqot_prepare.py.

Stage 1 (--stage 1): featureExtracter (seqL=3, 32x900 -> 256D sub-descriptor),
  lazy-triplet loss exactly as upstream (tools/loss.py triplet_loss,
  margin 0.5, lazy=False — the same call training_seqot.py makes),
  3 positives + 3 negatives per anchor (upstream use_pos_num/use_neg_num).
Stage 2 (--stage 2): dump per-keyframe 256D sub-descriptors with the trained
  stage-1 model, then train GeM pooling over seqlen=20 sub-descriptor
  windows, 6 positives + 6 negatives per anchor (upstream values).
--stage both runs 1 then 2.

Deviations from the official pipeline (documented on purpose):
  * Batched driver: the stock loops (train/training_seqot.py,
    train/training_gem.py) process one anchor at a time and disk-load every
    range image each step. We load each sequence's range-image stack into
    RAM once per epoch and forward --anchors-per-step anchors jointly
    (batch = A * (1 + n_pos + n_neg) windows); the per-anchor loss is the
    unchanged upstream triplet_loss, averaged over the A anchors.
  * Training pairs come from the pose-distance surrogate index written by
    _seqot_prepare.py (positive < 10 m, negative > 50 m), not from overlap
    reprojection. overlap column is 1.0/0.0, upstream threshold 0.3 still
    separates them.
  * Per-sequence anchor sampling across all 24 train sequences with a cap
    (--anchors-per-seq, default 300) so a full run fits in <= 12 GPU-hours
    on one 16 GB GPU. Epoch defaults: --epochs1 15, --epochs2 10 (upstream
    hard-codes 100 epochs on a single NCLT sequence; our multi-sequence
    epoch sees 24 x 300 anchors).
  * seqL windows are clamped at sequence edges (upstream replicates the
    center frame when a neighbor file is missing).
  * lr defaults 5e-6 for both stages (training_seqot.py's own default;
    config.yml ships 5e-7 for stage 1 / 5e-6 for stage 2).

Checkpoints (upstream {'epoch','state_dict','optimizer'} format) go under
--run-dir (default _external/seqot_runs/<name>): seqot_epochN.pth.tar /
seqot_latest.pth.tar / gem_epochN.pth.tar / gem_latest.pth.tar, plus
subdesc/<seq>.npy stage-2 inputs and train_log.txt.

Example (full run, inside docker):
  python scripts/_seqot_train.py --run-dir _external/seqot_runs/full --stage both
Smoke:
  python scripts/_seqot_train.py --run-dir _external/seqot_runs/smoke \
      --sequences kitti_02 nclt_2012-05-11 --epochs1 1 --epochs2 1 \
      --anchors-per-seq 8 --anchors-per-step 2
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SEQOT = REPO / "_external" / "SeqOT"
sys.path.insert(0, str(SEQOT))

from modules.seqTransformerCat import featureExtracter  # noqa: E402 (unchanged)
from modules.gem import GeM                              # noqa: E402 (unchanged)

# SeqOT's tools/ is a namespace package; a regular `tools` package in the
# docker image's site-packages shadows it, so load loss.py by file path.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("_seqot_loss", SEQOT / "tools" / "loss.py")
PNV_loss = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(PNV_loss)                       # SeqOT original, unchanged

MARGIN = 0.5          # upstream MARGIN_1
POS1, NEG1 = 3, 3     # upstream training_seqot.py use_pos_num/use_neg_num
POS2, NEG2 = 6, 6     # upstream training_gem.py use_pos_num/use_neg_num
SEQL = 3              # upstream training_seqot seqlen
GEM_SEQLEN = 20       # upstream training_gem seqlen


def log(run_dir, msg):
    print(msg, flush=True)
    with open(run_dir / "train_log.txt", "a") as f:
        f.write(msg + "\n")


def find_train_seqs(data_root):
    return sorted(d.name for d in data_root.iterdir()
                  if d.is_dir() and (d / "train_index.npy").exists())


def load_range_stack(seq_dir):
    files = sorted(seq_dir.glob("[0-9]" * 6 + ".npy"))
    return np.stack([np.load(f) for f in files]).astype(np.float32)


def group_index(index, pos_need, neg_need):
    """index rows (anchor, other, overlap) -> {anchor: (pos_ids, neg_ids)},
    keeping only anchors with enough positives AND negatives."""
    pos, neg = {}, {}
    for a, o, ov in index:
        (pos if ov > 0.3 else neg).setdefault(int(a), []).append(int(o))
    return {a: (np.array(pos[a]), np.array(neg[a]))
            for a in pos
            if a in neg and len(pos[a]) >= pos_need and len(neg[a]) >= neg_need}


def window(center, seqlen, n):
    lo = center - seqlen // 2
    return np.clip(np.arange(lo, lo + seqlen), 0, n - 1)


def anchor_batches(groups, anchors_per_seq, anchors_per_step, rng):
    anchors = np.array(sorted(groups))
    if len(anchors) > anchors_per_seq:
        anchors = rng.choice(anchors, size=anchors_per_seq, replace=False)
    else:
        anchors = rng.permutation(anchors)
    for i in range(0, len(anchors), anchors_per_step):
        yield anchors[i:i + anchors_per_step]


def train_windows(chunk, groups, pos_n, neg_n, seqlen, n, rng):
    """Window index matrix (A*(1+pos_n+neg_n), seqlen) for one step."""
    rows = []
    for a in chunk:
        p_ids, n_ids = groups[a]
        picks = [a] + list(rng.choice(p_ids, pos_n, replace=False)) \
                    + list(rng.choice(n_ids, neg_n, replace=False))
        rows += [window(i, seqlen, n) for i in picks]
    return np.stack(rows)


def triplet_step(model, optimizer, batch, n_anchor, pos_n, neg_n):
    model.train()
    optimizer.zero_grad()
    des = model(batch)                       # (A*(1+P+N), 256)
    des = des.view(n_anchor, 1 + pos_n + neg_n, -1)
    loss = torch.stack([
        PNV_loss.triplet_loss(des[a, 0:1], des[a, 1:1 + pos_n],
                              des[a, 1 + pos_n:], MARGIN, lazy=False)
        for a in range(n_anchor)]).mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run_stage1(args, run_dir, device, rng):
    model = featureExtracter(seqL=SEQL).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    for epoch in range(args.epochs1):
        t0, losses = time.time(), []
        for seq in rng.permutation(args.sequences):
            seq_dir = args.data_root / seq
            stack = load_range_stack(seq_dir)
            n = len(stack)
            groups = group_index(np.load(seq_dir / "train_index.npy"),
                                 POS1, NEG1)
            if not groups:
                log(run_dir, f"  [s1 e{epoch}] {seq}: no usable anchors, skip")
                continue
            for chunk in anchor_batches(groups, args.anchors_per_seq,
                                        args.anchors_per_step, rng):
                w = train_windows(chunk, groups, POS1, NEG1, SEQL, n, rng)
                batch = torch.from_numpy(stack[w]).to(device)  # (B, seqL, 32, 900)
                losses.append(triplet_step(model, optimizer, batch,
                                           len(chunk), POS1, NEG1))
            del stack
        scheduler.step()
        dt = time.time() - t0
        log(run_dir, f"[stage1] epoch {epoch}: loss {np.mean(losses):.4f} "
                     f"({len(losses)} steps, {dt:.1f}s, {dt / max(len(losses), 1):.2f}s/step)")
        state = {"epoch": epoch, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict()}
        torch.save(state, run_dir / f"seqot_epoch{epoch}.pth.tar")
        torch.save(state, run_dir / "seqot_latest.pth.tar")
    return model


def load_feature_extracter(ckpt_path, device):
    model = featureExtracter(seqL=SEQL).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model


def gen_subdescriptors(model, sequences, data_root, out_dir, device, batch_size,
                       run_dir):
    """256D sub-descriptor per keyframe (seqL=3 window, clamped), batched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for seq in sequences:
        stack = load_range_stack(data_root / seq)
        n = len(stack)
        des = np.zeros((n, 256), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, n, batch_size):
                idx = np.arange(i, min(i + batch_size, n))
                w = np.stack([window(j, SEQL, n) for j in idx])
                batch = torch.from_numpy(stack[w]).to(device)
                des[idx] = model(batch).cpu().numpy()
        np.save(out_dir / f"{seq}.npy", des)
        log(run_dir, f"  [subdesc] {seq}: {n} frames in {time.time() - t0:.1f}s")
        del stack


def run_stage2(args, run_dir, device, rng):
    if args.seqot_checkpoint is not None:
        ckpt = Path(args.seqot_checkpoint)
    else:
        ckpt = run_dir / "seqot_latest.pth.tar"
        if not ckpt.exists():
            raise SystemExit(f"stage 2 needs a stage-1 checkpoint: {ckpt} "
                             f"missing and --seqot-checkpoint not given")
    log(run_dir, f"[stage2] sub-descriptors from {ckpt}")
    fe = load_feature_extracter(ckpt, device)
    sub_dir = run_dir / "subdesc"
    gen_subdescriptors(fe, args.sequences, args.data_root, sub_dir, device,
                       args.infer_batch, run_dir)
    del fe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = GeM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    for epoch in range(args.epochs2):
        t0, losses = time.time(), []
        for seq in rng.permutation(args.sequences):
            seq_dir = args.data_root / seq
            descs = np.load(sub_dir / f"{seq}.npy")
            n = len(descs)
            groups = group_index(np.load(seq_dir / "train_index.npy"),
                                 POS2, NEG2)
            if not groups:
                continue
            for chunk in anchor_batches(groups, args.anchors_per_seq,
                                        args.anchors_per_step * 8, rng):
                w = train_windows(chunk, groups, POS2, NEG2, GEM_SEQLEN, n, rng)
                batch = torch.from_numpy(descs[w]).to(device)  # (B, 20, 256)
                model.train()
                optimizer.zero_grad()
                out = model(batch).squeeze(1)                  # (B, 256)
                out = out.view(len(chunk), 1 + POS2 + NEG2, -1)
                loss = torch.stack([
                    PNV_loss.triplet_loss(out[a, 0:1], out[a, 1:1 + POS2],
                                          out[a, 1 + POS2:], MARGIN, lazy=False)
                    for a in range(len(chunk))]).mean()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        scheduler.step()
        dt = time.time() - t0
        log(run_dir, f"[stage2] epoch {epoch}: loss {np.mean(losses):.4f} "
                     f"(p={float(model.p.item()):.3f}, {len(losses)} steps, {dt:.1f}s)")
        state = {"epoch": epoch, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict()}
        torch.save(state, run_dir / f"gem_epoch{epoch}.pth.tar")
        torch.save(state, run_dir / "gem_latest.pth.tar")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(REPO / "_external/seqot_data"))
    ap.add_argument("--run-dir", required=True,
                    help="checkpoint/log dir, e.g. _external/seqot_runs/full")
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    ap.add_argument("--sequences", nargs="+", default=None,
                    help="train dirs under data-root (default: all with train_index.npy)")
    ap.add_argument("--epochs1", type=int, default=15)
    ap.add_argument("--epochs2", type=int, default=10)
    ap.add_argument("--anchors-per-seq", type=int, default=300)
    ap.add_argument("--anchors-per-step", type=int, default=4)
    ap.add_argument("--lr1", type=float, default=5e-6)
    ap.add_argument("--lr2", type=float, default=5e-6)
    ap.add_argument("--seqot-checkpoint", default=None,
                    help="stage-1 checkpoint for --stage 2 (default: run_dir/seqot_latest.pth.tar)")
    ap.add_argument("--infer-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.sequences is None:
        args.sequences = find_train_seqs(args.data_root)
    if not args.sequences:
        ap.error(f"no train sequences (train_index.npy) under {args.data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    log(run_dir, f"SeqOT train: stage={args.stage} device={device} "
                 f"seqs={args.sequences}")
    (run_dir / "args.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2))

    if args.stage in ("1", "both"):
        run_stage1(args, run_dir, device, rng)
    if args.stage in ("2", "both"):
        run_stage2(args, run_dir, device, rng)
    log(run_dir, "done")


if __name__ == "__main__":
    main()
