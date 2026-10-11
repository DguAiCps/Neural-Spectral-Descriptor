#!/usr/bin/env python3
"""BEVPlace++ on split v2 -- part 2: training (2026-10-11).

Official code (_external/BEVPlace2, commit in _bevplace_official_bev.OFFICIAL_COMMIT)
imported unchanged: REIN / REM / NetVLAD (REIN.py), main.getClusters +
NetVLAD.init_params (k-means: 400 training images x 25 local features, faiss
Kmeans 64, niter 100), main.TripletLoss (margin 0.3, max over 10 negatives,
/batchSize), main.infer, main.saveCheckpoint, main.get_args, and the
kitti_dataset.TrainingDataset rules (positive < 5 m excluding the query,
random positive, 10 negatives, hardest-10 selection from a per-epoch feature
cache), kitti_dataset.collate_fn, nclt_dataset.InferDataset.

PAPER OVERRIDES (project rule 2026-10-11: where the paper and the code differ
the paper value is used; BEVPlace++ T-RO 2025 = arXiv 2408.01841v3, and
BEVPlace ICCV'23 = arXiv 2302.14325 where the T-RO text is silent):
  P1 optimizer AdamW, weight decay 1e-3 (T-RO Sec. V "Implementation
     details")                    -- overrides main.py:310-311 (Adam, no decay;
     --weightDecay is parsed at main.py:43 but never used)
  P2 lr x0.5 every 10 epochs (same)      -- main.py has no scheduler
     (--lrStep 10 / --lrGamma 0.5 parsed at main.py:41-42, never used)
  P3 50 epochs (same)                    -- main.py:38 default 40
  P4 negatives = all frames that are not positives, i.e. >= 5 m (T-RO
     Sec. IV-D "its negative samples are the other frames")
                                         -- kitti_dataset.py:206,221 (> 7 m)
  P5 hard mining only after the first 10 epochs (T-RO Sec. IV-D)
                                         -- main.py:87 (`if epoch>=0`, every
     epoch); epochs 0-9 use 10 random negatives (the code's non-mining
     sampling, kitti_dataset.py:258-259) and the code's training positive
     rule (random positive, :243)
  P6 rotation augmentation of the POINT CLOUD about z, angle uniform in
     [-pi, pi) (ICCV Sec. 5; T-RO Sec. VI-A "randomly rotating the point
     clouds")                            -- kitti_dataset.py:263-286 rotate the
     BEV image by an integer angle 0..359 with cv2.warpAffine.  Each training
     image is regenerated from the stored points (official generator,
     vectorised getBEV checked equal to the official one) and quantised to
     uint8 as the PNG the official loader reads.
  P7 200x200 BEV image, bilinear feature up-sampling, NetVLAD soft-assignment
     bias (see _bevplace_official_bev: crop_paper, paper_rem_class,
     add_netvlad_bias)                   -- gen_bev_images.py:28-36,
     REIN.py:132,137, REIN.py:23,37
Project settings (not paper): seed 1 (main.py:47 default 1024).
Protocol replacements (unchanged from the first version of this file):
  * data: all split-v2 TRAIN keyframes, positive/negative pools inside each
    sequence (main.py:275/303 use KITTI 00, first 3000 frames);
  * per-epoch validation / model_best: split-v2 VAL, intra-sequence 5 m R@1
    with a 30-keyframe gap, sensor families weighted equally
    (_pdv2_c3._family_pooled), strictly-higher rule of main.py:351
    (main.py:322-360 validate on KITTI 00-08 and NCLT = v2 test places).
Engineering (no change in what is computed): negatives computed on access,
query-to-negatives distances in row chunks (bit-identical), the feature-cache
and k-means passes read only the query image, faiss validation search in
blocks of 19 queries (bit-identical to the per-query loop), --resume.

Usage (inside ring_baseline; official flags pass through to main.get_args):
  python3 scripts/_bevplace_official_train.py \
      --runsPath _external/bevplace_official_pdv2/runs/ \
      --cachePath _external/bevplace_official_pdv2/cache/
  smoke:  ... --smoke 200 --nEpochs 2 --hard-mining-start 1 --runsPath .../smoke/runs/ \
              --cachePath .../smoke/cache/ --selection-out .../smoke/train_selection.json
  resume: ... --resume _external/bevplace_official_pdv2/runs/<Mon DD_HH-MM-SS>
  CPU check: python3 scripts/_bevplace_official_train.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from os.path import isfile, join
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from _bevplace_official_bev import (  # noqa: E402
    BEV, IMG, OFFICIAL_COMMIT, POINTS, UPSAMPLE_MODE, add_netvlad_bias, build_model,
    cache_path, load_official, paper_rem_class, png_u8, rotated_bev, seq_name, sha256)
import _bevplace_official_bev as BB  # noqa: E402
from _pdv2_pilot import TRAIN, VAL  # noqa: E402

import open3d  # noqa: E402,F401  must be imported before DataLoader workers fork:
#                importing it first inside a forked worker hangs (checked 2026-10-11)
M, KD, ND, REIN_MOD = load_official()   # official main / kitti_dataset / nclt_dataset / REIN
import cv2  # noqa: E402
import faiss  # noqa: E402
import h5py  # noqa: E402  (from _external/bevplace_pylib)
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils import data  # noqa: E402

RESULTS = REPO / "results/pdv2_bevplace_official"
SELECTION = RESULTS / "train_selection.json"
DEFAULT_ARGV = ["--nEpochs", "50",   # P3 paper (main.py default 40)
                "--seed", "1"]       # project seed (main.py default 1024)
HARD_MINING_START = 10                # P5 paper
CHUNK = 1024          # rows per chunk of the negative-distance evaluation
FAISS_BLOCK = 19      # < faiss.cvar.distance_compute_blas_threshold (20)
SKIP, RADIUS = 30, 5.0


# ------------------------------------------------------------ dataset
class _LazyNegatives:
    """negatives[i] built like kitti_dataset.TrainingDataset.__init__
    (same numpy calls, argsort order) with the paper rule (P4: every frame
    that is not a positive, >= 5 m) inside the frame's own sequence, computed
    on access and shifted by the sequence offset."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds.seq_of)

    def __getitem__(self, i):
        s = self.ds.seq_of[i]
        off, poses = self.ds.offsets[s], self.ds.seq_poses[s]
        q_pose = poses[i - off]
        dises = np.sqrt(np.sum(((q_pose - poses) ** 2)[:, [3, 7, 11]], axis=1))
        indexes = np.argsort(dises)
        negs = indexes[np.where(dises[indexes] >= self.ds.neg_thres)[0]]   # P4 (code: > 7)
        return negs + off


def _neg_dist_chunked(q_feat, feats, neg_ids, chunk=None):
    """== np.sqrt(np.sum((q_feat.reshape(1,-1)-feats[neg_ids.tolist()])**2,axis=1))
    (kitti_dataset.py:246-247), evaluated chunk by chunk (row-wise identical)."""
    chunk = chunk or CHUNK
    return np.concatenate([
        np.sqrt(np.sum((q_feat.reshape(1, -1) - feats[neg_ids[s:s + chunk].tolist()]) ** 2,
                       axis=1))
        for s in range(0, len(neg_ids), chunk)])


class PDV2TrainingDataset(KD.TrainingDataset):
    """kitti_dataset.TrainingDataset over several sequences of one data root
    (official layout: <root><seq>/bev_imgs/*.png, <root>poses/<seq>.txt, plus
    the stored points <points_root><seq>/<scan_id>.npy for P6).  Pools never
    cross sequences."""

    def __init__(self, dataset_path, seqs, max_frames=0, points_root=None):
        data.Dataset.__init__(self)
        self.pos_thres = 5    # kitti_dataset.py:205
        self.neg_thres = 5    # P4 paper (kitti_dataset.py:206: 7)
        self.num_neg = 10     # kitti_dataset.py:209 (= ICCV paper: 10 negatives)
        points_root = points_root or str(POINTS / "train") + "/"
        self.imgs_path, self.points_path, self.positives = [], [], []
        self.seq_poses, self.offsets, seq_of = [], [], []
        for s, seq in enumerate(seqs):
            imgs_p = os.listdir(dataset_path + seq + '/bev_imgs/')
            imgs_p.sort()
            poses = np.loadtxt(dataset_path + 'poses/' + seq + '.txt')
            if max_frames:
                imgs_p, poses = imgs_p[:max_frames], poses[:max_frames]
            assert len(imgs_p) == len(poses), (seq, len(imgs_p), len(poses))
            off = len(self.imgs_path)
            self.imgs_path += [dataset_path + seq + '/bev_imgs/' + i for i in imgs_p]
            self.points_path += [points_root + seq + '/' + i[:-4] + '.npy' for i in imgs_p]
            for qi in range(len(poses)):       # kitti_dataset.py:212-219
                q_pose = poses[qi]
                dises = np.sqrt(np.sum(((q_pose - poses) ** 2)[:, [3, 7, 11]], axis=1))
                indexes = np.argsort(dises)
                remap_index = indexes[np.where(dises[indexes] < self.pos_thres)[0]]
                self.positives.append(remap_index[1:] + off)  # exclude query itself
            self.seq_poses.append(poses)
            self.offsets.append(off)
            seq_of += [s] * len(poses)
        self.seq_of = np.asarray(seq_of)
        self.poses = np.concatenate(self.seq_poses)   # __len__ = len(self.poses)
        self.negatives = _LazyNegatives(self)
        self.mining = False
        self.query_only = False
        self.cache = None

    def _select(self, index):
        if self.mining:   # kitti_dataset.py:236-252, distance in chunks
            q_feat = self.h5feat[index]
            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1, -1) - pos_feat) ** 2, axis=1))
            min_idx = np.where(dis_pos == np.max(dis_pos))[0][0]  # noqa: F841 (official, unused)
            pos_idx = np.random.choice(self.positives[index], 1)[0]
            negs = self.negatives[index]
            dis_neg = _neg_dist_chunked(q_feat, self.h5feat, negs)
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg - 1:-1]
            neg_idx = negs[dis_inc_index_tmp[:self.num_neg]]
        else:             # P5: epochs before hard mining
            pos_idx = np.random.choice(self.positives[index], 1)[0]           # :243
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int),
                                       self.num_neg)                          # :258
            neg_idx = self.negatives[index][neg_idx]                          # :259
        return pos_idx, neg_idx

    def _image(self, i):
        """P6: rotated point cloud -> official generator -> uint8 as the
        stored PNG -> cv2.imread(path)'s 3 equal channels, CHW."""
        ang = np.random.uniform(-np.pi, np.pi)
        img = png_u8(rotated_bev(np.load(self.points_path[i]), ang))
        return np.repeat(img[None], 3, axis=0)

    def __getitem__(self, index):
        if self.query_only:   # feature-cache / k-means passes use only the query
            query = self._image(index).astype(np.float32) / 256
            return query, query, torch.empty(0, 3, IMG, IMG), index
        pos_idx, neg_idx = self._select(index)
        query = self._image(index)
        positive = self._image(pos_idx)
        query = (query.astype(np.float32)) / 256
        positive = (positive.astype(np.float32) / 256)
        negatives = []
        for neg_i in neg_idx:
            negative = self._image(neg_i)
            negative = (negative) / 256
            negatives.append(torch.from_numpy(negative.astype(np.float32)))
        negatives = torch.stack(negatives, 0)
        return query, positive, negatives, index


def train_epoch_paper(epoch, model, train_set):
    """main.train_epoch (main.py:75-154) verbatim except P5 (mining from
    HARD_MINING_START) and the query-only cache pass."""
    opt, device, optimizer, writer = M.opt, M.device, M.optimizer, M.writer
    epoch_loss = 0
    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize
    criterion = M.TripletLoss().to(device)
    model.eval()
    if epoch >= HARD_MINING_START:                       # P5 (main.py:87: epoch>=0)
        print('====> Building Cache for Hard Mining')
        train_set.mining = False
        train_set.query_only = True
        train_set.cache = join(opt.cachePath, 'train_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5:
            pool_size = model.global_feat_dim
            h5feat = h5.create_dataset("features", [len(train_set), pool_size], dtype=np.float32)
            training_data_loader = data.DataLoader(dataset=train_set, num_workers=opt.threads,
                                                   batch_size=opt.batchSize, shuffle=False,
                                                   collate_fn=KD.collate_fn)
            with torch.no_grad():
                for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader, 1):
                    query = query.to(device)
                    _, _, global_descs = model(query)
                    h5feat[indices, :] = global_descs.detach().cpu().numpy()
        train_set.query_only = False
        train_set.mining = True
        train_set.refreshCache()

    training_data_loader = data.DataLoader(dataset=train_set, num_workers=opt.threads,
                                           batch_size=opt.batchSize, shuffle=True,
                                           collate_fn=KD.collate_fn)
    model.train()
    for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader):
        B, C, H, W = query.shape
        input = torch.cat([query, positives, negatives])
        input = input.to(device)
        _, _, global_descs = model(input)
        global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, negatives.shape[0]])
        optimizer.zero_grad()
        loss = 0
        num_negs = negatives.shape[0] // B
        for i in range(len(global_descs_Q)):
            max_loss = torch.max(criterion(global_descs_Q[i], global_descs_P[i], global_descs_N[num_negs * i:num_negs * (i + 1)]))
            loss += max_loss
        loss /= opt.batchSize
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        epoch_loss += batch_loss
        if iteration % 50 == 0 or n_batches <= 10:
            print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, n_batches, batch_loss), flush=True)
            writer.add_scalar('Train/Loss', batch_loss, ((epoch - 1) * n_batches) + iteration)
    optimizer.zero_grad()
    avg_loss = epoch_loss / n_batches
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    return avg_loss


# ------------------------------------------------------------ validation
def recall_at_1(desc, poses, per_query_records=None):
    """compute_recall_multi_k(desc, poses, k_values=[1], distance_threshold=5,
    skip_frames=30) with the faiss search issued 19 queries at a time."""
    from baselines.eval_utils import _find_revisit_queries, _score_recalls_from_ranked
    assert faiss.cvar.distance_compute_blas_threshold > FAISS_BLOCK
    positions = poses[:, :3, 3].astype(np.float64)
    queries = _find_revisit_queries(positions, RADIUS, SKIP)
    if len(queries) == 0:
        return 0.0, 0
    emb = desc.astype(np.float32).copy()
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    search_k = min(1 + 2 * SKIP, len(emb))
    qidx = np.array([q for q, _ in queries])
    ranked = []
    for s in range(0, len(qidx), FAISS_BLOCK):
        _, ind = index.search(emb[qidx[s:s + FAISS_BLOCK]], search_k)
        for row, q in zip(ind, qidx[s:s + FAISS_BLOCK]):
            ranked.append(row[np.abs(row - q) > SKIP][:1])
    rec = _score_recalls_from_ranked(queries, ranked, positions, [1], RADIUS, poses=poses,
                                     per_query_records=per_query_records)
    return float(rec[1]), len(queries)


def _poses44(p12):
    p = np.tile(np.eye(4), (len(p12), 1, 1))
    p[:, :3, :] = np.asarray(p12, dtype=np.float64).reshape(-1, 3, 4)
    return p


def make_val_sets(max_frames=0):
    sets = []
    for s, q in VAL:
        ds = ND.InferDataset(seq=seq_name(s, q), dataset_path=str(BEV / "val") + "/")
        if max_frames:
            ds.imgs_path, ds.poses = ds.imgs_path[:max_frames], ds.poses[:max_frames]
        assert len(ds.imgs_path) == len(ds.poses), (s, q)
        sets.append((f"{s}/{q}", ds))
    return sets


def validate(val_sets):
    per = {}
    for name, ds in val_sets:
        desc = M.infer(ds)                       # official inference
        r1, nq = recall_at_1(desc, _poses44(ds.poses))
        per[name] = {"r1": r1, "n_queries": nq, "n_frames": len(ds)}
        print(f"    val {name}: R@1 {r1:.4f} ({nq} queries)", flush=True)
    return per


# ------------------------------------------------------------ training
def _rng_state():
    return {"random": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def _set_rng_state(st):
    random.setstate(st["random"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    torch.cuda.set_rng_state_all(st["cuda"])


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


PAPER_OVERRIDES = {
    "P1_optimizer": "AdamW(lr=opt.lr, weight_decay=opt.weightDecay=1e-3) [main.py:310-311 Adam, no decay]",
    "P2_schedule": "StepLR(step=opt.lrStep=10, gamma=opt.lrGamma=0.5) per epoch [main.py: none]",
    "P3_epochs": "50 [main.py:38 default 40]",
    "P4_negatives": ">= 5 m, all non-positive frames [kitti_dataset.py:206,221 > 7 m]",
    "P5_hard_mining": f"from epoch {HARD_MINING_START} (0-based); random negatives before "
                      "[main.py:87 every epoch]",
    "P6_augmentation": "point-cloud z-rotation, uniform [-pi, pi), BEV regenerated "
                       "[kitti_dataset.py:263-286 image warp, integer 0..359 deg]",
    "P7_network_image": "200x200 BEV, bilinear feature up-sampling, NetVLAD bias "
                        "[gen_bev_images.py:28-36, REIN.py:132,137, REIN.py:23,37]",
    "seed": "1 (project) [main.py:47 default 1024]",
}


def train(args, official_argv):
    from tensorboardX import SummaryWriter
    from _pdv2_c3 import _family_pooled
    global HARD_MINING_START
    HARD_MINING_START = args.hard_mining_start

    sys.argv = [M.__file__, "--mode", "train"] + DEFAULT_ARGV + official_argv
    opt = M.get_args()                      # official parser; later flags win
    assert opt.mode == "train" and opt.load_from == "", \
        "--load_from would skip the official NetVLAD k-means init"
    M.opt = opt
    device = torch.device("cuda")
    M.device = device

    random.seed(opt.seed)                   # main.py:243-246
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print('===> Building model')
    model = build_model(REIN_MOD)           # P7
    model = model.cuda()
    M.model = model
    root = str(BEV / "train") + "/"
    seqs = [seq_name(s, q) for s, q in TRAIN]

    if not args.resume:                     # main.py:272-282
        initcache = join(opt.cachePath, 'desc_cen.hdf5')
        if not isfile(initcache):
            train_set = PDV2TrainingDataset(root, seqs, args.smoke)
            train_set.query_only = True     # getClusters reads only the query
            print('===> Calculating descriptors and clusters')
            M.getClusters(train_set)
        else:
            print(f'===> reusing existing {initcache} (official behaviour)')
        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.pooling.init_params(clsts, traindescs)
            add_netvlad_bias(model)         # P7: init_params removes the bias
            model = model.cuda()
        M.model = model

    log_dir = args.resume or join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir=log_dir)
    logdir = writer.file_writer.get_logdir()
    os.makedirs(logdir, exist_ok=True)
    if not args.resume:
        with open(join(logdir, 'flags.json'), 'w') as f:
            f.write(json.dumps({k: v for k, v in vars(opt).items()}))
    print('===> Saving state to:', logdir)
    M.writer = writer

    print('===> Loading dataset(s)')
    t0 = time.time()
    train_set = PDV2TrainingDataset(root, seqs, args.smoke)
    val_sets = make_val_sets(args.smoke)
    print(f"    train: {len(train_set)} frames in {len(seqs)} sequences "
          f"({time.time() - t0:.0f} s); val: "
          f"{sum(len(d) for _, d in val_sets)} keyframes in {len(val_sets)} sequences",
          flush=True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=opt.lr, weight_decay=opt.weightDecay)          # P1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(opt.lrStep),
                                          gamma=opt.lrGamma)                  # P2
    M.optimizer = optimizer

    best_score, start = 0, 0
    record = {"_meta": {
        "official_repo": "https://github.com/zjuluolun/BEVPlace2",
        "official_commit": OFFICIAL_COMMIT,
        "paper": "BEVPlace++ T-RO 2025 (arXiv 2408.01841v3); BEVPlace ICCV 2023 "
                 "(arXiv 2302.14325) where the T-RO text is silent",
        "paper_overrides": PAPER_OVERRIDES, "hard_mining_start": HARD_MINING_START,
        "driver": "scripts/_bevplace_official_train.py",
        "flags": vars(opt), "smoke": args.smoke, "logdir": logdir,
        "train_sequences": {n: len(p) for n, p in zip(seqs, train_set.seq_poses)},
        "val_sequences": [n for n, _ in val_sets],
        "selection_rule": "per epoch: split-v2 VAL intra-sequence 5 m R@1, 30-keyframe "
                          "gap (compute_recall_multi_k semantics); sensor families "
                          "(kitti+kitti360, helipr, mulran) weighted equally, queries "
                          "pooled inside a family; model_best when strictly higher "
                          "than every earlier epoch (main.py is_best rule)"},
        "epochs": []}
    if args.resume:
        ck = torch.load(join(logdir, 'checkpoint.pth.tar'), map_location='cpu',
                        weights_only=False)
        model.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        best_score, start = ck['best_score'], ck['epoch'] + 1
        _set_rng_state(torch.load(join(logdir, 'rng_state.pt'), weights_only=False))
        record = json.loads(args.selection_out.read_text())
        record["epochs"] = [e for e in record["epochs"] if e["epoch"] < start]
        record.setdefault("resumed_at_epoch", []).append(start)
        print(f"===> resumed after epoch {ck['epoch']} (best {best_score:.4f})")

    for epoch in range(start, opt.nEpochs):   # main.py:318-360
        torch.cuda.reset_peak_memory_stats()
        lr = optimizer.param_groups[0]["lr"]
        t0 = time.time()
        avg_loss = train_epoch_paper(epoch, model, train_set)
        scheduler.step()                                                      # P2
        t_train = time.time() - t0

        print('===> Testing')
        t0 = time.time()
        per = validate(val_sets)
        mean_recall = _family_pooled(per)
        seq_mean = float(np.mean([v["r1"] for v in per.values() if v["n_queries"]]))
        t_val = time.time() - t0
        for name, v in per.items():
            writer.add_scalars('val', {name.replace('/', '_'): v["r1"]}, epoch)
        writer.add_scalars('val', {'family_macro_r1': mean_recall}, epoch)
        print('===> Family-equal R@1 on v2 val: %0.2f (sequence mean %0.2f)'
              % (mean_recall * 100, seq_mean * 100), flush=True)

        is_best = mean_recall > best_score
        if is_best:
            best_score = mean_recall

        M.saveCheckpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'recalls': mean_recall,
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, logdir)
        torch.save(_rng_state(), join(logdir, 'rng_state.pt'))

        record["epochs"].append({
            "epoch": epoch, "lr": lr, "hard_mining": epoch >= HARD_MINING_START,
            "train_avg_loss": avg_loss,
            "family_macro_r1": mean_recall, "seq_mean_r1": seq_mean,
            "per_seq": per, "is_best": bool(is_best), "train_s": round(t_train),
            "val_s": round(t_val),
            "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)})
        best = max(record["epochs"], key=lambda e: (e["family_macro_r1"], -e["epoch"]))
        record["best"] = {"epoch": best["epoch"], "family_macro_r1": best["family_macro_r1"],
                          "checkpoint": join(logdir, 'model_best.pth.tar')}
        _write_json(args.selection_out, record)
        print(f"===> epoch {epoch}: train {t_train:.0f} s, val {t_val:.0f} s, "
              f"best epoch {best['epoch']} ({best['family_macro_r1']:.4f})", flush=True)

    print('===> Best Recall: %0.2f' % (best_score * 100))
    writer.close()
    record["best"]["sha256"] = sha256(Path(record["best"]["checkpoint"]))
    record["finished"] = True
    _write_json(args.selection_out, record)
    print("BEVPLACE_OFFICIAL_TRAIN_DONE", flush=True)


# ------------------------------------------------------------ CPU selftest
def selftest():
    """CPU only.
    (1) pools: positives == official TrainingDataset; negatives == the P4 rule.
    (2) selection: with the same np.random state, our mining selection (pos,
        10 hardest negatives) == the official __getitem__'s, and our random
        negatives == the official non-mining sampling (frames identified from
        constant-valued PNGs through the official image path).
    (3) items through the official collate_fn and forked workers (P6 path,
        query-only path).
    (4) chunked distances; (5) batched recall == compute_recall_multi_k;
    (6) PaperREM with bicubic == official REM bit for bit, bilinear differs;
        zero NetVLAD bias == no bias; (7) h5py."""
    global CHUNK
    from baselines.eval_utils import compute_recall_multi_k
    rng = np.random.default_rng(0)
    out = {}
    with tempfile.TemporaryDirectory() as td:
        root, proot = td + "/bev/", td + "/pts/"
        os.makedirs(root + "poses")
        seqs = {"kitti_06": "data/preprocessed_v2/cache_fecdaef7_kitti_06.npz",
                "kitti_05": "data/preprocessed_v2/cache_fecdaef7_kitti_05.npz"}
        sizes = {"kitti_06": 250, "kitti_05": 200}
        for name, cp in seqs.items():
            P = np.load(REPO / cp)["poses"][:sizes[name]].astype(np.float64)
            np.savetxt(root + f"poses/{name}.txt", P[:, :3, :].reshape(-1, 12))
            os.makedirs(root + f"{name}/bev_imgs")
            os.makedirs(proot + name)
            for i in range(len(P)):   # constant image: value encodes the frame
                cv2.imwrite(root + f"{name}/bev_imgs/{i:06d}.png",
                            np.full((IMG, IMG), i + 1, np.float64))
                np.save(proot + f"{name}/{i:06d}.npy",
                        rng.uniform(-60, 60, (3000, 3)).astype(np.float32))
        ours = PDV2TrainingDataset(root, list(seqs), points_root=proot)
        feats = rng.standard_normal((len(ours), 8192)).astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        ours.h5feat = feats
        n_sel = 0
        for name, off in zip(seqs, ours.offsets):
            off_ds = KD.TrainingDataset(dataset_path=root, seq=name)   # official
            n = len(off_ds)
            assert n == sizes[name]
            poses = off_ds.poses
            paper_negs = []
            for i in range(n):
                assert np.array_equal(ours.positives[off + i], off_ds.positives[i] + off)
                d = np.sqrt(np.sum(((poses[i] - poses) ** 2)[:, [3, 7, 11]], axis=1))
                ix = np.argsort(d)
                ref = ix[np.where(d[ix] >= 5)[0]]
                assert np.array_equal(ours.negatives[off + i], ref + off)
                assert np.array_equal(np.sort(np.r_[ref, off_ds.positives[i], i]), np.arange(n))
                paper_negs.append(ref)
            off_ds.negatives = paper_negs          # same pools for the selection test
            off_ds.h5feat = feats[off:off + n]

            def decode(item):   # frame ids from the centre pixel of each image
                return (int(round(item[1][0, IMG // 2, IMG // 2] * 256)) - 1,
                        [int(round(float(v) * 256)) - 1 for v in item[2][:, 0, IMG // 2, IMG // 2]])
            for i in list(range(0, n, 23)) + [n - 1]:
                CHUNK = 64                         # force several chunks
                ours.mining = off_ds.mining = True
                np.random.seed(1000 + i)
                p, ng = ours._select(off + i)
                np.random.seed(1000 + i)
                bp, bn = decode(KD.TrainingDataset.__getitem__(off_ds, i))
                assert p == off + bp and list(ng) == [off + k for k in bn], (name, i)
                ours.mining = off_ds.mining = False
                np.random.seed(2000 + i)
                p, ng = ours._select(off + i)
                np.random.seed(2000 + i)
                np.random.choice(off_ds.positives[i], 1)   # our extra positive draw
                _, bn = decode(KD.TrainingDataset.__getitem__(off_ds, i))
                assert list(ng) == [off + k for k in bn], (name, i)
                assert int(p) - off in set(off_ds.positives[i].tolist())
                n_sel += 2
            print(f"  dataset {name}: pools match (positives official, negatives P4); "
                  f"selection matches official", flush=True)
        out["selection_cases_compared"] = n_sel
        CHUNK = 1024
        for mining, qonly in ((True, False), (False, False), (False, True)):
            ours.mining, ours.query_only = mining, qonly
            dl = data.DataLoader(ours, num_workers=2, batch_size=4, shuffle=True,
                                 collate_fn=KD.collate_fn)
            qb, pb, nb, ib = next(iter(dl))
            assert tuple(qb.shape) == (4, 3, IMG, IMG) and tuple(pb.shape) == (4, 3, IMG, IMG)
            assert tuple(nb.shape) == ((0 if qonly else 40), 3, IMG, IMG) and len(ib) == 4
            assert float(qb.max()) < 1 and float(qb.min()) >= 0
            out[f"collate_mining{int(mining)}_queryonly{int(qonly)}"] = list(nb.shape)
        ours.query_only = False

    # (4) chunked distance on a 12k-row bank vs one gather
    bank = rng.standard_normal((12000, 8192)).astype(np.float32)
    ids = rng.permutation(12000)[:11950]
    one = np.sqrt(np.sum((bank[7].reshape(1, -1) - bank[ids.tolist()]) ** 2, axis=1))
    assert np.array_equal(one, _neg_dist_chunked(bank[7], bank, ids))
    del bank, one
    out["chunked_distance_equal_12k"] = True

    # (5) batched recall vs compute_recall_multi_k on a real val track
    for s, q in (("helipr", "Roundabout02"), ("kitti", "09")):
        P = np.load(cache_path("val", s, q))["poses"].astype(np.float64)
        pos = P[:, :3, 3]
        W = rng.standard_normal((6, 256))
        f = np.concatenate([np.cos(pos[:, :2] / 7.0), np.sin(pos[:, :2] / 7.0),
                            np.cos(pos[:, :2] / 3.0)], 1)
        emb = (f @ W + 0.8 * rng.standard_normal((len(pos), 256))).astype(np.float32)
        recs = []
        ref, nq_ref = compute_recall_multi_k(emb, P, k_values=[1], distance_threshold=5.0,
                                             skip_frames=30, per_query_records=recs)
        mine = []
        r1, nq = recall_at_1(emb, P, per_query_records=mine)
        assert nq == nq_ref and r1 == ref[1], (s, q, r1, ref, nq, nq_ref)
        assert [r["top1_idx"] for r in mine] == [r["top1_idx"] for r in recs]
        out[f"recall_{s}_{q}"] = {"r1": r1, "n_queries": nq, "equal": True}
        print(f"  recall {s}/{q}: batched {r1:.4f} == per-query {ref[1]:.4f} ({nq} q)")

    # (6) network overrides (CPU; REIN.py hard-codes .cuda() on small tensors)
    cuda_orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    try:
        torch.manual_seed(0)
        off_model = REIN_MOD.REIN().eval()
        x = torch.rand(1, 3, IMG, IMG)
        with torch.no_grad():
            ref = off_model.rem(x)
            off_model.rem.__class__ = paper_rem_class(REIN_MOD)
            BB.UPSAMPLE_MODE = "bicubic"
            same = off_model.rem(x)
            BB.UPSAMPLE_MODE = UPSAMPLE_MODE
            bil = off_model.rem(x)
            g0 = off_model.pooling(bil[0])
            add_netvlad_bias(off_model)
            g1 = off_model.pooling(bil[0])
        assert all(torch.equal(a, b) for a, b in zip(ref, same)), "REM copy differs"
        assert not torch.equal(ref[0], bil[0]) and ref[0].shape == bil[0].shape
        bias_diff = float((g0 - g1).abs().max())
        assert bias_diff < 1e-6 and off_model.pooling.conv.bias.requires_grad
        out["paper_rem"] = {"bicubic_copy_equals_official": True,
                            "bilinear_max_abs_diff": float((ref[0] - bil[0]).abs().max()),
                            "out1_shape": list(bil[0].shape),
                            "zero_bias_vs_no_bias_max_abs_diff": bias_diff}
        m2 = build_model(REIN_MOD)
        assert "pooling.conv.bias" in m2.state_dict()
    finally:
        torch.Tensor.cuda = cuda_orig

    # (7) h5py round trip in the work dir
    BEV.parent.mkdir(parents=True, exist_ok=True)
    p = BEV.parent / "_h5_selftest.hdf5"
    with h5py.File(p, mode='w') as h5:
        d = h5.create_dataset("features", [10, 4], dtype=np.float32)
        d[[1, 2, 5], :] = np.ones((3, 4), np.float32)
    with h5py.File(p, mode='r') as h5:
        assert np.array(h5.get("features")).sum() == 12
    p.unlink()
    out["h5py"] = h5py.__version__
    out["official_modules"] = {m.__name__: m.__file__ for m in (M, KD, ND, REIN_MOD)}
    print(json.dumps(out, indent=1))
    print("TRAIN_SELFTEST_OK", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", type=int, default=0,
                    help="first N frames of every train/val sequence (timing test)")
    ap.add_argument("--hard-mining-start", type=int, default=HARD_MINING_START,
                    help="paper: 10; lower only for the smoke test")
    ap.add_argument("--resume", default="", help="run dir to continue")
    ap.add_argument("--selection-out", type=Path, default=SELECTION)
    ap.add_argument("--selftest", action="store_true")
    args, official_argv = ap.parse_known_args()
    if args.selftest:
        selftest()
        return
    train(args, official_argv)


if __name__ == "__main__":
    main()
