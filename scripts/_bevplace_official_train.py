#!/usr/bin/env python3
"""BEVPlace++ official pipeline on split v2 -- part 2: training (2026-10-11).

Runs the training of the OFFICIAL main.py (_external/BEVPlace2, commit in
_bevplace_official_bev.OFFICIAL_COMMIT) on the split-v2 TRAIN keyframes.
Executed unchanged from the official files (imported, not copied):
  REIN (REM 8 rotations + ResNet34 layers 1-2 from ImageNet + NetVLAD 64x128),
  main.getClusters (400 random training images x 25 local features, faiss
  k-means 64, niter 100) + NetVLAD.init_params, main.train_epoch (per-epoch
  feature cache of every training image -> hard negatives; random positive;
  10 negatives; rotation augmentation 0-359 deg; triplet margin 0.3, max over
  negatives, /batchSize), main.TripletLoss, main.infer, main.saveCheckpoint,
  main.get_args (all official defaults: Adam lr 1e-4 WITHOUT weight decay,
  batch 4, 40 epochs; lrStep/lrGamma/weightDecay/momentum are
  parsed but unused, exactly as in main.py).
This file replaces only what the official code hard-codes for its own data:
  1. dataset (main.py:275/303 kitti_dataset.TrainingDataset() = KITTI 00, first
     3000 frames): PDV2TrainingDataset below = the official class with its
     per-sequence rules (positive < 5 m excluding the query, negative > 7 m,
     argsort order) applied to every split-v2 TRAIN sequence separately and
     concatenated. Two engineering changes, both bit-identical in output:
       - negatives[i] is computed on access instead of stored (the stored
         lists would be ~8 GB for the 12-19k-frame HeLiPR sequences);
       - in __getitem__ the query-to-all-negatives feature distance is
         evaluated in row chunks instead of one gather of all negatives
         (one gather is 0.6 GB per item for a 19k-frame sequence, x 24
         loader workers).  Per-row arithmetic is unchanged.
  2. per-epoch validation and checkpoint choice (main.py:322-360: KITTI
     00/02/05/06/08 + NCLT, best = mean NCLT recall; all of these are split-v2
     test places): split-v2 VAL keyframes, intra-sequence 5 m R@1 with a
     30-keyframe gap (baselines.eval_utils.compute_recall_multi_k semantics),
     sensor families weighted equally with queries pooled inside a family
     (scripts/_pdv2_c3._family_pooled) -- the rule of the earlier v2 baseline
     runs.  The official structure is kept: evaluate every epoch, keep
     model_best when the mean recall is strictly higher (ties -> earlier).
     The faiss search is issued in blocks of 19 queries (below faiss's BLAS
     threshold of 20), which returns bit-identical results to the per-query
     loop of compute_recall_multi_k (checked in --selftest).
  3. seed 1 (project protocol; main.py:47 default 1024), passed as the
     official --seed flag.
  4. --resume RUN_DIR (not in main.py): continue after the last finished
     epoch from checkpoint.pth.tar (model + Adam state + best_score) and the
     saved RNG states, for interrupted runs.
Rule (2026-10-11, final): the released code's actual behaviour is used, not
the paper's values; see logs/baseline_official_20261011/bevplace.md §8.

Usage (inside ring_baseline; official flags pass through to main.get_args):
  python3 scripts/_bevplace_official_train.py \
      --runsPath _external/bevplace_official_pdv2/runs/ \
      --cachePath _external/bevplace_official_pdv2/cache/
  smoke:  ... --smoke 200 --nEpochs 1 --runsPath .../smoke_runs/ \
              --cachePath .../smoke_cache/ --selection-out .../smoke/train_selection.json
  resume: ... --resume _external/bevplace_official_pdv2/runs/<Mon DD_HH-MM-SS>
  CPU check: python3 scripts/_bevplace_official_train.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
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
    BEV, OFFICIAL_COMMIT, cache_path, load_official, seq_name, sha256)
from _pdv2_pilot import TRAIN, VAL  # noqa: E402

M, KD, ND, REIN_MOD = load_official()   # official main / kitti_dataset / nclt_dataset / REIN
import cv2  # noqa: E402
import faiss  # noqa: E402
import h5py  # noqa: E402  (from _external/bevplace_pylib)
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils import data  # noqa: E402

RESULTS = REPO / "results/pdv2_bevplace_official"
SELECTION = RESULTS / "train_selection.json"
PROTOCOL_ARGV = ["--seed", "1"]   # project seed; explicit flags given later win
CHUNK = 1024          # rows per chunk of the negative-distance evaluation
FAISS_BLOCK = 19      # < faiss.cvar.distance_compute_blas_threshold (20)
SKIP, RADIUS = 30, 5.0


# ------------------------------------------------------------ dataset
class _LazyNegatives:
    """negatives[i] exactly as kitti_dataset.TrainingDataset.__init__ builds
    it (same numpy calls on the same per-sequence pose array), computed on
    access and shifted by the sequence offset."""

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
        negs = indexes[np.where(dises[indexes] > self.ds.neg_thres)[0]]
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
    (official layout: <root><seq>/bev_imgs/*.png, <root>poses/<seq>.txt).
    Positive / negative pools never cross sequences (each official pool is one
    sequence; our sequences do not share a pose frame)."""

    def __init__(self, dataset_path, seqs, max_frames=0):
        data.Dataset.__init__(self)
        self.pos_thres = 5    # official values (kitti_dataset.py:205-209)
        self.neg_thres = 7
        self.num_neg = 10
        self.imgs_path, self.positives = [], []
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
        self.cache = None

    def __getitem__(self, index):
        # kitti_dataset.TrainingDataset.__getitem__ verbatim except the two
        # lines marked CHANGED (chunked distance; same values).
        if self.mining:
            q_feat = self.h5feat[index]

            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1, -1) - pos_feat) ** 2, axis=1))

            min_idx = np.where(dis_pos == np.max(dis_pos))[0][0]  # noqa: F841 (official, unused)
            pos_idx = np.random.choice(self.positives[index], 1)[0]

            negs = self.negatives[index]                              # CHANGED
            dis_neg = _neg_dist_chunked(q_feat, self.h5feat, negs)     # CHANGED

            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg - 1:-1]

            neg_idx = negs[dis_inc_index_tmp[:self.num_neg]]

        else:
            pos_idx = self.positives[index][0]

            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]

        query = cv2.imread(self.imgs_path[index])
        # rot augmentation
        mat = cv2.getRotationMatrix2D((query.shape[1] // 2, query.shape[0] // 2), np.random.randint(0, 360), 1)
        query = cv2.warpAffine(query, mat, query.shape[:2])

        query = query.transpose(2, 0, 1)

        positive = cv2.imread(join(self.imgs_path[pos_idx]))
        mat = cv2.getRotationMatrix2D((positive.shape[1] // 2, positive.shape[0] // 2), np.random.randint(0, 360), 1)
        positive = cv2.warpAffine(positive, mat, positive.shape[:2])
        positive = positive.transpose(2, 0, 1)

        query = (query.astype(np.float32)) / 256
        positive = (positive.astype(np.float32) / 256)

        negatives = []

        for neg_i in neg_idx:

            negative = cv2.imread(self.imgs_path[neg_i])
            mat = cv2.getRotationMatrix2D((negative.shape[1] // 2, negative.shape[0] // 2), np.random.randint(0, 360), 1)
            negative = cv2.warpAffine(negative, mat, negative.shape[:2])
            negative = negative.transpose(2, 0, 1)
            negative = (negative) / 256

            negatives.append(torch.from_numpy(negative.astype(np.float32)))

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index


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


def train(args, official_argv):
    from tensorboardX import SummaryWriter
    from _pdv2_c3 import _family_pooled

    sys.argv = [M.__file__, "--mode", "train"] + PROTOCOL_ARGV + official_argv
    opt = M.get_args()                      # official parser and defaults
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
    model = REIN_MOD.REIN()
    model = model.cuda()
    M.model = model
    root = str(BEV / "train") + "/"
    seqs = [seq_name(s, q) for s, q in TRAIN]

    if not args.resume:                     # main.py:272-282
        initcache = join(opt.cachePath, 'desc_cen.hdf5')
        if not isfile(initcache):
            train_set = PDV2TrainingDataset(root, seqs, args.smoke)
            print('===> Calculating descriptors and clusters')
            M.getClusters(train_set)
        else:
            print(f'===> reusing existing {initcache} (official behaviour)')
        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.pooling.init_params(clsts, traindescs)
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

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,   # main.py:310-311
                                  model.parameters()), lr=opt.lr)
    M.optimizer = optimizer

    best_score, start = 0, 0
    record = {"_meta": {
        "official_repo": "https://github.com/zjuluolun/BEVPlace2",
        "official_commit": OFFICIAL_COMMIT,
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
        best_score, start = ck['best_score'], ck['epoch'] + 1
        _set_rng_state(torch.load(join(logdir, 'rng_state.pt'), weights_only=False))
        record = json.loads(args.selection_out.read_text())
        record["epochs"] = [e for e in record["epochs"] if e["epoch"] < start]
        record.setdefault("resumed_at_epoch", []).append(start)
        print(f"===> resumed after epoch {ck['epoch']} (best {best_score:.4f})")

    for epoch in range(start, opt.nEpochs):   # main.py:318-360
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        M.train_epoch(epoch, model, train_set)
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
        }, is_best, logdir)
        torch.save(_rng_state(), join(logdir, 'rng_state.pt'))

        record["epochs"].append({
            "epoch": epoch, "family_macro_r1": mean_recall, "seq_mean_r1": seq_mean,
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
    """CPU only: (1) dataset adapter == official TrainingDataset on each
    sequence (pools, mining and non-mining items, same RNG); (2) chunked
    distances == one gather; (3) batched recall == compute_recall_multi_k;
    (4) h5py works in the work dir."""
    global CHUNK
    from baselines.eval_utils import compute_recall_multi_k
    rng = np.random.default_rng(0)
    out = {}
    with tempfile.TemporaryDirectory() as td:
        root = td + "/"
        os.makedirs(root + "poses")
        seqs = {"kitti_06": "data/preprocessed_v2/cache_fecdaef7_kitti_06.npz",
                "kitti_05": "data/preprocessed_v2/cache_fecdaef7_kitti_05.npz"}
        sizes = {"kitti_06": 260, "kitti_05": 300}
        for name, cp in seqs.items():
            P = np.load(REPO / cp)["poses"][:sizes[name]].astype(np.float64)
            np.savetxt(root + f"poses/{name}.txt", P[:, :3, :].reshape(-1, 12))
            os.makedirs(root + f"{name}/bev_imgs")
            for i in range(len(P)):
                img = (rng.random((201, 201)) < 0.05).astype(np.float64) * rng.integers(1, 11, (201, 201)) * 25.5
                cv2.imwrite(root + f"{name}/bev_imgs/{i:06d}.png", img)
        ours = PDV2TrainingDataset(root, list(seqs))
        feats = rng.standard_normal((len(ours), 8192)).astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        ours.h5feat = feats
        n_items = 0
        for name, off in zip(seqs, ours.offsets):
            off_ds = KD.TrainingDataset(dataset_path=root, seq=name)   # official
            n = len(off_ds)
            assert n == sizes[name]
            off_ds.h5feat = feats[off:off + n]
            for i in range(n):
                assert np.array_equal(ours.positives[off + i], off_ds.positives[i] + off)
                assert np.array_equal(ours.negatives[off + i], off_ds.negatives[i] + off)
            for mining in (False, True):
                ours.mining = off_ds.mining = mining
                for i in list(range(0, n, 37)) + [n - 1]:
                    CHUNK = 64                  # force several chunks
                    np.random.seed(1000 + i)
                    a = ours[off + i]
                    np.random.seed(1000 + i)
                    b = KD.TrainingDataset.__getitem__(off_ds, i)  # official method
                    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                    assert torch.equal(a[2], b[2]) and a[3] == off + b[3]
                    n_items += 1
            print(f"  dataset {name}: pools equal for {n} frames; items equal", flush=True)
        out["dataset_items_compared"] = n_items
        CHUNK = 1024
        # through the official collate_fn and forked loader workers
        ours.mining = True
        dl = data.DataLoader(ours, num_workers=2, batch_size=4, shuffle=True,
                             collate_fn=KD.collate_fn)
        qb, pb, nb, ib = next(iter(dl))
        assert tuple(qb.shape) == (4, 3, 201, 201) and tuple(pb.shape) == (4, 3, 201, 201)
        assert tuple(nb.shape) == (40, 3, 201, 201) and len(ib) == 4
        out["official_collate_batch"] = [list(qb.shape), list(pb.shape), list(nb.shape)]

    # (2) chunked distance on a 19k-row bank vs one gather
    bank = rng.standard_normal((12000, 8192)).astype(np.float32)
    ids = rng.permutation(12000)[:11950]
    one = np.sqrt(np.sum((bank[7].reshape(1, -1) - bank[ids.tolist()]) ** 2, axis=1))
    assert np.array_equal(one, _neg_dist_chunked(bank[7], bank, ids))
    del bank, one
    out["chunked_distance_equal_12k"] = True

    # (3) batched recall vs compute_recall_multi_k on a real val track
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

    # (4) h5py round trip in the work dir
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
