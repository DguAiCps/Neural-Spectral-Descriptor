#!/usr/bin/env python3
"""DiSCO official pipeline on split v2 -- part 2: training (2026-10-11).

Runs the OFFICIAL training code of _external/DiSCO (commit in
_disco_official_prep.OFFICIAL_COMMIT) on the split-v2 TRAIN keyframes.
Executed unchanged from the official files (imported, not copied):
  train_DiSCO.py  argument parsing and cfg overrides (all defaults), module
                  state, train_one_epoch (shuffle; 2 positives + 18 negatives
                  + 1 other negative per query; queries with < 2 positives
                  skipped as FAULTY TUPLE; from epoch 6 every 1400 queries the
                  descriptors of all training scans are recomputed and 5 of the
                  query's negatives are taken by KDTree among 4000 sampled ones;
                  random yaw on the query, positives rotated by the heading
                  difference; model.ckpt saved every 6000 queries),
                  get_latent_vectors, get_feature_representation,
                  get_random_hard_negatives, run_model, rotation_on_SCI,
                  get_learning_rate, log_string
  loading_pointclouds.get_query_tuple, load_pc_files
  models/DiSCO.py DiSCO (U-Net 20->...->1, 2-D FFT magnitude, centre 32x32 =
                  1024-D) and Corr2Softmax(200, 0)
  loss/loss_function.py quadruplet_loss (margins 0.5 / 0.2, non-lazy sum as
                  set by train_DiSCO.py:85, + yaw cross-entropy through the
                  differentiable phase correlation)
  evaluate.get_latent_vectors (inference for the per-epoch validation)
Recipe (all official defaults): Adam lr 1e-5 for the network (constant:
get_learning_rate(0) only), Adam lr 1e-3 for Corr2Softmax, 20 epochs, one
query per step (cfg.BATCH_NUM_QUERIES = 1; --batch_num_queries is parsed but
never applied), single GPU (train_DiSCO.py:26), no weight decay, no seed in
the official code (we seed SEED from the environment, default 1, user rule).

What this file replaces (each marked in the code):
  1. data: the official pickle of NCLT 2012-02-04 tuples (generator
     construct_query_dict: positives < 1.5 m, negatives > 2 m, 2-D
     northing/easting, frames 0.5-1.5 m apart) -> PDV2Queries over the
     split-v2 TRAIN keyframes, same radii and the same set operations per
     sequence (pools never cross sequences: our sequences do not share a
     pose frame), on 3-D positions (the KITTI cache poses are camera
     frames with y down).  'negatives' is built on access instead of
     stored (the stored lists would be ~1e9 Python ints); every official
     use shuffles it first, so the sampled tuples have the same
     distribution.  The quadruplet's "other negative" is drawn by the
     official code from all training keys (it cannot be restricted without
     editing train_one_epoch), i.e. it may come from another sequence.
  2. scans: load_pc_file reads the official float32 .npy of each scan; it is
     replaced by a decoder of the packed polar stores of
     scripts/_disco_official_prep.py (identical array).
  3. per-epoch evaluation: evaluate.evaluate_model (NCLT test runs, top-1%
     recall; split-v2 test places) -> split-v2 VAL intra-sequence 5 m R@1,
     30-keyframe gap (compute_recall_multi_k semantics) with the official
     retrieval metric (Euclidean on the 1024-D descriptor, KDTree in
     evaluate.get_recall), sensor families weighted equally
     (_pdv2_c3._family_pooled).  Logged under the official 'EVAL RECALL'.
  4. checkpoints: the official model.ckpt (overwritten every 6000 queries)
     is kept; in addition each epoch end writes epoch_XX.ckpt (same keys)
     and resume.pth (optimizers, RNG, latent vectors) for --resume.
Checkpoint rule (user decision 2026-10-11): the official code has no
selection (its evaluate.py loads LOG_DIR/model.ckpt, the last periodic save).
`chosen` = val_best: family-equal v2-val R@1, ties -> earlier epoch, the rule
of the other learned baselines.  official_final is recorded for reference.
Default training flag in scripts/_pdv2_disco_official_run.sh:
--fast-hard-negatives (exact k-NN, same output as the official KDTree; speed
only).

Usage (inside ring_baseline):
  python3 scripts/_disco_official_train.py                 # full run
  python3 scripts/_disco_official_train.py --resume RUN_DIR
  python3 scripts/_disco_official_train.py --smoke 200 --smoke-epochs 0 6
  python3 scripts/_disco_official_train.py --selftest      # CPU only
Official train_DiSCO.py flags pass through (e.g. --max_epoch).
"""
from __future__ import annotations

import argparse
import ast
import collections
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from _disco_official_prep import (  # noqa: E402
    GENERATOR, OFFICIAL, OFFICIAL_COMMIT, OFFICIAL_REPO, RESULTS, SEED, WORK, import_official,
    load_store, official_env, seeded, seq_name, sha256, unpack)
from _pdv2_pilot import TRAIN, VAL  # noqa: E402

RUNS = seeded(WORK / "runs")          # SEED from the environment (default 1)
SMOKE = seeded(WORK / "smoke")
SELECTION = RESULTS / "train_selection.json"
POS_R, NEG_R = 1.5, 2.0       # generator construct_query_dict (lines 160, 163)
SKIP, RADIUS = 30, 5.0        # v2 validation (compute_recall_multi_k semantics)


# ------------------------------------------------------------ data adapters
class PolarBank:
    """Replacement of loading_pointclouds.load_pc_file: key 'role/name/i' ->
    the (20, 40, 120) float32 occupancy of the packed store row i."""

    def __init__(self):
        self.arrays = {}

    def add(self, role, sensor, seq, max_frames=0, in_ram=False, bits=None):
        if bits is None:
            bits, _, _ = load_store(role, sensor, seq, mmap=not in_ram)
        if max_frames:
            bits = bits[:max_frames]
        self.arrays[f"{role}/{seq_name(sensor, seq)}"] = bits
        return len(bits)

    def load(self, filename):
        prefix, i = filename.rsplit("/", 1)
        return unpack(self.arrays[prefix][int(i)])


class _Entry(dict):
    """One entry of the official query dict ({'query', 'heading',
    'positives', 'negatives'}); 'negatives' comes from the owner on access."""

    def __init__(self, owner, gi, **kw):
        super().__init__(**kw)
        self._owner, self._gi = owner, gi

    def __getitem__(self, k):
        if k == "negatives":
            return self._owner.negatives(self._gi)
        return dict.__getitem__(self, k)


class PDV2Queries(dict):
    """The official TRAINING_QUERIES over several sequences (keys 0..N-1).

    Per sequence, as generator construct_query_dict (lines 156-190):
      ind_nn = tree.query_radius(X, r=1.5); positives = setdiff1d(ind_nn[i], [i])
      ind_r  = tree.query_radius(X, r=2);   negatives = setdiff1d(all, ind_r[i])
    with X the 3-D keyframe positions and indices offset into the global
    key space.  negatives(i) is rebuilt on access (sorted order; every
    official use shuffles first) and kept for the last few queries, so the
    in-place shuffles of one training step act on one list."""

    def __init__(self, seqs, positions, headings, query_keys, keep=4):
        from sklearn.neighbors import KDTree
        super().__init__()
        self.trees, self.X, self.offsets, seq_of = [], [], [], []
        off = 0
        for s, (X, H, keys) in enumerate(zip(positions, headings, query_keys)):
            tree = KDTree(X)
            ind_nn = tree.query_radius(X, r=POS_R)
            for i in range(len(X)):
                pos = (np.setdiff1d(ind_nn[i], [i]) + off).tolist()
                self[off + i] = _Entry(self, off + i, query=keys[i],
                                       heading=float(H[i]), positives=pos)
            self.trees.append(tree)
            self.X.append(X)
            self.offsets.append(off)
            seq_of += [s] * len(X)
            off += len(X)
        self.seq_of = np.asarray(seq_of)
        self.seqs = list(seqs)
        self._lru = collections.OrderedDict()
        self._keep = keep

    def negatives(self, gi):
        if gi in self._lru:
            self._lru.move_to_end(gi)
            return self._lru[gi]
        s = int(self.seq_of[gi])
        off, X = self.offsets[s], self.X[s]
        ind_r = self.trees[s].query_radius(X[gi - off:gi - off + 1], r=NEG_R)[0]
        neg = (np.setdiff1d(np.arange(len(X)), ind_r) + off).tolist()
        self._lru[gi] = neg
        if len(self._lru) > self._keep:
            self._lru.popitem(last=False)
        return neg

    def n_faulty(self, num_pos=2):
        return int(sum(len(dict.__getitem__(e, "positives")) < num_pos for e in self.values()))


def build_train(bank, smoke=0, in_ram=True):
    """PDV2Queries + bank rows for the split-v2 TRAIN sequences."""
    positions, headings, keys, sizes = [], [], [], {}
    for sensor, seq in TRAIN:
        bits, idx, meta = load_store("train", sensor, seq, mmap=not in_ram)
        n = bank.add("train", sensor, seq, max_frames=smoke, bits=bits)
        positions.append(idx["poses"][:n, :3, 3].astype(np.float64))
        headings.append(idx["heading"][:n])
        name = f"train/{seq_name(sensor, seq)}"
        keys.append([f"{name}/{i}" for i in range(n)])
        sizes[f"{sensor}/{seq}"] = n
    return PDV2Queries(TRAIN, positions, headings, keys), sizes


# ------------------------------------------------------------ official modules
def load_train_module(run_dir: Path, official_argv, queries):
    """Execute the unchanged train_DiSCO.py as module `disco_train_DiSCO`.
    Its import-time get_queries_dict(cfg.TRAIN_FILE) returns `queries`
    (cfg.TEST_FILE -> {}, TEST_QUERIES is never used in training)."""
    lp, cfg = sys.modules["loading_pointclouds"], sys.modules["config"]
    orig, argv, cvd = lp.get_queries_dict, sys.argv, os.environ.get("CUDA_VISIBLE_DEVICES")
    lp.get_queries_dict = lambda filename: queries if filename == cfg.TRAIN_FILE else {}
    sys.argv = [str(OFFICIAL / "train_DiSCO.py"), "--log_dir", f"{run_dir}/",
                "--results_dir", f"{run_dir}/results/"] + list(official_argv)
    try:
        spec = importlib.util.spec_from_file_location("disco_train_DiSCO",
                                                      OFFICIAL / "train_DiSCO.py")
        T = importlib.util.module_from_spec(spec)
        sys.modules["disco_train_DiSCO"] = T
        spec.loader.exec_module(T)
    finally:
        lp.get_queries_dict, sys.argv = orig, argv
        if cvd is None:   # train_DiSCO.py:26 sets 0; CUDA is already initialised
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    assert T.TRAINING_QUERIES is queries
    return T


def exact_hard_negatives(T):
    """OPT-IN (--fast-hard-negatives): train_DiSCO.get_random_hard_negatives
    builds a KDTree on 4000 x 1024 latent vectors at every step (~0.15 s CPU)
    only to take the 5 Euclidean nearest; this float64 brute force returns the
    same indices in the same order (ties aside; checked in selftest `cost`)."""
    def get_random_hard_negatives(query_vec, random_negs, num_to_take):
        latent_vecs = np.asarray(T.TRAINING_LATENT_VECTORS)[np.asarray(random_negs)]
        query_vec = np.squeeze(np.array([query_vec])).reshape(1, -1)
        d2 = ((latent_vecs.astype(np.float64) - query_vec.astype(np.float64)) ** 2).sum(1)
        indices = np.argsort(d2, kind="stable")[:num_to_take]
        return np.squeeze(np.array(random_negs)[indices]).tolist()
    return get_random_hard_negatives


# ------------------------------------------------------------ validation
def l2_top1_masked(D, sq, qs, k=8, block=256):
    """Exact float64 Euclidean nearest neighbour of each row q in qs over all
    rows with |j - q| > SKIP: squared-distance expansion (block GEMM) for a
    shortlist of k, exact distances on the shortlist, ties -> lower index."""
    out = []
    for b0 in range(0, len(qs), block):
        qb = np.asarray(qs[b0:b0 + block])
        d2 = sq[None, :] - 2.0 * (D[qb] @ D.T) + sq[qb][:, None]
        for r, q in enumerate(qb):
            row = d2[r]
            row[max(q - SKIP, 0):q + SKIP + 1] = np.inf
            kk = min(k, int(np.isfinite(row).sum()))
            if kk == 0:
                out.append(-1)
                continue
            cand = np.argpartition(row, kk - 1)[:kk]
            ex = np.sqrt(((D[cand] - D[q]) ** 2).sum(axis=1))
            out.append(int(cand[np.lexsort((cand, ex))[0]]))
    return out


def recall_at_1_l2(desc, poses, per_query_records=None):
    """compute_recall_multi_k(k=[1], 5 m, skip 30) semantics with the official
    DiSCO metric (Euclidean on the raw descriptor) instead of cosine."""
    from baselines.eval_utils import _find_revisit_queries, _score_recalls_from_ranked
    positions = poses[:, :3, 3].astype(np.float64)
    queries = _find_revisit_queries(positions, RADIUS, SKIP)
    if not queries:
        return 0.0, 0
    D = np.asarray(desc, dtype=np.float64)
    sq = np.einsum("ij,ij->i", D, D)
    top = l2_top1_masked(D, sq, [q for q, _ in queries])
    ranked = [np.array([j]) if j >= 0 else np.array([], dtype=np.int64) for j in top]
    rec = _score_recalls_from_ranked(queries, ranked, positions, [1], RADIUS, poses=poses,
                                     per_query_records=per_query_records)
    return float(rec[1]), len(queries)


def descriptors(EV, model, bank, role, sensor, seq, n):
    """Official evaluate.get_latent_vectors on one sequence (eval mode)."""
    prefix = f"{role}/{seq_name(sensor, seq)}"
    d = {i: {"query": f"{prefix}/{i}", "heading": 0.0} for i in range(n)}
    q_out, _, _, _ = EV.get_latent_vectors(model, d)
    return np.asarray(q_out, dtype=np.float32).reshape(n, -1)


def validate(EV, model, bank, val_sets):
    per = {}
    for sensor, seq, n, poses in val_sets:
        desc = descriptors(EV, model, bank, "val", sensor, seq, n)
        r1, nq = recall_at_1_l2(desc, poses)
        per[f"{sensor}/{seq}"] = {"r1": r1, "n_queries": nq, "n_frames": n}
        print(f"    val {sensor}/{seq}: R@1 {r1:.4f} ({nq} queries)", flush=True)
    return per


def make_val_sets(bank, smoke=0):
    out = []
    for sensor, seq in VAL:
        bits, idx, _ = load_store("val", sensor, seq)
        n = bank.add("val", sensor, seq, max_frames=smoke, bits=bits)
        out.append((sensor, seq, n, idx["poses"][:n].astype(np.float64)))
    return out


# ------------------------------------------------------------ training
def _rng_state(torch):
    return {"random": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def train(args, official_argv):
    torch = official_env()
    # official: one GPU (train_DiSCO.py:26 CUDA_VISIBLE_DEVICES=0).  CUDA is
    # initialised here so that the caller's single-GPU choice holds.
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, \
        "expose exactly one GPU (CUDA_VISIBLE_DEVICES=<id>)"
    cfg, lp, SC, LF, EV = import_official()
    from tensorboardX import SummaryWriter
    from _pdv2_c3 import _family_pooled

    bank = PolarBank()
    lp.load_pc_file = bank.load                                   # replacement 2
    queries, sizes = build_train(bank, args.smoke)                # replacement 1
    val_sets = make_val_sets(bank, args.smoke)
    smoke = bool(args.smoke)
    sel_path = args.selection_out or (SMOKE / "train_selection.json" if smoke else SELECTION)
    if args.resume:
        run_dir = Path(args.resume)
        log = run_dir / "log_train.txt"
        if log.exists():   # train_DiSCO.py opens it with 'w' at import
            log.rename(run_dir / f"log_train.before_resume_{datetime.now():%m%d_%H%M%S}.txt")
    else:
        base = SMOKE / "runs" if smoke else RUNS
        base.mkdir(parents=True, exist_ok=True)
        run_dir = base / datetime.now().strftime("%b%d_%H-%M-%S")
    T = load_train_module(run_dir, official_argv, queries)
    if args.fast_hard_negatives:
        T.get_random_hard_negatives = exact_hard_negatives(T)
    print(f"===> train: {len(queries)} keyframes, {queries.n_faulty()} with < 2 positives "
          f"(skipped by the official FAULTY TUPLE rule); val: "
          f"{sum(v[2] for v in val_sets)} keyframes in {len(val_sets)} sequences", flush=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ---- official train() (train_DiSCO.py:136-186), verbatim apart from comments
    if cfg.LOSS_FUNCTION == 'quadruplet':
        loss_function = T.SC_loss.quadruplet_loss
    elif cfg.LOSS_FUNCTION == 'triplet':
        loss_function = T.SC_loss.triplet_loss
    learning_rate = T.get_learning_rate(0)
    train_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'train'))
    model = T.SC.DiSCO(output_dim=cfg.FEATURE_OUTPUT_DIM)
    corr2soft = T.SC.Corr2Softmax(200., 0.)
    model = model.to(T.device)
    corr2soft = corr2soft.to(T.device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    assert cfg.OPTIMIZER == 'adam'
    optimizer = torch.optim.Adam(parameters, learning_rate)
    optimizer_c2s = torch.optim.Adam(filter(lambda p: p.requires_grad, corr2soft.parameters()), lr=1e-3)

    record = {"_meta": {
        "official_repo": OFFICIAL_REPO, "official_commit": OFFICIAL_COMMIT,
        "driver": "scripts/_disco_official_train.py", "run_dir": str(run_dir),
        "flags": vars(T.FLAGS), "cfg": {k: getattr(cfg, k) for k in (
            "BATCH_NUM_QUERIES", "TRAIN_POSITIVES_PER_QUERY", "TRAIN_NEGATIVES_PER_QUERY",
            "MAX_EPOCH", "BASE_LEARNING_RATE", "MARGIN_1", "MARGIN_2", "LOSS_FUNCTION",
            "LOSS_LAZY", "TRIPLET_USE_BEST_POSITIVES", "FEATURE_OUTPUT_DIM")},
        "seed": SEED, "smoke": args.smoke, "train_sequences": sizes,
        "fast_hard_negatives": bool(args.fast_hard_negatives),
        "n_faulty_queries": queries.n_faulty(),
        "val_sequences": [f"{s}/{q}" for s, q, _, _ in val_sets],
        "pools": "per sequence, 3-D positions, positives < 1.5 m, negatives > 2 m",
        "selection_rule": "chosen = val_best: split-v2 VAL intra-sequence 5 m R@1 "
                          "(30-keyframe gap, Euclidean top-1), families (kitti+kitti360, "
                          "helipr, mulran) weighted equally, ties -> earlier epoch (rule of "
                          "the other learned baselines; user decision 2026-10-11). "
                          "official_final = LOG_DIR/model.ckpt, the last periodic save that "
                          "the official evaluate.py loads, recorded for reference"},
        "epochs": []}
    start = 0
    if args.resume:
        st = torch.load(run_dir / "resume.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        corr2soft.load_state_dict(st["corr2soft"])
        optimizer.load_state_dict(st["optimizer"])
        optimizer_c2s.load_state_dict(st["optimizer_c2s"])
        T.TOTAL_ITERATIONS = st["total_iterations"]
        T.TRAINING_LATENT_VECTORS = st["latent"]
        T.HARD_NEGATIVES = st["hard_negatives"]
        random.setstate(st["rng"]["random"])
        np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"])
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        start = st["epoch"] + 1
        record = json.loads(sel_path.read_text())
        record["epochs"] = [e for e in record["epochs"] if e["epoch"] < start]
        record.setdefault("resumed_at_epoch", []).append(start)
        print(f"===> resumed after epoch {st['epoch']}", flush=True)

    model = T.nn.DataParallel(model)
    T.LOG_FOUT.write(cfg.cfg_str())
    T.LOG_FOUT.write("\n")
    T.LOG_FOUT.flush()

    epochs = args.smoke_epochs if smoke else range(start, cfg.MAX_EPOCH)
    for epoch in epochs:
        print(epoch)
        print()
        T.log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()
        torch.cuda.reset_peak_memory_stats()
        it0, t0 = T.TOTAL_ITERATIONS, time.time()

        T.train_one_epoch(model, optimizer, corr2soft, optimizer_c2s, train_writer, loss_function, epoch)
        t_train = time.time() - t0

        T.log_string('EVALUATING...')
        t0 = time.time()
        per = validate(EV, model, bank, val_sets)                 # replacement 3
        eval_recall = _family_pooled(per)
        t_val = time.time() - t0
        T.log_string('EVAL RECALL: %s' % str(eval_recall))
        train_writer.add_scalar("Val Recall", eval_recall, epoch)

        # ---- replacement 4: per-epoch checkpoint + resume state + record
        ck = run_dir / f"epoch_{epoch:02d}.ckpt"
        torch.save({"epoch": epoch, "iter": T.TOTAL_ITERATIONS,
                    "state_dict": model.module.state_dict(),
                    "corr2soft": corr2soft.state_dict()}, ck)
        st = {"epoch": epoch, "model": model.module.state_dict(),
              "corr2soft": corr2soft.state_dict(), "optimizer": optimizer.state_dict(),
              "optimizer_c2s": optimizer_c2s.state_dict(),
              "total_iterations": T.TOTAL_ITERATIONS,
              "latent": T.TRAINING_LATENT_VECTORS, "hard_negatives": T.HARD_NEGATIVES,
              "rng": _rng_state(torch)}
        torch.save(st, run_dir / "resume.tmp")
        os.replace(run_dir / "resume.tmp", run_dir / "resume.pth")
        n_it = T.TOTAL_ITERATIONS - it0
        record["epochs"].append({
            "epoch": epoch, "family_macro_r1": eval_recall,
            "seq_mean_r1": float(np.mean([v["r1"] for v in per.values() if v["n_queries"]])),
            "per_seq": per, "checkpoint": str(ck), "train_s": round(t_train),
            "trained_queries": n_it, "s_per_query": round(t_train / max(n_it, 1), 4),
            "val_s": round(t_val),
            "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)})
        best = max(record["epochs"], key=lambda e: (e["family_macro_r1"], -e["epoch"]))
        record["val_best"] = {"epoch": best["epoch"], "family_macro_r1": best["family_macro_r1"],
                              "checkpoint": best["checkpoint"]}
        _write_json(sel_path, record)
        print(f"===> epoch {epoch}: train {t_train:.0f} s ({n_it} queries), val {t_val:.0f} s, "
              f"family R@1 {eval_recall:.4f}, val-best epoch {best['epoch']}", flush=True)

    train_writer.close()
    official = run_dir / cfg.MODEL_FILENAME
    if official.exists():
        ck = torch.load(official, map_location="cpu", weights_only=False)
        record["official_final"] = {"checkpoint": str(official), "sha256": sha256(official),
                                    "epoch": ck["epoch"], "iter": ck["iter"]}
    record["val_best"]["sha256"] = sha256(Path(record["val_best"]["checkpoint"]))
    record["chosen"] = "val_best"
    record["finished"] = True
    _write_json(sel_path, record)
    print("DISCO_OFFICIAL_TRAIN_DONE", flush=True)


# ------------------------------------------------------------ CPU selftest
def _official_construct_query_dict():
    """construct_query_dict from the unchanged generator source."""
    tree = ast.parse(GENERATOR.read_text(), str(GENERATOR))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == "construct_query_dict"]
    import pickle
    from sklearn.neighbors import KDTree
    ns = {"np": np, "random": random, "KDTree": KDTree, "pickle": pickle}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(GENERATOR), "exec"), ns)
    return ns["construct_query_dict"]


SELFTEST_PARTS = ("pools", "loop", "recall", "cost")


def selftest(args):
    """CPU only (parts selectable with --parts).
    pools : PDV2Queries == official construct_query_dict (same 2-D
            coordinates for the comparison) on kitti/05 and helipr/Bridge02;
    loop  : official train_one_epoch (unchanged) on CPU on real train
            keyframes (CPU transcription for the polar images): epoch 0
            (random negatives, 26 keys) and epoch 6 (latent refresh at query
            29, hard negatives, periodic model.ckpt at query 101, 120 keys);
            FAULTY TUPLE count; official evaluate.get_latent_vectors;
    recall: Euclidean R@1 == brute-force sklearn KDTree (official retrieval);
    cost  : CPU time of one official step outside the network on the full
            72,306-key dictionary (python shuffles / set differences / KDTree).
    """
    from _disco_official_prep import fake_gputransform_module
    fake = fake_gputransform_module()
    torch = official_env(cpu_only=True)
    torch.set_num_threads(8)
    cfg, lp, SC, LF, EV = import_official(cpu_only=True, fake_gputransform=fake)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out = {}
    for part in args.parts:
        {"pools": _check_pools, "loop": _check_loop, "recall": _check_recall,
         "cost": _check_cost}[part](out, torch, cfg, lp, EV, fake)
    if args.out_json:
        _write_json(args.out_json, out)
    print("TRAIN_SELFTEST_OK", flush=True)


def _check_pools(out, torch, cfg, lp, EV, fake):
    import contextlib
    import io
    import pickle
    import pandas as pd
    cqd = _official_construct_query_dict()
    with tempfile.TemporaryDirectory() as td:
        for sensor, seq in (("kitti", "05"), ("helipr", "Bridge02")):
            idx = store_index("train", sensor, seq)
            P = idx["poses"][:700]
            X2 = P[:, :3, 3][:, [0, 2]] if sensor == "kitti" else P[:, :2, 3]
            df = pd.DataFrame({"file": [f"f{i}" for i in range(len(P))],
                               "northing": X2[:, 0], "easting": X2[:, 1],
                               "yaw": idx["heading"][:700]})
            fn = os.path.join(td, "q.pickle")
            with contextlib.redirect_stdout(io.StringIO()):   # it prints every index
                cqd(df, fn, pickle_flag=True)
            off = pickle.load(open(fn, "rb"))
            ours = PDV2Queries([(sensor, seq)], [X2], [idx["heading"][:700]],
                               [[f"f{i}" for i in range(len(P))]])
            for i in range(len(P)):
                a, b = off[i], ours[i]
                assert a["query"] == b["query"] and a["heading"] == b["heading"]
                assert sorted(a["positives"]) == sorted(b["positives"])
                assert sorted(a["negatives"]) == b["negatives"]
            out[f"pools_{sensor}_{seq}"] = {"n": len(P), "equal": True}
            print(f"  pools {sensor}/{seq}: equal to official construct_query_dict "
                  f"({len(P)} keys)", flush=True)


def _check_loop(out, torch, cfg, lp, EV, fake):
    import types
    from _disco_official_prep import official_hits, official_polar, pack
    from _pd_build_test_caches import _make_loader
    from tensorboardX import SummaryWriter
    bank = PolarBank()
    lp.load_pc_file = bank.load
    pos_l, head_l, key_l, n_per = [], [], [], 60
    cfg_ = types.SimpleNamespace(max_length=1, max_height=1, num_ring=40,
                                 num_sector=120, num_height=20)
    for sensor, seq in (("kitti", "05"), ("helipr", "Bridge02")):
        idx = store_index("train", sensor, seq)
        loader = _make_loader(sensor, seq)
        bits = np.stack([pack(official_polar(official_hits(sensor, loader[int(s)]["points"]),
                                             fake, cfg_))
                         for s in idx["scan_ids"][:n_per]])
        bank.add("train", sensor, seq, bits=bits)
        pos_l.append(idx["poses"][:n_per, :3, 3])
        head_l.append(idx["heading"][:n_per])
        key_l.append([f"train/{seq_name(sensor, seq)}/{i}" for i in range(n_per)])
    queries = PDV2Queries([], pos_l, head_l, key_l)
    # epoch 0 on 26 keys (the official tuple needs >= 18 negatives > 2 m),
    # epoch 6 on 120 keys (refresh at query 29, model.ckpt at query 101)
    small = PDV2Queries([], [pos_l[0][:26]], [head_l[0][:26]], [key_l[0][:26]])
    with tempfile.TemporaryDirectory() as td:
        T = load_train_module(Path(td), [], small)
        assert str(T.device) == "cpu"
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        writer = SummaryWriter(os.path.join(cfg.LOG_DIR, "train"))
        model = T.SC.DiSCO(output_dim=cfg.FEATURE_OUTPUT_DIM)
        corr2soft = T.SC.Corr2Softmax(200., 0.)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               T.get_learning_rate(0))
        opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, corr2soft.parameters()), lr=1e-3)
        model = T.nn.DataParallel(model)
        w0 = [p.detach().clone() for p in model.parameters()]
        t0 = time.time()
        T.train_one_epoch(model, opt, corr2soft, opt2, writer, T.SC_loss.quadruplet_loss, 0)
        t_e0 = time.time() - t0
        T.TRAINING_QUERIES = queries
        t0 = time.time()
        T.train_one_epoch(model, opt, corr2soft, opt2, writer, T.SC_loss.quadruplet_loss, 6)
        t_e6 = time.time() - t0
        assert np.shape(T.TRAINING_LATENT_VECTORS) == (len(queries), 1024)
        assert any(not torch.equal(a, b) for a, b in zip(w0, model.parameters()))
        ck = torch.load(Path(td) / "model.ckpt", map_location="cpu", weights_only=False)
        assert set(ck) == {"epoch", "iter", "state_dict", "optimizer", "corr2soft"}
        assert ck["epoch"] == 6
        writer.close()
        log = (Path(td) / "log_train.txt").read_text()
        n_faulty = log.count("FAULTY TUPLE")
        losses = [float(x.split(":")[1]) for x in log.splitlines() if x.startswith("batch loss")]
        assert losses and all(np.isfinite(losses))
        assert n_faulty == small.n_faulty() + queries.n_faulty(), n_faulty
        assert T.TOTAL_ITERATIONS == len(small) + len(queries) - n_faulty
        D = descriptors(EV, model, bank, "train", "kitti", "05", n_per)   # validation path
        assert D.shape == (n_per, 1024) and np.isfinite(D).all()
        out["official_loop"] = {
            "keys": [len(small), len(queries)], "queries_trained": T.TOTAL_ITERATIONS,
            "faulty_tuples_logged": n_faulty, "first_losses": losses[:3],
            "last_losses": losses[-3:], "model_ckpt_epoch": ck["epoch"],
            "cpu_s_epoch0_26q": round(t_e0, 1), "cpu_s_epoch6_120q": round(t_e6, 1)}
        print(f"  official loop on CPU: {json.dumps(out['official_loop'])}", flush=True)


def _check_recall(out, torch, cfg, lp, EV, fake):
    from sklearn.neighbors import KDTree
    from baselines.eval_utils import _find_revisit_queries
    from _disco_official_prep import cache_path
    for sensor, seq in (("helipr", "Roundabout02"), ("kitti", "09"), ("mulran", "Sejong02")):
        P = np.load(cache_path("val", sensor, seq))["poses"].astype(np.float64)
        rng = np.random.default_rng(0)
        pos = P[:, :3, 3]
        f = np.concatenate([np.cos(pos[:, :2] / 7.0), np.sin(pos[:, :2] / 7.0)], 1)
        emb = (f @ rng.standard_normal((4, 64)) * 3
               + rng.standard_normal((len(P), 64))).astype(np.float32)
        r1, nq = recall_at_1_l2(emb, P)
        tree = KDTree(emb)
        qs = _find_revisit_queries(pos, RADIUS, SKIP)
        hit = 0
        for q, _ in qs:
            _, ind = tree.query(emb[q:q + 1], k=min(len(emb), 2 * SKIP + 2))
            j = [i for i in ind[0] if abs(i - q) > SKIP][0]
            hit += np.linalg.norm(pos[j] - pos[q]) <= RADIUS
        assert nq == len(qs) and abs(r1 - hit / len(qs)) < 1e-12, (r1, hit / len(qs))
        out[f"recall_{sensor}_{seq}"] = {"r1": r1, "n_queries": nq, "equal_kdtree": True}
        print(f"  recall {sensor}/{seq}: {r1:.4f} == KDTree ({nq} queries)", flush=True)


def _check_cost(out, torch, cfg, lp, EV, fake):
    full = full_train_queries_no_images()
    lp.load_pc_file = lambda fn: np.zeros((20, 40, 120), np.float32)
    rng = np.random.default_rng(1)
    ok_keys = [k for k in range(len(full))      # the official loop skips the others
               if len(dict.__getitem__(full[k], "positives")) >= 2]
    keys = rng.choice(ok_keys, 60, replace=False)
    with tempfile.TemporaryDirectory() as td:
        T = load_train_module(Path(td), [], full)
        t0 = time.time()
        for k in keys:
            lp.get_query_tuple(full[int(k)], 2, 18, full, hard_neg=[], other_neg=True)
        t_plain = (time.time() - t0) / len(keys)
        lat = rng.standard_normal((len(full), 1024)).astype(np.float32)
        T.TRAINING_LATENT_VECTORS = lat
        t0 = time.time()
        for k in keys[:20]:
            e = full[int(k)]
            random.shuffle(e["negatives"])
            negs = e["negatives"][0:4000]
            hn = T.get_random_hard_negatives(lat[int(k)], negs, 5)
            lp.get_query_tuple(e, 2, 18, full, hn, other_neg=True)
        t_hard = (time.time() - t0) / 20
        fast = exact_hard_negatives(T)
        n_eq, t_k, t_f = 0, 0.0, 0.0
        for k in keys[:20]:
            negs = full[int(k)]["negatives"][0:4000]
            t0 = time.time()
            a = T.get_random_hard_negatives(lat[int(k)], negs, 5)
            t_k += time.time() - t0
            t0 = time.time()
            b = fast(lat[int(k)], negs, 5)
            t_f += time.time() - t0
            n_eq += int(a == b)
        assert n_eq == 20, n_eq
    per_seq = {f"{s}/{q}": int(sum(len(dict.__getitem__(full[k], "positives")) < 2
                                   for k in range(off, off + len(X))))
               for (s, q), off, X in zip(TRAIN, full.offsets, full.X)}
    out["cpu_step_overhead_s"] = {"random_negatives": round(t_plain, 4),
                                  "hard_negatives": round(t_hard, 4),
                                  "official_kdtree_mining_only": round(t_k / 20, 4),
                                  "exact_brute_force_mining_only": round(t_f / 20, 4),
                                  "brute_force_equal_official": f"{n_eq}/20",
                                  "keys": len(full), "n_faulty": full.n_faulty(),
                                  "n_faulty_per_seq": per_seq}
    print(f"  CPU per step outside the network (72k keys): {out['cpu_step_overhead_s']}",
          flush=True)


def store_index(role, sensor, seq):
    """index.npz of a store (scan_ids, poses, heading; no images needed)."""
    from _disco_official_prep import store_dir
    return np.load(store_dir(role, sensor, seq) / "index.npz")


def full_train_queries_no_images():
    pos, head, keys = [], [], []
    for sensor, seq in TRAIN:
        idx = store_index("train", sensor, seq)
        pos.append(idx["poses"][:, :3, 3].astype(np.float64))
        head.append(idx["heading"])
        keys.append([f"train/{seq_name(sensor, seq)}/{i}" for i in range(len(pos[-1]))])
    return PDV2Queries(TRAIN, pos, head, keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", type=int, default=0,
                    help="first N keyframes of every train/val sequence (GPU timing test)")
    ap.add_argument("--smoke-epochs", type=int, nargs="+", default=[0, 6],
                    help="epoch numbers passed to the official train_one_epoch in smoke mode")
    ap.add_argument("--resume", default="", help="run dir to continue (needs resume.pth)")
    ap.add_argument("--fast-hard-negatives", action="store_true",
                    help="opt-in exact brute-force k-NN in place of the per-step KDTree "
                         "of get_random_hard_negatives (same indices; not the default)")
    ap.add_argument("--selection-out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parts", nargs="+", choices=SELFTEST_PARTS, default=list(SELFTEST_PARTS),
                    help="selftest parts")
    ap.add_argument("--out-json", type=Path, default=None, help="selftest report")
    args, official_argv = ap.parse_known_args()
    if args.selftest:
        selftest(args)
        return
    train(args, official_argv)


if __name__ == "__main__":
    main()
