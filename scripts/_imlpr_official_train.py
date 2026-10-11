#!/usr/bin/env python3
"""ImLPR official pipeline on split v2 -- part 2: training (2026-10-11).

Runs the OFFICIAL trainer of _external/ImLPR (commit in
_imlpr_official_riv.OFFICIAL_COMMIT, files unmodified) on the split-v2 TRAIN
keyframes.  Imported and executed as they are:
  misc/utils.TrainingParams (config parser and code defaults), models/ImLPR
  (DINOv2 ViT-S/14 from torch.hub + MultiConv adapters + SALAD),
  models/losses (TruncatedSmoothAP + Patch-InfoNCE), datasets/base_datasets
  .TrainingDataset (yaw roll, 28-px cylindrical pad, /255, three masks),
  datasets/samplers.BatchSampler, datasets/dataset_utils.make_dataloaders,
  training/train_utils (multistaged_training_step, CustomCosineWarmupScheduler),
  preprocess/generate_training_tuples_baseline_img.construct_query_dict
  (10 m positives / 30 m non-negatives on an XY KD-tree), training/train
  ._set_seed.
Changed, each marked in the code:
  1. config: official config/config_helipr.txt with five keys replaced
     (model_name, dataset_folder, load_weights = empty -> DINOv2 init as in
     the paper, num_workers, train_file); lr 5e-5 / Adam / cosine warm-up 10
     / 100 epochs / batch 2048 / split 64 are the released code values.
  2. tuples: construct_query_dict is called once per pose-frame group and
     the results are concatenated with an id offset (--pickle-mode global:
     each KITTI sequence alone, HeLiPR Bridge01-04 together = the official
     "one KD-tree over all runs" for runs that share a frame; per_sequence:
     one tree per sequence).
  3. collate: the official make_collate_fn source with one hunk replaced:
     Patch-InfoNCE back-projection uses the pair's own sensor window and
     range scale (GEOMETRY) instead of the single uFoV / dFoV / sensor_range
     of the one-sensor config (patched source is exec'd; selftest shows it
     equals the official collate on a one-sensor batch).
  4. evaluation inside training (train.py:98-104: every 5th epoch, keep
     `_best` when strictly better): the official evaluate() is called with a
     keyword it does not accept (TypeError) and reads HeLiPR Town pickles
     (test place).  It is replaced by split-v2 VAL, intra-sequence 5 m R@1,
     30-keyframe gap, sensor families weighted equally
     (scripts/_pdv2_c3._family_pooled); the official cadence and strict-'>'
     `_best` rule are kept.
  5. do_train is copied below (official body) to add --resume and to write
     weights under _external/imlpr_official_pdv2/runs/ instead of
     _external/ImLPR/weights/.
  6. multistaged_training_step: official source with a None guard before
     train_utils.py:117 (a batch without any Patch-InfoNCE patch pair makes
     the official step raise TypeError; the skipped backward has a zero
     gradient), see _patched_step.
Paper over code (project rule, 2026-10-11): where the released code and the
paper (arXiv 2505.18364) differ, the paper value is used and the code line it
overrides is named (PAPER_OVERRIDES below; disclosed in the selection record):
  - optimizer / lr: AdamW, 5e-4 (App. A) instead of Adam (misc/utils.py:80),
    5e-5 (config_helipr.txt:12); weight decay is not stated in the paper ->
    torch AdamW default, which is also what train.py:61-66 passes;
  - trained backbone part: NOT switched to the paper's "last two blocks"; the
    released weights match the code path (refined rule: released weights /
    reproducing code over paper), see BACKBONE (paper_freeze is optional);
  - loss: L_P + lambda * L_TSAP, lambda = 2 (§3.5, App. A) instead of
    L_TSAP + 2 * L_P (train_utils.py:96);
  - Patch-InfoNCE image pairs: 1/8 of the positive image pairs of the batch
    (App. A) instead of a fixed 216 (config_helipr.txt:19);
  - SALAD: the code layout (64 clusters x 128 dims) is kept because the
    released weights have it (paper App. A writes (m, l, e) = (128, 64, 256);
    IMLPR_AGGREGATOR=paper switches);
  - normal k = 25 (App. A) instead of 15 (_imlpr_official_riv.py).
Environment shims: np.NINF (removed in NumPy 2, used by
truncated_smoothap.py:26) is restored; the official top-level folders have
no __init__.py and `datasets` is shadowed by HuggingFace datasets, so they
are registered in sys.modules first.

Subcommands (inside ring_baseline; TORCH_HOME should point at the pinned
DINOv2 hub copy, see scripts/_pdv2_imlpr_official_run.sh):
  pickle [--pickle-mode global|per_sequence]   training tuples (+ manifest)
  config [--pickle-mode M] [--num-workers N]   write config_pdv2.txt
  train [--resume]                            100 epochs (resumable)
  smoke [--val-frames N]                      2 steps of full batches + val
  selftest [--parts pickle,collate,dryrun]    CPU checks
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import inspect
import json
import os
import pickle
import random
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from _imlpr_official_riv import (  # noqa: E402
    DATA, OFFICIAL, OFFICIAL_COMMIT, SENSORS, TRAIN_H, WORK, role_dir, seq_name, sha256)
from _pdv2_pilot import TRAIN, VAL  # noqa: E402

RESULTS = REPO / "results/pdv2_imlpr_official"
SELECTION = RESULTS / "train_selection.json"
CONFIG = WORK / "config_pdv2.txt"
OFFICIAL_MODEL_CONFIG = OFFICIAL / "config/config_model_imlpr.txt"
MODEL_CONFIG = WORK / "config_model_pdv2.txt"     # written by model_config_text()
RUNS = WORK / "runs"
MODEL_NAME = "ImLPR_pdv2_official"
SEED = 1
PAD = 28                     # evaluate.py:92 / base_datasets.py:25
FAISS_BLOCK = 19             # < faiss BLAS threshold (20): bit-identical to one-by-one
SKIP, RADIUS = 30, 5.0
# Patch-InfoNCE back-projection per sensor: (uFoV, dFoV, sensor_range), the
# window and range scale the RIVs were rendered with (_imlpr_official_riv).
GEOMETRY = {s: (c["fov_up"], -c["fov_down"], c["max_range"]) for s, c in SENSORS.items()}
PAIR_GEOMETRY = os.environ.get("IMLPR_PAIR_GEOMETRY", "sensor")   # sensor | config (D6)


def sensor_of(rel_path: str) -> str:
    """'training/kitti_05/000123.npy' -> 'kitti'."""
    return Path(rel_path).parent.name.split("_", 1)[0]


# ------------------------------------------------------------ official code
_OFF = None


def load_official():
    global _OFF
    if _OFF is not None:
        return _OFF
    for name, val in (("NINF", -np.inf),):     # NumPy-2 alias shim (value unchanged)
        if not hasattr(np, name):
            setattr(np, name, val)
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    for pkg in ("misc", "models", "datasets", "eval", "training"):
        spec = importlib.machinery.ModuleSpec(pkg, None, is_package=True)
        spec.submodule_search_locations = [str(OFFICIAL / pkg)]
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(OFFICIAL / pkg)]
        mod.__spec__ = spec
        sys.modules[pkg] = mod
    import datasets.base_datasets as BD
    import datasets.dataset_utils as DU
    import datasets.samplers as SA
    import misc.utils as MU
    import models.ImLPR as MI
    import models.losses.loss as LO
    import training.train as TR
    import training.train_utils as TU
    mods = dict(BD=BD, DU=DU, SA=SA, MU=MU, MI=MI, LO=LO, TR=TR, TU=TU,
                make_collate_fn=DU.make_collate_fn,   # official, kept before patching
                multistaged_training_step=_patched_step(TU))
    for m in mods.values():
        if isinstance(m, types.ModuleType):
            assert Path(m.__file__).resolve().is_relative_to(OFFICIAL.resolve()), m.__file__
    spec = importlib.util.spec_from_file_location(
        "imlpr_generate_training_tuples",
        OFFICIAL / "preprocess/generate_training_tuples_baseline_img.py")
    TG = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TG)
    mods["TG"] = TG
    _OFF = types.SimpleNamespace(**mods)
    return _OFF


_STEP_OLD = """                local_feat.backward(gradient=all_local_feats_grad[i:i + minibatch_size])
"""
_STEP_NEW = """                if all_local_feats_grad is not None:  # PDV2 CHANGED (see _patched_step)
                    local_feat.backward(gradient=all_local_feats_grad[i:i + minibatch_size])
"""
_STEP_OLD2 = """    # Stage 3: recompute descriptors with grad, apply cached grads
"""
_STEP_OLD3 = """        loss = loss_truncated + 2 * loss_contrastive
"""
_STEP_NEW3 = """        loss = loss_contrastive + 2 * loss_truncated  # PDV2 CHANGED (paper: L_P + lambda * L_TSAP, lambda = 2)
"""
_STEP_NEW2 = """    # Stage 3: recompute descriptors with grad, apply cached grads
    if phase == 'train' and all_local_feats_grad is None:  # PDV2 ADDED (log)
        print("[pdv2] no Patch-InfoNCE patch pair in this batch: local-feature gradient is zero", flush=True)
"""


def _patched_step(TU):
    """training/train_utils.multistaged_training_step with one guard: when no
    image pair of the batch yields a patch pair, PointInfoNCELoss returns a
    constant 0 (pointinfonce.py:141-142), all_local_feats.grad stays None and
    the official line 117 raises TypeError.  The guard skips that backward,
    whose gradient would be zero.  Paper over code: the loss is
    L_P + 2 * L_TSAP (paper §3.5, App. A) instead of line 96's
    L_TSAP + 2 * L_P.  Everything else is the official source."""
    src = inspect.getsource(TU.multistaged_training_step)
    for old, new in ((_STEP_OLD, _STEP_NEW), (_STEP_OLD2, _STEP_NEW2), (_STEP_OLD3, _STEP_NEW3)):
        assert src.count(old) == 1, "official step changed"
        src = src.replace(old, new)
    g = dict(vars(TU))
    exec(compile(src, f"<patched {TU.__file__}:multistaged_training_step>", "exec"), g)
    return g["multistaged_training_step"]


# ------------------------------------------------------------ tuples (pickle)
def read_train_poses(root: Path, folder: str):
    """generate_training_tuples_baseline_img.py:80-91 for one run folder, with
    the README layout (files relative to dataset_folder: training/<run>/)."""
    import pandas as pd
    df = pd.read_csv(root / "training" / folder / "poses.txt", sep=r"\s+", header=None,
                     names=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    rel_prefix = "training/" + folder + "/"
    files = (np.arange(len(df)).astype(int)).astype(str)
    files = np.char.zfill(files, 6)
    df['file'] = rel_prefix + pd.Series(files) + ".npy"
    return df[['file', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]


def pose_groups(names: list[str], mode: str) -> list[list[str]]:
    """global: runs that share a pose frame form one KD-tree (official: all
    runs in one tree).  KITTI sequences each have their own frame (and are
    >= 49.8 m apart in UTM, results/split_audit/geo_overlap.json, so no pair
    within 30 m is lost); HeLiPR Bridge01-04 share one frame (GT starts at
    different coordinates; Bridge01/04 and 02/03 drive the same roads)."""
    if mode == "per_sequence":
        return [[n] for n in names]
    assert mode == "global", mode
    out = [[n] for n in names if not n.startswith("helipr_Bridge")]
    bridge = [n for n in names if n.startswith("helipr_Bridge")]
    return out + ([bridge] if bridge else [])


def build_tuples(root: Path, names: list[str], mode: str, out_path: Path) -> dict:
    """Official construct_query_dict per group, ids offset into one dict."""
    import pandas as pd
    off = load_official()
    dfs = {n: read_train_poses(root, n) for n in names}
    merged, offset, groups = {}, 0, pose_groups(names, mode)
    order = [n for g in groups for n in g]
    assert order == names, "group order must keep the sequence order"
    with tempfile.TemporaryDirectory() as td:
        for g in groups:
            df = pd.concat([dfs[n] for n in g], ignore_index=True)
            tmp = Path(td) / "q.pickle"
            off.TG.construct_query_dict(df, str(tmp))        # official
            with open(tmp, "rb") as f:
                q = pickle.load(f)
            assert sorted(q) == list(range(len(df)))
            for i in range(len(df)):
                d = dict(q[i])
                d["id"] = d["id"] + offset
                d["positives"] = [int(p) + offset for p in d["positives"]]
                d["non_negatives"] = [int(p) + offset for p in d["non_negatives"]]
                merged[d["id"]] = d
            offset += len(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # statistics: cross-sequence positives / non-negatives
    seq_of = np.concatenate([[k] * len(dfs[n]) for k, n in enumerate(names)])
    cross_pos = [sum(seq_of[p] != seq_of[i] for p in d["positives"]) for i, d in merged.items()]
    cross_nn = [sum(seq_of[p] != seq_of[i] for p in d["non_negatives"]) for i, d in merged.items()]
    return {"mode": mode, "groups": groups, "n_tuples": len(merged),
            "frames_per_sequence": {n: len(dfs[n]) for n in names},
            "no_positive": int(sum(len(d["positives"]) == 0 for d in merged.values())),
            "mean_positives": float(np.mean([len(d["positives"]) for d in merged.values()])),
            "frac_with_cross_seq_positive": float(np.mean(np.asarray(cross_pos) > 0)),
            "mean_cross_seq_positives": float(np.mean(cross_pos)),
            "mean_cross_seq_non_negatives": float(np.mean(cross_nn))}


def pickle_name(mode: str) -> str:
    return f"training_pdv2_{mode}.pickle"


def cmd_pickle(mode: str):
    names = [seq_name(s, q) for s, q in TRAIN]
    for s, q in TRAIN:
        idx = np.load(role_dir("train", s, q) / "index.npz")
        n = len(idx["scan_ids"])
        assert all((role_dir("train", s, q) / f"{p:06d}.npy").exists() for p in range(n)), \
            f"train RIVs of {s}/{q} incomplete"
    out = DATA / pickle_name(mode)
    man = build_tuples(DATA, names, mode, out)
    man.update(official="preprocess/generate_training_tuples_baseline_img.construct_query_dict "
                        f"@ {OFFICIAL_COMMIT[:12]}", pickle=str(out.relative_to(REPO)),
               sha256=sha256(out))
    (DATA / (pickle_name(mode) + ".manifest.json")).write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "groups"}, indent=1))
    print("PICKLE_DONE", flush=True)


# ------------------------------------------------------------ config
CONFIG_CHANGES = [   # (official line, our line) in config/config_helipr.txt
    ("model_name = ImLPR_training", f"model_name = {MODEL_NAME}"),
    ("dataset_folder = ../data/", "dataset_folder = data/"),        # CWD = WORK
    ("load_weights =../weights/ImLPR_default.pth", "load_weights ="),
    ("lr = 5e-5", "lr = 5e-4\noptimizer = AdamW"),                  # paper App. A
    ("num_workers = 16", None),                                       # filled in
    ("train_file = training_helipr.pickle", None),                    # filled in
]


def config_text(mode: str, num_workers: int) -> str:
    txt = (OFFICIAL / "config/config_helipr.txt").read_text()
    for old, new in CONFIG_CHANGES:
        if old.startswith("num_workers"):
            new = f"num_workers = {num_workers}"
        if old.startswith("train_file"):
            new = f"train_file = {pickle_name(mode)}"
        assert txt.count(old) == 1, old
        txt = txt.replace(old, new)
    return txt if txt.endswith("\n") else txt + "\n"


# Released-weights evidence over paper (rule refined 2026-10-11): ImLPR_default.pth
# has 64 clusters x 128 dims -> the code layout is the default.
AGGREGATOR = os.environ.get("IMLPR_AGGREGATOR", "code")    # code | paper
# Trained backbone part: the released weights reproduce the code path (task
# gradients in blocks 0-8, 10, 11 and the embeddings, none in block 9) -> no
# freezing by default; IMLPR_BACKBONE=paper applies paper_freeze().
BACKBONE = os.environ.get("IMLPR_BACKBONE", "code")        # code | paper


def model_config_text() -> str:
    """config/config_model_imlpr.txt; paper App. A (m, l, e) = (128, 64, 256)
    with V in R^{m x l} and m = number of clusters (§3.4) replaces
    num_clusters=64 / cluster_dim=128 (lines 8-9).  The released
    ImLPR_default.pth has 64 clusters x 128 dims (score.2: 64 rows,
    cluster_features.2: 128 rows): IMLPR_AGGREGATOR=code keeps the code."""
    txt = OFFICIAL_MODEL_CONFIG.read_text()
    if AGGREGATOR == "code":
        return txt if txt.endswith("\n") else txt + "\n"
    assert AGGREGATOR == "paper", AGGREGATOR
    for old, new in (("num_clusters=64", "num_clusters=128"), ("cluster_dim=128", "cluster_dim=64")):
        assert txt.count(old) == 1, old
        txt = txt.replace(old, new)
    return txt if txt.endswith("\n") else txt + "\n"


def ensure_model_config(path: Path = None) -> Path:
    path = path or MODEL_CONFIG
    txt = model_config_text()
    if not path.exists() or path.read_text() != txt:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(txt)
    return path


def paper_freeze(model) -> dict:
    """Paper §3.3 / §4.1 / App. A: only the last two DINOv2 blocks are trained.
    Frozen here: patch_embed, cls_token, pos_embed, mask_token and
    blocks[:-num_trainable_blocks].  Trained: the last blocks, the final norm
    (applied after them, as in SALAD's DINOv2 wrapper this code follows),
    the MultiConv adapters and SALAD.  The code itself (models/ImLPR.py:
    130-144) also updates blocks 0-8 and the patch embedding through the
    adapter inputs.  Frozen parameters never get a gradient, so the official
    optimizer (built on model.parameters()) skips them."""
    enc = model.encoder.encoder
    dino, n_train = enc.model, enc.num_trainable_blocks
    first_trained = len(dino.blocks) - n_train
    frozen, trained = 0, 0
    for name, p in dino.named_parameters():
        keep = name.startswith("norm.") or (
            name.startswith("blocks.") and int(name.split(".")[1]) >= first_trained)
        if not keep:
            p.requires_grad_(False)
            frozen += p.numel()
        else:
            trained += p.numel()
    return {"frozen_dinov2_params": frozen, "trained_dinov2_params": trained,
            "trained_blocks": list(range(first_trained, len(dino.blocks)))}


def cmd_config(mode: str, num_workers: int):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(config_text(mode, num_workers))
    ensure_model_config()
    print(CONFIG.read_text())
    print(MODEL_CONFIG.read_text())
    print(f"CONFIG_WRITTEN {CONFIG}", flush=True)


# ------------------------------------------------------------ collate patch
_COLLATE_OLD = """            query_range = (clouds_np[i][1][:, 28:-28]) * params.sensor_range
            positive_range = (clouds_np[j][1][:, 28:-28]) * params.sensor_range
"""
_COLLATE_OLD2 = """        sampled_pairs = sample_positive_pairs(positives_mask.numpy(), poses, params.num_positive_pairs)
"""
_COLLATE_NEW2 = """        _n_pos_pairs = len(extract_unique_positive_pairs(positives_mask.numpy()))  # PDV2 CHANGED (paper App. A)
        sampled_pairs = sample_positive_pairs(positives_mask.numpy(), poses,  # PDV2 CHANGED: 1/8 of the
                                              max(1, int(round(_n_pos_pairs / 8))))  # positive image pairs
"""
_COLLATE_NEW = """            _uFoV, _dFoV, _srange = _PDV2_PAIR_GEOMETRY(dataset, labels[i], labels[j])  # PDV2 CHANGED
            elevation_angles = np.linspace(_uFoV * np.pi / 180.0,  # PDV2 CHANGED (per pair)
                                           _dFoV * np.pi / 180.0,
                                           params.image_W, endpoint=False)
            query_range = (clouds_np[i][1][:, 28:-28]) * _srange  # PDV2 CHANGED
            positive_range = (clouds_np[j][1][:, 28:-28]) * _srange  # PDV2 CHANGED
"""


def make_collate_fn_pdv2(dataset, params, geometry=None, timed=True, fraction=True):
    """Official make_collate_fn source with two hunks replaced:
    (1) pair geometry: the pair's sensor window / range scale (geometry=None ->
        GEOMETRY; IMLPR_PAIR_GEOMETRY=config -> the config values, i.e. the
        official behaviour); (2) fraction=True: 1/8 of the positive image
        pairs of the batch (paper App. A) instead of params.num_positive_pairs."""
    off = load_official()
    if geometry is None and PAIR_GEOMETRY == "config":
        geometry = {s: (params.uFoV, params.dFoV, params.sensor_range) for s in GEOMETRY}
    geometry = geometry or GEOMETRY
    src = inspect.getsource(off.make_collate_fn)
    hunks = [(_COLLATE_OLD, _COLLATE_NEW)] + ([(_COLLATE_OLD2, _COLLATE_NEW2)] if fraction else [])
    for old, new in hunks:
        assert src.count(old) == 1, "official collate changed"
        src = src.replace(old, new)

    def pair_geometry(ds, li, lj):
        si = sensor_of(ds.queries[li].rel_scan_filepath)
        sj = sensor_of(ds.queries[lj].rel_scan_filepath)
        assert si == sj, (si, sj)          # positives never cross sensors
        return geometry[si]

    g = dict(vars(off.DU))
    g["_PDV2_PAIR_GEOMETRY"] = pair_geometry
    exec(compile(src, f"<patched {off.DU.__file__}:make_collate_fn>", "exec"), g)
    fn = g["make_collate_fn"](dataset, params)
    if not timed:
        return fn

    def collate(data_list):
        t0 = time.time()
        out = fn(data_list)
        print(f"[pdv2] collate {len(data_list)} items, {len(out[3])} pairs: "
              f"{time.time() - t0:.1f} s", flush=True)
        return out
    return collate


def install_collate_patch():
    """IMLPR_PAIR_GEOMETRY=sensor (default): per-pair sensor geometry;
    =config: the config uFoV/dFoV/sensor_range for every pair (official).
    The paper's 1/8 rule for Patch-InfoNCE image pairs applies in both."""
    off = load_official()
    assert PAIR_GEOMETRY in ("sensor", "config"), PAIR_GEOMETRY
    off.DU.make_collate_fn = lambda dataset, params: make_collate_fn_pdv2(dataset, params)


# ------------------------------------------------------------ v2 validation
def eval_input(img: np.ndarray) -> np.ndarray:
    """eval/evaluate.py:90-100: cylindrical pad 28, rows -> 126 (INTER_LINEAR,
    RESIZE_TO_HELIPR), CHW float32 / 255."""
    import cv2
    left, right = img[:, :PAD], img[:, -PAD:]
    img = np.concatenate([right, img, left], axis=1)
    if img.shape[0] != TRAIN_H:
        H, W, _ = img.shape
        img = cv2.resize(img, (W, TRAIN_H), interpolation=cv2.INTER_LINEAR)
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


class _Images:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return eval_input(np.load(self.paths[i]))


def encode(model, device, paths, batch=32, workers=4):
    """Descriptors of RIV files with the official eval input and forward
    (evaluate.py:_get_latent_vectors/_flush_batch: model.eval(), no_grad,
    `_, desc = model(x)`); batched instead of val_batch_size = 1."""
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    dl = DataLoader(_Images(paths), batch_size=batch, num_workers=workers, shuffle=False)
    out = []
    with torch.no_grad():
        for x in dl:
            _, desc = model(x.to(device))
            out.append(desc.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 8448), np.float32)


def recall_at_1(desc, poses, per_query_records=None):
    """baselines.eval_utils.compute_recall_multi_k(k=[1], 5 m, skip 30) with the
    faiss search issued 19 queries at a time (same results, checked by the
    BEVPlace++ driver's selftest and again here)."""
    import faiss
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


class V2Val:
    """Replacement of eval.evaluate.evaluate inside training: family-equal
    R@1 on the split-v2 VAL keyframes."""

    def __init__(self, max_frames=0, batch=32, workers=4, sets=None):
        self.batch, self.workers, self.last, self.calls = batch, workers, None, []
        if sets is None:
            sets = []
            for s, q in VAL:
                d = role_dir("val", s, q)
                idx = np.load(d / "index.npz")
                n = len(idx["scan_ids"]) if not max_frames else min(max_frames, len(idx["scan_ids"]))
                sets.append((f"{s}/{q}", [str(d / f"{p:06d}.npy") for p in range(n)],
                             idx["poses_cache"][:n]))
        self.sets = sets

    def __call__(self, model, device, params, **kwargs):
        from _pdv2_c3 import _family_pooled
        t0, per = time.time(), {}
        for name, paths, poses in self.sets:
            desc = encode(model, device, paths, self.batch, self.workers)
            r1, nq = recall_at_1(desc, poses)
            per[name] = {"r1": r1, "n_queries": nq, "n_frames": len(paths)}
            print(f"    val {name}: R@1 {r1:.4f} ({nq} queries, {len(paths)} frames)", flush=True)
        fam = _family_pooled(per) if any(v["n_queries"] for v in per.values()) else 0.0
        self.last = {"family_macro_r1": fam, "per_seq": per, "val_s": round(time.time() - t0, 1)}
        self.calls.append(self.last)
        print(f"[pdv2] v2-val family-equal R@1 {fam:.4f} ({time.time() - t0:.0f} s)", flush=True)
        return {"ave_recall": fam, "ave_one_percent_recall": 0.0}


# ------------------------------------------------------------ training
def _rng_state():
    import torch
    return {"random": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(st):
    import torch
    random.setstate(st["random"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def do_train_pdv2(params, evaluate, run_dir: Path, record: dict, record_path: Path,
                  resume: bool = False):
    """training/train.py:do_train (lines 16-169), copied; changes marked."""
    import torch
    import tqdm
    off = load_official()
    TU = off.TU
    # ----- Model -----
    from models.ImLPR import ImLPR
    model = ImLPR(params.model_params)
    if BACKBONE == "paper":                                      # CHANGED (optional, see BACKBONE)
        record["_meta"]["paper_freeze"] = paper_freeze(model)
        print(f"[pdv2] paper_freeze: {record['_meta']['paper_freeze']}", flush=True)
    else:
        assert BACKBONE == "code", BACKBONE
        record["_meta"]["paper_freeze"] = None

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_name = params.model_name
    print(f'Model name: {model_name}')
    run_dir.mkdir(parents=True, exist_ok=True)                 # CHANGED (weights folder)
    model_pathname = os.path.join(str(run_dir), model_name)

    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum(p.nelement() for p in model.parameters())
        print(f'Number of model parameters: {n_params}')
    print(f'Model device: {device}')

    # ----- Data -----
    dataloaders = off.DU.make_dataloaders(params)

    # ----- Loss(es) -----
    loss_fn = off.LO.make_losses(params)

    # ----- Optionally load previous weights -----
    prev_path = params.load_weights
    if os.path.exists(prev_path):
        previous_model = torch.load(prev_path, map_location="cpu")
        if torch.cuda.device_count() <= 1:
            previous_model = {k[7:] if k.startswith("module.") else k: v for k, v in previous_model.items()}
        missing = model.load_state_dict(previous_model, strict=False)
        print("[load] missing:", getattr(missing, "missing_keys", None))
        print("[load] unexpected:", getattr(missing, "unexpected_keys", None))
    else:
        print(f"[info] previous weights not found: {prev_path}")

    # ----- Optimizer -----
    if params.optimizer == 'Adam':
        optimizer_fn = torch.optim.Adam
    elif params.optimizer == 'AdamW':
        optimizer_fn = torch.optim.AdamW
    else:
        raise NotImplementedError(f"Unsupported optimizer: {params.optimizer}")

    optimizer = optimizer_fn(model.parameters(), lr=params.lr)

    # ----- Scheduler -----
    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == "CustomCosineWarmupScheduler":
            scheduler = TU.CustomCosineWarmupScheduler(optimizer, warmup=params.warmup_epochs, max_iters=params.epochs, min_value=0.1)
        else:
            raise NotImplementedError(f'Unsupported LR scheduler: {params.scheduler}')

    # ----- Step function (single vs multi-stage) -----
    if params.batch_split_size is None or params.batch_split_size == 0:
        train_step_fn = TU.training_step
    else:
        train_step_fn = off.multistaged_training_step          # CHANGED (guard, see _patched_step)

    # ----- Book-keeping -----
    stats = {'train': [], 'eval': []}
    phases = ['train', 'val'] if 'val' in dataloaders else ['train']
    if 'val' in dataloaders:
        stats['val'] = []

    best_recall = 0.0
    intra_flg = False

    start_epoch = 1                                             # CHANGED (resume)
    resume_path = run_dir / "resume.pth"
    if resume and resume_path.exists():
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ck["scheduler"])
        best_recall, stats, start_epoch = ck["best_recall"], ck["stats"], ck["epoch"] + 1
        _set_rng_state(ck["rng"])
        record["evals"] = [e for e in record["evals"] if e["at_start_of_epoch"] < start_epoch]
        record["epochs"] = [e for e in record["epochs"] if e["epoch"] < start_epoch]
        record.setdefault("resumed_at_epoch", []).append(start_epoch)
        print(f"[pdv2] resumed after epoch {ck['epoch']} (best {best_recall:.4f})", flush=True)

    # ----- Epoch loop -----
    for epoch in tqdm.tqdm(range(start_epoch, params.epochs + 1)):
        metrics = {'train': {}, 'val': {}}
        t_epoch = time.time()

        # Optional eval cadence
        if epoch % 5 == 0:
            intra_flg = (epoch % 1000 == 1)
            resuls_eval = evaluate(model, device, params, intra_flg=intra_flg)   # CHANGED (v2 val)
            is_best = resuls_eval.get('ave_recall', 0.0) > best_recall
            if is_best:
                best_recall = resuls_eval['ave_recall']
                torch.save(model.state_dict(), model_pathname + "_best.pth")
                print('Best model saved')
            record["evals"].append({"at_start_of_epoch": epoch, "trained_epochs": epoch - 1,
                                    "is_best": bool(is_best), **evaluate.last})   # ADDED
            _write_json(record_path, record)

        for phase in phases:
            running_stats, count_batches = [], 0
            global_iter = iter(dataloaders[phase]) if phase in dataloaders and dataloaders[phase] is not None else None

            while True:
                count_batches += 1
                if params.debug and count_batches > 2:
                    break
                try:
                    t_step = time.time()
                    temp_stats = train_step_fn(global_iter, model, phase, device, optimizer, loss_fn)
                    running_stats.append({'global': temp_stats})
                    print(f"[pdv2] epoch {epoch} step {count_batches}: {time.time() - t_step:.1f} s "
                          "(incl. waiting for the collate)", flush=True)   # ADDED
                except StopIteration:
                    break

            # Aggregate epoch stats
            epoch_stats = {}
            if running_stats:
                for substep in running_stats[0]:
                    epoch_stats[substep] = {}
                    for key in running_stats[0][substep]:
                        temp = [e[substep][key] for e in running_stats]
                        if isinstance(temp[0], dict):
                            epoch_stats[substep][key] = {k: np.mean([t[k] for t in temp]) for k in temp[0]}
                        elif isinstance(temp[0], np.ndarray):
                            epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                        else:
                            epoch_stats[substep][key] = np.mean(temp)

                stats[phase].append(epoch_stats)
                TU.print_stats(phase, epoch_stats)

                # For external logging (e.g., wandb)
                metrics[phase]['loss1'] = epoch_stats['global']['loss']
                if 'num_non_zero_triplets' in epoch_stats['global']:
                    metrics[phase]['active_triplets1'] = epoch_stats['global']['num_non_zero_triplets']
                if 'positive_ranking' in epoch_stats['global']:
                    metrics[phase]['positive_ranking'] = epoch_stats['global']['positive_ranking']
                if 'recall' in epoch_stats['global']:
                    metrics[phase]['recall@1'] = epoch_stats['global']['recall'][1]
                if 'ap' in epoch_stats['global']:
                    metrics[phase]['AP'] = epoch_stats['global']['ap']

        # Finalize epoch
        if scheduler is not None:
            scheduler.step()

        if params.save_freq > 0 and epoch % params.save_freq == 0:
            torch.save(model.state_dict(), model_pathname + f"_{epoch}.pth")

        if params.batch_expansion_th is not None and stats['train']:
            le_train_stats = stats['train'][-1]
            rnz = le_train_stats['global']['num_non_zero_triplets'] / le_train_stats['global']['num_triplets']
            if rnz < params.batch_expansion_th:
                dataloaders['train'].batch_sampler.expand_batch()

        tmp = run_dir / "resume.tmp"                             # ADDED (resume state)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "best_recall": best_recall, "stats": stats, "rng": _rng_state()}, tmp)
        os.replace(tmp, resume_path)
        record["epochs"].append({"epoch": epoch, "s": round(time.time() - t_epoch),
                                 "lr": optimizer.param_groups[0]["lr"],
                                 "train": {k: (float(v) if np.isscalar(v) else None)
                                           for k, v in (stats['train'][-1]['global'].items()
                                                        if stats['train'] else [])},
                                 "cuda_peak_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                                                  if torch.cuda.is_available() else None)})
        _write_json(record_path, record)

    # Save final model
    final_model_path = model_pathname + '_final.pth'
    print(f"Saving weights: {final_model_path}")
    torch.save(model.state_dict(), final_model_path)

    # Final evaluation
    stats_eval = evaluate(model, device, params, intra_flg=intra_flg)
    off.TR.print_eval_stats(stats_eval)
    record["final"] = {"checkpoint": final_model_path, **evaluate.last}   # ADDED
    print('.')
    return model


PAPER_OVERRIDES = {
    "optimizer_lr": "AdamW 5e-4 (paper App. A) over Adam 5e-5 (misc/utils.py:80, config_helipr.txt:12)",
    "trained_backbone": "code path kept (models/ImLPR.py:130-144: blocks 0-8, 10, 11, embeddings "
                        "get gradients, block 9 none): it reproduces the released weights; the "
                        "paper's 'last two blocks only' is contradicted by them (IMLPR_BACKBONE)",
    "loss_weighting": "L_P + 2 L_TSAP (paper §3.5, App. A) over L_TSAP + 2 L_P (train_utils.py:96)",
    "patch_image_pairs": "1/8 of the positive image pairs per batch (paper App. A) over 216 "
                         "(config_helipr.txt:19)",
    "aggregator": "code layout 64 clusters x 128 dims kept: released weights have it; the paper's "
                  "(m, l) = (128, 64) is not used (IMLPR_AGGREGATOR)",
    "optimizer_evidence": "released weights: max |w - w_pretrained| per DINOv2 block 0.2099-0.2143 = "
                          "sum of lr over the run for lr 5e-4, 9 steps/epoch, 100 epochs, code "
                          "scheduler (0.2130); lr 5e-5 would give 0.021",
    "normal_k": "25 (paper App. A) over 15 (generate_riv_images.py:431)",
}


def model_optimizer_probe(params):
    """The optimizer class train.py:59-66 builds for these params."""
    import torch
    cls = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[params.optimizer]
    return cls([torch.nn.Parameter(torch.zeros(1))], lr=params.lr)


def _meta(mode, cfg_text, run_dir, smoke=False):
    man_p = DATA / (pickle_name(mode) + ".manifest.json")
    return {"official_repo": "https://github.com/minwoo0611/ImLPR",
            "official_commit": OFFICIAL_COMMIT, "driver": "scripts/_imlpr_official_train.py",
            "seed": SEED, "smoke": smoke, "run_dir": str(run_dir), "config": cfg_text,
            "pickle_mode": mode,
            "pickle_manifest": json.loads(man_p.read_text()) if man_p.exists() else None,
            "patch_infonce_geometry": ({s: GEOMETRY[s] for s in ("kitti", "helipr")}
                                       if PAIR_GEOMETRY == "sensor" else "config values"),
            "config_uFoV_dFoV_sensor_range": "kept at the official values; superseded per "
                                              "pair by patch_infonce_geometry (collate patch)",
            "val_sequences": [f"{s}/{q}" for s, q in VAL],
            "paper_overrides": PAPER_OVERRIDES,
            "aggregator": AGGREGATOR, "backbone": BACKBONE, "model_config": model_config_text(),
            "selection_rule": ("official train.py cadence: evaluate at the start of every 5th "
                               "epoch (state after epoch-1 epochs), keep `_best.pth` when "
                               "strictly better; metric = split-v2 VAL intra-sequence 5 m R@1, "
                               "30-keyframe gap, sensor families (kitti+kitti360, helipr, "
                               "mulran) weighted equally, queries pooled in a family"),
            "torch_home": os.environ.get("TORCH_HOME")}


def _params(mode: str):
    off = load_official()
    os.chdir(WORK)                       # dataset_folder = data/ (relative, as official ../data/)
    cfg_text = CONFIG.read_text()
    assert f"train_file = {pickle_name(mode)}" in cfg_text, \
        f"{CONFIG} was written for another pickle mode"
    params = off.MU.TrainingParams(str(CONFIG), str(ensure_model_config()), debug=False)
    return params, cfg_text


def cmd_train(mode: str, resume: bool, val_batch: int, val_workers: int):
    import torch
    off = load_official()
    params, cfg_text = _params(mode)
    run_dir = RUNS / MODEL_NAME
    if resume and SELECTION.exists():
        record = json.loads(SELECTION.read_text())
        assert not record.get("finished"), "training already finished"
    else:
        assert not (run_dir / "resume.pth").exists(), f"{run_dir} has a run; use --resume"
        record = {"_meta": _meta(mode, cfg_text, run_dir), "evals": [], "epochs": []}
    off.TR._set_seed(SEED, deterministic=False)   # train.py:_set_seed (official CLI --seed)
    params.print()
    install_collate_patch()
    evaluate = V2Val(batch=val_batch, workers=val_workers)
    do_train_pdv2(params, evaluate, run_dir, record, SELECTION, resume=resume)
    best = run_dir / f"{MODEL_NAME}_best.pth"
    final = run_dir / f"{MODEL_NAME}_final.pth"
    be = [e for e in record["evals"] if e["is_best"]]
    record["best"] = {"checkpoint": str(best), "sha256": sha256(best),
                      "at_start_of_epoch": be[-1]["at_start_of_epoch"],
                      "family_macro_r1": be[-1]["family_macro_r1"]}
    record["final"]["sha256"] = sha256(final)
    record["finished"] = True
    record["torch"] = torch.__version__
    _write_json(SELECTION, record)
    print("IMLPR_OFFICIAL_TRAIN_DONE", flush=True)


def cmd_smoke(mode: str, val_frames: int, val_batch: int, val_workers: int):
    """Two full-size training steps (official debug path: 2 batches) and one
    val pass on the first val_frames keyframes of every val sequence:
    step time, collate time, GPU memory, val time.  Separate run dir."""
    import torch
    off = load_official()
    params, cfg_text = _params(mode)
    params.epochs, params.debug = 1, True
    run_dir = WORK / "smoke/runs"
    out = WORK / "smoke/train_selection.json"
    record = {"_meta": _meta(mode, cfg_text, run_dir, smoke=True), "evals": [], "epochs": []}
    off.TR._set_seed(SEED, deterministic=False)
    install_collate_patch()
    evaluate = V2Val(max_frames=val_frames, batch=val_batch, workers=val_workers)
    t0 = time.time()
    do_train_pdv2(params, evaluate, run_dir, record, out, resume=False)
    record["smoke_total_s"] = round(time.time() - t0)
    record["cuda_peak_gb"] = (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                              if torch.cuda.is_available() else None)
    _write_json(out, record)
    print(json.dumps({"epoch": record["epochs"], "val_s": record["final"]["val_s"],
                      "cuda_peak_gb": record["cuda_peak_gb"]}, indent=1, default=float))
    print("IMLPR_OFFICIAL_SMOKE_DONE", flush=True)


# ------------------------------------------------------------ CPU selftest
def _selftest_frames(td: Path, n_per_seq: int):
    """Real training RIVs for a few consecutive 3 m-subsampled keyframes of
    KITTI 05 and HeLiPR Bridge01/04 (accumulated), README layout under td."""
    from _imlpr_official_riv import (Track, cache_path, fast_frame, frame_points,
                                     lidar_poses, make_loader, select_frame_indices,
                                     to_train_height)
    from scipy.spatial.transform import Rotation
    names = []
    for s, q, start in (("kitti", "05", 300), ("helipr", "Bridge01", 2000),
                        ("helipr", "Bridge04", None)):
        with np.load(cache_path("train", s, q)) as z:
            sids, poses, ts = z["scan_ids"], z["poses"].astype(np.float64), z["timestamps"]
        lp = lidar_poses(s, q, poses)
        keep = np.asarray(select_frame_indices(lp[:, :3, 3], 3.0))
        if start is None:   # Bridge04: frames nearest to the chosen Bridge01 frames
            ref = ref_pos
            d = np.linalg.norm(lp[keep][:, None, :2, 3] - ref[None, :, :2], axis=2).min(1)
            i0 = int(np.argmin(d))
            sel = keep[i0:i0 + n_per_seq]
        else:
            sel = keep[start:start + n_per_seq]
        if q == "Bridge01":
            ref_pos = lp[sel][:, :3, 3]
        loader = make_loader(s, q)
        track = Track("train", s, q, loader) if SENSORS[s]["accumulate"] else None
        name = seq_name(s, q)
        d = td / "training" / name
        d.mkdir(parents=True)
        qq = Rotation.from_matrix(lp[sel][:, :3, :3]).as_quat()
        with open(d / "poses.txt", "w") as f:
            for t, P, r in zip(ts[sel], lp[sel], qq):
                f.write(f"{float(t)!r} {float(P[0, 3])!r} {float(P[1, 3])!r} {float(P[2, 3])!r} {float(r[0])!r} {float(r[1])!r} {float(r[2])!r} {float(r[3])!r}\n")
        for p, ci in enumerate(sel):
            xyz, refl, ring, _ = frame_points(s, "train", loader[int(sids[ci])]["points"],
                                              track, int(sids[ci]))
            np.save(d / f"{p:06d}.npy", to_train_height(fast_frame(xyz, refl, ring, s)))
        names.append(name)
        print(f"  selftest frames {name}: {len(sel)}", flush=True)
    return names


def selftest(parts, n_per_seq: int, out_json: Path | None):
    import torch
    off = load_official()
    rep = {"official_commit": OFFICIAL_COMMIT, "torch": torch.__version__,
           "numpy": np.__version__}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        names = _selftest_frames(td, n_per_seq)

        if "pickle" in parts:
            # (a) one group == official construct_query_dict on the same rows
            df = read_train_poses(td, names[0])
            off.TG.construct_query_dict(df, str(td / "ref.pickle"))
            build_tuples(td, names[:1], "global", td / "ours.pickle")
            a = pickle.load(open(td / "ref.pickle", "rb"))
            b = pickle.load(open(td / "ours.pickle", "rb"))
            same = sorted(a) == sorted(b) and all(
                a[k]["positives"] == b[k]["positives"] and a[k]["non_negatives"] ==
                b[k]["non_negatives"] and a[k]["rel_scan_filepath"] == b[k]["rel_scan_filepath"]
                and np.array_equal(a[k]["position"], b[k]["position"]) for k in a)
            # (b) global vs per_sequence on KITTI 05 + Bridge01/04
            g = build_tuples(td, names, "global", td / "training_g.pickle")
            p = build_tuples(td, names, "per_sequence", td / "training_p.pickle")
            rep["pickle"] = {"single_group_equals_official": bool(same),
                             "schema_keys": sorted(next(iter(b.values())).keys()),
                             "global": {k: g[k] for k in ("groups", "mean_cross_seq_positives",
                                                          "frac_with_cross_seq_positive")},
                             "per_sequence": {k: p[k] for k in ("mean_cross_seq_positives",)},
                             "official_dataset_loads": len(off.BD.TrainingDataset(
                                 str(td) + "/", "training_g.pickle").queries)}
            assert same and g["mean_cross_seq_positives"] > 0 and p["mean_cross_seq_positives"] == 0
            print("  pickle:", json.dumps(rep["pickle"]), flush=True)

        if "collate" in parts or "dryrun" in parts:
            cfg = td / "config.txt"
            txt = config_text("global", 0)
            txt = txt.replace("dataset_folder = data/", f"dataset_folder = {td}/")
            txt = txt.replace("train_file = training_pdv2_global.pickle",
                              "train_file = training_g.pickle")
            for o, n in (("batch_size = 2048", "batch_size = 8"),
                         ("batch_split_size = 64", "batch_split_size = 4"),
                         ("num_positive_pairs = 216", "num_positive_pairs = 4"),
                         ("epochs = 100", "epochs = 1")):
                assert txt.count(o) == 1, o
                txt = txt.replace(o, n)
            cfg.write_text(txt)
            params = off.MU.TrainingParams(str(cfg), str(ensure_model_config(td / "model.txt")),
                                           debug=False)

        if "collate" in parts:
            # patched collate == official collate on a one-sensor batch when the
            # pair geometry equals the config values
            ds = off.BD.TrainingDataset(params.dataset_folder, params.train_file)
            ids = [i for i, qd in ds.queries.items() if sensor_of(qd.rel_scan_filepath) == "kitti"][:8]
            geo = {"kitti": (params.uFoV, params.dFoV, params.sensor_range)}
            # official collate with num_positive_pairs = 1/8 of this batch's positive pairs
            n_pp = sum(1 for x in ids for y in ds.queries[x].positives if y in set(ids) and y > x)
            p_off = types.SimpleNamespace(**vars(params))
            p_off.num_positive_pairs = max(1, int(round(n_pp / 8)))
            outs = []
            for fn in (off.make_collate_fn(ds, p_off),
                       make_collate_fn_pdv2(ds, params, geometry=geo, timed=False)):
                np.random.seed(7)
                random.seed(7)
                items = [ds[i] for i in ids]
                outs.append(fn(items))
            a, b = outs
            eq = all(torch.equal(x, y) for x, y in zip(a[0], b[0])) and torch.equal(a[1], b[1]) \
                and torch.equal(a[2], b[2]) and np.array_equal(a[3], b[3]) and a[4] == b[4]
            # timing with our per-sensor geometry on a KITTI batch and a HeLiPR batch
            tim = {}
            for sen in ("kitti", "helipr"):
                ids_s = [i for i, qd in ds.queries.items()
                         if sensor_of(qd.rel_scan_filepath) == sen][:8]
                fn = make_collate_fn_pdv2(ds, params, timed=False)
                items = [ds[i] for i in ids_s]
                t0 = time.time()
                o = fn(items)
                tim[sen] = {"pairs": int(len(o[3])), "s": round(time.time() - t0, 2),
                            "patch_pairs": [len(x) for x in o[4]]}
            # Patch-InfoNCE yield without augmentation: patch pairs per image pair
            # with the per-sensor geometry vs the official one-sensor config values
            yld = {}
            for sen in ("kitti", "helipr"):
                its = []
                for i, qd in ds.queries.items():
                    if sensor_of(qd.rel_scan_filepath) != sen:
                        continue
                    x = np.load(Path(ds.dataset_path) / qd.rel_scan_filepath)
                    x = off.BD.normalize_to_unit(off.BD.pad_cylindrical_28(x).transpose(2, 0, 1))
                    its.append((x, i, list(qd.position)))
                for gname, g in (("per_sensor", GEOMETRY[sen]),
                                 ("official_config", (params.uFoV, params.dFoV, params.sensor_range))):
                    o = make_collate_fn_pdv2(ds, params, geometry={sen: g}, timed=False)(its)
                    pp = [len(x) for x in o[4]]
                    yld[f"{sen}/{gname}"] = {"image_pairs": len(pp), "patch_pairs": pp,
                                             "mean": float(np.mean(pp)) if pp else 0.0}
            rep["collate"] = {"patched_equals_official_one_sensor": bool(eq), "per_sensor": tim,
                              "batch_positive_pairs": n_pp,
                              "official_num_positive_pairs_set_to": p_off.num_positive_pairs,
                              "n_pairs": int(len(a[3])),
                              "n_patch_pairs": [len(x) for x in a[4]],
                              "patch_pair_yield_unaugmented": yld}
            assert eq
            print("  collate:", json.dumps(rep["collate"]), flush=True)

        if "batchstats" in parts:
            # positive image pairs per 2048 batch (official BatchSampler over the
            # real split-v2 training tuples) -> the paper's 1/8 rule vs the code's 216
            names_all = [seq_name(s, q) for s, q in TRAIN]
            if not all((DATA / "training" / n / "poses.txt").exists() for n in names_all):
                print("  batchstats: skipped (run the index stage first)", flush=True)
            else:
                bst = {}
                for mode in ("global", "per_sequence"):
                    man = build_tuples(DATA, names_all, mode, td / f"bs_{mode}.pickle")
                    dsb = off.BD.TrainingDataset(str(td) + "/", f"bs_{mode}.pickle")
                    random.seed(SEED)
                    bsm = off.SA.BatchSampler(dsb, batch_size=2048)
                    bsm.generate_batches()
                    cnt = []
                    for b in bsm.batch_idx:
                        if len(b) < 2048:
                            continue
                        S = set(b)
                        cnt.append(sum(1 for x in b for y in dsb.queries[x].positives
                                       if y in S and y > x))
                    bst[mode] = {"n_batches": len(bsm.batch_idx), "full_batches": len(cnt),
                                 "positive_pairs_per_batch": [int(min(cnt)), int(np.mean(cnt)),
                                                              int(max(cnt))],
                                 "one_eighth": int(round(np.mean(cnt) / 8)),
                                 "mean_positives_per_frame": man["mean_positives"]}
                rep["batchstats"] = bst
                print("  batchstats:", json.dumps(bst), flush=True)

        if "recall" in parts:
            # batched val recall == baselines.eval_utils.compute_recall_multi_k
            from baselines.eval_utils import compute_recall_multi_k
            from _imlpr_official_riv import cache_path
            rng = np.random.default_rng(0)
            rr = {}
            for s, q in (("helipr", "Roundabout02"), ("kitti", "09"), ("mulran", "Sejong02")):
                P = np.load(cache_path("val", s, q))["poses"].astype(np.float64)
                pos = P[:, :3, 3]
                W = rng.standard_normal((6, 256))
                f = np.concatenate([np.cos(pos[:, :2] / 7.0), np.sin(pos[:, :2] / 7.0),
                                    np.cos(pos[:, :2] / 3.0)], 1)
                emb = (f @ W + 0.8 * rng.standard_normal((len(pos), 256))).astype(np.float32)
                recs, mine = [], []
                ref, nq_ref = compute_recall_multi_k(emb, P, k_values=[1], distance_threshold=5.0,
                                                     skip_frames=30, per_query_records=recs)
                r1, nq = recall_at_1(emb, P, per_query_records=mine)
                same = nq == nq_ref and r1 == ref[1] and \
                    [r["top1_idx"] for r in mine] == [r["top1_idx"] for r in recs]
                rr[f"{s}/{q}"] = {"r1": r1, "n_queries": nq, "equal": bool(same)}
                assert same, (s, q)
            rep["recall"] = rr
            print("  recall:", json.dumps(rr), flush=True)

        if "dryrun" in parts:
            # official trainer loop (copied do_train) on CPU: 1 epoch of <= 2
            # steps with 8-image batches, v2-val replacement on the same frames,
            # then resume into epoch 2.
            torch.set_num_threads(4)
            install_collate_patch()
            sets = []
            for n in names:
                d = td / "training" / n
                P = np.loadtxt(d / "poses.txt")
                from scipy.spatial.transform import Rotation
                T = np.tile(np.eye(4), (len(P), 1, 1))
                T[:, :3, :3] = Rotation.from_quat(P[:, 4:8]).as_matrix()
                T[:, :3, 3] = P[:, 1:4]
                sets.append((n, [str(d / f"{p:06d}.npy") for p in range(len(P))], T))
            ev = V2Val(sets=sets, batch=4, workers=0)
            params.debug = True
            run_dir = td / "runs"
            record = {"_meta": {"selftest": True}, "evals": [], "epochs": []}
            off.TR._set_seed(SEED, deterministic=False)
            model0 = off.MI.ImLPR(params.model_params)
            sd0 = {k: v.clone() for k, v in model0.state_dict().items()}
            del model0
            off.TR._set_seed(SEED, deterministic=False)
            t0 = time.time()
            model = do_train_pdv2(params, ev, run_dir, record, td / "sel.json")
            t1 = time.time() - t0
            sd1 = model.state_dict()
            moved = [k for k in sd0 if not torch.equal(sd0[k], sd1[k])]
            blk = "encoder.encoder.model.blocks."
            blocks = sorted({int(k[len(blk):].split(".")[0]) for k in moved if k.startswith(blk)})
            changed = sorted({".".join(k.split(".")[:4]) for k in moved if not k.startswith(blk)})
            unchanged_other = sorted({".".join(k.split(".")[:4]) for k in sd0
                                      if k not in moved and not k.startswith(blk)})
            losses = [e["train"].get("loss") for e in record["epochs"]]
            # resume: continue with epochs = 2
            params.epochs = 2
            rec2 = json.loads(json.dumps(record, default=float))
            do_train_pdv2(params, ev, run_dir, rec2, td / "sel2.json", resume=True)
            rep["dryrun"] = {
                "cpu_s_epoch1": round(t1, 1), "train_loss_epoch1": losses,
                "changed_dinov2_blocks": blocks,
                "changed_param_groups_other": changed,
                "unchanged_param_groups_other": unchanged_other,
                "files": sorted(p.name for p in run_dir.iterdir()),
                "val_final": record["final"]["family_macro_r1"],
                "resumed_at_epoch": rec2.get("resumed_at_epoch"),
                "epochs_after_resume": [e["epoch"] for e in rec2["epochs"]]}
            assert all(np.isfinite(x) for x in losses if x is not None)
            assert rep["dryrun"]["epochs_after_resume"] == [1, 2]
            if BACKBONE == "paper":                                  # last two blocks only
                assert blocks == [10, 11], blocks
                assert not any(c.startswith(("encoder.encoder.model.patch_embed",
                                             "encoder.encoder.model.pos_embed",
                                             "encoder.encoder.model.cls_token")) for c in changed)
            else:                                                    # code path = released weights
                assert blocks == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11], blocks
            rep["dryrun"]["paper_freeze"] = record["_meta"]["paper_freeze"]
            rep["dryrun"]["backbone_mode"] = BACKBONE
            rep["dryrun"]["aggregator"] = AGGREGATOR
            rep["dryrun"]["optimizer"] = type(model_optimizer_probe(params)).__name__
            print("  dryrun:", json.dumps(rep["dryrun"]), flush=True)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rep, indent=1, default=str))
    print("TRAIN_SELFTEST_OK", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mode_kw = dict(choices=("global", "per_sequence"),
                   default=os.environ.get("IMLPR_PICKLE_MODE", "global"))
    p = sub.add_parser("pickle")
    p.add_argument("--pickle-mode", **mode_kw)
    p = sub.add_parser("config")
    p.add_argument("--pickle-mode", **mode_kw)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", 4)))
    for name in ("train", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--pickle-mode", **mode_kw)
        p.add_argument("--val-batch", type=int, default=32)
        p.add_argument("--val-workers", type=int, default=8)
        if name == "train":
            p.add_argument("--resume", action="store_true")
        else:
            p.add_argument("--val-frames", type=int, default=300)
    p = sub.add_parser("selftest")
    p.add_argument("--parts", default="pickle,collate,batchstats,recall,dryrun")
    p.add_argument("--n-per-seq", type=int, default=6)
    p.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "pickle":
        cmd_pickle(a.pickle_mode)
    elif a.cmd == "config":
        cmd_config(a.pickle_mode, a.num_workers)
    elif a.cmd == "train":
        cmd_train(a.pickle_mode, a.resume, a.val_batch, a.val_workers)
    elif a.cmd == "smoke":
        cmd_smoke(a.pickle_mode, a.val_frames, a.val_batch, a.val_workers)
    elif a.cmd == "selftest":
        selftest(a.parts.split(","), a.n_per_seq, a.out_json)


if __name__ == "__main__":
    main()
