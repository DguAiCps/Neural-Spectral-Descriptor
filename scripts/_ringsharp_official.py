#!/usr/bin/env python3
"""RING#-L on split v2 with the public official code and its default behaviour.

Official code: lus6-Jenny/RINGSharp (_external/RINGSharp, upstream cd55367;
our branch nsc-integration only adds the 'nsc' data adapter, portability
fixes, two upstream trainer bug fixes and math-preserving memory patches —
logs/baseline_official_20261011/ring_family.md lists every commit).

Training (official recipe, glnet/config/config_nclt.txt): 30 epochs, Adam
lr 1e-4, weight decay 1e-5, MultiStepLR milestone 20 (gamma 0.1), batch 16
(batch_size_limit 32 with expansion rate 1.0 = no expansion), aug_mode 3,
xyz_aug, model config ring_sharp_l_nsc.txt (= ring_sharp_l_nclt.txt except
the model name).  Data: the existing split-v2 training tuples
(_external/ringsharp_data_pdv2/train_nsc.pickle: KITTI 01/02/05/06 + HeLiPR
Bridge01-04 GT keyframes, positives 25 m, non-negatives 50 m), read-only.
Only num_workers differs (5, container memory).  NSC_MICRO_BATCH (env,
default 2: pair-preserving gradient accumulation, tools/nsc_micro.py) is
required to fit 16 GB; on a 96 GB card NSC_MICRO_BATCH=1 runs the
upstream full-batch step.  Training runs the official tools/train.py
unchanged (runpy) after scripts/_ringsharp_band.install().

Retrieval (official depth, tools/evaluate_ours_gl.py with the README flag
--n_rerank 1000): rotation branch (estimate_yaw, batches of 128) ranks ALL
candidates by 1 - score; the first 1000 are re-ordered by the translation
branch (rotate_bev_batch at yaw and yaw - pi, solve_translation, element-wise
min of the two errors); top-1 = the translation minimum.  Query side as in
evaluate_ours_gl.py:252-259 (RING#-L has trans_mode 0, so the query's
bev_trans is None): the query's 128-channel BEV features are rotated FIRST
and trans_cnn is applied to the rotated features; map side: trans_cnn of the
unrotated map features (:247-250, :706, :722).  Fixed 2026-10-16: before, the query's
trans_cnn output was rotated (wrong order; see TRANS_QUERY_PATH).  Per test query t
the candidates are every keyframe j <= t - 30 (past only).  The translation
errors come from trans_errors_exact, a vectorized copy of solve_translation's
error line (its per-candidate python loop computes the same
1 - max(corr)/(0.15 h w)); it is re-checked against solve_translation on
three queries per sequence (asserted); --official-loop uses solve_translation
for every query (slow).

Checkpoint selection: the official trainer has no rule (it saves every
epoch and *_final.pth = last epoch).  We keep the split-v2 rule used before:
highest validation family-macro 5 m R@1 (queries pooled within a sensor
family, families weighted equally, ties -> earlier epoch), now over epochs
1-30 and with the official retrieval depth; the protocol of that number is
unchanged (GT keyframes, revisit < 5 m, |i - j| > 30, baselines/eval_utils).

BEV (decision 2026-10-11): official generate_bev, 160x160x20 grid, +-70 m,
z band per sensor = the RING / RING++ DATASET_TABLE row
(_ring_official_lib.ringsharp_bounds) in training (scripts/_ringsharp_band),
validation and test.  NCLT keeps [1, 20] m; before, every sensor used it.

Subcommands (one heavy sequence per process; container ring_baseline):
  prep                      write the training config + manifest
  train                     official trainer, 30 epochs (about 40 h)
  val-bev --seq S           cache validation BEVs (CPU)
  val --epoch E --seq S     validation R@1 of epoch E at the official depth
  val-watch --exit-file F   validate every epoch checkpoint as it appears
                            (run on a second GPU while training runs)
  select                    family-macro selection over the epochs present
  dump --cond C --seq S     test ledger (schema pr-dump-v1)
  check                     CPU-only checks
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import _ring_official_lib as rol  # noqa: E402  (loaders, caches; torch-free)
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402
from _pdv2_pilot import TEST, VAL  # noqa: E402

RINGSHARP = REPO / "_external/RINGSharp"
RS_TOOLS = RINGSHARP / "tools"
STAGE = REPO / "_external/ringsharp_data_pdv2_official"
PICKLES = REPO / "_external/ringsharp_data_pdv2"      # read-only inputs
CONFIG = STAGE / "config_nsc_pdv2_official.txt"
MODEL_CONFIG = RINGSHARP / "glnet/config/ring_sharp_l_nsc.txt"
OFFICIAL_CONFIG = RINGSHARP / "glnet/config/config_nclt.txt"
# Seed (env SEED, default 1).  The official trainer seeds torch, torch.cuda
# and numpy with the constant random_seed = 123 when tools/trainer.py is
# imported (trainer.py:31-34); Python's `random` (batch sampler,
# glnet/datasets/samplers.py:35,131) is not seeded by the official code.
# SEED = 1 is that official run, unchanged.  SEED = k != 1 re-seeds the same
# three generators with 123 + k - 1 (nothing else) and writes everything
# training-dependent to its own places; config, training tuples and
# validation BEVs are shared.
SEED = int(os.environ.get("SEED", "1"))
assert SEED >= 1, SEED
OFFICIAL_TRAINER_SEED = 123
TRAINER_SEED = OFFICIAL_TRAINER_SEED + SEED - 1
EXP_BASE = "ring_sharp_l_nsc_pdv2_official"
EXP = EXP_BASE if SEED == 1 else f"{EXP_BASE}_s{SEED}"
WDIR = RS_TOOLS / "results/weights" / EXP
OUT = rol.OUT_ROOT if SEED == 1 else rol.OUT_ROOT.with_name(f"{rol.OUT_ROOT.name}_s{SEED}")
LOG_DIR = REPO / ("logs/ring_official" if SEED == 1 else f"logs/ring_official_s{SEED}")
VAL_DIR = OUT / "ringsharp_val"
VAL_BEV = rol.CACHE_ROOT / "ringsharp_val_bev"
SEL = OUT / "ringsharp_ckpt_selection.json"


def _seed_meta():
    """Extra output fields for SEED != 1 (SEED = 1 outputs stay as before)."""
    return {} if SEED == 1 else {"seed": SEED, "trainer_seed": TRAINER_SEED,
                                 "exp_name": EXP}
N_RERANK = 1000          # README: evaluate_ours_gl.py --n_rerank 1000
XCHECK_MAX = 512         # candidates re-scored with the solve_translation loop per check
BATCH = 128              # evaluate_ours_gl.py rank / rerank batch
FEAT_BATCH = 8
EPOCHS = 30
N_XCHECK = 3
# Query path of the translation branch; written into every val json, the
# selection json and every ledger, and checked before an existing output is
# reused (outputs made before the 2026-10-16 fix lack it and are refused).
TRANS_QUERY_PATH = ("trans_cnn(rotate_bev_batch(query 128-ch BEV features)) "
                    "[evaluate_ours_gl.py:252-259]")

CONFIG_TEXT = """[DEFAULT]
dataset = nsc
dataset_folder = /workspace/data

[TRAIN]
# official config_nclt.txt: num_workers = 16 (data-loader processes only;
# 5 here because of the container memory limit)
num_workers = 5
batch_size = 16
batch_size_limit = 32
batch_expansion_rate = 1.0
batch_expansion_th = 0.7
save_freq = 1
eval_freq = 1

lr = 1e-4
epochs = 30
scheduler_milestones = 20
aug_mode = 3
weight_decay = 1e-5

# split-v2 training tuples (KITTI 01/02/05/06 + HeLiPR Bridge01-04, GT
# keyframes, positives 25 m / non-negatives 50 m), built by
# scripts/_pdv2_c3.py ring-prep; read-only here
train_file = {train}
val_file = {val}
# no side-channel test evaluator (selection = post-hoc split-v2 validation)
test_file = {stage}/_no_such_evalset.pickle
# BEV bounds are not a config key: the 'nsc' connector gets the per-sensor
# band (scripts/_ring_official_lib.DATASET_TABLE) from
# scripts/_ringsharp_band.py, see manifest.json bev_bounds_per_sensor
"""


def _sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _split(seq, roster):
    s, q = seq.split("/")
    assert (s, q) in roster, seq
    return s, q


# ---------------------------------------------------------------- training
def cmd_prep():
    STAGE.mkdir(parents=True, exist_ok=True)
    train, val = PICKLES / "train_nsc.pickle", PICKLES / "val_nsc.pickle"
    CONFIG.write_text(CONFIG_TEXT.format(train=train, val=val, stage=STAGE))
    import configparser
    ours, off = configparser.ConfigParser(), configparser.ConfigParser()
    ours.read(CONFIG)
    off.read(OFFICIAL_CONFIG)
    diff = {k: {"official": off["TRAIN"].get(k), "ours": ours["TRAIN"].get(k)}
            for k in off["TRAIN"] if off["TRAIN"].get(k) != ours["TRAIN"].get(k)}
    (STAGE / "manifest.json").write_text(json.dumps({
        "recipe": "official glnet/config/config_nclt.txt",
        "train_keys_differing_from_official": diff,
        "model_config": str(MODEL_CONFIG),
        "train_pickle": str(train), "train_pickle_sha256": _sha_file(train),
        "val_pickle": str(val), "val_pickle_sha256": _sha_file(val),
        "train_manifest": json.loads((PICKLES / "manifest.json").read_text()),
        "exp_name": EXP_BASE,
        "env": {"NSC_MICRO_BATCH": "2 by default (16 GB); 1 = upstream full batch"},
        "rule": ("public official code with its default behaviour (config_nclt.txt, "
                 "README --n_rerank 1000); only protocol-required and run-only changes"),
        "bev_bounds_per_sensor": {k: rol.ringsharp_bounds(k) for k in rol.DATASET_TABLE},
        "bev_bounds_source": ("scripts/_ring_official_lib.DATASET_TABLE via "
                              "scripts/_ringsharp_band.NSCBandTrainingDataset "
                              "(decision 2026-10-11; before: z [1, 20] for every sensor)"),
    }, indent=2))
    print(f"wrote {CONFIG}\nkeys differing from official: {diff}", flush=True)


def cmd_train():
    if list(WDIR.glob(f"model_*_{EPOCHS}.pth")):
        print(f"SKIP: {WDIR} has epoch-{EPOCHS} weights", flush=True)
        return
    partial = sorted(WDIR.glob("model_*.pth"))
    assert not partial, (
        f"{WDIR} holds an unfinished run ({len(partial)} files). The official "
        "--resume does not restore the optimizer / LR schedule (report 9-6); "
        "decide by hand (move the folder away to restart from scratch).")
    assert CONFIG.exists(), "run prep first"
    env = dict(os.environ)
    env.setdefault("NSC_MICRO_BATCH", "2")
    env["PYTHONPATH"] = os.pathsep.join([str(RINGSHARP), env.get("PYTHONPATH", "")])
    cmd = [sys.executable, str(Path(__file__).resolve()), "_train-worker"]
    print("cwd", RS_TOOLS, "cmd", " ".join(cmd), "NSC_MICRO_BATCH",
          env["NSC_MICRO_BATCH"], "CUDA_VISIBLE_DEVICES",
          env.get("CUDA_VISIBLE_DEVICES"), flush=True)
    rc = subprocess.run(cmd, cwd=RS_TOOLS, env=env).returncode
    assert rc == 0, f"RING#-L training failed rc={rc}"


def _apply_seed():
    """SEED != 1: import the official trainer (its module-level seeding with
    random_seed = 123 runs now), then re-seed the same three generators with
    TRAINER_SEED; train.py's `from trainer import do_train` reuses this
    module, so nothing re-seeds afterwards.  SEED = 1: nothing (official).
    Needs RS_TOOLS on sys.path."""
    if SEED == 1:
        return OFFICIAL_TRAINER_SEED
    import numpy as np
    import torch
    import trainer
    torch.manual_seed(TRAINER_SEED)
    torch.cuda.manual_seed_all(TRAINER_SEED)
    np.random.seed(TRAINER_SEED)
    trainer.random_seed = TRAINER_SEED
    print(f"SEED {SEED}: trainer seed {TRAINER_SEED} (official 123), exp {EXP}", flush=True)
    return TRAINER_SEED


def _train_worker():
    """Official tools/train.py, unchanged, with the per-sensor band connector."""
    import runpy
    for p in (str(RS_TOOLS), str(RINGSHARP)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(RS_TOOLS)
    import _ringsharp_band
    cls = _ringsharp_band.install()
    print(f"nsc training dataset class: {cls.__module__}.{cls.__name__}", flush=True)
    _apply_seed()
    sys.argv = ["train.py", "--config", str(CONFIG), "--model_config",
                str(MODEL_CONFIG), "--exp_name", EXP]
    runpy.run_path(str(RS_TOOLS / "train.py"), run_name="__main__")


def epoch_weight(e: int) -> Path:
    hits = sorted(WDIR.glob(f"model_*_{e}.pth"))
    assert len(hits) == 1, (e, hits)
    return hits[0]


# ------------------------------------------------------- official functions
def _base():
    import _ringsharp_eval as base   # glnet imports, load_model (e2cnn check)
    return base


def _depth(n: int) -> int:
    return n if N_RERANK <= 0 else min(N_RERANK, n)


def trans_errors_exact(a, b):
    """errors of glnet.models.utils.solve_translation(a, b) (zero-mean
    normalize, scale 1), vectorized: same ops, then
    1 - max(corr_i) / (0.15 h w) per item without the python loop."""
    import torch
    import torch.nn as nn
    h, w = a.shape[-2:]
    inorm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False)
    a, b = inorm(a), inorm(b)
    a_fft = torch.fft.fft2(a, dim=(-2, -1), norm="ortho")
    b_fft = torch.fft.fft2(b, dim=(-2, -1), norm="ortho")
    corr = torch.fft.ifft2(a_fft * b_fft.conj(), dim=(-2, -1), norm="ortho")
    corr = torch.sqrt(corr.real ** 2 + corr.imag ** 2 + 1e-15)
    corr = torch.sum(corr, dim=-3)
    corr = torch.fft.fftshift(corr, dim=(-2, -1))
    return (1 - corr.amax(dim=(-2, -1)) / (0.15 * h * w)).float().cpu()


def rotation_branch(base, specs, qspec, cand):
    """estimate_yaw of the query against specs[cand] in batches of 128
    (evaluate_ours_gl.py) -> (dist = 1 - score, angle in rad) numpy."""
    import torch
    dists, angles = [], []
    hs = specs.shape[-2]
    with torch.no_grad():
        for b0 in range(0, len(cand), BATCH):
            ms = specs[cand[b0:b0 + BATCH]]
            _, sc, sh = base.estimate_yaw(qspec.repeat(ms.shape[0], 1, 1, 1), ms)
            dists.append(1 - sc.detach().cpu().numpy().reshape(-1))
            angles.append((sh * 2 * np.pi / hs).detach().cpu().numpy().reshape(-1))
    return np.concatenate(dists), np.concatenate(angles)


def translation_branch(base, trans_cnn, bts, qfeat, cand, angles, official_loop, dedup=True):
    """Official translation rerank of cand (angles from the rotation
    branch): min over (yaw, yaw - pi) of the solve_translation error.
    qfeat = the query's 128-channel BEV features (model output 'bev',
    1 x C x H x W); as evaluate_ours_gl.py:252-259 they are rotated by each
    candidate's angle and THEN passed through trans_cnn.  bts = trans_cnn of
    the unrotated map features (:247-250).
    dedup: the rotation branch yields at most 160 distinct angles, so
    trans_cnn(rotate(qfeat, a)) is computed once per distinct float angle a
    (the same float32 inputs the per-candidate call would get) and shared;
    xcheck_translation compares it with the per-candidate computation."""
    import torch
    errs = []
    with torch.no_grad():
        a1 = torch.tensor(angles, dtype=torch.float)
        a2 = torch.tensor(angles - np.pi, dtype=torch.float)
        if dedup:
            uniq, inv = torch.unique(torch.cat([a1, a2]), return_inverse=True)
            qmaps = torch.cat([trans_cnn(base.rotate_bev_batch(qfeat, uniq[k:k + BATCH]))
                               for k in range(0, len(uniq), BATCH)])
            inv1, inv2 = inv[:len(a1)], inv[len(a1):]
        for b0 in range(0, len(cand), BATCH):
            cbt = bts[cand[b0:b0 + BATCH]]
            if dedup:
                q1 = qmaps[inv1[b0:b0 + BATCH].to(qmaps.device)]
                q2 = qmaps[inv2[b0:b0 + BATCH].to(qmaps.device)]
            else:
                q1 = trans_cnn(base.rotate_bev_batch(qfeat, a1[b0:b0 + BATCH]))
                q2 = trans_cnn(base.rotate_bev_batch(qfeat, a2[b0:b0 + BATCH]))
            if official_loop:
                e1 = base.solve_translation(q1, cbt)[2]
                e2 = base.solve_translation(q2, cbt)[2]
            else:
                e1, e2 = trans_errors_exact(q1, cbt), trans_errors_exact(q2, cbt)
            errs.append(np.minimum(e1.detach().numpy().reshape(-1),
                                   e2.detach().numpy().reshape(-1)))
    return np.concatenate(errs)


def xcheck_translation(base, trans_cnn, bts, qfeat, cand, angles):
    """Per-candidate official path (solve_translation loop, no angle dedup)
    vs the path used (vectorized errors, deduplicated angles)."""
    cand, angles = cand[:XCHECK_MAX], angles[:XCHECK_MAX]
    a = translation_branch(base, trans_cnn, bts, qfeat, cand, angles, True, dedup=False)
    b = translation_branch(base, trans_cnn, bts, qfeat, cand, angles, False, dedup=True)
    d = float(np.abs(a - b).max())
    gap = float(np.partition(a, 1)[1] - a.min()) if len(a) > 1 else 1.0
    assert d < 1e-5 and (int(np.argmin(a)) == int(np.argmin(b)) or gap < 1e-5), \
        (d, np.argmin(a), np.argmin(b))
    return {"n": int(len(cand)), "max_abs_diff": d}


# ---------------------------------------------------------------- features
def features(base, model, trans_cnn, bev_batches, n, device):
    """RINGSharpL forward -> global (n,160) fp32, spec (n,1,160,160) fp32,
    bev_trans = trans_cnn(bev) (n,1,160,160) fp32 (official keeps fp32)."""
    import torch
    glob_, specs, bts = [], [], []
    with torch.no_grad():
        for bev in bev_batches:
            out = model({"pc": bev.to(device), "img": None})
            glob_.append(out["global"].float().cpu().numpy())
            specs.append(out["spec"].float().cpu())
            bts.append(trans_cnn(out["bev"]).float().cpu())
    g, s, b = np.concatenate(glob_), torch.cat(specs), torch.cat(bts)
    assert len(g) == n
    return g, s, b


def bev_batches_from_loader(base, loader, scan_ids, bounds):
    import torch
    for i0 in range(0, len(scan_ids), FEAT_BATCH):
        bevs = []
        for sid in scan_ids[i0:i0 + FEAT_BATCH]:
            pts = np.ascontiguousarray(loader[int(sid)]["points"][:, :3], dtype=np.float64)
            bevs.append(base.generate_bev(pts, Z=base.NSC_Z, Y=base.NSC_Y,
                                          X=base.NSC_X, bounds=bounds))
        yield torch.stack(bevs)


def tee_packed(bev_batches, sink):
    """Pass BEV batches through, keeping each frame's occupancy bits (sink)
    so a query's 128-channel features can be recomputed later."""
    for b in bev_batches:
        bb = b.numpy()
        assert set(np.unique(bb)) <= {0.0, 1.0}
        sink.append(np.packbits(bb.reshape(len(bb), -1) > 0, axis=1))
        yield b


def query_features(base, model, packed_row, device):
    """The query's 128-channel BEV features (RINGSharpL output 'bev'), as
    evaluate_ours_gl.py compute_embedding_item returns them for a query."""
    import torch
    shape = (base.NSC_Z, base.NSC_Y, base.NSC_X)
    b = np.unpackbits(packed_row, axis=1)[:, :int(np.prod(shape))]
    with torch.no_grad():
        return model({"pc": torch.from_numpy(b.reshape(1, *shape).astype(np.float32)).to(device),
                      "img": None})["bev"]


def _is_current(meta: dict) -> bool:
    return meta.get("query_translation_path") == TRANS_QUERY_PATH


def bev_batches_from_packed(base, packed):
    import torch
    shape = (base.NSC_Z, base.NSC_Y, base.NSC_X)
    m = int(np.prod(shape))
    for i0 in range(0, len(packed), FEAT_BATCH):
        b = np.unpackbits(packed[i0:i0 + FEAT_BATCH], axis=1)[:, :m]
        yield torch.from_numpy(b.reshape(-1, *shape).astype(np.float32))


# -------------------------------------------------------------- validation
def cmd_val_bev(seq):
    s, q = _split(seq, VAL)
    out = VAL_BEV / f"{s}_{q}.npz"
    scan_ids, _ = rol.load_keyframes(rol.split_cache("gt", "val", s, q))
    bounds = rol.ringsharp_bounds(s)
    if out.exists():
        with np.load(out) as z:
            if np.array_equal(z["scan_ids"], scan_ids) and np.allclose(z["bounds"], bounds):
                print(f"SKIP {out.name}", flush=True)
                return
    base = _base()
    loader = rol.make_loader(s, q)
    t0 = time.time()
    packed = []
    for bev in bev_batches_from_loader(base, loader, scan_ids, bounds):
        bev = bev.numpy()
        assert set(np.unique(bev)) <= {0.0, 1.0}
        packed.append(np.packbits(bev.reshape(len(bev), -1) > 0, axis=1))
    VAL_BEV.mkdir(parents=True, exist_ok=True)
    tmp = VAL_BEV / f".tmp_{s}_{q}.npz"
    np.savez_compressed(tmp, scan_ids=scan_ids, packed=np.concatenate(packed),
                        bounds=np.asarray(bounds))
    os.replace(tmp, out)
    print(f"VAL_BEV_DONE {seq}: {len(scan_ids)} kf, "
          f"{(time.time() - t0) / len(scan_ids):.3f} s/kf", flush=True)


def cmd_val(epoch, seq, official_loop):
    import torch
    from baselines.eval_utils import compute_recall_cosine_then_rerank
    s, q = _split(seq, VAL)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    out = VAL_DIR / f"val_e{epoch}_{s}_{q}.json"
    if out.exists():
        if _is_current(json.loads(out.read_text())):
            print(f"SKIP {out.name}", flush=True)
            return
        sys.exit(f"{out} was made before the 2026-10-16 translation fix; "
                 f"move {VAL_DIR} away, then rerun")
    base = _base()
    w = epoch_weight(epoch)
    device = "cuda"
    model, trans_cnn = base.load_model(str(w), device)
    scan_ids, poses = rol.load_keyframes(rol.split_cache("gt", "val", s, q))
    z = np.load(VAL_BEV / f"{s}_{q}.npz")
    assert np.array_equal(z["scan_ids"], scan_ids), "val BEV cache mismatch"
    assert np.allclose(z["bounds"], rol.ringsharp_bounds(s)), "val BEV cache band mismatch"
    t0 = time.time()
    g, specs, bts = features(base, model, trans_cnn,
                             bev_batches_from_packed(base, z["packed"]), len(scan_ids), device)
    t_feat = time.time() - t0
    specs, bts = specs.to(device), bts.to(device)
    xcheck = []

    def rerank_fn(qi, cand):
        cand = np.asarray(cand)
        dist, ang = rotation_branch(base, specs, specs[qi:qi + 1], cand)
        order = np.argsort(dist)
        m = _depth(len(cand))
        top = order[:m]
        qfeat = query_features(base, model, z["packed"][qi:qi + 1], device)
        if len(xcheck) < N_XCHECK and m >= 2 * BATCH:
            xcheck.append(xcheck_translation(base, trans_cnn, bts, qfeat, cand[top], ang[top]))
        errs = translation_branch(base, trans_cnn, bts, qfeat, cand[top], ang[top], official_loop)
        order[:m] = top[np.argsort(errs)]
        return cand[order]

    t0 = time.time()
    recalls, n_q = compute_recall_cosine_then_rerank(
        g, rerank_fn, poses, k_values=[1, 5, 10], n_coarse=len(g))
    t_ret = time.time() - t0
    out.write_text(json.dumps({
        "sensor": s, "sequence": q, "epoch": epoch, "n_frames": len(scan_ids),
        "n_queries": n_q, **{f"recall@{k}": recalls[k] for k in (1, 5, 10)},
        "weight": str(w), "weight_sha256": _sha_file(w),
        "retrieval": f"exhaustive rotation ranking -> translation rerank of top {N_RERANK}",
        "translation_errors": "solve_translation loop" if official_loop else "trans_errors_exact",
        "translation_xcheck": xcheck,
        "query_translation_path": TRANS_QUERY_PATH,
        "protocol": "revisit<5m, |i-j|>30, intra-sequence (baselines/eval_utils), GT keyframes",
        "bev_bounds": list(rol.ringsharp_bounds(s)),
        "timing": {"feature_s": round(t_feat, 1), "retrieval_s": round(t_ret, 1),
                   "retrieval_ms_per_query": round(1e3 * t_ret / max(n_q, 1), 1)},
        **_seed_meta(),
    }, indent=2))
    print(f"VAL_DONE e{epoch} {seq}: R@1 {recalls[1]:.4f} ({n_q} q, feat {t_feat:.0f}s, "
          f"ret {t_ret:.0f}s)", flush=True)
    del specs, bts
    torch.cuda.empty_cache()


def _weight_ready(e, exit_file, min_age):
    """Epoch-e checkpoint path once it is completely written: training has
    exited, or a later epoch / *_final exists, or it is min_age s old."""
    hits = sorted(WDIR.glob(f"model_*_{e}.pth"))
    if len(hits) != 1:
        return None
    if (exit_file.exists() or list(WDIR.glob(f"model_*_{e + 1}.pth"))
            or list(WDIR.glob("model_*_final.pth"))
            or time.time() - hits[0].stat().st_mtime >= min_age):
        return hits[0]
    return None


def cmd_val_watch(exit_file, epochs, poll, min_age, log_dir, max_tries=2):
    """Validate every epoch checkpoint as soon as it is written (resumable:
    finished val_e*.json files are skipped).  Each (epoch, sequence) runs in
    its own `val` process on the GPU this watcher sees (CUDA_VISIBLE_DEVICES).
    Ends when every requested epoch is validated, or when training has ended
    (exit_file written) and nothing new can appear."""
    exit_file, log_dir = Path(exit_file), Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    todo, tries, failed = list(epochs), {}, set()
    print(f"val-watch: epochs {todo[0]}-{todo[-1]}, CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')}, weights {WDIR}", flush=True)
    while todo:
        progressed = False
        for e in list(todo):
            if _weight_ready(e, exit_file, min_age) is None:
                break                      # checkpoints appear in epoch order
            for s, q in VAL:
                key = (e, f"{s}/{q}")
                if (VAL_DIR / f"val_e{e}_{s}_{q}.json").exists() or key in failed:
                    continue
                tries[key] = tries.get(key, 0) + 1
                log = log_dir / f"rs_val_e{e}_{s}_{q}.log"
                t0 = time.time()
                with open(log, "w") as fh:
                    rc = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "val",
                         "--epoch", str(e), "--seq", f"{s}/{q}"],
                        stdout=fh, stderr=subprocess.STDOUT).returncode
                print(f"[{time.strftime('%F %T')}] val e{e} {s}/{q} rc={rc} "
                      f"({time.time() - t0:.0f} s)", flush=True)
                if rc != 0 and tries[key] >= max_tries:
                    failed.add(key)
            if all((VAL_DIR / f"val_e{e}_{s}_{q}.json").exists() for s, q in VAL):
                todo.remove(e)
                progressed = True
            elif all((VAL_DIR / f"val_e{e}_{s}_{q}.json").exists() or (e, f"{s}/{q}") in failed
                     for s, q in VAL):
                todo.remove(e)             # gave up on its failed sequences
        if not todo:
            break
        if exit_file.exists() and _weight_ready(todo[0], exit_file, min_age) is None:
            print(f"training ended ({exit_file.read_text().strip()}); no checkpoint "
                  f"for epochs {todo}", flush=True)
            break
        if not progressed:
            time.sleep(poll)
    missing = [(e, f"{s}/{q}") for e in epochs for s, q in VAL
               if not (VAL_DIR / f"val_e{e}_{s}_{q}.json").exists()]
    print(f"VAL_WATCH_END missing {len(missing)}: {missing[:10]}", flush=True)
    return 0 if not missing else 1


def cmd_select(epochs):
    from _pdv2_c3 import _family_pooled
    if SEL.exists():
        if _is_current(json.loads(SEL.read_text())):
            print(f"{SEL} exists; frozen", flush=True)
            return
        sys.exit(f"{SEL} was made before the 2026-10-16 translation fix; move it away")
    table = {}
    for e in epochs:
        per = {}
        for s, q in VAL:
            d = json.loads((VAL_DIR / f"val_e{e}_{s}_{q}.json").read_text())
            assert _is_current(d), f"pre-fix validation result: val_e{e}_{s}_{q}.json"
            per[f"{s}/{q}"] = {"recall@1": d["recall@1"], "n_queries": d["n_queries"]}
        table[f"e{e}"] = {"family_macro_r1": _family_pooled(per), "per_seq": per}
        print(f"e{e}: family-macro R@1 {table[f'e{e}']['family_macro_r1']:.4f}", flush=True)
    best = min(table, key=lambda t: (-table[t]["family_macro_r1"], int(t[1:])))
    w = epoch_weight(int(best[1:]))
    SEL.write_text(json.dumps({
        "rule": ("highest split-v2 validation family-macro 5 m R@1 (queries "
                 "pooled within a sensor family, families weighted equally) "
                 f"over epochs {epochs[0]}-{epochs[-1]}, official retrieval depth "
                 "(exhaustive rotation ranking -> translation rerank of top "
                 f"{N_RERANK}, README); ties -> earlier epoch. The official trainer has "
                 "no selection rule (it keeps every epoch and *_final = last)."),
        "selected_epoch": best, "selected_weight": str(w), "weight_sha256": _sha_file(w),
        "official_final_epoch": f"e{EPOCHS}",
        "query_translation_path": TRANS_QUERY_PATH,
        "epochs": table, **_seed_meta()}, indent=2))
    print(f"selected {best}: {w.name}", flush=True)


# -------------------------------------------------------------------- test
def ledger(n, positions, min_past, rerank_top1):
    """Causal all-frames ledger (torch-free loop; rerank_top1(t, n_prefix))."""
    queries = np.arange(MIN_INDEX_GAP, n, dtype=np.int64)
    cols = {k: np.full(len(queries), np.nan) for k in
            ("score_trans", "score_rot", "top1_geo_dist", "score_rotbranch",
             "top1_geo_dist_rot")}
    j_star = np.full(len(queries), -1, dtype=np.int64)
    j_rot = np.full(len(queries), -1, dtype=np.int64)
    n_cand = np.zeros(len(queries), dtype=np.int64)
    n_trans = np.zeros(len(queries), dtype=np.int64)
    t0 = time.time()
    for qi, t in enumerate(queries):
        t = int(t)
        npf = t - MIN_INDEX_GAP + 1
        r = rerank_top1(t, npf)
        assert r["j"] <= t - MIN_INDEX_GAP and r["j_rot"] <= t - MIN_INDEX_GAP
        j_star[qi], j_rot[qi] = r["j"], r["j_rot"]
        cols["score_trans"][qi] = -r["err"]
        cols["score_rot"][qi] = -r["rot_of_j"]
        cols["score_rotbranch"][qi] = -r["rot_min"]
        cols["top1_geo_dist"][qi] = np.linalg.norm(positions[r["j"]] - positions[t])
        cols["top1_geo_dist_rot"][qi] = np.linalg.norm(positions[r["j_rot"]] - positions[t])
        n_cand[qi], n_trans[qi] = npf, r["n_trans"]
        if qi and qi % 500 == 0:
            rate = qi / (time.time() - t0)
            print(f"  ledger {qi}/{len(queries)} ({rate:.1f} q/s, ETA "
                  f"{(len(queries) - qi) / rate / 60:.0f} min)", flush=True)
    return dict(query_idx=queries, j_star=j_star, score_trans=cols["score_trans"],
                score_rot=cols["score_rot"], top1_geo_dist=cols["top1_geo_dist"],
                n_candidates=n_cand, min_past_gt_dist=min_past[queries],
                j_star_rot=j_rot, score_rotbranch=cols["score_rotbranch"],
                top1_geo_dist_rot=cols["top1_geo_dist_rot"], n_trans=n_trans)


def selected_weight():
    sel = json.loads(SEL.read_text())
    assert _is_current(sel), f"{SEL} predates the 2026-10-16 translation fix"
    p = Path(sel["selected_weight"])
    if not p.exists():
        p = REPO / sel["selected_weight"].split("/Neural-Spectral-Codec/", 1)[1]
    assert _sha_file(p) == sel["weight_sha256"], "selected weight sha mismatch"
    return p, sel


def cmd_dump(cond, seq, official_loop, overwrite):
    import torch
    s, q = _split(seq, TEST)
    out_dir = OUT / cond
    out = out_dir / f"dump_ringsharp_{s}_{q}.npz"
    if out.exists() and not overwrite:
        if _is_current(json.loads(str(np.load(out)["meta"]))):
            print(f"SKIP {out.name}", flush=True)
            return
        sys.exit(f"{out} was made before the 2026-10-16 translation fix; "
                 f"move the old ledgers away (or pass --overwrite)")
    base = _base()
    w, sel = selected_weight()
    device = "cuda"
    model, trans_cnn = base.load_model(str(w), device)
    cache = rol.split_cache(cond, "test", s, q)
    scan_ids, poses = rol.load_keyframes(cache)
    positions = poses[:, :3, 3]
    min_past = rol.min_past_dists(positions, MIN_INDEX_GAP)
    loader = rol.make_loader(s, q)
    t0 = time.time()
    packed = []
    _, specs, bts = features(base, model, trans_cnn,
                             tee_packed(bev_batches_from_loader(base, loader, scan_ids,
                                                                rol.ringsharp_bounds(s)), packed),
                             len(scan_ids), device)
    packed = np.concatenate(packed)
    t_feat = time.time() - t0
    del loader
    specs, bts = specs.to(device), bts.to(device)
    n = len(scan_ids)
    xq = set(sorted({n // 4, n // 2, n - 1})[:N_XCHECK])
    xcheck = []

    def rerank_top1(t, npf):
        cand = np.arange(npf)
        dist, ang = rotation_branch(base, specs, specs[t:t + 1], cand)
        order = np.argsort(dist)                       # official: np.argsort
        m = _depth(npf)
        top = order[:m]
        qfeat = query_features(base, model, packed[t:t + 1], device)
        if t in xq:
            xcheck.append({"t": t, **xcheck_translation(base, trans_cnn, bts, qfeat,
                                                        cand[top], ang[top])})
        errs = translation_branch(base, trans_cnn, bts, qfeat, cand[top], ang[top], official_loop)
        i = int(np.argmin(errs))
        return {"j": int(top[i]), "err": float(errs[i]), "rot_of_j": float(dist[top[i]]),
                "j_rot": int(order[0]), "rot_min": float(dist[order[0]]), "n_trans": m}

    t0 = time.time()
    arrays = ledger(n, positions, min_past, rerank_top1)
    t_led = time.time() - t0
    nq = len(arrays["query_idx"])
    meta = {
        "schema": "pr-dump-v1", "method": "ringsharp", "sensor": s, "sequence": q,
        "condition": cond, "n_frames": n, "n_queries": nq,
        "min_index_gap": MIN_INDEX_GAP, "n_rerank": N_RERANK, "batch": BATCH,
        "retrieval": ("official depth (evaluate_ours_gl.py, README --n_rerank 1000): "
                      "rotation branch ranks every keyframe j <= t - 30, translation "
                      "branch re-orders the first min(1000, |prefix|)"),
        "translation_errors": "solve_translation loop" if official_loop else "trans_errors_exact",
        "translation_xcheck": xcheck,
        "query_translation_path": TRANS_QUERY_PATH,
        "score_direction": {
            "score_trans": "PRIMARY: -translation error of the emitted top-1 (j_star)",
            "score_rot": "-(1 - rotation score) of the same emitted top-1",
            "score_rotbranch": ("-(1 - rotation score) of the rotation-branch top-1 "
                                "j_star_rot (what the official PR curve thresholds "
                                "when n_rerank < database size); distance in "
                                "top1_geo_dist_rot")},
        "poses_for_scoring": "poses_gt" if cond == "odom" else "poses (GT)",
        "bev": ("glnet generate_bev 160x160x20, bounds = DATASET_TABLE band "
                "(decision 2026-10-11)"),
        "bev_bounds": list(rol.ringsharp_bounds(s)),
        "artifacts": {"weight": str(w), "weight_sha256": sel["weight_sha256"],
                      "weight_selection": str(SEL), "weight_selection_rule": sel["rule"],
                      "keyframe_cache": str(cache)},
        "train_overlap": False,
        **_seed_meta(),
        "timing": {"feature_s_total": round(t_feat, 1),
                   "feature_s_per_kf": round(t_feat / n, 4),
                   "ledger_s_total": round(t_led, 1),
                   "ledger_ms_per_query": round(1e3 * t_led / max(nq, 1), 1)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    print(f"DUMP_DONE {out.name}: {nq} queries, feat {t_feat:.0f}s, ledger "
          f"{t_led:.0f}s, xcheck {xcheck}", flush=True)
    del specs, bts, model, trans_cnn
    torch.cuda.empty_cache()


# -------------------------------------------------------------- CPU checks
def cmd_check():
    import torch
    base = _base()
    rep = {}
    # 1. vectorized translation errors == solve_translation (CPU, random)
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for b in (1, 7, 33):
        a = torch.rand(b, 1, 160, 160, generator=g)
        c = torch.rand(b, 1, 160, 160, generator=g)
        c[0] = torch.roll(a[0], shifts=(5, -9), dims=(-2, -1))
        e_off = base.solve_translation(a, c)[2]
        e_vec = trans_errors_exact(a, c)
        worst = max(worst, float((e_off - e_vec).abs().max()))
        assert torch.equal(e_off, e_vec) or worst < 1e-6, worst
    rep["trans_errors_exact_vs_solve_translation_cpu"] = {"max_abs_diff": worst}
    # 1b. query path of the translation branch = official order
    rep["translation_query_path"] = _check_query_path(base)
    print("  translation", rep["trans_errors_exact_vs_solve_translation_cpu"], flush=True)
    # 2. ledger schema vs the existing split-v2 RING#-L ledger
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.standard_normal((90, 3)), axis=0)
    mp = rol.min_past_dists(pos, MIN_INDEX_GAP)

    def stub(t, npf):
        e = np.abs(np.sin(np.arange(npf) * 0.7 + t))
        j = int(np.argmin(e))
        return {"j": j, "err": float(e[j]), "rot_of_j": 0.5, "j_rot": 0,
                "rot_min": 0.1, "n_trans": _depth(npf)}

    arrays = ledger(90, pos, mp, stub)
    ref = REPO / "results/pdv2_c3/odom/dump_ringsharp_kitti_07.npz"
    z = np.load(ref)
    theirs = {k: str(z[k].dtype) for k in z.files if k != "meta"}
    mine = {k: str(v.dtype) for k, v in arrays.items()}
    rep["ledger_schema"] = {"reference": str(ref.relative_to(REPO)),
                            "reference_arrays_present_same_dtype":
                                all(mine.get(k) == v for k, v in theirs.items()),
                            "extra_arrays": sorted(set(mine) - set(theirs))}
    assert rep["ledger_schema"]["reference_arrays_present_same_dtype"]
    assert np.array_equal(arrays["query_idx"], np.arange(30, 90))
    print("  ledger", rep["ledger_schema"], flush=True)
    # 3. official model code untouched (no compute_spec override)
    from glnet.models.localizer.ring_sharp_l import RINGSharpL
    assert RINGSharpL.compute_spec.__module__ == "glnet.models.localizer.ring_sharp_l"
    rep["official_compute_spec"] = True
    # 3b. per-sensor band in val / test BEVs (CPU, real scans)
    rep["bev_band"] = _check_band(base)
    # 4. the training connector uses the band per sample
    rep["train_connector"] = _check_train_connector(base)
    # 5. recipe: our config vs the official one
    if not CONFIG.exists():
        cmd_prep()
    rep["recipe_diff"] = json.loads((STAGE / "manifest.json").read_text())[
        "train_keys_differing_from_official"]
    print("  recipe diff", rep["recipe_diff"], flush=True)
    p = rol.OUT_ROOT / "check_ringsharp_official_cpu.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, indent=2))
    print(f"CHECK_OK -> {p}", flush=True)


def _vox_reference(pts, bounds, Z=20, Y=160, X=160):
    """CHECK ONLY: numpy re-statement of glnet Vox_util.voxelize_xyz +
    generate_bev (mem = (p - min) / vox - 0.5, in bounds strictly inside,
    rounded, origin points and voxel (0, 0, 0) dropped, last two axes
    swapped) to show which points the BEV takes."""
    lo = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
    hi = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
    n = np.array([X, Y, Z], dtype=np.float64)
    vox = (hi - lo) / n
    # float32 like Vox_util (scale and offset of mem_T_ref rounded to float32)
    a = (1.0 / vox).astype(np.float32)
    t = ((-lo - vox / 2) * (1.0 / vox)).astype(np.float32)
    p = np.asarray(pts, dtype=np.float32)[:, :3]
    mem = (p * a + t).astype(np.float64)
    zero = t.astype(np.float64)
    ok = np.all((mem > -0.5) & (mem < n - 0.5), axis=1) & (mem[:, 2] != 0.0)
    ok &= np.linalg.norm(mem - zero, axis=1) >= 0.1
    idx = np.rint(mem[ok]).astype(np.int64)
    occ = np.zeros((Z, Y, X), dtype=np.uint8)
    occ[idx[:, 2], idx[:, 1], idx[:, 0]] = 1
    occ[0, 0, 0] = 0
    return occ.transpose(0, 2, 1)


def _check_query_path(base):
    """translation_branch == evaluate_ours_gl.py:252-262 written out
    (trans_cnn applied to the ROTATED query features), and != the pre-fix
    order rotate(trans_cnn(features)).  CPU: the official rotate_bev_batch
    hard-codes .cuda(), so Tensor.cuda is a no-op inside this check only."""
    import torch
    from glnet.models.backbones_2d.unet import last_conv_block
    torch.manual_seed(0)
    tc = last_conv_block(4, 4, bn=False).eval()
    qfeat, maps = torch.rand(1, 4, 160, 160), torch.rand(9, 4, 160, 160)
    ang = (np.array([3, 3, 17, -40, 3, 79, -80, 17, 0]) * 2 * np.pi / 160).astype(np.float32)
    cand = np.arange(9)
    orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    try:
        with torch.no_grad():
            bts = tc(maps)
            ours = translation_branch(base, tc, bts, qfeat, cand, ang, False)
            ours_nd = translation_branch(base, tc, bts, qfeat, cand, ang, False, dedup=False)

            def official(q):  # q: rotated query maps fed to solve_translation
                return base.solve_translation(q, bts)[2].numpy().reshape(-1)
            rot = lambda f, a: base.rotate_bev_batch(f, torch.tensor(a, dtype=torch.float))  # noqa: E731
            ref = np.minimum(official(tc(rot(qfeat, ang))), official(tc(rot(qfeat, ang - np.pi))))
            qb = tc(qfeat)
            old = np.minimum(official(rot(qb, ang)), official(rot(qb, ang - np.pi)))
    finally:
        torch.Tensor.cuda = orig
    out = {"max_abs_diff_vs_official_order": float(np.abs(ours - ref).max()),
           "max_abs_diff_dedup_vs_per_candidate": float(np.abs(ours - ours_nd).max()),
           "max_abs_diff_vs_pre_fix_order": float(np.abs(old - ref).max())}
    print("  translation query path", out, flush=True)
    assert out["max_abs_diff_vs_official_order"] < 1e-6, out
    assert out["max_abs_diff_dedup_vs_per_candidate"] < 1e-6, out
    assert out["max_abs_diff_vs_pre_fix_order"] > 1e-4, out
    return out


def _check_band(base):
    seqs = [("kitti", "00"), ("kitti360", "0000"), ("mulran", "DCC01"),
            ("nclt", "2012-01-08"), ("helipr", "Town01"), ("mcd", "kth_day_06")]
    rows = []
    for s, q in seqs:
        scan_ids, _ = rol.load_keyframes(rol.split_cache("odom", "test", s, q))
        loader = rol.make_loader(s, q)
        b_new, b_old = rol.ringsharp_bounds(s), base.NSC_BOUNDS
        for i in (0, len(scan_ids) // 2):
            pts = np.ascontiguousarray(loader[int(scan_ids[i])]["points"][:, :3], dtype=np.float64)
            new = base.generate_bev(pts, Z=base.NSC_Z, Y=base.NSC_Y, X=base.NSC_X,
                                    bounds=b_new).numpy().astype(np.uint8)
            old = base.generate_bev(pts, Z=base.NSC_Z, Y=base.NSC_Y, X=base.NSC_X,
                                    bounds=b_old).numpy().astype(np.uint8)
            ref = _vox_reference(pts, b_new)
            layers = np.flatnonzero(new.reshape(new.shape[0], -1).any(axis=1))
            dz = (b_new[5] - b_new[4]) / base.NSC_Z
            r = {"sensor": s, "seq": q, "scan_id": int(scan_ids[i]),
                 "bounds": list(b_new), "occupied_voxels": int(new.sum()),
                 "occupied_voxels_old_1_20": int(old.sum()),
                 "voxels_differing_from_band_reference": int((new != ref).sum()),
                 "occupied_layer_z_range_m": ([round(b_new[4] + layers[0] * dz, 3),
                                               round(b_new[4] + (layers[-1] + 1) * dz, 3)]
                                              if len(layers) else None),
                 "identical_to_old_bev": bool(np.array_equal(new, old))}
            rows.append(r)
            print("  band", json.dumps(r), flush=True)
        del loader
    # a few voxels may differ where points sit exactly on a voxel face (NCLT's
    # 5 mm raw grid lines up with the 0.875 m x/y voxel edges): float32 rounding
    assert all(r["voxels_differing_from_band_reference"]
               <= max(10, 0.005 * r["occupied_voxels"]) for r in rows)
    assert all(r["identical_to_old_bev"] for r in rows if r["sensor"] == "nclt")
    assert not any(r["identical_to_old_bev"] for r in rows if r["sensor"] != "nclt")
    return rows


def _check_train_connector(base):
    """make_datasets() picks NSCBandTrainingDataset after install(), and its
    BEV for a KITTI and a HeLiPR tuple equals generate_bev with the table
    band (xyz_aug off for a deterministic comparison)."""
    import _ringsharp_band
    from glnet.utils.params import TrainingParams
    from glnet.datasets.dataset_utils import make_datasets
    from glnet.datasets.nsc.nsc_dataset import parse_virtual_path
    if not CONFIG.exists():
        cmd_prep()
    cls = _ringsharp_band.install()
    params = TrainingParams(str(CONFIG), str(MODEL_CONFIG))
    params.model_params.xyz_aug = False
    ds = make_datasets(params, validation=False)["global_train"]
    out = {"class": f"{type(ds).__module__}.{type(ds).__name__}",
           "is_band_class": isinstance(ds, cls)}
    assert out["is_band_class"], out
    seen = {}
    for ndx in (0, len(ds) - 1):
        dtype, seq, sid = parse_virtual_path(ds.queries[ndx].rel_scan_filepath)
        bev = ds[ndx][5].numpy()
        pts = ds._cache.points(dtype, seq, sid)
        ref = base.generate_bev(pts, Z=base.NSC_Z, Y=base.NSC_Y, X=base.NSC_X,
                                bounds=rol.ringsharp_bounds(dtype)).numpy()
        old = base.generate_bev(pts, Z=base.NSC_Z, Y=base.NSC_Y, X=base.NSC_X,
                                bounds=base.NSC_BOUNDS).numpy()
        seen[f"{dtype}/{seq}/{sid}"] = {"bounds_used": list(ds.bounds),
                                        "equal_table_band_bev": bool(np.array_equal(bev, ref)),
                                        "equal_old_1_20_bev": bool(np.array_equal(bev, old))}
        assert seen[f"{dtype}/{seq}/{sid}"]["equal_table_band_bev"]
    out["samples"] = seen
    print("  train connector", json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "train", "check", "_train-worker"):
        sub.add_parser(name)
    p = sub.add_parser("val-watch")
    p.add_argument("--exit-file", required=True,
                   help="file the run script writes when training exits")
    p.add_argument("--epochs", type=int, nargs="+", default=list(range(1, EPOCHS + 1)))
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--min-age", type=int, default=120,
                   help="seconds a checkpoint must be old before it is read")
    p.add_argument("--log-dir", default=str(LOG_DIR))
    p = sub.add_parser("val-bev")
    p.add_argument("--seq", required=True)
    p = sub.add_parser("val")
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--official-loop", action="store_true")
    p = sub.add_parser("select")
    p.add_argument("--epochs", type=int, nargs="+", default=list(range(1, EPOCHS + 1)))
    p = sub.add_parser("dump")
    p.add_argument("--cond", choices=("odom", "gt"), default="odom")
    p.add_argument("--seq", required=True)
    p.add_argument("--official-loop", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prep":
        cmd_prep()
    elif a.cmd == "train":
        cmd_train()
    elif a.cmd == "_train-worker":
        _train_worker()
    elif a.cmd == "val-watch":
        sys.exit(cmd_val_watch(a.exit_file, sorted(a.epochs), a.poll, a.min_age, a.log_dir))
    elif a.cmd == "check":
        cmd_check()
    elif a.cmd == "val-bev":
        cmd_val_bev(a.seq)
    elif a.cmd == "val":
        cmd_val(a.epoch, a.seq, a.official_loop)
    elif a.cmd == "select":
        cmd_select(sorted(a.epochs))
    elif a.cmd == "dump":
        cmd_dump(a.cond, a.seq, a.official_loop, a.overwrite)


if __name__ == "__main__":
    main()
