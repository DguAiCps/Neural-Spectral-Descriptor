#!/usr/bin/env python3
"""SeqOT official pipeline on split v2 -- part 2: the two official training
phases, sub-descriptors and checkpoint selection on split-v2 VAL (2026-10-11).

Official code (_external/SeqOT @ OFFICIAL_COMMIT, unmodified, imported):
  modules/seqTransformerCat.featureExtracter(seqL=3), modules/gem.GeM,
  tools/loss.triplet_loss.  The training loops below are the loop bodies of
  train/training_seqot.py trainHandler.train() (lines 53-179) and
  train/training_gem.py trainHandler.train() (lines 56-183) with the same
  statements in the same order; what differs is only how a step's tensors are
  found (identical values):
    * anchors: the official loop walks every index row and skips rows whose
      (anchor, dir) was seen before (O(rows^2)); here the anchors are taken in
      the same first-occurrence order directly;
    * read_one_need_from_seq / read_one_batch_pos_neg (and the descriptor
      variants) scan all index rows for the anchor, load every partner window
      (skipping the anchor if ANY window is incomplete), put positives first
      and negatives in reverse order at the end, then keep the first
      use_pos positives and batch[pos_num:pos_num+use_neg]; here the same
      selection (first use_pos positives, last use_neg negatives in reverse
      index order, same skip rule) is precomputed and only the used windows
      are read from the range store (memmap rows == official .npy files);
    * 8 TRAIN sequences instead of one: the official index rows carry a
      sequence id (train_dir1 / train_dir2, always 0 officially); the indices
      of the 8 sequences are concatenated in the split-v2 TRAIN order and the
      window of a row is read from its own sequence.
  --selftest runs the official trainHandler classes unmodified on CPU on a
  small real set and checks that this driver gives bit-identical weights.

Recipe -- the official code's actual behaviour (user rule 2026-10-11):
  phase 1  featureExtracter(seqL=3), Adam lr 5e-7 (config.yml:20), StepLR(5,
           0.9) per epoch, epochs 1..99 (range(1, 100)), one anchor per step:
           1 + 3 positives + 3 negatives (overlap > 0.3 / < 0.3, 0.3 skipped),
           triplet_loss(margin 0.5, use_min=False, lazy=False), centred 3-scan
           windows of raw scans, checkpoint every epoch
  phase 2  official gen_sub_descriptors (centred 3-scan window, filled with
           the centre scan when it leaves the sequence, N + 1 rows with a zero
           last row) with the selected phase-1 weights, then GeM (p init 3),
           Adam lr 5e-6 (config.yml:44), StepLR(5, 0.9), 99 epochs, 1 + 6 + 6,
           centred 20-window
  paper    differs (lr 5e-6 / 5e-5, 20 epochs, 6 + 6, past windows); not used
  seed     1 for torch / numpy / random (the official code sets none)
  The released weights agree on GeM lr (pretrained_gem.pth.tar initial_lr
  5e-6, epoch 15); the released phase-1 checkpoint (epoch 12, lr 1e-6 since a
  restart) matches neither code nor paper, so the code value stays.

Selection (not in the official code, which saves every epoch and has no
rule): split-v2 VAL GT keyframes, per sequence R@1 at 5 m over all keyframes
with |j - q| > 30 (query set of baselines.eval_utils.compute_recall_multi_k,
exhaustive top-1), sensor families weighted equally with queries pooled per
family (_pdv2_c3._family_pooled); best epoch, ties -> earlier epoch.
VAL descriptors use the evaluation-time windows of the protocol: trailing
3-scan / 20-scan windows of raw scans ending at the keyframe (official
fallback: a window that leaves the sequence is filled with its last scan).
Phase 1 is scored with the untrained official GeM (p = 3), phase 2 with each
GeM epoch on the selected phase-1 sub-descriptors.  Both cosine (prereg
§1.5 score) and the official L2 are recorded; METRIC (default l2, user
decision 2026-10-11) chooses.

Subcommands (inside ring_baseline; CUDA_VISIBLE_DEVICES picks the GPU):
  train1 [--epochs E --lr X --smoke N]   phase 1, resumes from the last epoch
  select1 [--follow]                     score phase-1 epochs on VAL, choose
  subdesc                                train sub-descriptors (official)
  train2 [--epochs E --lr X]             phase 2 (GeM), resumes
  select2                                score GeM epochs on VAL, choose
  smoke                                  GPU timing / memory on small subsets
  selftest [--parts ...]                 CPU: driver == official classes
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
sys.dont_write_bytecode = True

from _seqot_official_prep import (  # noqa: E402
    FOV, GEM_SEQLEN, OFFICIAL, OFFICIAL_COMMIT, RESULTS, SEED, SEQL, TRAIN, VAL,
    WORK, _git, _write_json, label_dir, official_config, official_imports, open_store,
    range_image, seq_name, sha256)

RUNS = WORK / "runs"
SUBDESC = WORK / "subdesc"
SELECTION = RESULTS / "train_selection.json"
OVERLAP_THRESH = 0.3        # training_seqot.py:51, training_gem.py:54; paper III-C
MARGIN_1 = 0.5              # training_seqot.py:152, training_gem.py:155; paper alpha
STEP_SIZE, GAMMA = 5, 0.9   # StepLR, training_seqot.py:47, training_gem.py:45; paper "0.9 every 5"
RADIUS, SKIP = 5.0, 30      # VAL R@1 (compute_recall_multi_k defaults)
# Recipe = the official code's actual behaviour (user rule 2026-10-11; the
# paper differs: lr 5e-6 / 5e-5, 20 epochs, Npos = Nneg = 6, past windows).
LR1, LR2 = 5e-7, 5e-6        # config.yml:20, :44
EPOCHS1 = EPOCHS2 = 100      # training_seqot.py:54, training_gem.py:57 -> epochs 1..99
USE1 = (3, 3)                # training_seqot.py:116-117
USE2 = (6, 6)                # training_gem.py:120-121
TRAIN_WINDOW = "centered"    # read_samples.py:20,71-72,99 (training may read future scans;
#                              evaluation uses trailing windows, see trailing_final_descriptors)
NEG_RULE = "lt"              # read_samples.py:62-68: < 0.3 negative, == 0.3 skipped
CODE = dict(lr1=LR1, lr2=LR2, epochs=100, use1=USE1, use2=USE2, window=TRAIN_WINDOW, neg=NEG_RULE)
# checkpoint selection and the reported ledger: official L2 (user decision); cosine is recorded
METRIC = os.environ.get("SEQOT_SELECT_METRIC", "l2")
LOG_EVERY = 1000


# ------------------------------------------------------------ official modules
_OFF = {}


def official():
    """featureExtracter, GeM, triplet_loss from the unmodified files."""
    if not _OFF:
        with official_imports():
            from modules.seqTransformerCat import featureExtracter
            from modules.gem import GeM
            import tools.loss as L
            for m in (featureExtracter, GeM):
                assert Path(sys.modules[m.__module__].__file__).resolve().is_relative_to(OFFICIAL.resolve())
        _OFF.update(featureExtracter=featureExtracter, GeM=GeM, loss=L)
    return _OFF


def seed_all(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------ data sources
class StoreSource:
    """Raw scans of one sequence -> (32, 900) range images, via the range
    store of _seqot_official_prep (rows == raw indices for role train)."""

    def __init__(self, role, sensor, seq):
        self.mm, self.raw_ids, self.kf, self.kf_poses, self.n_raw = open_store(role, sensor, seq)
        self.contiguous = bool(len(self.raw_ids) == self.n_raw)
        if not self.contiguous:
            self.pos = {int(r): k for k, r in enumerate(self.raw_ids)}

    def exists(self, r):            # official: os.path.exists(<root>/<r:06d>.npy)
        return 0 <= r < self.n_raw

    def rows(self, raws):
        raws = np.asarray(raws, dtype=np.int64)
        if self.contiguous:
            return self.mm[raws]
        return self.mm[np.array([self.pos[int(r)] for r in raws])]


class DictSource:
    """Selftest: {raw: image} (missing keys == missing official files)."""

    def __init__(self, d):
        self.d = d

    def exists(self, r):
        return r in self.d

    def rows(self, raws):
        return np.stack([self.d[int(r)] for r in raws])


def centered(a, seq_len):
    """np.arange(a - seq_len//2, a - seq_len//2 + seq_len) (read_samples.py:20)."""
    lo = a - seq_len // 2
    return np.arange(lo, lo + seq_len)


def trailing(a, seq_len):
    return np.arange(a - seq_len + 1, a + 1)


def window(a, seq_len, mode):
    return centered(a, seq_len) if mode == "centered" else trailing(a, seq_len)


# ------------------------------------------------------------ anchor plan
def parse_index(arr):
    """Official training index ('<U32' rows: idx1, idx2, overlap)."""
    f1 = np.array([int(x) for x in arr[:, 0]], dtype=np.int64)
    f2 = np.array([int(x) for x in arr[:, 1]], dtype=np.int64)
    ov = arr[:, 2].astype(float)              # training_seqot.py:78
    return f1, f2, ov


def anchor_plan(indices, window_ok, use_pos, use_neg, thresh=OVERLAP_THRESH, neg_rule=NEG_RULE):
    """indices: [(f1, f2, ov)] per dir (sequence) in concatenation order;
    window_ok(dir, raw) -> bool.  Returns [(dir, anchor, [pos], [neg])] in
    the official step order with the official skip rules (see docstring).
    neg_rule 'lt' = code (ov < 0.3 negative, 0.3 skipped), 'le' = paper
    (every pair that is not positive is negative)."""
    plan = []
    for d, (f1, f2, ov) in enumerate(indices):
        order = np.argsort(f1, kind="stable")                # rows of an anchor stay in index order
        keys, start = np.unique(f1[order], return_index=True)
        end = np.append(start[1:], len(order))
        first = order[start]                                 # first row of each anchor
        for g in np.argsort(first):                          # first-occurrence order
            a = int(keys[g])
            if not window_ok(d, a):                          # read_one_need_*: incomplete
                continue
            rows = order[start[g]:end[g]]                    # index order
            pos = [int(f2[j]) for j in rows if ov[j] > thresh]
            neg = [int(f2[j]) for j in rows
                   if (ov[j] < thresh - 0.0 if neg_rule == "lt" else not ov[j] > thresh)]
            if not all(window_ok(d, x) for x in pos + neg):  # read_one_batch_*: incomplete
                continue
            sel_pos = pos[:use_pos]                          # sample_batch[0:use_pos]
            sel_neg = neg[::-1][:use_neg]                    # sample_batch[pos_num:pos_num+use_neg]
            if not sel_pos or not sel_neg:                   # neg_num == 0 or pos_num == 0
                continue
            plan.append((d, a, sel_pos, sel_neg))
    return plan


def _label_meta():
    out = {}
    for s, q in TRAIN:
        m = json.loads((label_dir(s, q) / "labels_meta.json").read_text())
        out[seq_name(s, q)] = {k: m.get(k) for k in ("overlap_formula", "scale", "n_rows")}
    return out


def load_train_indices():
    out = []
    for sensor, seq in TRAIN:
        p = label_dir(sensor, seq) / "normalized_data.npy"
        out.append(parse_index(np.load(p)))
    return out


# ------------------------------------------------------------ phase 1
def stage1_batch(src, a, pos, neg, mode, seql=SEQL):
    """torch.cat((current_batch, sample_batch)) of training_seqot.py:141."""
    win = [window(a, seql, mode)] + [window(x, seql, mode) for x in pos + neg]
    return np.stack([src.rows(w) for w in win]).astype(np.float32, copy=False)


def _lr_sched(model, lr):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr)                       # training_seqot.py:46
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)  # :47
    return opt, sch


def _resume(run_dir, prefix, model, opt, sch, map_location):
    import torch
    done = sorted(int(p.name[len(prefix):-len(".pth.tar")]) for p in run_dir.glob(f"{prefix}*.pth.tar"))
    if not done:
        return 0
    last = done[-1]
    ck = torch.load(run_dir / f"{prefix}{last}.pth.tar", map_location=map_location, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    opt.load_state_dict(ck["optimizer"])
    sch.last_epoch = ck["epoch"]
    sch._last_lr = [g["lr"] for g in opt.param_groups]
    rng = run_dir / f"rng_{last}.pt"
    if rng.exists():
        st = torch.load(rng, weights_only=False)
        torch.set_rng_state(st["torch"])
        np.random.set_state(st["numpy"])
        random.setstate(st["python"])
    print(f"resumed {run_dir} from epoch {last}", flush=True)
    return last


def _save_epoch(run_dir, prefix, i, model, opt):
    """training_seqot.py:171-176 (file name of :169, in run_dir)."""
    import torch
    tmp = run_dir / f"{prefix}{i}.pth.tar.tmp"
    torch.save({'epoch': i, 'state_dict': model.state_dict(), 'optimizer': opt.state_dict()}, tmp)
    torch.save({"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "python": random.getstate()}, run_dir / f"rng_{i}.pt")
    os.replace(tmp, run_dir / f"{prefix}{i}.pth.tar")


def train_phase(kind, plan, sources, run_dir, lr, epochs, dev, init_state=None,
                start_epoch=None, log=None, quiet=False, mode=TRAIN_WINDOW):
    """The official epoch loop of training_seqot.py:69-179 (kind 'seqot') or
    training_gem.py:73-183 (kind 'gem') over a precomputed anchor plan."""
    import torch
    O = official()
    if kind == "seqot":
        model = O["featureExtracter"](seqL=SEQL)
    else:
        model = O["GeM"]()
    model.to(dev)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt, sch = _lr_sched(model, lr)
    prefix = "amodel_seqot" if kind == "seqot" else "amodel_gem"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = start_epoch if start_epoch is not None else _resume(run_dir, prefix, model, opt, sch, dev)
    losses_all = {}
    for i in range(start + 1, epochs):
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        loss_each_epoch = 0
        used_num = 0
        losses = []
        for d, a, pos, neg in plan:
            if kind == "seqot":
                arr = stage1_batch(sources[d], a, pos, neg, mode)
                current_batch = torch.from_numpy(arr[:1]).to(dev)
                sample_batch = torch.from_numpy(arr[1:]).to(dev)
            else:
                arr = stage2_batch(sources[d], a, pos, neg, mode)
                current_batch = torch.from_numpy(arr[:1]).to(dev)
                sample_batch = torch.from_numpy(arr[1:]).to(dev)
            pos_num, neg_num = len(pos), len(neg)

            input_batch = torch.cat((current_batch, sample_batch), dim=0)
            input_batch.requires_grad_(True)
            model.train()
            opt.zero_grad()

            global_des = model(input_batch)
            if kind == "gem":
                global_des = global_des.squeeze(1)

            o1, o2, o3 = torch.split(global_des, [1, pos_num, neg_num], dim=0)
            loss = O["loss"].triplet_loss(o1, o2, o3, MARGIN_1, lazy=False)

            loss.backward()
            opt.step()
            if not quiet and used_num % LOG_EVERY == 0:
                print(str(used_num), loss, flush=True)
            if torch.isnan(loss):
                print(pos_num)
                print(neg_num)

            loss_each_epoch = loss_each_epoch + loss.item()
            losses.append(loss.item())
            used_num = used_num + 1

        print("epoch {} loss {}".format(i, loss_each_epoch / max(used_num, 1)), flush=True)
        sch.step()
        _save_epoch(run_dir, prefix, i, model, opt)
        losses_all[i] = losses
        rec = {"epoch": i, "loss": loss_each_epoch / max(used_num, 1), "steps": used_num,
               "seconds": round(time.time() - t0, 1), "lr_next": opt.param_groups[0]["lr"],
               "s_per_step": round((time.time() - t0) / max(used_num, 1), 4)}
        if dev.type == "cuda":
            rec["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
        if log:
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return model, losses_all


def _plan_meta(plan, n_seq):
    per = np.bincount([p[0] for p in plan], minlength=n_seq)
    return {seq_name(*TRAIN[k]): int(per[k]) for k in range(n_seq)}


def _recipe_meta(phase):
    return {"recipe": "official code defaults (user rule 2026-10-11)",
            "paper_differs": {"lr": "5e-6" if phase == 1 else "5e-5", "epochs": 20,
                              "use_pos_neg": "(6, 6)", "window": "past (III-A)",
                              "overlap": "Eq. 13 min(valid) denominator"}}


def cmd_train1(epochs, lr, smoke):
    import torch
    cfg = official_config()
    lr = lr if lr is not None else LR1
    assert int(cfg["training_seqot"]["seqlen"]) == SEQL
    assert (cfg["training_seqot"]["height"], cfg["training_seqot"]["width"]) == (32, 900)
    seed_all()
    dev = device()
    sources = [StoreSource("train", s, q) for s, q in TRAIN]
    for s in sources:
        assert s.contiguous
    plan = anchor_plan(load_train_indices(),
                       lambda d, a: all(sources[d].exists(int(x)) for x in window(a, SEQL, TRAIN_WINDOW)),
                       *USE1)
    run_dir = RUNS / ("smoke/stage1" if smoke else "stage1")
    if smoke:
        plan = plan[:smoke]
        epochs = 2
    meta = {"official_commit": OFFICIAL_COMMIT, "lr": lr, "epochs": f"1..{epochs - 1}",
            "anchors_per_epoch": len(plan), "anchors_per_sequence": _plan_meta(plan, len(TRAIN)),
            "use_pos_neg": USE1, "window": TRAIN_WINDOW, "negative_rule": NEG_RULE,
            "labels": _label_meta(), "seed": SEED, "device": str(dev),
            "torch": torch.__version__, **_recipe_meta(1)}
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "meta.json", meta)
    print(json.dumps(meta), flush=True)
    train_phase("seqot", plan, sources, run_dir, lr, epochs, dev, log=run_dir / "log.jsonl")
    (run_dir / "finished").write_text(f"{epochs - 1}\n")


# ------------------------------------------------------------ descriptors
def subdesc_windows(raws, n_avail_ok, seql, mode):
    """Window (raw indices) of the sub-descriptor of each raw scan in `raws`,
    with the official fallback of read_one_need_from_seq (read_samples.py:
    22-30): if any scan of the window is missing, the whole window is the
    scan itself."""
    out = []
    for a in raws:
        w = centered(a, seql) if mode == "centered" else trailing(a, seql)
        if not all(n_avail_ok(int(x)) for x in w):
            w = np.full(seql, a)
        out.append(w)
    return np.stack(out)


def sub_descriptors(model, src, raws, dev, batch, mode, seql=SEQL):
    """featureExtracter (eval) on the windows of `raws` -> (len(raws), 256)."""
    import torch
    W = subdesc_windows(raws, src.exists, seql, mode)
    out = np.zeros((len(raws), 256), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for b in range(0, len(raws), batch):
            x = np.stack([src.rows(w) for w in W[b:b + batch]]).astype(np.float32, copy=False)
            out[b:b + batch] = model(torch.from_numpy(x).to(dev)).cpu().numpy()
    return out


def gem_windows(keys, ok, seq_len, mode):
    """read_one_need_descriptor_from_seq_ft (read_samples.py:96-108) window
    rule: out-of-range -> the whole window is the key's own descriptor."""
    out = []
    for a in keys:
        w = centered(a, seq_len) if mode == "centered" else trailing(a, seq_len)
        if not all(ok(int(x)) for x in w):
            w = np.full(seq_len, a)
        out.append(w)
    return np.stack(out)


def gem_final(gem, sub_lookup, keys, ok, dev, mode, seq_len=GEM_SEQLEN, batch=4096):
    """GeM (eval) over the 20-windows of `keys`; sub_lookup(raw array) -> rows."""
    import torch
    W = gem_windows(keys, ok, seq_len, mode)
    out = np.zeros((len(keys), 256), dtype=np.float32)
    gem.eval()
    with torch.no_grad():
        for b in range(0, len(keys), batch):
            x = np.stack([sub_lookup(w) for w in W[b:b + batch]]).astype(np.float32, copy=False)
            out[b:b + batch] = gem(torch.from_numpy(x).to(dev)).squeeze(1).cpu().numpy()
    return out


def trailing_final_descriptors(fe, gem, src, kf, dev, batch=64):
    """Protocol (past-only) descriptors at the keyframes `kf` (raw scan ids):
    sub-descriptors of raw scans [s-19, s] with trailing 3-windows, then GeM
    over the trailing 20-window.  Every scan read is <= s."""
    need = np.unique(np.concatenate([trailing(int(s), GEM_SEQLEN) for s in kf]))
    need = need[(need >= 0) & (need < src.n_raw)]
    sub = sub_descriptors(fe, src, need, dev, batch, "trailing")
    pos = np.full(src.n_raw, -1, dtype=np.int64)
    pos[need] = np.arange(len(need))
    final = gem_final(gem, lambda w: sub[pos[w]], kf, lambda r: 0 <= r < src.n_raw, dev, "trailing")
    return final, sub, need


# ------------------------------------------------------------ VAL selection
def recall_at_1(desc, poses, metric):
    """compute_recall_multi_k query set (5 m, 30-frame gap) with an
    exhaustive top-1 over |j - q| > 30 (cos: normalised dot, l2: exact
    float64 Euclidean); ties -> lower index."""
    from baselines.eval_utils import _find_revisit_queries, _score_recalls_from_ranked
    positions = poses[:, :3, 3].astype(np.float64)
    queries = _find_revisit_queries(positions, RADIUS, SKIP)
    if not queries:
        return 0.0, 0
    D = np.asarray(desc, dtype=np.float64)
    if metric == "cos":
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
    sq = np.einsum("ij,ij->i", D, D)
    qs = np.array([q for q, _ in queries])
    ranked = []
    for b in range(0, len(qs), 512):
        qb = qs[b:b + 512]
        G = D[qb] @ D.T
        S = G if metric == "cos" else -(sq[None, :] - 2.0 * G + sq[qb][:, None])
        for r, q in enumerate(qb):
            row = S[r]
            row[max(q - SKIP, 0):q + SKIP + 1] = -np.inf
            ranked.append(np.array([int(np.argmax(row))]) if np.isfinite(row).any()
                          else np.array([], dtype=np.int64))
    rec = _score_recalls_from_ranked(queries, ranked, positions, [1], RADIUS)
    return float(rec[1]), len(queries)


def _family_scores(per):
    from _pdv2_c3 import _family_pooled
    return {m: _family_pooled({k: {"r1": v[f"r1_{m}"], "n_queries": v["n_queries"]}
                               for k, v in per.items()}) for m in ("cos", "l2")}


def val_score(fe, gem, dev, val_sets, batch=64, sub_cache=None):
    per = {}
    for (sensor, seq), src in val_sets:
        t0 = time.time()
        if sub_cache is not None:
            sub, need = sub_cache[(sensor, seq)]
            pos = np.full(src.n_raw, -1, dtype=np.int64)
            pos[need] = np.arange(len(need))
            final = gem_final(gem, lambda w: sub[pos[w]], src.kf, lambda r: 0 <= r < src.n_raw,
                              dev, "trailing")
        else:
            final, _, _ = trailing_final_descriptors(fe, gem, src, src.kf, dev, batch)
        r_cos, nq = recall_at_1(final, src.kf_poses, "cos")
        r_l2, _ = recall_at_1(final, src.kf_poses, "l2")
        per[f"{sensor}/{seq}"] = {"r1_cos": r_cos, "r1_l2": r_l2, "n_queries": nq,
                                  "n_keyframes": int(len(src.kf)), "s": round(time.time() - t0, 1)}
        print(f"    val {sensor}/{seq}: R@1 cos {r_cos:.4f} l2 {r_l2:.4f} ({nq} q)", flush=True)
    fam = _family_scores(per)
    return {"family_r1_cos": fam["cos"], "family_r1_l2": fam["l2"], "per_seq": per}


def _load_sel():
    if SELECTION.exists():
        return json.loads(SELECTION.read_text())
    return {"_meta": {"official_commit": OFFICIAL_COMMIT, "metric": METRIC,
                      "rule": "split-v2 VAL GT keyframes, R@1 5 m, |j-q| > 30 exhaustive, "
                              "families equal / queries pooled (_pdv2_c3._family_pooled); "
                              "best epoch, ties -> earlier epoch; trailing raw-scan windows"},
            "stage1": {"epochs": {}}, "stage2": {"epochs": {}}}


def _choose(stage, metric):
    ep = stage["epochs"]
    key = f"family_r1_{metric}"
    best = min(ep, key=lambda e: (-ep[e][key], int(e)))
    return int(best)


def _val_sets(limit=0):
    out = []
    for s, q in VAL:
        src = StoreSource("val", s, q)
        if limit:
            src.kf, src.kf_poses = src.kf[:limit], src.kf_poses[:limit]
        out.append(((s, q), src))
    return out


def _load_fe(path, dev):
    import torch
    fe = official()["featureExtracter"](seqL=SEQL).to(dev)
    fe.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"])
    return fe.eval()


def _load_gem(path, dev):
    import torch
    gem = official()["GeM"]().to(dev)
    if path is not None:
        gem.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"])
    return gem.eval()


def cmd_select1(follow, batch):
    seed_all()
    dev = device()
    run_dir = RUNS / "stage1"
    val_sets = _val_sets()
    gem0 = _load_gem(None, dev)        # official GeM init, p = 3
    while True:
        sel = _load_sel()
        st = sel["stage1"]
        ck = sorted(run_dir.glob("amodel_seqot*.pth.tar"),
                    key=lambda p: int(p.name[len("amodel_seqot"):-len(".pth.tar")]))
        todo = [p for p in ck if p.name[len("amodel_seqot"):-len(".pth.tar")] not in st["epochs"]]
        for p in todo:
            e = p.name[len("amodel_seqot"):-len(".pth.tar")]
            t0 = time.time()
            print(f"select1: epoch {e}", flush=True)
            res = val_score(_load_fe(p, dev), gem0, dev, val_sets, batch)
            res.update({"checkpoint": str(p.relative_to(REPO)), "sha256": sha256(p),
                        "val_s": round(time.time() - t0, 1)})
            sel = _load_sel()
            sel["stage1"]["epochs"][e] = res
            _write_json(SELECTION, sel)
            print(f"select1: epoch {e} family R@1 cos {res['family_r1_cos']:.4f} "
                  f"l2 {res['family_r1_l2']:.4f}", flush=True)
        finished = (run_dir / "finished").exists()
        if finished and not todo:
            break
        if not follow:
            break
        time.sleep(120)
    sel = _load_sel()
    st = sel["stage1"]
    if (run_dir / "finished").exists() and st["epochs"]:
        n_expected = int((run_dir / "finished").read_text())
        assert len(st["epochs"]) == n_expected, (len(st["epochs"]), n_expected)
        e = _choose(st, METRIC)
        st["chosen"] = {"epoch": e, "metric": METRIC, **{k: st["epochs"][str(e)][k]
                                                          for k in ("checkpoint", "sha256")}}
        st["finished"] = True
        _write_json(SELECTION, sel)
        print(f"SELECT1_DONE epoch {e} ({METRIC})", flush=True)


def _chosen(stage):
    sel = _load_sel()
    c = sel[stage].get("chosen")
    assert c and sel[stage].get("finished"), f"{stage} not selected yet"
    p = REPO / c["checkpoint"]
    assert sha256(p) == c["sha256"], f"{p} changed after selection"
    return p, c


def cmd_subdesc(batch):
    """train/gen_sub_descriptors.py:40-58 per TRAIN sequence with the chosen
    phase-1 weights: centered windows, N + 1 rows (last row zero)."""
    dev = device()
    p, c = _chosen("stage1")
    fe = _load_fe(p, dev)
    out_dir = SUBDESC / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, q in TRAIN:
        out = out_dir / f"{seq_name(s, q)}.npy"
        if out.exists():
            print(f"SKIP subdesc {s}/{q}", flush=True)
            continue
        src = StoreSource("train", s, q)
        t0 = time.time()
        des = np.zeros((src.n_raw // 1 + 1, 256))
        des[:src.n_raw] = sub_descriptors(fe, src, np.arange(src.n_raw), dev, batch, "centered")
        np.save(out_dir / f"{seq_name(s, q)}.tmp.npy", des.astype("float32"))
        os.replace(out_dir / f"{seq_name(s, q)}.tmp.npy", out)
        print(f"OK subdesc {s}/{q}: {src.n_raw} scans in {time.time() - t0:.0f} s", flush=True)
    _write_json(out_dir / "meta.json", {"stage1": c, "windows": "centered (official gen_sub_descriptors)",
                                        "rows": "n_raw + 1, last row zero (official)"})


# ------------------------------------------------------------ phase 2
class DescSource:
    def __init__(self, des):
        self.des = des

    def exists(self, r):              # read_samples.py:100 (i < 0 or i >= shape[0])
        return 0 <= r < self.des.shape[0]

    def rows(self, raws):
        return self.des[np.asarray(raws)]


def stage2_batch(src, a, pos, neg, mode, seq_len=GEM_SEQLEN):
    win = [window(a, seq_len, mode)] + [window(x, seq_len, mode) for x in pos + neg]
    return np.stack([src.rows(w) for w in win]).astype(np.float32, copy=False)


def cmd_train2(epochs, lr):
    cfg = official_config()
    lr = lr if lr is not None else LR2
    assert int(cfg["training_gem"]["seqlen"]) == GEM_SEQLEN
    _chosen("stage1")
    seed_all()
    dev = device()
    sources = [DescSource(np.load(SUBDESC / "train" / f"{seq_name(s, q)}.npy")) for s, q in TRAIN]
    plan = anchor_plan(load_train_indices(),
                       lambda d, a: all(sources[d].exists(int(x)) for x in window(a, GEM_SEQLEN, TRAIN_WINDOW)),
                       *USE2)
    run_dir = RUNS / "stage2"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "meta.json", {"lr": lr, "epochs": f"1..{epochs - 1}",
                                        "anchors_per_epoch": len(plan),
                                        "anchors_per_sequence": _plan_meta(plan, len(TRAIN)),
                                        "use_pos_neg": USE2, "window": TRAIN_WINDOW,
                                        "negative_rule": NEG_RULE, **_recipe_meta(2), "seed": SEED,
                                        "stage1": _load_sel()["stage1"]["chosen"]})
    train_phase("gem", plan, sources, run_dir, lr, epochs, dev, log=run_dir / "log.jsonl")
    (run_dir / "finished").write_text(f"{epochs - 1}\n")


def cmd_select2(batch):
    dev = device()
    p1, _ = _chosen("stage1")
    fe = _load_fe(p1, dev)
    run_dir = RUNS / "stage2"
    assert (run_dir / "finished").exists(), "phase 2 not finished"
    val_sets = _val_sets()
    cache_dir = SUBDESC / "val"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sub_cache = {}
    for (s, q), src in val_sets:       # phase-1 sub-descriptors once (trailing windows)
        f = cache_dir / f"{seq_name(s, q)}.npz"
        if f.exists() and str(np.load(f)["stage1_sha256"]) == sha256(p1):
            z = np.load(f)
            sub_cache[(s, q)] = (z["sub"], z["raw"])
            continue
        need = np.unique(np.concatenate([trailing(int(k), GEM_SEQLEN) for k in src.kf]))
        need = need[(need >= 0) & (need < src.n_raw)]
        sub = sub_descriptors(fe, src, need, dev, batch, "trailing")
        np.savez(f, sub=sub, raw=need, stage1_sha256=sha256(p1))
        sub_cache[(s, q)] = (sub, need)
    for p in sorted(run_dir.glob("amodel_gem*.pth.tar"),
                    key=lambda p: int(p.name[len("amodel_gem"):-len(".pth.tar")])):
        e = p.name[len("amodel_gem"):-len(".pth.tar")]
        if e in _load_sel()["stage2"]["epochs"]:
            continue
        res = val_score(None, _load_gem(p, dev), dev, val_sets, batch, sub_cache)
        res.update({"checkpoint": str(p.relative_to(REPO)), "sha256": sha256(p)})
        sel = _load_sel()
        sel["stage2"]["epochs"][e] = res
        _write_json(SELECTION, sel)
        print(f"select2: epoch {e} family R@1 cos {res['family_r1_cos']:.4f} "
              f"l2 {res['family_r1_l2']:.4f}", flush=True)
    sel = _load_sel()
    st = sel["stage2"]
    assert len(st["epochs"]) == int((run_dir / "finished").read_text())
    e = _choose(st, METRIC)
    st["chosen"] = {"epoch": e, "metric": METRIC, **{k: st["epochs"][str(e)][k]
                                                      for k in ("checkpoint", "sha256")}}
    st["finished"] = True
    sel["finished"] = True
    _write_json(SELECTION, sel)
    print(f"SELECT2_DONE epoch {e} ({METRIC})", flush=True)


# ------------------------------------------------------------ smoke (GPU timing)
def cmd_smoke(n_anchor, n_kf, batch):
    import torch
    dev = device()
    out = {"device": str(dev)}
    if dev.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(dev)
    sm = RUNS / "smoke"
    if sm.exists():
        shutil.rmtree(sm)
    t0 = time.time()
    cmd_train1(EPOCHS1, None, smoke=n_anchor)
    out["stage1"] = [json.loads(x) for x in (sm / "stage1/log.jsonl").read_text().splitlines()]
    fe = _load_fe(sm / "stage1/amodel_seqot1.pth.tar", dev)
    vs = _val_sets(limit=n_kf)
    torch.cuda.reset_peak_memory_stats(dev) if dev.type == "cuda" else None
    t1 = time.time()
    res = val_score(fe, _load_gem(None, dev), dev, vs, batch)
    out["val"] = {"keyframes": sum(len(s.kf) for _, s in vs), "s": round(time.time() - t1, 1),
                  "family_r1_cos": res["family_r1_cos"]}
    if dev.type == "cuda":
        out["val"]["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
    # phase 2 timing on random sub-descriptors of the right shape (no selection)
    sources = [DescSource(np.random.default_rng(0).standard_normal((StoreSource("train", s, q).n_raw + 1, 256))
                          .astype(np.float32) / 16) for s, q in TRAIN]
    plan = anchor_plan(load_train_indices(),
                       lambda d, a: all(sources[d].exists(int(x)) for x in window(a, GEM_SEQLEN, TRAIN_WINDOW)),
                       *USE2)[:n_anchor]
    t2 = time.time()
    train_phase("gem", plan, sources, sm / "stage2", LR2, 2, dev, log=sm / "stage2/log.jsonl", quiet=True)
    out["stage2_s_per_step"] = round((time.time() - t2) / max(len(plan), 1), 4)
    out["total_s"] = round(time.time() - t0, 1)
    _write_json(sm / "smoke.json", out)
    print("SMOKE", json.dumps(out), flush=True)


# ------------------------------------------------------------ CPU selftest
@contextlib.contextmanager
def cpu_cuda_shim():
    """The official read_samples / trainHandlers call .cuda() unconditionally."""
    import torch
    saved = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    try:
        yield
    finally:
        torch.Tensor.cuda = saved


def _official_train_modules():
    """train/training_seqot.py, train/training_gem.py, train/gen_sub_descriptors.py
    and test/test_gem_prepare.py imported unmodified (their `from tools...`
    imports resolve to the clone; read_samples' `from utils.utils import *`
    to tools/utils)."""
    import importlib.util as ilu
    mods = {}
    with official_imports(utils_as="package"):
        for name, rel in (("training_seqot", "train/training_seqot.py"),
                          ("training_gem", "train/training_gem.py"),
                          ("gen_sub_descriptors", "train/gen_sub_descriptors.py"),
                          ("test_gem_prepare", "test/test_gem_prepare.py")):
            spec = ilu.spec_from_file_location(f"seqot_official_{name}", OFFICIAL / rel)
            m = ilu.module_from_spec(spec)
            spec.loader.exec_module(m)
            mods[name] = m
    return mods


def _toy_set():
    """Real KITTI 06 range images (our adapter) of the raw scans around 12
    every-5th frames spread over the sequence, the overlaps between them
    (overlap_row, KITTI window) as an official index (all pairs j >= i,
    shuffled, + a duplicate row and a row with overlap exactly 0.3).
    Frame 0 (window needs raw -1) and a frame whose raw+1 file is withheld
    exercise the official incomplete-window skips."""
    from _pd_build_test_caches import _make_loader
    from _seqot_official_prep import lidar_poses, overlap_row, official_points
    L = _make_loader("kitti", "06")
    P = lidar_poses("kitti", "06", L)
    frames = [0, 20, 21, 22, 24, 27, 60, 61, 63, 120, 121, 200]   # every-5th frame ids
    withheld = 5 * 63 + 1
    imgs = {}
    for f in frames:
        for r in (5 * f - 1, 5 * f, 5 * f + 1):
            if r >= 0 and r != withheld:
                imgs[r] = range_image("kitti", L[r]["points"])
    pts = {f: official_points("kitti", L[5 * f]["points"]) for f in frames}
    rows = []
    for a_i, a in enumerate(frames):
        ov = overlap_row(pts[a], P[5 * a], [(pts[b], P[5 * b]) for b in frames[a_i:]], FOV["kitti"])
        for b, o in zip(frames[a_i:], ov):
            # like the official index, a subset of the pairs: the frame with
            # the withheld file (63) and frame 0 appear only as partners of
            # anchors 60 / 0, which the official reader must skip
            if (b == 63 and a not in (60, 63)) or (b == 0 and a != 0):
                continue
            rows.append([str(5 * a).zfill(6), str(5 * b).zfill(6), np.float64(o)])
    rows.append(list(rows[13]))                     # a duplicated row
    rows.append([str(5 * 20).zfill(6), str(5 * 61).zfill(6), np.float64(0.3)])
    rng = np.random.default_rng(7)
    arr = np.array(rows)
    arr = arr[rng.permutation(len(arr))]
    return imgs, arr


def _state_equal(a, b):
    return all(np.array_equal(a[k].cpu().numpy(), b[k].cpu().numpy()) for k in a) and a.keys() == b.keys()


def selftest_train1(rep, td):
    """Official training_seqot.trainHandler.train(), unmodified, on CPU for
    one epoch (resume from a seed-1 initial model saved as epoch 98, so its
    loop runs epoch 99 only) vs train_phase on the same data."""
    import torch
    mods = _official_train_modules()
    imgs, arr = _toy_set()
    root = td / "range_image" / "depth"
    root.mkdir(parents=True)
    for r, im in imgs.items():
        np.save(root / f"{r:06d}.npy", im)
    np.save(td / "index.npy", arr)
    np.save(td / "poses.npy", np.zeros((1, 4, 4)))
    torch.manual_seed(SEED)
    init = official()["featureExtracter"](seqL=SEQL).state_dict()
    torch.save({"epoch": 98, "state_dict": init}, td / "init.pth.tar")
    lr = float(official_config()["training_seqot"]["lr"])
    cwd = os.getcwd()
    os.chdir(td)
    buf = io.StringIO()
    t0 = time.time()
    try:
        with cpu_cuda_shim(), contextlib.redirect_stdout(buf):
            h = mods["training_seqot"].trainHandler(
                height=32, width=900, seqlen=SEQL, lr=lr, resume=True,
                pretrained_weights=str(td / "init.pth.tar"), train_set=str(td / "index.npy"),
                poses_file=str(td / "poses.npy"), range_image_root=str(root) + "/")
            h.train()
    finally:
        os.chdir(cwd)
    t_off = time.time() - t0
    off = torch.load(td / "amodel_seqot99.pth.tar", weights_only=False)
    off_losses = [float(l.split("tensor(")[1].split(",")[0]) for l in buf.getvalue().splitlines()
                  if "tensor(" in l]
    # ours
    src = DictSource(imgs)
    plan = anchor_plan([parse_index(arr)], lambda d, a: all(src.exists(int(x)) for x in centered(a, SEQL)),
                       *CODE["use1"], neg_rule=CODE["neg"])
    t0 = time.time()
    model, losses = train_phase("seqot", plan, [src], td / "ours1", lr, 100, torch.device("cpu"),
                                init_state=init, start_epoch=98, quiet=True, mode=CODE["window"])
    t_ours = time.time() - t0
    ours = torch.load(td / "ours1" / "amodel_seqot99.pth.tar", weights_only=False)
    case = {"anchors_official_steps": len(off_losses), "anchors_ours": len(plan),
            "plan": [(a, p, n) for _, a, p, n in plan],
            "branches_pos_neg_counts": sorted({(len(p), len(n)) for _, _, p, n in plan}),
            "losses_equal_printed_4dp": bool(len(off_losses) == len(losses[99]) and
                                             np.allclose(off_losses, losses[99], rtol=0, atol=1e-4)),
            "losses_official_printed": off_losses, "losses_ours": losses[99],
            "state_dict_bitwise_equal": _state_equal(off["state_dict"], ours["state_dict"]),
            "optimizer_state_equal": bool(all(
                np.array_equal(off["optimizer"]["state"][k]["exp_avg"].numpy(),
                               ours["optimizer"]["state"][k]["exp_avg"].numpy())
                for k in off["optimizer"]["state"])),
            "official_s": round(t_off, 1), "ours_s": round(t_ours, 1),
            "s_per_step_cpu": round(t_ours / max(len(plan), 1), 2)}
    rep["train1"] = case
    print("train1", json.dumps({k: v for k, v in case.items() if "losses_" not in k}), flush=True)
    return case["state_dict_bitwise_equal"] and case["optimizer_state_equal"] and \
        case["anchors_official_steps"] == case["anchors_ours"] and case["losses_equal_printed_4dp"]


def selftest_train2(rep, td):
    """Official training_gem.trainHandler.train() (unmodified: it forces
    resume=False and runs epochs 1..99) on CPU vs train_phase('gem'); every
    epoch checkpoint compared."""
    import torch
    mods = _official_train_modules()
    _, arr = _toy_set() if not (td / "index.npy").exists() else (None, np.load(td / "index.npy"))
    rng = np.random.default_rng(3)
    descs = (rng.standard_normal((1102, 256)) / 16).astype(np.float32)
    descs[-1] = 0.0                                  # official N + 1 rows, last zero
    np.save(td / "index.npy", arr)
    np.save(td / "poses.npy", np.zeros((1, 4, 4)))
    lr = float(official_config()["training_gem"]["lr"])
    run = td / "gem_official"
    run.mkdir()
    cwd = os.getcwd()
    os.chdir(run)
    t0 = time.time()
    try:
        with cpu_cuda_shim(), contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(SEED)
            h = mods["training_gem"].trainHandler(lr=lr, seqlen=GEM_SEQLEN, resume=False,
                                                  pretrained_weights=None, descriptors=descs,
                                                  train_set=str(td / "index.npy"),
                                                  poses_file=str(td / "poses.npy"))
            h.train()
    finally:
        os.chdir(cwd)
    t_off = time.time() - t0
    src = DescSource(descs)
    plan = anchor_plan([parse_index(arr)], lambda d, a: all(src.exists(int(x)) for x in centered(a, GEM_SEQLEN)),
                       *CODE["use2"], neg_rule=CODE["neg"])
    torch.manual_seed(SEED)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        train_phase("gem", plan, [src], td / "gem_ours", lr, 100, torch.device("cpu"), start_epoch=0,
                    quiet=True, mode=CODE["window"])
    t_ours = time.time() - t0
    eq, ps = [], []
    for i in range(1, 100):
        a = torch.load(run / f"amodel_gem{i}.pth.tar", weights_only=False)
        b = torch.load(td / "gem_ours" / f"amodel_gem{i}.pth.tar", weights_only=False)
        eq.append(_state_equal(a["state_dict"], b["state_dict"]))
        ps.append(float(b["state_dict"]["p"]))
    case = {"anchors": len(plan), "epochs_compared": len(eq), "all_epochs_bitwise_equal": all(eq),
            "p_epoch1_99": [ps[0], ps[-1]], "official_s": round(t_off, 1), "ours_s": round(t_ours, 1)}
    rep["train2"] = case
    print("train2", json.dumps(case), flush=True)
    return case["all_epochs_bitwise_equal"]


def selftest_descriptors(rep, td):
    """(1) official gen_sub_descriptors.testHandler.eval() == our centered
    sub-descriptors (batch 1: bitwise; batch 8: max |diff|); (2) official
    test_gem_prepare.testHandler.eval() == our centered GeM descriptors + L2
    (same top-22 lists, distances); (3) our trailing windows == the official
    centered pipeline shifted (sub: s -> s-1, final: s -> s-10) away from the
    edges; (4) trailing descriptors are prefix-invariant (causal)."""
    import torch
    mods = _official_train_modules()
    from _pd_build_test_caches import _make_loader
    L = _make_loader("kitti", "06")
    n = 30
    imgs = {r: range_image("kitti", L[r]["points"]) for r in range(n)}
    root = td / "sub_root"
    root.mkdir()
    for r, im in imgs.items():
        np.save(root / f"{r:06d}.npy", im)
    torch.manual_seed(SEED)
    fe = official()["featureExtracter"](seqL=SEQL)
    torch.save({"epoch": 1, "state_dict": fe.state_dict()}, td / "fe.pth.tar")
    work = td / "subdesc_official"
    work.mkdir()
    cwd = os.getcwd()
    os.chdir(work)
    try:
        with cpu_cuda_shim(), contextlib.redirect_stdout(io.StringIO()):
            h = mods["gen_sub_descriptors"].testHandler(seqlen=SEQL, pretrained_weights=str(td / "fe.pth.tar"),
                                                        range_image_database_root=str(root) + "/",
                                                        range_image_query_root=str(root) + "/")
            h.eval()
    finally:
        os.chdir(cwd)
    off = np.load(work / "des_list_database.npy")
    src = DictSource(imgs)
    src.n_raw = n
    cpu = torch.device("cpu")
    fe_l = _load_fe(td / "fe.pth.tar", cpu)
    ours1 = np.zeros((n + 1, 256))
    ours1[:n] = sub_descriptors(fe_l, src, np.arange(n), cpu, 1, "centered")
    ours1 = ours1.astype("float32")
    ours8 = sub_descriptors(fe_l, src, np.arange(n), cpu, 8, "centered")
    case = {"sub_shape": [list(off.shape), list(ours1.shape)],
            "sub_batch1_bitwise_equal": bool(np.array_equal(off, ours1)),
            "sub_batch8_max_abs_diff": float(np.abs(off[:n] - ours8).max())}
    # (3a) trailing sub(s) == official centered sub(s-1)
    tr = sub_descriptors(fe_l, src, np.arange(n), cpu, 1, "trailing")
    case["trailing_sub_eq_centered_shift"] = bool(np.array_equal(tr[2:n], off[1:n - 1]))
    # (2) GeM eval
    rng = np.random.default_rng(5)
    db = (rng.standard_normal((251, 256)) / 16).astype(np.float32)
    qy = (rng.standard_normal((30, 256)) / 16).astype(np.float32)
    gem = official()["GeM"]()
    with torch.no_grad():
        gem.p.fill_(3.3)
    torch.save({"epoch": 1, "state_dict": gem.state_dict()}, td / "gem.pth.tar")
    work2 = td / "gem_official_eval"
    work2.mkdir()
    os.chdir(work2)
    try:
        with cpu_cuda_shim(), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            h = mods["test_gem_prepare"].testHandler(seqlen=GEM_SEQLEN, pretrained_weights=str(td / "gem.pth.tar"),
                                                     descs_database=db, descs_query=qy)
            h.eval()
    finally:
        os.chdir(cwd)
    pred = np.load(work2 / "predicted_L2_dis.npz")["arr_0"].reshape(-1, 3)
    gem_l = _load_gem(td / "gem.pth.tar", cpu)
    dbk = np.arange(0, 251, 10)
    fdb = gem_final(gem_l, lambda w: db[w], dbk, lambda r: 0 <= r < 251, cpu, "centered")
    qk = np.arange(0, 30, 5)
    fq = gem_final(gem_l, lambda w: qy[w], qk, lambda r: 0 <= r < 30, cpu, "centered")
    d2 = ((fq[:, None, :].astype(np.float64) - fdb[None].astype(np.float64)) ** 2).sum(-1)
    lists_eq, dmax = True, 0.0
    for qi, q in enumerate(qk):
        r = pred[pred[:, 0] == q]
        order = np.argsort(d2[qi], kind="stable")[:22]
        lists_eq &= bool(np.array_equal(r[:, 1].astype(int), dbk[order]))
        dmax = max(dmax, float(np.abs(r[:, 2] - d2[qi][order]).max() / d2[qi][order].max()))
    case.update({"gem_eval_lists_equal": lists_eq, "gem_eval_max_rel_dist_diff": dmax})
    # (3b) GeM over the same sub-descriptor array: trailing window at s ==
    # official centered window at s-9 ([s-19, s]); with the sub-descriptor
    # shift of (3a) the full pipeline shift is 10
    subs = (rng.standard_normal((80, 256)) / 16).astype(np.float32)
    ok = lambda r: 0 <= r < 80  # noqa: E731
    keys = np.arange(30, 80)
    t_f = gem_final(gem_l, lambda w: subs[w], keys, ok, cpu, "trailing")
    off_c = []
    with cpu_cuda_shim():
        rs = mods["training_gem"].__dict__.get("read_one_need_descriptor_from_seq_ft")
        for k in keys:
            x, _ = rs(str(k - 9).zfill(6), subs, GEM_SEQLEN)
            off_c.append(gem_l(x).squeeze(1)[0].detach().numpy())
    case["trailing_final_eq_official_centered_shift"] = bool(np.array_equal(t_f, np.stack(off_c)))
    # (4) prefix invariance of the protocol descriptors
    src_full = DictSource(imgs)
    src_full.n_raw = n
    kf = np.array([3, 9, 15, 22, 29])
    f_full, _, _ = trailing_final_descriptors(fe_l, gem_l, src_full, kf, cpu, 4)
    src_cut = DictSource({r: imgs[r] for r in range(23)})
    src_cut.n_raw = 23
    f_cut, _, _ = trailing_final_descriptors(fe_l, gem_l, src_cut, kf[:4], cpu, 4)
    case["trailing_prefix_invariant_max_abs"] = float(np.abs(f_full[:4] - f_cut).max())
    rep["descriptors"] = case
    print("descriptors", json.dumps(case), flush=True)
    return case["sub_batch1_bitwise_equal"] and case["sub_batch8_max_abs_diff"] < 1e-5 and \
        case["trailing_sub_eq_centered_shift"] and lists_eq and dmax < 1e-5 and \
        case["trailing_final_eq_official_centered_shift"] and \
        case["trailing_prefix_invariant_max_abs"] < 1e-5


def selftest_resume(rep, td):
    """Our resume: 2 epochs straight == 1 epoch + resume (tiny GeM run)."""
    import torch
    rng = np.random.default_rng(4)
    descs = (rng.standard_normal((1102, 256)) / 16).astype(np.float32)
    arr = np.load(td / "index.npy")
    src = DescSource(descs)
    plan = anchor_plan([parse_index(arr)], lambda d, a: all(src.exists(int(x)) for x in centered(a, GEM_SEQLEN)), *USE2)
    with contextlib.redirect_stdout(io.StringIO()):
        seed_all()
        train_phase("gem", plan, [src], td / "r_a", 5e-3, 13, torch.device("cpu"), start_epoch=0, quiet=True)
        seed_all()
        train_phase("gem", plan, [src], td / "r_b", 5e-3, 7, torch.device("cpu"), start_epoch=0, quiet=True)
        seed_all()
        train_phase("gem", plan, [src], td / "r_b", 5e-3, 13, torch.device("cpu"), quiet=True)
    a = torch.load(td / "r_a" / "amodel_gem12.pth.tar", weights_only=False)
    b = torch.load(td / "r_b" / "amodel_gem12.pth.tar", weights_only=False)
    case = {"resume_equal": _state_equal(a["state_dict"], b["state_dict"]),
            "lr_equal": a["optimizer"]["param_groups"][0]["lr"] == b["optimizer"]["param_groups"][0]["lr"]}
    rep["resume"] = case
    print("resume", json.dumps(case), flush=True)
    return case["resume_equal"] and case["lr_equal"]


def selftest_recall(rep):
    """Exhaustive R@1 == compute_recall_multi_k where the latter's top-61
    shortlist cannot bind (unstructured descriptors: the 61 nearest are never
    all within +-30 frames), on a two-lap trajectory."""
    from baselines.eval_utils import compute_recall_multi_k
    rng = np.random.default_rng(0)
    n = 400
    t = np.linspace(0, 4 * np.pi, n)
    pos = np.stack([30 * np.cos(t), 30 * np.sin(t), 0 * t], 1)   # two laps
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, 3] = pos
    desc = rng.standard_normal((n, 32)).astype(np.float32)
    r_ours, nq = recall_at_1(desc, poses, "cos")
    r_ref, nq_ref = compute_recall_multi_k(desc, poses, k_values=[1])
    case = {"ours": r_ours, "compute_recall_multi_k": r_ref[1], "n_queries": [nq, nq_ref]}
    rep["recall"] = case
    print("recall", json.dumps(case), flush=True)
    return abs(r_ours - r_ref[1]) < 1e-12 and nq == nq_ref


def cmd_selftest(parts, out_json):
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    rep = {"official_commit": OFFICIAL_COMMIT, "torch": torch.__version__}
    ok = True
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if "recall" in parts:
            ok &= selftest_recall(rep)
        if "train1" in parts:
            ok &= selftest_train1(rep, td)
        if "train2" in parts:
            ok &= selftest_train2(rep, td)
        if "resume" in parts:
            if not (td / "index.npy").exists():
                np.save(td / "index.npy", _toy_set()[1])
            ok &= selftest_resume(rep, td)
        if "descriptors" in parts:
            ok &= selftest_descriptors(rep, td)
    st = _git("status", "--porcelain", check=False)
    rep["official_clone_status"] = st.splitlines()
    rep["all_ok"] = bool(ok)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out_json, rep)
    print("TRAIN_SELFTEST_OK" if ok else "TRAIN_SELFTEST_FAILED", flush=True)
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train1")
    p.add_argument("--epochs", type=int, default=EPOCHS1, help="official loop bound (100 -> epochs 1..99)")
    p.add_argument("--lr", type=float, default=None, help="default config.yml:20 5e-7")
    p.add_argument("--smoke", type=int, default=0)
    p = sub.add_parser("select1")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("subdesc")
    p.add_argument("--batch", type=int, default=1, help="1 = official gen_sub_descriptors")
    p = sub.add_parser("train2")
    p.add_argument("--epochs", type=int, default=EPOCHS2, help="official loop bound (100 -> epochs 1..99)")
    p.add_argument("--lr", type=float, default=None, help="default config.yml:44 5e-6")
    p = sub.add_parser("select2")
    p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("smoke")
    p.add_argument("--anchors", type=int, default=200)
    p.add_argument("--keyframes", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p = sub.add_parser("selftest")
    p.add_argument("--parts", nargs="+", default=["recall", "train1", "train2", "resume", "descriptors"])
    p.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "train1":
        cmd_train1(a.epochs, a.lr, a.smoke)
    elif a.cmd == "select1":
        cmd_select1(a.follow, a.batch)
    elif a.cmd == "subdesc":
        cmd_subdesc(a.batch)
    elif a.cmd == "train2":
        cmd_train2(a.epochs, a.lr)
    elif a.cmd == "select2":
        cmd_select2(a.batch)
    elif a.cmd == "smoke":
        cmd_smoke(a.anchors, a.keyframes, a.batch)
    elif a.cmd == "selftest":
        cmd_selftest(a.parts, a.out_json)


if __name__ == "__main__":
    main()
