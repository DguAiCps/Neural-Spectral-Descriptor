#!/usr/bin/env python3
"""HeLiOS official pipeline on split v2 -- part 3: training (2026-10-11).

Runs the OFFICIAL training code of _external/HeLiOS (minwoo0611/HeLiOS,
6915b75) on the split-v2 pickles of scripts/_helios_official_overlap.py.
Imported from the official files and executed unchanged:
  misc/utils.TrainingParams / ModelParams   official config values:
      config/config_baseline_helios.txt (only dataset_folder replaced, see
      write_config) and config/model_helios.txt (unchanged file)
  training/trainer.initialize_model_and_optimizer   model_factory, Adam lr 1e-3,
      weight decay 1e-4, MultiStepLR milestones 40, 70, gamma 0.1
  training/trainer.multistaged_training_step   batch 2048 in mini-batches of
      64: stage 1 embeddings without grad, stage 2 losses, stage 3 recompute
      and backward (train mode in all stages, as in the official code)
  training/trainer.aggregate_epoch_stats
  datasets/dataset_utils.make_dataloaders   PNVTrainingDataset + PNV loader,
      TrainTransform(aug_mode 1: jitter, random point removal, translation,
      block removal), TrainSetTransform (+-5 deg yaw, x/y flip), BatchSampler
      (pairs of positives), collate_fn (positive / semi-positive / negative
      masks and overlaps from the pickle, CartesianQuantizer 0.01), 4 workers
  models/losses/loss.make_losses   TruncatedSmoothAP (tau 0.01, 4 positives,
      euclidean) + two overlap-guided batch-hard triplet losses (margins 0.02
      and 2 x 0.19), weights 0.8 / 0.1 / 0.1 (config values; the paper's eq. 6
      weight 1 is not used: public-code rule)
The epoch loop of trainer.do_train (trainer.py:287-353) is repeated in
`run_training` with these differences (marked CHANGED):
  1. weights are written to the run directory instead of
     _external/HeLiOS/weights (trainer.create_weights_folder);
  2. the final official evaluate() is not called: it reads the Roundabout
     evaluation pickles (test data of the official split), which do not exist
     here; selection / test are separate stages (_helios_official_dump.py);
  3. resume.pth after every epoch (model, optimizer, scheduler, maximum_ap,
     stats, RNG states) and --resume; the official loop has no resume;
  4. seed 1 for random / numpy / torch (the official code sets none);
  5. a per-epoch JSON record (stats, times, GPU peak memory);
  6. guard_empty_miner_stats: a batch whose hard-triplet miner has no valid
     anchor would crash the official statistics line (also on torch 1.13 /
     pml 1.1.2); its six logged distance statistics become NaN, the loss (0 for
     zero triplets) and mining are the official ones.  Counted per epoch.
Saved like the official loop: <run>/HeLiOS_default_trainning_best.pth whenever
the validation-phase AP (TruncatedSmoothAP statistic on validation.pickle =
Roundabout01) beats every earlier epoch, _{10,20,...}.pth (save_freq default
10), _final.pth.

Usage (inside ring_baseline):
  python3 scripts/_helios_official_train.py                 # full run
  python3 scripts/_helios_official_train.py --resume <run dir>
  python3 scripts/_helios_official_train.py --smoke 300     # timing / memory
  python3 scripts/_helios_official_train.py --selftest      # CPU only
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

import _helios_official_prep as P  # noqa: E402

HELIOS = P.HELIOS
OFFICIAL_CONFIG = HELIOS / "config/config_baseline_helios.txt"
MODEL_CONFIG = HELIOS / "config/model_helios.txt"
# training seed: environment variable SEED (default 1).  SEED = 1 keeps every
# path of the first run; any other seed gets its own config / runs / smoke /
# results (suffix _s<SEED>).  Prepared scans, pickles and overlap files
# (_external/helios_official_pdv2/{data,overlap}) are shared by all seeds.
SEED = int(os.environ.get("SEED", "1"))
SFX = "" if SEED == 1 else f"_s{SEED}"
CONFIG = P.WORK / f"config_pdv2{SFX}.txt"
RUNS = P.WORK / f"runs{SFX}"
SMOKE = P.WORK / f"smoke{SFX}"
RESULTS = REPO / f"results/pdv2_helios_official{SFX}"
RECORD = RESULTS / "train_record.json"


PML112 = P.HENV / "pml112"   # pytorch-metric-learning 1.1.2 (official Dockerfile pin)


def official_path():
    """Official repo + vendored SparseTransformer on sys.path, and
    pytorch-metric-learning 1.1.2 (pip --target copy) ahead of the
    container's 2.9.0, with which the official loss fails
    (AvgNonZeroReducer has no `triplets_past_filter`; see selftest)."""
    for p in (str(HELIOS), str(HELIOS / "docker/thirdparty/SparseTransformer"), str(PML112)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    assert "pytorch_metric_learning" not in sys.modules or \
        sys.modules["pytorch_metric_learning"].__version__ == "1.1.2", "pml imported before official_path()"


def guard_empty_miner_stats():
    """Run-only guard (category 2; the official stack fails the same way:
    torch 1.13.1 + pml 1.1.2 raise the identical RuntimeError).
    HardTripletMinerWithMasks._update_stats (models/losses/loss.py:191-203)
    takes torch.max / min / mean over the valid anchors; when no row of a batch
    has both kinds of partner (e.g. no semi-positive in a small batch) the set
    is empty and torch.max raises.  Mining and the loss are NOT changed: with
    zero triplets the official loss returns 0 (zero_losses).  For such a call
    only the six logged distance statistics become NaN; every other call runs
    the official method.  Calls are counted in EMPTY_MINER_CALLS."""
    import models.losses.loss as L
    orig = L.HardTripletMinerWithMasks._update_stats
    if getattr(orig, "_pdv2_guard", False):
        return
    def _update_stats(self, pos_dist, neg_dist, valid_idx):  # noqa: E306
        if pos_dist[valid_idx].numel() == 0:
            EMPTY_MINER_CALLS[0] += 1
            self.stats.update({k: float("nan") for k in self.stats})
            return
        return orig(self, pos_dist, neg_dist, valid_idx)
    _update_stats._pdv2_guard = True
    _update_stats._pdv2_orig = orig
    L.HardTripletMinerWithMasks._update_stats = _update_stats


EMPTY_MINER_CALLS = [0]


def write_config(path: Path, dataset_folder: Path, **train_overrides) -> Path:
    """Official config_baseline_helios.txt with dataset_folder replaced
    (and, for smoke / selftest only, the given [TRAIN] keys)."""
    import configparser
    cp = configparser.ConfigParser()
    cp.read(OFFICIAL_CONFIG)
    cp["DEFAULT"]["dataset_folder"] = str(dataset_folder)
    for k, v in train_overrides.items():
        cp["TRAIN"][k] = str(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        cp.write(f)
    return path


def _rng_state():
    import torch
    st = {"random": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    import torch
    random.setstate(st["random"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def run_training(params, run_dir: Path, record_path: Path, resume: bool,
                 device: str, model_override=None, max_batches: int = 0):
    """trainer.do_train (trainer.py:287-353) with the CHANGED points of the
    module docstring.  model_override / max_batches: CPU selftest only."""
    import torch
    official_path()
    from datasets.dataset_utils import make_dataloaders
    from models.losses.loss import make_losses
    import training.trainer as T
    guard_empty_miner_stats()                              # CHANGED 6 (run-only)

    random.seed(SEED)                                      # CHANGED 4
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if model_override is None:
        model, optimizer, scheduler = T.initialize_model_and_optimizer(params, device)
    else:   # same optimizer / scheduler construction as the official function
        model = model_override.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
    T.log_model_info(model, params.model_name)

    dataloaders = make_dataloaders(params)
    loss_fn = make_losses(params)
    loss_weights = [params.truncated_weight, params.guided_p_sp_weight, params.guided_sp_n_weight]

    run_dir.mkdir(parents=True, exist_ok=True)              # CHANGED 1
    model_pathname = str(run_dir / params.model_name)

    stats = {"train": [], "val": []} if "val" in dataloaders else {"train": []}
    phases = ["train", "val"] if "val" in dataloaders else ["train"]
    maximum_ap = 0
    start = 1
    record = {"_meta": {"official_repo": "https://github.com/minwoo0611/HeLiOS",
                        "official_commit": P.git_head(HELIOS), "driver": "scripts/_helios_official_train.py",
                        "config": str(params.params_path), "model_config": str(params.model_params_path),
                        "run_dir": str(run_dir), "seed": SEED, "device": device,
                        "train_pickle": params.train_file, "val_pickle": params.val_file,
                        "n_train": len(dataloaders["train"].dataset),
                        "n_val": len(dataloaders["val"].dataset) if "val" in dataloaders else 0,
                        "selection_official": "model_best = highest validation-phase AP "
                                              "(trainer.py:300-302)",
                        "loss_weights": loss_weights},
              "epochs": []}
    if resume:                                              # CHANGED 3
        ck = torch.load(run_dir / "resume.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        maximum_ap, stats, start = ck["maximum_ap"], ck["stats"], ck["epoch"] + 1
        _set_rng_state(ck["rng"])
        record = json.loads(record_path.read_text())
        record["epochs"] = [e for e in record["epochs"] if e["epoch"] < start]
        record.setdefault("resumed_at_epoch", []).append(start)
        print(f"===> resumed after epoch {ck['epoch']} (maximum_ap {maximum_ap:.4f})", flush=True)

    for epoch in range(start, params.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        times, is_best = {}, False
        empty0 = EMPTY_MINER_CALLS[0]
        for phase in phases:
            t0 = time.time()
            running_stats = []
            count_batches = 0
            global_iter = iter(dataloaders[phase])
            while True:
                count_batches += 1
                if params.debug and count_batches > 2:
                    break
                if max_batches and count_batches > max_batches:
                    break
                try:
                    batch_stats = T.multistaged_training_step(
                        global_iter, model, phase, device, optimizer, loss_fn, loss_weights)
                    running_stats.append(batch_stats)
                except StopIteration:
                    break
            T.aggregate_epoch_stats(running_stats, phase, stats)
            if phase == "val" and stats[phase][-1]["global"]["ap"] > maximum_ap:
                maximum_ap = stats[phase][-1]["global"]["ap"]
                torch.save(model.state_dict(), f"{model_pathname}_best.pth")
                is_best = True
            times[phase] = round(time.time() - t0, 1)
            times[f"{phase}_batches"] = len(running_stats)
        if scheduler is not None:
            scheduler.step()
        if params.save_freq > 0 and epoch % params.save_freq == 0:
            torch.save(model.state_dict(), f"{model_pathname}_{epoch}.pth")

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "maximum_ap": maximum_ap, "stats": stats,
                    "rng": _rng_state()}, run_dir / "resume.pth.tmp")
        os.replace(run_dir / "resume.pth.tmp", run_dir / "resume.pth")
        g = {ph: stats[ph][-1]["global"] for ph in phases}
        record["epochs"].append({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            **{f"{ph}_{k}": float(g[ph][k]) for ph in phases
               for k in ("loss_truncated", "loss_triplet_ps", "loss_triplet_sn", "ap")},
            "saved_best": is_best,
            "empty_miner_calls": EMPTY_MINER_CALLS[0] - empty0,
            "maximum_ap": float(maximum_ap), **times,
            "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            if torch.cuda.is_available() else None})
        _write_json(record_path, record)
        print(f"===> epoch {epoch}: {times}, val ap {g.get('val', {}).get('ap', float('nan')):.4f}, "
              f"best {maximum_ap:.4f}", flush=True)

    final_model_path = f"{model_pathname}_final.pth"
    print(f"Saving weights: {final_model_path}")
    torch.save(model.state_dict(), final_model_path)
    # CHANGED 2: no official evaluate() here
    record["finished"] = True
    record["checkpoints"] = {p.name: P.sha256(p) for p in sorted(run_dir.glob("*.pth"))
                             if p.name != "resume.pth"}
    _write_json(record_path, record)
    print("HELIOS_OFFICIAL_TRAIN_DONE", flush=True)
    return record


# ------------------------------------------------------------ smoke
def subset_pickle(src: Path, dst: Path, n_per_folder: int):
    """First n queries of every folder; ids remapped, pools restricted."""
    with open(src, "rb") as f:
        q = pickle.load(f)
    folder = {i: v["rel_scan_filepath"].split("/")[1] for i, v in q.items()}
    keep, seen = [], {}
    for i in sorted(q):
        seen[folder[i]] = seen.get(folder[i], 0) + 1
        if seen[folder[i]] <= n_per_folder:
            keep.append(i)
    new = {old: k for k, old in enumerate(keep)}
    out = {}
    for old in keep:
        v = dict(q[old])
        for key, ov in (("positives", "pos_overlap"), ("non_negatives", "semi_pos_overlap")):
            m = np.array([e in new for e in v[key]], bool)
            v[key] = np.array([new[e] for e in v[key][m]], dtype=np.int64)
            v[ov] = [v[ov][0][m]]
        v["id"] = new[old]
        out[new[old]] = v
    with open(dst, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return len(out)


# ------------------------------------------------------------ CPU selftest
class _StandIn:
    """CPU stand-in for the network (sptr / SparseTransformer kernels are
    CUDA-only).  Takes the official mini-batch dict {'coords', 'features'}
    and returns {'global': L2-normalised (B, 32)}.  Only for exercising the
    official data / loss / multistage code on CPU."""

    @staticmethod
    def make():
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = nn.Sequential(nn.Linear(7, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 32))

            def forward(self, batch):
                c = batch["coords"].float()
                b = c[:, 0].long()
                nb = int(b.max()) + 1
                xyz = c[:, 1:] / 100.0
                cnt = torch.zeros(nb).index_add_(0, b, torch.ones(len(b)))
                mean = torch.zeros(nb, 3).index_add_(0, b, xyz) / cnt[:, None]
                sq = torch.zeros(nb, 3).index_add_(0, b, xyz * xyz) / cnt[:, None]
                x = torch.cat([mean, (sq - mean * mean).clamp_min(0).sqrt(), cnt[:, None] / 1e4], 1)
                return {"global": torch.nn.functional.normalize(self.f(x), dim=1)}

        return Net()


def _pml_loss_check(tmp: Path) -> dict:
    """The official losses (make_losses + trainer.compute_losses) with the
    container's pytorch-metric-learning 2.9.0 and with the official pin 1.1.2
    (_external/helios_env/pml112), separate processes, same inputs."""
    code = r'''
import sys, json, numpy as np, torch
pml_dir, out = sys.argv[1], sys.argv[2]
sys.path[:0] = [sys.argv[3], sys.argv[3] + "/docker/thirdparty/SparseTransformer"]
if pml_dir:
    sys.path.insert(0, pml_dir)
import pytorch_metric_learning as pml
from misc.utils import TrainingParams
from models.losses.loss import make_losses
from training.trainer import compute_losses
params = TrainingParams(sys.argv[4], sys.argv[5])
g = torch.Generator().manual_seed(0)
n = 64
emb = torch.nn.functional.normalize(torch.randn(n, 32, generator=g), dim=1).double()
ov = torch.rand(n, n, generator=g).double()
ov = (ov + ov.T) / 2
ov.fill_diagonal_(1.0)
pos = ov > 0.75
pos.fill_diagonal_(False)
semi = (ov > 0.3) & (ov <= 0.75)
neg = ~pos & ~semi
o = torch.where(pos | semi, ov, torch.zeros_like(ov))
emb.requires_grad_(True)
try:
    loss, stats = compute_losses(emb, pos, neg, semi, o, make_losses(params),
                                 [params.truncated_weight, params.guided_p_sp_weight, params.guided_sp_n_weight], "cpu")
    loss.backward()
    np.save(out, emb.grad.numpy())
    print(json.dumps({"pml": pml.__version__, "ok": True, "loss": float(loss),
                      "parts": [s["loss"] for s in stats]}))
except Exception as e:
    print(json.dumps({"pml": pml.__version__, "ok": False, "error": f"{type(e).__name__}: {e}"}))
'''
    script = tmp / "pml_check.py"
    script.write_text(code)
    res = {}
    for tag, d in (("container", ""), ("official_pin", str(PML112))):
        r = subprocess.run([sys.executable, str(script), d, str(tmp / f"grad_{tag}.npy"), str(HELIOS),
                            str(write_config(tmp / "cfg.txt", tmp)), str(MODEL_CONFIG)],
                           capture_output=True, text=True, cwd=str(HELIOS),
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
        assert r.returncode == 0, r.stderr[-3000:]
        res[tag] = json.loads(r.stdout.strip().splitlines()[-1])
    return res


def _guard_check() -> dict:
    """guard_empty_miner_stats vs the official method (pml 1.1.2, CPU):
    non-empty batches -> identical loss, gradient and statistics; a batch with
    no valid anchor -> the official method raises, the guarded call returns
    the official loss_fn value for zero triplets (0) with NaN statistics."""
    import torch
    import torch.nn.functional as F
    import models.losses.loss as L
    cur = L.HardTripletMinerWithMasks._update_stats
    L.HardTripletMinerWithMasks._update_stats = getattr(cur, "_pdv2_orig", cur)   # official

    def make(n, p_pos, p_semi, seed):
        g = torch.Generator().manual_seed(seed)
        emb = F.normalize(torch.randn(n, 32, generator=g), dim=1).double()
        r = torch.rand(n, n, generator=g)
        r = (r + r.T) / 2
        pos = r < p_pos
        semi = (r >= p_pos) & (r < p_pos + p_semi)
        pos.fill_diagonal_(False)
        semi.fill_diagonal_(False)                 # diagonal negative, as in the official collate
        neg = ~pos & ~semi
        ov = torch.where(pos, 0.5 + 0.5 * r, torch.where(semi, 0.5 * r, torch.zeros_like(r))).double()
        return emb, pos, semi, neg, ov

    def run(emb, pos, semi, neg, ov):
        out = {}
        for rel, margin, (pm, nm) in (("ps", 0.02, (pos, semi)), ("sn", 2 * 0.19, (semi, neg))):
            e = emb.clone().requires_grad_(True)
            fn = L.BatchHardGuidedTripletLossWithMasks(margin=margin)
            # as in training: the loss object has already reduced a non-empty batch
            # (pml 1.1.2 registers reducer.triplets_past_filter on that first call)
            we, wp, wsm, wn, wo = make(32, 0.2, 0.3, 99)
            fn(we, *((wp, wsm) if rel == "ps" else (wsm, wn)), wo, rel)
            try:
                loss, st = fn(e, pm, nm, ov, rel)
            except RuntimeError as ex:
                out[rel] = f"RuntimeError: {str(ex).splitlines()[0]}"
                continue
            loss.backward()
            ref = None
            if st["num_triplets"] == 0:      # what the official loss_fn gives for zero triplets
                e2 = emb.clone().requires_grad_(True)
                ref = float(fn.loss_fn(e2, torch.arange(len(e2)), (torch.zeros(0, dtype=torch.long),) * 3,
                                       ov, rel))
            out[rel] = (float(loss), e.grad.clone(), st, ref)
        return out

    cases = [(n, pp, ps_, sd) for sd, (n, pp, ps_) in enumerate(
        [(16, 0.1, 0.1), (32, 0.05, 0.3), (64, 0.2, 0.05), (48, 0.02, 0.02), (24, 0.3, 0.3), (40, 0.1, 0.0)])]
    official = {c: run(*make(*c)) for c in cases}
    guard_empty_miner_stats()
    rec = {}
    n_equal = n_nonempty = n_empty_ok = n_empty = 0
    for c in cases:
        g = run(*make(*c))
        for rel in ("ps", "sn"):
            o, gr = official[c][rel], g[rel]
            if isinstance(o, str):                 # official raised: empty valid set
                n_empty += 1
                ok = (not isinstance(gr, str)) and gr[3] is not None and gr[0] == gr[3] == 0.0 \
                    and all(v != v for k, v in gr[2].items() if k.endswith("pair_dist")) \
                    and float(gr[1].abs().max()) == 0.0
                n_empty_ok += ok
                rec[f"{c} {rel}"] = {"official": o, "guarded_loss": gr[0],
                                     "official_loss_fn_zero_triplets": gr[3], "stats_nan_grad_zero": bool(ok)}
            else:
                n_nonempty += 1
                same = o[0] == gr[0] and torch.equal(o[1], gr[1]) and o[2] == gr[2]
                n_equal += same
                rec[f"{c} {rel}"] = {"identical_loss_grad_stats": bool(same), "loss": gr[0],
                                     "n_triplets": gr[2]["num_triplets"]}
    rec["empty_ok"] = f"{n_empty_ok}/{n_empty}"
    rec["non_empty_identical"] = f"{n_equal}/{n_nonempty}"
    return rec


def selftest(out_json: Path):
    """CPU only: (1) official training entry imports with the patched
    MinkowskiEngine and the official model is built (author weights load
    strictly; forward needs CUDA: sptr); (2) the official losses under the
    official pin pytorch-metric-learning 1.1.2 (and the failure under 2.9.0);
    (3) a small real pickle (official toolbox scans of KITTI 06 + Bridge01/04,
    official overlap labels) through the official make_dataloaders /
    collate_fn: masks, overlaps, quantised batches; (4) run_training (the
    official multistage step, aggregate_epoch_stats, best / periodic / final
    saves, resume) for 2 + 1 epochs on CPU with the stand-in network;
    (5) the official collate time of one 2048 batch (estimate)."""
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    official_path()
    rep = {}
    t0 = time.time()
    import training.train  # noqa: F401  (official CLI module, __main__ guarded)
    import training.trainer as T  # noqa: F401
    import MinkowskiEngine as ME
    from misc.utils import ModelParams
    from models.model_factory import model_factory
    mp = ModelParams(str(MODEL_CONFIG))
    m = model_factory(mp)
    sd = torch.load(HELIOS / "weights/HeLiOS.pth", map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    rep["empty_miner_guard"] = _guard_check()
    print(json.dumps(rep["empty_miner_guard"]), flush=True)
    rep["imports"] = {"seconds": round(time.time() - t0, 1), "MinkowskiEngine": ME.__version__,
                      "torch": torch.__version__, "model_params": int(sum(p.numel() for p in m.parameters())),
                      "author_weights_strict_load": True}
    with tempfile.TemporaryDirectory(dir=str(P.WORK)) as td:
        td = Path(td)
        rep["pml_loss_check"] = _pml_loss_check(td)
        print(json.dumps(rep["pml_loss_check"]), flush=True)
        # (3) small real dataset in the official layout
        import _helios_official_overlap as O
        from _pd_build_test_caches import _make_loader
        from scipy.spatial import cKDTree
        apply_transformation, load_pc = O.official_transformer()
        tb = P.Toolbox()
        data = td / "data"
        rel, H, G, vox, stamps = [], [], [], [], []
        for sensor, seq, n in (("helipr", "Bridge01", 40), ("helipr", "Bridge04", 40), ("kitti", "06", 48)):
            z = np.load(P.cache_path("train", sensor, seq))
            lp = P.lidar_poses(sensor, seq, z["poses"].astype(np.float64))
            keep = np.flatnonzero(P.official_keyframes(lp[:, :3, 3], P.TOOLBOX_SPACING_M))[:n]
            loader = _make_loader(sensor, seq)
            for k in keep:
                sid = int(z["scan_ids"][k])
                r = P.rel_scan_path("train", sensor, seq, sid)
                (data / r).parent.mkdir(parents=True, exist_ok=True)
                pts, _ = P.process_scan(tb, loader[sid]["points"], P.fill_rng(sensor, seq, sid))
                pts.astype("<f4").tofile(data / r)
                tr = apply_transformation(load_pc(str(data / r)), O.pose_to_transformation(lp[k]))
                vox.append(O.pcl_voxel_grid(tr.astype(np.float32)))
                rel.append(r)
                H.append(P.horizontal(sensor, lp[k][None, :3, 3])[0])
                G.append(P.frame_group(sensor, seq))
                stamps.append(sid)
        n = len(vox)
        H, G = np.asarray(H), np.asarray(G)
        trees = [cKDTree(v.astype(np.float64)) for v in vox]
        I, J, OV = [], [], []
        for i in range(n):
            for j in O._candidates(i, H, G, n):
                ov = O.text5(O.symmetric_overlap(vox[i], vox[j], trees[i], trees[j]))
                if ov > 0:
                    I.append(i), J.append(j), OV.append(ov)
        q = O.build_queries(n, np.asarray(I), np.asarray(J), np.asarray(OV), np.asarray(rel),
                            np.asarray(stamps), H)
        for name in ("training.pickle", "validation.pickle"):
            with open(data / name, "wb") as f:
                pickle.dump(q, f, protocol=pickle.HIGHEST_PROTOCOL)
        cfg = write_config(td / "cfg_small.txt", data, batch_size=32, batch_split_size=8,
                           val_batch_size=32, epochs=2, num_workers=2, save_freq=2)
        from misc.utils import TrainingParams
        from datasets.dataset_utils import make_dataloaders
        params = TrainingParams(str(cfg), str(MODEL_CONFIG))
        dl = make_dataloaders(params)
        batch, pos, neg, semi, ovm = next(iter(dl["train"]))
        rep["collate"] = {"n_minibatches": len(batch), "minibatch_keys": sorted(batch[0]),
                          "coords_dtype": str(batch[0]["coords"].dtype),
                          "mask_shape": list(pos.shape),
                          "diag_is_negative (official)": bool(neg.diagonal().all()),
                          "pos_semi_neg_disjoint": bool(not (pos & semi).any() and not (pos & neg).any()
                                                         and not (semi & neg).any()),
                          "overlap_range": [float(ovm.min()), float(ovm.max())]}
        rep["small_pickle"] = {"n": n, "pairs_overlap_gt0": len(OV),
                               "positives_median": float(np.median([len(v["positives"]) for v in q.values()]))}
        print(json.dumps(rep["collate"]), flush=True)
        # (4) the training loop on CPU with the stand-in network
        run = td / "run"
        torch.manual_seed(0)
        rec = run_training(params, run, td / "rec.json", resume=False, device="cpu",
                           model_override=_StandIn.make(), max_batches=2)
        files_a = sorted(p.name for p in run.glob("*.pth"))
        # resume: continue the 2-epoch run to 4 epochs from resume.pth (epoch 2)
        rec_path = td / "rec.json"
        params3 = TrainingParams(str(write_config(td / "cfg_small4.txt", data, batch_size=32,
                                                  batch_split_size=8, val_batch_size=32, epochs=4,
                                                  num_workers=2, save_freq=2)), str(MODEL_CONFIG))
        rec2 = run_training(params3, run, rec_path, resume=True, device="cpu",
                            model_override=_StandIn.make(), max_batches=2)
        rep["cpu_loop"] = {"epochs_first": [e["epoch"] for e in rec["epochs"]],
                           "epochs_after_resume": [e["epoch"] for e in rec2["epochs"]],
                           "files": files_a, "losses_finite": bool(all(np.isfinite(e["train_ap"])
                                                                       for e in rec2["epochs"])),
                           "train_ap": [round(e["train_ap"], 4) for e in rec2["epochs"]],
                           "val_ap": [round(e["val_ap"], 4) for e in rec2["epochs"]]}
        print(json.dumps(rep["cpu_loop"]), flush=True)
        # (5) collate cost of one official 2048 batch on a synthetic pickle
        rep["collate_2048"] = _collate_timing(td, q, data)
        print(json.dumps(rep["collate_2048"]), flush=True)
    eg = rep["empty_miner_guard"]
    a_, b_ = eg["empty_ok"].split("/")
    c_, d_ = eg["non_empty_identical"].split("/")
    ok = (a_ == b_ and int(b_) > 0 and c_ == d_ and int(d_) > 0
          and rep["pml_loss_check"]["official_pin"]["ok"] and rep["pml_loss_check"]["official_pin"]["pml"] == "1.1.2"
          and rep["collate"]["pos_semi_neg_disjoint"]
          and rep["cpu_loop"]["epochs_after_resume"] == [1, 2, 3, 4] and rep["cpu_loop"]["losses_finite"])
    rep["all_ok"] = bool(ok)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=1))
    print("TRAIN_SELFTEST_OK" if ok else "TRAIN_SELFTEST_FAILED", flush=True)
    if not ok:
        sys.exit(1)


def _collate_timing(td: Path, q_small: dict, data: Path) -> dict:
    """One 2048-element batch through the official collate_fn (masks +
    quantizer + batched_coordinates) with pools of a realistic size."""
    official_path()
    from datasets.dataset_utils import make_collate_fn
    from datasets.pnv_train import PNVTrainingDataset, TrainTransform
    from datasets.augmentation import TrainSetTransform
    from misc.utils import ModelParams
    rng = np.random.default_rng(0)
    n = 6000
    first = next(iter(q_small.values()))
    big = {}
    for i in range(n):
        nb = np.sort(rng.choice(n, 900, replace=False))
        nb = nb[nb != i]
        posm = rng.random(len(nb)) < 0.1
        big[i] = {"id": i, "timestamp": i, "rel_scan_filepath": first["rel_scan_filepath"],
                  "positives": nb[posm], "non_negatives": nb[~posm], "position": np.zeros(2),
                  "pos_overlap": [rng.random(posm.sum())], "semi_pos_overlap": [rng.random((~posm).sum())]}
    with open(td / "big.pickle", "wb") as f:
        pickle.dump(big, f, protocol=pickle.HIGHEST_PROTOCOL)
    ds = PNVTrainingDataset(str(data), "../big.pickle", transform=TrainTransform(1),
                            set_transform=TrainSetTransform(1))
    coll = make_collate_fn(ds, ModelParams(str(MODEL_CONFIG)).quantizer, 64)
    items = [ds[int(i)] for i in rng.choice(n, 2048)]
    t0 = time.time()
    batch = coll(items)
    return {"pool_size_per_query": 900, "batch": 2048, "collate_s": round(time.time() - t0, 1),
            "n_minibatches": len(batch[0])}


def check_batches(n_epochs: int, out_json: Path):
    """CPU only: replay the official BatchSamplers on data/{training,validation}.pickle
    with the run's seeding (random.seed(SEED) before make_dataloaders; nothing
    else in the main process draws from `random`), build the official
    collate masks per batch (in_sorted_array semantics, diagonal negative) and
    count the rows the two hard-triplet miners can use: ps = has a positive
    and a semi-positive, sn = has a semi-positive and a negative."""
    import torch
    official_path()
    from misc.utils import TrainingParams
    from datasets.dataset_utils import make_dataloaders
    with tempfile.TemporaryDirectory(dir=str(P.WORK)) as td:
        params = TrainingParams(str(write_config(Path(td) / "cfg.txt", P.DATA)), str(MODEL_CONFIG))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dls = make_dataloaders(params)
    rep = {"seed": SEED, "epochs": n_epochs, "train_n": len(dls["train"].dataset),
           "val_n": len(dls["val"].dataset), "per_epoch": [], "empty": []}
    for epoch in range(1, n_epochs + 1):
        row = {"epoch": epoch}
        for phase in ("train", "val"):
            ds, sampler = dls[phase].dataset, dls[phase].batch_sampler
            sizes, min_ps, min_sn = [], [], []
            for b, batch in enumerate(sampler):                     # official generate_batches
                lab = np.asarray(batch)
                pos = np.stack([np.isin(lab, ds.queries[l].positives) for l in lab])
                semi = np.stack([np.isin(lab, ds.queries[l].non_negatives) for l in lab])
                neg = ~pos & ~semi
                v_ps = int((pos.any(1) & semi.any(1)).sum())
                v_sn = int((semi.any(1) & neg.any(1)).sum())
                sizes.append(len(lab))
                min_ps.append(v_ps)
                min_sn.append(v_sn)
                if v_ps == 0 or v_sn == 0:
                    rep["empty"].append({"epoch": epoch, "phase": phase, "batch": b, "size": len(lab),
                                         "valid_ps_rows": v_ps, "valid_sn_rows": v_sn,
                                         "batches_in_phase": None})
            for e in rep["empty"]:
                if e["epoch"] == epoch and e["phase"] == phase:
                    e["batches_in_phase"] = len(sizes)
            row[phase] = {"batches": len(sizes), "sizes_min_last": [min(sizes), sizes[-1]],
                          "valid_ps_rows_min": min(min_ps), "valid_sn_rows_min": min(min_sn)}
        rep["per_epoch"].append(row)
        print(json.dumps(row), flush=True)
    rep["n_empty_batches"] = len(rep["empty"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=1))
    print(json.dumps({"empty": rep["empty"][:20], "n_empty_batches": rep["n_empty_batches"]}), flush=True)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", default="", help="run directory to continue")
    ap.add_argument("--smoke", type=int, default=0,
                    help="first N queries of every folder, 1 epoch, separate run/record")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-batches", type=int, default=0, metavar="EPOCHS",
                    help="CPU: replay the official samplers for EPOCHS epochs and count miner-empty batches")
    ap.add_argument("--out-json", type=Path, default=P.LOG_DIR / f"helios_selftest_train{SFX}.json")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.out_json)
        return
    if a.check_batches:
        check_batches(a.check_batches, P.LOG_DIR / f"helios_batch_check{SFX}.json")
        return
    official_path()
    import torch
    from misc.utils import TrainingParams
    assert torch.cuda.is_available(), "training needs CUDA (sptr kernels)"
    if a.smoke:
        sm = SMOKE
        sm.mkdir(parents=True, exist_ok=True)
        for role, name in (("train", "training.pickle"), ("val", "validation.pickle")):
            subset_pickle(P.DATA / name, sm / name, a.smoke)
        cfg = write_config(sm / "config_smoke.txt", P.DATA, train_file=f"../{sm.name}/training.pickle",
                           val_file=f"../{sm.name}/validation.pickle", epochs=1)
        params = TrainingParams(str(cfg), str(MODEL_CONFIG))
        params.print()
        run_training(params, sm / "run", sm / "train_record.json", resume=False, device="cuda")
        return
    cfg = write_config(CONFIG, P.DATA)
    params = TrainingParams(str(cfg), str(MODEL_CONFIG))
    params.print()
    if a.resume:
        run_dir = Path(a.resume)
    else:
        run_dir = RUNS / datetime.now().strftime("%b%d_%H-%M-%S")
    run_training(params, run_dir, RECORD, resume=bool(a.resume), device="cuda")


if __name__ == "__main__":
    main()
