#!/usr/bin/env python3
"""DeepRING on split v2 -- PAPER-BASED RE-IMPLEMENTATION, driver (2026-10-11).

No public official DeepRING code exists (scripts/_deepring_reimpl_model.py
docstring lists what was checked).  This driver chains:

  bev        per keyframe: our loader points -> RING repo dataset class
             read_pc / normalize_pc with the shared DATASET_TABLE row
             (scripts/_ring_official_lib.official_points, the same mapping as
             the official RING / RING++ pipeline) -> RING repo
             utils.core.generate_bev_occ (CUDA voxelocc, 120 x 120 occupancy)
             -> bit-packed shards _external/deepring_reimpl_pdv2/bev/{role}/
             {sensor}_{seq}/.  role train / val = split-v2 GT keyframes
             (data/preprocessed_v2), test_odom = KISS-ICP keyframes
             (data/preprocessed_v2_odom).  For test_odom the identical BEVs of
             the RING pipeline (_external/ring_official_cache/odom, same calls,
             same table row, same scan_ids) are copied when they exist.
  train      paper recipe (Sec. III-D): 20 epochs x 60 episodes, Adam lr 1e-3,
             weight decay 1e-4, MultiStepLR gamma 0.1 at epochs 5 and 12;
             episodes of 24 classes x (1 shot + 6 queries) inside one training
             sequence; sinogram = RING repo utils.core.generate_RING on the
             BEV (GPU); M_f = |row DFT(UNet(sinogram))|; eq. 8 correlation as
             RING's own circorr2 score (scripts/_deepring_reimpl_model.py);
             eq. 9 W C + b; cross entropy.  Seed = env SEED (default 1;
             other seeds: runs/s<SEED>, results/pdv2_deepring_reimpl_s<SEED>).
             Every epoch: split-v2 VAL
             intra-sequence 5 m R@1 with a 30-keyframe gap (past and future
             candidates, compute_recall_multi_k semantics, ranking by the
             DeepRING logit W C + b over all frames), families weighted
             equally (scripts/_pdv2_c3._family_pooled).  Selected = highest
             family-macro R@1, ties -> earlier epoch.  The paper states no
             selection rule; the last epoch (20) is recorded as well.
  dump       46 split-v2 test sequences: per query t >= 30, exhaustive eq. 8
             (RING circorr2 score) over every keyframe j <= t - 30, top-1 = argmax W C + b, written
             by scripts/_causal_ring_pr.dump_variant (unchanged, schema
             pr-dump-v1): score_corr = (W C + b) of the top-1 minus 1 (the
             shared writer's constant offset; PR metrics are shift-invariant).
  aggregate  scripts/_pdv2_c3.cmd_aggregate_baselines (unchanged), column
             score_corr -> results/pdv2_deepring_reimpl/odom/pr_summary_baselines.json
  check      CPU-only checks -> results/pdv2_deepring_reimpl/check_cpu.json
  gpu-check  GPU (seconds to minutes): official GPU kernels vs CPU paths,
             one full 168-scan episode (time, peak memory)

Run inside docker ring_baseline from the repo root with `ulimit -s 81920`
(RING README: voxelocc keeps per-point arrays on the stack); see
scripts/_pdv2_deepring_reimpl_run.sh.
"""
from __future__ import annotations

import os
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"   # NGC default 1 forces TF32 matmul; the eq. 8 scorer stays fp32

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import _ring_official_lib as rol  # noqa: E402  (before anything importing 'utils')
from _pdv2_pilot import TRAIN, VAL, TEST  # noqa: E402
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402

V2 = REPO / "data/preprocessed_v2"
ODOM = REPO / "data/preprocessed_v2_odom"
ROLES = {"train": (TRAIN, V2, "", "poses"),
         "val": (VAL, V2, "val_", "poses"),
         "test_odom": (TEST, ODOM, "test_", "poses_gt")}
WORK = REPO / "_external/deepring_reimpl_pdv2"
BEV_ROOT = WORK / "bev"


def _seed_from_env() -> int:
    """Training seed from the environment variable SEED (default 1)."""
    raw = os.environ.get("SEED") or "1"
    assert raw.isdigit() and raw[0] != "0", f"SEED must be a positive integer, got {raw!r}"
    return int(raw)


SEED = _seed_from_env()
# training-dependent locations: unchanged for seed 1, own dirs for other seeds;
# BEV caches (BEV_ROOT) are shared by all seeds
RUN_DIR = WORK / f"runs/s{SEED}"
RESULTS = REPO / ("results/pdv2_deepring_reimpl" if SEED == 1
                  else f"results/pdv2_deepring_reimpl_s{SEED}")
SELECTION = RESULTS / "train_selection.json"
METHOD = "deepring"
SHARD = 1000
EMBED_BATCH = 256
VAL_RADIUS, VAL_SKIP = 5.0, 30
NOTE = ("DeepRING (Lu et al., ICRA 2023) paper-based re-implementation: no public "
        "official code exists; RING repo preprocessing/radon, RINGSharp UNet and RING "
        "circorr2 score as eq. 8 (unchanged) + paper eqs. 7 and 9 and Sec. III-D recipe")


# ------------------------------------------------------------------ helpers
def _split(seq, role):
    s, q = seq.split("/")
    assert (s, q) in ROLES[role][0], f"{seq} is not a split-v2 {role} sequence"
    return s, q


def cache_path(role, sensor, seq) -> Path:
    _, cdir, mid, _ = ROLES[role]
    hits = sorted(glob.glob(str(cdir / f"cache_????????_{sensor}_{mid}{seq}.npz")))
    assert len(hits) == 1, (role, sensor, seq, hits)
    return Path(hits[0])


def load_cache(role, sensor, seq):
    """scan_ids, (n, 4, 4) poses (GT; poses_gt for test_odom)."""
    with np.load(cache_path(role, sensor, seq)) as z:
        return z["scan_ids"].astype(np.int64), z[ROLES[role][3]].astype(np.float64)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


# ------------------------------------------------------------------ BEV stage
def occupancy_bev(core, sensor, pts, backend):
    """One keyframe -> (120, 120) uint8 occupancy, n points after the official
    point processing.  backend 'gpu' = RING repo generate_bev_occ (production);
    'cpu' = rol.occupancy_numpy, the CPU transcription of the same kernel
    (checks and CPU dry runs only)."""
    _, pcn = rol.official_points(sensor, pts)
    if len(pcn) == 0:
        return np.zeros((120, 120), np.uint8), 0
    if backend == "gpu":
        import torch
        with torch.no_grad():
            occ = core.generate_bev_occ(pcn).cpu().numpy().reshape(120, 120)
        rol.check_bev(occ, pcn)        # silent-CUDA-failure guard (exact CPU transcription)
    else:
        occ = rol.occupancy_numpy(pcn)
    assert np.isin(occ, (0, 1)).all(), "occupancy BEV is not binary"
    return occ.astype(np.uint8), len(pcn)


def bev_dir(root, role, sensor, seq):
    return Path(root) / role / f"{sensor}_{seq}"


def _reuse_ring_cache(sensor, q, scan_ids, d):
    """Copy the RING pipeline's test-odom occupancy BEVs (identical calls)."""
    rd = rol.CACHE_ROOT / "odom" / f"{sensor}_{q}"
    if not (rd / "done.json").exists():
        return None
    info = json.loads((rd / "done.json").read_text())
    if (info.get("scan_ids_sha256") != sha_arr(scan_ids)
            or info.get("table_row") != rol.DATASET_TABLE[sensor]
            or info.get("ring_commit") != rol.RING_COMMIT):
        print(f"  RING cache {rd} does not match (scan_ids / table row / commit); building", flush=True)
        return None
    ids, occ = [], []
    for f in sorted(rd.glob("shard_*.npz")):
        with np.load(f) as z:
            ids.append(z["scan_ids"])
            occ.append(z["bev_ring"].reshape(-1, 120, 120))
    ids, occ = np.concatenate(ids), np.concatenate(occ)
    assert np.array_equal(ids, scan_ids) and np.isin(occ, (0, 1)).all()
    for c0 in range(0, len(ids), SHARD):
        c1 = min(c0 + SHARD, len(ids))
        np.savez_compressed(d / f"shard_{c0:06d}.npz", scan_ids=ids[c0:c1],
                            bev_packed=np.packbits(occ[c0:c1].astype(np.uint8), axis=-1),
                            n_points=np.full(c1 - c0, -1, np.int64))
    return {"source": f"copied from {rd.relative_to(REPO)} (RING official pipeline, "
                      "same official_points + generate_bev_occ)",
            "ring_done": info}


def cmd_bev(role, seq, backend="gpu", root=BEV_ROOT, limit=0, reuse_ring=True):
    sensor, q = _split(seq, role)
    cp = cache_path(role, sensor, q)
    scan_ids, _ = load_cache(role, sensor, q)
    if limit:
        scan_ids = scan_ids[:limit]
    d = bev_dir(root, role, sensor, q)
    done = d / "done.json"
    if done.exists() and json.loads(done.read_text())["scan_ids_sha256"] == sha_arr(scan_ids):
        print(f"SKIP {role} {seq}: BEV cache complete", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    row = rol.DATASET_TABLE[sensor]
    meta = {"sensor": sensor, "sequence": q, "role": role,
            "cache": str(cp.relative_to(REPO)), "cache_sha256": sha256_file(cp),
            "n_frames": int(len(scan_ids)), "limit": limit,
            "scan_ids_sha256": sha_arr(scan_ids), "table_row": row,
            "ring_commit": rol.RING_COMMIT, "backend": backend,
            "generator": "_ring_official_lib.official_points -> RING utils.core.generate_bev_occ"
                         if backend == "gpu" else
                         "_ring_official_lib.official_points -> occupancy_numpy (CPU transcription; CHECK ONLY)"}
    if role == "test_odom" and backend == "gpu" and reuse_ring and not limit:
        got = _reuse_ring_cache(sensor, q, scan_ids, d)
        if got:
            meta.update(got)
            _write_json(done, meta)
            print(f"BEV_DONE {role} {seq}: {len(scan_ids)} keyframes ({got['source']})", flush=True)
            return
    core = rol.import_ring_core() if backend == "gpu" else None
    loader = rol.make_loader(sensor, q)
    n, t_all, n_built, n_empty = len(scan_ids), 0.0, 0, 0
    for c0 in range(0, n, SHARD):
        c1 = min(c0 + SHARD, n)
        shard = d / f"shard_{c0:06d}.npz"
        if shard.exists():
            with np.load(shard) as z:
                if np.array_equal(z["scan_ids"], scan_ids[c0:c1]):
                    continue
        occ = np.zeros((c1 - c0, 120, 120), np.uint8)
        npts = np.zeros(c1 - c0, np.int64)
        t0 = time.time()
        for k in range(c1 - c0):
            occ[k], npts[k] = occupancy_bev(core, sensor, loader[int(scan_ids[c0 + k])]["points"], backend)
        tmp = d / f".tmp_shard_{c0:06d}.npz"
        np.savez_compressed(tmp, scan_ids=scan_ids[c0:c1],
                            bev_packed=np.packbits(occ, axis=-1), n_points=npts)
        os.replace(tmp, shard)
        dt = time.time() - t0
        t_all, n_built, n_empty = t_all + dt, n_built + (c1 - c0), n_empty + int((npts == 0).sum())
        print(f"  [{role} {seq}] BEV {c1}/{n} ({dt / (c1 - c0):.3f} s/kf)", flush=True)
    meta.update({"source": "built", "s_per_kf_this_run": round(t_all / n_built, 4) if n_built else None,
                 "n_frames_no_points_this_run": n_empty})
    _write_json(done, meta)
    print(f"BEV_DONE {role} {seq}: {n} keyframes", flush=True)


def load_bev(role, sensor, q, root=BEV_ROOT, limit=0):
    """-> packed (n, 120, 15) uint8, done.json; checks keyframes and table row."""
    scan_ids, _ = load_cache(role, sensor, q)
    if limit:
        scan_ids = scan_ids[:limit]
    d = bev_dir(root, role, sensor, q)
    info = json.loads((d / "done.json").read_text())
    assert info["scan_ids_sha256"] == sha_arr(scan_ids), f"{d}: keyframes changed"
    assert info["table_row"] == rol.DATASET_TABLE[sensor], \
        f"{d}: DATASET_TABLE row changed since the BEVs were built"
    ids, packed = [], []
    for c0 in range(0, len(scan_ids), SHARD):
        with np.load(d / f"shard_{c0:06d}.npz") as z:
            ids.append(z["scan_ids"])
            packed.append(z["bev_packed"])
    assert np.array_equal(np.concatenate(ids), scan_ids)
    packed = np.concatenate(packed)
    for c0 in range(0, len(packed), 2000):     # refuse rows from a silently failed kernel
        bad = rol.degenerate_rows(np.unpackbits(packed[c0:c0 + 2000], axis=-1, count=120))
        assert not len(bad), f"{d}: degenerate occupancy row {c0 + int(bad[0])}; rebuild this BEV cache"
    return packed, info


def unpack(packed_rows):
    import torch
    occ = np.unpackbits(packed_rows, axis=-1, count=120)        # (m, 120, 120)
    return torch.from_numpy(occ.astype(np.float32)).unsqueeze(1)


def cmd_jobs(roles):
    for role in roles:
        for s, q in ROLES[role][0]:
            print(role, f"{s}/{q}")


def cmd_status(root=BEV_ROOT):
    missing = [f"{role} {s}/{q}" for role in ROLES for s, q in ROLES[role][0]
               if not (bev_dir(root, role, s, q) / "done.json").exists()]
    n = sum(len(ROLES[r][0]) for r in ROLES)
    print(f"{n - len(missing)}/{n} BEV caches complete")
    for m in missing:
        print("  missing", m)
    print("BEV_COMPLETE" if not missing else "BEV_INCOMPLETE")


# --------------------------------------------------------------- model glue
class Sino:
    """Sinogram of occupancy BEVs (B, 1, 120, 120) with the RING repo:
    'gpu' = utils.core.generate_RING (torch-radon ParallelBeam, production);
    'cpu' = utils.core.generate_RING_cpu (skimage radon; CPU dry runs only)."""

    def __init__(self, backend):
        self.backend = backend
        self.core = rol.import_ring_core()

    def __call__(self, bev):
        import torch
        with torch.no_grad():
            if self.backend == "gpu":
                s, _ = self.core.generate_RING(bev)
                return s.reshape(bev.shape)
            return torch.stack([self.core.generate_RING_cpu(b)[0] for b in bev.cpu()]).to(bev.device)


def embed_into(model, packed, sino, device, bank):
    """M_f of every packed BEV, in chunks, into a SpecBank."""
    import torch
    bs = EMBED_BATCH if device == "cuda" else 16   # CPU: keep peak RAM < 1 GB
    model.eval()
    with torch.no_grad():
        for i0 in range(0, len(packed), bs):
            bank.put(i0, model.embed(sino(unpack(packed[i0:i0 + bs]).to(device))))


def set_seeds(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scorer_fp32_guard(device):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda":
        assert not torch.backends.cuda.matmul.allow_tf32, \
            "TF32 cuBLAS is forced (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE); the eq. 8 scorer must run in fp32"


# ------------------------------------------------------------------ validation
def validate(model, val_sets, sino, device, block=32):
    """split-v2 VAL: intra-sequence 5 m R@1, 30-frame gap, past + future
    candidates (compute_recall_multi_k semantics), ranking by W C + b."""
    import torch
    from baselines.eval_utils import _find_revisit_queries, _score_recalls_from_ranked
    from _deepring_reimpl_model import SpecBank
    W, b = model.wb()
    per = {}
    for name, packed, poses in val_sets:
        n = len(packed)
        bank = SpecBank(n, device)
        embed_into(model, packed, sino, device, bank)
        positions = poses[:, :3, 3].astype(np.float64)
        queries = _find_revisit_queries(positions, VAL_RADIUS, VAL_SKIP)
        if not queries:
            per[name] = {"r1": 0.0, "n_queries": 0, "n_frames": n}
            continue
        qidx = np.array([q for q, _ in queries])
        cols = torch.arange(n, device=device)
        ranked = []
        with torch.no_grad():
            for s0 in range(0, len(qidx), block):
                qb = qidx[s0:s0 + block]
                S = bank.corr(qb, n).double() * W + b
                qt = torch.as_tensor(qb, device=device).unsqueeze(1)
                S[(cols.unsqueeze(0) - qt).abs() <= VAL_SKIP] = -float("inf")
                ranked += [np.array([j]) for j in S.argmax(dim=1).cpu().numpy()]
        r1 = _score_recalls_from_ranked(queries, ranked, positions, [1], VAL_RADIUS)[1]
        per[name] = {"r1": float(r1), "n_queries": len(queries), "n_frames": n}
        print(f"    val {name}: R@1 {r1:.4f} ({len(queries)} queries)", flush=True)
        del bank
    return per


# ------------------------------------------------------------------ training
def train(args):
    import torch
    from _deepring_reimpl_model import RECIPE, DeepRING, EpisodeSampler, episode_loss
    from _pdv2_c3 import _family_pooled

    device = args.device
    set_seeds(SEED)
    scorer_fp32_guard(device)
    sino = Sino(args.sino_backend)
    run_dir, sel_path = Path(args.run_dir), Path(args.selection_out)
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs, n_ep = args.epochs or RECIPE["epochs"], args.episodes or RECIPE["episodes_per_epoch"]
    ways, nq = args.ways or RECIPE["ways"], args.queries or RECIPE["queries_per_class"]

    t0 = time.time()
    tr_names, tr_packed, tr_pos = [], [], []
    for s, q in TRAIN:
        if args.train_seqs and f"{s}/{q}" not in args.train_seqs:
            continue
        packed, _ = load_bev("train", s, q, args.bev_root, args.max_frames)
        _, poses = load_cache("train", s, q)
        tr_names.append(f"{s}/{q}")
        tr_packed.append(packed)
        tr_pos.append(poses[:len(packed), :3, 3])
    val_sets = []
    for s, q in VAL:
        if args.val_seqs and f"{s}/{q}" not in args.val_seqs:
            continue
        packed, _ = load_bev("val", s, q, args.bev_root, args.val_max_frames)
        _, poses = load_cache("val", s, q)
        val_sets.append((f"{s}/{q}", packed, poses[:len(packed)]))
    sampler = EpisodeSampler(tr_pos, np.random.default_rng(SEED), ways=ways, queries=nq,
                             same_m=RECIPE["same_class_m"], diff_m=RECIPE["diff_class_m"])
    print(f"train {sum(map(len, tr_packed))} keyframes in {len(tr_names)} sequences, "
          f"val {sum(len(v[1]) for v in val_sets)} in {len(val_sets)} ({time.time() - t0:.0f} s)",
          flush=True)

    model = DeepRING().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=RECIPE["milestones"],
                                                 gamma=RECIPE["gamma"])
    record = {"_meta": {
        "note": NOTE, "driver": "scripts/_deepring_reimpl.py", "seed": SEED,
        "recipe": RECIPE, "overrides": {"epochs": epochs, "episodes": n_ep, "ways": ways,
                                        "queries": nq, "max_frames": args.max_frames,
                                        "val_max_frames": args.val_max_frames},
        "device": device, "sino_backend": args.sino_backend, "run_dir": str(run_dir),
        "train_sequences": {n: len(p) for n, p in zip(tr_names, tr_packed)},
        "val_sequences": [v[0] for v in val_sets],
        "selection_rule": "per epoch: split-v2 VAL intra-sequence 5 m R@1, 30-keyframe gap "
                          "(compute_recall_multi_k semantics, ranking by W C + b over all "
                          "frames outside the gap); sensor families (kitti+kitti360, helipr, "
                          "mulran) weighted equally, queries pooled inside a family; selected "
                          "= highest family-macro R@1, ties -> earlier epoch. The paper states "
                          "no selection rule; 'last' records the final epoch"},
        "epochs": []}
    start = 0
    ckpts = sorted(run_dir.glob("epoch_*.pth"))
    if ckpts:
        ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        sampler.rng.bit_generator.state = ck["sampler_rng"]
        random.setstate(ck["rng"]["random"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        start = ck["epoch"]
        record = json.loads(sel_path.read_text())
        record["epochs"] = [e for e in record["epochs"] if e["epoch"] <= start]
        record.setdefault("resumed_after_epoch", []).append(start)
        record["_meta"]["overrides"]["epochs"] = epochs
        print(f"resumed after epoch {start} from {ckpts[-1]}", flush=True)

    for epoch in range(start + 1, epochs + 1):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        t0, losses, accs = time.time(), [], []
        ep_seq = {n: 0 for n in tr_names}
        for _ in range(n_ep):
            s, shots, qs = sampler.sample()
            ep_seq[tr_names[s]] += 1
            idx = np.concatenate([shots, qs.reshape(-1)])
            M = model.embed(sino(unpack(tr_packed[s][idx]).to(device)))
            loss, acc = episode_loss(model, M, ways, nq)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
            accs.append(acc.item())
        lr = opt.param_groups[0]["lr"]
        sched.step()
        t_train = time.time() - t0
        t0 = time.time()
        per = validate(model, val_sets, sino, device)
        fam = _family_pooled(per) if any(v["n_queries"] for v in per.values()) else 0.0
        t_val = time.time() - t0
        W, b = model.wb()
        ck_path = run_dir / f"epoch_{epoch:02d}.pth"
        rng = {"random": random.getstate(), "numpy": np.random.get_state(),
               "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None}
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "sampler_rng": sampler.rng.bit_generator.state,
                    "rng": rng, "family_macro_r1": fam}, ck_path)
        record["epochs"].append({
            "epoch": epoch, "lr": lr, "loss_mean": float(np.mean(losses)),
            "episode_acc_mean": float(np.mean(accs)), "W": W, "b": b,
            "episodes_per_sequence": ep_seq,
            "family_macro_r1": fam, "per_seq": per, "checkpoint": str(ck_path),
            "train_s": round(t_train, 1), "val_s": round(t_val, 1),
            "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            if device == "cuda" else None})
        best = max(record["epochs"], key=lambda e: (e["family_macro_r1"], -e["epoch"]))
        record["best"] = {k: best[k] for k in ("epoch", "family_macro_r1", "checkpoint")}
        record["last"] = {k: record["epochs"][-1][k] for k in ("epoch", "family_macro_r1", "checkpoint")}
        _write_json(sel_path, record)
        print(f"=== epoch {epoch}: loss {np.mean(losses):.4f} acc {np.mean(accs):.3f} "
              f"W {W:.3g} b {b:.3g} | val family R@1 {fam:.4f} | train {t_train:.0f} s "
              f"val {t_val:.0f} s | best epoch {best['epoch']}", flush=True)

    record["best"]["sha256"] = sha256_file(Path(record["best"]["checkpoint"]))
    record["last"]["sha256"] = sha256_file(Path(record["last"]["checkpoint"]))
    record["finished"] = True
    _write_json(sel_path, record)
    print("DEEPRING_TRAIN_DONE", flush=True)
    return record


# ------------------------------------------------------------------ test ledgers
class BlockScorer:
    """scorer(t, n_prefix) for _causal_ring_pr.dump_variant: W C + b of query
    t against rows [0, t - 30].  Queries are evaluated in blocks (one bank
    pass per block); every row returned for t lies in t's own prefix."""

    def __init__(self, bank, W, b, n, block):
        self.bank, self.W, self.b, self.n, self.block = bank, W, b, n, block
        self.cache = {}

    def __call__(self, t, n_prefix):
        if t not in self.cache:
            self.cache.clear()
            ts = list(range(t, min(t + self.block, self.n)))
            C = self.bank.corr(ts, ts[-1] - MIN_INDEX_GAP + 1).double().cpu().numpy()
            for k, tt in enumerate(ts):
                self.cache[tt] = C[k, :tt - MIN_INDEX_GAP + 1]
        C = self.cache.pop(t)
        assert len(C) == n_prefix == t - MIN_INDEX_GAP + 1
        return self.W * C + self.b


def load_model(checkpoint, device):
    import torch
    from _deepring_reimpl_model import DeepRING
    model = DeepRING().to(device)
    if checkpoint is not None:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def xcheck_scores(bank, t, block):
    """blocked bank scores == single-query bank scores == the RING repo's
    batch_circorr (corr_max_direct) on up to 256 prefix rows of the same raw
    M_f (kept by the bank for these rows)."""
    import torch
    from _deepring_reimpl_model import corr_max_direct
    npf = t - MIN_INDEX_GAP + 1
    ts = list(range(t, min(t + block, bank.n)))
    with torch.no_grad():
        blk = bank.corr(ts, ts[-1] - MIN_INDEX_GAP + 1)[0, :npf].double().cpu()
        one = bank.corr([t], npf)[0].double().cpu()
        m = min(npf, 256)
        M = bank.rows(np.concatenate([np.arange(m), [t]])).double().cpu()
        ref = corr_max_direct(M[-1:], M[:m])[0]
    scale = float(ref.abs().max())
    return {"t": int(t), "n_prefix": int(npf),
            "rel_diff_block_vs_single": float((blk - one).abs().max()) / max(float(one.abs().max()), 1e-30),
            "rel_diff_single_vs_direct": float((one[:m] - ref).abs().max()) / max(scale, 1e-30),
            "same_argmax_block_single": int(blk.argmax()) == int(one.argmax()),
            "same_argmax_direct_first_rows": int(one[:m].argmax()) == int(ref.argmax())}


def dump_one(sensor, q, checkpoint, device, sino_backend, block, out_path,
             bev_root=BEV_ROOT, limit=0, extra_meta=None):
    import torch
    from _causal_ring_pr import dump_variant
    from _deepring_reimpl_model import SpecBank, UNET_UPSTREAM
    scorer_fp32_guard(device)
    cp = cache_path("test_odom", sensor, q)
    _, poses_gt = load_cache("test_odom", sensor, q)
    packed, info = load_bev("test_odom", sensor, q, bev_root, limit)
    poses_gt = poses_gt[:len(packed)]
    n = len(packed)
    model = load_model(checkpoint, device)
    W, b = model.wb()
    sino = Sino(sino_backend)
    t0 = time.time()
    bank = SpecBank(n, device, keep=set(range(256)) | {n // 4, n // 2, n - 1})
    embed_into(model, packed, sino, device, bank)
    if device == "cuda":
        torch.cuda.synchronize()
    t_enc = time.time() - t0
    positions = poses_gt[:, :3, 3]
    min_past = rol.min_past_dists(positions, MIN_INDEX_GAP)
    xc = []
    for t in sorted({n // 4, n // 2, n - 1}):
        if t >= MIN_INDEX_GAP:
            r = xcheck_scores(bank, t, block)
            xc.append(r)
            assert r["rel_diff_block_vs_single"] < 1e-4 and r["rel_diff_single_vs_direct"] < 1e-4, r
    t0 = time.time()
    arrays, n_corr = dump_variant(None, positions, min_past, 0, device, None,
                                  tag=f"[{sensor}/{q}] deepring",
                                  scorer=BlockScorer(bank, W, b, n, block), n_frames=n)
    t_led = time.time() - t0
    nq = len(arrays["query_idx"])
    meta = {
        "schema": "pr-dump-v1", "method": METHOD, "variant": NOTE,
        "sensor": sensor, "sequence": q, "condition": "odom",
        "min_index_gap": MIN_INDEX_GAP, "n_frames": int(n), "n_queries": int(nq),
        "primary_score": "score_corr",
        "scores": {"score_corr": "DeepRING classifier logit W*C + b (eq. 9 before softmax; "
                                 "C = eq. 8 as RING's circorr2 score of M_f) of the exhaustive "
                                 "prefix top-1, MINUS 1 (constant offset of the shared writer "
                                 "_causal_ring_pr.dump_variant); similarity, higher = accept"},
        "W": W, "b": b,
        "pipeline": "official_points -> generate_bev_occ -> generate_RING sinogram (RING repo "
                    f"@ {rol.RING_COMMIT}) -> UNet ({UNET_UPSTREAM}, is_circular=True) -> row "
                    "|DFT| -> eq. 8 (RING circorr2 score) vs every keyframe j <= t - 30 -> argmax W C + b",
        "dataset_table_row": rol.DATASET_TABLE[sensor],
        "scorer": f"SpecBank: per-frame InstanceNorm + fft2(dim=-2) bank (RING circorr2 hoist), query blocks of {block}",
        "scorer_crosscheck": xc,
        "artifacts": {"checkpoint": str(checkpoint) if checkpoint else None,
                      "checkpoint_sha256": sha256_file(Path(checkpoint)) if checkpoint else None,
                      "selection": str(SELECTION.relative_to(REPO))},
        "caches": {"operating_cache": str(cp.relative_to(REPO)), "bev": info,
                   "positions": "poses_gt (scoring only)"},
        "timing": {"encode_s": round(t_enc, 1), "encode_ms_per_kf": round(1e3 * t_enc / max(n, 1), 2),
                   "ledger_s": round(t_led, 1), "ms_per_query": round(1e3 * t_led / max(nq, 1), 2),
                   "n_correlations": int(n_corr)},
    }
    meta.update(extra_meta or {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta, default=str), **arrays)
    print(f"DUMP_DONE {out_path.name}: {nq} queries, encode {t_enc:.0f} s, ledger {t_led:.0f} s, "
          f"xcheck {[(r['t'], round(r['rel_diff_single_vs_direct'], 8)) for r in xc]}", flush=True)
    return out_path


def default_checkpoint():
    sel = json.loads(SELECTION.read_text())
    assert sel.get("finished"), f"{SELECTION}: training not finished"
    ck = Path(sel["best"]["checkpoint"])
    assert sha256_file(ck) == sel["best"]["sha256"], "checkpoint changed after selection"
    return ck


def cmd_dump(seqs, checkpoint, block, worker, overwrite):
    checkpoint = Path(checkpoint) if checkpoint else default_checkpoint()
    todo = seqs or [f"{s}/{q}" for s, q in TEST]
    out = RESULTS / "odom"
    if worker:
        assert len(todo) == 1
        sensor, q = _split(todo[0], "test_odom")
        dump_one(sensor, q, checkpoint, "cuda", "gpu", block, out / f"dump_{METHOD}_{sensor}_{q}.npz")
        return
    for key in todo:          # one sequence per subprocess (memory discipline)
        sensor, q = _split(key, "test_odom")
        if (out / f"dump_{METHOD}_{sensor}_{q}.npz").exists() and not overwrite:
            print(f"[skip] {key}", flush=True)
            continue
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "dump", "--worker",
                             "--seqs", key, "--checkpoint", str(checkpoint),
                             "--block", str(block)]).returncode
        if rc:
            print(f"WORKER_FAILED {key} rc={rc}", flush=True)
            sys.exit(rc)
    print("DEEPRING_DUMP_DONE", flush=True)


def cmd_aggregate():
    import _pdv2_c3 as c3
    c3.OUT_ROOT = RESULTS
    c3.BASELINES = {METHOD: "score_corr"}
    c3.cmd_aggregate_baselines("odom")
    p = RESULTS / "odom" / "pr_summary_baselines.json"
    s = json.loads(p.read_text())
    s["_meta"].pop("seqot", None)   # hard-coded note of the reused aggregator
    s["_meta"][METHOD] = NOTE + "; checkpoint by v2-val family-equal R@1, " + \
        str(SELECTION.relative_to(REPO))
    p.write_text(json.dumps(s, indent=2))


# ------------------------------------------------------------------ checks
CHECK_SCANS = [("train", "kitti", "05"), ("train", "helipr", "Bridge01"),
               ("val", "kitti360", "0004"), ("val", "mulran", "Sejong01"),
               ("test_odom", "nclt", "2012-01-08"), ("test_odom", "mcd", "kth_day_06")]


def _check_unet(rep):
    """Imported UNet unchanged; how exact the rotation-equivariance /
    translation-invariance of the authors' circular UNet is."""
    import torch
    from _deepring_reimpl_model import DeepRING, corr_max
    torch.manual_seed(0)
    model = DeepRING().eval()
    x = torch.relu(torch.randn(2, 1, 120, 120)) * 5
    with torch.no_grad():
        M = model.embed(x)
        out = {"n_params": int(sum(p.numel() for p in model.parameters())),
               "M_shape": list(M.shape)}
        for a in (1, 8, 37):
            Mr = model.embed(torch.roll(x, a, dims=-2))
            out[f"theta_roll_{a}_rel_err_vs_rolled_output"] = float(
                (Mr - torch.roll(M, a, dims=-2)).abs().max() / M.abs().max())
            out[f"theta_roll_{a}_corr_ratio"] = float(corr_max(M[:1], Mr[:1]) / corr_max(M[:1], M[:1]))
        for d in (1, 8):
            Mt = model.embed(torch.roll(x, d, dims=-1))
            out[f"tau_roll_{d}_rel_err_M"] = float((Mt - M).abs().max() / M.abs().max())
    rep["unet"] = out
    print("  unet", out, flush=True)


def _check_scorer(rep):
    """corr_max == eq. 8 written out == SpecBank (single and blocked);
    prefix rows only; first-max argmax."""
    import torch
    from _deepring_reimpl_model import SpecBank, corr_max, corr_max_direct
    g = torch.Generator().manual_seed(0)
    M = torch.rand(70, 1, 120, 120, generator=g, dtype=torch.float64).float() * 3
    M[60] = torch.roll(M[7], 17, dims=-2)            # rotated revisit
    ref = corr_max_direct(M.double(), M.double())
    a = corr_max(M, M).double()
    bank = SpecBank(70, "cpu")
    bank.put(0, M)
    blk = bank.corr(list(range(40, 70)), 70).double()
    one = torch.stack([bank.corr([t], 70)[0] for t in range(40, 70)]).double()
    tr = BlockScorer(bank, 1.0, 0.0, 70, 8)
    pref = [tr(t, t - MIN_INDEX_GAP + 1) for t in range(MIN_INDEX_GAP, 70)]
    worst_pref = max(float(np.abs(p - ref[t, :t - MIN_INDEX_GAP + 1].numpy()).max())
                     for p, t in zip(pref, range(MIN_INDEX_GAP, 70)))
    scale = float(ref.abs().max())
    out = {"rel_corr_max_vs_direct": float((a - ref).abs().max()) / scale,
           "rel_bank_block_vs_direct": float((blk - ref[40:]).abs().max()) / scale,
           "rel_bank_single_vs_direct": float((one - ref[40:]).abs().max()) / scale,
           "rel_blockscorer_prefix_vs_direct": worst_pref / scale,
           "revisit_found": int(one[20, :60 - MIN_INDEX_GAP + 1].argmax()) == 7}
    assert all(v < 1e-5 for k, v in out.items() if k.startswith("rel_")) and out["revisit_found"], out
    rep["scorer"] = out
    print("  scorer", out, flush=True)


def _check_preprocessing(rep, tmp):
    """Adapter BEV (cpu transcription of voxelocc) == official class reading
    the raw file (KITTI / MulRan / NCLT); packbits round trip; batched CPU
    sinogram == per-sample RING generate_RING_cpu; per-keyframe CPU time."""
    import torch
    core = rol.import_ring_core()
    sino = Sino("cpu")
    rows = []
    for role, sensor, q in CHECK_SCANS:
        scan_ids, _ = load_cache(role, sensor, q)
        loader = rol.make_loader(sensor, q)
        occs = []
        for i in (0, len(scan_ids) // 2, len(scan_ids) - 1):
            sid = int(scan_ids[i])
            t0 = time.time()
            pts = np.asarray(loader[sid]["points"])
            t_load = time.time() - t0
            t0 = time.time()
            occ, npts = occupancy_bev(None, sensor, pts, "cpu")
            t_bev = time.time() - t0
            r = {"role": role, "seq": f"{sensor}/{q}", "scan_id": sid, "points_after_official": npts,
                 "occupied_cells": int(occ.sum()), "t_load_s": round(t_load, 3),
                 "t_official_points_plus_occ_s": round(t_bev, 3)}
            if sensor in ("kitti", "mulran", "nclt"):
                _, pcn_f = rol.official_points_from_file(sensor, rol.raw_scan_path(loader, sensor, sid))
                r["equal_to_official_raw_file_read"] = bool(np.array_equal(rol.occupancy_numpy(pcn_f), occ))
            packed = np.packbits(occ[None], axis=-1)
            r["packbits_roundtrip"] = bool(torch.equal(unpack(packed)[0, 0], torch.from_numpy(occ).float()))
            occs.append(occ)
            rows.append(r)
            print("  prep", r, flush=True)
        bev = torch.from_numpy(np.stack(occs).astype(np.float32)).unsqueeze(1)
        s_batch = sino(bev)
        s_each = torch.stack([core.generate_RING_cpu(b)[0] for b in bev])
        rows[-1]["cpu_sinogram_batched_equals_per_sample"] = bool(torch.equal(s_batch, s_each))
        rows[-1]["sinogram_max"] = float(s_batch.max())
        del loader
    rep["preprocessing"] = rows
    assert all(r.get("equal_to_official_raw_file_read", True) and r["packbits_roundtrip"] for r in rows)
    assert all(r.get("cpu_sinogram_batched_equals_per_sample", True) for r in rows)


def _check_sampler(rep):
    from _deepring_reimpl_model import RECIPE, EpisodeSampler
    pos = [load_cache("train", s, q)[1][:, :3, 3] for s, q in TRAIN]
    t0 = time.time()
    sm = EpisodeSampler(pos, np.random.default_rng(SEED), ways=RECIPE["ways"],
                        queries=RECIPE["queries_per_class"])
    t_init = time.time() - t0
    t0 = time.time()
    seen, worst = np.zeros(len(pos), int), {"min_shot_sep": 1e9, "max_query_dist": 0.0,
                                            "min_query_to_other_shot": 1e9}
    for _ in range(200):
        s, shots, qs = sm.sample()
        seen[s] += 1
        P = pos[s]
        D = np.linalg.norm(P[shots][:, None] - P[shots][None], axis=-1) + np.eye(len(shots)) * 1e9
        worst["min_shot_sep"] = min(worst["min_shot_sep"], float(D.min()))
        for k in range(len(shots)):
            dq = np.linalg.norm(P[qs[k]][:, None] - P[shots][None], axis=-1)   # (6, 24)
            worst["max_query_dist"] = max(worst["max_query_dist"], float(dq[:, k].max()))
            worst["min_query_to_other_shot"] = min(worst["min_query_to_other_shot"],
                                                   float(np.delete(dq, k, axis=1).min()))
            assert shots[k] not in qs[k] and len(set(qs[k])) == len(qs[k])
    out = {"episodes": 200, "init_s": round(t_init, 1),
           "ms_per_episode": round(1e3 * (time.time() - t0) / 200, 1),
           "episodes_per_sequence": {f"{a}/{b}": int(c) for (a, b), c in zip(TRAIN, seen)},
           **{k: round(v, 2) for k, v in worst.items()}}
    assert worst["min_shot_sep"] > 30 and worst["max_query_dist"] <= 10 \
        and worst["min_query_to_other_shot"] > 20, out
    rep["episode_sampler"] = out
    print("  sampler", out, flush=True)


def _pick_val_window():
    """Smallest val prefix with >= 5 revisit queries (dry-run validation)."""
    from baselines.eval_utils import _find_revisit_queries
    best = None
    for s, q in VAL:
        P = load_cache("val", s, q)[1][:, :3, 3]
        for L in range(100, min(len(P), 1200) + 1, 50):
            if len(_find_revisit_queries(P[:L], VAL_RADIUS, VAL_SKIP)) >= 5:
                if best is None or L < best[1]:
                    best = (f"{s}/{q}", L)
                break
    return best


def _check_dry_run(rep, tmp):
    """CPU dry run of the whole chain on tiny data (BEV cpu transcription +
    CPU radon): train 2 epochs x 2 episodes (4 ways x 2 queries) with per-epoch
    validation and selection, resume equivalence, test ledger on 80 keyframes
    of nclt/2013-01-10 (untrained, format only; deleted afterwards)."""
    import torch
    root = Path(tmp) / "bev"
    tr = ["kitti/05", "helipr/Bridge01"]
    vname, vlen = _pick_val_window()
    for seq in tr:
        cmd_bev("train", seq, "cpu", root, limit=300)
    cmd_bev("val", vname, "cpu", root, limit=vlen)
    base = dict(device="cpu", sino_backend="cpu", bev_root=root, epochs=2, episodes=2,
                ways=4, queries=2, max_frames=300, val_max_frames=vlen,
                train_seqs=tr, val_seqs=[vname])
    t0 = time.time()
    recA = train(argparse.Namespace(**base, run_dir=Path(tmp) / "runA",
                                    selection_out=Path(tmp) / "selA.json"))
    t_full = time.time() - t0
    trB = dict(base, epochs=1)
    train(argparse.Namespace(**trB, run_dir=Path(tmp) / "runB", selection_out=Path(tmp) / "selB.json"))
    recB = train(argparse.Namespace(**base, run_dir=Path(tmp) / "runB",
                                    selection_out=Path(tmp) / "selB.json"))
    a = torch.load(Path(tmp) / "runA/epoch_02.pth", weights_only=False)["state_dict"]
    b = torch.load(Path(tmp) / "runB/epoch_02.pth", weights_only=False)["state_dict"]
    same = all(torch.equal(a[k], b[k]) for k in a)
    out = {"train_seqs": tr, "val": [vname, vlen], "wall_s_2_epochs": round(t_full, 1),
           "epochs": [{k: e[k] for k in ("epoch", "lr", "loss_mean", "W", "b", "family_macro_r1",
                                         "train_s", "val_s")} for e in recA["epochs"]],
           "best": recA["best"]["epoch"], "resume_bit_identical": same,
           "resumed_after_epoch": recB.get("resumed_after_epoch")}
    assert same and recA["finished"], out
    # test ledger format on 80 keyframes, untrained network
    cmd_bev("test_odom", "nclt/2013-01-10", "cpu", root, limit=80, reuse_ring=False)
    p = dump_one("nclt", "2013-01-10", Path(tmp) / "runA/epoch_02.pth", "cpu", "cpu", 8,
                 Path(tmp) / "dump_deepring_nclt_2013-01-10.npz", bev_root=root, limit=80,
                 extra_meta={"DRY_RUN": "CPU, 2x2 tiny episodes, format check only"})
    z = np.load(p)
    ring_ref = np.load(REPO / "results/pd2026_pr/dump_ring_kitti_05.npz")
    seqot = np.load(REPO / "results/pdv2_c3/odom/dump_seqot_nclt_2013-01-10.npz")
    m = len(z["query_idx"])
    mine = {k: str(z[k].dtype) for k in z.files if k != "meta"}
    out["ledger"] = {
        "arrays_dtypes": mine,
        "same_as_ring_pr_dump_v1": mine == {k: str(ring_ref[k].dtype) for k in ring_ref.files if k != "meta"},
        "schema": json.loads(str(z["meta"]))["schema"],
        **{f"{k}_equal_seqot_v2_shared_rows": bool(np.array_equal(z[k], seqot[k][:m]))
           for k in ("query_idx", "min_past_gt_dist", "n_candidates")},
        "causal": bool((z["j_star"] <= z["query_idx"] - MIN_INDEX_GAP).all()),
        "scorer_crosscheck": json.loads(str(z["meta"]))["scorer_crosscheck"]}
    assert out["ledger"]["same_as_ring_pr_dump_v1"] and out["ledger"]["causal"] and all(
        out["ledger"][f"{k}_equal_seqot_v2_shared_rows"]
        for k in ("query_idx", "min_past_gt_dist", "n_candidates")), out["ledger"]
    rep["dry_run"] = out
    print("  dry run", json.dumps(out)[:2000], flush=True)


def _unet_flops_and_activations():
    """Analytic forward FLOPs of the UNet at 120 x 120 and saved-for-backward
    bytes of one episode sample (CPU, autograd hooks, B = 2)."""
    import torch
    from _deepring_reimpl_model import DeepRING, corr_max
    model = DeepRING()
    flops = [0]

    def hook(mod, inp, out):
        if isinstance(mod, torch.nn.Conv2d):
            k = mod.kernel_size[0] * mod.kernel_size[1]
            flops[0] += 2 * out.numel() * mod.in_channels * k // out.shape[0]
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        model.embed(torch.rand(1, 1, 120, 120))
    for h in hs:
        h.remove()
    saved = [0]

    def pack(t):
        saved[0] += t.numel() * t.element_size()
        return t
    x = torch.rand(2, 1, 120, 120) * 5
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        M = model.embed(x)
        loss = corr_max(M[1:], M[:1]).sum()
    t0 = time.time()
    loss.backward()
    xb = torch.rand(4, 1, 120, 120) * 5
    t0 = time.time()
    Mb = model.embed(xb)
    corr_max(Mb[1:], Mb[:1]).sum().backward()
    return {"unet_fwd_gflop_per_scan": round(flops[0] / 1e9, 2),
            "saved_activation_mb_per_scan": round(saved[0] / 2 / 2 ** 20, 1),
            "cpu_fwd_bwd_s_per_scan_B4": round((time.time() - t0) / 4, 2),
            "cpu_threads": torch.get_num_threads()}


def cmd_check():
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    rep = {"_meta": {"date": time.strftime("%Y-%m-%d %H:%M"), "note": NOTE,
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "torch": torch.__version__}}
    tmp = tempfile.mkdtemp(prefix="deepring_check_", dir=str(WORK.parent))
    try:
        for f in (_check_unet, _check_scorer, _check_sampler):
            print(f"[check] {f.__name__}", flush=True)
            f(rep)
        print("[check] _check_preprocessing", flush=True)
        _check_preprocessing(rep, tmp)
        print("[check] cost", flush=True)
        rep["cost"] = _unet_flops_and_activations()
        print("  cost", rep["cost"], flush=True)
        print("[check] _check_dry_run", flush=True)
        _check_dry_run(rep, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _write_json(RESULTS / "check_cpu.json", rep)
    print(f"CHECK_OK -> {RESULTS / 'check_cpu.json'}", flush=True)


def cmd_gpu_check():
    """Server, before the BEV stage: voxelocc == CPU transcription, batched
    generate_RING == per-sample, GPU scorer == eq. 8 written out (fp32),
    one full 168-scan episode (time, peak memory)."""
    import torch
    from _deepring_reimpl_model import (RECIPE, DeepRING, SpecBank, corr_max_direct,
                                        episode_loss)
    assert torch.cuda.is_available()
    scorer_fp32_guard("cuda")
    core = rol.import_ring_core()
    rep, occs = {"bev": []}, []
    for role, sensor, q in CHECK_SCANS:
        scan_ids, _ = load_cache(role, sensor, q)
        loader = rol.make_loader(sensor, q)
        for i in (0, len(scan_ids) // 2):
            pts = loader[int(scan_ids[i])]["points"]
            g, _ = occupancy_bev(core, sensor, pts, "gpu")
            c, _ = occupancy_bev(None, sensor, pts, "cpu")
            rep["bev"].append({"seq": f"{sensor}/{q}", "equal": bool(np.array_equal(g, c))})
            occs.append(g)
    bev = torch.from_numpy(np.stack(occs).astype(np.float32)).unsqueeze(1).cuda()
    s_b = Sino("gpu")(bev)
    s_e = torch.cat([core.generate_RING(b)[0].reshape(1, 1, 120, 120) for b in bev])
    rep["sinogram_batched_vs_per_sample_max_abs"] = float((s_b - s_e).abs().max())
    torch.manual_seed(0)
    model = DeepRING().cuda()
    with torch.no_grad():
        M = model.embed(s_b)
    bank = SpecBank(len(M), "cuda")
    bank.put(0, M)
    got = bank.corr(list(range(len(M))), len(M)).double().cpu()
    ref = corr_max_direct(M.double().cpu(), M.double().cpu())
    rep["gpu_scorer_rel_err"] = float((got - ref).abs().max() / ref.abs().max())
    # one full paper episode
    ways, nq = RECIPE["ways"], RECIPE["queries_per_class"]
    x = s_b[torch.randint(len(s_b), (ways * (1 + nq),))]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        loss, _ = episode_loss(model, model.embed(x), ways, nq)
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    rep["episode_168_s"] = [round(t, 3) for t in ts]
    rep["episode_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    print(json.dumps(rep, indent=1))
    assert all(r["equal"] for r in rep["bev"]) and rep["gpu_scorer_rel_err"] < 1e-5 \
        and rep["sinogram_batched_vs_per_sample_max_abs"] < 1e-4, rep
    _write_json(RESULTS / "check_gpu.json", rep)
    print("GPU_CHECK_OK", flush=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bev")
    p.add_argument("--role", choices=tuple(ROLES), required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--no-reuse-ring", action="store_true")
    p = sub.add_parser("jobs")
    p.add_argument("--roles", nargs="+", default=list(ROLES))
    sub.add_parser("status")
    p = sub.add_parser("train")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sino-backend", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--bev-root", type=Path, default=BEV_ROOT)
    p.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p.add_argument("--selection-out", type=Path, default=SELECTION)
    for k in ("epochs", "episodes", "ways", "queries", "max-frames", "val-max-frames"):
        p.add_argument(f"--{k}", type=int, default=0, help="0 = paper value / all frames")
    p.add_argument("--train-seqs", nargs="+", default=None)
    p.add_argument("--val-seqs", nargs="+", default=None)
    p = sub.add_parser("dump")
    p.add_argument("--seqs", nargs="+", default=None)
    p.add_argument("--checkpoint", default=None, help="default: best of train_selection.json")
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    sub.add_parser("aggregate")
    sub.add_parser("check")
    sub.add_parser("gpu-check")
    a = ap.parse_args()
    if a.cmd == "bev":
        cmd_bev(a.role, a.seq, reuse_ring=not a.no_reuse_ring)
    elif a.cmd == "jobs":
        cmd_jobs(a.roles)
    elif a.cmd == "status":
        cmd_status()
    elif a.cmd == "train":
        train(a)
    elif a.cmd == "dump":
        cmd_dump(a.seqs, a.checkpoint, a.block, a.worker, a.overwrite)
    elif a.cmd == "aggregate":
        cmd_aggregate()
    elif a.cmd == "check":
        cmd_check()
    elif a.cmd == "gpu-check":
        cmd_gpu_check()


if __name__ == "__main__":
    main()
