#!/usr/bin/env python3
"""HeLiOS from-scratch pilot with the author-recommended recipe (2026-11-08).

Diagnostic only (not a main-table replacement).  Seed 1, split-v2 train8 /
val9, two learning-rate candidates run sequentially, everything else as the
existing official run (scripts/_helios_official_train.py, unchanged, imported):

  candidate     lr     batch_size  batch_split_size  epochs  milestones
  lr5e-4        5e-4   2048        32                20      40, 70 (official)
  lr1e-4        1e-4   2048        32                20      40, 70 (official)
  (existing     1e-3   2048        64                80      40, 70; seed1 val-best epoch 10)

Training = TR.run_training (official multistage step, losses, sampler,
resume.pth per epoch) with config overrides lr / batch_split_size / epochs /
num_workers=2 (server sharing rule; official 4) / save_freq=1 (a checkpoint per
epoch).  The model is initialised from scratch by the official
initialize_model_and_optimizer; the author checkpoint is never loaded.  The
init state is hashed by a wrapper around T.initialize_model_and_optimizer.

Selection metric (fixed by coordination.json): split-v2 VAL 9 sequences, all
keyframes, intra-sequence 5 m R@1 with a 30-keyframe gap
(_helios_official_dump.recall_at_1), families equal (_pdv2_c3._family_pooled),
ties -> earlier epoch.  Same scans, mapping and preprocessing as the existing
selection (_helios_official_dump.cmd_select).  The test set is not touched.

Subcommands (inside ring_baseline, CUDA_VISIBLE_DEVICES=0):
  probe       encode speed / equality of the official per-scan path vs a
              prefetching per-scan path on an existing seed1 checkpoint
  determinism init-state hash and first-epoch sampler hash (CPU)
  smoke       tiny 1-epoch run (official subset_pickle) + 1 eval sequence
  train  --cand C [--epochs N] [--resume]
  evaluate --cand C [--follow]   score epoch checkpoints at EVAL_EVERY
  decide      20-epoch decision + extension rule (protocol.json)
  verify      recompute family R@1 from the saved per-sequence ledgers
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
os.environ.setdefault("SEED", "1")

import _helios_official_prep as P  # noqa: E402
import _helios_official_train as TR  # noqa: E402
import _helios_official_dump as D  # noqa: E402
from _pdv2_pilot import VAL  # noqa: E402

assert TR.SEED == 1, "pilot is seed 1 only"
WORK = REPO / "_external/overnight_repro_20261108/helios_scratch"
CANDIDATES = {"lr5e-4": 5e-4, "lr1e-4": 1e-4}
ORDER = ["lr5e-4", "lr1e-4"]
SPLIT = 32
EPOCHS_INITIAL = 20
EPOCHS_EXTENDED = 40
NUM_WORKERS = 2
EVAL_EVERY = 2            # fixed before any candidate result (probe timing)
EVAL_BATCH = 32           # scans per sparse batch in the selection encoder (64 OOM at the 4.5 GiB cap)
TRAIN_MEM_GIB = 10.0      # per-process caps: training + evaluator <= 15 GiB per job
EVAL_MEM_GIB = 4.5
OLD_SEED1_R1 = 0.5372937495222714
OLD_SEED1_CKPT = P.WORK / "runs/Oct12_12-22-10/HeLiOS_default_trainning_10.pth"
AUTHOR_CKPTS = [P.HELIOS / "weights/HeLiOS.pth", P.HELIOS / "weights/HeLiOS-S.pth"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wjson(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def cand_dir(cand: str) -> Path:
    return WORK / cand


def write_cand_config(cand: str, epochs: int) -> Path:
    return TR.write_config(cand_dir(cand) / f"config_{cand}_e{epochs}.txt", P.DATA,
                           lr=CANDIDATES[cand], batch_split_size=SPLIT, epochs=epochs,
                           num_workers=NUM_WORKERS, save_freq=1)


def state_hash(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------ encoding
def val_paths(sensor: str, seq: str):
    z = np.load(P.seq_dir("val", sensor, seq) / "index.npz")
    sids = z["scan_ids"][z["generate"]]
    return sids, [str(P.DATA / P.rel_scan_path("val", sensor, seq, s)) for s in sids]


def encode_prefetch(model, params, paths, device, workers=1, batch=1):
    """Official compute_embedding (quantizer, ME batched_coordinates, ones
    features, model(batch)['global']) with file reading and quantisation
    prefetched by a DataLoader worker.  batch=1 is the official one scan per
    forward; batch>1 puts several scans in one sparse batch (see probe)."""
    import torch
    import MinkowskiEngine as ME
    from datasets.pnv_raw import PNVPointCloudLoader

    quant = params.model_params.quantizer

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            pc = torch.tensor(PNVPointCloudLoader()(paths[i]), dtype=torch.float32)
            coords, _ = quant(pc)
            return coords

    dl = torch.utils.data.DataLoader(_DS(), batch_size=batch, num_workers=workers, collate_fn=list,
                                     prefetch_factor=4 if workers else None)
    model.eval()
    out, i = None, 0
    for coords in dl:
        with torch.no_grad():
            bcoords = ME.utils.batched_coordinates(coords)
            feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
            emb = model({"coords": bcoords.to(device), "features": feats.to(device)})["global"]
            emb = emb.detach().cpu().numpy()
        if out is None:
            out = np.zeros((len(paths), emb.shape[1]), dtype=emb.dtype)
        out[i:i + len(emb)] = emb
        i += len(emb)
    assert i == len(paths)
    return out.astype(np.float32)


def mem_cap(gib: float):
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2 ** 30 / total), 0)


def eval_checkpoint(ckpt: Path, out_json: Path, device="cuda", workers=1, batch=EVAL_BATCH):
    """Split-v2 VAL family R@1 of one checkpoint; resumable per sequence."""
    from _pdv2_c3 import _family_pooled
    mem_cap(EVAL_MEM_GIB)
    E, params = D._official(WORK / "eval_config.txt")
    sha = P.sha256(ckpt)
    rec = json.loads(out_json.read_text()) if out_json.exists() else {}
    if rec.get("sha256") != sha or rec.get("encode_batch") != batch:
        rec = {"checkpoint": str(ckpt), "sha256": sha, "per_seq": {}, "started": now(),
               "encode_batch": batch, "encoder": "official compute_embedding path, " +
               ("one scan per forward" if batch == 1 else f"{batch} scans per sparse batch")}
    model = None
    for sensor, seq in VAL:
        key = f"{sensor}/{seq}"
        if key in rec["per_seq"]:
            continue
        if model is None:
            model = D.official_model(E, params, ckpt, device)
        sids, paths = val_paths(sensor, seq)
        t0 = time.time()
        desc = encode_prefetch(model, params, paths, device, workers, batch)
        dt = time.time() - t0
        poses = np.load(P.cache_path("val", sensor, seq))["poses"].astype(np.float64)
        assert len(poses) == len(desc), (key, len(poses), len(desc))
        recs = []
        r1, nq = D.recall_at_1(desc, poses, per_query_records=recs)
        led = out_json.parent / "ledgers" / f"{out_json.stem}__{sensor}_{seq}.npz"
        led.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(led, query_idx=np.array([r["query_idx"] for r in recs], np.int64),
                            top1_idx=np.array([r["top1_idx"] for r in recs], np.int64),
                            top1_geo_dist_m=np.array([r["top1_geo_dist_m"] for r in recs], np.float64),
                            correct=np.array([r["success_at_k1"] for r in recs], bool),
                            desc_norm=np.linalg.norm(desc, axis=1).astype(np.float32))
        rec["per_seq"][key] = {"r1": r1, "n_queries": nq, "n_frames": int(len(sids)),
                               "encode_s": round(dt, 1), "ledger": str(led.relative_to(REPO))}
        print(f"  {ckpt.name} {key}: R@1 {r1:.4f} ({nq} q, {len(sids)} f, {dt:.0f} s)", flush=True)
        wjson(out_json, rec)
    rec["family_macro_r1"] = _family_pooled(rec["per_seq"])
    rec["finished"] = now()
    wjson(out_json, rec)
    print(f"== {ckpt.name}: family R@1 {rec['family_macro_r1']:.6f}", flush=True)
    return rec


# ------------------------------------------------------------ probe
def cmd_probe(n: int, out: Path):
    import torch
    E, params = D._official(WORK / "eval_config.txt")
    model = D.official_model(E, params, OLD_SEED1_CKPT, "cuda")
    rep = {"checkpoint": str(OLD_SEED1_CKPT), "sha256": P.sha256(OLD_SEED1_CKPT), "n": n,
           "gpu_mem_before_gib": round(torch.cuda.mem_get_info()[0] / 2 ** 30, 2)}
    sensor, seq = "helipr", "Roundabout02"
    sids, paths = val_paths(sensor, seq)
    t0 = time.time()
    ref, _, _ = D.descriptors(E, params, model, "val", sensor, seq, "cuda", limit=n)
    rep["official_s"] = round(time.time() - t0, 2)
    for w in (1, 2):
        t0 = time.time()
        fast = encode_prefetch(model, params, paths[:n], "cuda", workers=w)
        rep[f"prefetch_w{w}_s"] = round(time.time() - t0, 2)
        rep[f"prefetch_w{w}_maxabs"] = float(np.abs(fast - ref).max())
        rep[f"prefetch_w{w}_bitwise_equal"] = bool(np.array_equal(fast, ref))
    rep["cuda_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    rep["val_frames_total"] = int(sum(len(val_paths(s, q)[0]) for s, q in VAL))
    rep["est_full_val_s_prefetch_w1"] = round(rep["prefetch_w1_s"] / n * rep["val_frames_total"])
    wjson(out, rep)
    print(json.dumps(rep, indent=1), flush=True)


# ------------------------------------------------------------ determinism
def init_state(seed: int = 1):
    """Same order as TR.run_training: seed random/numpy/torch, then the
    official initialize_model_and_optimizer."""
    import torch
    TR.official_path()
    from misc.utils import TrainingParams
    import training.trainer as T
    params = TrainingParams(str(write_cand_config("lr5e-4", EPOCHS_INITIAL)), str(TR.MODEL_CONFIG))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, _, _ = T.initialize_model_and_optimizer(params, "cpu")
    return state_hash(model.state_dict()), params


def cmd_determinism(out: Path):
    import torch
    TR.official_path()
    from datasets.dataset_utils import make_dataloaders
    rep = {"time": now()}
    h1, params = init_state()
    h2, _ = init_state()
    rep["init_state_sha256"] = [h1, h2]
    rep["init_reproducible"] = h1 == h2
    author = {}
    for ck in AUTHOR_CKPTS:
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        author[ck.name] = state_hash(sd)
    rep["author_state_sha256"] = author
    rep["init_differs_from_author"] = h1 not in author.values()
    hashes = []
    for _ in range(2):
        random.seed(TR.SEED)
        np.random.seed(TR.SEED)
        torch.manual_seed(TR.SEED)
        dls = make_dataloaders(params)
        batches = list(dls["train"].batch_sampler)
        hashes.append(hashlib.sha256(json.dumps([list(map(int, b)) for b in batches]).encode()).hexdigest())
        rep["epoch1_train_batches"] = len(batches)
        rep["epoch1_batch_sizes"] = sorted({len(b) for b in batches})
    rep["epoch1_sampler_sha256"] = hashes
    rep["sampler_reproducible"] = hashes[0] == hashes[1]
    rep["note"] = ("BatchSampler draws from python random only; batch_size 2048 equals the existing "
                   "seed1 run, so its batch composition is the same sequence; batch_split_size only "
                   "changes the mini-batch partition inside the multistage step.  Augmentation in "
                   "DataLoader workers depends on num_workers (2 here, 4 in the existing run).")
    rep["all_ok"] = bool(rep["init_reproducible"] and rep["init_differs_from_author"]
                         and rep["sampler_reproducible"])
    wjson(out, rep)
    print(json.dumps(rep, indent=1), flush=True)
    if not rep["all_ok"]:
        sys.exit(1)


# ------------------------------------------------------------ training
def _hook_init(run_dir: Path):
    """Wrap T.initialize_model_and_optimizer to record (and save) the init state."""
    import torch
    import training.trainer as T
    orig = T.initialize_model_and_optimizer

    def wrapped(params, device):
        model, opt, sch = orig(params, device)
        sd = model.state_dict()
        h = state_hash(sd)
        wjson(run_dir / "init_state.json", {"init_state_sha256": h, "time": now(),
                                            "author_loaded": False})
        torch.save(sd, run_dir / "init_state.pth")
        print(f"===> init state sha256 {h}", flush=True)
        return model, opt, sch
    T.initialize_model_and_optimizer = wrapped


def cmd_train(cand: str, epochs: int, resume: bool):
    import torch
    TR.official_path()
    from misc.utils import TrainingParams
    torch.set_num_threads(2)
    mem_cap(TRAIN_MEM_GIB)
    d = cand_dir(cand)
    run_dir = d / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = write_cand_config(cand, epochs)
    params = TrainingParams(str(cfg), str(TR.MODEL_CONFIG))
    params.print()
    if not resume:
        _hook_init(run_dir)
    TR.run_training(params, run_dir, d / "train_record.json", resume=resume, device="cuda")


def epochs_to_eval(epochs: int):
    return [e for e in range(1, epochs + 1) if e % EVAL_EVERY == 0]


def cmd_evaluate(cand: str, epochs: int, follow: bool):
    d = cand_dir(cand)
    run_dir = d / "run"
    for e in epochs_to_eval(epochs):
        ck = run_dir / f"HeLiOS_default_trainning_{e}.pth"
        out = d / "val" / f"epoch{e:03d}.json"
        if out.exists() and json.loads(out.read_text()).get("finished"):
            continue
        def recorded():      # the record is written after the epoch file and resume.pth
            r = d / "train_record.json"
            return r.exists() and any(x["epoch"] == e for x in json.loads(r.read_text())["epochs"])
        while follow and not recorded() and (WORK / "flags" / f"{cand}.TRAINING").exists():
            time.sleep(60)
        if not (ck.exists() and recorded()):
            print(f"not ready: {ck}", flush=True)
            continue
        eval_checkpoint(ck, out)
    summarize(cand)


def summarize(cand: str):
    d = cand_dir(cand)
    rows, row_batches = [], []
    for p in sorted((d / "val").glob("epoch*.json")):
        r = json.loads(p.read_text())
        if r.get("finished"):
            rows.append({"epoch": int(p.stem[5:]), "family_macro_r1": r["family_macro_r1"],
                         "sha256": r["sha256"], "checkpoint": r["checkpoint"]})
            row_batches.append(r.get("encode_batch"))
    s = {"candidate": cand, "lr": CANDIDATES[cand], "batch_split_size": SPLIT, "eval_every": EVAL_EVERY,
         "rows": rows, "time": now()}
    # raw difference only against the old seed1 e10 checkpoint rescored with the same
    # B32 encoder (coordination validation_encoding_decision); OLD_SEED1_R1 is the
    # official one-scan value and is not comparable, no additive correction
    cal_p = WORK / f"old_seed1_e10_b{EVAL_BATCH}.json"
    cal = json.loads(cal_p.read_text()) if cal_p.exists() else {}
    cal_r1 = cal.get("family_macro_r1")
    cmp = {"status": "pending", "raw_diff": None, "baseline": str(cal_p), "baseline_sha256": cal.get("sha256"),
           "baseline_encode_batch": cal.get("encode_batch"), "baseline_family_macro_r1": cal_r1,
           "rows_encode_batch": sorted(set(row_batches), key=str)}
    if rows:
        best = max(rows, key=lambda r: (r["family_macro_r1"], -r["epoch"]))
        s["val_best"] = best
        if cal.get("finished"):
            ok = (cal.get("encode_batch") == EVAL_BATCH and isinstance(cal_r1, (int, float))
                  and bool(np.isfinite(cal_r1)) and all(b == EVAL_BATCH for b in row_batches))
            cmp["status"] = "matched_b32" if ok else "not_comparable"
            if ok:
                cmp["raw_diff"] = best["family_macro_r1"] - cal_r1
    s["val_best_vs_old_seed1_b32"] = cmp
    wjson(d / "val_summary.json", s)
    return s


def cmd_decide():
    """Pre-registered: extend to 40 epochs only the candidate with the higher
    20-epoch val-best (ties -> lr5e-4) and only if its best evaluated epoch is
    one of the last two evaluated epochs (<= epoch 20), i.e. still improving."""
    sums = {c: summarize(c) for c in ORDER}
    last2 = epochs_to_eval(EPOCHS_INITIAL)[-2:]
    for c in ORDER:
        assert [r["epoch"] for r in sums[c]["rows"] if r["epoch"] <= EPOCHS_INITIAL] == \
            epochs_to_eval(EPOCHS_INITIAL), f"{c}: 20-epoch evaluation incomplete"

    def best20(c):
        return max((r for r in sums[c]["rows"] if r["epoch"] <= EPOCHS_INITIAL),
                   key=lambda r: (r["family_macro_r1"], -r["epoch"]))
    b = {c: best20(c) for c in ORDER}
    top = max(ORDER, key=lambda c: (b[c]["family_macro_r1"], -ORDER.index(c)))
    extend = b[top]["epoch"] in last2
    dec = {"time": now(), "best_at_20": b, "top_candidate": top, "last_two_evaluated": last2,
           "extend": top if extend else None,
           "rule": cmd_decide.__doc__.strip()}
    wjson(WORK / "decision_20.json", dec)
    print(json.dumps(dec, indent=1), flush=True)
    return dec


def cmd_verify():
    """Recompute R@1 and family macro from the saved ledgers and compare."""
    from _pdv2_c3 import _family_pooled
    out = {"time": now(), "candidates": {}}
    ok = True
    for c in ORDER:
        for p in sorted((cand_dir(c) / "val").glob("epoch*.json")):
            r = json.loads(p.read_text())
            if not r.get("finished"):
                continue
            per = {}
            for k, v in r["per_seq"].items():
                z = np.load(REPO / v["ledger"])
                per[k] = {"r1": float(z["correct"].mean()) if len(z["correct"]) else 0.0,
                          "n_queries": int(len(z["correct"]))}
            fam = _family_pooled(per)
            good = abs(fam - r["family_macro_r1"]) < 1e-12 and \
                all(per[k]["n_queries"] == r["per_seq"][k]["n_queries"] for k in per)
            ok &= good
            out["candidates"].setdefault(c, {})[p.stem] = {"stored": r["family_macro_r1"],
                                                            "recomputed": fam, "ok": good,
                                                            "sha256": r["sha256"]}
    out["all_ok"] = bool(ok)
    wjson(WORK / "verify_ledgers.json", out)
    print(json.dumps(out, indent=1), flush=True)


def cmd_smoke(n: int):
    """Official subset_pickle (first n queries per folder), 1 epoch at
    lr5e-4 / split 32, then one VAL sequence encoded with its checkpoint."""
    import torch
    TR.official_path()
    from misc.utils import TrainingParams
    torch.set_num_threads(2)
    mem_cap(TRAIN_MEM_GIB)
    sm = WORK / "smoke"
    sm.mkdir(parents=True, exist_ok=True)
    for name in ("training.pickle", "validation.pickle"):
        TR.subset_pickle(P.DATA / name, sm / name, n)
    rel = os.path.relpath(sm, P.DATA)
    cfg = TR.write_config(sm / "config_smoke.txt", P.DATA, train_file=f"{rel}/training.pickle",
                          val_file=f"{rel}/validation.pickle", epochs=1, lr=5e-4,
                          batch_split_size=SPLIT, num_workers=NUM_WORKERS, save_freq=1)
    params = TrainingParams(str(cfg), str(TR.MODEL_CONFIG))
    _hook_init(sm / "run")
    rec = TR.run_training(params, sm / "run", sm / "train_record.json", resume=False, device="cuda")
    e = rec["epochs"][0]
    assert np.isfinite(e["train_loss_truncated"]) and e["train_batches"] >= 1, e
    E, p2 = D._official(WORK / "eval_config.txt")
    model = D.official_model(E, p2, sm / "run/HeLiOS_default_trainning_1.pth", "cuda")
    sids, paths = val_paths("kitti", "09")
    t0 = time.time()
    desc = encode_prefetch(model, p2, paths[:64], "cuda", 1, batch=32)
    print(json.dumps({"smoke_epoch": e, "encode_64_b32_s": round(time.time() - t0, 1),
                      "desc_shape": list(desc.shape), "finite": bool(np.isfinite(desc).all())}), flush=True)
    print("SMOKE_OK", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--n", type=int, default=300)
    sub.add_parser("determinism")
    p = sub.add_parser("smoke")
    p.add_argument("--n", type=int, default=150)
    p = sub.add_parser("train")
    p.add_argument("--cand", choices=ORDER, required=True)
    p.add_argument("--epochs", type=int, default=EPOCHS_INITIAL)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--cand", choices=ORDER, required=True)
    p.add_argument("--epochs", type=int, default=EPOCHS_INITIAL)
    p.add_argument("--follow", action="store_true")
    p = sub.add_parser("eval-old")
    p.add_argument("--batch", type=int, default=EVAL_BATCH)
    p = sub.add_parser("eval-official")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    sub.add_parser("decide")
    sub.add_parser("verify")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.cmd == "probe":
        cmd_probe(a.n, WORK / "probe_encode.json")
    elif a.cmd == "determinism":
        cmd_determinism(WORK / "determinism.json")
    elif a.cmd == "smoke":
        cmd_smoke(a.n)
    elif a.cmd == "train":
        cmd_train(a.cand, a.epochs, a.resume)
    elif a.cmd == "evaluate":
        cmd_evaluate(a.cand, a.epochs, a.follow)
    elif a.cmd == "eval-old":
        eval_checkpoint(OLD_SEED1_CKPT, WORK / f"old_seed1_e10_b{a.batch}.json", batch=a.batch)
    elif a.cmd == "eval-official":     # post-hoc confirmation, one scan per forward
        assert a.checkpoint.resolve() not in [c.resolve() for c in AUTHOR_CKPTS]
        if not (WORK / "flags" / "OFFICIAL_APPROVED").exists():   # root schedules this stage
            print(f"OFFICIAL_PENDING {a.checkpoint}: root creates flags/OFFICIAL_APPROVED to allow", flush=True)
            return
        eval_checkpoint(a.checkpoint, a.out, batch=1)
    elif a.cmd == "decide":
        cmd_decide()
    elif a.cmd == "verify":
        cmd_verify()


if __name__ == "__main__":
    main()
