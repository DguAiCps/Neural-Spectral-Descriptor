#!/usr/bin/env python3
"""HeLiOS from-scratch pilot: official single-scan (B1) VAL9 confirmation (2026-11-08).

scratch_seed1_pilot_diagnostic: seed-1 from-scratch pilot (author/public weights
never loaded).  Diagnostic confirmation of the B32-frozen pair only; it never
replaces main-table values and never touches the test set.

Exactly two frozen models (fixed order, decided before any B1 result):
  1. old_e10    helios_official_pdv2/runs/Oct12_12-22-10/HeLiOS_default_trainning_10.pth (existing seed1)
  2. candidate  helios_scratch/lr5e-4/run/HeLiOS_default_trainning_10.pth (B32 overall val-best)
No new training, no epoch reselection, no extra candidates (lr1e-4 is not re-scored).
The existing Oct-2026 B1 value of old_e10 (train_selection.json) is not reused: it
has no per-query ledger and used get_latent_vectors instead of encode_prefetch.

Encoder (identical to the running public-FT B1 check):
scripts/_helios_scratch_pilot_20261108.eval_checkpoint with batch=1
(encode_prefetch, unchanged: official compute_embedding order -- quantizer,
ME.utils.batched_coordinates, ones features, model(batch)['global'] -- one scan
per forward).  The scratch CLI `eval-official` keeps its AUTHOR_CKPTS
exclusion and OFFICIAL_APPROVED flag; this isolated wrapper does not call or
modify it.  Only S.WORK is pointed at this role's output folder (this process
only) so that the regenerated eval_config.txt lands here; its bytes are
checked against helios_scratch/eval_config.txt.  Instrumentation is read-only:
a forward hook counts forwards and checks one scan per forward, and pass-through
wrappers record the current sequence for progress/ETA.

Subcommands (inside ring_baseline):
  preflight         CPU: sources, frozen hashes, B32 references, VAL inputs and
                    pose-derived queries == B32 ledgers
  eval --which W    GPU0: B1 VAL9 of one frozen model (resumable per sequence),
                    then a 256-frame encoder diagnostic (B1 repeat, B1 vs B32)
  verify            CPU: ledgers -> R@1/family recompute, B1 vs B32, pair deltas
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _helios_scratch_pilot_20261108 as S  # noqa: E402  (imported unchanged)

REPO = S.REPO
SC = REPO / "_external/overnight_repro_20261108/helios_scratch"
OUT = REPO / "_external/overnight_repro_20261108/helios_scratch_official_check"
LABEL = "scratch_seed1_pilot_diagnostic"
ORDER = ["old_e10", "candidate"]
FROZEN = {
    "old_e10": {"candidate": "old_seed1", "epoch": 10,
                "sha256": "3a6424dbd89f835ae708bb819dc804c0a274ad1ce56c58204df676fa0a188258",
                "checkpoint": "/workspace/Neural-Spectral-Codec/_external/helios_official_pdv2/runs/Oct12_12-22-10/HeLiOS_default_trainning_10.pth",
                "b32_json": "old_seed1_e10_b32.json",
                "b32_family_macro_r1": 0.5413847510985155},
    "candidate": {"candidate": "lr5e-4", "epoch": 10,
                  "sha256": "e4a4b47c17dbdc46ca5534fee175d7557e42bc0de02a7ba7ca4ebfaf3817e22e",
                  "checkpoint": "/workspace/Neural-Spectral-Codec/_external/overnight_repro_20261108/"
                                "helios_scratch/lr5e-4/run/HeLiOS_default_trainning_10.pth",
                  "b32_json": "lr5e-4/val/epoch010.json",
                  "b32_family_macro_r1": 0.5535016902149547},
}
AUTHOR_SHA = {"88e1ec050be7f80619f458abb96e19e2e0606d7922279022436f0e0dc1e42515",   # HeLiOS.pth
              "7f73a58eb8f485fe75c2bf3359268676accecc3774e42f38974d9c36aae93675"}  # HeLiOS-S.pth
SOURCES = {
    "scripts/_helios_scratch_pilot_20261108.py": "d5e19ced5caea94a0054404e61cfb013f3f3b065b16da8dc7e32a0bf00fab086",
    "scripts/_helios_scratch_pilot_20261108_finalize.py": "a576194398bb75f86568262af9473b89cd33f4a50c4cd10e9286983d3d87b62c",
    "baselines/eval_utils.py": "c740445ee390385b9908f35e88f0ece644ecceb3e594a29772401f955ab5f085",
    "scripts/_helios_official_dump.py": "f55ae58645348206709b8f9ab746f41803cae2e3bf9596f32e87bf98f3e75020",
    "scripts/_helios_official_prep.py": "0e6d057c072a5421ae6a80fb1d9467e85d7b204b751d932e04b36b28462c6a35",
    "scripts/_helios_official_train.py": "6f9393aeadf08268d19373ad11132add35348a140f7f2a48df6bd15d2d8a46a6",
    "scripts/_pdv2_c3.py": "8f5400ba8a22580b6e866ff2a4e4e4ba9881b42d3b2fb810e4f3d66d7bf31f7e",
    "scripts/_pdv2_pilot.py": "2abd10227db023f9ce7b5a944e65e5a20670b1826e06200e26a65235776625ed",
}
VAL_KEYS = [f"{a}/{b}" for a, b in S.VAL]
DIAG_N = 256
_ORIG_VP, _ORIG_ENC, _ORIG_OM = S.val_paths, S.encode_prefetch, S.D.official_model

assert S.EVAL_MEM_GIB == 4.5 and S.EVAL_BATCH == 32 and S.TR.SEED == 1
# public-weight exclusion (scratch track): no frozen path is an author checkpoint
assert all(Path(f["checkpoint"]) not in S.AUTHOR_CKPTS and f["sha256"] not in AUTHOR_SHA for f in FROZEN.values())


def b1_json(which: str) -> Path:
    f = FROZEN[which]
    return OUT / f["candidate"] / "val" / f"epoch{f['epoch']:03d}.json"


def seq_checks_json(which: str) -> Path:
    f = FROZEN[which]
    return OUT / f["candidate"] / f"b1_seq_checks_e{f['epoch']:03d}.json"


def b32_json(which: str) -> Path:
    return SC / FROZEN[which]["b32_json"]


def fsha(p: Path) -> str:
    return S.P.sha256(Path(p))


def rj(p: Path):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------ preflight (CPU)
def cmd_preflight():
    from baselines.eval_utils import _find_revisit_queries
    rep = {"time": S.now(), "label": LABEL, "order": ORDER, "frozen": FROZEN, "checks": {}}
    c = rep["checks"]
    c["sources"] = {k: fsha(REPO / k) == v for k, v in SOURCES.items()}
    c["frozen_ckpt_sha"] = {w: fsha(Path(f["checkpoint"])) == f["sha256"] for w, f in FROZEN.items()}
    c["author_ckpts_excluded"] = {str(a): fsha(a) not in {f["sha256"] for f in FROZEN.values()}
                                  for a in S.AUTHOR_CKPTS if a.exists()}
    c["author_sha_constants"] = {str(a): fsha(a) in AUTHOR_SHA for a in S.AUTHOR_CKPTS if a.exists()}
    c["b32_reference"] = {}
    for w, f in FROZEN.items():
        r = rj(b32_json(w))
        c["b32_reference"][w] = (r.get("finished") is not None and r["encode_batch"] == 32
                                 and r["sha256"] == f["sha256"] and r["checkpoint"] == f["checkpoint"]
                                 and r["family_macro_r1"] == f["b32_family_macro_r1"]
                                 and list(r["per_seq"]) == VAL_KEYS)
    inputs, n_total = {}, 0
    for sensor, seq in S.VAL:
        k = f"{sensor}/{seq}"
        sids, paths = S.val_paths(sensor, seq)
        poses = np.load(S.P.cache_path("val", sensor, seq))["poses"].astype(np.float64)
        q = np.array([a for a, _ in _find_revisit_queries(poses[:, :3, 3], S.D.RADIUS, S.D.SKIP)], np.int64)
        missing = sum(not os.path.exists(p) for p in paths)
        led_eq = {}
        for w in ORDER:
            b = rj(b32_json(w))["per_seq"][k]
            z = np.load(REPO / b["ledger"])
            led_eq[w] = bool(np.array_equal(z["query_idx"], q)) and b["n_frames"] == len(sids)
        h = hashlib.sha256("\n".join(paths).encode()).hexdigest()
        inputs[k] = {"n_frames": int(len(sids)), "n_poses": int(len(poses)), "n_queries": int(len(q)),
                     "missing_scan_files": int(missing), "paths_sha256": h,
                     "queries_equal_b32_ledgers": led_eq}
        n_total += len(sids)
    rep["inputs"] = inputs
    rep["val_frames_total"] = int(n_total)
    rep["val_queries_total"] = int(sum(v["n_queries"] for v in inputs.values()))
    c["inputs_ok"] = all(v["n_frames"] == v["n_poses"] and v["missing_scan_files"] == 0
                         and all(v["queries_equal_b32_ledgers"].values()) for v in inputs.values())
    rep["all_ok"] = bool(all(c["sources"].values()) and all(c["frozen_ckpt_sha"].values())
                         and all(c["author_ckpts_excluded"].values()) and all(c["author_sha_constants"].values())
                         and all(c["b32_reference"].values()) and c["inputs_ok"]
                         and rep["val_queries_total"] == 10084 and n_total == 99496)
    S.wjson(OUT / "preflight.json", rep)
    print(json.dumps({"all_ok": rep["all_ok"], "checks": c, "frames": n_total,
                      "queries": rep["val_queries_total"]}, indent=1), flush=True)
    if not rep["all_ok"]:
        sys.exit(2)


# ------------------------------------------------------------ eval (GPU0)
PROG: dict = {}
ENV: dict = {}


def _write_progress(which: str, force=False):
    t = time.time()
    if not force and t - PROG.get("last_write", 0) < 60:
        return
    PROG["last_write"] = t
    done_frames = PROG["frames_done_before"] + PROG.get("seq_forwards", 0)
    warm = PROG.get("warm")
    rate = None
    if warm and PROG["proc_forwards"] > warm[0]:
        rate = (PROG["proc_forwards"] - warm[0]) / max(t - warm[1], 1e-9)
    rem = PROG["frames_total"] - done_frames
    S.wjson(OUT / f"progress_{which}.json", {
        "time": S.now(), "which": which, "pid": os.getpid(), "seq": PROG.get("seq"),
        "seq_frames": PROG.get("seq_frames"), "seq_forwards": PROG.get("seq_forwards", 0),
        "frames_done_model": int(done_frames), "frames_total_model": PROG["frames_total"],
        "process_forwards": PROG["proc_forwards"], "process_elapsed_s": round(t - PROG["t0"], 1),
        "rate_fps_after_warmup": None if rate is None else round(rate, 3),
        "eta_model_s": None if not rate else round(rem / rate),
        "max_scans_per_forward_seq": PROG.get("seq_max_scans", 0),
        "cuda_peak_gib": round(__import__("torch").cuda.max_memory_allocated() / 2 ** 30, 3)})


def _done_seqs(which: str) -> dict:
    p = b1_json(which)
    if not p.exists():
        return {}
    r = rj(p)
    return r["per_seq"] if r.get("sha256") == FROZEN[which]["sha256"] and r.get("encode_batch") == 1 else {}


def _instrument(which: str):
    """Read-only progress/ETA and per-sequence B1 checks (forwards == frames,
    one scan per forward).  Inputs, model and outputs are not changed."""
    import torch

    def val_paths(sensor, seq):
        PROG["seq"] = f"{sensor}/{seq}"
        return _ORIG_VP(sensor, seq)

    def encode_prefetch(model, params, paths, device, workers=1, batch=1):
        PROG.update(seq_frames=len(paths), seq_forwards=0, seq_max_scans=0)
        PROG["frames_done_before"] = sum(v["n_frames"] for v in _done_seqs(which).values())
        _write_progress(which, force=True)
        t0 = time.time()
        desc = _ORIG_ENC(model, params, paths, device, workers, batch)
        p = seq_checks_json(which)
        chk = rj(p) if p.exists() else {"seqs": {}, "env_by_pid": {}}
        chk["seqs"][PROG["seq"]] = {"n_frames": len(paths), "forwards": PROG["seq_forwards"],
                                    "max_scans_per_forward": PROG["seq_max_scans"], "batch": batch,
                                    "workers": workers, "desc_shape": list(desc.shape),
                                    "desc_dtype": str(desc.dtype), "desc_finite": bool(np.isfinite(desc).all()),
                                    "encode_s": round(time.time() - t0, 1), "pid": os.getpid(), "time": S.now()}
        chk["env_by_pid"][str(os.getpid())] = ENV
        S.wjson(p, chk)
        return desc

    def official_model(E, params, checkpoint, device):
        m = _ORIG_OM(E, params, checkpoint, device)
        ENV.update(param_dtypes=sorted({str(p.dtype) for p in m.parameters()}),
                   param_devices=sorted({str(p.device) for p in m.parameters()}),
                   model_class=type(m).__name__)

        def hook(mod, inp, out):
            PROG["proc_forwards"] += 1
            PROG["seq_forwards"] = PROG.get("seq_forwards", 0) + 1
            if PROG["proc_forwards"] == 50:
                PROG["warm"] = (50, time.time())
            coords = inp[0]["coords"]
            ns = int(coords[:, 0].max().item()) + 1 if len(coords) else 0
            PROG["seq_max_scans"] = max(PROG.get("seq_max_scans", 0), ns)
            if "first_forward" not in ENV:
                ENV["first_forward"] = {"training": bool(mod.training), "global_dtype": str(out["global"].dtype),
                                        "global_shape": list(out["global"].shape),
                                        "coords_device": str(coords.device),
                                        "features_dtype": str(inp[0]["features"].dtype)}
            _write_progress(which)
        m.register_forward_hook(hook)
        return m

    S.val_paths, S.encode_prefetch, S.D.official_model = val_paths, encode_prefetch, official_model
    PROG.update(t0=time.time(), proc_forwards=0, frames_total=99496, frames_done_before=0)
    ENV.update(torch=torch.__version__, cuda=torch.version.cuda, device_name=torch.cuda.get_device_name(0),
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), torch_threads=torch.get_num_threads(),
               cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
               pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), pid=os.getpid())
    try:
        import MinkowskiEngine as ME
        ENV["minkowski_engine"] = ME.__version__
    except Exception as e:  # noqa: BLE001
        ENV["minkowski_engine"] = repr(e)


def _restore():
    S.val_paths, S.encode_prefetch, S.D.official_model = _ORIG_VP, _ORIG_ENC, _ORIG_OM


def _encoder_diag(which: str, ckpt: Path):
    """First DIAG_N kitti/09 frames with the same model: B1 twice and B32 once;
    descriptor max|diff| and min cosine.  Unpatched functions; diagnostic only,
    not a ledger, not an uncertainty bound or significance threshold."""
    import torch
    out = OUT / FROZEN[which]["candidate"] / f"encoder_diag_e{FROZEN[which]['epoch']:03d}.json"
    if out.exists():
        return
    _restore()
    E, params = S.D._official(OUT / "eval_config.txt")
    model = _ORIG_OM(E, params, ckpt, "cuda")
    _, paths = _ORIG_VP("kitti", "09")
    paths = paths[:DIAG_N]
    a = _ORIG_ENC(model, params, paths, "cuda", 1, 1)
    b = _ORIG_ENC(model, params, paths, "cuda", 1, 1)
    c = _ORIG_ENC(model, params, paths, "cuda", 1, 32)

    def cmp(x, y):
        cos = np.sum(x * y, 1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        return {"max_abs": float(np.abs(x - y).max()), "min_cosine": float(cos.min()),
                "bitwise_equal": bool(np.array_equal(x, y))}
    S.wjson(out, {"time": S.now(), "which": which, "n": len(paths), "sequence": "kitti/09",
                  "checkpoint_sha256": fsha(ckpt), "b1_repeat": cmp(a, b), "b1_vs_b32": cmp(a, c),
                  "desc_norm_b1_minmax": [float(np.linalg.norm(a, axis=1).min()),
                                          float(np.linalg.norm(a, axis=1).max())],
                  "note": "encoder numerical diagnostic only; not an uncertainty bound or significance threshold"})
    del model
    torch.cuda.empty_cache()


def cmd_eval(which: str, workers: int):
    import torch
    assert which in ORDER
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and torch.cuda.is_available()
    torch.set_num_threads(2)
    S.mem_cap(S.EVAL_MEM_GIB)      # also set inside eval_checkpoint; here for the diagnostic-only path
    f = FROZEN[which]
    ckpt = Path(f["checkpoint"])
    assert fsha(ckpt) == f["sha256"], "frozen checkpoint hash mismatch"
    if which == "candidate":       # fixed order: old_e10 must be finished first
        assert rj(b1_json("old_e10")).get("label") == LABEL, "old_e10 B1 not finished"
    S.WORK = OUT                   # eval_config.txt location only (this process)
    out = b1_json(which)
    if out.exists() and rj(out).get("label") == LABEL:
        print(f"already finished {out}", flush=True)
    else:
        _instrument(which)
        rec = S.eval_checkpoint(ckpt, out, device="cuda", workers=workers, batch=1)
        _write_progress(which, force=True)
        _restore()
        assert fsha(ckpt) == f["sha256"], "checkpoint changed during evaluation"
        assert rec["encode_batch"] == 1 and list(rec["per_seq"]) == VAL_KEYS
        chk = rj(seq_checks_json(which))
        seq_ok = {k: (k in chk["seqs"] and chk["seqs"][k]["forwards"] == rec["per_seq"][k]["n_frames"]
                      and chk["seqs"][k]["max_scans_per_forward"] == 1 and chk["seqs"][k]["batch"] == 1
                      and chk["seqs"][k]["desc_finite"]) for k in VAL_KEYS}
        assert all(seq_ok.values()), seq_ok
        ev_cfg = OUT / "eval_config.txt"
        rec.update(label=LABEL, role=which, frozen=f, order=ORDER, workers=workers,
                   env=dict(ENV, cuda_peak_gib_last_process=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                            mem_cap_gib=S.EVAL_MEM_GIB),
                   b1_seq_checks={"file": str(seq_checks_json(which).relative_to(REPO)),
                                  "sha256": fsha(seq_checks_json(which)), "all_ok": True,
                                  "rule": "forwards == n_frames, max one scan per forward, batch 1, finite"},
                   eval_config_sha256=fsha(ev_cfg),
                   eval_config_equals_scratch=fsha(ev_cfg) == fsha(SC / "eval_config.txt"),
                   b32_reference={"json": str(b32_json(which).relative_to(REPO)),
                                  "json_sha256": fsha(b32_json(which)),
                                  "family_macro_r1": f["b32_family_macro_r1"]},
                   note="official single-scan (B1) confirmation; scratch_seed1_pilot_diagnostic; no selection")
        S.wjson(out, rec)
        print(f"== B1 {which}: family R@1 {rec['family_macro_r1']:.6f}", flush=True)
    _encoder_diag(which, ckpt)


# ------------------------------------------------------------ verify (CPU)
FAM = {"kitti": "kitti", "kitti360": "kitti", "helipr": "helipr", "mulran": "mulran"}


def _fam(per: dict):
    acc = {}
    for k, (c, n) in per.items():
        a = acc.setdefault(FAM[k.split("/")[0]], [0, 0])
        a[0] += c
        a[1] += n
    return float(np.mean([c / n for c, n in acc.values() if n])), acc


def _load(js: Path):
    r = rj(js)
    led, per, ok = {}, {}, True
    for k in VAL_KEYS:
        v = r["per_seq"][k]
        z = dict(np.load(REPO / v["ledger"]))
        n = len(z["correct"])
        ok &= n == v["n_queries"] and abs((z["correct"].sum() / n if n else 0.0) - v["r1"]) < 1e-12
        ok &= bool(np.all((z["top1_geo_dist_m"] < 5.0) == z["correct"]))
        ok &= bool(np.all(np.isfinite(z["desc_norm"])))
        led[k] = z
        per[k] = (int(z["correct"].sum()), n)
    fam, acc = _fam(per)
    from _pdv2_c3 import _family_pooled
    ok &= abs(fam - r["family_macro_r1"]) < 1e-12 and abs(_family_pooled(r["per_seq"]) - fam) < 1e-12
    return r, led, per, fam, acc, bool(ok)


def cmd_verify():
    res = {"time": S.now(), "label": LABEL, "order": ORDER, "models": {}, "checks": {}}
    ch = res["checks"]
    L = {}
    for w in ORDER:
        f = FROZEN[w]
        r1, l1, p1, f1, a1, ok1 = _load(b1_json(w))
        r32, l32, p32, f32, a32, ok32 = _load(b32_json(w))
        same_q = all(np.array_equal(l1[k]["query_idx"], l32[k]["query_idx"]) for k in VAL_KEYS)
        agree = {k: {"same_top1": int((l1[k]["top1_idx"] == l32[k]["top1_idx"]).sum()),
                     "both_correct": int((l1[k]["correct"] & l32[k]["correct"]).sum()),
                     "b1_only_correct": int((l1[k]["correct"] & ~l32[k]["correct"]).sum()),
                     "b32_only_correct": int((~l1[k]["correct"] & l32[k]["correct"]).sum()),
                     "n": int(len(l1[k]["correct"]))} for k in VAL_KEYS}
        tot = {x: sum(a[x] for a in agree.values()) for x in agree[VAL_KEYS[0]]}
        res["models"][w] = {
            "frozen": f, "b1_json_sha256": fsha(b1_json(w)), "b32_json_sha256": fsha(b32_json(w)),
            "b1_family_macro_r1": f1, "b32_family_macro_r1": f32,
            "b1_minus_b32_pp": 100 * (f1 - f32),
            "b1_families": {k: list(v) for k, v in a1.items()}, "b32_families": {k: list(v) for k, v in a32.items()},
            "b1_per_seq_r1": {k: p1[k][0] / p1[k][1] for k in VAL_KEYS},
            "b32_per_seq_r1": {k: p32[k][0] / p32[k][1] for k in VAL_KEYS},
            "b1_empty_top1": int(sum((l1[k]["top1_idx"] < 0).sum() for k in VAL_KEYS)),
            "query_agreement_b1_vs_b32": tot,
            "b1_encode_s_total": round(sum(v["encode_s"] for v in r1["per_seq"].values()), 1),
            "b1_env": r1.get("env"), "eval_config_equals_scratch": r1.get("eval_config_equals_scratch")}
        ch[f"{w}_b1_ledgers_recomputed"] = ok1
        ch[f"{w}_b32_ledgers_recomputed"] = ok32
        ch[f"{w}_same_queries_b1_b32"] = same_q
        ch[f"{w}_b1_sha_frozen"] = r1["sha256"] == f["sha256"] and r1["encode_batch"] == 1 and \
            r1.get("label") == LABEL and r1.get("finished") is not None
        ch[f"{w}_b32_value_frozen"] = abs(f32 - f["b32_family_macro_r1"]) < 1e-12
        ch[f"{w}_ckpt_sha_now"] = fsha(Path(f["checkpoint"])) == f["sha256"]
        # one scan per forward: read from the per-sequence forward-hook record (env has no such key)
        sc = rj(seq_checks_json(w)).get("seqs", {})
        ch[f"{w}_one_scan_per_forward"] = bool(
            (r1.get("b1_seq_checks") or {}).get("sha256") == fsha(seq_checks_json(w))
            and all(k in sc and sc[k]["max_scans_per_forward"] == 1 and sc[k]["batch"] == 1
                    and sc[k]["forwards"] == r1["per_seq"][k]["n_frames"] for k in VAL_KEYS))
        ch[f"{w}_eval_config_equals_scratch"] = bool(r1.get("eval_config_equals_scratch"))
        L[w] = (l1, a1, f1, a32, f32)
    la, aa, fa, aa32, fa32 = L["old_e10"]
    lc, ac, fc, ac32, fc32 = L["candidate"]
    ch["same_queries_pair"] = all(np.array_equal(la[k]["query_idx"], lc[k]["query_idx"]) for k in VAL_KEYS)
    famd = lambda x, y: {k: 100 * (x[k][0] / x[k][1] - y[k][0] / y[k][1]) for k in x}  # noqa: E731
    pair_q = {"cand_only_correct": int(sum((lc[k]["correct"] & ~la[k]["correct"]).sum() for k in VAL_KEYS)),
              "old_e10_only_correct": int(sum((~lc[k]["correct"] & la[k]["correct"]).sum() for k in VAL_KEYS)),
              "both_correct": int(sum((lc[k]["correct"] & la[k]["correct"]).sum() for k in VAL_KEYS)),
              "n": int(sum(len(lc[k]["correct"]) for k in VAL_KEYS))}
    d1, d32 = 100 * (fc - fa), 100 * (fc32 - fa32)
    res["pair"] = {"b1_candidate_minus_old_e10_pp": d1, "b1_family_delta_pp": famd(ac, aa),
                   "b32_candidate_minus_old_e10_pp": d32, "b32_family_delta_pp": famd(ac32, aa32),
                   "b1_sign": "candidate_higher" if d1 > 0 else ("old_e10_higher" if d1 < 0 else "tie"),
                   "b32_sign": "candidate_higher" if d32 > 0 else ("old_e10_higher" if d32 < 0 else "tie"),
                   "sign_same_b1_b32": (d1 > 0) == (d32 > 0) and (d1 < 0) == (d32 < 0),
                   "b1_query_pair_counts": pair_q,
                   "note": "descriptive raw differences on VAL9; no significance threshold; "
                           "frozen pair, no reselection; scratch_seed1_pilot_diagnostic"}
    for w in ORDER:
        dp = OUT / FROZEN[w]["candidate"] / f"encoder_diag_e{FROZEN[w]['epoch']:03d}.json"
        res["models"][w]["encoder_diag"] = rj(dp) if dp.exists() else None
    res["all_ok"] = bool(all(ch.values()))
    S.wjson(OUT / "verification.json", res)
    summ = {"time": res["time"], "label": LABEL, "all_ok": res["all_ok"], "order": ORDER,
            "b1": {w: res["models"][w]["b1_family_macro_r1"] for w in ORDER},
            "b32": {w: res["models"][w]["b32_family_macro_r1"] for w in ORDER},
            "b1_minus_b32_pp": {w: res["models"][w]["b1_minus_b32_pp"] for w in ORDER},
            "pair": {k: res["pair"][k] for k in ("b1_candidate_minus_old_e10_pp", "b1_family_delta_pp",
                                                 "b32_candidate_minus_old_e10_pp", "b1_sign", "b32_sign",
                                                 "sign_same_b1_b32")}}
    S.wjson(OUT / "summary.json", summ)
    print(json.dumps(summ, indent=1), flush=True)
    if not res["all_ok"]:
        sys.exit(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    p = sub.add_parser("eval")
    p.add_argument("--which", choices=ORDER, required=True)
    p.add_argument("--workers", type=int, default=1, choices=[1, 2])
    sub.add_parser("verify")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "preflight":
        cmd_preflight()
    elif a.cmd == "eval":
        cmd_eval(a.which, a.workers)
    elif a.cmd == "verify":
        cmd_verify()


if __name__ == "__main__":
    main()
