#!/usr/bin/env python3
"""RING#-L evaluation on the frozen pd2026 42-sequence TEST roster.

Pipeline per sequence — identical to scripts/_ringsharp_eval.py (official
RINGSharp code end to end): keyframe clouds -> official generate_bev
(160x160x20, +-70 m, z [1,20] m) -> RINGSharpL forward (160D 'global' +
160x160 'spec') -> compute_recall_cosine_then_rerank (revisit < 5 m,
>= 30-keyframe gap, intra-sequence, R@1/5/10) with the official rotation
branch (estimate_yaw) + translation branch (trans_cnn + solve_translation)
rerank. Same --n-coarse 200 / --n-trans 20 defaults and the same documented
deviation from the official n_rerank=1000.

Differences from scripts/_ringsharp_eval.py (which is left untouched):
  * roster/protocol: scan_ids + poses come from the pd2026 test operating
    caches listed in scripts/_pd_test_cache_manifest.json (42 sequences,
    kitti/helipr/mulran/nclt/kitti360/mcd), not the paper-era val caches;
  * kitti360 (src/data/kitti360_loader.py, HDL-64E) and mcd
    (src/data/mcd_loader.py, OS1-64 gen2) are wired in additively;
  * one JSON per sequence: results/pd2026_baselines/ringsharp_<sensor>_<seq>.json
    (one heavy sequence per process — shell out per sequence);
  * per-sequence runtime is recorded (feature s/keyframe, retrieval ms/query).

WEIGHT CONVENTION (documented, mirrors the corrected-NCLT lower-bound
convention of commit aed302d):
  * RING# ships NO official pretrained checkpoints (the TRO'25 release is
    code-only; see _external/ringsharp_env/repro_nclt_official.sh). The
    de-facto official weights in this repo are the corrected-data retrain
    _external/RINGSharp/tools/results/weights/ring_sharp_l_nsc_fix8/
    model_ring_sharp_l_nsc_20260728_0247_final.pth — the checkpoint behind
    the committed paper-era results/ringsharp_eval_fix8.json.
  * They are used ZERO-SHOT here: no retraining on the pd2026 train split.
    As in commit aed302d, the train/eval mismatch is documented instead of
    retrained away.
  * That checkpoint was trained on the paper-era 24-sequence split
    (_external/ringsharp_data/manifest.json), which overlaps 10 pd TEST
    sequences: helipr Town02/KAIST04/DCC04/Riverside04 and mulran
    DCC01/DCC02/KAIST01/KAIST02/Riverside01/Riverside02. On those, recall is
    train-contaminated (optimistic upper bound) and the result JSON carries
    "train_overlap": true. The other 32 sequences are strictly zero-shot.

Verification mode: --paper-cache resolves dtype/seq against the paper-era
val caches (data/preprocessed/cache_{056e0a02,33919e6e}_<dtype>_val_<seq>.npz)
so the port can be checked against the committed results/ringsharp_eval_fix8.json
numbers; output goes to _verify_ringsharp_paper_<dtype>_<seq>.json.

Examples (inside the ring_baseline container):
  python scripts/_pdb_ringsharp_eval.py --paper-cache kitti/05   # verify
  python scripts/_pdb_ringsharp_eval.py mcd/tuhh_night_09        # roster
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("NSD_DATA_ROOT", "/workspace/data")
sys.path.insert(0, str(REPO / "scripts"))

# Single-sources the glnet imports, the e2cnn filter-rebuild load pitfall and
# the paper-era cache conventions.
import _ringsharp_eval as base                                   # noqa: E402
from baselines.eval_utils import compute_recall_cosine_then_rerank  # noqa: E402

MANIFEST = REPO / "scripts" / "_pd_test_cache_manifest.json"
OUT_DIR_DEFAULT = REPO / "results" / "pd2026_baselines"
DEFAULT_WEIGHT = (REPO / "_external/RINGSharp/tools/results/weights/"
                  "ring_sharp_l_nsc_fix8/model_ring_sharp_l_nsc_20260728_0247_final.pth")

# Paper-era RING#-L training split (_external/ringsharp_data/manifest.json
# train_sequences) intersected with the pd TEST roster: contaminated sequences.
TRAIN_OVERLAP = {
    ("helipr", "Town02"), ("helipr", "KAIST04"), ("helipr", "DCC04"),
    ("helipr", "Riverside04"),
    ("mulran", "DCC01"), ("mulran", "DCC02"), ("mulran", "KAIST01"),
    ("mulran", "KAIST02"), ("mulran", "Riverside01"), ("mulran", "Riverside02"),
}


def load_manifest():
    m = json.loads(MANIFEST.read_text())
    return {(e["sensor"], e["sequence"]): REPO / e["expected"]["operating_cache"]
            for e in m["sequences"]}


def make_pd_loader(sensor: str, seq: str, data_root: str):
    """Loader factory: base.make_nsc_loader + kitti360/mcd + helipr layout fix."""
    sys.path.insert(0, str(REPO / "src"))
    if sensor == "kitti360":
        from data.kitti360_loader import Kitti360Loader
        return Kitti360Loader(os.path.join(data_root, "kitti360"), seq, lazy_load=True)
    if sensor == "mcd":
        from data.mcd_loader import MCDLoader
        return MCDLoader(os.path.join(data_root, "mcd_extracted"), seq, lazy_load=True)
    if sensor == "helipr":
        from data.helipr_loader import HeLiPRLoader
        p = os.path.join(data_root, "helipr", seq)
        if not os.path.isdir(os.path.join(p, "LiDAR")):
            p = os.path.join(p, seq)  # legacy double-nested layout
        return HeLiPRLoader(p, lazy_load=True)
    return base.make_nsc_loader(sensor, seq, data_root)


def compute_seq_features(model, trans_cnn, loader, label, scan_ids, device, batch):
    """base.compute_seq_features with an injected loader (kitti360/mcd support)."""
    n = len(scan_ids)
    glob, specs, bevs_packed, bts = [], [], [], []
    t0 = time.time()
    for i0 in range(0, n, batch):
        idx = range(i0, min(i0 + batch, n))
        bevs = []
        for i in idx:
            pts = np.ascontiguousarray(
                loader[int(scan_ids[i])]["points"][:, :3], dtype=np.float64)
            bevs.append(base.generate_bev(
                pts, Z=base.NSC_Z, Y=base.NSC_Y, X=base.NSC_X, bounds=base.NSC_BOUNDS))
        bev = torch.stack(bevs).to(device)
        with torch.no_grad():
            out = model({"pc": bev, "img": None})
            bts.append(trans_cnn(out["bev"]).half().cpu())
        glob.append(out["global"].float().cpu().numpy())
        specs.append(out["spec"].half().cpu())
        bevs_packed.append(np.packbits(
            (bev.cpu().numpy() > 0).reshape(len(bevs), -1), axis=1))
        if (i0 // batch) % 20 == 0:
            print(f"    [{label}] {i0 + len(bevs)}/{n} ({time.time() - t0:.0f}s)",
                  flush=True)
    return (np.concatenate(glob), torch.cat(specs), np.concatenate(bevs_packed),
            torch.cat(bts))


def evaluate_sequence(model, trans_cnn, sensor, seq, scan_ids, poses, device, args):
    label = f"{sensor}_{seq}"
    loader = make_pd_loader(sensor, seq, DATA_ROOT)
    t_feat0 = time.time()
    glob, specs, bevs_packed, bts = compute_seq_features(
        model, trans_cnn, loader, label, scan_ids, device, args.batch)
    t_feat = time.time() - t_feat0
    del loader
    n = len(glob)
    print(f"    [{label}] features: {t_feat:.0f}s ({t_feat / n:.3f} s/kf), "
          f"trans cache {bts.element_size() * bts.nelement() / 1e9:.2f} GB",
          flush=True)
    n_coarse = n if args.n_coarse < 0 else min(args.n_coarse, n)

    def rerank_fn(qi, cand):
        # Official rotation-branch circular correlation + translation branch
        # re-ordering of the top --n-trans; verbatim from _ringsharp_eval.main.
        cand = np.asarray(cand)
        qspec = specs[qi].float().to(device)
        cspec = specs[cand].float().to(device)
        with torch.no_grad():
            _, scores, angles = base.estimate_yaw(
                qspec.unsqueeze(0).expand(len(cand), -1, -1, -1), cspec)
        rot_dist = (1 - scores).cpu().numpy().reshape(-1)
        angles = (angles * 2 * np.pi / specs.shape[-2]).cpu().numpy().reshape(-1)
        order = np.argsort(rot_dist)
        m = min(args.n_trans, len(cand))
        top = order[:m]
        with torch.no_grad():
            bt = bts[np.concatenate(([qi], cand[top]))].to(device).float()
            qbt, cbt = bt[:1], bt[1:]
            ang = torch.from_numpy(angles[top]).float().to(device)
            qrot = base.rotate_bev_batch(qbt, ang)
            qrot_e = base.rotate_bev_batch(qbt, ang - np.pi)
            _, _, e1, _ = base.solve_translation(qrot, cbt)
            _, _, e2, _ = base.solve_translation(qrot_e, cbt)
            errs = np.minimum(e1.cpu().numpy().reshape(-1),
                              e2.cpu().numpy().reshape(-1))
        order[:m] = top[np.argsort(errs)]
        return cand[order]

    t0 = time.time()
    recalls, n_q = compute_recall_cosine_then_rerank(
        glob, rerank_fn, poses, k_values=base.K_VALUES, n_coarse=n_coarse)
    t_eval = time.time() - t0
    print(f"  [{label}] queries={n_q} "
          + " ".join(f"R@{k}={recalls[k]:.4f}" for k in base.K_VALUES)
          + f" ({t_eval:.0f}s, {1000 * t_eval / max(n_q, 1):.0f} ms/query)",
          flush=True)
    return {
        "n_frames": n,
        "n_queries": n_q,
        **{f"recall@{k}": recalls[k] for k in base.K_VALUES},
        "timing": {
            "feature_s_total": round(t_feat, 1),
            "feature_s_per_kf": round(t_feat / n, 4),
            "retrieval_s_total": round(t_eval, 1),
            "retrieval_ms_per_query": round(1000 * t_eval / max(n_q, 1), 1),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sequences", nargs="+", help="sensor/seq from the 42-seq "
                    "roster (one heavy sequence per process)")
    ap.add_argument("--weight", default=str(DEFAULT_WEIGHT))
    ap.add_argument("--n-coarse", type=int, default=200,
                    help="-1 = full database (official exhaustive ranking)")
    ap.add_argument("--n-trans", type=int, default=20,
                    help="candidates re-ordered by the translation branch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0, help="smoke only")
    ap.add_argument("--paper-cache", action="store_true",
                    help="VERIFY mode: resolve dtype/seq against the paper-era "
                         "val caches to compare with results/ringsharp_eval_fix8.json")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roster = load_manifest()
    targets = [tuple(s.split("/", 1)) for s in args.sequences]
    if not args.paper_cache:
        for t in targets:
            assert t in roster, f"not in the frozen 42-seq roster: {t[0]}/{t[1]}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, trans_cnn = base.load_model(args.weight, device)

    for sensor, seq in targets:
        if args.paper_cache:
            scan_ids, poses = base.load_cache(sensor, seq)
            out = out_dir / f"_verify_ringsharp_paper_{sensor}_{seq}.json"
        else:
            with np.load(roster[(sensor, seq)], allow_pickle=True) as c:
                scan_ids = c["scan_ids"].astype(np.int64)
                poses = c["poses"].astype(np.float64)
            out = out_dir / f"ringsharp_{sensor}_{seq}.json"
        if args.max_frames:
            scan_ids, poses = scan_ids[:args.max_frames], poses[:args.max_frames]
        print(f"[{sensor}_{seq}] {len(scan_ids)} keyframes "
              f"({'paper-era val cache' if args.paper_cache else 'pd2026 test cache'})",
              flush=True)
        res = evaluate_sequence(model, trans_cnn, sensor, seq, scan_ids, poses,
                                device, args)
        payload = {
            f"{sensor}_{seq}": res,
            "_meta": {
                "weight": args.weight,
                "weight_convention": (
                    "no official RING#-L checkpoints exist; repo corrected-data "
                    "retrain (paper-era 24-seq split) used ZERO-SHOT on the pd "
                    "roster, train/test overlap documented per sequence "
                    "(aed302d lower-bound convention)"),
                "train_overlap": (sensor, seq) in TRAIN_OVERLAP,
                "zero_shot_wrt_pd_train": True,
                "cache": ("paper-era val" if args.paper_cache
                          else str(roster.get((sensor, seq), "")).replace(str(REPO) + "/", "")),
                "n_coarse": args.n_coarse,
                "n_trans": args.n_trans,
                "max_frames": args.max_frames,
                "protocol": "revisit<5m, skip 30 frames, intra-sequence",
            },
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
