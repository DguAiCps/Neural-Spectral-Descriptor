#!/usr/bin/env python3
"""Evaluate a trained RING#-L checkpoint on OUR val split (intra-seq R@1/5/10).

Pipeline per val sequence (official RINGSharp code end to end):
  1. keyframe point clouds from our caches/loaders -> official generate_bev
     (grid 160x160x20, +-70 m, z-band [1,20] m — same as training),
  2. official RINGSharpL forward -> 160-D rotation-invariant 'global'
     descriptor + (1,160,160) 'spec' (rotation-equivariant sinogram spectrum),
  3. retrieval scored with baselines/eval_utils.compute_recall_cosine_then_rerank
     (query = revisit < 5 m with >= 30-keyframe gap — identical protocol to
     every other baseline in this repo):
       coarse    : cosine over 'global' (n_coarse candidates; --n-coarse -1
                   uses the full database = official exhaustive ranking),
       rerank    : official rotation-branch circular correlation of specs
                   (glnet estimate_yaw), then official translation branch
                   (trans_cnn + solve_translation) re-orders the top
                   --n-trans candidates by translation correlation error —
                   this is RINGSharp's own two-stage PR-by-PE scoring
                   (evaluate_ours_gl.py) transplanted onto our protocol.

Protocol deviation (documented): their official evaluation reranks
n_rerank=1000 candidates with the translation branch; intra-sequence
databases here are up to ~7.5k frames and per-candidate translation solving
is O(C*H*W log HW), so we default to --n-trans 20 (covers R@1/5/10) and
--n-coarse 200. Set --n-coarse -1 --n-trans 100 for a fidelity check on a
small sequence.

Writes results/ringsharp_eval.json. Example (inside docker):
  python scripts/_ringsharp_eval.py --weight <model_..._final.pth>
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RINGSHARP = REPO / "_external" / "RINGSharp"
sys.path.insert(0, str(RINGSHARP))
sys.path.insert(1, str(REPO))
sys.path.insert(2, str(REPO / "src"))

from glnet.utils.params import ModelParams                        # noqa: E402
from glnet.models.model_factory import model_factory              # noqa: E402
from glnet.models.backbones_2d.unet import last_conv_block        # noqa: E402
from glnet.models.utils import estimate_yaw, rotate_bev_batch     # noqa: E402
from glnet.models.utils import solve_translation                  # noqa: E402
from glnet.datasets.nsc.nsc_dataset import (                      # noqa: E402
    make_nsc_loader, NSC_BOUNDS, NSC_X, NSC_Y, NSC_Z)
from glnet.utils.data_utils.point_clouds import generate_bev      # noqa: E402
from baselines.eval_utils import compute_recall_cosine_then_rerank  # noqa: E402

K_VALUES = [1, 5, 10]
CACHE_KEYS = {"kitti": "056e0a02", "nclt": "056e0a02",
              "helipr": "056e0a02", "mulran": "33919e6e"}
VAL_SEQS = [("kitti", "00"), ("kitti", "05"), ("kitti", "08"),
            ("nclt", "2012-01-08"), ("nclt", "2013-01-10"),
            ("helipr", "Town01"),
            ("mulran", "DCC03"), ("mulran", "KAIST03"), ("mulran", "Riverside03")]


def load_cache(dtype, seq):
    p = REPO / "data" / "preprocessed" / f"cache_{CACHE_KEYS[dtype]}_{dtype}_val_{seq}.npz"
    with np.load(p, allow_pickle=True) as c:
        return c["scan_ids"].astype(np.int64), c["poses"].astype(np.float64)


def load_model(weight_path, device):
    mp = ModelParams(str(RINGSHARP / "glnet/config/ring_sharp_l_nsc.txt"), "nsc", "/tmp")
    model = model_factory(mp).to(device)
    trans_cnn = last_conv_block(mp.feature_dim, mp.feature_dim, bn=False).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    state = {k.replace("module.", "", 1): v for k, v in ckpt["model"].items()}
    # e2cnn pitfall: this e2cnn version registers the expanded `filter` buffer
    # at R2Conv.__init__, but train(True) deletes it — so a train-mode-saved
    # checkpoint has no `filter` keys. Switch to train mode BEFORE the strict
    # load (keys then match exactly), then eval() re-expands the filters from
    # the LOADED weights (verified below).
    model.train()
    model.load_state_dict(state, strict=True)
    trans_cnn.load_state_dict(ckpt["trans_cnn"], strict=True)
    model.eval()
    trans_cnn.eval()
    # sanity: eval-mode (pre-expanded filter) output must equal train-mode
    # (expand-on-the-fly) output for the same input -> filters were rebuilt
    # from the loaded weights, not left at their zero init.
    import os as _os
    _os.environ["NSC_ACT_CKPT"] = "0"
    x = (torch.rand(1, 20, 160, 160) > 0.98).float().to(device)
    with torch.no_grad():
        a = model({"pc": x, "img": None})["global"]
        model.train()
        b = model({"pc": x, "img": None})["global"]
        model.eval()
    assert torch.allclose(a, b, atol=1e-4), \
        f"e2cnn filter rebuild mismatch: max diff {(a-b).abs().max().item():.3e}"
    print(f"e2cnn filter-rebuild sanity OK (max |eval-train| diff "
          f"{(a-b).abs().max().item():.2e})")
    return model, trans_cnn


def compute_seq_features(model, dtype, seq, scan_ids, device, batch, max_frames=None):
    """Returns global desc (n,160) fp32, specs (n,1,160,160) fp16 cpu,
    packed occupancy bevs (n, packed) uint8 cpu."""
    loader = make_nsc_loader(dtype, seq, "/workspace/data")
    n = len(scan_ids) if not max_frames else min(max_frames, len(scan_ids))
    glob, specs, bevs_packed = [], [], []
    t0 = time.time()
    for i0 in range(0, n, batch):
        idx = range(i0, min(i0 + batch, n))
        bevs = []
        for i in idx:
            pts = np.ascontiguousarray(
                loader[int(scan_ids[i])]["points"][:, :3], dtype=np.float64)
            bevs.append(generate_bev(pts, Z=NSC_Z, Y=NSC_Y, X=NSC_X, bounds=NSC_BOUNDS))
        bev = torch.stack(bevs).to(device)
        with torch.no_grad():
            out = model({"pc": bev, "img": None})
        glob.append(out["global"].float().cpu().numpy())
        specs.append(out["spec"].half().cpu())
        bevs_packed.append(np.packbits(
            (bev.cpu().numpy() > 0).reshape(len(bevs), -1), axis=1))
        if (i0 // batch) % 20 == 0:
            print(f"    [{dtype}_{seq}] {i0+len(bevs)}/{n} "
                  f"({(time.time()-t0):.0f}s)", flush=True)
    del loader
    return (np.concatenate(glob), torch.cat(specs), np.concatenate(bevs_packed))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weight", required=True)
    ap.add_argument("sequences", nargs="*", help="dtype/seq (default: all 9 val)")
    ap.add_argument("--n-coarse", type=int, default=200,
                    help="-1 = full database (official exhaustive ranking)")
    ap.add_argument("--n-trans", type=int, default=20,
                    help="candidates re-ordered by the translation branch")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results" / "ringsharp_eval.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, trans_cnn = load_model(args.weight, device)

    targets = VAL_SEQS if not args.sequences else \
        [tuple(s.split("/")) for s in args.sequences]

    results, total_q = {}, 0
    for dtype, seq in targets:
        scan_ids, poses = load_cache(dtype, seq)
        if args.max_frames:
            scan_ids, poses = scan_ids[:args.max_frames], poses[:args.max_frames]
        print(f"[{dtype}_{seq}] {len(scan_ids)} keyframes")
        glob, specs, bevs_packed = compute_seq_features(
            model, dtype, seq, scan_ids, device, args.batch)
        n = len(glob)
        n_coarse = n if args.n_coarse < 0 else min(args.n_coarse, n)
        shape_bev = (NSC_Z, NSC_Y, NSC_X)

        def unpack_bev(idxs):
            b = np.unpackbits(bevs_packed[idxs], axis=1)
            b = b[:, :NSC_Z * NSC_Y * NSC_X].reshape(len(idxs), *shape_bev)
            return torch.from_numpy(b).float().to(device)

        def rerank_fn(qi, cand):
            cand = np.asarray(cand)
            qspec = specs[qi].float().to(device)
            cspec = specs[cand].float().to(device)
            with torch.no_grad():
                _, scores, angles = estimate_yaw(
                    qspec.unsqueeze(0).expand(len(cand), -1, -1, -1), cspec)
            rot_dist = (1 - scores).cpu().numpy().reshape(-1)
            angles = (angles * 2 * np.pi / specs.shape[-2]).cpu().numpy().reshape(-1)
            order = np.argsort(rot_dist)
            # translation branch on the best --n-trans by rotation correlation.
            # bev_trans = trans_cnn(model 128-ch BEV features), exactly as in
            # evaluate_ours_gl.py — recomputed from the cached occupancy BEVs
            # via one model forward over [query] + candidates.
            m = min(args.n_trans, len(cand))
            top = order[:m]
            with torch.no_grad():
                occ = unpack_bev(np.concatenate(([qi], cand[top])))
                feat128 = model({"pc": occ, "img": None})["bev"]
                bt = trans_cnn(feat128)
                qbt, cbt = bt[:1], bt[1:]
                ang = torch.from_numpy(angles[top]).float().to(device)
                # Official semantics: single query (B=1) rotated by the m
                # per-candidate angles -> (m, C, H, W); pass qbt as B=1, not
                # expanded, so rotate_bev_batch returns 4D (expanding to B=m
                # makes it 5D and breaks solve_translation's InstanceNorm2d).
                qrot = rotate_bev_batch(qbt, ang)
                qrot_e = rotate_bev_batch(qbt, ang - np.pi)
                _, _, e1, _ = solve_translation(qrot, cbt)
                _, _, e2, _ = solve_translation(qrot_e, cbt)
                errs = np.minimum(e1.cpu().numpy().reshape(-1),
                                  e2.cpu().numpy().reshape(-1))
            order[:m] = top[np.argsort(errs)]
            return cand[order]

        t0 = time.time()
        recalls, n_q = compute_recall_cosine_then_rerank(
            glob, rerank_fn, poses, k_values=K_VALUES, n_coarse=n_coarse)
        dt = time.time() - t0
        print(f"  [{dtype}_{seq}] queries={n_q} "
              + " ".join(f"R@{k}={recalls[k]:.4f}" for k in K_VALUES)
              + f" ({dt:.0f}s)")
        results[f"{dtype}_{seq}"] = {"n_queries": n_q,
                                     **{f"recall@{k}": recalls[k] for k in K_VALUES}}
        total_q += n_q

    if total_q:
        mean = {f"recall@{k}": sum(r[f"recall@{k}"] * r["n_queries"]
                                   for r in results.values()) / total_q
                for k in K_VALUES}
        results["_query_weighted_mean"] = {"n_queries": total_q, **mean}
        print("query-weighted mean: "
              + " ".join(f"R@{k}={mean[f'recall@{k}']:.4f}" for k in K_VALUES))

    results["_meta"] = {"weight": args.weight, "n_coarse": args.n_coarse,
                        "n_trans": args.n_trans,
                        "protocol": "revisit<5m, skip 30 frames, intra-sequence"}
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
