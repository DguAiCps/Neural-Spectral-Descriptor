#!/usr/bin/env python3
"""RING#-L cross-session evaluation (docker ring_baseline, GPU).

Implements the "RING-style cross-session protocol under a common keyframe
roster" (scripts/_xs_eval_lib.py; NOT the official RING roster): map session
at 20 m spacing, query session at 5 m, one fixed query roster reused at
every threshold. Reuses, without modifying, scripts/_pdb_ringsharp_eval.py
(make_pd_loader raw-scan loader factory, compute_seq_features feature
extraction, base.load_model with the e2cnn filter-rebuild sanity check, and
the official rerank pieces base.estimate_yaw / base.rotate_bev_batch /
base.solve_translation) plus scripts/_pdb_ringsharp_pd_eval.sha256_of.
Poses come ONLY from the canonical global-GT source inside lib.make_pair
(results/pd2026/keys_pd_s1_{sensor}_{seq}_gp.npz); loader / cache / method
artifacts never contribute poses.

WEIGHT CONVENTION (pd-retrain, mirrors scripts/_pdb_ringsharp_pd_eval.py):
the checkpoint is the pd2026 place-disjoint RING#-L retrain selected by the
pre-declared pd-validation rule ("highest pd-validation sequence-macro 5 m
Recall@1 over epochs 1-5; earlier epoch on ties", declared before any test
contact), frozen in results/pd2026_baselines/ringsharp_pd_ckpt_selection.json
(selected_weight + weight_sha256). The weight file's sha256 is re-verified
against the frozen value before loading, and both are echoed into every
result payload.

Per protocol: features (official 160D 'global', (1,160,160) 'spec' and the
trans_cnn feature map) are computed for the spacing-sliced frames ONLY
(features are per-frame; the sliced scan_ids fed to the loader are asserted
equal to pair db_scan_ids / q_scan_ids) and cached per session as
results/pd2026_xsession/xs_rsfeat_{sensor}_{seq}.npz, keyed by the spaced
scan_ids + frozen weight sha. The spacing-derived scan_ids depend only on
the full session + frozen constants — roster-independent, so the two
protocols per site share the db-session cache. Coarse = cosine over
'global' via lib.rank_cross_session (DB-only, n_coarse = min(200, n_db));
rerank = official rotation-branch circular correlation over the
concatenated [db; query] spec rows (query row = n_db + roster index,
candidates always < n_db) followed by the official translation branch
re-ordering the top N_TRANS = 20 (same documented deviation from the
official n_rerank=1000 as scripts/_ringsharp_eval.py).

Outputs: results/pd2026_xsession/xs_ringsharp_{protocol}.json

Run inside docker ring_baseline (repo at /workspace/Neural-Spectral-Codec):
  python3 scripts/_xs_ringsharp_eval.py [--protocols p1 p2 ...] [--overwrite]
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
sys.path.insert(0, str(REPO / "scripts"))

import _xs_eval_lib as lib                                       # noqa: E402
import _pdb_ringsharp_eval as pdrs  # loader factory, feature     # noqa: E402
                                    # extraction, base.load_model + official
                                    # estimate_yaw/rotate_bev_batch/
                                    # solve_translation
from _pdb_ringsharp_pd_eval import sha256_of                     # noqa: E402

SELECTION = (REPO / "results" / "pd2026_baselines"
             / "ringsharp_pd_ckpt_selection.json")
N_TRANS = 20  # translation-branch re-order depth (as _pdb_ringsharp_eval)
BATCH = 8     # feature-extraction batch (as _pdb_ringsharp_eval)


def resolve_weight(selected_weight: str) -> Path:
    """The selection JSON freezes the docker path; remap onto this checkout
    when the repo is mounted elsewhere."""
    p = Path(selected_weight)
    docker_prefix = "/workspace/Neural-Spectral-Codec/"
    if not p.exists() and selected_weight.startswith(docker_prefix):
        p = REPO / selected_weight[len(docker_prefix):]
    if not p.exists():
        raise FileNotFoundError(
            f"selected pd2026 RING#-L weight not found: {selected_weight}")
    return p


def session_features(pair: dict, side: str, device: str, get_model,
                     frozen_sha: str):
    """RING#-L features for one side's spaced frames, cached per session
    (keyed by the spacing-derived scan_ids + the frozen weight sha).
    Returns (glob f32, specs f16, bts f16, cache_path)."""
    sensor = pair["sensor"]
    seq = pair["db_seq"] if side == "db" else pair["query_seq"]
    idx = pair["db_idx"] if side == "db" else pair["q_idx"]
    session_sids = (pair["db_session_scan_ids"] if side == "db"
                    else pair["q_session_scan_ids"])
    spaced_sids = pair["db_scan_ids"] if side == "db" else pair["q_scan_ids"]
    spacing = lib.CONST["db_spacing_m" if side == "db" else "query_spacing_m"]
    # R4 identity: the sliced scan_ids fed to the loader must be exactly the
    # canonical session scan_ids under the spacing index set.
    assert np.array_equal(session_sids[idx], spaced_sids), (sensor, seq, side)

    cpath = lib.OUT_DIR / f"xs_rsfeat_{sensor}_{seq}.npz"
    if cpath.exists():
        with np.load(cpath) as z:
            if (np.array_equal(z["scan_ids"], spaced_sids)
                    and str(z["weight_sha256"]) == frozen_sha):
                print(f"  feat cache hit: {cpath.name}", flush=True)
                return (z["glob"].copy(), z["specs"].copy(),
                        z["bts"].copy(), str(cpath))
        print(f"  feat cache {cpath.name} stale (scan_ids/weight-sha "
              f"mismatch), rebuilding", flush=True)

    model, trans_cnn = get_model()
    loader = pdrs.make_pd_loader(sensor, seq, pdrs.DATA_ROOT)
    t0 = time.time()
    glob_desc, specs_t, bevs_packed, bts_t = pdrs.compute_seq_features(
        model, trans_cnn, loader, f"{sensor}_{seq}", spaced_sids, device,
        BATCH)
    del loader, bevs_packed  # rerank_fn uses specs + bts only
    glob_desc = glob_desc.astype(np.float32)
    specs = specs_t.numpy()  # fp16 (n, 1, 160, 160)
    bts = bts_t.numpy()      # fp16 trans_cnn feature maps
    del specs_t, bts_t
    print(f"  [{sensor}_{seq}] features for {len(spaced_sids)} spaced frames "
          f"({time.time() - t0:.0f}s)", flush=True)
    lib.OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = lib.OUT_DIR / f".tmp_xs_rsfeat_{sensor}_{seq}.npz"
    np.savez_compressed(tmp, scan_ids=spaced_sids, spacing_m=float(spacing),
                        weight_sha256=frozen_sha, glob=glob_desc,
                        specs=specs, bts=bts)
    os.replace(tmp, cpath)
    return glob_desc, specs, bts, str(cpath)


def make_rerank(specs_cat: torch.Tensor, bts_cat: torch.Tensor, n_db: int,
                device: str, rr: dict):
    """Official RING#-L rerank over [db; query] rows; verbatim semantics of
    _pdb_ringsharp_eval.evaluate_sequence.rerank_fn (rotation-branch circular
    correlation, then translation branch re-ordering the top N_TRANS)."""

    def rerank_fn(qi, cand):
        tq = time.time()
        cand = np.asarray(cand)
        assert np.all(cand < n_db)  # DB-only candidates
        qspec = specs_cat[qi].float().to(device)
        cspec = specs_cat[cand].float().to(device)
        with torch.no_grad():
            _, scores, angles = pdrs.base.estimate_yaw(
                qspec.unsqueeze(0).expand(len(cand), -1, -1, -1), cspec)
        rot_dist = (1 - scores).cpu().numpy().reshape(-1)
        angles = (angles * 2 * np.pi
                  / specs_cat.shape[-2]).cpu().numpy().reshape(-1)
        order = np.argsort(rot_dist)
        m = min(N_TRANS, len(cand))
        top = order[:m]
        with torch.no_grad():
            bt = bts_cat[np.concatenate(([qi], cand[top]))].to(device).float()
            qbt, cbt = bt[:1], bt[1:]
            ang = torch.from_numpy(angles[top]).float().to(device)
            qrot = pdrs.base.rotate_bev_batch(qbt, ang)
            qrot_e = pdrs.base.rotate_bev_batch(qbt, ang - np.pi)
            _, _, e1, _ = pdrs.base.solve_translation(qrot, cbt)
            _, _, e2, _ = pdrs.base.solve_translation(qrot_e, cbt)
            errs = np.minimum(e1.cpu().numpy().reshape(-1),
                              e2.cpu().numpy().reshape(-1))
        order[:m] = top[np.argsort(errs)]
        rr["s"] += time.time() - tq
        rr["n"] += 1
        return cand[order]

    return rerank_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocols", nargs="*", default=list(lib.PROTOCOLS))
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run protocols whose output JSON exists (feature "
                         "caches are reused; delete the npz to re-encode)")
    args = ap.parse_args()
    unknown = [p for p in args.protocols if p not in lib.PROTOCOLS]
    if unknown:
        raise SystemExit(f"unknown protocols: {unknown} "
                         f"(known: {sorted(lib.PROTOCOLS)})")

    sel = json.loads(SELECTION.read_text())
    frozen_sha = sel["weight_sha256"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded = {}

    def get_model():
        """Lazy load: cache-hit-only runs never touch the checkpoint. The
        file sha must match the frozen pd-val selection before loading."""
        if "m" not in loaded:
            wp = resolve_weight(sel["selected_weight"])
            sha = sha256_of(str(wp))
            assert sha == frozen_sha, (
                f"weight sha mismatch: {wp} has {sha}, selection froze "
                f"{frozen_sha}")
            print(f"weight={wp}\nsha256={sha} (matches frozen selection)",
                  flush=True)
            loaded["m"] = pdrs.base.load_model(str(wp), device)
        return loaded["m"]

    for protocol in args.protocols:
        name = f"xs_ringsharp_{protocol}"
        out_p = lib.OUT_DIR / f"{name}.json"
        if out_p.exists() and not args.overwrite:
            print(f"=== {protocol}: {out_p.name} exists, skip", flush=True)
            continue
        pair = lib.make_pair(protocol)
        n_db = len(pair["db_pos"])
        print(f"=== {protocol}: n_db={n_db} n_spaced_q={len(pair['q_pos'])} "
              f"n_roster={len(pair['roster'])} "
              f"roster_sha={pair['roster_sha256'][:12]}", flush=True)
        db_glob, db_specs, db_bts, db_cache = session_features(
            pair, "db", device, get_model, frozen_sha)
        q_glob, q_specs, q_bts, q_cache = session_features(
            pair, "query", device, get_model, frozen_sha)

        specs_cat = torch.from_numpy(np.concatenate([db_specs, q_specs]))
        bts_cat = torch.from_numpy(np.concatenate([db_bts, q_bts]))
        del db_specs, q_specs, db_bts, q_bts
        n_coarse = min(int(lib.CONST["n_coarse_baseline"]), n_db)
        rr = {"s": 0.0, "n": 0}
        t0 = time.time()
        ranked = lib.rank_cross_session(
            db_glob, q_glob, pair["roster"],
            rerank_fn=make_rerank(specs_cat, bts_cat, n_db, device, rr),
            n_coarse=n_coarse)
        scored = lib.score_rankings(ranked, pair["db_pos"], pair["q_pos"],
                                    pair["roster"])
        payload = lib.result_payload(
            "ringsharp", pair, scored,
            extra={
                "weight": sel["selected_weight"],
                "weight_sha256": frozen_sha,
                "ckpt_selection": {
                    "path": str(SELECTION.relative_to(REPO)),
                    "rule": sel["rule"],
                    "selected_epoch": sel["selected_epoch"],
                },
                "coarse": {"metric": "cosine over official 160D RING#-L "
                                     "'global' descriptor",
                           "n_coarse": n_coarse},
                "rerank": "official RING#-L rotation branch (estimate_yaw "
                          "circular correlation, dist = 1 - score) + "
                          "translation branch (rotate_bev_batch + "
                          f"solve_translation) re-ordering top {N_TRANS}",
                "n_trans": N_TRANS,
                "bev": {"grid_zyx": [pdrs.base.NSC_Z, pdrs.base.NSC_Y,
                                     pdrs.base.NSC_X],
                        "bounds": list(pdrs.base.NSC_BOUNDS)},
                "feat_cache": {"db": db_cache, "query": q_cache},
                "timing": {"rank_s": round(time.time() - t0, 1),
                           "rerank_ms_per_query":
                               round(1e3 * rr["s"] / max(rr["n"], 1), 1)},
            })
        p = lib.write_result(payload, name)
        m = payload["metrics"]
        print(f"  {name}: R@1@5m={m['R@1@5m']:.4f} "
              f"R@1@20m={m['R@1@20m']:.4f} "
              f"R1pct@20m={m['R1pct@20m']:.4f} -> {p.name}", flush=True)
        del specs_cat, bts_cat, ranked, db_glob, q_glob
        if device == "cuda":
            torch.cuda.empty_cache()
    print("XS_RINGSHARP_DONE", flush=True)


if __name__ == "__main__":
    main()
