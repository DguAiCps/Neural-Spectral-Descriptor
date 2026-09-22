#!/usr/bin/env python3
"""SeqOT cross-session evaluation (stage 1 GPU descriptors, stage 2 host-capable).

Implements the "RING-style cross-session protocol under a common keyframe
roster" (scripts/_xs_eval_lib.py) — our common-roster protocol, NOT the
official RING evaluation roster.

Reused modules (all left untouched):
  * scripts/_seqot_eval.py — the SeqOT descriptor pipeline: featureExtracter
    (SEQL=3 clamped window) -> trained GeM over GEM_SEQLEN=20 sub-descriptor
    windows, via load_range_stack / compute_final_descriptors; checkpoint
    loading identical to scripts/_pdb_seqot_eval.py.
  * scripts/_pdb_seqot_prepare.py output layout —
    _external/seqot_data_pd2026/{sensor}_{seq}/ (NNNNNN.npy + meta.json);
    scan_ids come from the same pd2026 operating cache the prep used
    (resolved through scripts/_pd_test_cache_manifest.json and cross-checked
    against meta.json["cache"]).
  * scripts/_xs_eval_lib.py — pair construction, DB-only cosine ranking,
    fixed-roster scoring, result schema.

Pose single source: positions come ONLY from pair["db_pos"]/pair["q_pos"]
(canonical global-GT results/pd2026/keys_pd_s1_{sensor}_{seq}_gp.npz). The
poses.npy next to the range images and the operating-cache poses are
per-session re-centred and are NEVER read here.

ORDER IS CRITICAL: stage 1 computes final descriptors on the FULL native
per-session keyframe stack, so the SEQL=3 / GEM_SEQLEN=20 temporal windows
see the native keyframe rate; the 20 m / 5 m spacing subsample is applied
only afterwards (lib.take in stage 2). Subsampling first would pool frames
~20-100 m apart and be silently wrong.

Stage 1 (GPU, docker ring_baseline; torch imported lazily) caches
results/pd2026_xsession/desc_seqot_{sensor}_{seq}.npz — full-session final
descriptors + scan_ids, roster-independent, shared by the two protocols of a
site. Skipped when present (--overwrite only rewrites protocol JSONs; delete
an npz to recompute it). Stage 2 (host-capable, torch-free): scan_id asserts
-> lib.take -> plain cosine lib.rank_cross_session (SeqOT's native retrieval
metric; no rerank) -> lib.score_rankings ->
results/pd2026_xsession/xs_seqot_{protocol}.json.

Run (inside docker ring_baseline):
  python3 scripts/_xs_seqot_eval.py
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

import _xs_eval_lib as lib

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "scripts" / "_pd_test_cache_manifest.json"
DATA_ROOT_DEFAULT = REPO / "_external" / "seqot_data_pd2026"
RUN_DIR = REPO / "_external" / "seqot_runs_pd2026" / "full"


def desc_path(sensor: str, seq: str) -> Path:
    return lib.OUT_DIR / f"desc_seqot_{sensor}_{seq}.npz"


def prep_cache_scan_ids(sensor: str, seq: str, meta: dict) -> np.ndarray:
    """scan_ids from the SAME pd2026 operating cache the range-image prep
    used (manifest-resolved, cross-checked against meta.json). scan_ids
    ONLY — the cache poses are per-session re-centred and must not be read."""
    man = json.loads(MANIFEST.read_text())
    hits = [e for e in man["sequences"]
            if e["sensor"] == sensor and e["sequence"] == seq]
    assert len(hits) == 1, (sensor, seq)
    cache_p = REPO / hits[0]["expected"]["operating_cache"]
    assert cache_p.name == meta["cache"], (cache_p.name, meta["cache"])
    with np.load(cache_p) as z:
        return z["scan_ids"].copy()


def ensure_descriptors(sessions, args) -> None:
    """Stage 1: full-native-stack SeqOT descriptors for every session that
    has no cached npz yet. torch/GPU needed only when something is missing."""
    missing = [s for s in sessions if not desc_path(*s).exists()]
    for s in sessions:
        if s not in missing:
            print(f"  {desc_path(*s).name}: exists, skip", flush=True)
    if not missing:
        return
    import torch                 # lazy: stage 2 alone stays torch-free
    import _seqot_eval as base   # SeqOT modules + descriptor pipeline

    data_root = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = base.featureExtracter(seqL=base.SEQL).to(device)
    fe.load_state_dict(torch.load(args.seqot_checkpoint, map_location=device,
                                  weights_only=False)["state_dict"])
    fe.eval()
    gem = base.GeM().to(device)
    gem.load_state_dict(torch.load(args.gem_checkpoint, map_location=device,
                                   weights_only=False)["state_dict"])
    gem.eval()

    lib.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sensor, seq in missing:
        seq_dir = data_root / f"{sensor}_{seq}"
        meta = json.loads((seq_dir / "meta.json").read_text())
        assert not meta["truncated"], (sensor, seq, "prep was truncated")
        sids = prep_cache_scan_ids(sensor, seq, meta)
        t0 = time.time()
        stack = base.load_range_stack(seq_dir, 0)   # FULL native stack
        assert len(stack) == len(sids) == meta["n_frames"], \
            (sensor, seq, len(stack), len(sids), meta["n_frames"])
        final = base.compute_final_descriptors(fe, gem, stack, device,
                                               args.infer_batch)
        dt = time.time() - t0
        p = desc_path(sensor, seq)
        np.savez(p, desc=final.astype(np.float32), scan_ids=sids,
                 seqot_checkpoint=str(args.seqot_checkpoint),
                 gem_checkpoint=str(args.gem_checkpoint),
                 seql=int(base.SEQL), gem_seqlen=int(base.GEM_SEQLEN))
        print(f"  {p.name}: {len(final)} frames, {dt:.0f}s "
              f"({dt / max(len(final), 1):.3f} s/kf incl. load)", flush=True)
        del stack


def load_desc(sensor: str, seq: str) -> dict:
    p = desc_path(sensor, seq)
    with np.load(p) as z:
        return {"desc": z["desc"].astype(np.float32),
                "scan_ids": z["scan_ids"].copy(),
                "seqot_checkpoint": str(z["seqot_checkpoint"]),
                "gem_checkpoint": str(z["gem_checkpoint"]),
                "seql": int(z["seql"]), "gem_seqlen": int(z["gem_seqlen"]),
                "path": str(p)}


def session_desc(pair: dict, side: str, cache: dict):
    """Assert the artifact's FULL-session scan_ids against the canonical
    pair, THEN slice the already-windowed final descriptors by spacing."""
    seq = pair["db_seq"] if side == "db" else pair["query_seq"]
    idx = pair["db_idx"] if side == "db" else pair["q_idx"]
    session_sids = (pair["db_session_scan_ids"] if side == "db"
                    else pair["q_session_scan_ids"])
    key = (pair["sensor"], seq)
    if key not in cache:
        cache[key] = load_desc(*key)
    d = cache[key]
    assert np.array_equal(d["scan_ids"], session_sids), (seq, "desc scan_ids")
    return lib.take({"desc": d["desc"]}, idx)["desc"], d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocols", nargs="*", default=list(lib.PROTOCOLS))
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite protocol JSONs (desc npz caches are kept; "
                         "delete one to recompute it)")
    ap.add_argument("--seqot-checkpoint",
                    default=str(RUN_DIR / "seqot_latest.pth.tar"))
    ap.add_argument("--gem-checkpoint",
                    default=str(RUN_DIR / "gem_latest.pth.tar"))
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--infer-batch", type=int, default=16)
    args = ap.parse_args()

    todo = []
    for protocol in args.protocols:
        name = f"xs_seqot_{protocol}"
        if (lib.OUT_DIR / f"{name}.json").exists() and not args.overwrite:
            print(f"  {name}: exists, skip", flush=True)
        else:
            todo.append(protocol)

    sessions = sorted({(lib.PROTOCOLS[p]["sensor"], lib.PROTOCOLS[p][k])
                       for p in todo for k in ("db_seq", "query_seq")})
    ensure_descriptors(sessions, args)

    desc_cache = {}
    for protocol in todo:
        pair = lib.make_pair(protocol)
        print(f"=== {protocol}: n_db={len(pair['db_pos'])} "
              f"n_spaced_q={len(pair['q_pos'])} n_roster={len(pair['roster'])} "
              f"roster_sha={pair['roster_sha256'][:12]}", flush=True)
        db_desc, db_meta = session_desc(pair, "db", desc_cache)
        q_desc, q_meta = session_desc(pair, "query", desc_cache)
        assert db_meta["seqot_checkpoint"] == q_meta["seqot_checkpoint"]
        assert db_meta["gem_checkpoint"] == q_meta["gem_checkpoint"]
        ranked = lib.rank_cross_session(
            db_desc, q_desc, pair["roster"],
            n_coarse=int(lib.CONST["n_coarse_baseline"]))
        scored = lib.score_rankings(ranked, pair["db_pos"], pair["q_pos"],
                                    pair["roster"])
        payload = lib.result_payload(
            "seqot", pair, scored,
            extra={"descriptor_dim": int(db_desc.shape[1]),
                   "seqot_checkpoint": db_meta["seqot_checkpoint"],
                   "gem_checkpoint": db_meta["gem_checkpoint"],
                   "seql": db_meta["seql"],
                   "gem_seqlen": db_meta["gem_seqlen"],
                   "n_coarse": int(lib.CONST["n_coarse_baseline"]),
                   "desc_sources": [db_meta["path"], q_meta["path"]],
                   "windowing": "final descriptors computed on the full "
                                "native session stack before spacing "
                                "subsampling"})
        name = f"xs_seqot_{protocol}"
        p = lib.write_result(payload, name)
        m = payload["metrics"]
        print(f"  {name}: R@1@5m={m['R@1@5m']:.4f} "
              f"R@1@20m={m['R@1@20m']:.4f} "
              f"R1pct@20m={m['R1pct@20m']:.4f} -> {p.name}", flush=True)
    print("XS_SEQOT_DONE", flush=True)


if __name__ == "__main__":
    main()
