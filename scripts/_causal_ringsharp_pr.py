#!/usr/bin/env python3
"""RING#-L all-frames per-query ledger for the pre-registered causal PR eval.

Track C of docs/CAUSAL_PR_PREREGISTRATION.md (§1.1 causal candidates, §1.2
all-frames query set, §1.5 RING#-L native-score row): per sensor/seq dump one
row per query frame t >= 30 with everything scripts/_pr_eval_lib.py needs
downstream — no PR accounting happens here.  Output schema and worker
discipline mirror scripts/_causal_pr_dump.py (schema "pr-dump-v1").

Pipeline per sequence (official RINGSharp code, imported unchanged from
scripts/_pdb_ringsharp_eval.py — that file stays byte-identical):
  * features ONCE per sequence via _pdb_ringsharp_eval.compute_seq_features
    (official generate_bev 160x160x20 -> RINGSharpL forward -> 160D 'global'
    + 160x160 'spec' + cached trans_cnn features).  CAUSALITY: each frame's
    features depend only on that frame's point cloud, so whole-sequence
    precompute leaks nothing (prereg Track C: per-frame methods reuse
    descriptors, only the search is re-run causally).
  * per query t: coarse = 160D 'global' cosine top-min(n_coarse, |prefix|)
    computed WITHIN the prefix j <= t - 30 by a per-query gemv over prefix
    rows only (_causal_pr_dump.prefix_topk, imported — bit-identical
    candidate selection, never global-search-then-filter).
  * rerank = the official rotation + translation two-branch rerank, verbatim
    from _pdb_ringsharp_eval.evaluate_sequence's rerank_fn: batched
    estimate_yaw over all coarse candidates -> rot_dist = 1 - corr; the top
    min(n_trans, |cand|) by rot_dist go through the translation branch
    (rotate_bev_batch at +yaw and +yaw-pi -> solve_translation -> errs =
    elementwise min) and are re-ordered by errs.  CAUSALITY: both branches
    are pairwise (t, j) computations over the already-causal candidate set —
    no cross-candidate or future state.
  * emitted top-1 = first element after the translation re-order
    (= argmin errs within the rot-branch top-n_trans, matching the official
    `order[:m] = top[np.argsort(errs)]`).

SCORE DIRECTION (§1.5 "translation-branch errs.min() (primary) + rot_dist
(comparison)"; ledger stores similarities, higher = accept):
  score_trans = -errs.min()  = negated translation correlation error of the
                emitted top-1 (primary decision score);
  score_rot   = -rot_dist    of the SAME emitted top-1 (NOT the rot-branch
                minimum — the emitted match is chosen by the translation
                branch).  Both negations are recorded in meta.

Weights: the pd2026 place-disjoint retrain checkpoint selected by the
pre-declared rule in results/pd2026_baselines/ringsharp_pd_ckpt_selection.json
(epoch e1); its sha256 is re-computed and asserted against the frozen
selection record before any evaluation.

Documented protocol deviation (inherited from scripts/_ringsharp_eval.py):
official n_rerank=1000 -> --n-trans 20 on top of --n-coarse 200 (now
min(200, |prefix|)); no silent caps beyond these disclosed defaults.

Output, one npz per sequence (resumable; skipped when present):
  results/pd2026_pr/dump_ringsharp_{sensor}_{seq}.npz
  arrays query_idx / j_star / score_trans / score_rot / top1_geo_dist /
  min_past_gt_dist / n_candidates + json-string meta.

MEMORY / WORKER DISCIPLINE (_causal_pr_dump convention): the CLI accepts a
sequence list (default = the Strict-v2 37-seq roster), but the driver runs
each sequence in its own --worker subprocess; the worker holds the large
per-sequence spec/trans caches and frees GPU tensors before exit.  The heavy
glnet/e2cnn imports happen only inside the worker (ring_baseline container);
--help and --self-test are CPU-safe and torch-free.

Examples (inside the ring_baseline container):
  python scripts/_causal_ringsharp_pr.py --self-test          # torch-free
  python scripts/_causal_ringsharp_pr.py --seqs kitti/05      # one sequence
  python scripts/_causal_ringsharp_pr.py                      # all 37, resumable
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _causal_pr_dump import prefix_topk  # noqa: E402  (torch-free)
from _pr_eval_lib import MIN_INDEX_GAP   # noqa: E402  (torch-free)

METHOD = "ringsharp"
OUT = REPO / "results/pd2026_pr"
MANIFEST = REPO / "scripts/_pd_test_cache_manifest.json"
CKPT_SELECTION = REPO / "results/pd2026_baselines/ringsharp_pd_ckpt_selection.json"
# Strict-v2 = frozen 42-seq manifest minus these five (prereg roster rule).
STRICT_V2_EXCLUDED = {
    ("kitti", "00"), ("kitti360", "0004"), ("helipr", "Bridge04"),
    ("nclt", "2012-02-04"), ("nclt", "2012-03-17"),
}


def load_roster():
    """Torch-free replica of _pdb_ringsharp_eval.load_manifest (that module
    imports glnet at top level, so it is only importable in the worker) plus
    the Strict-v2 exclusion.  Returns (ordered strict list, cache map)."""
    m = json.loads(MANIFEST.read_text())
    cache = {(e["sensor"], e["sequence"]): REPO / e["expected"]["operating_cache"]
             for e in m["sequences"]}
    strict = [(e["sensor"], e["sequence"]) for e in m["sequences"]
              if (e["sensor"], e["sequence"]) not in STRICT_V2_EXCLUDED]
    assert len(strict) == 37, f"Strict-v2 roster is {len(strict)} != 37"
    return strict, cache


def sha256_of(path: Path) -> str:
    """Replica of _pdb_ringsharp_pd_eval.sha256_of (same top-level glnet
    import problem prevents importing it here)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_weight():
    """The pd2026 retrain checkpoint from the frozen pre-declared selection;
    container path remapped onto this checkout, sha256 asserted."""
    sel = json.loads(CKPT_SELECTION.read_text())
    w = sel["selected_weight"]
    p = Path(w)
    if not p.exists():  # container path /workspace/Neural-Spectral-Codec/...
        p = REPO / w.split("/Neural-Spectral-Codec/", 1)[1]
    assert p.exists(), f"selected weight not found: {w} (remap {p})"
    sha = sha256_of(p)
    assert sha == sel["weight_sha256"], (
        f"weight sha mismatch: {p}\n  computed {sha}\n  frozen   "
        f"{sel['weight_sha256']} ({CKPT_SELECTION.name})")
    return p, sha, sel


def min_past_dists(positions: np.ndarray) -> np.ndarray:
    """min 3D distance from t to ANY frame <= t - 30 (query classification
    input) — per-query prefix scan, replicated verbatim from
    _causal_pr_dump.run_worker; never a full-sequence statistic."""
    n = len(positions)
    mp = np.full(n, np.nan)
    for t in range(MIN_INDEX_GAP, n):
        d = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t],
                           axis=1)
        mp[t] = d.min()
    return mp


def dump_seq(globn: np.ndarray, positions: np.ndarray, rerank_top1,
             n_coarse: int) -> dict:
    """Causal all-frames ledger loop (torch-free; the official rerank enters
    as the rerank_top1(t, cand) -> (j_star, err_trans, rot_dist) callable so
    --self-test can drive this loop with a synthetic stub).

    globn must be row-L2-normalized (coarse cosine = dot; matches the
    faiss.normalize_L2 + IndexFlatIP convention of baselines/eval_utils)."""
    n = len(globn)
    queries = np.arange(MIN_INDEX_GAP, n, dtype=np.int64)
    j_star = np.full(len(queries), -1, dtype=np.int64)
    score_trans = np.full(len(queries), np.nan)
    score_rot = np.full(len(queries), np.nan)
    top1_geo = np.full(len(queries), np.nan)
    n_cand = np.zeros(len(queries), dtype=np.int64)
    for qi, t in enumerate(queries):
        t = int(t)
        n_prefix = t - MIN_INDEX_GAP + 1  # prefix rows [0, t - 30]
        if n_prefix <= 0:
            continue  # empty pool -> NaN row (not expected: t >= 30)
        cand = prefix_topk(globn, t, n_prefix, n_coarse)
        assert cand.size and cand.max() <= t - MIN_INDEX_GAP, \
            "candidate beyond the causal prefix"
        j, err_trans, rot_dist = rerank_top1(t, cand)
        j = int(j)
        j_star[qi] = j
        score_trans[qi] = -float(err_trans)  # similarity: negated trans error
        score_rot[qi] = -float(rot_dist)     # similarity: negated 1 - corr
        top1_geo[qi] = float(np.linalg.norm(positions[j] - positions[t]))
        n_cand[qi] = int(cand.size)
        if qi and qi % 200 == 0:
            print(f"  ledger queries {qi}/{len(queries)}", flush=True)
    return dict(query_idx=queries, j_star=j_star, score_trans=score_trans,
                score_rot=score_rot, top1_geo_dist=top1_geo,
                n_candidates=n_cand)


def make_rerank_top1(specs, bts, device, n_trans):
    """Official rotation + translation rerank, verbatim math from
    _pdb_ringsharp_eval.evaluate_sequence.rerank_fn (batched estimate_yaw
    over all candidates, translation branch on the rot-branch top-n_trans),
    returning the emitted top-1 with its native scores instead of the full
    re-ordered list."""
    import torch
    import _pdb_ringsharp_eval as pdrs
    base = pdrs.base

    def rerank_top1(qi, cand):
        cand = np.asarray(cand)
        qspec = specs[qi].float().to(device)
        cspec = specs[cand].float().to(device)
        with torch.no_grad():
            _, scores, angles = base.estimate_yaw(
                qspec.unsqueeze(0).expand(len(cand), -1, -1, -1), cspec)
        rot_dist = (1 - scores).cpu().numpy().reshape(-1)
        angles = (angles * 2 * np.pi / specs.shape[-2]).cpu().numpy().reshape(-1)
        order = np.argsort(rot_dist)
        m = min(n_trans, len(cand))
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
        # Emitted top-1 after `order[:m] = top[np.argsort(errs)]` = the errs
        # minimum within the rot-branch top-m (argmin = first minimum, i.e.
        # the stable-sort head; float-tie behavior is the only difference
        # from the official unstable argsort and is vanishingly unlikely).
        i_best = int(np.argmin(errs))
        j = int(cand[top[i_best]])
        return j, float(errs[i_best]), float(rot_dist[top[i_best]])

    return rerank_top1


def out_path(sensor: str, seq: str, max_frames: int) -> Path:
    smoke = f"_maxf{max_frames}" if max_frames else ""
    return OUT / f"dump_{METHOD}_{sensor}_{seq}{smoke}.npz"


def run_worker(args, sensor: str, seq: str):
    import torch
    import _pdb_ringsharp_eval as pdrs  # glnet/e2cnn imports (container env)

    _, cache_map = load_roster()
    weight, weight_sha, sel = selected_weight()
    print(f"weight={weight}\nsha256={weight_sha} (asserted vs "
          f"{CKPT_SELECTION.name})", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, trans_cnn = pdrs.base.load_model(str(weight), device)

    cache_p = cache_map[(sensor, seq)]
    with np.load(cache_p, allow_pickle=True) as c:
        scan_ids = c["scan_ids"].astype(np.int64)
        # Scoring uses ground-truth poses (poses_gt on odometry-keyframe caches).
        poses = (c["poses_gt"] if "poses_gt" in c.files
                 else c["poses"]).astype(np.float64)
    if args.max_frames:
        scan_ids, poses = scan_ids[:args.max_frames], poses[:args.max_frames]
    label = f"{sensor}_{seq}"
    print(f"[{label}] {len(scan_ids)} keyframes (pd2026 test cache "
          f"{cache_p.name})", flush=True)

    loader = pdrs.make_pd_loader(sensor, seq, pdrs.DATA_ROOT)
    t0 = time.time()
    glob, specs, bevs_packed, bts = pdrs.compute_seq_features(
        model, trans_cnn, loader, label, scan_ids, device, args.batch)
    t_feat = time.time() - t0
    del loader, bevs_packed
    n = len(glob)
    assert n == len(poses), (sensor, seq, "feature/pose length")
    print(f"[{label}] features: {t_feat:.0f}s ({t_feat / n:.3f} s/kf)",
          flush=True)

    globn = glob / np.linalg.norm(glob, axis=1, keepdims=True)
    positions = poses[:, :3, 3]
    min_past = min_past_dists(positions)
    rerank = make_rerank_top1(specs, bts, device, args.n_trans)

    t0 = time.time()
    arrays = dump_seq(globn.astype(np.float32), positions, rerank,
                      args.n_coarse)
    t_led = time.time() - t0
    arrays["min_past_gt_dist"] = min_past[arrays["query_idx"]]

    meta = {
        "schema": "pr-dump-v1",
        "method": METHOD,
        "sensor": sensor, "sequence": seq,
        "n_frames": n, "n_queries": int(len(arrays["query_idx"])),
        "n_coarse": args.n_coarse, "n_trans": args.n_trans,
        "batch": args.batch, "max_frames": args.max_frames,
        "min_index_gap": MIN_INDEX_GAP,
        "device": device,
        "score_direction": {
            "score_trans": ("PRIMARY: similarity = -errs.min() = negated "
                            "translation-branch correlation error of the "
                            "emitted top-1 (native err: lower = better; "
                            "negated so higher = accept)"),
            "score_rot": ("COMPARISON: similarity = -rot_dist = -(1 - "
                          "rotation circular-correlation score) of the SAME "
                          "emitted top-1 (not the rot-branch minimum; native "
                          "dist: lower = better; negated so higher = accept)"),
        },
        "artifacts": {
            "weight": str(weight),
            "weight_sha256": weight_sha,
            "weight_selection": str(CKPT_SELECTION),
            "weight_selection_rule": sel["rule"],
            "cache": str(cache_p),
            "manifest": str(MANIFEST),
        },
        "train_overlap": False,  # pd2026 retrain is place-disjoint from test
        "protocol": ("causal all-frames PR ledger (RING#-L, prereg §1.2/§1.5):"
                     " per-frame features precomputed (frame-local, no leak);"
                     " coarse = 160D global cosine top-min(n_coarse,|prefix|)"
                     " searched WITHIN the prefix j <= t - 30 (per-query gemv,"
                     " never global-then-filter); official rotation"
                     " (estimate_yaw) + translation (rotate_bev_batch +"
                     " solve_translation) rerank of the rot-branch top-n_trans;"
                     " emitted top-1 = translation-reordered head. Documented"
                     " deviation (inherited from _ringsharp_eval.py): official"
                     " n_rerank=1000 -> n_trans=20 on n_coarse=200; no other"
                     " caps."),
        "timing": {
            "feature_s_total": round(t_feat, 1),
            "feature_s_per_kf": round(t_feat / n, 4),
            "ledger_s_total": round(t_led, 1),
            "ledger_ms_per_query": round(
                1000 * t_led / max(len(arrays["query_idx"]), 1), 1),
        },
    }
    out = out_path(sensor, seq, args.max_frames)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    fin = np.isfinite(arrays["score_trans"])
    print(f"[{label}] n_queries={len(arrays['query_idx'])} "
          f"finite={int(fin.sum())} -> {out.name}", flush=True)

    # Free the large per-sequence state explicitly (R6/worker discipline);
    # the subprocess exit then releases everything else.
    del specs, bts, model, trans_cnn, rerank
    if device == "cuda":
        torch.cuda.empty_cache()


def self_test():
    """CPU-safe synthetic check of the torch-free causal ledger loop: R2
    (prefix-only candidates), R3 (all t >= 30), score negation, and prefix
    truncation invariance.  The official rerank is replaced by a
    deterministic stub that depends only on (t, cand)."""
    rng = np.random.default_rng(0)
    n = 140
    glob = rng.standard_normal((n, 16)).astype(np.float32)
    globn = glob / np.linalg.norm(glob, axis=1, keepdims=True)
    positions = np.cumsum(rng.standard_normal((n, 3)), axis=0)
    seen = []

    def stub(t, cand):
        cand = np.asarray(cand)
        seen.append((t, int(cand.max())))
        errs = np.abs(np.sin(cand * 0.7 + t * 0.01))
        i = int(np.argmin(errs))
        return int(cand[i]), float(errs[i]), float(np.abs(np.cos(cand[i] * 0.3)))

    arrays = dump_seq(globn, positions, stub, n_coarse=50)
    # R3: every t >= 30 is a query row.
    assert np.array_equal(arrays["query_idx"], np.arange(30, n)), "query set"
    # R2: every candidate the rerank ever saw was within the prefix.
    assert all(cmax <= t - MIN_INDEX_GAP for t, cmax in seen), "causality"
    # Scores: finite everywhere (prefix non-empty for t >= 30) and negated
    # (stub errors are non-negative -> similarities are non-positive).
    for k in ("score_trans", "score_rot"):
        assert np.isfinite(arrays[k]).all() and (arrays[k] <= 0).all(), k
    # Prefix-invariance (prereg §4.1 flavor): truncating the sequence at t0
    # reproduces the shared rows bit-exactly.
    t0 = 100
    a2 = dump_seq(globn[:t0 + 1], positions[:t0 + 1], stub, n_coarse=50)
    m = t0 + 1 - 30
    for k in ("query_idx", "j_star", "n_candidates"):
        assert np.array_equal(arrays[k][:m], a2[k]), f"truncation: {k}"
    for k in ("score_trans", "score_rot", "top1_geo_dist"):
        assert np.array_equal(arrays[k][:m], a2[k]), f"truncation: {k}"
    # min_past_dists: prefix scan matches brute force at a sample query.
    mp = min_past_dists(positions)
    t = 77
    ref = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t],
                         axis=1).min()
    assert np.isclose(mp[t], ref), "min_past_gt_dist"
    # Roster + frozen-selection records parse (no torch needed).
    strict, cache_map = load_roster()
    assert all(s in cache_map for s in strict)
    sel = json.loads(CKPT_SELECTION.read_text())
    assert sel["selected_weight"].endswith(".pth") and sel["weight_sha256"]
    print("SELF_TEST_OK", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seqs", nargs="+", metavar="SENSOR/SEQ", default=None,
                    help="subset of the Strict-v2 37-seq roster "
                         "(default: all 37, resumable)")
    ap.add_argument("--n-coarse", type=int, default=200,
                    help="coarse pool = min(n_coarse, |prefix|)")
    ap.add_argument("--n-trans", type=int, default=20,
                    help="rot-branch candidates re-ordered by the "
                         "translation branch (documented deviation from the "
                         "official n_rerank=1000)")
    ap.add_argument("--batch", type=int, default=8, help="feature batch")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="smoke only (output gets a _maxf suffix)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-safe synthetic ledger check (no torch/glnet)")
    ap.add_argument("--worker", action="store_true",
                    help="internal: run a single sequence in-process")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    strict, cache_map = load_roster()
    if args.seqs is None:
        targets = strict
    else:
        targets = []
        for s in args.seqs:
            assert s.count("/") == 1, f"--seqs entries are sensor/seq: {s!r}"
            t = tuple(s.split("/"))
            assert t in strict, f"not in the Strict-v2 37-seq roster: {s}"
            targets.append(t)

    if args.worker:
        assert len(targets) == 1, "--worker takes exactly one sequence"
        run_worker(args, *targets[0])
        return

    OUT.mkdir(parents=True, exist_ok=True)
    # Driver: one heavy sequence per subprocess (memory discipline; the
    # worker holds GB-scale spec/trans caches).  Resumable: existing dumps
    # are skipped before a worker is spawned.
    for sensor, seq in targets:
        out = out_path(sensor, seq, args.max_frames)
        if out.exists() and not args.overwrite:
            print(f"[{sensor}/{seq}] {out.name} exists, skipping", flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--seqs", f"{sensor}/{seq}",
               "--n-coarse", str(args.n_coarse),
               "--n-trans", str(args.n_trans),
               "--batch", str(args.batch), "--worker"]
        if args.max_frames:
            cmd += ["--max-frames", str(args.max_frames)]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"=== spawning worker: {sensor}/{seq} ===", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"WORKER_FAILED {sensor}/{seq} rc={rc}", flush=True)
            sys.exit(rc)
    print("CAUSAL_RINGSHARP_PR_DONE", flush=True)


if __name__ == "__main__":
    main()
