#!/usr/bin/env python3
"""Causal all-frames per-query SeqOT ledger for the pre-registered PR eval.

Track C of docs/CAUSAL_PR_PREREGISTRATION.md (§1.1 causal candidates, §1.2
all-frames query set, §1.5 native scores, §3 track C): one npz ledger per
Strict-v2 test sequence with SeqOT's native decision score under the causal
protocol.  NEW FILE — no existing evaluator is touched; the SeqOT model code
(_external/SeqOT via scripts/_seqot_eval.py) is imported, never edited.

MANDATED DEVIATION (prereg §1.5 "SeqOT: trailing window; §3 주의"): the
intra-sequence pipeline (scripts/_seqot_eval.py::window) uses CENTERED
windows — fe seqL=3 over [t-1..t+1] and GeM seqlen=20 over [t-10..t+9] —
which read future frames and are a future leak under the causal protocol.
Descriptors are therefore RECOMPUTED here with TRAILING windows: fe over
[t-2..t], GeM over [t-19..t], clipped at the sequence start by np.clip index
clamping (frame-0 repetition), mirroring how the centered version clips at
the edges.  Every window index is <= t, so descriptor t is a function of
frames <= t only — the whole-sequence precompute is causal-safe and each row
is prefix-invariant (verified by --self-test).

Retrieval per query t (PAST-ONLY, R2): exhaustive cosine over the prefix
j <= t - 30 — a per-query gemv over L2-normalized 256D final descriptors that
only ever reads prefix rows (never global-search-then-filter).  score_cos =
cosine similarity of the prefix argmax (native SeqOT score per §1.5; the
intra driver's faiss normalize_L2 + IndexFlatIP equals normalized dot).
SCORE DIRECTION: cosine similarity is already higher-is-accept — stored
WITHOUT negation (recorded in meta).

Output per sequence (schema "pr-dump-v1"):
  results/pd2026_pr/dump_seqot_{sensor}_{seq}.npz
  arrays query_idx (all t >= 30) / score_cos / j_star / top1_geo_dist /
  min_past_gt_dist (per-query prefix scan, replicating
  scripts/_causal_pr_dump.py) / n_candidates (= prefix size) + json-string
  meta.  Geometry comes from the pd2026 test operating caches via
  _pd_apply_frozen.resolve_caches — the same caches
  scripts/_pdb_seqot_prepare.py wrote poses.npy from (asserted equal, and
  the prepared meta.json cache name is asserted against the resolved cache).

MEMORY DISCIPLINE (scripts/_causal_pr_dump.py convention): the CLI accepts a
sequence list (default: the Strict-v2 37-seq roster from
scripts/_pdb_strict_rev3.strict_roster), but the driver runs each sequence in
its own --worker subprocess (the range stack is ~1 GB for long sequences).
Resumable: existing dumps are skipped unless --overwrite.  --max-frames is
smoke-only and writes *.smoke.npz so the real ledger namespace stays clean.

CPU-safe self-test (no GPU, no data; torch tier auto-skips if torch is
absent):  python scripts/_causal_seqot_pr.py --self-test
Heavy runs happen later inside the ring_baseline container via the causal
baseline chain script (scripts/_causal_baseline_chain.sh).
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

from _pd_apply_frozen import resolve_caches  # noqa: E402  (torch-free)
from _pdb_strict_rev3 import strict_roster   # noqa: E402  (torch-free)
from _pr_eval_lib import MIN_INDEX_GAP       # noqa: E402  (torch-free)

OUT = REPO / "results/pd2026_pr"
DATA_ROOT_DEFAULT = REPO / "_external/seqot_data_pd2026"
RUNS = REPO / "_external/seqot_runs_pd2026/full"
SEQL = 3         # must equal _seqot_eval.SEQL (asserted in the worker)
GEM_SEQLEN = 20  # must equal _seqot_eval.GEM_SEQLEN (asserted in the worker)


def trailing_window(center: int, seqlen: int, n: int) -> np.ndarray:
    """Trailing variant of scripts/_seqot_eval.py::window (line 54): frames
    [center - seqlen + 1 .. center], clipped the same way the centered
    version clips (np.clip into [0, n-1]; at the sequence start the clamp
    repeats frame 0).  Every index is <= center — no future frame is read."""
    lo = center - seqlen + 1
    return np.clip(np.arange(lo, lo + seqlen), 0, n - 1)


def compute_final_descriptors_trailing(fe, gem, stack, device, infer_batch):
    """scripts/_seqot_eval.py::compute_final_descriptors (lines 66-81) with
    trailing_window substituted for window in BOTH stages; batching and
    dtypes are otherwise identical.  sub[t] reads frames [t-2..t] and
    final[t] pools sub[[t-19..t]], so final[t] depends on frames <= t only."""
    import torch
    n = len(stack)
    sub = np.zeros((n, 256), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, infer_batch):
            idx = np.arange(i, min(i + infer_batch, n))
            w = np.stack([trailing_window(j, SEQL, n) for j in idx])
            sub[idx] = fe(torch.from_numpy(stack[w]).to(device)).cpu().numpy()
        final = np.zeros((n, 256), dtype=np.float32)
        gem_batch = 512
        for i in range(0, n, gem_batch):
            idx = np.arange(i, min(i + gem_batch, n))
            w = np.stack([trailing_window(j, GEM_SEQLEN, n) for j in idx])
            out = gem(torch.from_numpy(sub[w]).to(device)).squeeze(1)
            final[idx] = out.cpu().numpy()
    return final


def min_past_dists(positions: np.ndarray) -> np.ndarray:
    """min 3D distance from t to ANY frame <= t - 30 (query classification
    input) — per-query prefix scan replicating scripts/_causal_pr_dump.py
    (never a full-sequence statistic).  NaN for t < 30."""
    n = len(positions)
    out = np.full(n, np.nan)
    for t in range(MIN_INDEX_GAP, n):
        d = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t],
                           axis=1)
        out[t] = d.min()
    return out


def dump_ledger(final_n: np.ndarray, positions: np.ndarray,
                min_past: np.ndarray) -> dict:
    """Per-query ledger arrays: exhaustive prefix cosine top-1 per t >= 30.
    The gemv reads rows [0, t - 30] only (R2: no global-then-filter); the
    per-row dot has a fixed accumulation order, so scores are bit-identical
    under sequence truncation (self-test asserts this)."""
    n = len(final_n)
    queries = np.arange(MIN_INDEX_GAP, n, dtype=np.int64)
    j_star = np.full(len(queries), -1, dtype=np.int64)
    score_cos = np.full(len(queries), np.nan)
    top1_geo = np.full(len(queries), np.nan)
    n_cand = np.zeros(len(queries), dtype=np.int64)
    for qi, t in enumerate(queries):
        t = int(t)
        n_prefix = t - MIN_INDEX_GAP + 1  # prefix rows [0, t - 30]
        if n_prefix <= 0:
            continue  # empty candidate pool -> NaN row (not expected, t >= 30)
        col = final_n[:n_prefix] @ final_n[t]
        j = int(np.argmax(col))  # exhaustive prefix top-1 (first-max ties)
        assert j <= t - MIN_INDEX_GAP, "candidate beyond the causal prefix"
        j_star[qi] = j
        score_cos[qi] = float(col[j])
        top1_geo[qi] = float(np.linalg.norm(positions[j] - positions[t]))
        n_cand[qi] = n_prefix
        if qi and qi % 1000 == 0:
            print(f"  ledger queries {qi}/{len(queries)}", flush=True)
    return dict(query_idx=queries, j_star=j_star, score_cos=score_cos,
                top1_geo_dist=top1_geo, min_past_gt_dist=min_past[queries],
                n_candidates=n_cand)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def out_path_for(sensor: str, seq: str, max_frames: int) -> Path:
    suffix = ".smoke.npz" if max_frames else ".npz"
    return OUT / f"dump_seqot_{sensor}_{seq}{suffix}"


def run_worker(args, sensor: str, seq: str):
    import torch
    import _seqot_eval as base

    assert base.SEQL == SEQL and base.GEM_SEQLEN == GEM_SEQLEN, \
        "window constants drifted from scripts/_seqot_eval.py"

    out_path = out_path_for(sensor, seq, args.max_frames)
    if out_path.exists() and not args.overwrite:
        print(f"[{sensor}/{seq}] {out_path.name} exists, skipping", flush=True)
        return

    seq_dir = Path(args.data_root) / f"{sensor}_{seq}"
    assert (seq_dir / "poses.npy").exists(), \
        f"{seq_dir} not prepared (run scripts/_pdb_seqot_prepare.py)"
    prep_meta = json.loads((seq_dir / "meta.json").read_text())

    # Geometry provenance: the pd2026 test operating cache (R5) — the same
    # cache _pdb_seqot_prepare wrote this sequence's range stack + poses from.
    cache_p, _, _, marker = resolve_caches(sensor, seq)
    assert marker == "test", (sensor, seq, marker, "expected a test cache")
    assert prep_meta["cache"] == cache_p.name, \
        (sensor, seq, prep_meta["cache"], cache_p.name,
         "prepared range stack came from a different cache")
    cache = np.load(cache_p)
    poses = cache["poses"]

    t0 = time.time()
    stack = base.load_range_stack(seq_dir, args.max_frames)
    n = len(stack)
    poses_npy = np.load(seq_dir / "poses.npy")
    if args.max_frames:
        print(f"[{sensor}/{seq}] SMOKE: truncated to {n} frames "
              f"-> {out_path.name}", flush=True)
        poses, poses_npy = poses[:n], poses_npy[:n]
    assert len(poses) == n == len(poses_npy), (sensor, seq, "frame count")
    assert np.array_equal(poses_npy, poses.astype(np.float64)), \
        (sensor, seq, "poses.npy != operating-cache poses")
    t_load = time.time() - t0

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    device = torch.device(dev)
    fe = base.featureExtracter(seqL=SEQL).to(device)
    fe.load_state_dict(torch.load(args.seqot_checkpoint, map_location=device,
                                  weights_only=False)["state_dict"])
    fe.eval()
    gem = base.GeM().to(device)
    gem.load_state_dict(torch.load(args.gem_checkpoint, map_location=device,
                                   weights_only=False)["state_dict"])
    gem.eval()

    t1 = time.time()
    final = compute_final_descriptors_trailing(fe, gem, stack, device,
                                               args.infer_batch)
    t_desc = time.time() - t1
    norms = np.linalg.norm(final, axis=1, keepdims=True)
    assert (norms > 0).all(), (sensor, seq, "zero-norm final descriptor")
    final_n = final / norms  # cosine == normalized dot (intra driver: faiss
    #                          normalize_L2 + IndexFlatIP)

    positions = poses[:, :3, 3]
    min_past = min_past_dists(positions)
    t2 = time.time()
    arrays = dump_ledger(final_n, positions, min_past)
    t_ret = time.time() - t2

    meta = {
        "schema": "pr-dump-v1",
        "method": "seqot",
        "sensor": sensor, "sequence": seq,
        "window": "trailing",
        "seql": SEQL, "gem_seqlen": GEM_SEQLEN,
        "min_index_gap": MIN_INDEX_GAP,
        "n_frames": n, "n_queries": int(len(arrays["query_idx"])),
        "infer_batch": args.infer_batch, "device": dev,
        "max_frames": args.max_frames,
        "primary_score": "score_cos",
        "score_direction": ("score_cos = cosine similarity of the 256D "
                            "trailing-window final descriptors at the prefix "
                            "argmax; higher = accept. Native score is already "
                            "a similarity — stored WITHOUT negation."),
        "descriptor": ("256D SeqOT final descriptor (fe seqL=3 -> GeM "
                       "seqlen=20, both trailing), per-row L2-normalized "
                       "before the prefix gemv"),
        "deviation": ("prereg §1.5/§3 SeqOT deviation: the intra-sequence "
                      "pipeline (scripts/_seqot_eval.py) uses CENTERED "
                      "windows (fe [t-1..t+1], GeM [t-10..t+9]) — a future "
                      "leak under the causal protocol. Descriptors were "
                      "recomputed with TRAILING windows [t-2..t] / "
                      "[t-19..t], clipped at the sequence start by index "
                      "clamping (frame-0 repetition), mirroring the centered "
                      "version's edge clamping. Ledger scores are therefore "
                      "not comparable to the offline (centered) SeqOT rows."),
        "protocol": ("causal all-frames PR ledger: trailing-window SeqOT "
                     "descriptors (causal per frame), exhaustive per-query "
                     "cosine over the prefix j <= t - 30 (prefix gemv, never "
                     "global-then-filter), native cosine top-1 score; "
                     "scores/dists per docs/CAUSAL_PR_PREREGISTRATION.md "
                     "§1.2/§1.5 track C"),
        "artifacts": {
            "seqot_checkpoint": str(args.seqot_checkpoint),
            "seqot_checkpoint_sha256": sha256_of(Path(args.seqot_checkpoint)),
            "gem_checkpoint": str(args.gem_checkpoint),
            "gem_checkpoint_sha256": sha256_of(Path(args.gem_checkpoint)),
        },
        "caches": {"operating_cache": str(cache_p), "marker": marker,
                   "seqot_data_dir": str(seq_dir),
                   "seqot_prepare_meta": prep_meta},
        "timing": {"load_s": round(t_load, 1),
                   "descriptor_s_total": round(t_desc, 1),
                   "descriptor_s_per_kf": round(t_desc / max(n, 1), 4),
                   "ledger_s_total": round(t_ret, 1)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    fin = np.isfinite(arrays["score_cos"])
    print(f"[{sensor}/{seq}] n_queries={len(arrays['query_idx'])} "
          f"finite={int(fin.sum())} desc {t_desc:.1f}s ledger {t_ret:.1f}s "
          f"-> {out_path.name}", flush=True)


def self_test():
    """CPU-safe synthetic checks of every NEW code path (R6). The torch tier
    (random-weight fe+GeM prefix-invariance) auto-skips when torch/SeqOT are
    not importable on this host — rerun inside ring_baseline for full cover."""
    rng = np.random.default_rng(0)

    # 1. trailing windows: right length, ordered, never a future index.
    assert trailing_window(0, 3, 50).tolist() == [0, 0, 0]
    assert trailing_window(1, 3, 50).tolist() == [0, 0, 1]
    assert trailing_window(5, 3, 50).tolist() == [3, 4, 5]
    assert trailing_window(49, 20, 50).tolist() == list(range(30, 50))
    for _ in range(500):
        n = int(rng.integers(1, 60))
        t = int(rng.integers(0, n))
        seqlen = int(rng.integers(1, 25))
        w = trailing_window(t, seqlen, n)
        assert len(w) == seqlen and w.min() >= 0 and w.max() <= t
        assert np.all(np.diff(w) >= 0)
    print("PASS trailing_window: len==seqlen, ordered, every index <= t")

    # 2. ledger prefix-invariance: truncating the sequence never changes the
    #    surviving rows (bit-identical — the causality proof for retrieval).
    n, d = 80, 16
    bank = rng.standard_normal((n, d)).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    positions = (rng.standard_normal((n, 3)) * 50.0).astype(np.float64)
    full = dump_ledger(bank, positions, min_past_dists(positions))
    cut = 60
    trunc = dump_ledger(bank[:cut], positions[:cut],
                        min_past_dists(positions[:cut]))
    m = len(trunc["query_idx"])
    assert m == cut - MIN_INDEX_GAP
    for k in full:
        assert np.array_equal(full[k][:m], trunc[k], equal_nan=True), k
    print(f"PASS ledger prefix-invariance: first {m} queries bit-identical "
          f"after truncation at {cut}")

    # 3. ledger candidates + min_past vs brute force.
    assert np.array_equal(full["n_candidates"],
                          full["query_idx"] - MIN_INDEX_GAP + 1)
    mp = min_past_dists(positions)
    assert np.isnan(mp[:MIN_INDEX_GAP]).all()
    for t in (MIN_INDEX_GAP, 45, n - 1):
        expect = min(float(np.linalg.norm(positions[j] - positions[t]))
                     for j in range(t - MIN_INDEX_GAP + 1))
        assert abs(float(mp[t]) - expect) < 1e-9
    for qi, t in enumerate(full["query_idx"][:5]):
        col = bank[:t - MIN_INDEX_GAP + 1] @ bank[t]
        assert full["j_star"][qi] == int(np.argmax(col))
        assert full["score_cos"][qi] == float(col.max())
    print("PASS n_candidates == prefix size; min_past + top-1 match brute "
          "force; frames < 30 are NaN")

    # 4. torch tier: trailing DESCRIPTORS are prefix-invariant end to end
    #    (random weights — the property is architectural, not learned).
    try:
        import torch
        import _seqot_eval as base
    except Exception as e:  # noqa: BLE001 — any import failure => skip
        print(f"SKIP torch descriptor test ({type(e).__name__}: {e}); "
              "rerun --self-test inside the ring_baseline container")
        return
    assert base.SEQL == SEQL and base.GEM_SEQLEN == GEM_SEQLEN
    torch.manual_seed(0)
    fe = base.featureExtracter(seqL=SEQL).eval()
    gem = base.GeM().eval()
    stack = rng.random((26, 32, 900), dtype=np.float32)
    dev = torch.device("cpu")
    f_full = compute_final_descriptors_trailing(fe, gem, stack, dev, 8)
    f_trunc = compute_final_descriptors_trailing(fe, gem, stack[:21], dev, 8)
    assert np.allclose(f_full[:21], f_trunc, atol=1e-4, rtol=1e-4)
    print("PASS trailing descriptors prefix-invariant (random-weight fe+GeM, "
          "CPU, n=26 vs truncation at 21)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seqs", nargs="+", default=None, metavar="SENSOR/SEQ",
                    help="subset, e.g. kitti/05 (default: the Strict-v2 "
                         "37-sequence roster)")
    ap.add_argument("--seqot-checkpoint",
                    default=str(RUNS / "seqot_latest.pth.tar"),
                    help="pd2026 retrain featureExtracter checkpoint")
    ap.add_argument("--gem-checkpoint",
                    default=str(RUNS / "gem_latest.pth.tar"),
                    help="pd2026 retrain GeM checkpoint")
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT),
                    help="prepared range stacks (scripts/_pdb_seqot_prepare.py)")
    ap.add_argument("--infer-batch", type=int, default=16)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="smoke only; writes *.smoke.npz, never the ledger")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--worker", action="store_true",
                    help="internal: run a single sequence in-process")
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-safe synthetic checks (no GPU, no data); the "
                         "torch tier auto-skips when torch is unavailable")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    roster = strict_roster()
    if args.seqs is None:
        targets = roster
    else:
        targets = []
        for s in args.seqs:
            assert s.count("/") == 1, \
                f"--seqs entries must be sensor/seq, got {s!r}"
            sensor, seq = s.split("/")
            assert (sensor, seq) in roster, \
                f"{s} not in the Strict-v2 37-sequence roster"
            targets.append((sensor, seq))

    if args.worker:
        assert len(targets) == 1, "--worker takes exactly one sequence"
        run_worker(args, *targets[0])
        return

    # Driver: one heavy sequence per subprocess (memory discipline), skipping
    # already-dumped sequences BEFORE spawning (resumable).
    OUT.mkdir(parents=True, exist_ok=True)
    for sensor, seq in targets:
        out_path = out_path_for(sensor, seq, args.max_frames)
        if out_path.exists() and not args.overwrite:
            print(f"skip {sensor}/{seq}: {out_path.name} exists", flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--seqs", f"{sensor}/{seq}",
               "--seqot-checkpoint", args.seqot_checkpoint,
               "--gem-checkpoint", args.gem_checkpoint,
               "--data-root", args.data_root,
               "--infer-batch", str(args.infer_batch),
               "--device", args.device,
               "--max-frames", str(args.max_frames),
               "--worker"]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"=== spawning worker: {sensor}/{seq} ===", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"WORKER_FAILED {sensor}/{seq} rc={rc}", flush=True)
            sys.exit(rc)
    print("CAUSAL_SEQOT_PR_DONE", flush=True)


if __name__ == "__main__":
    main()
