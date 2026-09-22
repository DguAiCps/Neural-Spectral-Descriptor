#!/usr/bin/env python3
"""RING / RING++ all-frames per-query score ledgers for the causal PR eval.

Track C of docs/CAUSAL_PR_PREREGISTRATION.md (baselines under the causal
protocol): per (variant, sensor/seq) dump one row per query frame t >= 30
with everything scripts/_pr_eval_lib.py needs downstream — no PR accounting
happens here.  Output contract mirrors scripts/_causal_pr_dump.py (schema
"pr-dump-v1").

Method (both variants share everything except the BEV channel count):
  * BEVs: shared per-sequence cache
    _external/ringsharp_data/pdb_ring_cache/bev_{sensor}_{seq}.npz, opened
    READ-ONLY with a full-sequence scan_ids match (the
    _xs_ring_eval.load_sliced_bevs guard); this driver NEVER calls
    _pdb_ring_eval.compute_bevs, which rebuilds-and-overwrites on mismatch.
  * TIRING specs: _pdb_ring_eval.compute_specs (official radon sinogram ->
    row-FFT magnitude), computed ONCE per (sequence, variant).  Per-frame
    descriptors depend only on that frame, so precomputing the whole bank is
    causal-safe; only the CANDIDATE RANGE per query is prefix-limited, and
    that is index arithmetic (bank[:t - 30 + 1]), not data-dependent
    selection.
  * Scoring per query t: EXHAUSTIVE official circular correlation
    (glnet estimate_yaw, the _pdb_ring_eval.eval_variant rerank_fn loop,
    batch 256) of spec[t] against every prefix frame j <= t - 30.  No coarse
    stage, no caps — §1.5 fixes the RING/RING++ score as the exhaustive
    correlation distance.  Queries run in index order against a growing view
    of a persistent spec bank (fp16, device-resident by default;
    --specs-on-cpu streams candidate batches instead).
  * SCORE DIRECTION: official dist = 1 - normalized circular correlation
    (glnet/models/utils.py estimate_yaw).  Stored score_corr =
    -(1 - corr_max) = corr_max - 1 of the prefix argmin(dist): a similarity,
    higher = accept.  The negation is recorded in meta.

Per-query row: score_corr, j_star (prefix argmin; np.argmax first-max
tie-break, deterministic), top1_geo_dist (3D pose distance of the returned
top-1), min_past_gt_dist (min 3D distance to ANY frame <= t - 30, per-query
prefix scan replicating _causal_pr_dump), n_candidates (= prefix size,
exhaustive).  With t >= 30 the prefix is never empty, so NaN rows are not
expected (guard kept for contract parity).

Output, one npz per (variant, seq):
  results/pd2026_pr/dump_{ring,ringpp}_{sensor}_{seq}.npz
  arrays query_idx / j_star / score_corr / top1_geo_dist / min_past_gt_dist /
  n_candidates + json-string meta.

Roster: Strict-v2 37 sequences (_pd_test_cache_manifest.json minus the five
excluded); scan_ids/poses from the pd2026 test operating caches via
_pdb_ring_eval.load_cache.  MEMORY DISCIPLINE (_causal_pr_dump convention):
the CLI accepts a sequence list, but the driver runs each sequence in its own
--worker subprocess.  Resumable per (sequence, variant) output file.

Run inside docker ring_baseline (GPU; torch_radon needs CUDA):
  python3 scripts/_causal_ring_pr.py --seqs kitti/05 helipr/Town01
  python3 scripts/_causal_ring_pr.py --roster        # print the 37 ids
  python3 scripts/_causal_ring_pr.py --self-test     # CPU-safe synthetic
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

from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402  (torch-free)

OUT = REPO / "results/pd2026_pr"
MANIFEST = REPO / "scripts/_pd_test_cache_manifest.json"
VARIANTS = ("ring", "ringpp")           # ring first: 1-ch correlation is ~6x cheaper
BEV_KEY = {"ring": "bevs_ring", "ringpp": "bevs_ringpp"}
BATCH_DEFAULT = 256                     # the _pdb_ring_eval rerank_fn batch
STRICT_V2_EXCLUDED = {
    ("kitti", "00"), ("kitti360", "0004"), ("helipr", "Bridge04"),
    ("nclt", "2012-02-04"), ("nclt", "2012-03-17"),
}


def manifest_map() -> dict:
    """(sensor, seq) -> operating-cache relpath.  Torch-free mirror of
    _pdb_ring_eval.load_manifest (kept local so --roster/--help never import
    torch); the worker still loads scan_ids/poses via ringmod.load_cache."""
    m = json.loads(MANIFEST.read_text())
    return {(s["sensor"], s["sequence"]): s["expected"]["operating_cache"]
            for s in m["sequences"]}


def strict_v2_roster() -> list:
    return [(s, q) for (s, q) in manifest_map() if (s, q) not in STRICT_V2_EXCLUDED]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------ scoring

def correlate_prefix(bank, t: int, n_prefix: int, batch: int, device: str,
                     yaw_fn) -> np.ndarray:
    """Normalized circular-correlation scores of query t against prefix rows
    [0, n_prefix) of the spec bank (higher = more similar).  Mirrors the
    _pdb_ring_eval.eval_variant rerank_fn batching; only rows < n_prefix are
    ever read (R2: candidate range is index arithmetic on a per-frame bank)."""
    import torch
    q = bank[t].float().to(device)
    out = np.empty(n_prefix, dtype=np.float64)
    with torch.no_grad():
        for j0 in range(0, n_prefix, batch):
            j1 = min(j0 + batch, n_prefix)
            cspec = bank[j0:j1].float().to(device)
            _, scores, _ = yaw_fn(
                q.unsqueeze(0).expand(j1 - j0, -1, -1, -1), cspec)
            out[j0:j1] = scores.detach().cpu().numpy().reshape(-1)
    return out


def precompute_fft_bank(specs, device: str, gpu_limit_bytes: int = 9 << 30):
    """Per-frame InstanceNorm + fft2(dim=-2, norm='ortho') computed ONCE.

    circorr2.circular_corr re-normalizes and re-FFTs both operands on every
    call, so a candidate j is re-transformed for every query whose prefix
    contains it (O(N^2) transforms). Both steps are per-frame (InstanceNorm2d
    (affine=False) normalizes each (sample, channel) over hxw independently;
    the FFT is per-frame), so hoisting them is mathematically exact — the
    fast scorer below reproduces the official corr = ifft2(fft2(q) *
    conj(fft2(c))) with identical operand values. Verified against completed
    slow-path dumps (kitti/05, kitti/08).
    Returns (fft_bank complex64, on_gpu flag)."""
    import torch
    import torch.nn as nn
    inorm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False)
    n, c, h, w = specs.shape
    on_gpu = device == "cuda" and (n * c * h * w * 8) <= gpu_limit_bytes
    out = torch.empty((n, c, h, w), dtype=torch.complex64,
                      device=device if on_gpu else "cpu")
    with torch.no_grad():
        for i0 in range(0, n, 512):
            i1 = min(i0 + 512, n)
            x = specs[i0:i1].float().to(device)
            x = inorm(x)
            f = torch.fft.fft2(x, dim=-2, norm="ortho")
            out[i0:i1] = f if on_gpu else f.cpu()
    return out, on_gpu


def correlate_prefix_fast(fft_bank, on_gpu: bool, t: int, n_prefix: int,
                          chunk: int, device: str) -> np.ndarray:
    """Vectorized replica of circorr2.circular_corr scoring over the prefix:
    corr = |ifft2(F_q * conj(F_c), dim=-2)| summed over channels and width,
    max over the rotation axis, / (0.15*c*h*w). Only rows < n_prefix are read
    (R2: candidate range is index arithmetic on a per-frame bank)."""
    import torch
    q = fft_bank[t]
    if not on_gpu:
        q = q.to(device)
    c, h, w = q.shape
    denom = 0.15 * c * h * w
    out = np.empty(n_prefix, dtype=np.float64)
    with torch.no_grad():
        for j0 in range(0, n_prefix, chunk):
            j1 = min(j0 + chunk, n_prefix)
            cf = fft_bank[j0:j1]
            if not on_gpu:
                cf = cf.to(device, non_blocking=True)
            corr = torch.fft.ifft2(q.unsqueeze(0) * cf.conj(),
                                   dim=-2, norm="ortho")
            mag = torch.sqrt(corr.real ** 2 + corr.imag ** 2 + 1e-15)
            s = mag.sum(dim=-3).sum(dim=-1).max(dim=-1)[0] / denom
            out[j0:j1] = s.double().cpu().numpy()
    return out


def dump_variant(bank, positions: np.ndarray, min_past: np.ndarray,
                 batch: int, device: str, yaw_fn, tag: str = "",
                 scorer=None, n_frames: int = None) -> tuple:
    """Per-query ledger arrays for one variant (see module docstring).
    Returns (arrays dict, n_correlations). When `scorer(t, n_prefix)` is
    given it replaces the slow per-batch official-call path (bank/yaw_fn
    unused then; pass n_frames)."""
    n = n_frames if scorer is not None else len(bank)
    queries = np.arange(MIN_INDEX_GAP, n, dtype=np.int64)
    j_star = np.full(len(queries), -1, dtype=np.int64)
    score_corr = np.full(len(queries), np.nan)
    top1_geo = np.full(len(queries), np.nan)
    n_cand = np.zeros(len(queries), dtype=np.int64)
    n_corr = 0
    t0 = time.time()
    for qi, t in enumerate(queries):
        t = int(t)
        n_prefix = t - MIN_INDEX_GAP + 1  # prefix rows [0, t - 30]
        if n_prefix <= 0:
            continue  # empty pool -> NaN row (not expected, t >= 30)
        if scorer is not None:
            s = scorer(t, n_prefix)
        else:
            s = correlate_prefix(bank, t, n_prefix, batch, device, yaw_fn)
        j = int(np.argmax(s))  # argmin of dist = 1 - s; first-max tie-break
        assert j <= t - MIN_INDEX_GAP, "candidate beyond the causal prefix"
        j_star[qi] = j
        score_corr[qi] = float(s[j]) - 1.0  # -(1 - corr): negated official dist
        top1_geo[qi] = float(np.linalg.norm(positions[j] - positions[t]))
        n_cand[qi] = n_prefix
        n_corr += n_prefix
        if qi and qi % 200 == 0:
            rate = (qi + 1) / (time.time() - t0)
            eta = (len(queries) - qi - 1) / rate / 60
            print(f"  {tag} queries {qi}/{len(queries)} "
                  f"({rate:.1f} q/s, ETA {eta:.0f} min)", flush=True)
    arrays = dict(query_idx=queries, j_star=j_star, score_corr=score_corr,
                  top1_geo_dist=top1_geo, min_past_gt_dist=min_past[queries],
                  n_candidates=n_cand)
    return arrays, n_corr


# ------------------------------------------------------------------- worker

def load_bev_cache_readonly(cache_dir: Path, sensor: str, seq: str,
                            scan_ids: np.ndarray):
    """READ-ONLY open of the shared per-sequence BEV cache with the
    _xs_ring_eval.load_sliced_bevs guard: full-sequence scan_ids must match;
    NEVER build/rebuild (compute_bevs overwrites on mismatch)."""
    p = cache_dir / f"bev_{sensor}_{seq}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"missing BEV cache {p} — build it separately inside docker "
            f"ring_baseline (python3 scripts/_pdb_ring_eval.py {sensor}/{seq} "
            f"--bev-only); this driver never builds BEVs")
    z = np.load(p)
    if not np.array_equal(z["scan_ids"], scan_ids):
        raise AssertionError(
            f"{p.name}: cached scan_ids != operating-cache scan_ids; refusing "
            f"(this driver never rebuilds the BEV cache)")
    return z, p


def run_worker(args, sensor: str, seq: str):
    import torch
    import _pdb_ring_eval as ringmod  # reuse load_cache/compute_specs/
                                      # CACHE_DIR/GRID/BOUNDS ONLY — never
                                      # compute_bevs (cache clobber hazard)
    from glnet.models.utils import estimate_yaw  # official correlation

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA unavailable "
                           "(torch_radon needs CUDA; run in docker ring_baseline)")
    cache_rel = manifest_map()[(sensor, seq)]
    scan_ids, poses = ringmod.load_cache(cache_rel)
    positions = poses[:, :3, 3]
    n = len(scan_ids)
    print(f"[{sensor}/{seq}] {n} keyframes cache={cache_rel}", flush=True)

    # min 3D distance from t to ANY frame <= t - 30 (query classification
    # input; per-query prefix scan replicating _causal_pr_dump — never a
    # full-sequence statistic).
    min_past = np.full(n, np.nan)
    for t in range(MIN_INDEX_GAP, n):
        d = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t],
                           axis=1)
        min_past[t] = d.min()

    bev_npz, bev_path = load_bev_cache_readonly(
        ringmod.CACHE_DIR, sensor, seq, scan_ids)
    bev_sha = sha256_of(bev_path)
    cache_sha = sha256_of(REPO / cache_rel)

    for variant in [v for v in VARIANTS if v in args.variants]:
        out_path = OUT / f"dump_{variant}_{sensor}_{seq}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[{sensor}/{seq}] {variant}: {out_path.name} exists, "
                  f"skipping", flush=True)
            continue
        t0 = time.time()
        bevs = bev_npz[BEV_KEY[variant]]
        specs = ringmod.compute_specs(bevs, device)  # fp16 CPU tensor
        del bevs
        spec_s = time.time() - t0
        if args.fast:
            fft_bank, on_gpu = precompute_fft_bank(specs, device)
            n_var = len(specs)
            del specs
            print(f"[{sensor}/{seq}] {variant}: fft bank "
                  f"{tuple(fft_bank.shape)} ({spec_s:.0f}s specs), on "
                  f"{'gpu' if on_gpu else 'cpu (streamed)'}", flush=True)
            scorer = (lambda t, n_prefix: correlate_prefix_fast(
                fft_bank, on_gpu, t, n_prefix, args.fast_chunk, device))
            t1 = time.time()
            arrays, n_corr = dump_variant(None, positions, min_past,
                                          args.batch, device, None,
                                          tag=f"[{sensor}/{seq}] {variant}",
                                          scorer=scorer, n_frames=n_var)
            del fft_bank
        else:
            bank = specs if args.specs_on_cpu else specs.to(device)
            del specs
            print(f"[{sensor}/{seq}] {variant}: specs {tuple(bank.shape)} "
                  f"({spec_s:.0f}s), bank on "
                  f"{'cpu (streamed)' if args.specs_on_cpu else device}",
                  flush=True)
            t1 = time.time()
            arrays, n_corr = dump_variant(bank, positions, min_past,
                                          args.batch, device, estimate_yaw,
                                          tag=f"[{sensor}/{seq}] {variant}")
        wall = time.time() - t1
        if not args.fast:
            del bank
        if device == "cuda":
            torch.cuda.empty_cache()
        nq = len(arrays["query_idx"])
        meta = {
            "schema": "pr-dump-v1",
            "method": variant,
            "sensor": sensor, "sequence": seq,
            "min_index_gap": MIN_INDEX_GAP,
            "n_frames": n, "n_queries": nq,
            "grid": ringmod.GRID, "bounds": list(ringmod.BOUNDS),
            "bev_channels": 1 if variant == "ring" else 6,
            "device": device, "specs_on_cpu": bool(args.specs_on_cpu),
            "fast_path": bool(args.fast),
            "fast_note": ("per-frame InstanceNorm+FFT hoisted out of the "
                          "pair loop (mathematically exact vs circorr2; "
                          "verified vs slow-path dumps)" if args.fast else ""),
            "score_convention": (
                "similarity, higher = accept; score_corr = corr_max - 1 = "
                "-(official dist = 1 - normalized circular correlation, "
                "glnet estimate_yaw) of the prefix argmin(dist) — negated "
                "per prereg §1.5 / output contract"),
            "budget": {
                "candidates": "exhaustive over prefix j <= t - 30, no caps",
                "n_correlations": int(n_corr),
                "batch": args.batch,
                "wall_s": round(wall, 1),
                "ms_per_query": round(1e3 * wall / max(nq, 1), 1),
                "spec_s": round(spec_s, 1),
            },
            "artifacts": {
                "bev_cache": str(bev_path), "bev_cache_sha256": bev_sha,
                "operating_cache": str(REPO / cache_rel),
                "operating_cache_sha256": cache_sha,
            },
            "protocol": (
                "causal all-frames PR ledger (docs/CAUSAL_PR_PREREGISTRATION"
                ".md §1.1/§1.2/§1.5): all queries t >= 30; per-frame TIRING "
                "specs (official radon sinogram -> row-FFT magnitude, "
                "_pdb_ring_eval.compute_specs) precomputed once — per-frame, "
                "causal-safe; per query t the OFFICIAL circular correlation "
                "(estimate_yaw) is evaluated exhaustively against every "
                "prefix frame j <= t - 30 (candidate range = index "
                "arithmetic, never global-search-then-filter); "
                "min_past_gt_dist = per-query prefix scan"),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
        fin = np.isfinite(arrays["score_corr"])
        print(f"[{sensor}/{seq}] {variant}: n_queries={nq} "
              f"finite={int(fin.sum())} corr={n_corr} ({wall:.0f}s) "
              f"-> {out_path.name}", flush=True)


# ---------------------------------------------------------------- self-test

def _toy_yaw(q, c):
    """CPU stand-in with the estimate_yaw call contract (corr, score, angle);
    score = cosine of the flattened pair.  --self-test only — verifies the
    LOOP mechanics (prefix limits, argmin, negation, truncation invariance),
    not the official correlation itself."""
    import torch
    qf = q.reshape(len(c), -1).float()
    cf = c.reshape(len(c), -1).float()
    s = torch.nn.functional.cosine_similarity(qf, cf, dim=1)
    return None, s, None


def self_test():
    import torch
    rng = np.random.default_rng(0)
    n = 64
    specs = torch.from_numpy(
        rng.standard_normal((n, 2, 8, 6)).astype(np.float16))
    specs[50] = specs[7]  # planted revisit: query 50 duplicates frame 7 (<= 20)
    positions = (rng.standard_normal((n, 3)) * 100).astype(np.float64)
    min_past = np.full(n, np.nan)
    for t in range(MIN_INDEX_GAP, n):
        min_past[t] = np.linalg.norm(
            positions[:t - MIN_INDEX_GAP + 1] - positions[t], axis=1).min()
    # brute-force min_past cross-check
    for t in range(MIN_INDEX_GAP, n):
        ref = min(np.linalg.norm(positions[j] - positions[t])
                  for j in range(0, t - MIN_INDEX_GAP + 1))
        assert abs(min_past[t] - ref) < 1e-9

    def run(yaw_fn, label):
        arrays, n_corr = dump_variant(specs, positions, min_past, 5, "cpu",
                                      yaw_fn, tag=f"[self-test {label}]")
        q = arrays["query_idx"]
        assert np.array_equal(q, np.arange(MIN_INDEX_GAP, n)), "R3 all-frames"
        assert np.all(arrays["n_candidates"] == q - MIN_INDEX_GAP + 1), \
            "exhaustive prefix pool size"
        assert np.all(arrays["j_star"] <= q - MIN_INDEX_GAP), "R2 causality"
        assert np.all(np.isfinite(arrays["score_corr"])), "no NaN expected"
        assert n_corr == int((q - MIN_INDEX_GAP + 1).sum())
        # planted revisit found, score = corr - 1 with corr at its max
        i50 = int(np.flatnonzero(q == 50)[0])
        assert arrays["j_star"][i50] == 7, "planted duplicate not retrieved"
        _, s_ref, _ = yaw_fn(specs[50].float().unsqueeze(0),
                             specs[7].float().unsqueeze(0))
        assert abs(arrays["score_corr"][i50] - (float(s_ref[0]) - 1.0)) < 1e-6, \
            "negation record mismatch (score != corr - 1)"
        assert abs(arrays["top1_geo_dist"][i50]
                   - np.linalg.norm(positions[7] - positions[50])) < 1e-9
        # batch invariance
        arrays_b1, _ = dump_variant(specs, positions, min_past, 1, "cpu",
                                    yaw_fn, tag=f"[self-test {label} b1]")
        assert np.allclose(arrays["score_corr"], arrays_b1["score_corr"],
                           atol=1e-6), "batching changes scores"
        # truncation invariance: row t computed from the full bank == row t
        # computed from the bank truncated at t (prefix-only data touched)
        t_cut = 45
        arrays_cut, _ = dump_variant(specs[:t_cut + 1], positions[:t_cut + 1],
                                     min_past, 5, "cpu", yaw_fn,
                                     tag=f"[self-test {label} cut]")
        it = int(np.flatnonzero(q == t_cut)[0])
        assert arrays_cut["j_star"][-1] == arrays["j_star"][it]
        assert abs(arrays_cut["score_corr"][-1]
                   - arrays["score_corr"][it]) < 1e-6, "truncation variance"
        print(f"self-test [{label}]: OK", flush=True)

    run(_toy_yaw, "toy")
    try:
        sys.path.insert(0, str(REPO / "_external" / "RINGSharp"))
        from glnet.models.utils import estimate_yaw
    except Exception as e:  # host without glnet deps: mechanics already proven
        print(f"self-test: official estimate_yaw unavailable ({e!r}); "
              f"toy-scorer mechanics test passed, official-scorer pass "
              f"runs in docker ring_baseline", flush=True)
    else:
        run(estimate_yaw, "official-cpu")
    print("SELF_TEST_OK", flush=True)


# ------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seqs", nargs="+", metavar="SENSOR/SEQ",
                    help="e.g. kitti/05 (Strict-v2 roster only)")
    ap.add_argument("--variant", choices=("both",) + VARIANTS, default="both")
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT,
                    help="estimate_yaw candidate batch (default 256)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fast", dest="fast", action="store_true", default=True,
                    help="vectorized scorer with per-frame precomputed "
                         "InstanceNorm+FFT (default; exact math)")
    ap.add_argument("--no-fast", dest="fast", action="store_false")
    ap.add_argument("--fast-chunk", type=int, default=1024,
                    help="candidate rows per fused GPU op in --fast mode")
    ap.add_argument("--specs-on-cpu", action="store_true",
                    help="keep the spec bank on CPU and stream candidate "
                         "batches (less GPU memory, more PCIe traffic)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--roster", action="store_true",
                    help="print the Strict-v2 37 sensor/seq ids and exit")
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-safe synthetic loop-mechanics test, no data")
    ap.add_argument("--worker", action="store_true",
                    help="internal: run a single sequence in-process")
    args = ap.parse_args()
    args.variants = VARIANTS if args.variant == "both" else (args.variant,)

    if args.self_test:
        self_test()
        return
    if args.roster:
        for sensor, seq in strict_v2_roster():
            print(f"{sensor}/{seq}")
        return
    if not args.seqs:
        ap.error("--seqs required (or --roster / --self-test)")

    mm = manifest_map()
    for s in args.seqs:
        assert s.count("/") == 1, f"--seqs entries must be sensor/seq, got {s!r}"
        sensor, seq = s.split("/")
        if (sensor, seq) in STRICT_V2_EXCLUDED:
            sys.exit(f"{s} is excluded from the Strict-v2 roster (R5)")
        if (sensor, seq) not in mm:
            sys.exit(f"{s} not in scripts/_pd_test_cache_manifest.json")

    if args.worker:
        assert len(args.seqs) == 1, "--worker takes exactly one sequence"
        sensor, seq = args.seqs[0].split("/")
        run_worker(args, sensor, seq)
        return

    # Driver: one heavy sequence per subprocess (memory discipline,
    # _causal_pr_dump convention).
    OUT.mkdir(parents=True, exist_ok=True)
    for s in args.seqs:
        sensor, seq = s.split("/")
        outs = [OUT / f"dump_{v}_{sensor}_{seq}.npz" for v in args.variants]
        if not args.overwrite and all(p.exists() for p in outs):
            print(f"=== {s}: all dumps exist, skipping ===", flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--seqs", s, "--variant", args.variant,
               "--batch", str(args.batch), "--device", args.device,
               "--fast-chunk", str(args.fast_chunk), "--worker"]
        if not args.fast:
            cmd.append("--no-fast")
        if args.specs_on_cpu:
            cmd.append("--specs-on-cpu")
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"=== spawning worker: {s} ===", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"WORKER_FAILED {s} rc={rc}", flush=True)
            sys.exit(rc)
    print("CAUSAL_RING_PR_DONE", flush=True)


if __name__ == "__main__":
    main()
