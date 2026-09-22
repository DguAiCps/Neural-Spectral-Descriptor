#!/usr/bin/env python3
"""Torch-free shared library for the cross-session benchmark.

Protocol (frozen in scripts/_xs_protocols.json): "RING-style cross-session
protocol under a common keyframe roster" — map session subsampled at 20 m,
query session at 5 m, query roster = spaced queries with >= 1 map pose within
20 m, built ONCE and reused at EVERY report threshold (fixed denominator; no
per-threshold re-filtering). Primary R@1@5m, auxiliary R@1@20m and R@1%@20m
with k_1pct = max(int(round(n_db / 100.0)), 1).

Hard rules enforced here:
  * Candidate generation is DB-only: scores = db @ query, top pool_k over DB
    rows exclusively; every candidate array is asserted < n_db. [db; query]
    concatenation exists ONLY for alignment-bank indexing (align_scores /
    column_align_scores keep their original single-array signatures).
  * Poses come from ONE canonical source for every method:
    results/pd2026/keys_pd_s1_{sensor}_{seq}_gp.npz (global-GT-fixed frames;
    the pd2026 operating caches are per-session re-centred and must never
    feed cross-session distances).
  * Every per-session array is sliced by the same spacing index set via
    take() (asserts equal leading lengths first).

NSD fusion helpers are lifted verbatim from scripts/_height_candidate_fusion.py
(align_scores :301, column_align_bank :308, column_align_scores :318,
minmax :329, l2 :67), scripts/_pbev_branch_fusion.py (variant_inputs :75) and
scripts/_hspec_branch_fusion.py (hspec_inputs :37) — those modules import
torch at module level and cannot be loaded on the host. Bit-identity of the
lifted copies is verified once against the docker originals by
scripts/_xs_verify_lift.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROTOCOLS_PATH = REPO / "scripts" / "_xs_protocols.json"
KEYS_DIR = REPO / "results" / "pd2026"
OUT_DIR = REPO / "results" / "pd2026_xsession"

_SPEC = json.loads(PROTOCOLS_PATH.read_text())
CONST = _SPEC["constants"]
PROTOCOLS = _SPEC["protocols"]
SITES = _SPEC["sites"]
PROTOCOL_NAME = _SPEC["name"]


# ---------------------------------------------------------------- sessions

def load_global_session(sensor: str, seq: str) -> dict:
    """Canonical scan_ids + global positions for one session (pose single
    source; seed choice irrelevant — poses are identical across seeds)."""
    p = KEYS_DIR / f"keys_pd_s1_{sensor}_{seq}_gp.npz"
    with np.load(p, allow_pickle=True) as z:
        scan_ids = z["scan_ids"].copy()
        pos = z["poses"][:, :3, 3].astype(np.float64)
    if CONST["planar"]:
        pos = pos.copy()
        pos[:, 2] = 0.0
    return {"scan_ids": scan_ids, "positions": pos, "path": str(p)}


def spacing_indices(positions: np.ndarray, spacing_m: float | None) -> np.ndarray:
    """Greedy trajectory-order downsampling, verbatim from
    scripts/evaluate_cross_session._spacing_indices."""
    if spacing_m is None or spacing_m <= 0 or len(positions) == 0:
        return np.arange(len(positions), dtype=np.int64)
    keep = [0]
    for i in range(1, len(positions)):
        if np.linalg.norm(positions[i] - positions[keep[-1]]) >= spacing_m:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def _min_db_dists(db_pos: np.ndarray, q_pos: np.ndarray) -> np.ndarray:
    """Per-query min distance to any DB pose (chunked; n_db is small)."""
    out = np.empty(len(q_pos), dtype=np.float64)
    for i0 in range(0, len(q_pos), 1024):
        d = np.linalg.norm(q_pos[i0:i0 + 1024, None, :] - db_pos[None, :, :],
                           axis=-1)
        out[i0:i0 + 1024] = d.min(axis=1)
    return out


def build_roster(db_pos: np.ndarray, q_pos: np.ndarray,
                 radius_m: float) -> np.ndarray:
    """Indices into the SPACED query arrays with >= 1 DB pose within radius.
    Built once; the same roster is the denominator at every threshold."""
    return np.flatnonzero(_min_db_dists(db_pos, q_pos) < radius_m)


def make_pair(protocol: str) -> dict:
    spec = PROTOCOLS[protocol]
    sensor = spec["sensor"]
    db_s = load_global_session(sensor, spec["db_seq"])
    q_s = load_global_session(sensor, spec["query_seq"])
    db_idx = spacing_indices(db_s["positions"], CONST["db_spacing_m"])
    q_idx = spacing_indices(q_s["positions"], CONST["query_spacing_m"])
    db_pos = db_s["positions"][db_idx]
    q_pos = q_s["positions"][q_idx]
    roster = build_roster(db_pos, q_pos, CONST["roster_radius_m"])
    db_sids = db_s["scan_ids"][db_idx]
    q_sids = q_s["scan_ids"][q_idx]
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(db_sids).tobytes())
    h.update(np.ascontiguousarray(q_sids[roster]).tobytes())
    # Sanity: roster distances must be consistent with the radius rule.
    md = _min_db_dists(db_pos, q_pos[roster])
    assert len(roster) == 0 or (np.median(md) <= CONST["roster_radius_m"]
                                and md.max() < CONST["roster_radius_m"]), protocol
    return {
        "protocol": protocol, "sensor": sensor,
        "db_seq": spec["db_seq"], "query_seq": spec["query_seq"],
        "db_idx": db_idx, "q_idx": q_idx,
        "db_pos": db_pos, "q_pos": q_pos, "roster": roster,
        "db_scan_ids": db_sids, "q_scan_ids": q_sids,
        "db_session_scan_ids": db_s["scan_ids"], "q_session_scan_ids": q_s["scan_ids"],
        "roster_sha256": h.hexdigest(),
        "n_raw_db": len(db_s["scan_ids"]), "n_raw_q": len(q_s["scan_ids"]),
    }


def take(arrays: dict, idx: np.ndarray) -> dict:
    """THE spacing-consistency enforcement point: every array must share the
    leading length, and all are sliced with one index set."""
    lens = {k: len(v) for k, v in arrays.items()}
    assert len(set(lens.values())) == 1, f"leading-length mismatch: {lens}"
    return {k: v[idx] for k, v in arrays.items()}


# ---------------------------------------------------------------- ranking

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32).copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def db_only_candidates(db_desc: np.ndarray, q_vec: np.ndarray,
                       pool_k: int) -> np.ndarray:
    """DB-only candidate generation (advisor-mandated form)."""
    scores = db_desc @ q_vec
    candidates = np.argsort(-scores, kind="stable")[:pool_k]
    assert np.all(candidates < len(db_desc))
    return candidates


def rank_cross_session(db_desc: np.ndarray, q_desc: np.ndarray,
                       roster: np.ndarray, *, rerank_fn=None,
                       n_coarse: int = 200) -> list:
    """Cosine coarse ranking over DB rows only, optional method rerank.

    rerank_fn(q_concat_idx, candidates) -> reordered candidates, where the
    driver has concatenated [db; query] method arrays so existing rerank
    workers keep their single-array signature: the query row is
    n_db + roster_index_into_spaced_arrays, candidates stay in [0, n_db).
    """
    dbn = normalize(db_desc)
    qn = normalize(q_desc)
    n_db = len(dbn)
    ranked_lists = []
    for ri in roster:
        cand = db_only_candidates(dbn, qn[ri], min(n_coarse, n_db))
        if rerank_fn is not None:
            cand = np.asarray(rerank_fn(n_db + int(ri), cand))
            assert np.all(cand < n_db)
        ranked_lists.append(cand)
    return ranked_lists


# ---------------------------------------------------------------- scoring

def score_rankings(ranked_lists: list, db_pos: np.ndarray, q_pos: np.ndarray,
                   roster: np.ndarray) -> dict:
    """Fixed-roster metrics. len(roster) is the denominator at EVERY
    threshold — never re-filtered per threshold."""
    thresholds = [float(t) for t in CONST["thresholds_m"]]
    k_values = [int(k) for k in CONST["k_values"]]
    n_db = len(db_pos)
    k_1pct = max(int(round(n_db / 100.0)), 1)
    n_roster = len(roster)
    assert len(ranked_lists) == n_roster

    hits = {}
    for t in thresholds:
        for k in k_values:
            hits[(k, t)] = 0
        hits[("1pct", t)] = 0
    for ranked, qi in zip(ranked_lists, roster):
        ranked = np.asarray(ranked)
        d = np.linalg.norm(db_pos[ranked] - q_pos[qi], axis=1)
        for t in thresholds:
            for k in k_values:
                hits[(k, t)] += bool((d[:k] < t).any())
            hits[("1pct", t)] += bool((d[:k_1pct] < t).any())

    denom = max(n_roster, 1)
    metrics = {}
    for t in thresholds:
        ts = f"{t:g}m"
        for k in k_values:
            metrics[f"R@{k}@{ts}"] = hits[(k, t)] / denom
        metrics[f"R1pct@{ts}"] = hits[("1pct", t)] / denom
    metrics["k_1pct"] = k_1pct

    md = _min_db_dists(db_pos, q_pos[roster]) if n_roster else np.empty(0)
    diagnostics = {f"matchable@{t:g}m": int((md < t).sum()) for t in thresholds}
    diagnostics["n_roster"] = n_roster
    # The invariant the advisor mandated: one denominator for all thresholds.
    diagnostics["denominator_per_threshold"] = {f"{t:g}m": denom for t in thresholds}
    return {"metrics": metrics, "diagnostics": diagnostics}


# ------------------------------------------------- lifted NSD fusion helpers
# Verbatim math from scripts/_height_candidate_fusion.py (torch-importing
# module) — verified bit-identical by scripts/_xs_verify_lift.py.

def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def minmax(x: np.ndarray):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / max(hi - lo, 1e-8)


def align_scores(fft: np.ndarray, norm: np.ndarray, query: int,
                 candidates: np.ndarray):
    corr = np.fft.irfft(fft[candidates] * np.conj(fft[query]),
                        n=(fft.shape[-1] - 1) * 2, axis=-1)
    corr = corr.sum(axis=tuple(range(1, corr.ndim - 1))).max(axis=-1)
    return corr / (norm[candidates] * norm[query])


def column_align_bank(grid: np.ndarray):
    x = grid.astype(np.float32)
    col_norm = np.linalg.norm(x, axis=1, keepdims=True)
    valid = (col_norm[:, 0] > 1e-8).astype(np.float32)
    x = x / np.clip(col_norm, 1e-8, None)
    return (np.fft.rfft(x, axis=-1).astype(np.complex64),
            np.fft.rfft(valid, axis=-1).astype(np.complex64))


def column_align_scores(bank, query: int, candidates: np.ndarray):
    fft, mask_fft = bank
    n_sector = (fft.shape[-1] - 1) * 2
    corr = np.fft.irfft(fft[candidates] * np.conj(fft[query]),
                        n=n_sector, axis=-1).sum(axis=1)
    count = np.fft.irfft(mask_fft[candidates] * np.conj(mask_fft[query]),
                         n=n_sector, axis=-1)
    score = np.where(count > 0.5, corr / np.maximum(count, 1.0), -1.0)
    return score.max(axis=-1)


def variant_inputs(sk: np.ndarray, n_freqs: int):
    """scripts/_pbev_branch_fusion.variant_inputs, verbatim."""
    skf = sk[:, :, :n_freqs]
    hkey = l2(np.abs(skf).reshape(len(skf), -1).astype(np.float32))
    hfft = np.zeros((len(skf), 16, 31), np.complex64)
    hfft[:, :, 1:n_freqs + 1] = skf
    hnorm = np.clip(np.linalg.norm(np.abs(skf).reshape(len(skf), -1), axis=1),
                    1e-8, None)
    recon = np.fft.irfft(hfft, n=60, axis=-1).astype(np.float32)
    return hkey, hfft, hnorm, column_align_bank(recon)


def hspec_inputs(raw_height: np.ndarray, n_freqs: int):
    """scripts/_hspec_branch_fusion.hspec_inputs, verbatim (keeps the
    f16 -> f32 cast before the rfft)."""
    coeffs = np.fft.rfft(raw_height.astype(np.float32), axis=-1, norm="ortho")
    sk = coeffs[:, :, 1:n_freqs + 1].astype(np.complex64)
    hkey = l2(np.abs(sk).reshape(len(sk), -1))
    n_bins = raw_height.shape[-1] // 2 + 1
    hfft = np.zeros((len(sk), raw_height.shape[1], n_bins), np.complex64)
    hfft[:, :, 1:n_freqs + 1] = sk
    hnorm = np.clip(np.linalg.norm(np.abs(sk).reshape(len(sk), -1), axis=1),
                    1e-8, None)
    recon = np.fft.irfft(hfft, n=raw_height.shape[-1], axis=-1).astype(np.float32)
    return hkey, hfft, hnorm, column_align_bank(recon)


def _concat_bank(db_bank, q_bank):
    """Concatenate per-side (hkey, hfft, hnorm, colbank) tuples so the
    original single-array alignment helpers index [db; query] rows."""
    hkey = np.concatenate([db_bank[0], q_bank[0]])
    hfft = np.concatenate([db_bank[1], q_bank[1]])
    hnorm = np.concatenate([db_bank[2], q_bank[2]])
    colbank = (np.concatenate([db_bank[3][0], q_bank[3][0]]),
               np.concatenate([db_bank[3][1], q_bank[3][1]]))
    return hkey, hfft, hnorm, colbank


def nsd_rank_fused(db_key: np.ndarray, q_key: np.ndarray,
                   db_bank, q_bank, roster: np.ndarray,
                   w, pool_k: int) -> list:
    """Cross-session NSD retrieval: two DB-only top-pool_k cosine indexes
    (416D key + variant magnitude) -> union -> per-query min-max 4-channel
    fusion at the frozen tuple. Semantics of _pd_apply_frozen.evaluate_fixed
    with candidate generation replaced by the DB-only form (no skip window,
    no self-exclusion; union kept at pool_k per index — never widened to the
    full DB, preserving the normalization regime the tuple was selected in)."""
    dbk = normalize(db_key)
    qk = normalize(q_key)
    n_db = len(dbk)
    hkey_cat, hfft_cat, hnorm_cat, colbank_cat = _concat_bank(db_bank, q_bank)
    db_mag, q_mag = hkey_cat[:n_db], hkey_cat[n_db:]
    ranked_lists = []
    for ri in roster:
        cn = db_only_candidates(dbk, qk[ri], min(pool_k, n_db))
        ch = db_only_candidates(db_mag, q_mag[ri], min(pool_k, n_db))
        cand = np.unique(np.concatenate([cn, ch]))
        assert np.all(cand < n_db)
        sn = minmax(dbk[cand] @ qk[ri])
        sh = minmax(db_mag[cand] @ q_mag[ri])
        sa = minmax(align_scores(hfft_cat, hnorm_cat, n_db + int(ri), cand))
        sd = minmax(column_align_scores(colbank_cat, n_db + int(ri), cand))
        score = w[0] * sn + w[1] * sh + w[2] * sa + w[3] * sd
        ranked_lists.append(cand[np.argsort(-score, kind="stable")])
    return ranked_lists


# ---------------------------------------------------------------- output

def result_payload(method: str, pair: dict, scored: dict, *,
                   variant=None, seed=None, op=None, extra=None) -> dict:
    payload = {
        "schema": "xs-eval-v1",
        "protocol_name": PROTOCOL_NAME,
        "method": method,
        "protocol": pair["protocol"],
        "db": {"sensor": pair["sensor"], "seq": pair["db_seq"],
               "spacing_m": CONST["db_spacing_m"],
               "n_raw": pair["n_raw_db"], "n_db": len(pair["db_pos"])},
        "query": {"sensor": pair["sensor"], "seq": pair["query_seq"],
                  "spacing_m": CONST["query_spacing_m"],
                  "n_raw": pair["n_raw_q"], "n_spaced": len(pair["q_pos"]),
                  "n_roster": len(pair["roster"]),
                  "roster_radius_m": CONST["roster_radius_m"]},
        "planar": CONST["planar"],
        "roster_sha256": pair["roster_sha256"],
        "pose_source": CONST["pose_source"],
        "metrics": scored["metrics"],
        "diagnostics": scored["diagnostics"],
    }
    if variant is not None:
        payload["variant"] = variant
    if seed is not None:
        payload["seed"] = seed
    if op is not None:
        payload["op"] = op
    if extra:
        payload.update(extra)
    return payload


def write_result(payload: dict, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2))
    return p
