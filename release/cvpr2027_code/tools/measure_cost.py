#!/usr/bin/env python3
"""Measure the cost of NSD when keyframes arrive one at a time (cost columns of Table 1).

    python tools/encode_sequence.py --dataset mulran --root MulRan --sequence Sejong01 --out encoded/mulran_Sejong01.npz
    python tools/measure_cost.py --dataset mulran --root MulRan --sequence Sejong01 \
        --encoded encoded/mulran_Sejong01.npz --weights assets/weights/s4.pth [--n-map 10000]

The map is first filled with the keyframes of the encoded file. Then 96 keyframes arrive one by one, and the same
96 arrivals are repeated three times. The first query sees ``--n-map`` eligible candidates. Each arrival t runs

    encode   raw scan -> spectral key d and complex DFT features z
    update   add the place to both indexes and recompute the refined spectra of keyframes t-10..t from the
             window t-20..t (the spectrum of t-10 is final from this arrival on)
    query    decide q = t - 15: candidate search over j <= q - 30, alignment, temporal evidence, acceptance score

Scans are held in memory; reading from disk, LiDAR odometry and the wait for the 15 later keyframes are not timed.
After the timing, the returned places and scores are compared with the full-sequence computation of
``nsd.model.refine`` and ``nsd.matching.match_sequence``.

The steps are the package computations with repeated work removed: the two projections of the encoder share their
polar coordinates, the window graph keeps its fixed edges and gathers the neighbour terms of a refining step at
once, the acceptance score runs its two banks together, and the alignment cues sum the 16 range bins before the
inverse FFT. All but the last are bitwise equal to the package; the last changes the cues in float32 rounding.

Resident state counts what stays in memory per place (r, m, refined spectrum, pose, arc length), the last 21 raw
(d, z) used by the refining window, and the constants (network and assets).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from nsd import deterministic, index, matching  # noqa: E402
from nsd import encoding as E  # noqa: E402
from nsd.data import DATASETS, open_sequence  # noqa: E402
from nsd.model import (ETA, SLOTS, WINDOW, build_graph, edge_geometry, load_refiner, odometry_yaw,  # noqa: E402
                       refine, temporal_edges)

ARRIVALS = 96
REPETITIONS = 3
WARMUP = 12
RAW = 4 * WINDOW + 1          # raw (d, z) kept for the refining window t-20..t
PROVISIONAL = 2 * WINDOW + 1  # refined spectra rewritten at each arrival: t-10..t


def encode(points: np.ndarray, dataset: str) -> tuple:
    """``nsd.encoding.encode_scan`` with the polar coordinates computed once for both projections.

    ``range_image`` and ``max_height_bev`` each select the finite points and compute x^2 + y^2 and the azimuth of
    the same points; here the two projections share them, and the azimuth is computed only for the points that
    one of the two projections keeps (most of the others are empty returns at the origin). The arithmetic is
    otherwise that of the package, and the result is compared bitwise with the encoded file at every arrival.
    """
    x, y, z = E._finite_xyz(points)
    horizontal_sq = x ** 2 + y ** 2
    horizontal = np.sqrt(horizontal_sq)
    rng = np.sqrt(horizontal_sq + z ** 2)
    in_image = (rng >= E.MIN_RANGE_M) & (rng <= E.MAX_RANGE_M) & np.isfinite(rng)
    in_bev = (horizontal >= E.BEV_MIN_RANGE_M) & (horizontal < E.BEV_MAX_RANGE_M) & (z > E.BEV_MIN_HEIGHT_M)
    used = in_image | in_bev
    azimuth = np.zeros_like(x)
    azimuth[used] = E._azimuth(x[used], y[used])

    # invariant key d (E.range_image, then the package steps)
    elevation = np.arctan2(z[in_image], horizontal[in_image])
    low, high = np.deg2rad(E.ELEVATION_RANGE_DEG[dataset][0]), np.deg2rad(E.ELEVATION_RANGE_DEG[dataset][1])
    row = np.clip(np.floor((elevation - low) / (high - low) * E.N_ROWS).astype(int), 0, E.N_ROWS - 1)
    col = np.clip(np.floor(azimuth[in_image] / (2 * np.pi) * E.N_AZIMUTH).astype(int), 0, E.N_AZIMUTH - 1)
    image = np.full(E.N_ROWS * E.N_AZIMUTH, np.inf, dtype=np.float32)
    np.minimum.at(image, row * E.N_AZIMUTH + col, rng[in_image])
    image[image == np.inf] = 0.0
    image = E.fill_empty_cells(image.reshape(E.N_ROWS, E.N_AZIMUTH))
    d = E.octave_band_statistics(E.row_dft_magnitudes(image))

    # complex features z (E.max_height_bev, then the package steps)
    n_rings = int(E.BEV_MAX_RANGE_M - E.BEV_MIN_RANGE_M)
    ring = np.clip(np.floor(horizontal[in_bev] - E.BEV_MIN_RANGE_M).astype(int), 0, n_rings - 1)
    sector = np.clip(np.floor(azimuth[in_bev] / (2 * np.pi) * E.BEV_SECTORS).astype(int), 0, E.BEV_SECTORS - 1)
    bev = np.zeros(n_rings * E.BEV_SECTORS, dtype=np.float32)
    np.maximum.at(bev, ring * E.BEV_SECTORS + sector, z[in_bev].astype(np.float32))
    features = E.low_frequency_coefficients(E.average_rings(bev.reshape(n_rings, E.BEV_SECTORS)))
    return d, features


def refine_spectra(spectrum: torch.Tensor, neighbour: torch.Tensor, mask: torch.Tensor, phase: torch.Tensor,
                   attention: torch.Tensor) -> torch.Tensor:
    """``nsd.model.refine_spectra`` with the neighbour terms of a step computed in one operation.

    The terms and the order in which they are summed (slot by slot) are those of the package, so the result is
    bitwise equal; on the 21-node window this avoids most of the per-operation overhead of the slot loop.
    """
    norm = spectrum.abs().flatten(1).norm(dim=1).clamp_min(1e-12)
    y = spectrum / norm[:, None, None]
    y = y * phase.conj()[:, None]
    eta = torch.full((2, len(spectrum), 1), ETA, device=spectrum.device)
    for step in range(2):
        a = attention[step].repeat_interleave(2, dim=-1) * mask[..., None]
        terms = a[:, :, None] * (y[neighbour] - y[:, None])
        update = torch.zeros_like(y)
        for slot in range(SLOTS):
            if slot != WINDOW:
                update = update + terms[:, slot]
        y = y + eta[step, :, :, None] * update
    return y * phase[:, None]


def returned_place(fixed: tuple, bank: tuple, weights: np.ndarray) -> int:
    """``nsd.matching.returned_place`` with the 16 range bins summed before the inverse FFT.

    ``fixed`` holds the key and aux cosines of the candidates; ``bank`` is ``matching.alignment_bank`` of the
    candidates followed by the query (last row). The package takes the inverse FFT of every range bin and sums the
    16 correlation profiles; since the inverse FFT is linear, summing the cross-spectra first gives the same
    profile with one inverse FFT per candidate instead of 16. The cues differ from the package in float32
    rounding only (order 1e-7); the returned places are compared with the package after the timing.
    """
    spectrum, norm, (columns, mask) = bank
    k = len(norm) - 1
    shifts = matching.N_SHIFTS
    spectral = (np.fft.irfft((spectrum[:k] * np.conj(spectrum[k])).sum(1), n=shifts, axis=-1).max(-1)
                / (norm[:k] * norm[k]))
    profile = np.fft.irfft((columns[:k] * np.conj(columns[k])).sum(1), n=shifts, axis=-1)
    count = np.fft.irfft(mask[:k] * np.conj(mask[k]), n=shifts, axis=-1)
    column = np.where(count > 0.5, profile / np.maximum(count, 1.0), -1.0).max(-1)
    fused = (weights[0] * matching.minmax(fixed[0]) + weights[1] * matching.minmax(fixed[1])).astype(np.float32)
    fused = fused + weights[2] * matching.minmax(spectral) + weights[3] * matching.minmax(column)
    return int(np.argmax(fused))


def acceptance_score(z: torch.Tensor, query: torch.Tensor, candidate: torch.Tensor, weight: torch.Tensor,
                     whitening: torch.Tensor, mix: float, calibration: dict) -> torch.Tensor:
    """``nsd.matching.acceptance_banks`` followed by ``temporal_score`` for the frames of one query.

    ``z`` holds the refined spectra of the frames the plan touches; ``query`` and ``candidate`` index into it.
    The package runs every operation once per bank (P(z') and P(W z')); here the two banks are stacked and each
    operation runs once on both. The operations are those of the package, so the score is bitwise equal.
    """
    shifts = matching.N_SHIFTS
    white = torch.einsum("kij,njk->nik", whitening, z)
    banks = torch.stack((matching.compress(z), matching.compress(white)))              # [2, M, 16, 8]
    # matching._peak_bank
    spectrum = F.pad(banks, (1, shifts // 2 + 1 - matching.N_FREQ - 1))
    norm = banks.abs().flatten(2).norm(dim=2).clamp_min(1e-8)
    grid = torch.fft.irfft(spectrum, n=shifts, dim=-1)
    column_norm = grid.norm(dim=2, keepdim=True)
    mask = (column_norm[:, :, 0] > 1e-8).to(grid.dtype)
    columns = torch.fft.rfft(grid / column_norm.clamp_min(1e-8), dim=-1)
    mask = torch.fft.rfft(mask, dim=-1)
    # matching._pair_peaks
    q, j = query.flatten(), candidate.flatten()
    profile = torch.fft.irfft(spectrum[:, j] * spectrum[:, q].conj(), n=shifts, dim=-1).sum(-2)
    spec = profile.max(-1).values / (norm[:, j] * norm[:, q])
    profile = torch.fft.irfft(columns[:, j] * columns[:, q].conj(), n=shifts).sum(-2)
    count = torch.fft.irfft(mask[:, j] * mask[:, q].conj(), n=shifts)
    col = torch.where(count > 0.5, profile / count.clamp_min(1.0), -1.0).max(-1).values
    # matching.temporal_score
    peaks = torch.stack((spec, col), -1).reshape(2, *query.shape, 2)
    peaks = (peaks * weight[..., None]).sum(-2).max(-2).values                          # per bank [B, 2]
    mixed = (1 - mix) * peaks[0] + mix * peaks[1]
    value = (30.0 * mixed[:, 0] + mixed[:, 1]) / 2.0
    return (value - calibration["mean"]) * calibration["slope"] / calibration["scale"] + calibration["intercept"]


class OnlineNSD:
    """State of the online method. All arrays are preallocated; rows [0, n) are in use."""

    def __init__(self, net, assets: dict, capacity: int, device: str) -> None:
        self.net, self.assets, self.device = net, assets, device
        self.r = np.zeros((capacity, 288), np.float32)
        self.m = np.zeros((capacity, 128), np.float32)
        self.z_refined = np.zeros((capacity, 16, 8), np.complex64)
        self.poses = np.zeros((capacity, 4, 4), np.float64)
        self.travel = np.zeros(capacity, np.float64)
        self.d_raw = np.zeros((RAW, 288), np.float32)
        self.z_raw = np.zeros((RAW, 16, 8), np.complex64)
        self.n = 0
        # Edges and slot tables of the window graph are the same at every arrival; node inputs, edge geometry and
        # transport phases are set in update.
        self.edges = temporal_edges(RAW)
        graph = build_graph(self.d_raw, self.z_raw, np.broadcast_to(np.eye(4), (RAW, 4, 4)), device=device)
        self.graph = {k: graph[k] for k in ("edges", "neighbour", "mask")}

    def fill(self, d: np.ndarray, z: np.ndarray, poses: np.ndarray) -> None:
        """Initial map from already encoded keyframes."""
        n = len(d)
        self.r[:n] = index.remetric_key(d, self.assets["remetric_weights"])
        self.m[:n] = index.aux_magnitude_index(z)
        self.z_refined[:n] = refine(self.net, d, z, poses)
        self.poses[:n], self.travel[:n] = poses, matching.arc_length(poses)
        self.d_raw[:], self.z_raw[:] = d[-RAW:], z[-RAW:]
        self.n = n

    def update(self, d: np.ndarray, z: np.ndarray, pose: np.ndarray) -> None:
        t = self.n
        self.r[t] = index.remetric_key(d[None], self.assets["remetric_weights"])[0]
        self.m[t] = index.aux_magnitude_index(z[None])[0]
        self.poses[t] = pose
        self.travel[t] = self.travel[t - 1] + np.linalg.norm(pose[:3, 3] - self.poses[t - 1, :3, 3])
        self.d_raw[:-1], self.z_raw[:-1] = self.d_raw[1:], self.z_raw[1:]
        self.d_raw[-1], self.z_raw[-1] = d, z
        self.n = t + 1
        window = self.poses[self.n - RAW:self.n]
        # The transport yaw is measured from the first keyframe of the sequence, not of the window.
        yaw = odometry_yaw(np.concatenate((self.poses[:1], window)))[1:]
        phase = np.exp(1j * yaw[:, None] * np.arange(1, 9)[None]).astype(np.complex64)
        graph = dict(self.graph, d=torch.from_numpy(self.d_raw).to(self.device),
                     spectrum=torch.from_numpy(self.z_raw).to(self.device),
                     geometry=torch.from_numpy(edge_geometry(window, self.edges)).to(self.device),
                     phase=torch.from_numpy(phase).to(self.device))
        with torch.no_grad():
            refined = refine_spectra(graph["spectrum"], graph["neighbour"], graph["mask"], graph["phase"],
                                     self.net.attention(graph))
        self.z_refined[self.n - PROVISIONAL:self.n] = refined[-PROVISIONAL:].cpu().numpy()

    def query(self, q: int) -> tuple:
        """Returned place and acceptance score of query q; reads rows up to q + 5 only."""
        ids = index.candidates(self.r, self.m, q)
        bank = matching.alignment_bank(self.z_refined[np.r_[ids, q]])
        fixed = (self.r[ids] @ self.r[q], self.m[ids] @ self.m[q])
        j = int(ids[returned_place(fixed, bank, self.assets["fusion_weights"])])
        plan = matching.temporal_plan(self.travel[:self.n], [q], [j])
        frames, local = np.unique(np.r_[plan["query"].ravel(), plan["candidate"].ravel()], return_inverse=True)
        local = torch.from_numpy(local.reshape(2, *plan["query"].shape)).to(self.device)
        whitening = torch.as_tensor(self.assets["whitening_matrix"]).to(self.device, torch.complex64)
        with torch.no_grad():
            score = acceptance_score(torch.from_numpy(self.z_refined[frames]).to(self.device), local[0], local[1],
                                     torch.from_numpy(plan["weight"]).to(self.device), whitening,
                                     self.assets["whitened_mix"], self.assets["score_calibration"])
        return j, float(score[0])

    def resident_bytes(self) -> dict:
        per_place = sum(a[:self.n].nbytes for a in (self.r, self.m, self.z_refined, self.poses, self.travel))
        constants = sum(p.numel() * p.element_size() for p in self.net.state_dict().values())
        constants += self.assets["remetric_weights"].nbytes + self.assets["whitening_matrix"].nbytes
        return {"places": per_place, "raw_window": self.d_raw.nbytes + self.z_raw.nbytes, "constants": constants}


def clock(device: str) -> float:
    if device != "cpu":
        torch.cuda.synchronize()
    return time.perf_counter()


def summary(ms: list) -> dict:
    return {"mean": round(float(np.mean(ms)), 4), "p95": round(float(np.percentile(ms, 95)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--root", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--encoded", required=True, type=Path, help="file written by tools/encode_sequence.py")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--assets", default=str(PKG / "assets"))
    ap.add_argument("--n-map", type=int, default=10000, help="eligible candidates of the first timed query")
    ap.add_argument("--device", default="cpu", help="device of the network and of the acceptance score")
    ap.add_argument("--out", type=Path, help="write the result as JSON")
    args = ap.parse_args()

    deterministic()
    assets = matching.load_assets(args.assets)
    net = load_refiner(args.weights, args.device)
    with np.load(args.encoded) as s:
        scan_ids, d, z, poses = s["scan_ids"], s["d"], s["z"], s["poses"]
    first = args.n_map + index.MIN_GAP - 1 + matching.LAG     # arrival whose query sees n_map candidates
    last = first + ARRIVALS
    assert last <= len(d), f"the sequence has {len(d)} keyframes, {last} are needed"
    seq = open_sequence(args.dataset, args.root, args.sequence)
    scans = {t: seq.points(int(scan_ids[t])) for t in range(first - WARMUP, last)}

    state = OnlineNSD(net, assets, last, args.device)
    state.fill(d[:first - WARMUP], z[:first - WARMUP], poses[:first - WARMUP])

    def arrive(t: int) -> tuple:
        t0 = clock(args.device)
        dt, zt = encode(scans[t], args.dataset)
        t1 = clock(args.device)
        state.update(dt, zt, poses[t])
        t2 = clock(args.device)
        returned, score = state.query(t - matching.LAG)
        t3 = clock(args.device)
        assert np.array_equal(dt, d[t]) and np.array_equal(zt, z[t]), "encoding differs from the encoded file"
        return returned, score, 1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2)

    for t in range(first - WARMUP, first):
        arrive(t)
    start = (state.d_raw.copy(), state.z_raw.copy())
    ledgers, times = [], {"encode": [], "state_update": [], "query": []}
    for _ in range(REPETITIONS):
        state.n, state.d_raw[:], state.z_raw[:] = first, start[0], start[1]
        rows = [arrive(t) for t in range(first, last)]
        ledgers.append([(row[0], row[1]) for row in rows])
        for key, column in zip(times, (2, 3, 4)):
            times[key] += [row[column] for row in rows]
    times["update"] = [a + b for a, b in zip(times["encode"], times["state_update"])]
    times["processing"] = [a + b for a, b in zip(times["update"], times["query"])]

    # Full-sequence reference on the same keyframes.
    z_full = refine(net, d[:last], z[:last], poses[:last])
    reference = matching.match_sequence(d[:last], z[:last], z_full, poses[:last], assets, args.device)
    queries = np.arange(first, last) - matching.LAG
    ref_returned = reference["returned"][queries - index.MIN_GAP]
    ref_score = reference["score"][queries - index.MIN_GAP]
    final = last - 2 * WINDOW                                 # refined spectra [0, final) no longer change
    check = {"refined_max_abs_diff": float(np.abs(state.z_refined[:final] - z_full[:final]).max()),
             "returned_equal": all([j for j, _ in run] == ref_returned.tolist() for run in ledgers),
             "score_max_abs_diff": max(float(np.abs(np.array([s for _, s in run]) - ref_score).max())
                                       for run in ledgers)}

    resident = state.resident_bytes()
    total = sum(resident.values())
    result = {"sequence": f"{args.dataset}/{args.sequence}", "n_map": args.n_map, "device": args.device,
              "arrivals": ARRIVALS, "repetitions": REPETITIONS, "places_at_end": state.n,
              "ms": {k: summary(v) for k, v in times.items()},
              "resident_bytes": dict(resident, total=total),
              "resident_mb": round(total / 1e6, 3),
              "resident_kb_per_place": round(total / 1e3 / (queries[-1] - index.MIN_GAP + 1), 3),
              "check": check}
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
