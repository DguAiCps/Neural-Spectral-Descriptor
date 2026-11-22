"""Candidate alignment and acceptance (Sec. 3.3).

Given the candidate union of nsd.index for a query q, this module

1. scores every candidate with four cues, each min-max normalised over the candidate set,
       key cosine        r_q . r_j                      (fixed, nsd.index)
       aux cosine        m_q . m_j                      (fixed, nsd.index)
       spectral corr.    max over 60 circular shifts of the cross-spectrum of the refined z'
       column alignment  best mean column cosine between the 16x60 grids reconstructed from z'
   and returns the candidate with the largest weighted sum (weights 2, 1, 2, 2; ties -> smallest id);
2. computes the acceptance score of the returned pair (q, j) from temporal evidence: the neighbours
   q-5..q+5 are matched to frames within j +- 10 by odometry arc length, in the same and in the reverse
   driving direction. Each pair contributes the 60-shift peaks of two banks, P(z') and P(W z'), with
   P(z) = z / sqrt(|z|) (phase kept, magnitude compressed) and W the fixed per-frequency whitening matrix.
   The weighted neighbour sum is maximised over the two directions, the banks are mixed 75:25
   (lambda = 0.25), the two channels averaged as (30 * spectral + column) / 2, and a fixed positive affine
   calibration is applied. Thresholding this score gives the accept decision.

Query schedule (online, fixed lag): at keyframe t the system decides q = t - 15, so queries are
30 <= q < N - 15 and candidates j <= q - 30. The acceptance score of q reads z' only up to q + 5, and
z'_i itself depends on keyframes up to i + 10 (nsd.model), so nothing after t is used provided z' was
produced with the online semantics of nsd.model.

Step 1 and the temporal plan are float32/float64 numpy; the acceptance score is torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import index

LAG = 15              # q = t - LAG is decided at keyframe t
N_SHIFTS = 60         # circular yaw shifts of the 16 x 60 reconstructed grid
N_FREQ = 8            # stored positive DFT frequencies 1..8
NEIGHBOURS = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
CANDIDATE_RADIUS = 10  # candidate neighbours are searched in j +- 10
SCORE_CHUNK = 128      # queries scored together (shared bank of the frames they touch)


# ------------------------------------------------------------------ 1. alignment scores and winner
def alignment_bank(z: np.ndarray) -> tuple:
    """Per-frame inputs of the two alignment cues computed from the refined coefficients z' [N, 16, 8].

    Returns (spectrum [N,16,31] complex64 with frequencies 1..8 filled, magnitude norm [N],
    column bank (rfft of the unit-normalised columns of the 16x60 grid, rfft of the valid-column mask)).
    """
    z = z[:, :, :N_FREQ]
    spectrum = np.zeros((len(z), 16, N_SHIFTS // 2 + 1), np.complex64)
    spectrum[:, :, 1:N_FREQ + 1] = z
    norm = np.clip(np.linalg.norm(np.abs(z).reshape(len(z), -1), axis=1), 1e-8, None)
    grid = np.fft.irfft(spectrum, n=N_SHIFTS, axis=-1).astype(np.float32)
    column_norm = np.linalg.norm(grid, axis=1, keepdims=True)
    valid = (column_norm[:, 0] > 1e-8).astype(np.float32)
    columns = grid / np.clip(column_norm, 1e-8, None)
    return spectrum, norm, (np.fft.rfft(columns, axis=-1).astype(np.complex64),
                            np.fft.rfft(valid, axis=-1).astype(np.complex64))


def minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalisation over the candidate set."""
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / max(hi - lo, 1e-8)


def spectral_alignment(bank: tuple, q: int, ids: np.ndarray) -> np.ndarray:
    """Peak over 60 shifts of the summed cross-correlation, normalised by the two magnitude norms."""
    spectrum, norm, _ = bank
    corr = np.fft.irfft(spectrum[ids] * np.conj(spectrum[q]), n=N_SHIFTS, axis=-1)
    return corr.sum(axis=1).max(axis=-1) / (norm[ids] * norm[q])


def column_alignment(bank: tuple, q: int, ids: np.ndarray) -> np.ndarray:
    """Best (over 60 shifts) mean cosine of the columns that are valid in both grids (-1 if none)."""
    columns, mask = bank[2]
    corr = np.fft.irfft(columns[ids] * np.conj(columns[q]), n=N_SHIFTS, axis=-1).sum(axis=1)
    count = np.fft.irfft(mask[ids] * np.conj(mask[q]), n=N_SHIFTS, axis=-1)
    return np.where(count > 0.5, corr / np.maximum(count, 1.0), -1.0).max(axis=-1)


def returned_place(r: np.ndarray, m: np.ndarray, bank: tuple, q: int, ids: np.ndarray,
                   weights: np.ndarray) -> int:
    """Fuse the four min-max normalised cues with weights (key, aux, spectral, column); return the argmax id."""
    fixed = (weights[0] * minmax(r[ids] @ r[q]) + weights[1] * minmax(m[ids] @ m[q])).astype(np.float32)
    fused = (fixed + weights[2] * minmax(spectral_alignment(bank, q, ids))
             + weights[3] * minmax(column_alignment(bank, q, ids)))
    return int(ids[np.argmax(fused)])


# ------------------------------------------------------------------ 2. temporal evidence
def arc_length(poses: np.ndarray) -> np.ndarray:
    """Cumulative odometry travel distance along the keyframes (float64)."""
    xyz = poses[:, :3, 3]
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]


def _nearest_by_distance(travel: np.ndarray, centre: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Frame within centre +- 10 whose travel distance is closest to travel[centre] + offset."""
    target = travel[centre] + offset
    left = np.maximum(0, centre - CANDIDATE_RADIUS)
    right = np.minimum(len(travel) - 1, centre + CANDIDATE_RADIUS)
    lower = np.clip(np.searchsorted(travel, target, side="left") - 1, left, right)
    upper = np.clip(lower + 1, left, right)
    return np.where(np.abs(travel[lower] - target) <= np.abs(travel[upper] - target), lower, upper)


def temporal_plan(travel: np.ndarray, q: np.ndarray, j: np.ndarray) -> dict:
    """Pairs and weights of the temporal evidence for returned pairs (q, j).

    For both directions s in (+1, -1), neighbour q+o (o in +-1..5) is paired with the candidate-side frame
    at arc length travel[j] + s * (travel[q+o] - travel[q]). A neighbour is valid if both frames exist and
    the candidate-side frame differs from j. Several query neighbours mapped to the same candidate frame
    share one unit of weight. The centre pair gets weight 1 before normalisation; after normalisation
    everything is scaled by 11/15 and 4/15 is added to the centre.

    Returns arrays of shape [B, 2, 11] (centre first): query, candidate (clipped to valid range), weight.
    """
    q, j = np.asarray(q, np.int64), np.asarray(j, np.int64)
    n = len(travel)
    qs, js, valids = [], [], []
    for direction in (1, -1):
        qn = q[:, None] + NEIGHBOURS[None]
        jn = _nearest_by_distance(travel, j[:, None], direction * (travel[np.clip(qn, 0, n - 1)] - travel[q, None]))
        qs.append(qn)
        js.append(jn)
        valids.append((qn >= 0) & (qn < n) & (jn >= 0) & (jn < n) & (jn != j[:, None]))
    qn, jn, valid = np.stack(qs), np.stack(js), np.stack(valids)
    same = (jn[..., :, None] == jn[..., None, :]) & valid[..., None, :]
    w = valid / np.maximum(same.sum(-1), 1)
    w = np.concatenate((np.ones(w.shape[:2] + (1,)), w), -1)
    w /= w.sum(-1, keepdims=True)
    w *= 11 / 15
    w[..., 0] += 4 / 15
    shape = qn.shape[:2] + (1,)
    qi = np.concatenate((np.broadcast_to(q[None, :, None], shape), qn), -1)
    ji = np.concatenate((np.broadcast_to(j[None, :, None], shape), jn), -1)
    return {"query": np.clip(qi, 0, n - 1).transpose(1, 0, 2).copy(),
            "candidate": np.clip(ji, 0, n - 1).transpose(1, 0, 2).copy(),
            "weight": w.transpose(1, 0, 2).astype(np.float32)}


# ------------------------------------------------------------------ 3. acceptance score
def compress(x: torch.Tensor) -> torch.Tensor:
    """P(z) = z / sqrt(|z|), and 0 where z = 0."""
    amplitude = x.abs()
    return torch.where(amplitude > 0, x / amplitude.clamp_min(1e-12).sqrt(), 0)


def acceptance_banks(z: torch.Tensor, whitening: np.ndarray) -> torch.Tensor:
    """Stack P(z') and P(W z') -> [N, 2, 16, 8]. W is [8, 16, 16]: one 16x16 radial mix per frequency."""
    matrix = torch.as_tensor(whitening).to(z.device, z.dtype)
    white = torch.einsum("kij,njk->nik", matrix, z)
    return torch.stack((compress(z), compress(white)), dim=1)


def _peak_bank(coefficients: torch.Tensor) -> tuple:
    """Torch version of alignment_bank on [M, 16, 8] coefficients."""
    spectrum = F.pad(coefficients, (1, N_SHIFTS // 2 + 1 - N_FREQ - 1))
    norm = coefficients.abs().flatten(1).norm(dim=1).clamp_min(1e-8)
    grid = torch.fft.irfft(spectrum, n=N_SHIFTS, dim=-1)
    column_norm = grid.norm(dim=1, keepdim=True)
    mask = (column_norm[:, 0] > 1e-8).to(grid.dtype)
    columns = torch.fft.rfft(grid / column_norm.clamp_min(1e-8), dim=-1)
    return spectrum, norm, columns, torch.fft.rfft(mask, dim=-1)


def _pair_peaks(bank: tuple, query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """(spectral peak, column peak) of each (query, candidate) pair; same definitions as step 1."""
    spectrum, norm, columns, mask = bank
    profile = torch.fft.irfft(spectrum[candidate] * spectrum[query].conj(), n=N_SHIFTS, dim=-1).sum(-2)
    spec = profile.max(-1).values / (norm[candidate] * norm[query])
    profile = torch.fft.irfft(columns[candidate] * columns[query].conj(), n=N_SHIFTS).sum(-2)
    count = torch.fft.irfft(mask[candidate] * mask[query].conj(), n=N_SHIFTS)
    col = torch.where(count > 0.5, profile / count.clamp_min(1.0), -1.0).max(-1).values
    return torch.stack((spec, col), -1)


def temporal_score(banks: torch.Tensor, plan: dict, mix: float, calibration: dict) -> torch.Tensor:
    """Acceptance score of a chunk of pairs. plan tensors are [B, 2, 11]; banks from acceptance_banks.

    Differentiable with respect to banks (and hence z').
    """
    q, j, weight = (plan[k] for k in ("query", "candidate", "weight"))
    ids, inverse = torch.unique(torch.cat((q.flatten(), j.flatten())), return_inverse=True)
    iq, ij = inverse.chunk(2)
    peaks = []
    for b in range(2):
        p = _pair_peaks(_peak_bank(banks[ids, b]), iq, ij).reshape(*q.shape, 2)
        peaks.append((p * weight[..., None]).sum(-2).max(-2).values)   # neighbour sum, best direction
    mixed = (1 - mix) * peaks[0] + mix * peaks[1]
    value = (30.0 * mixed[:, 0] + mixed[:, 1]) / 2.0
    return (value - calibration["mean"]) * calibration["slope"] / calibration["scale"] + calibration["intercept"]


# ------------------------------------------------------------------ full sequence
def load_assets(folder) -> dict:
    """Fixed constants: remetric weights omega, whitening matrix W, fusion weights, mix lambda, calibration."""
    folder = Path(folder)
    spec = json.loads((folder / "matching.json").read_text())
    w = spec["fusion_weights"]
    return {"remetric_weights": np.load(folder / "remetric_weights.npy"),
            "whitening_matrix": np.load(folder / "whitening_matrix.npy"),
            "fusion_weights": np.asarray([w["key"], w["aux"], w["spectral_alignment"], w["column_alignment"]], np.float32),
            "whitened_mix": float(spec["whitened_mix"]),
            "score_calibration": spec["score_calibration"]}


def match_sequence(d: np.ndarray, z: np.ndarray, z_refined: np.ndarray, poses: np.ndarray, assets: dict,
                   device: str = "cuda") -> dict:
    """Run indexing, alignment and acceptance for every query of one sequence.

    Inputs follow the per-sequence interface (d, z, z_refined, odometry poses); assets is load_assets(<assets folder>).
    Returns the per-query ledger: query, returned (keyframe index of the returned place), score.
    """
    r = index.remetric_key(d, assets["remetric_weights"])
    m = index.aux_magnitude_index(z)
    bank = alignment_bank(z_refined)
    weights = assets["fusion_weights"]
    query = np.arange(index.MIN_GAP, len(d) - LAG, dtype=np.int64)
    returned = np.array([returned_place(r, m, bank, int(q), index.candidates(r, m, int(q)), weights)
                         for q in query], np.int64)
    plan = temporal_plan(arc_length(poses), query, returned)
    score = []
    with torch.no_grad():
        zt = torch.from_numpy(np.ascontiguousarray(z_refined)).to(device)
        banks = acceptance_banks(zt, assets["whitening_matrix"])
        for start in range(0, len(query), SCORE_CHUNK):
            chunk = {k: torch.from_numpy(v[start:start + SCORE_CHUNK]).to(device) for k, v in plan.items()}
            score.append(temporal_score(banks, chunk, assets["whitened_mix"], assets["score_calibration"]).cpu().numpy())
    score = np.concatenate(score).astype(np.float64) if score else np.zeros(0)
    return {"query": query, "returned": returned, "score": score}
