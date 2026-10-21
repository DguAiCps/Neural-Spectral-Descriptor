#!/usr/bin/env python3
"""DeepRING (Lu, Xu, Tang, Xiong, Wang, ICRA 2023, arXiv 2210.11029) --
PAPER-BASED RE-IMPLEMENTATION, model and episode pieces (2026-10-11).

THERE IS NO PUBLIC OFFICIAL DeepRING CODE (checked 2026-10-11): no repository
on the authors' GitHub accounts (lus6-Jenny, MaverickPeter), none in GitHub
code search, no link in the paper or on the first author's publication page
(which links code for RING), nothing in the full histories of
lus6-Jenny/RING and lus6-Jenny/RINGSharp, and the request for the code in
lus6-Jenny/RING issue #1 (2023-12-23) is open and unanswered.  Everything
below is therefore a re-implementation from the paper text; where the same
authors' released code contains the building block, that code is used
unchanged:

  sinogram    RING repo utils.core.generate_RING (radon ParallelBeam(120,
              linspace(0, 2 pi, 120))) on the RING occupancy BEV -- called by
              the driver, not here.  Paper Sec. V-B: "the same data
              preprocessing ... the same resolution of 120 x 120 ... for Scan
              Context, RING and our method".
  UNet        RINGSharp glnet/models/backbones_2d/unet.py (upstream cd55367,
              byte-identical in our clone, sha256 checked on import), class
              UNet(in_channels=1, bn=False, is_circular=True).  Paper
              Sec. III-B: "UNet equipped with circular convolutions"; this file
              has the is_circular option and the commented-out global
              max/avg pooling of DeepRING's aggregation ablation (Tab. III).
  row DFT     |FFT_tau| per theta row, norm='ortho', +1e-15 inside the sqrt
              = RINGSharp ring_sharp_l.forward_row_fft (the RING repo
              utils.core.forward_row_fft is the same without the epsilon).
Written from the paper (no released counterpart):
  eq. 8       C(M1, M2) = max_a sum_theta sum_omega M1(theta + a, omega) M2(theta, omega)
              -- un-normalized, as printed (corr_max; FFT over theta).
  eq. 9       C~ = Softmax(W C + b), W, b in R  -- nn.Linear(1, 1).
  loss        cross entropy over the support classes (Sec. III-C).
  episodes    24 classes x (1 shot + 6 queries); same class = within 10 m,
              different classes = beyond 20 m (Sec. III-D) -> EpisodeSampler.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
UNET_FILE = REPO / "_external/RINGSharp/glnet/models/backbones_2d/unet.py"
UNET_UPSTREAM = "lus6-Jenny/RINGSharp @ cd55367 glnet/models/backbones_2d/unet.py"
UNET_SHA256 = "3d1fefd05bf351692814ad89e2fcc361c03cbe24d089924d8e8759b90131f002"
N_THETA = N_TAU = 120          # paper: 120 x 120 (RING cfg.num_ring / num_sector)

# Paper Sec. III-D (implementation details) -- the training recipe.
RECIPE = {"epochs": 20, "episodes_per_epoch": 60, "optimizer": "Adam",
          "lr": 1e-3, "weight_decay": 1e-4, "milestones": [5, 12], "gamma": 0.1,
          "ways": 24, "shots": 1, "queries_per_class": 6,
          "same_class_m": 10.0, "diff_class_m": 20.0}


def import_unet():
    """RINGSharp UNet class from its file, after checking the bytes equal the
    upstream blob (loaded by path: no 'glnet' package import)."""
    data = UNET_FILE.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    assert sha == UNET_SHA256, f"{UNET_FILE} differs from {UNET_UPSTREAM}: {sha}"
    spec = importlib.util.spec_from_file_location("ringsharp_unet_cd55367", UNET_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UNet


def row_fft_mag(x: torch.Tensor) -> torch.Tensor:
    """RINGSharp ring_sharp_l.forward_row_fft (first output): DFT magnitude of
    every theta row along tau (eq. 7)."""
    median_output = torch.fft.fft2(x, dim=-1, norm='ortho')
    return torch.sqrt(median_output.real ** 2 + median_output.imag ** 2 + 1e-15)


class DeepRING(nn.Module):
    """sinogram (B, 1, 120, 120) -> M_f (B, 1, 120, 120); eq. 9 head."""

    def __init__(self):
        super().__init__()
        UNet = import_unet()
        self.unet = UNet(1, bn=False, is_circular=True)
        self.fc = nn.Linear(1, 1)          # W C + b, W, b in R (eq. 9)

    def embed(self, sino: torch.Tensor) -> torch.Tensor:
        return row_fft_mag(self.unet(sino))

    def logits(self, C: torch.Tensor) -> torch.Tensor:
        return self.fc(C.unsqueeze(-1)).squeeze(-1)

    def wb(self):
        return float(self.fc.weight.detach().reshape(())), float(self.fc.bias.detach().reshape(()))


def corr_max(Mq: torch.Tensor, Ms: torch.Tensor) -> torch.Tensor:
    """eq. 8 for all pairs: C[q, s] = max_a sum_{c, theta, omega}
    Mq[q, c, theta + a, omega] * Ms[s, c, theta, omega]  (indices mod 120).
    Cross-correlation theorem along theta (rfft; exact for real inputs)."""
    A = torch.fft.rfft(Mq, dim=-2)                     # (Q, C, 61, W)
    B = torch.fft.rfft(Ms, dim=-2)                     # (S, C, 61, W)
    G = torch.einsum('qckw,sckw->qsk', A, B.conj())    # (Q, S, 61)
    corr = torch.fft.irfft(G, n=Mq.shape[-2], dim=-1)  # (Q, S, 120): shift a
    return corr.max(dim=-1).values


def corr_max_direct(Mq: torch.Tensor, Ms: torch.Tensor) -> torch.Tensor:
    """Reference for checks: eq. 8 written out with torch.roll."""
    H = Mq.shape[-2]
    Sf = Ms.reshape(len(Ms), -1)
    cols = [torch.roll(Mq, shifts=-a, dims=-2).reshape(len(Mq), -1) @ Sf.T
            for a in range(H)]
    return torch.stack(cols, dim=-1).max(dim=-1).values


class SpecBank:
    """Retrieval form of eq. 8 for one sequence: rfft_theta of every M_f kept
    as a real (61, n, 240) bank [Re | Im]; a block of queries is scored
    against rows [0, ncols) with one batched matmul over the 61 theta
    frequencies, then irfft over the shift and max (same value as corr_max).
    Only rows < ncols are read."""

    def __init__(self, n: int, device: str):
        self.n, self.device = n, device
        self.K = N_THETA // 2 + 1
        self.bank = torch.empty((self.K, n, 2 * N_TAU), dtype=torch.float32, device=device)

    def put(self, i0: int, M: torch.Tensor):
        A = torch.fft.rfft(M.to(self.device, torch.float32), dim=-2)[:, 0]   # (m, 61, 120)
        i1 = i0 + len(A)
        self.bank[:, i0:i1, :N_TAU] = A.real.permute(1, 0, 2)
        self.bank[:, i0:i1, N_TAU:] = A.imag.permute(1, 0, 2)

    def rows(self, idx) -> torch.Tensor:
        """M_f of bank rows idx, (m, 1, 120, 120), by the inverse rfft (checks)."""
        idx = torch.as_tensor(np.asarray(idx), device=self.device)
        A = torch.complex(self.bank[:, idx, :N_TAU], self.bank[:, idx, N_TAU:])   # (61, m, 120)
        return torch.fft.irfft(A.permute(1, 0, 2), n=N_THETA, dim=-2).unsqueeze(1)

    def corr(self, qidx, ncols: int) -> torch.Tensor:
        """C (len(qidx), ncols) for queries qidx (rows of this bank)."""
        qidx = torch.as_tensor(np.asarray(qidx), device=self.device)
        A = self.bank[:, qidx, :]                                   # (61, B, 240) [Ar|Ai]
        Ar, Ai = A[..., :N_TAU], A[..., N_TAU:]
        q_re = A.transpose(1, 2)                                    # [Ar; Ai]
        q_im = torch.cat([Ai, -Ar], dim=-1).transpose(1, 2)         # [Ai; -Ar]
        nb = len(qidx)
        Gk = torch.bmm(self.bank[:, :ncols, :], torch.cat([q_re, q_im], dim=2))
        G = torch.complex(Gk[..., :nb], Gk[..., nb:])               # (61, ncols, B)
        corr = torch.fft.irfft(G.permute(2, 1, 0), n=N_THETA, dim=-1)   # (B, ncols, 120)
        return corr.max(dim=-1).values


class EpisodeSampler:
    """Episodes of paper Sec. III-C/D inside one training sequence.

    Class k = its shot s_k; members(k) = keyframes within same_m (10 m) of
    s_k.  Paper: "scans whose positions are within 10 m from each other belong
    to the same class and scans whose positions are beyond 20 m from each
    other belong to different classes".  Shots are drawn at random with
    pairwise distance > same_m + diff_m (30 m), so that every query is
    > 20 m from every other class's shot (triangle inequality) and every
    (query, shot) pair the loss sees is labelled by the paper's rule.
    One sequence per episode (our sequences do not share a pose frame),
    chosen with probability proportional to its keyframe count."""

    def __init__(self, seq_positions, rng: np.random.Generator, ways=24, queries=6,
                 same_m=10.0, diff_m=20.0, max_tries=50):
        from scipy.spatial import cKDTree
        self.pos = [np.asarray(p, dtype=np.float64) for p in seq_positions]
        self.rng, self.ways, self.queries = rng, ways, queries
        self.sep = same_m + diff_m
        self.max_tries = max_tries
        self.members = [[np.asarray(sorted(set(m) - {i}), dtype=np.int64)
                         for i, m in enumerate(cKDTree(p).query_ball_point(p, same_m))]
                        for p in self.pos]
        n = np.array([len(p) for p in self.pos], dtype=np.float64)
        self.p_seq = n / n.sum()

    def sample(self):
        """-> (seq index, shots (ways,), queries (ways, queries)) frame indices."""
        for _ in range(self.max_tries):
            s = int(self.rng.choice(len(self.pos), p=self.p_seq))
            pos, mem = self.pos[s], self.members[s]
            shots = []
            for i in self.rng.permutation(len(pos)):
                if len(mem[i]) < self.queries:
                    continue
                if shots and np.min(np.linalg.norm(pos[shots] - pos[i], axis=1)) <= self.sep:
                    continue
                shots.append(int(i))
                if len(shots) == self.ways:
                    break
            if len(shots) < self.ways:
                continue
            qs = np.stack([self.rng.choice(mem[i], self.queries, replace=False) for i in shots])
            return s, np.asarray(shots, dtype=np.int64), qs.astype(np.int64)
        raise RuntimeError("could not draw an episode")


def episode_loss(model: DeepRING, M: torch.Tensor, ways: int, queries: int):
    """M = embeddings of [shots (ways); queries (ways*queries, class-major)]."""
    C = corr_max(M[ways:], M[:ways])                       # (ways*queries, ways)
    logits = model.logits(C)
    labels = torch.arange(ways, device=M.device).repeat_interleave(queries)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc
