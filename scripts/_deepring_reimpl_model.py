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
  eq. 8       the max circular correlation over theta, computed as RING's own
              retrieval score (RING repo utils/circorr2.py circular_corr with
              zero_mean_normalize=True, i.e. 1 - utils.core.batch_circorr dist
              as in evaluation/evaluate.py): nn.InstanceNorm2d per descriptor,
              fft2(dim=-2) cross-correlation, |corr| summed over channels and
              tau, max over the shift, / (0.15 c h w) -> corr_max, SpecBank.
              RING-based on the user's suggestion (2026-10-21): eq. 8 as
              printed (un-normalized signed sum) gave first-step logits of
              10^2-10^3, an exploding loss, and descriptors collapsing to one
              map (loss ln 24: all 20 epochs of seed 3, 10 of seed 1).
Written from the paper (no released counterpart):
  eq. 9       C~ = Softmax(W C + b), W, b in R  -- nn.Linear(1, 1), started at
              W = 1, b = 0 (the paper gives no initial values; RING's score is
              higher for more similar scans, so W > 0 points the classifier the
              right way).  The default init draws W from U(-1, 1): with the
              RING score, seed 3 (W = -0.974) still collapsed in its first
              epoch while seeds 1 and 2 trained (2026-10-21, user decision).
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
          "same_class_m": 10.0, "diff_class_m": 20.0, "fc_init": {"W": 1.0, "b": 0.0}}


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
        with torch.no_grad():              # W = 1, b = 0; see the module docstring
            self.fc.weight.fill_(RECIPE["fc_init"]["W"])
            self.fc.bias.fill_(RECIPE["fc_init"]["b"])

    def embed(self, sino: torch.Tensor) -> torch.Tensor:
        return row_fft_mag(self.unet(sino))

    def logits(self, C: torch.Tensor) -> torch.Tensor:
        return self.fc(C.unsqueeze(-1)).squeeze(-1)

    def wb(self):
        return float(self.fc.weight.detach().reshape(())), float(self.fc.bias.detach().reshape(()))


def ring_fft(M: torch.Tensor) -> torch.Tensor:
    """circorr2.circular_corr per-input step: InstanceNorm2d (affine=False,
    eps 1e-5; F.instance_norm is what the module calls), fft2(dim=-2, ortho)."""
    return torch.fft.fft2(F.instance_norm(M), dim=-2, norm="ortho")


def ring_score(prod: torch.Tensor) -> torch.Tensor:
    """circorr2.circular_corr from the spectral product (..., c, h, w) of two
    ring_fft inputs: |ifft2(dim=-2)| (+1e-15) summed over channels and tau,
    max over the shift, / (0.15 c h w)."""
    corr = torch.fft.ifft2(prod, dim=-2, norm="ortho")
    mag = torch.sqrt(corr.real ** 2 + corr.imag ** 2 + 1e-15)
    c, h, w = prod.shape[-3:]
    return mag.sum(dim=-3).sum(dim=-1).max(dim=-1).values / (0.15 * c * h * w)


def corr_max(Mq: torch.Tensor, Ms: torch.Tensor) -> torch.Tensor:
    """eq. 8 as RING's circorr2 score for all pairs, C (Q, S); differentiable."""
    return ring_score(ring_fft(Mq).unsqueeze(1) * ring_fft(Ms).conj().unsqueeze(0))


def corr_max_direct(Mq: torch.Tensor, Ms: torch.Tensor) -> torch.Tensor:
    """Reference for checks: the RING repo's own utils.core.batch_circorr, one
    query at a time; returns 1 - dist (Q, S) float64."""
    import _ring_official_lib as rol
    core = rol.import_ring_core()
    rows = []
    for i in range(len(Mq)):
        dist, _ = core.batch_circorr(Mq[i:i + 1].repeat(len(Ms), 1, 1, 1), Ms)
        rows.append(1.0 - np.asarray(dist, dtype=np.float64).reshape(-1))
    return torch.from_numpy(np.stack(rows))


class SpecBank:
    """Retrieval form of corr_max for one sequence: the exact per-frame hoist
    of circorr2 used for the RING rows of Table 2 (_causal_ring_pr.
    precompute_fft_bank / correlate_prefix_fast): ring_fft once per frame into
    a complex64 (n, 1, 120, 120) bank; a block of queries is scored against
    rows [0, ncols) in chunks of at most `pairs` query-candidate pairs.  Only
    rows < ncols are read.  Rows in `keep` also retain their raw M_f (checks)."""

    def __init__(self, n: int, device: str, keep=(), pairs: int = 4096):
        self.n, self.device, self.pairs = n, device, pairs
        self.bank = torch.empty((n, 1, N_THETA, N_TAU), dtype=torch.complex64, device=device)
        self.keep, self.raw = {int(i) for i in keep}, {}

    def put(self, i0: int, M: torch.Tensor):
        M = M.to(self.device, torch.float32)
        self.bank[i0:i0 + len(M)] = ring_fft(M)
        for i in self.keep & set(range(i0, i0 + len(M))):
            self.raw[i] = M[i - i0].cpu()

    def rows(self, idx) -> torch.Tensor:
        """raw M_f of kept rows idx, (m, 1, 120, 120) (checks)."""
        return torch.stack([self.raw[int(i)] for i in idx])

    def corr(self, qidx, ncols: int) -> torch.Tensor:
        """C (len(qidx), ncols) for queries qidx (rows of this bank)."""
        qidx = torch.as_tensor(np.asarray(qidx), device=self.device)
        A = self.bank[qidx].unsqueeze(1)                            # (B, 1, c, h, w)
        k = max(1, self.pairs // len(qidx))
        out = torch.empty((len(qidx), ncols), dtype=torch.float32, device=self.device)
        for j0 in range(0, ncols, k):
            j1 = min(j0 + k, ncols)
            out[:, j0:j1] = ring_score(A * self.bank[j0:j1].conj().unsqueeze(0))
        return out


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
