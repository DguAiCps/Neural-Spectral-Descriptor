"""Training of DiffAttnConv (Sec. 3.4, "Training objective"; Sec. 4.1, "Training and model selection").

Reported configuration: random initialisation (seed s sets the initialisation and the sample order), AdamW
(learning rate 1e-3, weight decay 1e-5), gradient-norm clipping at 1, 64 pairs per update, 8 updates per training
sequence in turn, 512 updates; the final weights are used.

Training pairs. For every query q of a training sequence (30 <= q < N-15) one candidate j is fixed before training: the
place-selection winner over the union of the two index lists, computed on the *unrefined* spectra (see
:func:`nsd.train.build_pool`). Label y = 1 if the nearest eligible past keyframe of q lies within 5 m and j lies within
5 m of q; pairs whose nearest eligible past keyframe is in (5 m, 25 m] are excluded. Each update draws 32 positives
and 32 negatives (with replacement only if a class has fewer than 32 pairs).

Loss. Binary cross-entropy on a * S(q, j) + b, where S is the acceptance score of the refined spectra (Eq. (6)) and the
positive affine map (a, b) was fitted once on the training pairs with the untrained model (= uniform subtraction) and
is then frozen (``assets/loss_calibration.json``).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import index, matching, metrics
from .model import NSDRefiner

UPDATES = 512
BATCH = 64
UPDATES_PER_SEQUENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
CLIP = 1.0


# --------------------------------------------------------------------------------------------- initialisation
def initial_model(seed: int) -> NSDRefiner:
    """Random initialisation of the reported model for ``seed``.

    Draw order on the CPU generator: input projection, edge-type embedding, edge MLP, then per layer the Q, K, V
    projections (PyTorch default init, then Xavier-uniform) and the geometry MLP; this order defines the initial weights
    of each seed. Then the Q projections and the last geometry layers are set
    to zero: all attention weights are uniform and the untrained model computes uniform subtraction exactly.
    """
    net = NSDRefiner()
    torch.manual_seed(seed)
    net.input_proj.reset_parameters()
    net.edge_encoder.type_embedding.reset_parameters()
    net.edge_encoder.mlp[0].reset_parameters()
    net.edge_encoder.mlp[2].reset_parameters()
    for conv in net.convs:
        for layer in (conv.query, conv.key, conv.value, conv.geometry[0], conv.geometry[2]):
            layer.reset_parameters()
        for layer in (conv.query, conv.key, conv.value):
            nn.init.xavier_uniform_(layer.weight)
    with torch.no_grad():
        for conv in net.convs:
            conv.query.weight.zero_()
            conv.geometry[2].weight.zero_()
            conv.geometry[2].bias.zero_()
    return net


# --------------------------------------------------------------------------------------------- acceptance score
def acceptance_score(z: torch.Tensor, plan: Dict[str, torch.Tensor], whitening, mix: float,
                     calibration: Dict[str, float]) -> torch.Tensor:
    """Differentiable acceptance score S(q, j) of Eqs. (5)-(6) for a batch of pairs (calibrated scale).

    The same computation as at test time (:func:`nsd.matching.acceptance_banks`, :func:`nsd.matching.temporal_score`):

    z            [N, 16, 8] complex64  refined spectra of the sequence
    plan         query / candidate [B, 2, 11] int64 and weight [B, 2, 11] float32: for each direction rho (axis 1)
                 the centre pair (slot 0) and the ten distance-matched pairs; weights already hold the temporal-evidence
                 weighting (1-beta) on the centre plus beta times the mean over matched pairs, beta = 11/15
    whitening    [8, 16, 16] decorrelating matrices Gamma_k (one per frequency)
    mix          lambda, weight of the decorrelated bank (0.25)
    calibration  fixed positive affine map of the raw score: (S - mean) * slope / scale + intercept

        T_C   = max_rho sum_u w_u C(q+u, j_u)          for C in {C_spec, C_col} and each bank
        U_C   = (1 - lambda) T_C[P(z)] + lambda T_C[P(Gamma z)]
        S_raw = (30 U_spec + U_col) / 2
    """
    return matching.temporal_score(matching.acceptance_banks(z, whitening), plan, mix, calibration)


# --------------------------------------------------------------------------------------------- training pairs
def build_pool(d: np.ndarray, z: np.ndarray, poses: np.ndarray, positions: np.ndarray, omega: np.ndarray,
               fusion_weights: np.ndarray) -> Dict[str, np.ndarray]:
    """Fixed training pairs of one training sequence (computed once, before training).

    d, z, poses   shared-interface fields of the sequence (``poses`` gives the arc length of the temporal evidence)
    positions     [N, 3] reference positions for the labels
    omega         remetric weights of the spectral key index; fusion_weights (w_r, w_m, w_s, w_c)

    For each query 30 <= q < N-15 the candidate j is the place-selection winner (Eq. (4)) over the union of the two
    index lists, scored with the *unrefined* spectra z. Returns the per-pair arrays
    query_idx, j_star, min_past_gt_dist, top1_geo_dist, eligible, correct and the temporal-evidence plan
    plan_query / plan_candidate / plan_weight ([B, 2, 11]) used by :func:`acceptance_score`.
    """
    r = index.remetric_key(d, omega)
    m = index.aux_magnitude_index(z)
    bank = matching.alignment_bank(z)
    q = np.arange(30, len(d) - matching.LAG, dtype=np.int64)
    j = np.array([matching.returned_place(r, m, bank, int(t), index.candidates(r, m, int(t)), fusion_weights)
                  for t in q], np.int64)
    nearest = metrics.min_past_distance(positions, q)
    returned = metrics.returned_distance(positions, q, j)
    plan = matching.temporal_plan(matching.arc_length(poses), q, j)
    return dict(query_idx=q, j_star=j, min_past_gt_dist=nearest, top1_geo_dist=returned,
                eligible=(nearest <= metrics.POSITIVE_M) | (nearest > metrics.NEGATIVE_M),
                correct=(nearest <= metrics.POSITIVE_M) & (returned <= metrics.CORRECT_M),
                **{"plan_" + k: v for k, v in plan.items()})


# --------------------------------------------------------------------------------------------- training loop
def sample_batch(eligible: np.ndarray, correct: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Class-balanced batch of pair indices (with replacement only for a class smaller than its share)."""
    classes = [np.flatnonzero(eligible & (correct == c)) for c in (0, 1)]
    classes = [v for v in classes if len(v)]
    n = size // len(classes)
    return rng.permutation(np.concatenate([rng.choice(v, n, replace=len(v) < n) for v in classes]))


def train(graphs: List[Dict[str, torch.Tensor]], pools: List[Dict[str, np.ndarray]], seed: int,
          whitening: torch.Tensor, mix: float, score_calibration: Dict[str, float],
          loss_calibration: Dict[str, float], updates: int = UPDATES,
          log: Optional[Callable[[int, float, float], None]] = None) -> NSDRefiner:
    """Train from scratch; returns the model after ``updates`` updates.

    graphs / pools are the training sequences in a fixed order (``nsd.model.build_graph`` dicts and
    :func:`build_pool` dicts on the same device). The batch of sequence number ``o`` in epoch ``e`` (1-based) is drawn
    from ``numpy.random.default_rng((seed, e, o))``. Each update runs the model on the whole sequence graph.
    ``log(update, loss, grad_norm)`` is called after every update.
    """
    device = graphs[0]["d"].device
    per_epoch = UPDATES_PER_SEQUENCE * len(graphs)
    assert updates % per_epoch == 0
    net = initial_model(seed).to(device)
    params = list(net.parameters())
    opt = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    plans = [{k: torch.from_numpy(p["plan_" + k]).to(device) for k in ("query", "candidate", "weight")} for p in pools]
    lc = loss_calibration
    step = 0
    for epoch in range(1, updates // per_epoch + 1):
        for ordinal, (graph, pool, plan) in enumerate(zip(graphs, pools, plans)):
            rng = np.random.default_rng((seed, epoch, ordinal))
            for _ in range(UPDATES_PER_SEQUENCE):
                chosen = sample_batch(pool["eligible"], pool["correct"], rng, BATCH)
                ix = torch.from_numpy(chosen).to(device)
                opt.zero_grad(set_to_none=True)
                z = net(graph)
                score = acceptance_score(z, {k: v[ix] for k, v in plan.items()}, whitening, mix, score_calibration)
                logit = (score - lc["mean"]) * lc["slope"] / lc["scale"] + lc["intercept"]
                label = torch.from_numpy(pool["correct"][chosen]).to(device).float()
                loss = F.binary_cross_entropy_with_logits(logit, label)
                loss.backward()
                for p in params:  # every parameter takes part in clipping and the optimiser step
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                grad = torch.nn.utils.clip_grad_norm_(params, CLIP)
                assert math.isfinite(float(loss.detach())) and math.isfinite(float(grad))
                opt.step()
                step += 1
                if log is not None:
                    log(step, float(loss.detach()), float(grad))
    return net
