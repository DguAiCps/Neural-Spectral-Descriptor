"""DiffAttnConv and temporal spectrum refining (Sec. 3.2).

Input per sequence (keyframes in acquisition order, see ``nsd/__init__.py``):

    d      float32   [N, 288]    invariant spectral key
    z      complex64 [N, 16, 8]  complex DFT features (range bins x frequencies 1..8)
    poses  float64   [N, 4, 4]   odometry poses; for validation/test these are the LiDAR odometry poses.
           For the training sequences the reported model used the dataset reference poses
           (KITTI: camera poses, see ``lidar_to_pose``).

Optional field:

    lidar_to_pose  float64 [4, 4]  extrinsic mapping the LiDAR frame into the frame the poses are expressed in.
                   Only used for the yaw of the transport (Sec. 3.2, "Yaw transport"); edge geometry uses the
                   poses as given. Identity (None) for KISS-ICP poses; the KITTI calibration ``Tr`` for the
                   KITTI training sequences, whose reference poses are camera poses.

Output: the refined spectra ``z_refined`` complex64 [N, 16, 8].

Online semantics. Each node is connected to its five preceding and five following keyframes. Two refining steps make
``z_refined[i]`` depend on keyframes i-10 .. i+10 only, and the temporal evidence of a query q reads refined spectra up to
q+5, so the decision for q made at step t = q+15 needs keyframes up to t. Running :func:`refine` on the full sequence
therefore gives the online result.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WINDOW = 5            # temporal neighbours on each side
HIDDEN = 128          # feature width
HEADS = 4             # attention heads; head h serves frequencies 2h+1 and 2h+2
EDGE_DIM = 32         # edge embedding width
ROT_ENCODE = 16       # sinusoidal encoding width of the relative rotation angle
TYPE_EMBED = 16       # learned embedding of the (temporal) edge type
ETA = -0.5            # contrast strength of both refining steps
SLOTS = 2 * WINDOW + 1


# --------------------------------------------------------------------------------------------- temporal graph
def temporal_edges(n: int) -> np.ndarray:
    """Directed edges (source, target) between keyframes at most ``WINDOW`` apart, self excluded.

    Edges are ordered by source node, then by offset -5..-1, +1..+5. The order fixes the summation order of the
    aggregations below.
    """
    offsets = np.array([o for o in range(-WINDOW, WINDOW + 1) if o != 0])
    source = np.broadcast_to(np.arange(n)[:, None], (n, len(offsets)))
    target = source + offsets[None]
    valid = (target >= 0) & (target < n)
    return np.stack((source[valid], target[valid]))


def edge_geometry(poses: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Relative odometry of every edge: [log(1 + distance) / 5, rotation angle / pi] (float32, [E, 2])."""
    source, target = edges
    distance = np.linalg.norm(poses[source, :3, 3] - poses[target, :3, 3], axis=1)
    rotation = poses[target, :3, :3] @ np.transpose(poses[source, :3, :3], (0, 2, 1))
    trace = np.clip(np.trace(rotation, axis1=1, axis2=2), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    geometry = np.zeros((len(source), 2), np.float32)
    geometry[:, 0] = np.log1p(distance) / 5.0
    geometry[:, 1] = angle / np.pi
    return geometry


def odometry_yaw(poses: np.ndarray, lidar_to_pose: Optional[np.ndarray] = None) -> np.ndarray:
    """Yaw psi_i of the LiDAR in the frame of the first keyframe (Sec. 3.2, "Yaw transport")."""
    rotations = np.asarray(poses, dtype=np.float64)[:, :3, :3]
    if lidar_to_pose is not None:
        rotations = rotations @ np.asarray(lidar_to_pose, dtype=np.float64)[:3, :3]
    relative = rotations[0].T[None] @ rotations
    return np.arctan2(relative[:, 1, 0], relative[:, 0, 0])


def build_graph(d: np.ndarray, z: np.ndarray, poses: np.ndarray,
                lidar_to_pose: Optional[np.ndarray] = None, device="cpu") -> Dict[str, torch.Tensor]:
    """Temporal graph of one sequence and the tensors used by the refining steps.

    Returns a dict with
        d          [N, 288]  float32       node input
        edges      [2, E]    int64         (source j, target i): messages flow from j into i
        geometry   [E, 2]    float32       relative distance / rotation of the edge
        spectrum   [N, 16, 8] complex64    z
        neighbour  [N, 11]   int64         neighbour index per slot (slot = j - i + 5; unused slots point to i)
        mask       [N, 11]   float32       1 where the slot holds a neighbour (slot 5, the node itself, is 0)
        phase      [N, 8]    complex64     exp(i k psi_i), k = 1..8
    """
    n = len(d)
    z = np.ascontiguousarray(z, dtype=np.complex64)
    assert d.shape == (n, 288) and z.shape == (n, 16, 8) and poses.shape == (n, 4, 4)
    edges = temporal_edges(n)
    source, target = edges
    slot = source - target + WINDOW
    neighbour = np.broadcast_to(np.arange(n)[:, None], (n, SLOTS)).copy()
    mask = np.zeros((n, SLOTS), np.float32)
    neighbour[target, slot], mask[target, slot] = source, 1.0
    yaw = odometry_yaw(poses, lidar_to_pose)
    phase = np.exp(1j * yaw[:, None] * np.arange(1, 9)[None]).astype(np.complex64)
    out = dict(d=torch.from_numpy(np.asarray(d, np.float32)), edges=torch.from_numpy(edges),
               geometry=torch.from_numpy(edge_geometry(poses, edges)), spectrum=torch.from_numpy(z),
               neighbour=torch.from_numpy(neighbour), mask=torch.from_numpy(mask), phase=torch.from_numpy(phase))
    return {k: v.to(device) for k, v in out.items()}


# --------------------------------------------------------------------------------------------- graph helpers
def neighbour_mean(value: torch.Tensor, target: torch.Tensor, n: int) -> torch.Tensor:
    """Mean of ``value`` over the incoming edges of each target, returned per edge ([E, H])."""
    total = torch.zeros(n, value.shape[1], device=value.device, dtype=value.dtype)
    total.index_add_(0, target, value)
    count = torch.bincount(target, minlength=n).clamp_min(1)
    return (total / count[:, None])[target]


def neighbour_softmax(logits: torch.Tensor, target: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of edge logits ([E, H]) over the incoming edges of each target node."""
    index = target[:, None].expand_as(logits)
    peak = logits.new_zeros(n, logits.shape[1]).scatter_reduce_(0, index, logits.detach(), "amax", include_self=False)
    weight = (logits - peak.index_select(0, target)).exp()
    total = weight.new_zeros(n, logits.shape[1]).scatter_add_(0, index, weight) + 1e-16
    return weight / total.index_select(0, target)


# --------------------------------------------------------------------------------------------- modules
class EdgeEncoder(nn.Module):
    """e_ij = MLP([dist_ij, sin/cos(rot_ij * f), t]) with a learned temporal edge-type embedding t (Sec. 3.2)."""

    def __init__(self) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(2, TYPE_EMBED)
        freqs = torch.exp(torch.arange(0, ROT_ENCODE, 2).float() * -(math.log(10000.0) / ROT_ENCODE))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(1 + ROT_ENCODE + TYPE_EMBED, 2 * EDGE_DIM), nn.ReLU(),
                                 nn.Linear(2 * EDGE_DIM, EDGE_DIM))

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        edge_type = torch.zeros(len(geometry), dtype=torch.long, device=geometry.device)
        angles = geometry[:, 1].unsqueeze(-1) * self.freqs
        raw = torch.cat([geometry[:, 0:1], torch.sin(angles), torch.cos(angles), self.type_embedding(edge_type)], -1)
        return self.mlp(raw)


class DiffAttnConv(nn.Module):
    """One DiffAttnConv layer (Eqs. (1)-(2)).

        q_i = W_Q h_i,  k_ij = tanh(W_K (h_j - h_i)),  v_ij = tanh(W_V (h_j - h_i))
        a_ij = q_i . k_ij / sqrt(d_h),  b_ij = MLP(e_ij)
        lambda_ij = a_ij + b_ij + (a_ij - mean_j a_ij) * tanh(b_ij - mean_j b_ij)
        alpha_ij = softmax_j lambda_ij,  u_i = sum_j alpha_ij v_ij

    Returns u ([N, 128]) and stores alpha ([E, 4]) in ``self.attention``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.head_dim = HIDDEN // HEADS
        self.scale = math.sqrt(self.head_dim)
        self.query = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.key = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.value = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.geometry = nn.Sequential(nn.Linear(EDGE_DIM, EDGE_DIM), nn.ReLU(), nn.Linear(EDGE_DIM, HEADS))
        self.attention = None

    def forward(self, h: torch.Tensor, edges: torch.Tensor, edge_embedding: torch.Tensor) -> torch.Tensor:
        source, target = edges
        center = h[target]
        delta = h[source] - center
        q = self.query(center).view(-1, HEADS, self.head_dim)
        k = self.key(delta).tanh().view(-1, HEADS, self.head_dim)
        a = (q * k).sum(-1) / self.scale
        b = self.geometry(edge_embedding)
        n = len(h)
        score = a + b
        score = score + (a - neighbour_mean(a, target, n)) * (b - neighbour_mean(b, target, n)).tanh()
        self.attention = neighbour_softmax(score, target, n)
        v = self.value(delta).tanh().view(-1, HEADS, self.head_dim)
        messages = (v * self.attention[..., None]).flatten(1)
        out = torch.zeros(n, HIDDEN, dtype=h.dtype, device=h.device)
        return out.index_add(0, target, messages)


def refine_spectra(spectrum: torch.Tensor, neighbour: torch.Tensor, mask: torch.Tensor, phase: torch.Tensor,
                   attention: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Yaw transport, two contrast steps (Eq. (3)) and return to the sensor frame.

        y0_i = z_i e^{-i k psi_i} / ||z_i||_F
        y(l+1)_i = y(l)_i + eta * sum_j alpha(l)_ij (y(l)_j - y(l)_i),   eta = -0.5
        z~_i = y(2)_i e^{i k psi_i}

    ``attention`` is [2, N, 11, 4] (per step, slot and head); head h weights frequencies 2h+1, 2h+2 of all 16 range
    bins. ``attention=None`` gives uniform weights over the available neighbours (uniform subtraction). A node without
    neighbours keeps its normalised input.
    """
    n = len(spectrum)
    norm = spectrum.abs().flatten(1).norm(dim=1).clamp_min(1e-12)
    y = spectrum / norm[:, None, None]
    y = y * phase.conj()[:, None]
    if attention is None:
        uniform = mask / mask.sum(1, keepdim=True).clamp_min(1)
        attention = uniform[None, :, :, None].expand(2, -1, -1, HEADS)
    eta = torch.full((2, n, 1), ETA, device=spectrum.device)
    for step in range(2):
        update = torch.zeros_like(y)
        for slot in range(SLOTS):
            if slot == WINDOW:
                continue
            a = attention[step, :, slot].repeat_interleave(2, dim=-1) * mask[:, slot, None]
            update = update + a[:, None] * (y[neighbour[:, slot]] - y)
        y = y + eta[step, :, :, None] * update
    return y * phase[:, None]


class NSDRefiner(nn.Module):
    """Temporal spectrum refining: two DiffAttnConv layers drive two contrast steps on the complex spectra.

    h0 = ReLU(LN(W_in log(1 + d)));  h(l+1) = h(l) + LN(u(l)) (with a ReLU after the first LN).
    The attention of layer l+1 weights refining step l.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(288, HIDDEN)
        self.input_norm = nn.LayerNorm(HIDDEN)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(HIDDEN), nn.LayerNorm(HIDDEN)])
        self.edge_encoder = EdgeEncoder()
        self.convs = nn.ModuleList([DiffAttnConv(), DiffAttnConv()])
        self.last_attention = None

    def attention(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Attention of both layers scattered to slots: [2, N, 11, 4] (slot = j - i + 5)."""
        h = F.relu(self.input_norm(self.input_proj(torch.log1p(graph["d"]))))
        edge_embedding = self.edge_encoder(graph["geometry"])
        source, target = graph["edges"]
        dense = []
        for layer, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            previous = h
            h = norm(conv(h, graph["edges"], edge_embedding))
            if layer == 0:
                h = F.relu(h)
            h = h + previous
            slots = torch.zeros(len(h), SLOTS, HEADS, dtype=h.dtype, device=h.device)
            slots[target, source - target + WINDOW] = conv.attention
            dense.append(slots)
        return torch.stack(dense)

    def forward(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Refined spectra z~ [N, 16, 8] (complex64)."""
        self.last_attention = self.attention(graph)
        return refine_spectra(graph["spectrum"], graph["neighbour"], graph["mask"], graph["phase"],
                              self.last_attention)


def load_refiner(path, device="cpu") -> NSDRefiner:
    """Load released weights (a plain state dict)."""
    net = NSDRefiner()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return net.to(device).eval()


@torch.no_grad()
def refine(net: NSDRefiner, d: np.ndarray, z: np.ndarray, poses: np.ndarray,
           lidar_to_pose: Optional[np.ndarray] = None) -> np.ndarray:
    """z_refined for one sequence (numpy complex64 [N, 16, 8])."""
    device = next(net.parameters()).device
    return net(build_graph(d, z, poses, lidar_to_pose, device)).cpu().numpy()
