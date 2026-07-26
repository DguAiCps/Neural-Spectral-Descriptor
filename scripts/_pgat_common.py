#!/usr/bin/env python3
"""
Shared helpers for the P-GAT baseline adapters (_pgat_train.py / _pgat_eval.py).

P-GAT source (_external/P-GAT, commit a66a701) is imported untouched via
sys.path insertion. This module adds:

1. pgat_streams(): a functionally-identical, out-of-place replication of
   AttentionalGraph.forward + the PoseGAT output head. Their forward mutates
   its input tensor in place (`features[:, i] = ...`), which trips autograd's
   saved-tensor version check on current PyTorch (tensors saved by the
   attention/linear layers share storage with the mutated base). The
   replication preserves their exact update order, including the asymmetric
   cross-attention schedule: stream 0 is cross-updated first using the
   self-updated stream 1; stream 1 is then cross-updated using the ALREADY
   cross-updated stream 0.
2. verify_against_native(): checks the replication against their native
   forward under torch.no_grad() (where in-place is harmless) on a given batch.
3. Window/subgraph tensor gather helpers shared by trainer and eval.
"""

import os
import sys

import torch
import torch.nn.functional as F

# NGC containers enable TF32 matmul by default; TF32 error (~1e-3) makes the
# cosine in PoseGAT.forward exceed 1, pushing scores past the /2.0001 headroom
# and outside BCELoss's [0, 1] domain (device-side assert). Force fp32 like
# their original environment.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGAT_ROOT = os.path.join(REPO, "_external", "P-GAT")
if PGAT_ROOT not in sys.path:
    sys.path.insert(0, PGAT_ROOT)

from attentional_graph.modeling.pose_gat import PoseGAT  # noqa: E402,F401


def build_model(pose_dim, feature_dim, keypoint_hidden_dim, include_pose,
                num_heads, num_layers, dropout):
    return PoseGAT(
        pose_dim=pose_dim,
        feature_dim=feature_dim,
        keypoint_enc_hidden_dim=list(keypoint_hidden_dim),
        include_pose=include_pose,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def pgat_streams(model, f0, f1, p0, p1, m0, m1):
    """Out-of-place replication of AttentionalGraph.forward.

    Args (per stream): f [B, n, D] features, p [B, n, pose_dim] poses,
                       m [B, n] bool key_padding_mask (True = padding).
    Returns (f0, f1) post-attention features (pre output_fc).
    """
    ag = model.attentional_graph
    if ag.include_pose:
        f0 = ag.embeding(f0, p0, m0)
        f1 = ag.embeding(f1, p1, m1)
    for layer in range(ag.num_layers):
        # self attention (their iteration 0 then 1; streams independent here)
        s0 = ag.attention_aggregation[2 * layer](f0, key_padding_mask=m0)
        f0 = ag.aggregating(f0, m0, s0, 2 * layer)
        s1 = ag.attention_aggregation[2 * layer](f1, key_padding_mask=m1)
        f1 = ag.aggregating(f1, m1, s1, 2 * layer)
        # cross attention: stream 0 first, stream 1 sees the UPDATED stream 0
        c0 = ag.attention_aggregation[2 * layer + 1](f0, f1, key_padding_mask=m1)
        f0 = ag.aggregating(f0, m0, c0, 2 * layer + 1)
        c1 = ag.attention_aggregation[2 * layer + 1](f1, f0, key_padding_mask=m0)
        f1 = ag.aggregating(f1, m1, c1, 2 * layer + 1)
    return f0, f1


def pgat_forward(model, features, poses, masks):
    """Replication of PoseGAT.forward (scores) + per-node embeddings.

    Args: features [B, 2, n, D], poses [B, 2, n, pose_dim], masks [B, 2, n].
    Returns (scores [B, n, n], e0 [B, n, D], e1 [B, n, D]) where e* are the
    output_fc + L2-normalized node descriptors of each stream.
    """
    f0, f1 = pgat_streams(
        model,
        features[:, 0], features[:, 1],
        poses[:, 0], poses[:, 1],
        masks[:, 0], masks[:, 1],
    )
    e0 = F.normalize(model.output_fc(f0), dim=2)
    e1 = F.normalize(model.output_fc(f1), dim=2)
    scores = torch.einsum("bnd, bmd -> bnm", e0, e1) / 2.0001 + 0.5
    return scores, e0, e1


def verify_against_native(model, features, poses, masks, atol=1e-5):
    """Compare pgat_forward against the untouched PoseGAT.forward (no_grad).

    Returns (ok, max_abs_diff). Native forward mutates its input, so it gets a
    clone. Raises nothing; caller decides what to do on mismatch.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        scores_rep, _, _ = pgat_forward(model, features, poses, masks)
        scores_nat = model(features.clone(), poses.clone(), masks.clone())
    if was_training:
        model.train()
    diff = (scores_rep - scores_nat).abs().max().item()
    return diff <= atol, diff


def gather_subgraph_features(features, sub_nodes, sub_masks):
    """features [N, D], sub_nodes [B, n] int (-1 pad), sub_masks [B, n] bool
    (True = pad) -> [B, n, D] with zeros at padding (their generator pads
    features with zeros)."""
    idx = sub_nodes.clamp(min=0).long()
    out = features[idx]
    out = out * (~sub_masks).unsqueeze(-1).to(out.dtype)
    return out


def block_scores_switches(xy, nodes_a, masks_a, nodes_b, masks_b,
                          pos_thresh, neg_thresh):
    """On-the-fly replication of their global adjacency / switches matrices
    for one batch of subgraph pairs.

    Their generator: adjacency[i, j] = 1 iff dist2d(i, j) < pos_thresh;
    switches[i, j] = 1 iff dist < pos_thresh or dist >= neg_thresh (the
    [pos, neg) band is ignored by the loss). Identical here, computed from the
    stored ground-plane positions for the node id lists of each pair. Padded
    rows/cols get score 0 / switch 0 (their versions carry real values of
    neighboring nodes there, but weight_loss masks them out either way).

    Args: xy [N, 2]; nodes_* [B, n] int (-1 pad); masks_* [B, n] bool (True=pad).
    Returns (scores [B, n, n] float, switches [B, n, n] float).
    """
    xa = xy[nodes_a.clamp(min=0).long()]
    xb = xy[nodes_b.clamp(min=0).long()]
    dist = torch.cdist(xa, xb)
    valid = (~masks_a).unsqueeze(2) & (~masks_b).unsqueeze(1)
    scores = ((dist < pos_thresh) & valid).float()
    switches = (((dist < pos_thresh) | (dist >= neg_thresh)) & valid).float()
    return scores, switches
