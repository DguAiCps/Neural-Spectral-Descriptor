#!/usr/bin/env python3
"""Post-hoc similarity-edge classifier (B3), trained on the frozen model.

For every TRAIN sequence: forward the frozen seed-1 checkpoint (fixed diag_wccn
key) on the paper graph, take the top-M cosine candidates per node (temporal
window excluded), and describe each candidate edge with inference-time features:
key/context/final cosines, rank and reciprocal-rank, local density, query
margin, range-sketch cyclic-shift alignment evidence (score / shift / peak
margin / entropy), and normalized index gap. Labels come from GT poses:
positive < 5 m, negative >= 25 m, the ambiguous 5-25 m band is dropped.
A small MLP is trained on these features; at eval time it replaces the cosine
top-k / Bayesian rules for similarity-edge selection (targets the pose-oracle
gap identified by run_oracle_ceiling_eval.py).
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import torch, yaml
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes
from run_kitti_operating_point import _normalize
from encoding.phase_alignment import phase_alignment_edge_features

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
RVEC = REPO / "artifacts/key_remetrize_r.npy"
OUT = REPO / "results/_edge_cls"
CKPT = REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth"
TRAIN_SEQS = {
    "kitti": ["01", "02", "06", "07"],
    "nclt": ["2012-05-11", "2012-08-04", "2012-11-04", "2012-11-16", "2013-02-23"],
    "helipr": ["Town02", "Roundabout01", "Bridge01", "KAIST04", "DCC04", "Riverside04"],
    "mulran": ["DCC01", "DCC02", "KAIST01", "KAIST02", "Riverside01", "Riverside02",
               "Sejong01", "Sejong02"],
}
HOLDOUT = {"kitti/07", "nclt/2013-02-23", "helipr/Bridge01", "mulran/Sejong02"}
M_CAND, SKIP, POS_D, NEG_D = 30, 30, 5.0, 25.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FEAT = 14


def range_phase_flat(nsd_layouts: np.ndarray, n_freqs: int = 4) -> np.ndarray:
    coeffs = np.fft.rfft(nsd_layouts.astype(np.float32), axis=2, norm="ortho")[:, :, 1:n_freqs + 1]
    n = len(coeffs)
    return np.concatenate(
        [coeffs.real.reshape(n, -1), coeffs.imag.reshape(n, -1)], axis=1).astype(np.float32)


def edge_features(embA, d_rm, x_phase, poses, want_labels=True):
    """Return (src, dst, feats(E,14)) and labels if requested."""
    n = len(embA)
    f_final = _normalize(embA)
    ctx = _normalize(embA[:, 288:])
    xy = poses[:, :2, 3]

    sims_full = f_final @ f_final.T
    idx = np.arange(n)
    band = np.abs(idx[:, None] - idx[None, :]) < SKIP
    sims = sims_full.copy()
    sims[band] = -2.0
    order = np.argsort(-sims, axis=1)[:, :M_CAND]              # (n, M)
    cand_sims = np.take_along_axis(sims, order, axis=1)

    top10 = cand_sims[:, :10]
    density = top10.mean(axis=1)
    margin = cand_sims[:, 0] - cand_sims[:, 1]

    # reciprocal rank lookup
    rank_of = np.full((n, n), M_CAND, dtype=np.float32)
    np.put_along_axis(rank_of, order, np.arange(M_CAND, dtype=np.float32)[None, :], axis=1)

    src = np.repeat(idx, M_CAND)
    dst = order.reshape(-1)
    feats = np.empty((n * M_CAND, N_FEAT), dtype=np.float32)
    feats[:, 0] = cand_sims.reshape(-1)                                     # cos final
    feats[:, 1] = np.einsum("ij,ij->i", d_rm[src], d_rm[dst])               # cos re-metrized key
    feats[:, 2] = np.einsum("ij,ij->i", ctx[src], ctx[dst])                 # cos context
    feats[:, 3] = np.tile(np.arange(M_CAND, dtype=np.float32), n) / M_CAND  # rank
    feats[:, 4] = rank_of[dst, src] / M_CAND                                # reciprocal rank
    feats[:, 5] = margin[src]
    feats[:, 6] = density[src]
    feats[:, 7] = density[dst]

    ei = torch.from_numpy(np.stack([src, dst])).long()
    ph = phase_alignment_edge_features(
        torch.from_numpy(x_phase), ei, None, n_rows=16, n_freqs=4, n_sectors=60,
        include_score=True, similarity_only=False).numpy()
    feats[:, 8:8 + ph.shape[1]] = ph[:, :5]
    feats[:, 13] = np.abs(src - dst) / float(n)

    if not want_labels:
        return src, dst, feats, None
    d = np.linalg.norm(xy[src] - xy[dst], axis=1)
    labels = np.full(len(src), -1, dtype=np.int8)
    labels[d < POS_D] = 1
    labels[d >= NEG_D] = 0
    return src, dst, feats, labels


class EdgeMLP(nn.Module):
    def __init__(self, d_in=N_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(RVEC)}
    r = np.load(RVEC).astype(np.float32)
    model = _make_model(rm_cfg, CKPT, DEVICE)

    # ---- 1. build per-sequence feature datasets ----
    for sensor, seqs in TRAIN_SEQS.items():
        for seq in seqs:
            out_path = OUT / f"data_{sensor}_{seq}.npz"
            if out_path.exists():
                print("SKIP", out_path.name, flush=True)
                continue
            cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
            d_raw = cache["descriptors"].astype(np.float32)
            poses = cache["poses"]
            graph = _build_eval_graph(
                keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
                cache=cache, config=rm_cfg, device=DEVICE, temporal_edge_mode="bidirectional",
                temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
                sensor_key=sensor)
            with torch.no_grad():
                embA = model(graph.to(DEVICE)).cpu().numpy()
            d_rm = _normalize(d_raw * r[None, :])
            x_phase = range_phase_flat(cache["nsd_layouts"])
            src, dst, feats, labels = edge_features(embA, d_rm, x_phase, poses)
            keep = labels >= 0
            np.savez_compressed(out_path, feats=feats[keep], labels=labels[keep])
            print(f"OK {sensor}/{seq}: {keep.sum():,} edges "
                  f"(pos {int(labels[keep].sum()):,})", flush=True)

    # ---- 2. train the classifier ----
    tr_x, tr_y, ho_x, ho_y = [], [], [], []
    for sensor, seqs in TRAIN_SEQS.items():
        for seq in seqs:
            d = np.load(OUT / f"data_{sensor}_{seq}.npz")
            tgt = (ho_x, ho_y) if f"{sensor}/{seq}" in HOLDOUT else (tr_x, tr_y)
            tgt[0].append(d["feats"]); tgt[1].append(d["labels"])
    X = np.concatenate(tr_x); y = np.concatenate(tr_y).astype(np.float32)
    Xh = np.concatenate(ho_x); yh = np.concatenate(ho_y).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn, Xhn = (X - mu) / sd, (Xh - mu) / sd
    pos_w = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"train {len(X):,} (pos {int(y.sum()):,}, pos_w {pos_w:.1f}) "
          f"holdout {len(Xh):,}", flush=True)

    torch.manual_seed(0)
    clf = EdgeMLP().to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=DEVICE))
    Xt = torch.from_numpy(Xn).float().to(DEVICE)
    yt = torch.from_numpy(y).to(DEVICE)
    for epoch in range(8):
        perm = torch.randperm(len(Xt), device=DEVICE)
        tot = 0.0
        for b in range(0, len(Xt), 65536):
            sel = perm[b:b + 65536]
            opt.zero_grad()
            loss = lossf(clf(Xt[sel]), yt[sel])
            loss.backward(); opt.step()
            tot += float(loss) * len(sel)
        with torch.no_grad():
            ph = torch.sigmoid(clf(torch.from_numpy(Xhn).float().to(DEVICE))).cpu().numpy()
        os_ = np.argsort(-ph)
        auc = float(((ph[yh == 1][:, None] > ph[yh == 0][None, :1000]).mean()))
        k = max(int(yh.sum()), 1)
        prec_at_pos = float(yh[os_[:k]].mean())
        print(f"epoch {epoch}: loss {tot/len(Xt):.4f}  holdout AUC~{auc:.3f} "
              f"P@|pos| {prec_at_pos:.3f}", flush=True)

    torch.save({"state_dict": clf.state_dict(), "mu": mu, "sd": sd}, OUT / "classifier.pt")
    print("saved", OUT / "classifier.pt", flush=True)


if __name__ == "__main__":
    main()
