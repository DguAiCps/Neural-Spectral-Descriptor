#!/usr/bin/env python3
"""Causal edge classifier (B3) for the pre-registered causal PR protocol.

Same construction and recipe as scripts/_pd_build_edge_classifier.py (14
features, pos <5 m / neg >=25 m, 14-64-64-1 MLP, 8 epochs) but with CAUSAL
pass-1 graphs and CAUSAL prefix-only features
(scripts/_causal_edge_features.py; docs/CAUSAL_PR_PREREGISTRATION.md §2.1):

  --variant pd_causal  checkpoints/pd_causal_s{seed} (causal-retrained)
                       -> artifacts/edge_classifier_pd2026_causal.pt
  --variant pd         checkpoints/pd_s{seed} (Frozen-GNN causal adaptation:
                       frozen bidirectional-trained GNN, causal graphs and
                       causal features; pre-registration §0/§2.3)
                       -> artifacts/edge_classifier_pd2026_frozengnn.pt

Both variants use configs/training_place_disjoint_causal.yaml for graph
construction (temporal_neighbors=20 = past-only half-window 10, keeping total
incoming temporal degree ~10 for BOTH tracks, §2.2) and share
artifacts/key_remetrize_r_pd2026.npy (omega reuse: the fit is checkpoint-free,
§2.3).  mu/sd standardization is recomputed over the CAUSAL feature
distribution (§2.1).  TRAIN sequences only, PD_HOLDOUT split unchanged.
--seed picks the feature checkpoint (default 1, mirroring how
_pd_build_edge_classifier picks pd_s1; feat_dir/output names are
seed-agnostic like the pd chain).  Never touches the frozen bidirectional
classifiers (edge_classifier.pt / *_pd2026.pt / *_pdnn2026.pt).
"""
from __future__ import annotations
import argparse, copy, sys
from pathlib import Path
import numpy as np
import torch, yaml
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model
from build_and_train_edge_classifier import EdgeMLP, range_phase_flat
from run_kitti_operating_point import _normalize
from _pd_build_edge_classifier import PD_TRAIN_SEQS, PD_HOLDOUT, _pd_cache, _layouts
from _causal_edge_features import causal_edge_features, causal_pass1_graph

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VARIANTS = {
    "pd_causal": dict(
        ckpt_dir="pd_causal_s{seed}",
        rvec=REPO / "artifacts/key_remetrize_r_pd2026.npy",
        config=REPO / "configs/training_place_disjoint_causal.yaml",
        out=REPO / "artifacts/edge_classifier_pd2026_causal.pt",
        feat_dir=REPO / "results/_edge_cls_pd2026_causal",
    ),
    "pd": dict(
        ckpt_dir="pd_s{seed}",
        rvec=REPO / "artifacts/key_remetrize_r_pd2026.npy",
        config=REPO / "configs/training_place_disjoint_causal.yaml",
        out=REPO / "artifacts/edge_classifier_pd2026_frozengnn.pt",
        feat_dir=REPO / "results/_edge_cls_pd2026_frozengnn",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), default=1,
                    help="feature-checkpoint seed (pre-registered chain uses s1)")
    ap.add_argument("--only", default=None,
                    help="sensor/seq: build only this sequence's features, then exit "
                         "(per-process isolation; a full-run process OOMs the cgroup)")
    ap.add_argument("--train-only", action="store_true",
                    help="skip feature building; train from existing feat_dir npz")
    args = ap.parse_args()
    V = VARIANTS[args.variant]
    V["feat_dir"].mkdir(parents=True, exist_ok=True)
    assert V["out"].name not in {"edge_classifier.pt", "edge_classifier_pd2026.pt",
                                 "edge_classifier_pdnn2026.pt"}, \
        "must never overwrite a frozen bidirectional classifier"

    cfg = _apply_encoder_preset(yaml.safe_load(open(V["config"])), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(V["rvec"])}
    r = np.load(V["rvec"]).astype(np.float32)
    ckpt = REPO / "checkpoints" / V["ckpt_dir"].format(seed=args.seed) / "best_model.pth"
    model = None if args.train_only else _make_model(rm_cfg, ckpt, DEVICE)
    print(f"variant={args.variant} seed={args.seed} ckpt={ckpt}", flush=True)

    # ---- 1. build per-sequence CAUSAL feature datasets ----
    for sensor, seqs in PD_TRAIN_SEQS.items():
        for seq in seqs:
            if args.train_only:
                continue
            if args.only and f"{sensor}/{seq}" != args.only:
                continue
            out_path = V["feat_dir"] / f"data_{sensor}_{seq}.npz"
            if out_path.exists():
                print("SKIP", out_path.name, flush=True)
                continue
            cache = _pd_cache(sensor, seq)
            d_raw = cache["descriptors"].astype(np.float32)
            poses = cache["poses"]
            graph = causal_pass1_graph(cache, d_raw, rm_cfg, DEVICE, sensor)
            with torch.no_grad():
                embA = model(graph.to(DEVICE)).cpu().numpy()
            d_rm = _normalize(d_raw * r[None, :])
            phase_flat = range_phase_flat(_layouts(sensor, seq, cache))
            src, dst, feats, labels = causal_edge_features(
                embA, d_rm, phase_flat, poses)
            keep = labels >= 0
            np.savez_compressed(out_path, feats=feats[keep], labels=labels[keep])
            print(f"OK {sensor}/{seq}: {keep.sum():,} edges "
                  f"(pos {int(labels[keep].sum()):,})", flush=True)

    if args.only:
        return

    # ---- 2. train the classifier (same recipe as paper-main) ----
    tr_x, tr_y, ho_x, ho_y = [], [], [], []
    for sensor, seqs in PD_TRAIN_SEQS.items():
        for seq in seqs:
            d = np.load(V["feat_dir"] / f"data_{sensor}_{seq}.npz")
            tgt = (ho_x, ho_y) if f"{sensor}/{seq}" in PD_HOLDOUT else (tr_x, tr_y)
            tgt[0].append(d["feats"]); tgt[1].append(d["labels"])
    X = np.concatenate(tr_x); y = np.concatenate(tr_y).astype(np.float32)
    Xh = np.concatenate(ho_x); yh = np.concatenate(ho_y).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn, Xhn = (X - mu) / sd, (Xh - mu) / sd
    pos_w = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"train {len(X):,} (pos {int(y.sum()):,}, pos_w {pos_w:.1f}) "
          f"holdout {len(Xh):,} (pos {int(yh.sum()):,})", flush=True)

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

    torch.save({"state_dict": clf.state_dict(), "mu": mu, "sd": sd}, V["out"])
    print("saved", V["out"], flush=True)


if __name__ == "__main__":
    main()
