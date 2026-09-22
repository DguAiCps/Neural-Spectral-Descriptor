#!/usr/bin/env python3
"""Edge classifier (B3) for the place-disjoint retrain protocol (2026-08).

Same construction as scripts/build_and_train_edge_classifier.py (14 features,
pos <5 m / neg >=25 m, 14-64-64-1 MLP, 8 epochs) but on the pd protocol:

  --variant pd    checkpoints/pd_s1   + artifacts/key_remetrize_r_pd2026.npy
                  -> artifacts/edge_classifier_pd2026.pt
  --variant pdnn  checkpoints/pdnn_s1 + artifacts/key_remetrize_r_pdnn2026.npy
                  -> artifacts/edge_classifier_pdnn2026.pt   (no nclt)

TRAIN sequences only (protocol rule: the classifier may never see val/test).
Node descriptors/poses come from the pd2026 training caches (canonical for the
pd checkpoints). nsd_layouts come from the committed operating caches where
those match the pd2026 descriptors (kitti/helipr/mulran, per-frame cosine 1.0)
and from data/preprocessed_pd2026/layouts_*.npz (scripts/_pd_build_layouts.py)
for nclt + helipr Bridge02/Bridge03/Roundabout02.

Never touches artifacts/edge_classifier.pt or results/_edge_cls/.
"""
from __future__ import annotations
import argparse, copy, glob, sys
from pathlib import Path
import numpy as np
import torch, yaml
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes
from build_and_train_edge_classifier import EdgeMLP, edge_features, range_phase_flat, N_FEAT
from run_kitti_operating_point import _normalize  # noqa: F401 (used by edge_features)

PD_CACHE = REPO / "data/preprocessed_pd2026"
OP_CACHE = REPO / "data/preprocessed_cross_sensor_operating"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PD_TRAIN_SEQS = {
    "kitti": ["01", "02", "06", "07"],
    "nclt": ["2012-05-11", "2012-08-04", "2012-11-04", "2012-11-16", "2013-02-23"],
    "helipr": ["Bridge01", "Bridge02", "Bridge03", "Roundabout01", "Roundabout02"],
    "mulran": ["Sejong01", "Sejong02"],
}
PD_HOLDOUT = {"kitti/07", "nclt/2013-02-23", "helipr/Bridge01", "mulran/Sejong02"}

VARIANTS = {
    "pd": dict(
        ckpt=REPO / "checkpoints/pd_s1/best_model.pth",
        rvec=REPO / "artifacts/key_remetrize_r_pd2026.npy",
        config=REPO / "configs/training_place_disjoint.yaml",
        out=REPO / "artifacts/edge_classifier_pd2026.pt",
        feat_dir=REPO / "results/_edge_cls_pd2026",
        train_seqs=PD_TRAIN_SEQS,
        holdout=PD_HOLDOUT,
    ),
    "pdnn": dict(
        ckpt=REPO / "checkpoints/pdnn_s1/best_model.pth",
        rvec=REPO / "artifacts/key_remetrize_r_pdnn2026.npy",
        config=REPO / "configs/training_place_disjoint_no_nclt.yaml",
        out=REPO / "artifacts/edge_classifier_pdnn2026.pt",
        feat_dir=REPO / "results/_edge_cls_pdnn2026",
        train_seqs={k: v for k, v in PD_TRAIN_SEQS.items() if k != "nclt"},
        holdout={h for h in PD_HOLDOUT if not h.startswith("nclt/")},
    ),
}


def _pd_cache(sensor: str, seq: str) -> np.lib.npyio.NpzFile:
    hits = sorted(glob.glob(str(PD_CACHE / f"cache_*_{sensor}_{seq}.npz")))
    assert len(hits) == 1, (sensor, seq, hits)
    return np.load(hits[0])


def _layouts(sensor: str, seq: str, cache: np.lib.npyio.NpzFile) -> np.ndarray:
    """nsd_layouts aligned to the pd2026 keyframes (see module docstring)."""
    pd_lay = PD_CACHE / f"layouts_{sensor}_{seq}_layout60.npz"
    if pd_lay.exists():
        d = np.load(pd_lay)
        assert (d["keyframe_ids"] == cache["keyframe_ids"]).all(), (sensor, seq)
        return d["nsd_layouts"]
    op = np.load(OP_CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
    assert (op["keyframe_ids"] == cache["keyframe_ids"]).all(), (sensor, seq)
    if sensor == "nclt":
        raise RuntimeError(f"nclt/{seq}: pre-z-fix operating layouts are not pd-consistent; "
                           "run scripts/_pd_build_layouts.py first")
    # descriptor identity check: current loader == operating-cache loader
    da, db = cache["descriptors"].astype(np.float32), op["descriptors"].astype(np.float32)
    cos = (da * db).sum(1) / (np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1) + 1e-12)
    assert cos.min() > 0.9999, (sensor, seq, float(cos.min()))
    return op["nsd_layouts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--only", default=None,
                    help="sensor/seq: build only this sequence's features, then exit "
                         "(per-process isolation; a full-run process OOMs the cgroup)")
    ap.add_argument("--train-only", action="store_true",
                    help="skip feature building; train from existing feat_dir npz")
    args = ap.parse_args()
    V = VARIANTS[args.variant]
    V["feat_dir"].mkdir(parents=True, exist_ok=True)
    assert "pd" in V["out"].name, "must never overwrite the frozen paper-main classifier"

    cfg = _apply_encoder_preset(yaml.safe_load(open(V["config"])), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(V["rvec"])}
    r = np.load(V["rvec"]).astype(np.float32)
    model = None if args.train_only else _make_model(rm_cfg, V["ckpt"], DEVICE)

    # ---- 1. build per-sequence feature datasets ----
    for sensor, seqs in V["train_seqs"].items():
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
            graph = _build_eval_graph(
                keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
                cache=cache, config=rm_cfg, device=DEVICE, temporal_edge_mode="bidirectional",
                temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
                sensor_key=sensor)
            with torch.no_grad():
                embA = model(graph.to(DEVICE)).cpu().numpy()
            d_rm = _normalize(d_raw * r[None, :])
            x_phase = range_phase_flat(_layouts(sensor, seq, cache))
            src, dst, feats, labels = edge_features(embA, d_rm, x_phase, poses)
            keep = labels >= 0
            np.savez_compressed(out_path, feats=feats[keep], labels=labels[keep])
            print(f"OK {sensor}/{seq}: {keep.sum():,} edges "
                  f"(pos {int(labels[keep].sum()):,})", flush=True)

    if args.only:
        return

    # ---- 2. train the classifier (same recipe as paper-main) ----
    tr_x, tr_y, ho_x, ho_y = [], [], [], []
    for sensor, seqs in V["train_seqs"].items():
        for seq in seqs:
            d = np.load(V["feat_dir"] / f"data_{sensor}_{seq}.npz")
            tgt = (ho_x, ho_y) if f"{sensor}/{seq}" in V["holdout"] else (tr_x, tr_y)
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
