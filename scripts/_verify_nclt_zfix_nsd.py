#!/usr/bin/env python3
"""Effect of the NCLT z-axis fix on NSD's own closed-form 288D key.

The NCLT loader keeps the body-frame z-DOWN convention, which inverts height
semantics for max-height descriptors (SC++ collapses; z-flip rescues it to
~0.93 on 13-01). NSD's range-image encoder is affected differently: elevation
rows are mirrored and points outside the configured window are clipped. This
script measures how much the deployed 288D key itself gains once NCLT scans
are z-flipped and encoded with the physically correct elevation window.

Variants (release retrieval protocol, |i-j|>=30):
  cache    : cached 288D descriptors (release numbers; sanity reference)
  reenc    : re-encoded from raw scans, release convention (z down,
             window [-30.67, 10.67]) -- validates the encode path vs cache
  zfix     : z flipped up, mirrored window [-10.67, 30.67]

Run: docker run --rm --gpus all -v $REPO:/ws -v /mnt/d/NSD_datasets:/data -w /ws \
       nvcr.io/nvidia/pyg:26.01-py3 python scripts/_verify_nclt_zfix_nsd.py [seq]
Output: results/_alias_source/nclt_zfix_nsd.json
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset  # noqa: E402
from run_kitti_operating_point import _make_encoder  # noqa: E402

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
NCLT_ROOT = Path("/data/nclt")
OUT = REPO / "results/_alias_source"
POS_M, SKIP = 5.0, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NCLT_DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"),
                    ("intensity", "u1"), ("padding", "u1"), ("extra", "<u4")])


def load_scan(path):
    raw = np.fromfile(path, dtype=NCLT_DT)
    x = raw["x"].astype(np.float32) * 0.005 - 100.0
    y = raw["y"].astype(np.float32) * 0.005 - 100.0
    z = raw["z"].astype(np.float32) * 0.005 - 100.0
    pts = np.column_stack([x, y, z])
    m = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 200.0).all(axis=1)
    return pts[m]


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def find_queries(pos):
    q = []
    for j in range(SKIP, len(pos)):
        d = np.linalg.norm(pos[: j - SKIP + 1] - pos[j], axis=1)
        if (d < POS_M).any():
            q.append(j)
    return np.asarray(q, dtype=int)


def r1(emb, pos, queries):
    emb = l2(emb.astype(np.float64))
    idx = np.arange(len(emb))
    sims = emb[queries] @ emb.T
    miss = 0
    for qi, j in enumerate(queries):
        cand = np.abs(idx - j) >= SKIP
        t1 = int(np.argmax(sims[qi, cand]))
        if np.linalg.norm(pos[cand][t1] - pos[j]) >= POS_M:
            miss += 1
    return 1.0 - miss / max(len(queries), 1)


def make_enc(elev_range):
    cfg = yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml"))
    cfg = _apply_encoder_preset(copy.deepcopy(cfg), "no_interdiff")
    cfg["encoding"]["elevation_range"] = list(elev_range)
    return _make_encoder(cfg, DEVICE)


def encode_all(enc, scans, flip):
    out = []
    with torch.no_grad():
        for p in scans:
            q = p * np.array([1, 1, -1], dtype=np.float32) if flip else p
            out.append(enc.encode_points(q).detach().cpu().numpy())
    return np.asarray(out, dtype=np.float64)


def main():
    seqs = sys.argv[1:] or ["2013-01-10"]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "nclt_zfix_nsd.json"
    out = json.load(open(out_path)) if out_path.exists() else {}
    for seq in seqs:
        z = np.load(CACHE / f"nclt_operating_{seq}_layout60_stride1.npz")
        pos = z["poses"][:, :3, 3]
        sel = z["scan_ids"].astype(int)
        files = sorted((NCLT_ROOT / seq / "velodyne_sync").glob("*.bin"))
        scans = [load_scan(files[s]) for s in sel]
        queries = find_queries(pos)
        res = {"n": int(len(sel)), "n_q": int(len(queries))}
        res["cache"] = r1(z["descriptors"], pos, queries)

        enc_rel = make_enc((-30.67, 10.67))
        reenc = encode_all(enc_rel, scans, flip=False)
        res["reenc"] = r1(reenc, pos, queries)
        cachec = l2(z["descriptors"].astype(np.float64))
        res["reenc_vs_cache_cos"] = float(np.mean(np.sum(l2(reenc) * cachec, axis=1)))

        enc_fix = make_enc((-10.67, 30.67))
        zfix = encode_all(enc_fix, scans, flip=True)
        res["zfix"] = r1(zfix, pos, queries)

        out[seq] = res
        print(f"[{seq}] cache={res['cache']:.4f} reenc={res['reenc']:.4f} "
              f"(cos {res['reenc_vs_cache_cos']:.4f}) zfix={res['zfix']:.4f}", flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
